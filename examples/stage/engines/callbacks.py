#!/usr/bin/env python
# coding=utf-8
"""
训练回调函数

包含：
- AMPDiagnosticCallback: 监控 AMP GradScaler 状态
- JointLoggingCallback: 美化训练日志输出
- E2EEvaluationCallback: 端到端评估回调 (Stage 1/3/4)
- Stage1EvaluationCallback: Stage1 分类评估回调
"""

import logging
import os
from typing import Dict, Optional

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class AMPDiagnosticCallback(TrainerCallback):
    """
    监控 AMP GradScaler 状态，用于诊断 fp16 溢出问题

    工业实践：当 scale 下降时，通常意味着检测到了 overflow 并跳过了该步更新
    """

    def __init__(self):
        self.prev_scale = None
        self.overflow_count = 0
        self.scaler = None

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时尝试获取 scaler 引用"""
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, torch.cuda.amp.GradScaler):
                self.scaler = obj
                logger.info(f"[AMP-DIAG] Found GradScaler, initial scale={self.scaler.get_scale():.1f}")
                break

    def on_step_end(self, args, state, control, **kwargs):
        """每步结束后检查 GradScaler 状态"""
        if self.scaler is None:
            return

        try:
            current_scale = self.scaler.get_scale()
            if self.prev_scale is not None:
                if current_scale < self.prev_scale:
                    self.overflow_count += 1
                    logger.warning(
                        f"[AMP-DIAG] Step {state.global_step}: "
                        f"Scale decreased {self.prev_scale:.1f} -> {current_scale:.1f} "
                        f"(overflow detected, total overflows: {self.overflow_count})"
                    )
            self.prev_scale = current_scale

            # 每 500 步打印一次 scale 状态
            if state.global_step % 500 == 0:
                logger.info(f"[AMP-DIAG] Step {state.global_step}: scale={current_scale:.1f}, total_overflows={self.overflow_count}")
        except Exception:
            pass  # scaler 可能不可用


class JointLoggingCallback(TrainerCallback):
    """记录联合训练的详细日志（美化版）"""

    def __init__(self, total_steps: int = None, mode: str = "joint"):
        self.total_steps = total_steps
        self.mode = mode
        self.best_parent_acc = 0.0
        self.best_rel_acc = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        total = self.total_steps or state.max_steps or 1

        # 跳过没有 loss 的日志（如 eval 结果）
        if "loss" not in logs:
            return

        # 进度条
        progress = step / total
        bar_width = 20
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        # 学习率（从 logs 或 state 获取）
        lr = logs.get("learning_rate")
        if lr is None and hasattr(state, 'last_lr') and state.last_lr is not None:
            lr = state.last_lr
        if lr is None:
            lr = args.learning_rate
        lr_str = f"{lr:.2e}" if lr and lr > 0 else "N/A"

        # 构建输出
        header = f"Step {step:>5}/{total} [{bar}] {progress*100:>5.1f}%  lr={lr_str}"

        if self.mode == "stage1":
            # Stage1 模式：只显示分类 loss
            cls_loss = logs.get("cls_loss", logs.get("loss", 0))
            tasks = f"  loss={logs['loss']:.4f}  │  cls={cls_loss:.3f}"
        else:
            # Joint/Stage34 模式：显示全部指标
            cls_loss = logs.get("cls_loss", 0)
            parent_loss = logs.get("parent_loss", 0)
            rel_loss = logs.get("rel_loss", 0)
            parent_acc = logs.get("parent_acc", 0)
            rel_acc = logs.get("rel_acc", 0)

            # 更新最佳
            if parent_acc > self.best_parent_acc:
                self.best_parent_acc = parent_acc
            if rel_acc > self.best_rel_acc:
                self.best_rel_acc = rel_acc

            # 任务指标（带趋势指示）
            parent_indicator = "▲" if parent_acc >= self.best_parent_acc else " "
            rel_indicator = "▲" if rel_acc >= self.best_rel_acc else " "

            tasks = (
                f"  loss={logs['loss']:.4f}  │  "
                f"cls={cls_loss:.3f}  │  "
                f"parent={parent_loss:.3f} ({parent_acc:>5.1%}){parent_indicator}  │  "
                f"rel={rel_loss:.3f} ({rel_acc:>5.1%}){rel_indicator}"
            )

        logger.info(header)
        logger.info(tasks)


class E2EEvaluationCallback(TrainerCallback):
    """
    端到端评估 Callback (Stage 1/3/4)

    在每次评估时运行完整的端到端评估：
    - Stage 1: 分类 (Line-level Macro/Micro F1)
    - Stage 3: Parent 准确率
    - Stage 4: Relation 准确率 + Macro F1

    使用 engines/evaluator.py 统一接口
    """

    def __init__(
        self,
        eval_dataloader,
        data_collator=None,
        compute_teds: bool = False,
        save_predictions: bool = False,
        output_dir: str = None,
        # Best model 保存配置
        best_model_metric: str = "parent_accuracy",
        trainer=None,  # JointTrainer 实例，用于保存模型
    ):
        self.eval_dataloader = eval_dataloader
        self.data_collator = data_collator
        self.compute_teds = compute_teds
        self.save_predictions = save_predictions
        self.output_dir = output_dir

        # Best model 配置（始终保存，指标越大越好）
        self.best_model_metric = best_model_metric
        self.trainer = trainer
        self.best_metric_value = float('-inf')
        self.best_step = None

        # 历史评估记录：[(step, line_macro, line_micro, line_acc, parent_acc, rel_macro, rel_micro, rel_acc, teds, sec_parent, sec_rel), ...]
        self.history = []

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """在 Trainer.evaluate() 之后运行端到端评估"""
        if model is None:
            return

        device = next(model.parameters()).device
        global_step = state.global_step

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"End-to-End Evaluation (Stage 1/3/4) at Step {global_step}")
        logger.info("=" * 60)

        # 使用 Evaluator（统一接口，支持 page/doc 级别）
        from .evaluator import Evaluator

        evaluator = Evaluator(model, device)
        output = evaluator.evaluate(
            self.eval_dataloader,
            compute_teds=self.compute_teds,
            verbose=True,
            save_predictions=self.save_predictions,
            output_dir=self.output_dir,
        )

        # 打印结果
        line_macro = output.line_macro_f1 * 100
        line_micro = output.line_micro_f1 * 100
        line_acc = output.line_accuracy * 100
        parent_acc = output.parent_accuracy * 100
        rel_acc = output.relation_accuracy * 100
        rel_macro = output.relation_macro_f1 * 100
        rel_micro = output.relation_micro_f1 * 100
        teds = output.teds_score * 100 if output.teds_score is not None else None
        num_lines = output.num_lines
        # Section 指标
        sec_parent_acc = output.section_parent_accuracy * 100
        sec_rel_acc = output.section_relation_accuracy * 100
        sec_edge_acc = output.section_edge_accuracy * 100

        def fmt_delta(d, threshold=0.5):
            if d >= threshold:
                return f"↑{d:+.1f}"
            elif d <= -threshold:
                return f"↓{d:+.1f}"
            else:
                return f" {d:+.1f}"

        avg_n = min(3, len(self.history))

        logger.info("")
        logger.info("╔════════════════════════════════════════════════════════════════════════╗")
        logger.info(f"║              Evaluation Results @ Step {global_step:<6}                        ║")
        logger.info("╠════════════════════════════════════════════════════════════════════════╣")

        if avg_n > 0:
            recent = self.history[-avg_n:]
            avg_line_macro = sum(h[1] for h in recent) / avg_n
            avg_line_micro = sum(h[2] for h in recent) / avg_n
            avg_line_acc = sum(h[3] for h in recent) / avg_n
            avg_parent = sum(h[4] for h in recent) / avg_n
            avg_rel_macro = sum(h[5] for h in recent) / avg_n
            avg_rel_micro = sum(h[6] for h in recent) / avg_n
            avg_rel_acc = sum(h[7] for h in recent) / avg_n
            avg_sec_parent = sum(h[9] for h in recent) / avg_n
            avg_sec_rel = sum(h[10] for h in recent) / avg_n
            avg_sec_edge = sum(h[11] for h in recent) / avg_n

            delta_line_macro = line_macro - avg_line_macro
            delta_line_micro = line_micro - avg_line_micro
            delta_line_acc = line_acc - avg_line_acc
            delta_parent = parent_acc - avg_parent
            delta_rel_macro = rel_macro - avg_rel_macro
            delta_rel_micro = rel_micro - avg_rel_micro
            delta_rel_acc = rel_acc - avg_rel_acc
            delta_sec_parent = sec_parent_acc - avg_sec_parent
            delta_sec_rel = sec_rel_acc - avg_sec_rel
            delta_sec_edge = sec_edge_acc - avg_sec_edge

            logger.info(f"║  Metric          │ Current  │  Avg({avg_n})  │  Delta       ║")
            logger.info("║──────────────────┼──────────┼──────────┼──────────────║")
            logger.info(f"║  Line(MacroF1)   │  {line_macro:>5.1f}%  │  {avg_line_macro:>5.1f}%  │  {fmt_delta(delta_line_macro):>6}      ║")
            logger.info(f"║  Line(MicroF1)   │  {line_micro:>5.1f}%  │  {avg_line_micro:>5.1f}%  │  {fmt_delta(delta_line_micro):>6}      ║")
            logger.info(f"║  Line(Acc)       │  {line_acc:>5.1f}%  │  {avg_line_acc:>5.1f}%  │  {fmt_delta(delta_line_acc):>6}      ║")
            logger.info(f"║  Parent(Acc)     │  {parent_acc:>5.1f}%  │  {avg_parent:>5.1f}%  │  {fmt_delta(delta_parent):>6}      ║")
            logger.info(f"║  Sec-Parent(Acc) │  {sec_parent_acc:>5.1f}%  │  {avg_sec_parent:>5.1f}%  │  {fmt_delta(delta_sec_parent):>6}      ║")
            logger.info(f"║  Rel(MacroF1)    │  {rel_macro:>5.1f}%  │  {avg_rel_macro:>5.1f}%  │  {fmt_delta(delta_rel_macro):>6}      ║")
            logger.info(f"║  Rel(MicroF1)    │  {rel_micro:>5.1f}%  │  {avg_rel_micro:>5.1f}%  │  {fmt_delta(delta_rel_micro):>6}      ║")
            logger.info(f"║  Rel(Acc)        │  {rel_acc:>5.1f}%  │  {avg_rel_acc:>5.1f}%  │  {fmt_delta(delta_rel_acc):>6}      ║")
            logger.info(f"║  Sec-Rel(Acc)    │  {sec_rel_acc:>5.1f}%  │  {avg_sec_rel:>5.1f}%  │  {fmt_delta(delta_sec_rel):>6}      ║")
            logger.info(f"║  Sec-Edge(Acc) ★ │  {sec_edge_acc:>5.1f}%  │  {avg_sec_edge:>5.1f}%  │  {fmt_delta(delta_sec_edge):>6}      ║")

            summary = f"[Step {global_step}] Line={line_macro:.1f}% | Parent={parent_acc:.1f}% ({fmt_delta(delta_parent)}) | SecEdge={sec_edge_acc:.1f}% ({fmt_delta(delta_sec_edge)})"
        else:
            logger.info(f"║  Metric          │ Current  │                           ║")
            logger.info("║──────────────────┼──────────┼───────────────────────────║")
            logger.info(f"║  Line(MacroF1)   │  {line_macro:>5.1f}%  │                           ║")
            logger.info(f"║  Line(MicroF1)   │  {line_micro:>5.1f}%  │                           ║")
            logger.info(f"║  Line(Acc)       │  {line_acc:>5.1f}%  │                           ║")
            logger.info(f"║  Parent(Acc)     │  {parent_acc:>5.1f}%  │                           ║")
            logger.info(f"║  Sec-Parent(Acc) │  {sec_parent_acc:>5.1f}%  │                           ║")
            logger.info(f"║  Rel(MacroF1)    │  {rel_macro:>5.1f}%  │                           ║")
            logger.info(f"║  Rel(MicroF1)    │  {rel_micro:>5.1f}%  │                           ║")
            logger.info(f"║  Rel(Acc)        │  {rel_acc:>5.1f}%  │                           ║")
            logger.info(f"║  Sec-Rel(Acc)    │  {sec_rel_acc:>5.1f}%  │                           ║")
            logger.info(f"║  Sec-Edge(Acc) ★ │  {sec_edge_acc:>5.1f}%  │                           ║")
            summary = f"[Step {global_step}] Line={line_macro:.1f}% | Parent={parent_acc:.1f}% | SecEdge={sec_edge_acc:.1f}%"

        # TEDS 分数（如果计算了）
        if teds is not None:
            logger.info("║──────────────────┼──────────┼───────────────────────────║")
            logger.info(f"║  TEDS Score      │  {teds:>5.1f}%  │                           ║")
            summary += f" | TEDS={teds:.1f}%"

        logger.info("╠════════════════════════════════════════════════════════════════════════╣")
        logger.info(f"║  Lines evaluated: {num_lines:<53} ║")
        logger.info("╚════════════════════════════════════════════════════════════════════════╝")
        logger.info(summary)

        # history: (step, line_macro, line_micro, line_acc, parent_acc, rel_macro, rel_micro, rel_acc, teds, sec_parent, sec_rel, sec_edge)
        self.history.append((global_step, line_macro, line_micro, line_acc, parent_acc, rel_macro, rel_micro, rel_acc, teds, sec_parent_acc, sec_rel_acc, sec_edge_acc))

        # Best model 保存
        if self.trainer is not None:
            self._maybe_save_best_model(
                global_step=global_step,
                metrics={
                    "parent_accuracy": parent_acc,
                    "relation_macro_f1": rel_macro,
                    "section_parent": sec_parent_acc,
                    "section_edge": sec_edge_acc,  # parent + relation 都对
                    "teds": teds if teds is not None else 0.0,
                    "line_macro_f1": line_macro,
                },
                model=model,
            )

        # 写回 metrics dict（供 Trainer.evaluate() 返回，HF 用于选 best）
        metrics_dict = kwargs.get("metrics", {})
        metrics_dict.update({
            "line_macro_f1": line_macro / 100,
            "line_micro_f1": line_micro / 100,
            "line_accuracy": line_acc / 100,
            "parent_accuracy": parent_acc / 100,
            "relation_macro_f1": rel_macro / 100,
            "relation_accuracy": rel_acc / 100,
            "section_parent": sec_parent_acc / 100,
            "section_edge": sec_edge_acc / 100,
        })
        if teds is not None:
            metrics_dict["teds"] = teds / 100

    def _maybe_save_best_model(self, global_step: int, metrics: dict, model):
        """检查是否需要保存 best model（指标越大越好）"""
        current_value = metrics.get(self.best_model_metric, 0.0)

        if current_value > self.best_metric_value:
            self.best_metric_value = current_value
            self.best_step = global_step

            # 保存到 output_dir/best/
            best_dir = os.path.join(self.output_dir, "best")
            logger.info(f"")
            logger.info(f"🏆 New best model! {self.best_model_metric}={current_value:.2f}% at step {global_step}")
            logger.info(f"   Saving to: {best_dir}")

            # 调用 trainer 的保存方法
            self.trainer._save(output_dir=best_dir)

            # 保存 best_info.json 记录元信息
            import json
            best_info = {
                "step": global_step,
                "metric": self.best_model_metric,
                "value": current_value,
                "all_metrics": metrics,
            }
            with open(os.path.join(best_dir, "best_info.json"), "w") as f:
                json.dump(best_info, f, indent=2)


class Stage1EvaluationCallback(TrainerCallback):
    """
    Stage1 分类评估 Callback

    只评估 Stage1 的分类指标（Line-level Macro/Micro F1），
    不运行 Stage 3/4 的 Parent/Relation 评估。

    使用 Evaluator 类进行 Line-level 评估（和 E2EEvaluationCallback 相同的评估逻辑）。

    用于 --mode stage1 训练时。
    """

    def __init__(
        self,
        eval_dataloader,
        id2label: Dict[int, str] = None,
        output_dir: str = None,
        trainer=None,  # JointTrainer 实例，用于保存 best model
    ):
        self.eval_dataloader = eval_dataloader
        self.id2label = id2label
        self.output_dir = output_dir
        self.trainer = trainer

        # Best model 配置（stage1 用 focus_macro_f1 = mean(F1_section, F1_fstline, F1_paraline)）
        self.best_model_metric = "focus_macro_f1"
        self.best_metric_value = float('-inf')
        self.best_step = None

        self.history = []  # [(step, line_macro, line_micro, line_acc, focus_macro), ...]

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """运行 Stage1 分类评估（Line-level）"""
        if model is None:
            return

        device = next(model.parameters()).device
        global_step = state.global_step

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Stage1 Classification Evaluation at Step {global_step}")
        logger.info("=" * 60)

        # 使用 Evaluator 进行 Line-level 评估（与联合训练评估一致）
        from .evaluator import Evaluator

        evaluator = Evaluator(model, device)
        output = evaluator.evaluate(
            self.eval_dataloader,
            compute_teds=False,  # Stage1 不计算 TEDS
            verbose=True,
            save_predictions=False,
            output_dir=None,
        )

        # 提取 Line-level 指标
        line_macro = output.line_macro_f1 * 100
        line_micro = output.line_micro_f1 * 100
        line_acc = output.line_accuracy * 100
        focus_macro = output.focus_macro_f1 * 100  # mean(F1_section, F1_fstline, F1_paraline)
        num_lines = output.num_lines

        def fmt_delta(d, threshold=0.5):
            if d >= threshold:
                return f"↑{d:+.1f}"
            elif d <= -threshold:
                return f"↓{d:+.1f}"
            else:
                return f" {d:+.1f}"

        avg_n = min(3, len(self.history))

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════╗")
        logger.info(f"║        Stage1 Results @ Step {global_step:<6}                        ║")
        logger.info("╠══════════════════════════════════════════════════════════════╣")

        if avg_n > 0:
            recent = self.history[-avg_n:]
            avg_macro = sum(h[1] for h in recent) / avg_n
            avg_micro = sum(h[2] for h in recent) / avg_n
            avg_acc = sum(h[3] for h in recent) / avg_n
            avg_focus = sum(h[4] for h in recent) / avg_n
            delta_macro = line_macro - avg_macro
            delta_micro = line_micro - avg_micro
            delta_acc = line_acc - avg_acc
            delta_focus = focus_macro - avg_focus

            logger.info(f"║  Metric       │ Current  │  Avg({avg_n})  │  Delta       ║")
            logger.info("║───────────────┼──────────┼──────────┼──────────────║")
            logger.info(f"║  Macro F1     │  {line_macro:>5.1f}%  │  {avg_macro:>5.1f}%  │  {fmt_delta(delta_macro):>6}      ║")
            logger.info(f"║  Focus F1 ★   │  {focus_macro:>5.1f}%  │  {avg_focus:>5.1f}%  │  {fmt_delta(delta_focus):>6}      ║")
            logger.info(f"║  Micro F1     │  {line_micro:>5.1f}%  │  {avg_micro:>5.1f}%  │  {fmt_delta(delta_micro):>6}      ║")
            logger.info(f"║  Accuracy     │  {line_acc:>5.1f}%  │  {avg_acc:>5.1f}%  │  {fmt_delta(delta_acc):>6}      ║")
        else:
            logger.info(f"║  Metric       │ Current  │                           ║")
            logger.info("║───────────────┼──────────┼───────────────────────────║")
            logger.info(f"║  Macro F1     │  {line_macro:>5.1f}%  │                           ║")
            logger.info(f"║  Focus F1 ★   │  {focus_macro:>5.1f}%  │                           ║")
            logger.info(f"║  Micro F1     │  {line_micro:>5.1f}%  │                           ║")
            logger.info(f"║  Accuracy     │  {line_acc:>5.1f}%  │                           ║")

        logger.info("╠══════════════════════════════════════════════════════════════╣")
        logger.info(f"║  Focus F1 = mean(section, fstline, paraline)                 ║")
        logger.info(f"║  Lines evaluated: {num_lines:<43} ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")

        self.history.append((global_step, line_macro, line_micro, line_acc, focus_macro))

        # Best model 保存（stage1 用 focus_macro_f1）
        if self.trainer is not None and self.output_dir is not None:
            self._maybe_save_best_model(
                global_step=global_step,
                metrics={
                    "focus_macro_f1": focus_macro,
                    "line_macro_f1": line_macro,
                    "line_micro_f1": line_micro,
                    "line_accuracy": line_acc,
                },
                model=model,
            )

        # 写回 metrics dict（供 Trainer.evaluate() 返回）
        metrics_dict = kwargs.get("metrics", {})
        metrics_dict.update({
            "focus_macro_f1": focus_macro / 100,
            "line_macro_f1": line_macro / 100,
            "line_micro_f1": line_micro / 100,
            "line_accuracy": line_acc / 100,
        })

    def _maybe_save_best_model(self, global_step: int, metrics: dict, model):
        """检查是否需要保存 best model（指标越大越好）"""
        current_value = metrics.get(self.best_model_metric, 0.0)

        if current_value > self.best_metric_value:
            self.best_metric_value = current_value
            self.best_step = global_step

            # 保存到 output_dir/best/
            best_dir = os.path.join(self.output_dir, "best")
            logger.info(f"")
            logger.info(f"🏆 New best model! {self.best_model_metric}={current_value:.2f}% at step {global_step}")
            logger.info(f"   Saving to: {best_dir}")

            # 调用 trainer 的保存方法
            self.trainer._save(output_dir=best_dir)

            # 保存 best_info.json 记录元信息
            import json
            best_info = {
                "step": global_step,
                "metric": self.best_model_metric,
                "value": current_value,
                "all_metrics": metrics,
            }
            with open(os.path.join(best_dir, "best_info.json"), "w") as f:
                json.dump(best_info, f, indent=2)

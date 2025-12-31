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
    ):
        self.eval_dataloader = eval_dataloader
        self.data_collator = data_collator
        self.compute_teds = compute_teds
        self.save_predictions = save_predictions
        self.output_dir = output_dir
        # 历史评估记录：[(step, line_f1, line_acc, parent_acc, rel_f1, rel_acc), ...]
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
        line_acc = output.line_accuracy * 100
        parent_acc = output.parent_accuracy * 100
        rel_acc = output.relation_accuracy * 100
        rel_macro = output.relation_macro_f1 * 100
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
        logger.info(f"║           Evaluation Results @ Step {global_step:<6}                  ║")
        logger.info("╠══════════════════════════════════════════════════════════════╣")

        if avg_n > 0:
            recent = self.history[-avg_n:]
            avg_line_f1 = sum(h[1] for h in recent) / avg_n
            avg_line_acc = sum(h[2] for h in recent) / avg_n
            avg_parent = sum(h[3] for h in recent) / avg_n
            avg_rel_f1 = sum(h[4] for h in recent) / avg_n
            avg_rel_acc = sum(h[5] for h in recent) / avg_n

            delta_line_f1 = line_macro - avg_line_f1
            delta_line_acc = line_acc - avg_line_acc
            delta_parent = parent_acc - avg_parent
            delta_rel_f1 = rel_macro - avg_rel_f1
            delta_rel_acc = rel_acc - avg_rel_acc

            logger.info(f"║  Metric       │ Current  │  Avg({avg_n})  │  Delta       ║")
            logger.info("║───────────────┼──────────┼──────────┼──────────────║")
            logger.info(f"║  Line(F1)     │  {line_macro:>5.1f}%  │  {avg_line_f1:>5.1f}%  │  {fmt_delta(delta_line_f1):>6}      ║")
            logger.info(f"║  Line(Acc)    │  {line_acc:>5.1f}%  │  {avg_line_acc:>5.1f}%  │  {fmt_delta(delta_line_acc):>6}      ║")
            logger.info(f"║  Parent(Acc)  │  {parent_acc:>5.1f}%  │  {avg_parent:>5.1f}%  │  {fmt_delta(delta_parent):>6}      ║")
            logger.info(f"║  Rel(F1)      │  {rel_macro:>5.1f}%  │  {avg_rel_f1:>5.1f}%  │  {fmt_delta(delta_rel_f1):>6}      ║")
            logger.info(f"║  Rel(Acc)     │  {rel_acc:>5.1f}%  │  {avg_rel_acc:>5.1f}%  │  {fmt_delta(delta_rel_acc):>6}      ║")

            summary = f"[Step {global_step}] Line={line_macro:.1f}% | Parent={parent_acc:.1f}% ({fmt_delta(delta_parent)}) | Rel={rel_macro:.1f}% ({fmt_delta(delta_rel_f1)})"
        else:
            logger.info(f"║  Metric       │ Current  │                           ║")
            logger.info("║───────────────┼──────────┼───────────────────────────║")
            logger.info(f"║  Line(F1)     │  {line_macro:>5.1f}%  │                           ║")
            logger.info(f"║  Line(Acc)    │  {line_acc:>5.1f}%  │                           ║")
            logger.info(f"║  Parent(Acc)  │  {parent_acc:>5.1f}%  │                           ║")
            logger.info(f"║  Rel(F1)      │  {rel_macro:>5.1f}%  │                           ║")
            logger.info(f"║  Rel(Acc)     │  {rel_acc:>5.1f}%  │                           ║")
            summary = f"[Step {global_step}] Line={line_macro:.1f}% | Parent={parent_acc:.1f}% | Rel={rel_macro:.1f}%"

        logger.info("╠══════════════════════════════════════════════════════════════╣")
        logger.info(f"║  Lines evaluated: {num_lines:<43} ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")
        logger.info(summary)

        self.history.append((global_step, line_macro, line_acc, parent_acc, rel_macro, rel_acc))


class Stage1EvaluationCallback(TrainerCallback):
    """
    Stage1 分类评估 Callback

    只评估 Stage1 的分类指标（Line-level Macro/Micro F1），
    不运行 Stage 3/4 的 Parent/Relation 评估。

    用于 --mode stage1 训练时。
    """

    def __init__(self, eval_dataloader, id2label: Dict[int, str] = None):
        self.eval_dataloader = eval_dataloader
        self.id2label = id2label
        self.history = []  # [(step, line_f1, line_acc), ...]

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """运行 Stage1 分类评估"""
        if model is None:
            return

        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        device = next(model.parameters()).device
        global_step = state.global_step

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Stage1 Classification Evaluation at Step {global_step}")
        logger.info("=" * 60)

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                # 移动到设备
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                image = batch["image"].to(device)
                labels = batch["labels"]

                # Stage1 forward
                outputs = model.stage1(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    image=image,
                )
                logits = outputs.logits  # [batch, seq_len, num_labels]
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                # 收集预测和标签（过滤 -100）
                for pred, label in zip(preds, labels.numpy()):
                    for p, l in zip(pred, label):
                        if l != -100:
                            all_preds.append(p)
                            all_labels.append(l)

        model.train()

        if not all_labels:
            logger.warning("No valid labels found for evaluation")
            return

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)

        line_acc = accuracy * 100
        line_macro = macro_f1 * 100
        line_micro = micro_f1 * 100

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
            avg_f1 = sum(h[1] for h in recent) / avg_n
            avg_acc = sum(h[2] for h in recent) / avg_n
            delta_f1 = line_macro - avg_f1
            delta_acc = line_acc - avg_acc

            logger.info(f"║  Metric       │ Current  │  Avg({avg_n})  │  Delta       ║")
            logger.info("║───────────────┼──────────┼──────────┼──────────────║")
            logger.info(f"║  Macro F1     │  {line_macro:>5.1f}%  │  {avg_f1:>5.1f}%  │  {fmt_delta(delta_f1):>6}      ║")
            logger.info(f"║  Micro F1     │  {line_micro:>5.1f}%  │         │              ║")
            logger.info(f"║  Accuracy     │  {line_acc:>5.1f}%  │  {avg_acc:>5.1f}%  │  {fmt_delta(delta_acc):>6}      ║")
        else:
            logger.info(f"║  Metric       │ Current  │                           ║")
            logger.info("║───────────────┼──────────┼───────────────────────────║")
            logger.info(f"║  Macro F1     │  {line_macro:>5.1f}%  │                           ║")
            logger.info(f"║  Micro F1     │  {line_micro:>5.1f}%  │                           ║")
            logger.info(f"║  Accuracy     │  {line_acc:>5.1f}%  │                           ║")

        logger.info("╠══════════════════════════════════════════════════════════════╣")
        logger.info(f"║  Tokens evaluated: {len(all_labels):<42} ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")

        self.history.append((global_step, line_macro, line_acc))

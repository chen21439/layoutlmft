#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 联合训练脚本 (基于 HuggingFace Trainer)

端到端训练流程:
1. Stage 1: LayoutXLM 分类 (产生分类 loss + hidden states)
2. Stage 2: 从 hidden states 提取 line-level 特征
3. Stage 3: ParentFinder 训练 (产生 parent loss) - 使用 ParentFinderGRU + Soft-Mask
4. Stage 4: RelationClassifier 训练 (产生 relation loss)

总 Loss = λ1 * L_cls + λ2 * L_par + λ3 * L_rel (论文公式)

基于 HuggingFace Trainer 实现，享受以下优势：
- 自动梯度累积和步数计数
- 自动混合精度训练
- 自动 checkpoint 管理
- 自动日志和评估

Usage:
    python examples/stage/train_joint.py --env test --dataset hrds

    # 快速测试
    python examples/stage/train_joint.py --env test --dataset hrds --quick

    # 使用完整论文方法（GRU + Soft-Mask）
    python examples/stage/train_joint.py --env test --dataset hrds --use_gru --use_soft_mask

    # 自定义 loss 权重
    python examples/stage/train_joint.py --env test --lambda_cls 1.0 --lambda_parent 0.5 --lambda_rel 0.5
"""

import logging
import os
import sys
import json
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, Counter

import numpy as np

# ==================== GPU 设置（必须在 import torch 之前）====================
# CUDA_VISIBLE_DEVICES must be set before importing torch
def _setup_gpu_early():
    """在 import torch 之前设置 GPU，避免 DataParallel 问题"""
    # 优先从命令行参数中提取 --gpu
    gpu_id = None
    env = "test"  # 默认值
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
        if arg == "--env" and i + 1 < len(sys.argv):
            env = sys.argv[i + 1]

    # 如果命令行指定了 GPU，直接使用
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return

    # 否则从配置文件加载
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, PROJECT_ROOT)
        from configs.config_loader import load_config
        config = load_config(env)
        if config.gpu.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices
    except Exception as e:
        pass  # 如果加载失败，使用默认 GPU 设置

_setup_gpu_early()

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.modeling_outputs import TokenClassifierOutput
from torch.optim import AdamW

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
STAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE_ROOT)

import layoutlmft.data.datasets.hrdoc
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.data.labels import LABEL_LIST, NUM_LABELS, get_id2label, get_label2id
from layoutlmft.models.layoutxlm import (
    LayoutXLMForTokenClassification,
    LayoutXLMConfig,
    LayoutXLMTokenizerFast,
)
from layoutlmft.models.relation_classifier import (
    LineFeatureExtractor,
    MultiClassRelationClassifier,
    compute_geometry_features,
    FocalLoss,
    RELATION_LABELS,
    RELATION_NAMES,
    NUM_RELATIONS,
)

from joint_data_collator import HRDocJointDataCollator, HRDocDocumentLevelCollator
from train_parent_finder import (
    SimpleParentFinder,
    ParentFinderGRU,
    build_child_parent_matrix,
)
from tasks.parent_finding import ChildParentDistributionMatrix, build_child_parent_matrix_from_dataset

# 从共享模块导入 JointModel
EXAMPLES_ROOT = os.path.dirname(STAGE_ROOT)
sys.path.insert(0, EXAMPLES_ROOT)
from models.joint_model import JointModel
from models.build import build_joint_model

from util.eval_utils import (
    compute_macro_f1,
    log_per_class_metrics,
    aggregate_token_to_line_predictions,
)
from util.checkpoint_utils import get_latest_checkpoint, get_best_model
from util.experiment_manager import ensure_experiment
from data import HRDocDataLoader, HRDocDataLoaderConfig, tokenize_page_with_line_boundary

# HRDoc 评估工具
HRDOC_UTILS_PATH = os.path.join(PROJECT_ROOT, "HRDoc", "utils")
sys.path.insert(0, HRDOC_UTILS_PATH)

logger = logging.getLogger(__name__)


# ==================== 参数定义 ====================

@dataclass
class JointModelArguments:
    """联合模型参数"""
    model_name_or_path: str = field(
        default=None, metadata={"help": "Stage 1 model path (if None, auto-detect from experiment)"}
    )
    model_path: str = field(
        default=None, metadata={"help": "Alias for model_name_or_path (shorter name)"}
    )
    use_gru: bool = field(default=True, metadata={"help": "Use GRU decoder (paper method)"})
    use_soft_mask: bool = field(default=True, metadata={"help": "Use Soft-Mask (paper method)"})
    use_focal_loss: bool = field(default=True, metadata={"help": "Use Focal Loss"})
    lambda_cls: float = field(default=1.0, metadata={"help": "Classification loss weight"})
    lambda_parent: float = field(default=1.0, metadata={"help": "Parent loss weight"})
    lambda_rel: float = field(default=1.0, metadata={"help": "Relation loss weight"})
    stage1_micro_batch_size: int = field(default=1, metadata={"help": "Stage1 micro-batch size (can be larger when Stage1 is frozen)"})
    stage1_no_grad: bool = field(default=True, metadata={"help": "Freeze Stage1 (saves memory, only train Stage3/4)"})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing for Stage1 (saves ~50-70% GPU memory, slower training)"})
    freeze_visual: bool = field(default=False, metadata={"help": "Freeze visual encoder (ResNet) while training Transformer. Use with stage1_no_grad=False."})


@dataclass
class JointDataArguments:
    """联合训练数据参数"""
    env: str = field(default="test", metadata={"help": "Environment: dev, test"})
    dataset: str = field(default="hrds", metadata={"help": "Dataset: hrds, hrdh, tender"})
    gpu: Optional[str] = field(default=None, metadata={"help": "GPU ID to use (e.g., '0', '0,1'). Overrides config file."})
    covmatch: Optional[str] = field(default=None, metadata={"help": "Covmatch split name (e.g., doc_covmatch_dev50_seed42). If not specified, uses config default."})
    max_train_samples: int = field(default=-1, metadata={"help": "Max train samples (-1 for all)"})
    max_eval_samples: int = field(default=-1, metadata={"help": "Max eval samples (-1 for all)"})
    force_rebuild: bool = field(default=False, metadata={"help": "Force rebuild dataset (delete HuggingFace cache and regenerate)"})
    document_level: bool = field(default=True, metadata={"help": "Use document-level batching (preserves cross-page relations)"})


@dataclass
class JointTrainingArguments(TrainingArguments):
    """
    扩展 HuggingFace TrainingArguments，添加联合训练特定参数
    """
    # 覆盖 output_dir 添加默认值
    output_dir: str = field(default="./output/joint", metadata={"help": "Output directory"})

    # 禁用 TensorBoard（Python 3.12 移除了 distutils，TensorBoard 初始化会报错）
    report_to: str = field(default="none", metadata={"help": "Disable TensorBoard to avoid distutils.version issue in Python 3.12+"})

    # 评估设置（覆盖默认值）
    evaluation_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy"})
    eval_steps: int = field(default=500, metadata={"help": "Evaluation steps"})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy"})
    save_steps: int = field(default=500, metadata={"help": "Save steps"})
    save_total_limit: int = field(default=3, metadata={"help": "Maximum number of checkpoints to keep"})
    logging_steps: int = field(default=100, metadata={"help": "Logging steps"})

    # 实验管理
    exp: str = field(default=None, metadata={"help": "Experiment ID"})
    new_exp: bool = field(default=False, metadata={"help": "Create a new experiment"})
    exp_name: str = field(default="", metadata={"help": "Name for new experiment"})

    # 快速测试
    quick: bool = field(default=False, metadata={"help": "Quick test mode"})

    # 最小训练步数（小数据集时自动增加epoch）
    min_steps: int = field(default=1000, metadata={"help": "Minimum training steps. If calculated steps < min_steps, epochs will be increased."})

    # Stage 3/4 单独学习率
    learning_rate_stage34: float = field(default=5e-4, metadata={"help": "Learning rate for Stage 3/4"})

    # 功能开关
    disable_stage34: bool = field(default=False, metadata={"help": "Disable Stage 3/4"})

    # 断点续训（默认自动检测）
    resume_from_checkpoint: str = field(
        default="auto",
        metadata={"help": "Path to checkpoint to resume from. 'auto'=auto-detect, 'none'=start fresh."}
    )

    # 兼容旧参数
    dry_run: bool = field(default=False, metadata={"help": "Dry run mode"})

    # DataLoader 设置（默认单进程，禁用预取，避免内存爆炸）
    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of dataloader workers (0=single process)"})
    dataloader_pin_memory: bool = field(default=False, metadata={"help": "Pin memory for faster GPU transfer"})
    dataloader_drop_last: bool = field(default=False, metadata={"help": "Drop last incomplete batch"})


# ==================== 自定义 Trainer ====================
# 注意：JointModel 从 models.joint_model 导入（第123行），不在此处定义

class JointTrainer(Trainer):
    """
    联合训练 Trainer，继承 HuggingFace Trainer

    自定义内容：
    1. create_optimizer: 为 Stage 1 和 Stage 3/4 设置不同学习率
    2. compute_loss: 直接使用 JointModel 的 forward 返回的 loss
    3. log: 记录多个 loss（cls_loss, parent_loss, rel_loss）
    """

    def __init__(
        self,
        model_args: JointModelArguments = None,
        learning_rate_stage34: float = 5e-4,
        document_level: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_args = model_args
        self.learning_rate_stage34 = learning_rate_stage34
        self.document_level = document_level

        # 用于记录各 loss
        self._current_loss_dict = {}

    def create_optimizer(self):
        """创建优化器，为不同模块设置不同学习率"""

        if self.optimizer is not None:
            return self.optimizer

        model = self.model

        # 分组参数
        optimizer_grouped_parameters = [
            # Stage 1: 使用 args.learning_rate
            {
                "params": [p for n, p in model.stage1.named_parameters() if p.requires_grad],
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            # Stage 3: 使用 learning_rate_stage34
            {
                "params": [p for n, p in model.stage3.named_parameters() if p.requires_grad],
                "lr": self.learning_rate_stage34,
                "weight_decay": 0.0,
            },
            # Stage 4: 使用 learning_rate_stage34
            {
                "params": [p for n, p in model.stage4.named_parameters() if p.requires_grad],
                "lr": self.learning_rate_stage34,
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算 loss，JointModel.forward 已经返回组合后的 loss"""

        outputs = model(**inputs)
        loss = outputs.loss  # TokenClassifierOutput 使用属性访问

        # 从模型实例获取完整的 outputs 字典（包含各阶段 loss）
        outputs_dict = getattr(model, "_outputs_dict", {})

        # 记录各 loss 用于日志
        self._current_loss_dict = {
            "cls_loss": outputs_dict.get("cls_loss", torch.tensor(0.0)).item() if "cls_loss" in outputs_dict else 0.0,
        }
        if "parent_loss" in outputs_dict:
            self._current_loss_dict["parent_loss"] = outputs_dict["parent_loss"].item()
        if "rel_loss" in outputs_dict:
            self._current_loss_dict["rel_loss"] = outputs_dict["rel_loss"].item()
        # accuracy 从模型实例属性读取
        if hasattr(model, "_parent_acc"):
            self._current_loss_dict["parent_acc"] = model._parent_acc
        if hasattr(model, "_rel_acc"):
            self._current_loss_dict["rel_acc"] = model._rel_acc

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """扩展日志，添加各阶段 loss"""

        # 添加各 loss 到日志
        if self._current_loss_dict:
            logs.update(self._current_loss_dict)

        super().log(logs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写评估方法

        文档级别模式下跳过 Trainer 内置的 prediction_loop（形状不一致会失败），
        只触发 callbacks（E2EEvaluationCallback 会运行正确的评估）。
        """
        if self.document_level:
            # 文档级别：跳过 prediction_loop，直接调用 callbacks
            self.model.eval()

            # 手动遍历 callbacks 并调用 on_evaluate（传递 model）
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, 'on_evaluate'):
                    callback.on_evaluate(
                        self.args, self.state, self.control,
                        model=self.model, metrics={}
                    )

            self.model.train()
            return {}
        else:
            # 页面级别：正常评估
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """保存模型，分别保存各 Stage + Trainer 状态"""

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Stage 1: 保存完整模型（config + 权重），可独立加载
        stage1_dir = os.path.join(output_dir, "stage1")
        self.model.stage1.save_pretrained(stage1_dir)

        # Stage 3/4: 保存为 PyTorch 格式（用于单独加载推理，续训时由 pytorch_model.bin 恢复）
        torch.save(self.model.stage3.state_dict(), os.path.join(output_dir, "stage3.pt"))
        torch.save(self.model.stage4.state_dict(), os.path.join(output_dir, "stage4.pt"))

        # 保存完整模型状态（标准格式，供 Trainer resume 使用）
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        # 保存 tokenizer 到根目录（两种格式，确保 Fast tokenizer 可加载）
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir, legacy_format=True)   # sentencepiece
            self.tokenizer.save_pretrained(output_dir, legacy_format=False)  # tokenizer.json

        # 保存 trainer_state.json（用于续训）
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        logger.info(f"Model saved to {output_dir}")


# ==================== Callback 定义 ====================

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
        # 尝试通过 gc 获取 GradScaler 实例
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
        except Exception as e:
            pass  # scaler 可能不可用


class JointLoggingCallback(TrainerCallback):
    """记录联合训练的详细日志（美化版）"""

    def __init__(self, total_steps: int = None):
        self.total_steps = total_steps
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

        # 学习率
        lr = logs.get("learning_rate", 0)
        lr_str = f"{lr:.2e}" if lr > 0 else "N/A"

        # 各任务指标
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

        # 构建输出
        header = f"Step {step:>5}/{total} [{bar}] {progress*100:>5.1f}%  lr={lr_str}"

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
    端到端评估 Callback

    在每次评估时运行完整的端到端评估：
    - Stage 1: 分类 (Line-level Macro/Micro F1)
    - Stage 3: Parent 准确率
    - Stage 4: Relation 准确率 + Macro F1

    使用新的 engines/evaluator.py 统一接口
    """

    def __init__(self, eval_dataloader, data_collator, compute_teds: bool = False):
        self.eval_dataloader = eval_dataloader
        self.data_collator = data_collator
        self.compute_teds = compute_teds
        self._evaluator = None
        # 历史评估记录：[(step, line_f1, parent_acc, rel_f1), ...]
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

        # 使用新的 Evaluator（统一接口，支持 page/doc 级别）
        from engines.evaluator import Evaluator

        evaluator = Evaluator(model, device)
        output = evaluator.evaluate(
            self.eval_dataloader,
            compute_teds=self.compute_teds,
            verbose=True,
        )

        # 打印结果
        line_macro = output.line_macro_f1 * 100
        line_acc = output.line_accuracy * 100
        parent_acc = output.parent_accuracy * 100
        rel_acc = output.relation_accuracy * 100
        rel_macro = output.relation_macro_f1 * 100
        num_lines = output.num_lines

        # 格式化 delta（带符号和颜色指示）
        def fmt_delta(d, threshold=0.5):
            if d >= threshold:
                return f"↑{d:+.1f}"
            elif d <= -threshold:
                return f"↓{d:+.1f}"
            else:
                return f" {d:+.1f}"

        # 计算与历史平均值的对比
        avg_n = min(3, len(self.history))

        # 表格输出
        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════╗")
        logger.info(f"║           Evaluation Results @ Step {global_step:<6}                  ║")
        logger.info("╠══════════════════════════════════════════════════════════════╣")

        if avg_n > 0:
            recent = self.history[-avg_n:]
            avg_line = sum(h[1] for h in recent) / avg_n
            avg_parent = sum(h[2] for h in recent) / avg_n
            avg_rel = sum(h[3] for h in recent) / avg_n

            delta_line = line_macro - avg_line
            delta_parent = parent_acc - avg_parent
            delta_rel = rel_macro - avg_rel

            logger.info(f"║  Metric       │ Current  │  Avg({avg_n})  │  Delta       ║")
            logger.info("║───────────────┼──────────┼──────────┼──────────────║")
            logger.info(f"║  Line(F1)     │  {line_macro:>5.1f}%  │  {avg_line:>5.1f}%  │  {fmt_delta(delta_line):>6}      ║")
            logger.info(f"║  Parent(Acc)  │  {parent_acc:>5.1f}%  │  {avg_parent:>5.1f}%  │  {fmt_delta(delta_parent):>6}      ║")
            logger.info(f"║  Rel(F1)      │  {rel_macro:>5.1f}%  │  {avg_rel:>5.1f}%  │  {fmt_delta(delta_rel):>6}      ║")

            summary = f"[Step {global_step}] Line={line_macro:.1f}% | Parent={parent_acc:.1f}% ({fmt_delta(delta_parent)}) | Rel={rel_macro:.1f}% ({fmt_delta(delta_rel)})"
        else:
            logger.info(f"║  Metric       │ Current  │                           ║")
            logger.info("║───────────────┼──────────┼───────────────────────────║")
            logger.info(f"║  Line(F1)     │  {line_macro:>5.1f}%  │                           ║")
            logger.info(f"║  Parent(Acc)  │  {parent_acc:>5.1f}%  │                           ║")
            logger.info(f"║  Rel(F1)      │  {rel_macro:>5.1f}%  │                           ║")
            summary = f"[Step {global_step}] Line={line_macro:.1f}% | Parent={parent_acc:.1f}% | Rel={rel_macro:.1f}%"

        logger.info("╠══════════════════════════════════════════════════════════════╣")
        logger.info(f"║  Lines evaluated: {num_lines:<43} ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")
        logger.info(summary)

        # 保存到历史记录
        self.history.append((global_step, line_macro, parent_acc, rel_macro))

    def _evaluate_e2e_legacy(self, model, device, global_step: int) -> Dict[str, float]:
        """运行端到端评估 - 调用共享的 hrdoc_eval.evaluate_e2e"""
        from util.hrdoc_eval import evaluate_e2e
        from types import SimpleNamespace

        # 创建 args 对象传递 compute_teds 设置
        # hrdoc_eval.evaluate_e2e 通过 args.quick 控制是否计算 TEDS
        eval_args = SimpleNamespace(quick=not self.compute_teds)

        # 调用共享的评估函数
        results = evaluate_e2e(
            model=model,
            eval_loader=self.eval_dataloader,
            device=device,
            args=eval_args,
            global_step=global_step,
        )

        model.train()
        return results


# ==================== 数据处理 ====================

def compute_line_bboxes(token_bboxes: List[List[int]], token_line_ids: List[int]) -> List[List[float]]:
    """从 token bboxes 计算 line bboxes"""
    line_bbox_dict = defaultdict(lambda: [1e9, 1e9, -1e9, -1e9])

    for bbox, line_id in zip(token_bboxes, token_line_ids):
        if line_id < 0:
            continue
        x1, y1, x2, y2 = bbox
        line_bbox_dict[line_id][0] = min(line_bbox_dict[line_id][0], x1)
        line_bbox_dict[line_id][1] = min(line_bbox_dict[line_id][1], y1)
        line_bbox_dict[line_id][2] = max(line_bbox_dict[line_id][2], x2)
        line_bbox_dict[line_id][3] = max(line_bbox_dict[line_id][3], y2)

    if not line_bbox_dict:
        return []

    max_line_id = max(line_bbox_dict.keys())
    result = []
    for i in range(max_line_id + 1):
        if i in line_bbox_dict:
            result.append(line_bbox_dict[i])
        else:
            result.append([0, 0, 0, 0])

    return result


def prepare_datasets(tokenizer, data_args: JointDataArguments, training_args: JointTrainingArguments):
    """
    准备训练和评估数据集

    使用统一的数据加载模块，实现按行边界切分的 tokenization：
    - 确保一整行不会被截断到两个 chunk 中
    - 如果当前 chunk 放不下完整的一行，该行会被放到下一个 chunk
    """
    # 创建数据加载器配置
    num_workers = 1 if data_args.max_train_samples and data_args.max_train_samples < 100 else 4

    loader_config = HRDocDataLoaderConfig(
        data_dir=os.environ.get("HRDOC_DATA_DIR"),
        dataset_name=data_args.dataset,  # 使用数据集名称区分缓存（hrds, hrdh, tender 等）
        max_length=512,
        preprocessing_num_workers=num_workers,
        max_train_samples=data_args.max_train_samples if data_args.max_train_samples > 0 else None,
        max_val_samples=data_args.max_eval_samples if data_args.max_eval_samples > 0 else None,
        force_rebuild=data_args.force_rebuild,
        document_level=data_args.document_level,  # False=页面级别（快），True=文档级别（慢）
    )

    # 创建数据加载器（统一使用 HRDocDataLoader）
    data_loader = HRDocDataLoader(
        tokenizer=tokenizer,
        config=loader_config,
        include_line_info=True,  # 联合训练需要 line_ids, line_parent_ids, line_relations
    )

    # 统一使用 load_raw_datasets() 方法加载数据
    data_loader.load_raw_datasets()

    # 准备 tokenized 数据集
    tokenized_datasets = data_loader.prepare_datasets()

    train_dataset = tokenized_datasets.get("train")
    eval_dataset = tokenized_datasets.get("validation")

    # 获取原始训练数据集（用于构建 M_cp 矩阵）
    raw_train_dataset = data_loader._raw_datasets.get("train")

    logger.info(f"Train dataset: {len(train_dataset) if train_dataset else 0} samples")
    logger.info(f"Eval dataset: {len(eval_dataset) if eval_dataset else 0} samples")

    return train_dataset, eval_dataset, raw_train_dataset


# ==================== Line-level 评估指标 ====================

def compute_line_level_metrics(eval_pred):
    """计算 line-level 评估指标"""
    predictions, labels = eval_pred

    # predictions: [batch, seq_len, num_labels]
    # labels: [batch, seq_len]
    predictions = np.argmax(predictions, axis=2)

    # 计算准确率和 F1
    all_preds = []
    all_labels = []

    for pred, label in zip(predictions, labels):
        for p, l in zip(pred, label):
            if l != -100:
                all_preds.append(p)
                all_labels.append(l)

    if not all_labels:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }


# ==================== 配置加载 ====================

def load_config_and_setup(data_args: JointDataArguments, training_args: JointTrainingArguments, model_args: JointModelArguments):
    """加载配置并设置环境"""

    from configs.config_loader import load_config

    config = load_config(data_args.env)
    if training_args.quick:
        config.quick_test.enabled = True
    config = config.get_effective_config()

    # 设置 HuggingFace 缓存目录（与 train_stage1.py 保持一致）
    if config.paths.hf_cache_dir:
        os.environ["HF_HOME"] = config.paths.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = config.paths.hf_cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(config.paths.hf_cache_dir, "datasets")

    # 初始化实验管理器
    exp_manager, exp_dir = ensure_experiment(
        config,
        exp=training_args.exp,
        new_exp=training_args.new_exp,
        name=training_args.exp_name or f"Joint {data_args.dataset.upper()}",
    )

    # 设置模型路径（优先级：model_path > model_name_or_path > 自动检测）
    if model_args.model_path:
        model_args.model_name_or_path = model_args.model_path
        logger.info(f"Using manually specified model: {model_args.model_name_or_path}")

    if model_args.model_name_or_path is None:
        stage1_dir = exp_manager.get_stage_dir(training_args.exp, "stage1", data_args.dataset)
        stage1_model = get_latest_checkpoint(stage1_dir)

        if stage1_model:
            model_args.model_name_or_path = stage1_model
            logger.info(f"Using Stage 1 model from experiment: {stage1_model}")
        else:
            if hasattr(config.paths, 'stage1_model_path') and config.paths.stage1_model_path:
                legacy_dir = f"{config.paths.stage1_model_path}_{data_args.dataset}"
                stage1_model = get_latest_checkpoint(legacy_dir)
                if stage1_model:
                    model_args.model_name_or_path = stage1_model

            if model_args.model_name_or_path is None:
                model_args.model_name_or_path = config.model.local_path or config.model.name_or_path
                logger.warning(f"No Stage 1 model found, using pretrained: {model_args.model_name_or_path}")

    # 设置输出目录
    if training_args.output_dir == "./output/joint":
        training_args.output_dir = exp_manager.get_stage_dir(training_args.exp, "joint", data_args.dataset)

    # 数据目录
    data_dir = config.dataset.get_data_dir(data_args.dataset)
    os.environ["HRDOC_DATA_DIR"] = data_dir

    # tender 数据集默认不使用缓存（数据量小，避免缓存问题）
    if data_args.dataset == "tender" and not data_args.force_rebuild:
        data_args.force_rebuild = True
        logger.info("tender dataset: force_rebuild enabled by default")

    # Covmatch 目录 (命令行参数优先于配置文件)
    covmatch_from_cli = data_args.covmatch is not None
    if covmatch_from_cli:
        # 使用命令行指定的 covmatch
        config.dataset.covmatch = data_args.covmatch
        logger.info(f"Using covmatch from command line: {data_args.covmatch}")
    covmatch_dir = config.dataset.get_covmatch_dir(data_args.dataset)
    if os.path.exists(covmatch_dir):
        os.environ["HRDOC_SPLIT_DIR"] = covmatch_dir
        logger.info(f"Covmatch directory: {covmatch_dir}")
    else:
        if covmatch_from_cli:
            # 命令行明确指定了 covmatch，但目录不存在，退出
            logger.error(f"Covmatch directory not found: {covmatch_dir}")
            logger.error(f"Specified covmatch '{data_args.covmatch}' does not exist.")
            logger.error(f"Available covmatch directories can be found in: {os.path.dirname(covmatch_dir)}")
            # 列出可用的 covmatch 目录
            parent_dir = os.path.dirname(covmatch_dir)
            if os.path.exists(parent_dir):
                available = [d for d in os.listdir(parent_dir) if d.startswith("doc_covmatch")]
                if available:
                    logger.error(f"Available covmatch options: {', '.join(sorted(available))}")
                else:
                    logger.error(f"No covmatch directories found in {parent_dir}")
            sys.exit(1)
        else:
            # 使用配置文件默认值，只是警告
            logger.warning(f"Covmatch directory not found: {covmatch_dir}, using default directory structure")

    # GPU 设置
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    return config, data_dir, exp_manager


# ==================== 主函数 ====================

def main():
    # 解析参数
    parser = HfArgumentParser((JointModelArguments, JointDataArguments, JointTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # 加载配置
    config, data_dir, exp_manager = load_config_and_setup(data_args, training_args, model_args)

    # Quick 模式
    if training_args.quick:
        logger.info("=" * 60)
        logger.info("QUICK TEST MODE")
        logger.info("=" * 60)
        if data_args.max_train_samples == -1:
            data_args.max_train_samples = 10
        if data_args.max_eval_samples == -1:
            data_args.max_eval_samples = 10
        # max_steps=-1 表示按 epoch 训练，quick 模式强制使用 max_steps
        if training_args.max_steps <= 0 or training_args.max_steps > 20:
            training_args.max_steps = 20
        # 设置评估策略为按步数评估（而不是按 epoch）
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = 10
        training_args.save_steps = 10
        training_args.logging_steps = 5

    # Document-level 模式：调整 eval/save steps（文档级别训练较慢）
    if data_args.document_level and not training_args.quick:
        training_args.eval_steps = 25
        training_args.save_steps = 25
        training_args.logging_steps = 10

    # 设置随机种子
    set_seed(training_args.seed)

    # 打印配置
    logger.info("=" * 60)
    logger.info("Joint Training Configuration (HuggingFace Trainer)")
    logger.info("=" * 60)
    logger.info(f"Environment:    {data_args.env}")
    logger.info(f"Dataset:        {data_args.dataset}")
    logger.info(f"Quick Mode:     {training_args.quick}")
    logger.info(f"Model Path:     {model_args.model_name_or_path}")
    logger.info(f"Output Dir:     {training_args.output_dir}")
    logger.info("-" * 60)
    logger.info("Loss Weights:")
    logger.info(f"  lambda_cls:    {model_args.lambda_cls}")
    logger.info(f"  lambda_parent: {model_args.lambda_parent}")
    logger.info(f"  lambda_rel:    {model_args.lambda_rel}")
    logger.info("-" * 60)
    logger.info("Training Parameters:")
    logger.info(f"  max_steps:     {training_args.max_steps}")
    logger.info(f"  min_steps:     {training_args.min_steps}")
    logger.info(f"  batch_size:    {training_args.per_device_train_batch_size}")
    logger.info(f"  grad_accum:    {training_args.gradient_accumulation_steps}")
    logger.info(f"  learning_rate: {training_args.learning_rate} (Stage 1)")
    logger.info(f"  lr_stage34:    {training_args.learning_rate_stage34} (Stage 3/4)")
    logger.info(f"  fp16:          {training_args.fp16}")
    logger.info(f"  eval_steps:    {training_args.eval_steps}")
    logger.info(f"  save_steps:    {training_args.save_steps}")
    logger.info(f"  logging_steps: {training_args.logging_steps}")
    logger.info(f"  doc_level:     {data_args.document_level}")
    logger.info(f"  stage1_no_grad:       {model_args.stage1_no_grad}")
    logger.info(f"  gradient_checkpoint:  {model_args.gradient_checkpointing}")
    logger.info("=" * 60)

    if training_args.dry_run:
        logger.info("[Dry run mode - exiting]")
        return

    os.makedirs(training_args.output_dir, exist_ok=True)

    # 标记 stage 开始
    exp_manager.mark_stage_started(training_args.exp, "joint", data_args.dataset)

    # 加载 tokenizer（统一从 model_path 根目录加载）
    logger.info("Loading tokenizer...")
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(model_args.model_name_or_path)
    assert tokenizer.is_fast, "LayoutXLM requires fast tokenizer"

    # 准备数据集
    logger.info("Preparing datasets...")
    train_dataset, eval_dataset, raw_train_dataset = prepare_datasets(tokenizer, data_args, training_args)
    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset) if eval_dataset else 0}")

    # 如果 eval_dataset 为空，禁用评估
    if eval_dataset is None or len(eval_dataset) == 0:
        logger.warning("No eval dataset available, disabling evaluation during training")
        training_args.evaluation_strategy = "no"
        training_args.eval_steps = None

    # 自动调整 epoch（小数据集时确保达到最小训练步数）
    if not training_args.quick and train_dataset is not None and training_args.max_steps <= 0:
        # 计算当前配置下的总训练步数
        effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        steps_per_epoch = max(len(train_dataset) // effective_batch_size, 1)
        current_total_steps = steps_per_epoch * int(training_args.num_train_epochs)

        # 如果总步数小于 min_steps，自动增加 epoch
        if current_total_steps < training_args.min_steps:
            # 计算需要多少个 epoch 才能达到 min_steps
            required_epochs = (training_args.min_steps + steps_per_epoch - 1) // steps_per_epoch
            new_total_steps = steps_per_epoch * required_epochs
            logger.info(f"Auto-adjusting for small dataset (min_steps={training_args.min_steps}):")
            logger.info(f"  steps_per_epoch: {steps_per_epoch}")
            logger.info(f"  current_steps:   {current_total_steps} (epochs={int(training_args.num_train_epochs)})")
            logger.info(f"  adjusted_steps:  {new_total_steps} (epochs={required_epochs})")
            training_args.num_train_epochs = float(required_epochs)

    # Data collator（根据模式选择）
    if data_args.document_level:
        logger.info("Using DOCUMENT-LEVEL collator (slow, for inference)")
        data_collator = HRDocDocumentLevelCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=512,
        )
    else:
        logger.info("Using PAGE-LEVEL collator (fast training)")
        data_collator = HRDocJointDataCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=512,
        )

    # 加载模型
    logger.info("Loading models...")

    # 检测是否从 joint checkpoint 续训
    joint_checkpoint = None

    if training_args.resume_from_checkpoint and training_args.resume_from_checkpoint.lower() != "none":
        if training_args.resume_from_checkpoint == "auto":
            joint_checkpoint = get_last_checkpoint(training_args.output_dir)
        else:
            joint_checkpoint = training_args.resume_from_checkpoint

        # 验证 checkpoint 有效性（必须有 pytorch_model.bin + trainer_state.json）
        if joint_checkpoint:
            has_pytorch_model = os.path.isfile(os.path.join(joint_checkpoint, "pytorch_model.bin"))
            has_trainer_state = os.path.isfile(os.path.join(joint_checkpoint, "trainer_state.json"))

            if has_pytorch_model and has_trainer_state:
                logger.info(f"Found valid checkpoint: {joint_checkpoint}")
            else:
                logger.warning(f"Invalid checkpoint (missing pytorch_model.bin or trainer_state.json): {joint_checkpoint}")
                logger.warning("Please delete old checkpoints and restart training")
                joint_checkpoint = None

    # Stage 1: LayoutXLM
    # 优先级：model_path 显式指定 > joint_checkpoint/stage1 > 自动检测
    joint_model_path = None  # 如果 model_path 是 joint checkpoint，记录下来用于后续加载完整权重

    if model_args.model_path:
        # 用户显式指定了 model_path，检测是否是 joint checkpoint 结构
        specified_path = model_args.model_name_or_path
        stage1_subdir = os.path.join(specified_path, "stage1")
        joint_pytorch_model = os.path.join(specified_path, "pytorch_model.bin")

        if os.path.isfile(os.path.join(stage1_subdir, "config.json")) and os.path.isfile(joint_pytorch_model):
            # 是 joint checkpoint 结构
            stage1_path = stage1_subdir
            joint_model_path = specified_path
            logger.info(f"Loading from joint checkpoint: {specified_path}")
            logger.info(f"  - Stage1 config from: {stage1_path}")
        else:
            # 标准 LayoutXLM 目录结构
            stage1_path = specified_path
            logger.info(f"Loading Stage 1 from specified path: {stage1_path}")
    elif joint_checkpoint:
        # 续训且未指定 model_path，从 checkpoint 加载
        stage1_path = os.path.join(joint_checkpoint, "stage1")
        logger.info(f"Resuming: Loading Stage 1 from checkpoint: {stage1_path}")
    else:
        # 首次训练，使用自动检测的路径
        stage1_path = model_args.model_name_or_path
        logger.info(f"Fresh start: Loading Stage 1 from: {stage1_path}")

    stage1_config = LayoutXLMConfig.from_pretrained(stage1_path)
    stage1_config.num_labels = NUM_LABELS
    stage1_config.id2label = get_id2label()
    stage1_config.label2id = get_label2id()

    if joint_model_path:
        # Joint checkpoint: config from stage1/, weights will be loaded later from joint pytorch_model.bin
        # 创建模型结构，权重稍后从 joint checkpoint 加载
        stage1_model = LayoutXLMForTokenClassification(config=stage1_config)
        logger.info("Created Stage1 model structure (weights will be loaded from joint checkpoint)")
    else:
        # Regular checkpoint or base model: load config and weights together
        stage1_model = LayoutXLMForTokenClassification.from_pretrained(
            stage1_path,
            config=stage1_config,
        )

    # Enable Gradient Checkpointing for Stage1 (saves ~50-70% GPU memory)
    if model_args.gradient_checkpointing:
        # LayoutXLM/LayoutLMv2 使用 config.gradient_checkpointing 而不是方法
        stage1_model.config.gradient_checkpointing = True
        logger.info("Gradient Checkpointing ENABLED for Stage1 (memory saving mode)")

    # Stage 2: Feature Extractor
    feature_extractor = LineFeatureExtractor()

    # Stage 3: ParentFinder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_args.use_gru:
        logger.info("Using ParentFinderGRU (paper method)")
        stage3_model = ParentFinderGRU(
            hidden_size=768,
            gru_hidden_size=512,
            num_classes=NUM_LABELS,
            dropout=0.1,
            use_soft_mask=model_args.use_soft_mask,
        )

        if model_args.use_soft_mask:
            if joint_checkpoint or joint_model_path:
                # 续训或从 joint checkpoint 加载：M_cp 会从 checkpoint 的 state_dict 恢复，跳过构建
                logger.info("M_cp will be restored from checkpoint (skipping build)")
            else:
                # 首次训练：从数据集构建 M_cp
                logger.info("Building Child-Parent Distribution Matrix (M_cp)...")
                cp_matrix = build_child_parent_matrix_from_dataset(raw_train_dataset, num_classes=NUM_LABELS)
                stage3_model.set_child_parent_matrix(cp_matrix.get_tensor(device))
                logger.info("M_cp initialized successfully")

                # 打印 M_cp 矩阵用于调试
                if hasattr(stage3_model, 'M_cp'):
                    np.set_printoptions(precision=3, suppress=True, linewidth=200)
                    logger.info(f"M_cp matrix shape: {stage3_model.M_cp.shape}")
                    logger.info(f"M_cp matrix (rows=parent[ROOT,0-13], cols=child[0-13]):\n{stage3_model.M_cp.cpu().numpy()}")
                    # 打印每个 child 类别最可能的 parent
                    m_cp = stage3_model.M_cp.cpu().numpy()
                    logger.info("Top parents for each child class:")
                    for child_idx, child_name in enumerate(LABEL_LIST):
                        parent_probs = m_cp[:, child_idx]
                        top_parents = np.argsort(parent_probs)[::-1][:3]
                        top_info = []
                        for p_idx in top_parents:
                            p_name = "ROOT" if p_idx == 0 else LABEL_LIST[p_idx - 1]
                            top_info.append(f"{p_name}:{parent_probs[p_idx]:.3f}")
                        logger.info(f"  {child_name}: {', '.join(top_info)}")
    else:
        logger.info("Using SimpleParentFinder")
        stage3_model = SimpleParentFinder(hidden_size=768, dropout=0.1)

    # Stage 4: RelationClassifier (论文对齐：使用 GRU 隐状态，只有3类，不使用几何特征)
    # 公式：P_rel_(i,j) = softmax(LinearProj(Concat(h_i, h_j)))
    # h_i, h_j 是 GRU 隐状态，维度是 gru_hidden_size=512
    gru_hidden_size = 512 if model_args.use_gru else 768
    stage4_model = MultiClassRelationClassifier(
        hidden_size=gru_hidden_size,  # GRU hidden size（论文对齐）
        num_relations=NUM_RELATIONS,  # 3类: connect, contain, equality
        use_geometry=False,  # 论文不使用几何特征
        dropout=0.1,
    )

    # 联合模型
    model = JointModel(
        stage1_model=stage1_model,
        stage3_model=stage3_model,
        stage4_model=stage4_model,
        feature_extractor=feature_extractor,
        lambda_cls=model_args.lambda_cls,
        lambda_parent=model_args.lambda_parent if not training_args.disable_stage34 else 0.0,
        lambda_rel=model_args.lambda_rel if not training_args.disable_stage34 else 0.0,
        use_focal_loss=model_args.use_focal_loss,
        use_gru=model_args.use_gru,
        stage1_micro_batch_size=model_args.stage1_micro_batch_size,
        freeze_visual=model_args.freeze_visual,
        stage1_no_grad=model_args.stage1_no_grad,
    )
    logger.info(f"Stage1: micro_batch={model_args.stage1_micro_batch_size}, no_grad={model_args.stage1_no_grad}, grad_ckpt={model_args.gradient_checkpointing}, freeze_visual={model_args.freeze_visual}")

    # 如果从 joint checkpoint 加载，加载完整权重
    if joint_model_path:
        joint_weights_path = os.path.join(joint_model_path, "pytorch_model.bin")
        logger.info(f"Loading joint model weights from: {joint_weights_path}")
        state_dict = torch.load(joint_weights_path, map_location="cpu")
        # 加载权重，strict=False 允许部分匹配（如果模型结构略有不同）
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading joint model: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading joint model: {unexpected_keys}")
        logger.info("Joint model weights loaded successfully")

    # 打印模型参数量
    stage1_params = sum(p.numel() for p in model.stage1.parameters())
    stage3_params = sum(p.numel() for p in model.stage3.parameters())
    stage4_params = sum(p.numel() for p in model.stage4.parameters())
    total_params = stage1_params + stage3_params + stage4_params
    logger.info(f"Model parameters: Stage1={stage1_params:,}, Stage3={stage3_params:,}, Stage4={stage4_params:,}, Total={total_params:,}")

    # 创建评估 DataLoader（用于端到端评估）
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0,
        )

    # 创建 callbacks
    callbacks = [JointLoggingCallback(), AMPDiagnosticCallback()]
    if eval_dataloader is not None:
        # 添加端到端评估 callback（评估 Parent 和 Relation 准确率）
        callbacks.append(E2EEvaluationCallback(
            eval_dataloader=eval_dataloader,
            data_collator=data_collator,
            compute_teds=not training_args.quick,
        ))

    # 创建 Trainer
    trainer = JointTrainer(
        model=model,
        args=training_args,
        model_args=model_args,
        learning_rate_stage34=training_args.learning_rate_stage34,
        document_level=data_args.document_level,  # 文档级别模式下跳过 prediction_loop
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_line_level_metrics,
        callbacks=callbacks,
    )

    # 训练
    if joint_checkpoint:
        logger.info(f"Resuming from checkpoint: {joint_checkpoint}")
    else:
        logger.info("Starting fresh training")

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=joint_checkpoint)

    # 保存最终模型
    trainer.save_model()
    trainer.save_state()

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 最终评估
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()  # 文档级别模式会自动跳过 prediction_loop
        if eval_metrics:
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

    # 更新实验状态
    best_metric = metrics.get("train_loss", 0.0)
    exp_manager.mark_stage_completed(
        training_args.exp, "joint", data_args.dataset,
        best_checkpoint="final",
        metrics={"train_loss": float(best_metric)},
    )

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

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
    # 尝试从命令行参数中提取 --env
    env = "test"  # 默认值
    for i, arg in enumerate(sys.argv):
        if arg == "--env" and i + 1 < len(sys.argv):
            env = sys.argv[i + 1]
            break

    # 加载配置获取 GPU 设置
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

from joint_data_collator import HRDocJointDataCollator
from train_parent_finder import (
    SimpleParentFinder,
    ParentFinderGRU,
    ChildParentDistributionMatrix,
    build_child_parent_matrix,
)
from util.eval_utils import (
    compute_macro_f1,
    log_per_class_metrics,
    aggregate_token_to_line_predictions,
)
from util.checkpoint_utils import get_latest_checkpoint, get_best_model
from util.experiment_manager import ensure_experiment

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
    use_gru: bool = field(default=True, metadata={"help": "Use GRU decoder (paper method)"})
    use_soft_mask: bool = field(default=True, metadata={"help": "Use Soft-Mask (paper method)"})
    use_focal_loss: bool = field(default=True, metadata={"help": "Use Focal Loss"})
    lambda_cls: float = field(default=1.0, metadata={"help": "Classification loss weight"})
    lambda_parent: float = field(default=1.0, metadata={"help": "Parent loss weight"})
    lambda_rel: float = field(default=1.0, metadata={"help": "Relation loss weight"})


@dataclass
class JointDataArguments:
    """联合训练数据参数"""
    env: str = field(default="test", metadata={"help": "Environment: dev, test"})
    dataset: str = field(default="hrds", metadata={"help": "Dataset: hrds, hrdh"})
    max_train_samples: int = field(default=-1, metadata={"help": "Max train samples (-1 for all)"})
    max_eval_samples: int = field(default=-1, metadata={"help": "Max eval samples (-1 for all)"})


@dataclass
class JointTrainingArguments(TrainingArguments):
    """
    扩展 HuggingFace TrainingArguments，添加联合训练特定参数
    """
    # 覆盖 output_dir 添加默认值
    output_dir: str = field(default="./output/joint", metadata={"help": "Output directory"})

    # 评估设置（覆盖默认值）
    evaluation_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy"})
    eval_steps: int = field(default=500, metadata={"help": "Evaluation steps"})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy"})
    save_steps: int = field(default=500, metadata={"help": "Save steps"})
    logging_steps: int = field(default=100, metadata={"help": "Logging steps"})

    # 实验管理
    exp: str = field(default=None, metadata={"help": "Experiment ID"})
    new_exp: bool = field(default=False, metadata={"help": "Create a new experiment"})
    exp_name: str = field(default="", metadata={"help": "Name for new experiment"})

    # 快速测试
    quick: bool = field(default=False, metadata={"help": "Quick test mode"})

    # Stage 3/4 单独学习率
    learning_rate_stage34: float = field(default=5e-4, metadata={"help": "Learning rate for Stage 3/4"})

    # 功能开关
    disable_stage34: bool = field(default=False, metadata={"help": "Disable Stage 3/4"})

    # 兼容旧参数
    dry_run: bool = field(default=False, metadata={"help": "Dry run mode"})


# ==================== 联合模型定义 ====================

class JointModel(nn.Module):
    """
    联合模型：包含 Stage 1/2/3/4 的所有模块

    论文公式: L_total = L_cls + α₁·L_par + α₂·L_rel
    """

    def __init__(
        self,
        stage1_model: LayoutXLMForTokenClassification,
        stage3_model: nn.Module,
        stage4_model: MultiClassRelationClassifier,
        feature_extractor: LineFeatureExtractor,
        lambda_cls: float = 1.0,
        lambda_parent: float = 1.0,
        lambda_rel: float = 1.0,
        use_focal_loss: bool = True,
        use_gru: bool = False,
    ):
        super().__init__()

        self.stage1 = stage1_model
        self.stage3 = stage3_model
        self.stage4 = stage4_model
        self.feature_extractor = feature_extractor

        self.lambda_cls = lambda_cls
        self.lambda_parent = lambda_parent
        self.lambda_rel = lambda_rel
        self.use_gru = use_gru

        # 关系分类损失
        if use_focal_loss:
            self.relation_criterion = FocalLoss(gamma=2.0)
        else:
            self.relation_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor = None,
        labels: torch.Tensor = None,
        line_ids: Optional[torch.Tensor] = None,
        line_parent_ids: Optional[torch.Tensor] = None,
        line_relations: Optional[torch.Tensor] = None,
        line_bboxes: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """前向传播，返回 loss 和各阶段输出"""

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # ==================== Stage 1: Classification ====================
        stage1_outputs = self.stage1(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            labels=labels,
            output_hidden_states=True,
        )

        cls_loss = stage1_outputs.loss
        logits = stage1_outputs.logits
        hidden_states = stage1_outputs.hidden_states[-1]

        outputs = {
            "loss": cls_loss * self.lambda_cls,
            "cls_loss": cls_loss,
            "logits": logits,
        }

        # 如果没有 line 信息，直接返回（使用 TokenClassifierOutput 格式）
        if line_ids is None or line_parent_ids is None:
            return TokenClassifierOutput(
                loss=outputs["loss"],
                logits=logits,
            )

        # ==================== Stage 2: Feature Extraction ====================
        # 保持梯度流，让 Stage 3/4 的 loss 可以回传到 Stage 1
        text_seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :text_seq_len, :]

        line_features, line_mask = self.feature_extractor.extract_line_features(
            text_hidden, line_ids, pooling="mean"
        )

        # ==================== Stage 3: Parent Finding ====================
        parent_loss = torch.tensor(0.0, device=device)
        parent_correct = 0
        parent_total = 0
        gru_hidden = None  # GRU 隐状态，用于 Stage 4

        if self.lambda_parent > 0:
            if self.use_gru:
                # 论文对齐：获取 GRU 隐状态用于 Stage 4
                parent_logits, gru_hidden = self.stage3(
                    line_features, line_mask, return_gru_hidden=True
                )
                # gru_hidden: [B, L+1, gru_hidden_size]，包括 ROOT

                for b in range(batch_size):
                    sample_parent_ids = line_parent_ids[b]
                    sample_mask = line_mask[b]
                    num_lines = int(sample_mask.sum().item())

                    for child_idx in range(num_lines):
                        gt_parent = sample_parent_ids[child_idx].item()

                        if gt_parent == -100:
                            continue
                        if gt_parent >= child_idx:
                            continue

                        target_idx = gt_parent + 1 if gt_parent >= 0 else 0
                        child_logits = parent_logits[b, child_idx + 1, :child_idx + 2]

                        if torch.isinf(child_logits).all():
                            continue

                        child_logits = torch.where(
                            torch.isinf(child_logits),
                            torch.full_like(child_logits, -1e4),
                            child_logits
                        )

                        target = torch.tensor([target_idx], device=device)
                        loss = F.cross_entropy(child_logits.unsqueeze(0), target)

                        if not torch.isnan(loss):
                            parent_loss = parent_loss + loss
                            parent_total += 1

                        pred_parent = child_logits.argmax().item()
                        if pred_parent == target_idx:
                            parent_correct += 1
            else:
                for b in range(batch_size):
                    sample_features = line_features[b]
                    sample_mask = line_mask[b]
                    sample_parent_ids = line_parent_ids[b]

                    num_lines = sample_mask.sum().item()
                    if num_lines <= 1:
                        continue

                    for child_idx in range(1, int(num_lines)):
                        gt_parent = sample_parent_ids[child_idx].item()

                        if gt_parent < 0 or gt_parent >= child_idx:
                            continue

                        parent_candidates = sample_features[:child_idx]
                        child_feat = sample_features[child_idx]

                        scores = self.stage3(parent_candidates, child_feat)

                        target = torch.tensor([gt_parent], device=device)
                        loss = F.cross_entropy(scores.unsqueeze(0), target)
                        parent_loss = parent_loss + loss

                        pred_parent = scores.argmax().item()
                        if pred_parent == gt_parent:
                            parent_correct += 1
                        parent_total += 1

            if parent_total > 0:
                parent_loss = parent_loss / parent_total
                # 注意：不将 parent_acc 放入 outputs，因为 Trainer 会尝试 detach 它
                # 而 float 没有 detach 方法。accuracy 通过 _current_loss_dict 记录到日志
                self._parent_acc = parent_correct / parent_total

            outputs["parent_loss"] = parent_loss
            outputs["loss"] = outputs["loss"] + parent_loss * self.lambda_parent

        # ==================== Stage 4: Relation Classification ====================
        # 论文对齐：使用 GRU 隐状态 h_i, h_j 作为输入
        # P_rel_(i,j) = softmax(LinearProj(Concat(h_i, h_j)))
        rel_loss = torch.tensor(0.0, device=device)
        rel_correct = 0
        rel_total = 0

        if self.lambda_rel > 0 and line_relations is not None:
            # 检查是否有 GRU 隐状态（只有使用 GRU 时才有）
            if gru_hidden is None:
                # 如果没有 GRU 隐状态，则使用 line_features（兼容非 GRU 模式）
                gru_hidden = line_features  # [B, L, H]
                use_gru_offset = False
            else:
                # gru_hidden 包含 ROOT，形状是 [B, L+1, gru_hidden_size]
                # 位置 0 是 ROOT，位置 1~L 是原始行
                use_gru_offset = True

            for b in range(batch_size):
                sample_mask = line_mask[b]
                sample_parent_ids = line_parent_ids[b]
                sample_relations = line_relations[b]

                num_lines = int(sample_mask.sum().item())

                for child_idx in range(num_lines):
                    parent_idx = sample_parent_ids[child_idx].item()
                    rel_label = sample_relations[child_idx].item()

                    # 跳过无效样本：parent_id 无效 或 rel_label 是 padding (-100)
                    if parent_idx < 0 or parent_idx >= num_lines:
                        continue
                    if rel_label == -100:  # padding / none
                        continue

                    # 获取 GRU 隐状态
                    if use_gru_offset:
                        # gru_hidden 索引需要 +1（因为位置 0 是 ROOT）
                        # parent_idx=-1 (ROOT) 对应 gru_hidden 索引 0
                        # parent_idx=0 对应 gru_hidden 索引 1
                        parent_gru_idx = parent_idx + 1
                        child_gru_idx = child_idx + 1
                        parent_feat = gru_hidden[b, parent_gru_idx]
                        child_feat = gru_hidden[b, child_gru_idx]
                    else:
                        parent_feat = gru_hidden[b, parent_idx]
                        child_feat = gru_hidden[b, child_idx]

                    rel_logits = self.stage4(
                        parent_feat.unsqueeze(0),
                        child_feat.unsqueeze(0),
                    )

                    target = torch.tensor([rel_label], device=device)
                    loss = F.cross_entropy(rel_logits, target)
                    rel_loss = rel_loss + loss

                    pred_rel = rel_logits.argmax(dim=1).item()
                    if pred_rel == rel_label:
                        rel_correct += 1
                    rel_total += 1

            if rel_total > 0:
                rel_loss = rel_loss / rel_total
                # 注意：不将 rel_acc 放入 outputs，因为 Trainer 会尝试 detach 它
                # 而 float 没有 detach 方法。accuracy 通过 _current_loss_dict 记录到日志
                self._rel_acc = rel_correct / rel_total

            outputs["rel_loss"] = rel_loss
            outputs["loss"] = outputs["loss"] + rel_loss * self.lambda_rel

        # 返回 TokenClassifierOutput 格式，兼容 HuggingFace Trainer
        # 同时保留额外的 loss 信息在 outputs 字典中供 compute_loss 使用
        self._outputs_dict = outputs  # 保存完整的 outputs 供 compute_loss 使用
        return TokenClassifierOutput(
            loss=outputs["loss"],
            logits=outputs["logits"],
        )


# ==================== 自定义 Trainer ====================

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_args = model_args
        self.learning_rate_stage34 = learning_rate_stage34

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

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """保存模型，分别保存各 Stage"""

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Stage 1: 使用 HuggingFace 格式保存
        stage1_dir = os.path.join(output_dir, "stage1")
        self.model.stage1.save_pretrained(stage1_dir)

        # Stage 3/4: 保存为 PyTorch 格式
        torch.save(self.model.stage3.state_dict(), os.path.join(output_dir, "stage3.pt"))
        torch.save(self.model.stage4.state_dict(), os.path.join(output_dir, "stage4.pt"))

        # 保存 tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(stage1_dir)

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
    """记录联合训练的详细日志"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # 格式化日志输出
        step = state.global_step
        log_parts = [f"Step {step}"]

        if "loss" in logs:
            log_parts.append(f"loss={logs['loss']:.4f}")
        if "cls_loss" in logs:
            log_parts.append(f"cls={logs['cls_loss']:.4f}")
        if "parent_loss" in logs:
            log_parts.append(f"parent={logs['parent_loss']:.4f}")
        if "rel_loss" in logs:
            log_parts.append(f"rel={logs['rel_loss']:.4f}")
        if "parent_acc" in logs:
            log_parts.append(f"p_acc={logs['parent_acc']:.2%}")
        if "rel_acc" in logs:
            log_parts.append(f"r_acc={logs['rel_acc']:.2%}")

        if len(log_parts) > 1:
            logger.info(" | ".join(log_parts))


class E2EEvaluationCallback(TrainerCallback):
    """
    端到端评估 Callback

    在每次评估时运行完整的端到端评估：
    - Stage 1: 分类 (Line-level Macro/Micro F1)
    - Stage 3: Parent 准确率
    - Stage 4: Relation 准确率 + Macro F1
    - TEDS: Tree Edit Distance Similarity (可选)
    """

    def __init__(self, eval_dataloader, data_collator, compute_teds: bool = False):
        self.eval_dataloader = eval_dataloader
        self.data_collator = data_collator
        self.compute_teds = compute_teds

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

        # 运行端到端评估
        e2e_results = self._evaluate_e2e(model, device, global_step)

        # 打印结果（紧凑表格格式）
        if e2e_results:
            # 提取指标
            line_macro = e2e_results.get('line_macro_f1', 0) * 100
            line_micro = e2e_results.get('line_micro_f1', 0) * 100
            line_acc = e2e_results.get('line_accuracy', 0) * 100
            parent_acc = e2e_results.get('parent_accuracy', 0) * 100
            rel_acc = e2e_results.get('relation_accuracy', 0) * 100
            rel_macro = e2e_results.get('relation_macro_f1', 0) * 100
            teds = e2e_results.get('macro_teds', 0) * 100
            num_lines = e2e_results.get('num_lines', 0)

            # 紧凑格式打印
            logger.info("┌─────────────────────────────────────────────────────────┐")
            logger.info(f"│  Stage1(Line) │ MacroF1: {line_macro:5.2f}% │ Acc: {line_acc:5.2f}%          │")
            logger.info(f"│  Stage3(Par)  │ Accuracy: {parent_acc:5.2f}%                           │")
            logger.info(f"│  Stage4(Rel)  │ MacroF1: {rel_macro:5.2f}% │ Acc: {rel_acc:5.2f}%          │")
            if teds > 0:
                logger.info(f"│  TEDS         │ {teds:5.2f}%                                   │")
            logger.info(f"│  Lines: {num_lines:<6}                                          │")
            logger.info("└─────────────────────────────────────────────────────────┘")

            # 一行摘要（方便快速对比）
            summary = f"[Step {global_step}] Line={line_macro:.1f}% | Parent={parent_acc:.1f}% | Rel={rel_macro:.1f}%"
            if teds > 0:
                summary += f" | TEDS={teds:.1f}%"
            logger.info(summary)

    def _evaluate_e2e(self, model, device, global_step: int) -> Dict[str, float]:
        """运行端到端评估"""
        from util.hrdoc_eval import (
            ID2CLASS, CLASS2ID, ID2RELATION, RELATION2ID,
            extract_gt_from_batch, compute_teds_score
        )
        from sklearn.metrics import f1_score, accuracy_score

        model.eval()

        # Stage 1 分类
        all_gt_classes = []
        all_pred_classes = []

        # Stage 3 Parent
        all_gt_parents = []
        all_pred_parents = []

        # Stage 4 Relation
        all_gt_relations = []
        all_pred_relations = []

        # TEDS
        all_gt_docs = []
        all_pred_docs = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="E2E Eval"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Stage 1
                stage1_outputs = model.stage1(
                    input_ids=batch["input_ids"],
                    bbox=batch["bbox"],
                    attention_mask=batch["attention_mask"],
                    image=batch.get("image"),
                    output_hidden_states=True,
                )
                logits = stage1_outputs.logits
                hidden_states = stage1_outputs.hidden_states[-1]
                batch_size = batch["input_ids"].shape[0]

                # 提取 GT
                gt_docs = extract_gt_from_batch(batch, ID2CLASS)

                for b in range(batch_size):
                    gt_doc = gt_docs[b]
                    if not gt_doc:
                        continue

                    line_ids = batch.get("line_ids")
                    if line_ids is None:
                        continue

                    line_ids_b = line_ids[b]
                    sample_logits = logits[b]

                    # 提取每行的预测类别（使用多数投票）
                    token_preds = [sample_logits[i].argmax().item() for i in range(sample_logits.shape[0])]
                    line_preds = aggregate_token_to_line_predictions(
                        token_preds, line_ids_b.cpu().tolist(), method="majority"
                    )

                    # Stage 2: 提取特征
                    text_seq_len = batch["input_ids"].shape[1]
                    text_hidden = hidden_states[b:b+1, :text_seq_len, :]

                    line_features, line_mask = model.feature_extractor.extract_line_features(
                        text_hidden, line_ids_b.unsqueeze(0), pooling="mean"
                    )

                    line_features = line_features[0]
                    line_mask = line_mask[0]
                    actual_num_lines = int(line_mask.sum().item())

                    if actual_num_lines == 0:
                        continue

                    # Stage 3: 预测父节点
                    pred_parents = [-1] * actual_num_lines
                    gru_hidden = None  # 用于 Stage 4

                    if hasattr(model, 'use_gru') and model.use_gru:
                        # 论文对齐：获取 GRU 隐状态用于 Stage 4
                        parent_logits, gru_hidden = model.stage3(
                            line_features.unsqueeze(0),
                            line_mask.unsqueeze(0),
                            return_gru_hidden=True
                        )
                        # gru_hidden: [1, L+1, gru_hidden_size]，包括 ROOT
                        gru_hidden = gru_hidden[0]  # [L+1, gru_hidden_size]
                        for child_idx in range(actual_num_lines):
                            child_logits = parent_logits[0, child_idx + 1, :child_idx + 2]
                            pred_parent_idx = child_logits.argmax().item()
                            pred_parents[child_idx] = pred_parent_idx - 1
                    else:
                        for child_idx in range(1, actual_num_lines):
                            parent_candidates = line_features[:child_idx]
                            child_feat = line_features[child_idx]
                            scores = model.stage3(parent_candidates, child_feat)
                            pred_parents[child_idx] = scores.argmax().item()

                    # Stage 4: 预测关系（论文对齐：使用 GRU 隐状态，不使用几何特征）
                    pred_relations = [0] * actual_num_lines

                    for child_idx in range(actual_num_lines):
                        parent_idx = pred_parents[child_idx]
                        if parent_idx < 0 or parent_idx >= actual_num_lines:
                            continue

                        if gru_hidden is not None:
                            # 论文对齐：使用 GRU 隐状态
                            # gru_hidden 包含 ROOT，所以需要 +1 偏移
                            parent_gru_idx = parent_idx + 1
                            child_gru_idx = child_idx + 1
                            parent_feat = gru_hidden[parent_gru_idx]
                            child_feat = gru_hidden[child_gru_idx]
                        else:
                            # 非 GRU 模式：使用 encoder 输出
                            parent_feat = line_features[parent_idx]
                            child_feat = line_features[child_idx]

                        rel_logits = model.stage4(
                            parent_feat.unsqueeze(0),
                            child_feat.unsqueeze(0),
                        )
                        pred_relations[child_idx] = rel_logits.argmax(dim=1).item()

                    # 收集结果（使用 gt_doc 中保存的 line_id 来对齐预测）
                    for idx, gt_item in enumerate(gt_doc):
                        if idx >= actual_num_lines:
                            break
                        line_id = gt_item["line_id"]  # 使用实际的 line_id

                        # Stage 1: 分类
                        gt_class = CLASS2ID.get(gt_item["class"], 0)
                        pred_class = line_preds.get(line_id, 0)  # 用 line_id 而不是 idx
                        all_gt_classes.append(gt_class)
                        all_pred_classes.append(pred_class)

                        # Stage 3: Parent
                        all_gt_parents.append(gt_item["parent_id"])
                        all_pred_parents.append(pred_parents[idx] if idx < len(pred_parents) else -1)

                        # Stage 4: Relation
                        gt_rel = RELATION2ID.get(gt_item.get("relation", "none"), 0)
                        all_gt_relations.append(gt_rel)
                        all_pred_relations.append(pred_relations[idx] if idx < len(pred_relations) else 0)

                    # TEDS: 保存文档结构（使用 line_id 对齐）
                    if self.compute_teds:
                        pred_doc = []
                        for idx, gt_item in enumerate(gt_doc):
                            if idx >= actual_num_lines:
                                break
                            line_id = gt_item["line_id"]
                            class_id = line_preds.get(line_id, 0)  # 用 line_id
                            class_name = ID2CLASS.get(class_id, f"class_{class_id}")
                            pred_doc.append({
                                "class": class_name,
                                "text": f"line_{line_id}",
                                "parent_id": pred_parents[idx],
                                "relation": ID2RELATION.get(pred_relations[idx], "none"),
                            })
                        all_gt_docs.append(gt_doc)
                        all_pred_docs.append(pred_doc)

        # 计算指标
        results = {}

        if all_gt_classes:
            # Stage 1: 分类指标
            results["line_macro_f1"] = f1_score(all_gt_classes, all_pred_classes, average="macro", zero_division=0)
            results["line_micro_f1"] = f1_score(all_gt_classes, all_pred_classes, average="micro", zero_division=0)
            results["line_accuracy"] = accuracy_score(all_gt_classes, all_pred_classes)

            # Stage 3: Parent 准确率
            parent_correct = sum(1 for g, p in zip(all_gt_parents, all_pred_parents) if g == p)
            results["parent_accuracy"] = parent_correct / len(all_gt_parents)

            # Stage 4: Relation 指标（只计算有父节点的）
            rel_pairs = [(g, p) for g, p, gp in zip(all_gt_relations, all_pred_relations, all_gt_parents) if gp >= 0]
            if rel_pairs:
                gt_rels = [g for g, p in rel_pairs]
                pred_rels = [p for g, p in rel_pairs]
                rel_correct = sum(1 for g, p in rel_pairs if g == p)
                results["relation_accuracy"] = rel_correct / len(rel_pairs)
                results["relation_macro_f1"] = f1_score(gt_rels, pred_rels, average="macro", zero_division=0)

            results["num_lines"] = len(all_gt_classes)

        # TEDS
        if self.compute_teds and all_gt_docs:
            try:
                teds_score = compute_teds_score(all_gt_docs, all_pred_docs)
                if teds_score is not None:
                    results["macro_teds"] = teds_score
            except Exception as e:
                logger.warning(f"TEDS computation failed: {e}")

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


def build_child_parent_matrix_from_dataset(dataset, num_classes=NUM_LABELS):
    """从 HuggingFace Dataset 构建 Child-Parent Distribution Matrix (M_cp)"""
    logger.info("从数据集构建 Child-Parent Distribution Matrix...")

    cp_matrix = ChildParentDistributionMatrix(num_classes=num_classes)

    for example in tqdm(dataset, desc="统计父子关系"):
        ner_tags = example.get("ner_tags", [])
        line_ids = example.get("line_ids", [])
        line_parent_ids = example.get("line_parent_ids", [])

        if not line_parent_ids or not ner_tags:
            continue

        line_labels = {}
        for tag, line_id in zip(ner_tags, line_ids):
            if line_id >= 0 and line_id not in line_labels and tag >= 0:
                line_labels[line_id] = tag

        for child_idx, parent_idx in enumerate(line_parent_ids):
            if child_idx not in line_labels:
                continue

            child_label = line_labels[child_idx]
            parent_label = line_labels.get(parent_idx, -1) if parent_idx >= 0 else -1

            cp_matrix.update(child_label, parent_label)

    cp_matrix.build()
    return cp_matrix


def prepare_datasets(tokenizer, data_args: JointDataArguments, training_args: JointTrainingArguments):
    """准备训练和评估数据集"""

    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            padding="max_length",
            truncation=True,
            max_length=512,
            is_split_into_words=True,
            return_overflowing_tokens=True,
        )

        all_labels = []
        all_bboxes = []
        all_images = []
        all_line_ids = []
        all_line_parent_ids = []
        all_line_relations = []
        all_line_bboxes = []

        label2id = get_label2id()

        for batch_idx in range(len(tokenized["input_ids"])):
            word_ids = tokenized.word_ids(batch_index=batch_idx)
            org_idx = tokenized["overflow_to_sample_mapping"][batch_idx]

            label = examples["ner_tags"][org_idx]
            bbox = examples["bboxes"][org_idx]
            image = examples["image"][org_idx]
            line_ids_raw = examples.get("line_ids", [[]])[org_idx]
            line_parent_ids_raw = examples.get("line_parent_ids", [[]])[org_idx]
            line_relations_raw = examples.get("line_relations", [[]])[org_idx]

            token_labels = []
            token_bboxes = []
            token_line_ids = []
            prev_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    token_labels.append(-100)
                    token_bboxes.append([0, 0, 0, 0])
                    token_line_ids.append(-1)
                elif word_idx != prev_word_idx:
                    lbl = label[word_idx]
                    token_labels.append(lbl if isinstance(lbl, int) else label2id.get(lbl, 0))
                    token_bboxes.append(bbox[word_idx])
                    if word_idx < len(line_ids_raw):
                        token_line_ids.append(line_ids_raw[word_idx])
                    else:
                        token_line_ids.append(-1)
                else:
                    token_labels.append(-100)
                    token_bboxes.append(bbox[word_idx])
                    if word_idx < len(line_ids_raw):
                        token_line_ids.append(line_ids_raw[word_idx])
                    else:
                        token_line_ids.append(-1)
                prev_word_idx = word_idx

            all_labels.append(token_labels)
            all_bboxes.append(token_bboxes)
            all_images.append(image)
            all_line_ids.append(token_line_ids)
            all_line_parent_ids.append(line_parent_ids_raw)
            all_line_relations.append(line_relations_raw)

            line_bboxes = compute_line_bboxes(token_bboxes, token_line_ids)
            all_line_bboxes.append(line_bboxes)

        tokenized["labels"] = all_labels
        tokenized["bbox"] = all_bboxes
        tokenized["image"] = all_images
        tokenized["line_ids"] = all_line_ids
        tokenized["line_parent_ids"] = all_line_parent_ids
        tokenized["line_relations"] = all_line_relations
        tokenized["line_bboxes"] = all_line_bboxes

        return tokenized

    remove_columns = datasets["train"].column_names

    # 训练数据
    train_data = datasets["train"]
    if data_args.max_train_samples > 0:
        train_data = train_data.select(range(min(len(train_data), data_args.max_train_samples)))
        logger.info(f"Limited train samples to {len(train_data)}")

    num_proc = 1 if (data_args.max_train_samples > 0 and data_args.max_train_samples < 100) else 4
    train_dataset = train_data.map(
        tokenize_and_align,
        batched=True,
        remove_columns=remove_columns,
        num_proc=num_proc,
    )

    # 评估数据
    eval_dataset = None
    if "validation" in datasets:
        eval_data = datasets["validation"]
    elif "test" in datasets:
        eval_data = datasets["test"]
    else:
        eval_data = None

    if eval_data is not None:
        if data_args.max_eval_samples > 0:
            eval_data = eval_data.select(range(min(len(eval_data), data_args.max_eval_samples)))
        eval_dataset = eval_data.map(
            tokenize_and_align,
            batched=True,
            remove_columns=remove_columns,
            num_proc=4,
        )

    return train_dataset, eval_dataset, datasets["train"]


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

    # 初始化实验管理器
    exp_manager, exp_dir = ensure_experiment(
        config,
        exp=training_args.exp,
        new_exp=training_args.new_exp,
        name=training_args.exp_name or f"Joint {data_args.dataset.upper()}",
    )

    # 设置模型路径
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

    # Covmatch 目录
    covmatch_dir = config.dataset.get_covmatch_dir(data_args.dataset)
    if os.path.exists(covmatch_dir):
        os.environ["HRDOC_SPLIT_DIR"] = covmatch_dir

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
    logger.info(f"  batch_size:    {training_args.per_device_train_batch_size}")
    logger.info(f"  grad_accum:    {training_args.gradient_accumulation_steps}")
    logger.info(f"  learning_rate: {training_args.learning_rate} (Stage 1)")
    logger.info(f"  lr_stage34:    {training_args.learning_rate_stage34} (Stage 3/4)")
    logger.info(f"  fp16:          {training_args.fp16}")
    logger.info("=" * 60)

    if training_args.dry_run:
        logger.info("[Dry run mode - exiting]")
        return

    os.makedirs(training_args.output_dir, exist_ok=True)

    # 标记 stage 开始
    exp_manager.mark_stage_started(training_args.exp, "joint", data_args.dataset)

    # 加载 tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(model_args.model_name_or_path)

    # 准备数据集
    logger.info("Preparing datasets...")
    train_dataset, eval_dataset, raw_train_dataset = prepare_datasets(tokenizer, data_args, training_args)
    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset) if eval_dataset else 0}")

    # Data collator
    data_collator = HRDocJointDataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=512,
    )

    # 加载模型
    logger.info("Loading models...")

    # Stage 1: LayoutXLM
    stage1_config = LayoutXLMConfig.from_pretrained(model_args.model_name_or_path)
    stage1_config.num_labels = NUM_LABELS
    stage1_config.id2label = get_id2label()
    stage1_config.label2id = get_label2id()

    stage1_model = LayoutXLMForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=stage1_config,
    )

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
            logger.info("Building Child-Parent Distribution Matrix (M_cp)...")
            cp_matrix = build_child_parent_matrix_from_dataset(raw_train_dataset, num_classes=NUM_LABELS)
            stage3_model.set_child_parent_matrix(cp_matrix.get_tensor(device))
            logger.info("M_cp initialized successfully")
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
    )

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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_line_level_metrics,
        callbacks=callbacks,
    )

    # 训练
    logger.info("Starting training...")
    train_result = trainer.train()

    # 保存最终模型
    trainer.save_model()
    trainer.save_state()

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 评估
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
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

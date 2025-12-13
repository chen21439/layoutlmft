#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 联合训练脚本

端到端训练流程 (复用各 Stage 的现有实现):
1. Stage 1: LayoutXLM 分类 (产生分类 loss + hidden states)
2. Stage 2: 从 hidden states 提取 line-level 特征 (torch.no_grad，仅前向)
3. Stage 3: ParentFinder 训练 (产生 parent loss) - 复用 train_parent_finder.py
4. Stage 4: RelationClassifier 训练 (产生 relation loss) - 复用 train_multiclass_relation.py

总 Loss = λ1 * L_cls + λ2 * L_par + λ3 * L_rel (论文公式)

注意：
- Stage 2 的特征提取在 torch.no_grad() 下进行
- Stage 1 的梯度通过 L_cls 回传
- Stage 3/4 的梯度独立回传（特征是 detached 的输入）
- 使用 ParentFinderGRU + Soft-Mask（论文完整方法）或 SimpleParentFinder（简化方法）

Usage:
    python examples/stage/train_joint.py --env test --dataset hrds

    # 快速测试
    python examples/stage/train_joint.py --env test --dataset hrds --quick

    # 使用完整论文方法（GRU + Soft-Mask）
    python examples/stage/train_joint.py --env test --dataset hrds --use_gru --use_soft_mask

    # 只训练 Stage 1（禁用 Stage 3/4）
    python examples/stage/train_joint.py --env test --dataset hrds --disable_stage34

    # 自定义 loss 权重
    python examples/stage/train_joint.py --env test --lambda_cls 1.0 --lambda_parent 0.5 --lambda_rel 0.5
"""

import logging
import os
import sys
import json
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint

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
    RELATION_LABELS,
    RELATION_NAMES,
    FocalLoss,
)

from joint_data_collator import HRDocJointDataCollator
from train_parent_finder import (
    SimpleParentFinder,
    ParentFinderGRU,
    ChildParentDistributionMatrix,
    build_child_parent_matrix,
)
from util.eval_utils import compute_macro_f1, log_per_class_metrics
from util.checkpoint_utils import get_latest_checkpoint, get_best_model
from util.experiment_manager import ensure_experiment

# HRDoc 评估工具
HRDOC_UTILS_PATH = os.path.join(PROJECT_ROOT, "HRDoc", "utils")
sys.path.insert(0, HRDOC_UTILS_PATH)

logger = logging.getLogger(__name__)


@dataclass
class JointTrainingArguments:
    """联合训练参数"""

    # 环境和数据集
    env: str = field(default="test", metadata={"help": "Environment: dev, test"})
    dataset: str = field(default="hrds", metadata={"help": "Dataset: hrds, hrdh"})
    quick: bool = field(default=False, metadata={"help": "Quick test mode (few samples, small batch)"})
    max_train_samples: int = field(default=-1, metadata={"help": "Max train samples (-1 for all)"})

    # 实验管理
    exp: str = field(default=None, metadata={"help": "Experiment ID (default: current or latest)"})
    new_exp: bool = field(default=False, metadata={"help": "Create a new experiment"})
    exp_name: str = field(default="", metadata={"help": "Name for new experiment"})

    # 模型路径
    model_name_or_path: str = field(
        default=None, metadata={"help": "Stage 1 model path (if None, auto-detect from experiment)"}
    )
    output_dir: str = field(default=None, metadata={"help": "Output directory"})

    # Loss 权重 (论文公式: L = L_cls + α₁·L_par + α₂·L_rel)
    lambda_cls: float = field(default=1.0, metadata={"help": "Classification loss weight (λ1)"})
    lambda_parent: float = field(default=1.0, metadata={"help": "Parent loss weight (α₁)"})
    lambda_rel: float = field(default=1.0, metadata={"help": "Relation loss weight (α₂)"})

    # 训练参数
    max_steps: int = field(default=5000, metadata={"help": "Max training steps"})
    per_device_train_batch_size: int = field(default=2, metadata={"help": "Batch size"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation"})
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate for Stage 1"})
    learning_rate_stage34: float = field(default=5e-4, metadata={"help": "Learning rate for Stage 3/4"})
    warmup_steps: int = field(default=500, metadata={"help": "Warmup steps"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm"})
    fp16: bool = field(default=True, metadata={"help": "Use FP16"})
    seed: int = field(default=42, metadata={"help": "Random seed"})

    # 评估和保存
    eval_steps: int = field(default=500, metadata={"help": "Evaluation interval"})
    save_steps: int = field(default=500, metadata={"help": "Save interval"})
    save_total_limit: int = field(default=3, metadata={"help": "Max checkpoints to keep"})
    logging_steps: int = field(default=50, metadata={"help": "Logging interval"})

    # Stage 3 配置（论文方法，默认开启）
    use_gru: bool = field(default=True, metadata={"help": "Use GRU decoder (paper method, default=True)"})
    use_soft_mask: bool = field(default=True, metadata={"help": "Use Soft-Mask (paper method, default=True)"})

    # 功能开关
    disable_stage34: bool = field(default=False, metadata={"help": "Disable Stage 3/4 training"})
    use_focal_loss: bool = field(default=True, metadata={"help": "Use Focal Loss for L_cls and L_rel (paper method)"})
    use_hrdoc_eval: bool = field(default=True, metadata={"help": "Use HRDoc evaluation tools"})

    # 其他
    resume_from_checkpoint: str = field(default=None, metadata={"help": "Resume from checkpoint"})
    dry_run: bool = field(default=False, metadata={"help": "Dry run mode"})


class JointModel(nn.Module):
    """
    联合模型：包含 Stage 1/2/3/4 的所有模块

    Stage 1: LayoutXLM (分类) - 产生 L_cls
    Stage 2: LineFeatureExtractor (特征提取，无参数，torch.no_grad)
    Stage 3: ParentFinder (SimpleParentFinder 或 ParentFinderGRU) - 产生 L_par
    Stage 4: MultiClassRelationClassifier - 产生 L_rel

    论文公式: L_total = L_cls + α₁·L_par + α₂·L_rel
    """

    def __init__(
        self,
        stage1_model: LayoutXLMForTokenClassification,
        stage3_model: nn.Module,  # SimpleParentFinder 或 ParentFinderGRU
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

        # 关系分类损失 (论文使用 Focal Loss)
        if use_focal_loss:
            self.relation_criterion = FocalLoss(gamma=2.0)
        else:
            self.relation_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
        labels: torch.Tensor,
        line_ids: Optional[torch.Tensor] = None,
        line_parent_ids: Optional[torch.Tensor] = None,
        line_relations: Optional[torch.Tensor] = None,
        line_bboxes: Optional[torch.Tensor] = None,
        return_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: [B, seq_len]
            bbox: [B, seq_len, 4]
            attention_mask: [B, seq_len]
            image: [B, 3, H, W]
            labels: [B, seq_len] - 分类标签
            line_ids: [B, seq_len] - token 到 line 的映射
            line_parent_ids: [B, max_lines] - 每行的父节点 ID
            line_relations: [B, max_lines] - 每行的关系类型
            line_bboxes: [B, max_lines, 4] - 每行的 bbox
            return_outputs: 是否返回详细输出

        Returns:
            dict: 包含 loss 和各阶段输出
        """
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
        hidden_states = stage1_outputs.hidden_states[-1]  # 最后一层

        outputs = {
            "loss": cls_loss * self.lambda_cls,
            "cls_loss": cls_loss,
            "logits": logits,
        }

        # 如果没有 line 信息或禁用 Stage 3/4，直接返回
        if line_ids is None or line_parent_ids is None:
            return outputs

        # ==================== Stage 2: Feature Extraction ====================
        # 端到端训练：保持梯度流，让 Stage 3/4 的 loss 可以回传到 Stage 1
        text_seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :text_seq_len, :]

        line_features, line_mask = self.feature_extractor.extract_line_features(
            text_hidden, line_ids, pooling="mean"
        )  # [B, max_lines, hidden_size], [B, max_lines]

        # ==================== Stage 3: Parent Finding ====================
        parent_loss = torch.tensor(0.0, device=device)
        parent_correct = 0
        parent_total = 0

        if self.lambda_parent > 0:
            # 调试：检查输入特征
            if torch.isnan(line_features).any() or torch.isinf(line_features).any():
                logger.warning(
                    f"[DEBUG] line_features issue: "
                    f"nan={torch.isnan(line_features).sum().item()}, "
                    f"inf={torch.isinf(line_features).sum().item()}"
                )

            if self.use_gru:
                # 使用 ParentFinderGRU（论文完整方法）
                # 输入: line_features [B, max_lines, H], line_mask [B, max_lines]
                # 输出: parent_logits [B, L+1, L+1]
                parent_logits = self.stage3(line_features, line_mask)

                # 检查 NaN（只在有 NaN 时才报警，-inf 是正常的因果 mask）
                num_nan = torch.isnan(parent_logits).sum().item()
                if num_nan > 0:
                    logger.warning(f"[DEBUG] parent_logits has NaN: {num_nan}")
                    # 跳过这个 batch，避免 NaN 传播
                    pass
                else:
                    # 计算损失
                    for b in range(batch_size):
                        sample_parent_ids = line_parent_ids[b]  # [max_lines]
                        sample_mask = line_mask[b]
                        num_lines = int(sample_mask.sum().item())

                        for child_idx in range(num_lines):
                            gt_parent = sample_parent_ids[child_idx].item()

                            # 跳过无效标签
                            # gt_parent = -1 表示 ROOT（有效）
                            # gt_parent >= 0 且 < child_idx 表示有效的父节点
                            # gt_parent = -100 表示需要忽略
                            if gt_parent == -100:
                                continue
                            if gt_parent >= child_idx:  # 父节点不能是自己或之后的节点
                                continue

                            # gt_parent=-1 表示 ROOT，对应 logits 中的 index 0
                            # gt_parent>=0 表示实际行，对应 logits 中的 index = gt_parent + 1
                            target_idx = gt_parent + 1 if gt_parent >= 0 else 0

                            # child_idx 在 logits 中的位置是 child_idx + 1（因为 index 0 是 ROOT）
                            child_logits = parent_logits[b, child_idx + 1, :child_idx + 2]  # [child_idx + 2]

                            # 检查 logits 是否全是 -inf（无效候选）
                            if torch.isinf(child_logits).all():
                                continue

                            # 将 -inf 替换为一个很小的值，避免 softmax 问题
                            child_logits = torch.where(
                                torch.isinf(child_logits),
                                torch.full_like(child_logits, -1e4),
                                child_logits
                            )

                            target = torch.tensor([target_idx], device=device)
                            loss = F.cross_entropy(child_logits.unsqueeze(0), target)

                            # 检查 loss 是否为 NaN
                            if not torch.isnan(loss):
                                parent_loss = parent_loss + loss
                                parent_total += 1

                            # 准确率统计
                            pred_parent = child_logits.argmax().item()
                            if pred_parent == target_idx:
                                parent_correct += 1
            else:
                # 使用 SimpleParentFinder（简化方法）
                for b in range(batch_size):
                    sample_features = line_features[b]  # [max_lines, H]
                    sample_mask = line_mask[b]  # [max_lines]
                    sample_parent_ids = line_parent_ids[b]  # [max_lines]

                    num_lines = sample_mask.sum().item()
                    if num_lines <= 1:
                        continue

                    # 逐行预测父节点
                    for child_idx in range(1, num_lines):
                        gt_parent = sample_parent_ids[child_idx].item()

                        # 跳过无效标签
                        if gt_parent < 0 or gt_parent >= child_idx:
                            continue

                        # 候选父节点：所有前面的行
                        parent_candidates = sample_features[:child_idx]  # [child_idx, H]
                        child_feat = sample_features[child_idx]  # [H]

                        # 计算 scores
                        scores = self.stage3(parent_candidates, child_feat)  # [child_idx]

                        # 损失
                        target = torch.tensor([gt_parent], device=device)
                        loss = F.cross_entropy(scores.unsqueeze(0), target)
                        parent_loss = parent_loss + loss

                        # 准确率统计
                        pred_parent = scores.argmax().item()
                        if pred_parent == gt_parent:
                            parent_correct += 1
                        parent_total += 1

            if parent_total > 0:
                parent_loss = parent_loss / parent_total
                outputs["parent_acc"] = parent_correct / parent_total

            outputs["parent_loss"] = parent_loss
            outputs["loss"] = outputs["loss"] + parent_loss * self.lambda_parent

        # ==================== Stage 4: Relation Classification ====================
        rel_loss = torch.tensor(0.0, device=device)
        rel_correct = 0
        rel_total = 0

        if self.lambda_rel > 0 and line_relations is not None and line_bboxes is not None:
            for b in range(batch_size):
                sample_features = line_features[b]  # [max_lines, H]
                sample_mask = line_mask[b]  # [max_lines]
                sample_parent_ids = line_parent_ids[b]  # [max_lines]
                sample_relations = line_relations[b]  # [max_lines]
                sample_bboxes = line_bboxes[b]  # [max_lines, 4]

                num_lines = sample_mask.sum().item()

                for child_idx in range(num_lines):
                    parent_idx = sample_parent_ids[child_idx].item()
                    rel_label = sample_relations[child_idx].item()

                    # 跳过无效样本
                    if parent_idx < 0 or parent_idx >= num_lines:
                        continue
                    if rel_label < 0 or rel_label >= 4:  # 0=none, 1=connect, 2=contain, 3=equality
                        continue

                    # 获取特征
                    parent_feat = sample_features[parent_idx]  # [H]
                    child_feat = sample_features[child_idx]  # [H]

                    # 几何特征
                    parent_bbox = sample_bboxes[parent_idx]  # [4]
                    child_bbox = sample_bboxes[child_idx]  # [4]
                    geom_feat = compute_geometry_features(parent_bbox, child_bbox)  # [8]

                    # 预测
                    rel_logits = self.stage4(
                        parent_feat.unsqueeze(0),
                        child_feat.unsqueeze(0),
                        geom_feat.unsqueeze(0).to(device),
                    )  # [1, 4]

                    # 损失
                    target = torch.tensor([rel_label], device=device)
                    loss = F.cross_entropy(rel_logits, target)
                    rel_loss = rel_loss + loss

                    # 准确率统计
                    pred_rel = rel_logits.argmax(dim=1).item()
                    if pred_rel == rel_label:
                        rel_correct += 1
                    rel_total += 1

            if rel_total > 0:
                rel_loss = rel_loss / rel_total
                outputs["rel_acc"] = rel_correct / rel_total

            outputs["rel_loss"] = rel_loss
            outputs["loss"] = outputs["loss"] + rel_loss * self.lambda_rel

        return outputs


def load_config_and_setup(args: JointTrainingArguments):
    """加载配置并设置环境，返回 (config, data_dir, exp_manager)"""

    # 加载 YAML 配置
    from configs.config_loader import load_config

    config = load_config(args.env)
    if args.quick:
        config.quick_test.enabled = True
    config = config.get_effective_config()

    # 初始化实验管理器
    exp_manager, exp_dir = ensure_experiment(
        config,
        exp=args.exp,
        new_exp=args.new_exp,
        name=args.exp_name or f"Joint {args.dataset.upper()}",
    )

    # 设置 Stage 1 模型路径（优先使用实验中训练好的模型）
    if args.model_name_or_path is None:
        # 1. 先从当前实验查找 Stage 1 模型
        stage1_dir = exp_manager.get_stage_dir(args.exp, "stage1", args.dataset)
        stage1_model = get_latest_checkpoint(stage1_dir)

        if stage1_model:
            args.model_name_or_path = stage1_model
            logger.info(f"Using Stage 1 model from experiment: {stage1_model}")
        else:
            # 2. 回退到 legacy 路径
            if hasattr(config.paths, 'stage1_model_path') and config.paths.stage1_model_path:
                legacy_dir = f"{config.paths.stage1_model_path}_{args.dataset}"
                stage1_model = get_latest_checkpoint(legacy_dir)
                if stage1_model:
                    args.model_name_or_path = stage1_model
                    logger.info(f"Using Stage 1 model from legacy path: {stage1_model}")

        # 3. 最后回退到预训练模型
        if args.model_name_or_path is None:
            args.model_name_or_path = config.model.local_path or config.model.name_or_path
            logger.warning(f"No Stage 1 model found, using pretrained: {args.model_name_or_path}")

    # 设置输出目录（实验管理）
    if args.output_dir is None:
        args.output_dir = exp_manager.get_stage_dir(args.exp, "joint", args.dataset)

    # 数据目录
    data_dir = config.dataset.get_data_dir(args.dataset)
    os.environ["HRDOC_DATA_DIR"] = data_dir

    # Covmatch 目录
    covmatch_dir = config.dataset.get_covmatch_dir(args.dataset)
    if os.path.exists(covmatch_dir):
        os.environ["HRDOC_SPLIT_DIR"] = covmatch_dir

    # GPU 设置
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    # HF cache
    if config.paths.hf_cache_dir:
        os.environ["HF_HOME"] = config.paths.hf_cache_dir

    return config, data_dir, exp_manager


def create_dataloaders(tokenizer, data_dir: str, args: JointTrainingArguments):
    """创建数据加载器"""

    # 加载数据集
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))

    # 预处理函数
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            padding="max_length",
            truncation=True,
            max_length=512,
            is_split_into_words=True,
            return_overflowing_tokens=True,
        )

        # 对齐标签
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

            # 计算 line-level bboxes
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

    # 处理数据集
    remove_columns = datasets["train"].column_names

    # 限制训练样本数（用于快速测试）
    train_data = datasets["train"]
    if args.max_train_samples > 0 and len(train_data) > args.max_train_samples:
        train_data = train_data.select(range(args.max_train_samples))
        logger.info(f"Limited train samples to {args.max_train_samples}")

    train_dataset = train_data.map(
        tokenize_and_align,
        batched=True,
        remove_columns=remove_columns,
        num_proc=1 if args.max_train_samples > 0 and args.max_train_samples < 100 else 4,  # 少量数据用单进程
    )

    eval_dataset = None
    if "validation" in datasets:
        eval_dataset = datasets["validation"].map(
            tokenize_and_align,
            batched=True,
            remove_columns=remove_columns,
            num_proc=4,
        )
    elif "test" in datasets:
        eval_dataset = datasets["test"].map(
            tokenize_and_align,
            batched=True,
            remove_columns=remove_columns,
            num_proc=4,
        )

    # Data collator
    data_collator = HRDocJointDataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=512,
    )

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
    )

    eval_loader = None
    if eval_dataset:
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0,
        )

    # 返回原始数据集用于构建 M_cp
    return train_loader, eval_loader, datasets["train"]


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

    # 转换为列表
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


# ==================== HRDoc 评估相关函数 ====================

# ID 到类别名的映射（与 HRDoc 一致）
ID2CLASS = {
    0: "title", 1: "author", 2: "mail", 3: "affili", 4: "section",
    5: "fstline", 6: "paraline", 7: "table", 8: "figure", 9: "caption",
    10: "equation", 11: "footer", 12: "header", 13: "footnote"
}

# 关系 ID 到名称的映射
ID2RELATION = {0: "none", 1: "connect", 2: "contain", 3: "equality"}


def build_child_parent_matrix_from_dataset(dataset, num_classes=NUM_LABELS):
    """
    从 HuggingFace Dataset 构建 Child-Parent Distribution Matrix (M_cp)

    与 train_parent_finder.py 中的 build_child_parent_matrix 功能相同，
    但直接从 dataset 读取而非 features_dir

    Args:
        dataset: HuggingFace Dataset，包含 ner_tags, line_ids, line_parent_ids
        num_classes: 语义类别数

    Returns:
        cp_matrix: ChildParentDistributionMatrix 对象
    """
    logger.info("从数据集构建 Child-Parent Distribution Matrix...")

    cp_matrix = ChildParentDistributionMatrix(num_classes=num_classes)

    for example in tqdm(dataset, desc="统计父子关系"):
        ner_tags = example.get("ner_tags", [])
        line_ids = example.get("line_ids", [])
        line_parent_ids = example.get("line_parent_ids", [])

        if not line_parent_ids or not ner_tags:
            continue

        # 计算每行的语义标签（取该行第一个 token 的标签）
        line_labels = {}
        for tag, line_id in zip(ner_tags, line_ids):
            if line_id >= 0 and line_id not in line_labels and tag >= 0:
                line_labels[line_id] = tag

        # 统计父子关系
        for child_idx, parent_idx in enumerate(line_parent_ids):
            if child_idx not in line_labels:
                continue

            child_label = line_labels[child_idx]
            # parent_idx = -1 表示 ROOT
            parent_label = line_labels.get(parent_idx, -1) if parent_idx >= 0 else -1

            cp_matrix.update(child_label, parent_label)

    # 构建分布矩阵
    cp_matrix.build()

    return cp_matrix


def inference_to_json(
    model,
    dataloader,
    device,
    output_dir: str,
    id2label: Dict[int, str] = None,
) -> List[Dict]:
    """
    运行推理并将结果保存为 HRDoc 格式的 JSON 文件

    每个文档保存为一个 JSON 文件，格式：
    [
        {"class": "title", "text": "...", "parent_id": -1, "relation": "none"},
        {"class": "section", "text": "...", "parent_id": 0, "relation": "contain"},
        ...
    ]

    Args:
        model: JointModel
        dataloader: 评估数据加载器
        device: 设备
        output_dir: 输出目录
        id2label: 类别 ID 到名称的映射

    Returns:
        all_results: 所有文档的推理结果列表
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    if id2label is None:
        id2label = ID2CLASS

    all_results = []
    doc_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Stage 1: 分类
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

            for b in range(batch_size):
                # 获取该样本的数据
                sample_logits = logits[b]  # [seq_len, num_classes]
                sample_line_ids = batch.get("line_ids", None)
                if sample_line_ids is not None:
                    sample_line_ids = sample_line_ids[b]  # [seq_len]

                sample_line_parent_ids = batch.get("line_parent_ids", None)
                sample_line_bboxes = batch.get("line_bboxes", None)

                # 聚合到 line-level
                if sample_line_ids is None:
                    continue

                # 计算每行的预测类别
                line_preds = {}
                line_texts = {}  # 简化：用类别名作为 text

                for token_idx, line_id in enumerate(sample_line_ids.cpu().tolist()):
                    if line_id < 0:
                        continue
                    pred_class = sample_logits[token_idx].argmax().item()
                    if line_id not in line_preds:
                        line_preds[line_id] = pred_class

                num_lines = len(line_preds)
                if num_lines == 0:
                    continue

                # Stage 2: 提取特征
                text_seq_len = batch["input_ids"].shape[1]
                text_hidden = hidden_states[b:b+1, :text_seq_len, :]

                line_features, line_mask = model.feature_extractor.extract_line_features(
                    text_hidden, sample_line_ids.unsqueeze(0), pooling="mean"
                )  # [1, max_lines, H], [1, max_lines]

                line_features = line_features[0]  # [max_lines, H]
                line_mask = line_mask[0]  # [max_lines]
                actual_num_lines = line_mask.sum().item()

                # Stage 3: 预测父节点
                pred_parents = [-1] * actual_num_lines  # 默认都指向 ROOT

                if model.use_gru:
                    parent_logits = model.stage3(
                        line_features.unsqueeze(0),
                        line_mask.unsqueeze(0)
                    )  # [1, L+1, L+1]

                    for child_idx in range(actual_num_lines):
                        child_logits = parent_logits[0, child_idx + 1, :child_idx + 2]
                        pred_parent_idx = child_logits.argmax().item()
                        # 0 表示 ROOT (-1)，其他表示实际行号
                        pred_parents[child_idx] = pred_parent_idx - 1
                else:
                    for child_idx in range(1, actual_num_lines):
                        parent_candidates = line_features[:child_idx]
                        child_feat = line_features[child_idx]
                        scores = model.stage3(parent_candidates, child_feat)
                        pred_parents[child_idx] = scores.argmax().item()

                # Stage 4: 预测关系
                pred_relations = ["none"] * actual_num_lines

                if sample_line_bboxes is not None:
                    sample_bboxes = sample_line_bboxes[b]  # [max_lines, 4]

                    for child_idx in range(actual_num_lines):
                        parent_idx = pred_parents[child_idx]
                        if parent_idx < 0 or parent_idx >= actual_num_lines:
                            continue

                        parent_feat = line_features[parent_idx]
                        child_feat = line_features[child_idx]
                        parent_bbox = sample_bboxes[parent_idx]
                        child_bbox = sample_bboxes[child_idx]

                        geom_feat = compute_geometry_features(parent_bbox, child_bbox)

                        rel_logits = model.stage4(
                            parent_feat.unsqueeze(0),
                            child_feat.unsqueeze(0),
                            geom_feat.unsqueeze(0).to(device),
                        )
                        pred_rel = rel_logits.argmax(dim=1).item()
                        pred_relations[child_idx] = ID2RELATION.get(pred_rel, "none")

                # 构建 JSON 格式结果
                doc_result = []
                for line_idx in range(actual_num_lines):
                    class_id = line_preds.get(line_idx, 0)
                    class_name = id2label.get(class_id, f"class_{class_id}")

                    doc_result.append({
                        "class": class_name,
                        "text": f"line_{line_idx}",  # 简化处理
                        "parent_id": pred_parents[line_idx],
                        "relation": pred_relations[line_idx],
                    })

                # 保存 JSON 文件
                json_path = os.path.join(output_dir, f"doc_{doc_idx:04d}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_result, f, ensure_ascii=False, indent=2)

                all_results.append(doc_result)
                doc_idx += 1

    logger.info(f"Inference completed. {doc_idx} documents saved to {output_dir}")
    return all_results


def evaluate_with_hrdoc(
    gt_dir: str,
    pred_dir: str,
    use_teds: bool = True,
) -> Dict[str, float]:
    """
    使用 HRDoc 工具评估预测结果

    Args:
        gt_dir: Ground truth JSON 文件目录
        pred_dir: 预测结果 JSON 文件目录
        use_teds: 是否计算 TEDS（需要 apted 库）

    Returns:
        评估指标字典
    """
    results = {}

    try:
        # 分类 F1
        from sklearn.metrics import f1_score

        gt_classes = []
        pred_classes = []

        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".json")])
        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".json")])

        # 匹配文件
        common_files = set(gt_files) & set(pred_files)

        for filename in common_files:
            gt_data = json.load(open(os.path.join(gt_dir, filename)))
            pred_data = json.load(open(os.path.join(pred_dir, filename)))

            if len(gt_data) != len(pred_data):
                logger.warning(f"Skipping {filename}: different number of lines")
                continue

            for gt_item, pred_item in zip(gt_data, pred_data):
                gt_class = gt_item.get("class", "unknown")
                pred_class = pred_item.get("class", "unknown")

                # 转为 ID
                class2id = {v: k for k, v in ID2CLASS.items()}
                gt_classes.append(class2id.get(gt_class, 0))
                pred_classes.append(class2id.get(pred_class, 0))

        if gt_classes:
            results["line_macro_f1"] = f1_score(gt_classes, pred_classes, average="macro")
            results["line_micro_f1"] = f1_score(gt_classes, pred_classes, average="micro")
            logger.info(f"Line-level Macro F1: {results['line_macro_f1']:.4f}")
            logger.info(f"Line-level Micro F1: {results['line_micro_f1']:.4f}")

        # TEDS
        if use_teds:
            try:
                from doc_utils import generate_doc_tree_from_log_line_level, tree_edit_distance

                teds_list = []

                for filename in common_files:
                    gt_data = json.load(open(os.path.join(gt_dir, filename)))
                    pred_data = json.load(open(os.path.join(pred_dir, filename)))

                    if len(gt_data) != len(pred_data):
                        continue

                    gt_texts = [f"{t['class']}:{t.get('text', '')}" for t in gt_data]
                    gt_parents = [t["parent_id"] for t in gt_data]
                    gt_relations = [t.get("relation", "none") for t in gt_data]

                    pred_texts = [f"{t['class']}:{t.get('text', '')}" for t in pred_data]
                    pred_parents = [t["parent_id"] for t in pred_data]
                    pred_relations = [t.get("relation", "none") for t in pred_data]

                    try:
                        gt_tree = generate_doc_tree_from_log_line_level(gt_texts, gt_parents, gt_relations)
                        pred_tree = generate_doc_tree_from_log_line_level(pred_texts, pred_parents, pred_relations)
                        _, teds = tree_edit_distance(pred_tree, gt_tree)
                        teds_list.append(teds)
                    except Exception as e:
                        logger.warning(f"TEDS error for {filename}: {e}")
                        continue

                if teds_list:
                    results["macro_teds"] = sum(teds_list) / len(teds_list)
                    logger.info(f"Macro TEDS: {results['macro_teds']:.4f}")

            except ImportError as e:
                logger.warning(f"TEDS evaluation skipped: {e}")

    except Exception as e:
        logger.error(f"HRDoc evaluation failed: {e}")

    return results


def evaluate(model, eval_loader, device, args):
    """评估模型"""
    model.eval()

    total_loss = 0
    total_cls_loss = 0
    total_parent_loss = 0
    total_rel_loss = 0

    all_preds = []
    all_labels = []
    parent_correct = 0
    parent_total = 0
    rel_correct = 0
    rel_total = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                bbox=batch["bbox"],
                attention_mask=batch["attention_mask"],
                image=batch.get("image"),
                labels=batch["labels"],
                line_ids=batch.get("line_ids"),
                line_parent_ids=batch.get("line_parent_ids"),
                line_relations=batch.get("line_relations"),
                line_bboxes=batch.get("line_bboxes"),
            )

            total_loss += outputs["loss"].item()
            total_cls_loss += outputs["cls_loss"].item()

            if "parent_loss" in outputs:
                total_parent_loss += outputs["parent_loss"].item()
            if "rel_loss" in outputs:
                total_rel_loss += outputs["rel_loss"].item()

            # 收集分类预测
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            labels = batch["labels"]

            for pred, label in zip(preds, labels):
                mask = label != -100
                all_preds.extend(pred[mask].cpu().tolist())
                all_labels.extend(label[mask].cpu().tolist())

            # 收集 Stage 3/4 准确率
            if "parent_acc" in outputs:
                parent_correct += outputs["parent_acc"] * len(batch["input_ids"])
                parent_total += len(batch["input_ids"])
            if "rel_acc" in outputs:
                rel_correct += outputs["rel_acc"] * len(batch["input_ids"])
                rel_total += len(batch["input_ids"])

    num_batches = len(eval_loader)

    # 计算分类指标
    macro_f1, per_class = compute_macro_f1(all_preds, all_labels, num_classes=NUM_LABELS)
    accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_preds)

    results = {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

    if total_parent_loss > 0:
        results["parent_loss"] = total_parent_loss / num_batches
        if parent_total > 0:
            results["parent_acc"] = parent_correct / parent_total

    if total_rel_loss > 0:
        results["rel_loss"] = total_rel_loss / num_batches
        if rel_total > 0:
            results["rel_acc"] = rel_correct / rel_total

    # 打印每类指标
    log_per_class_metrics(per_class, class_names=LABEL_LIST, title="Eval Per-Class Metrics")

    return results


def evaluate_e2e(model, eval_loader, device, args, global_step: int = 0):
    """
    端到端评估：使用 HRDoc 工具进行完整评估

    1. 运行推理，生成预测 JSON
    2. 调用 classify_eval 计算 line-level F1
    3. 调用 teds_eval 计算 TEDS

    Args:
        model: JointModel
        eval_loader: 评估数据加载器
        device: 设备
        args: 训练参数
        global_step: 当前步数（用于输出目录命名）

    Returns:
        评估结果字典
    """
    if not args.use_hrdoc_eval:
        return {}

    logger.info("=" * 60)
    logger.info(f"Running End-to-End Evaluation (Step {global_step})")
    logger.info("=" * 60)

    # 创建临时目录存放预测结果
    pred_dir = os.path.join(args.output_dir, f"eval_pred_step{global_step}")
    os.makedirs(pred_dir, exist_ok=True)

    # 运行推理
    inference_to_json(model, eval_loader, device, pred_dir, id2label=ID2CLASS)

    # TODO: 需要 GT JSON 目录
    # 当前简化处理：仅输出推理结果，不进行 TEDS 评估
    # 完整评估需要将 eval 数据集转为 HRDoc JSON 格式

    results = {
        "e2e_eval_dir": pred_dir,
    }

    logger.info(f"Predictions saved to: {pred_dir}")
    logger.info("Note: Full TEDS evaluation requires GT JSON files")
    logger.info("=" * 60)

    return results


def train(args: JointTrainingArguments):
    """主训练函数"""

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # 加载配置和实验管理器
    config, data_dir, exp_manager = load_config_and_setup(args)

    # Quick 模式：覆盖参数以便快速测试
    if args.quick:
        logger.info("=" * 60)
        logger.info("QUICK TEST MODE - Using minimal settings")
        logger.info("=" * 60)
        # 减少数据量
        if args.max_train_samples == -1:
            args.max_train_samples = 10  # 只用 10 个样本
        # 减少训练步数
        args.max_steps = min(args.max_steps, 20)
        # 减小 batch size
        args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = 1
        # 减少评估/保存频率
        args.eval_steps = 10
        args.save_steps = 10
        args.logging_steps = 5

    # 设置随机种子
    set_seed(args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 打印配置
    logger.info("=" * 60)
    logger.info("Joint Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Environment:    {args.env}")
    logger.info(f"Dataset:        {args.dataset}")
    logger.info(f"Quick Mode:     {args.quick}")
    logger.info(f"Model Path:     {args.model_name_or_path}")
    logger.info(f"Output Dir:     {args.output_dir}")
    logger.info("-" * 60)
    logger.info("Loss Weights (Paper: L = L_cls + α₁·L_par + α₂·L_rel):")
    logger.info(f"  lambda_cls:    {args.lambda_cls}")
    logger.info(f"  lambda_parent: {args.lambda_parent} (α₁)")
    logger.info(f"  lambda_rel:    {args.lambda_rel} (α₂)")
    logger.info("-" * 60)
    logger.info("Model Configuration:")
    logger.info(f"  use_gru:       {args.use_gru} (paper method)")
    logger.info(f"  use_soft_mask: {args.use_soft_mask} (paper method)")
    logger.info(f"  use_focal_loss:{args.use_focal_loss} (paper method)")
    logger.info("-" * 60)
    logger.info("Training Parameters:")
    logger.info(f"  max_steps:     {args.max_steps}")
    logger.info(f"  max_samples:   {args.max_train_samples} (-1=all)")
    logger.info(f"  batch_size:    {args.per_device_train_batch_size}")
    logger.info(f"  grad_accum:    {args.gradient_accumulation_steps}")
    logger.info(f"  learning_rate: {args.learning_rate} (Stage 1)")
    logger.info(f"  lr_stage34:    {args.learning_rate_stage34} (Stage 3/4)")
    logger.info(f"  fp16:          {args.fp16}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[Dry run mode - exiting]")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 标记 stage 开始
    exp_manager.mark_stage_started(args.exp, "joint", args.dataset)

    # 加载 tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(args.model_name_or_path)

    # 创建数据加载器
    logger.info("Creating dataloaders...")
    train_loader, eval_loader, raw_train_dataset = create_dataloaders(tokenizer, data_dir, args)
    logger.info(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader) if eval_loader else 0}")

    # 加载模型
    logger.info("Loading models...")

    # Stage 1: LayoutXLM
    stage1_config = LayoutXLMConfig.from_pretrained(args.model_name_or_path)
    stage1_config.num_labels = NUM_LABELS
    stage1_config.id2label = get_id2label()
    stage1_config.label2id = get_label2id()

    stage1_model = LayoutXLMForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=stage1_config,
    )

    # Stage 2: Feature Extractor (无参数)
    feature_extractor = LineFeatureExtractor()

    # Stage 3: ParentFinder (论文方法选择)
    if args.use_gru:
        logger.info("Using ParentFinderGRU (paper method)")
        stage3_model = ParentFinderGRU(
            hidden_size=768,
            gru_hidden_size=512,
            num_classes=NUM_LABELS,
            dropout=0.1,
            use_soft_mask=args.use_soft_mask,
        )

        # 设置 M_cp（如果启用 Soft-Mask）
        if args.use_soft_mask:
            logger.info("Building Child-Parent Distribution Matrix (M_cp) from training data...")
            cp_matrix = build_child_parent_matrix_from_dataset(raw_train_dataset, num_classes=NUM_LABELS)
            stage3_model.set_child_parent_matrix(cp_matrix.get_tensor(device))
            logger.info("  M_cp initialized successfully")
    else:
        logger.info("Using SimpleParentFinder (simplified method)")
        stage3_model = SimpleParentFinder(hidden_size=768, dropout=0.1)

    # Stage 4: RelationClassifier
    stage4_model = MultiClassRelationClassifier(
        hidden_size=768,
        num_relations=4,
        use_geometry=True,
        dropout=0.1,
    )

    # 联合模型
    model = JointModel(
        stage1_model=stage1_model,
        stage3_model=stage3_model,
        stage4_model=stage4_model,
        feature_extractor=feature_extractor,
        lambda_cls=args.lambda_cls,
        lambda_parent=args.lambda_parent if not args.disable_stage34 else 0.0,
        lambda_rel=args.lambda_rel if not args.disable_stage34 else 0.0,
        use_focal_loss=args.use_focal_loss,
        use_gru=args.use_gru,
    )
    model.to(device)

    # 打印模型参数量
    stage1_params = sum(p.numel() for p in model.stage1.parameters())
    stage3_params = sum(p.numel() for p in model.stage3.parameters())
    stage4_params = sum(p.numel() for p in model.stage4.parameters())
    total_params = stage1_params + stage3_params + stage4_params
    logger.info(f"Model parameters: Stage1={stage1_params:,}, Stage3={stage3_params:,}, Stage4={stage4_params:,}, Total={total_params:,}")

    # 优化器：不同学习率
    optimizer_grouped_parameters = [
        # Stage 1 参数
        {
            "params": [p for n, p in model.stage1.named_parameters() if p.requires_grad],
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        # Stage 3 参数
        {
            "params": [p for n, p in model.stage3.named_parameters() if p.requires_grad],
            "lr": args.learning_rate_stage34,
            "weight_decay": 0.0,
        },
        # Stage 4 参数
        {
            "params": [p for n, p in model.stage4.named_parameters() if p.requires_grad],
            "lr": args.learning_rate_stage34,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # 学习率调度器
    num_training_steps = args.max_steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None

    # 训练循环
    logger.info("Starting training...")
    model.train()

    global_step = 0
    best_metric = 0
    best_checkpoint = None

    train_iter = iter(train_loader)

    progress_bar = tqdm(total=args.max_steps, desc="Training")

    while global_step < args.max_steps:
        # 获取 batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    bbox=batch["bbox"],
                    attention_mask=batch["attention_mask"],
                    image=batch.get("image"),
                    labels=batch["labels"],
                    line_ids=batch.get("line_ids"),
                    line_parent_ids=batch.get("line_parent_ids"),
                    line_relations=batch.get("line_relations"),
                    line_bboxes=batch.get("line_bboxes"),
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids=batch["input_ids"],
                bbox=batch["bbox"],
                attention_mask=batch["attention_mask"],
                image=batch.get("image"),
                labels=batch["labels"],
                line_ids=batch.get("line_ids"),
                line_parent_ids=batch.get("line_parent_ids"),
                line_relations=batch.get("line_relations"),
                line_bboxes=batch.get("line_bboxes"),
            )
            loss = outputs["loss"] / args.gradient_accumulation_steps
            loss.backward()

        # Gradient accumulation
        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)

        # Logging
        if global_step % args.logging_steps == 0:
            log_str = f"Step {global_step}: loss={outputs['loss'].item():.4f}"
            log_str += f", cls={outputs['cls_loss'].item():.4f}"
            if "parent_loss" in outputs:
                log_str += f", parent={outputs['parent_loss'].item():.4f}"
            if "rel_loss" in outputs:
                log_str += f", rel={outputs['rel_loss'].item():.4f}"
            logger.info(log_str)

        # Evaluation
        if global_step % args.eval_steps == 0 and eval_loader:
            logger.info(f"\n--- Evaluation at step {global_step} ---")
            eval_results = evaluate(model, eval_loader, device, args)

            log_str = f"Eval: loss={eval_results['loss']:.4f}"
            log_str += f", acc={eval_results['accuracy']:.4f}"
            log_str += f", macro_f1={eval_results['macro_f1']:.4f}"
            if "parent_acc" in eval_results:
                log_str += f", parent_acc={eval_results['parent_acc']:.4f}"
            if "rel_acc" in eval_results:
                log_str += f", rel_acc={eval_results['rel_acc']:.4f}"
            logger.info(log_str)

            # 保存最佳模型
            current_metric = eval_results["macro_f1"]
            if current_metric > best_metric:
                best_metric = current_metric
                best_checkpoint = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                # 保存
                os.makedirs(best_checkpoint, exist_ok=True)
                model.stage1.save_pretrained(os.path.join(best_checkpoint, "stage1"))
                torch.save(model.stage3.state_dict(), os.path.join(best_checkpoint, "stage3.pt"))
                torch.save(model.stage4.state_dict(), os.path.join(best_checkpoint, "stage4.pt"))
                tokenizer.save_pretrained(os.path.join(best_checkpoint, "stage1"))

                logger.info(f"New best model saved to {best_checkpoint} (macro_f1={best_metric:.4f})")

            model.train()

        # Save checkpoint
        if global_step % args.save_steps == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            model.stage1.save_pretrained(os.path.join(checkpoint_dir, "stage1"))
            torch.save(model.stage3.state_dict(), os.path.join(checkpoint_dir, "stage3.pt"))
            torch.save(model.stage4.state_dict(), os.path.join(checkpoint_dir, "stage4.pt"))
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "stage1"))

            logger.info(f"Checkpoint saved to {checkpoint_dir}")

    progress_bar.close()

    # 最终保存
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.stage1.save_pretrained(os.path.join(final_dir, "stage1"))
    torch.save(model.stage3.state_dict(), os.path.join(final_dir, "stage3.pt"))
    torch.save(model.stage4.state_dict(), os.path.join(final_dir, "stage4.pt"))
    tokenizer.save_pretrained(os.path.join(final_dir, "stage1"))

    # 更新实验状态
    exp_manager.mark_stage_completed(
        args.exp, "joint", args.dataset,
        best_checkpoint=os.path.basename(best_checkpoint) if best_checkpoint else "final",
        metrics={"macro_f1": best_metric} if best_metric > 0 else None,
    )

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best macro_f1: {best_metric:.4f}")
    logger.info(f"Best checkpoint: {best_checkpoint}")
    logger.info(f"Final model: {final_dir}")
    logger.info("=" * 60)


def main():
    parser = HfArgumentParser((JointTrainingArguments,))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    train(args)


if __name__ == "__main__":
    main()

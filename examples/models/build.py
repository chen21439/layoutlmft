#!/usr/bin/env python
# coding=utf-8
"""
模型构建工厂 - 统一的模型加载和组装接口

提供两种构建方式：
1. build_joint_model(): 从各组件构建新模型（训练时使用）
2. load_joint_model(): 从 checkpoint 加载已训练模型（推理时使用）

此文件只负责模型的构建和加载，不包含训练循环。
"""

import os
import logging
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_joint_model(
    stage1_model,
    stage3_model: nn.Module,
    stage4_model: nn.Module,
    feature_extractor,
    lambda_cls: float = 1.0,
    lambda_parent: float = 1.0,
    lambda_rel: float = 1.0,
    use_focal_loss: bool = True,
    use_gru: bool = False,
):
    """
    从各组件构建 JointModel（训练时使用）

    Args:
        stage1_model: LayoutXLMForTokenClassification 实例
        stage3_model: ParentFinderGRU 或 SimpleParentFinder 实例
        stage4_model: MultiClassRelationClassifier 实例
        feature_extractor: LineFeatureExtractor 实例
        lambda_cls: 分类损失权重
        lambda_parent: 父节点损失权重
        lambda_rel: 关系损失权重
        use_focal_loss: 是否使用 Focal Loss
        use_gru: 是否使用 GRU

    Returns:
        JointModel 实例
    """
    from .joint_model import JointModel

    return JointModel(
        stage1_model=stage1_model,
        stage3_model=stage3_model,
        stage4_model=stage4_model,
        feature_extractor=feature_extractor,
        lambda_cls=lambda_cls,
        lambda_parent=lambda_parent,
        lambda_rel=lambda_rel,
        use_focal_loss=use_focal_loss,
        use_gru=use_gru,
    )


def load_joint_model(
    model_path: str,
    device: torch.device = None,
    config: Any = None,
) -> Tuple[nn.Module, Any]:
    """
    从 checkpoint 加载 JointModel（推理时使用）

    统一的模型加载接口，训练脚本和推理脚本都使用此函数。

    Args:
        model_path: checkpoint 目录路径，包含 stage1/, stage3.pt, stage4.pt
        device: 目标设备
        config: 配置对象（可选，用于 tokenizer fallback 路径）

    Returns:
        (model, tokenizer) 元组
    """
    from .joint_model import JointModel
    from layoutlmft.data.labels import NUM_LABELS, get_id2label, get_label2id
    from layoutlmft.models.layoutxlm import (
        LayoutXLMForTokenClassification,
        LayoutXLMConfig,
        LayoutXLMTokenizerFast,
    )
    from layoutlmft.models.relation_classifier import (
        LineFeatureExtractor,
        MultiClassRelationClassifier,
    )

    # 延迟导入，避免循环依赖
    import sys
    STAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stage_dir = os.path.join(STAGE_ROOT, "stage")
    if stage_dir not in sys.path:
        sys.path.insert(0, stage_dir)
    from train_parent_finder import ParentFinderGRU, SimpleParentFinder

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading joint model from: {model_path}")

    # ==================== Stage 1: LayoutXLM ====================
    stage1_path = os.path.join(model_path, "stage1")
    if not os.path.exists(stage1_path):
        raise ValueError(f"Stage 1 model not found: {stage1_path}")

    stage1_config = LayoutXLMConfig.from_pretrained(stage1_path)
    stage1_config.num_labels = NUM_LABELS
    stage1_config.id2label = get_id2label()
    stage1_config.label2id = get_label2id()

    stage1_model = LayoutXLMForTokenClassification.from_pretrained(
        stage1_path, config=stage1_config,
    )
    logger.info(f"Loaded Stage 1 from: {stage1_path}")

    # ==================== Tokenizer ====================
    tokenizer = _load_tokenizer(stage1_path, config)

    # ==================== Stage 2: Feature Extractor ====================
    feature_extractor = LineFeatureExtractor()

    # ==================== Stage 3: ParentFinder ====================
    stage3_path = os.path.join(model_path, "stage3.pt")
    stage3_state = torch.load(stage3_path, map_location="cpu")
    use_gru = any("gru" in k for k in stage3_state.keys())

    if use_gru:
        gru_hidden_size = stage3_state.get("gru.weight_hh_l0", torch.zeros(1, 512)).shape[1]
        stage3_model = ParentFinderGRU(
            hidden_size=768, gru_hidden_size=gru_hidden_size,
            num_classes=NUM_LABELS, dropout=0.0, use_soft_mask=False,
        )
        logger.info(f"Using ParentFinderGRU (gru_hidden_size={gru_hidden_size})")
    else:
        stage3_model = SimpleParentFinder(hidden_size=768, dropout=0.0)
        logger.info("Using SimpleParentFinder")

    stage3_model.load_state_dict(stage3_state, strict=False)
    logger.info(f"Loaded Stage 3 from: {stage3_path}")

    # ==================== Stage 4: RelationClassifier ====================
    stage4_path = os.path.join(model_path, "stage4.pt")
    stage4_state = torch.load(stage4_path, map_location="cpu")
    hidden_size = stage4_state["fc.weight"].shape[1] // 2 if "fc.weight" in stage4_state else (512 if use_gru else 768)

    stage4_model = MultiClassRelationClassifier(
        hidden_size=hidden_size, num_relations=3, use_geometry=False, dropout=0.0,
    )
    stage4_model.load_state_dict(stage4_state)
    logger.info(f"Loaded Stage 4 from: {stage4_path}")

    # ==================== 组装 JointModel ====================
    model = JointModel(
        stage1_model=stage1_model,
        stage3_model=stage3_model,
        stage4_model=stage4_model,
        feature_extractor=feature_extractor,
        use_gru=use_gru,
    )

    model = model.to(device)
    model.eval()

    return model, tokenizer


def _load_tokenizer(stage1_path: str, config: Any = None):
    """
    加载 tokenizer，按优先级尝试多个来源

    优先级：checkpoint目录 > config.model.local_path > 远程模型
    """
    from layoutlmft.models.layoutxlm import LayoutXLMTokenizerFast

    tokenizer_sources = [stage1_path]

    # 从 config 获取本地路径
    if config and hasattr(config, 'model') and hasattr(config.model, 'local_path') and config.model.local_path:
        tokenizer_sources.append(config.model.local_path)

    # 远程模型作为最后 fallback
    tokenizer_sources.append("microsoft/layoutxlm-base")

    for source in tokenizer_sources:
        try:
            tokenizer = LayoutXLMTokenizerFast.from_pretrained(source)
            logger.info(f"Loaded tokenizer from: {source}")
            return tokenizer
        except Exception as e:
            logger.debug(f"Failed to load tokenizer from {source}: {e}")
            continue

    raise ValueError("Failed to load tokenizer from any source")


def get_latest_joint_checkpoint(
    config,
    exp: str = None,
    dataset: str = None,
) -> Optional[str]:
    """
    自动检测最新的 Joint 模型 checkpoint

    Args:
        config: 配置对象
        exp: 实验 ID（可选）
        dataset: 数据集名称（可选，用于过滤）

    Returns:
        checkpoint 路径，如果没找到返回 None
    """
    import glob

    # 延迟导入
    import sys
    STAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stage_util_dir = os.path.join(STAGE_ROOT, "stage", "util")
    if stage_util_dir not in sys.path:
        sys.path.insert(0, stage_util_dir)
    from experiment_manager import get_experiment_manager

    exp_manager = get_experiment_manager(config)
    exp_dir = exp_manager.get_experiment_dir(exp)

    joint_dirs = glob.glob(os.path.join(exp_dir, "joint_*"))

    if dataset:
        dataset_dirs = [d for d in joint_dirs if dataset in d]
        if dataset_dirs:
            joint_dirs = dataset_dirs

    latest_model = None
    latest_mtime = 0

    for joint_dir in joint_dirs:
        # 检查根目录
        stage3_path = os.path.join(joint_dir, "stage3.pt")
        if os.path.exists(stage3_path):
            mtime = os.path.getmtime(stage3_path)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_model = joint_dir

        # 检查 checkpoint 子目录
        for subdir in glob.glob(os.path.join(joint_dir, "checkpoint-*")):
            stage3_path = os.path.join(subdir, "stage3.pt")
            if os.path.exists(stage3_path):
                mtime = os.path.getmtime(stage3_path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_model = subdir

    return latest_model

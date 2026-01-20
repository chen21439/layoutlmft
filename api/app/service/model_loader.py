#!/usr/bin/env python
# coding=utf-8
"""
Model Loader - Singleton pattern for model loading

Loads model from a single checkpoint directory that contains:
- stage1/: Backbone (LayoutXLM) weights
- cls_head.pt: Classification head weights
- pytorch_model.bin: Construct model weights (parent/sibling heads)
- config.json: Construct model config
"""

import os
import sys
import logging
from typing import Optional, Tuple, Any
from threading import Lock

# Add project paths
# 注意顺序：STAGE_ROOT 必须在 COMP_HRDOC_ROOT 之前，因为 load_joint_model 在 stage/models/build.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
EXAMPLES_ROOT = os.path.join(PROJECT_ROOT, "examples")
sys.path.insert(0, EXAMPLES_ROOT)
COMP_HRDOC_ROOT = os.path.join(EXAMPLES_ROOT, "comp_hrdoc")
sys.path.insert(0, COMP_HRDOC_ROOT)
STAGE_ROOT = os.path.join(EXAMPLES_ROOT, "stage")
sys.path.insert(0, STAGE_ROOT)  # 最后插入，优先级最高

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton model loader for inference."""

    _instance: Optional["ModelLoader"] = None
    _lock: Lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._tokenizer = None
        self._predictor = None
        self._construct_model = None  # Construct model for TOC
        self._device = None
        self._checkpoint_path = None
        self._is_joint_training_model = False  # 是否是联合训练模型
        self._initialized = True

    def load(
        self,
        checkpoint_path: str,
        device: str = None,
        config: Any = None,
    ) -> None:
        """
        Load model from checkpoint.

        支持两种格式：
        1. 联合训练模型（包含 stage1/ 子目录，没有 stage3.pt）
           - 所有组件在同一目录：stage1/ + cls_head.pt + pytorch_model.bin
        2. 标准 JointModel（包含 stage3.pt, stage4.pt）
           - 用于兼容旧格式

        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to use ('cuda' or 'cpu')
            config: Optional config object
        """
        if self._model is not None and self._checkpoint_path == checkpoint_path:
            logger.info(f"Model already loaded from: {checkpoint_path}")
            return

        import torch
        from engines.predictor import Predictor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model from: {checkpoint_path}")
        logger.info(f"Device: {device}")

        # 检测 checkpoint 格式
        stage1_subdir = os.path.join(checkpoint_path, "stage1")
        stage3_file = os.path.join(checkpoint_path, "stage3.pt")
        is_joint_training_format = os.path.isdir(stage1_subdir) and not os.path.exists(stage3_file)

        if is_joint_training_format:
            # 联合训练格式：创建 JointModel 并加载权重
            logger.info("Detected joint training checkpoint format")
            self._model, self._tokenizer = self._load_joint_training_model(
                checkpoint_path, device, config
            )
            # 联合训练格式：Construct 在同一目录
            self._load_construct_model(checkpoint_path, device)
            self._is_joint_training_model = True
        else:
            # 标准 JointModel 格式（兼容旧格式）
            logger.info("Detected standard JointModel checkpoint format")
            from models.build import load_joint_model
            self._model, self._tokenizer = load_joint_model(
                checkpoint_path, torch.device(device), config
            )
            self._is_joint_training_model = False

        self._predictor = Predictor(self._model, torch.device(device))
        self._device = device
        self._checkpoint_path = checkpoint_path

        logger.info("Model loaded successfully")

    def _load_joint_training_model(self, checkpoint_path: str, device: str, config: Any = None):
        """加载联合训练 checkpoint（stage1/ + cls_head.pt + pytorch_model.bin）

        创建完整的 JointModel 以复用 Predictor 的特征提取逻辑。
        """
        import torch
        from layoutlmft.models.layoutxlm import (
            LayoutXLMForTokenClassification,
            LayoutXLMConfig,
            LayoutXLMTokenizerFast,
        )
        from layoutlmft.data.labels import NUM_LABELS, get_id2label, get_label2id
        from models.joint_model import JointModel

        # 延迟导入 stage3/stage4 相关模块
        from train_parent_finder import SimpleParentFinder
        from layoutlmft.models.relation_classifier import MultiClassRelationClassifier

        stage1_path = os.path.join(checkpoint_path, "stage1")
        logger.info(f"Loading backbone from: {stage1_path}")

        # 加载 backbone
        stage1_config = LayoutXLMConfig.from_pretrained(stage1_path)
        stage1_config.num_labels = NUM_LABELS
        stage1_config.id2label = get_id2label()
        stage1_config.label2id = get_label2id()

        backbone = LayoutXLMForTokenClassification.from_pretrained(
            stage1_path,
            config=stage1_config,
        )

        # 创建 dummy stage3/stage4（Predictor.extract_features 不需要它们）
        dummy_stage3 = SimpleParentFinder(hidden_size=768, dropout=0.0)
        dummy_stage4 = MultiClassRelationClassifier(hidden_size=768, num_relations=3, dropout=0.0)

        # 创建 JointModel
        model = JointModel(
            stage1_model=backbone,
            stage3_model=dummy_stage3,
            stage4_model=dummy_stage4,
            use_gru=False,
        )

        # 加载 cls_head 权重（如果存在）
        cls_head_path = os.path.join(checkpoint_path, "cls_head.pt")
        if os.path.exists(cls_head_path):
            logger.info(f"Loading cls_head from: {cls_head_path}")
            cls_head_state = torch.load(cls_head_path, map_location="cpu")
            model.cls_head.load_state_dict(cls_head_state)
        else:
            logger.warning(f"cls_head weights not found: {cls_head_path}")

        model = model.to(device)
        model.eval()

        # 加载 tokenizer
        try:
            tokenizer = LayoutXLMTokenizerFast.from_pretrained(checkpoint_path)
            logger.info(f"Loaded tokenizer from: {checkpoint_path}")
        except Exception:
            tokenizer = LayoutXLMTokenizerFast.from_pretrained(stage1_path)
            logger.info(f"Loaded tokenizer from: {stage1_path}")

        logger.info("Joint training model loaded successfully")
        return model, tokenizer

    def _load_construct_model(self, construct_checkpoint: str, device: str) -> None:
        """Load Construct model for TOC generation."""
        import torch
        import json

        if not os.path.exists(construct_checkpoint):
            logger.warning(f"Construct checkpoint not found: {construct_checkpoint}")
            return

        logger.info(f"Loading Construct model from: {construct_checkpoint}")

        # Load config
        config_path = os.path.join(construct_checkpoint, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {'hidden_size': 768, 'num_layers': 3, 'num_categories': 14}

        # Build model (从 comp_hrdoc 导入，避免与 stage/models 冲突)
        from comp_hrdoc.models.construct_only import build_construct_from_features
        model = build_construct_from_features(
            hidden_size=config.get('hidden_size', 768),
            num_categories=config.get('num_categories', 14),
            num_heads=config.get('num_heads', 12),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
        )

        # Load weights
        weights_path = os.path.join(construct_checkpoint, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            # 过滤掉 RoPE 缓存 buffer（联合训练格式可能包含这些）
            state_dict = {k: v for k, v in state_dict.items()
                          if 'cos_cached' not in k and 'sin_cached' not in k}
            model.load_state_dict(state_dict, strict=False)
        else:
            # Fallback to old format
            construct_weights = os.path.join(construct_checkpoint, "construct_model.pt")
            if os.path.exists(construct_weights):
                model.construct_module.load_state_dict(
                    torch.load(construct_weights, map_location=device)
                )

        model = model.to(device)
        model.eval()
        self._construct_model = model

        logger.info("Construct model loaded successfully")

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    @property
    def predictor(self):
        if self._predictor is None:
            raise RuntimeError("Predictor not loaded. Call load() first.")
        return self._predictor

    @property
    def device(self):
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def construct_model(self):
        """Construct model for TOC generation (may be None)."""
        return self._construct_model

    @property
    def has_construct_model(self) -> bool:
        return self._construct_model is not None

    @property
    def is_joint_training_model(self) -> bool:
        """是否是联合训练模型（Stage1 + Construct，无 stage3/stage4）"""
        return self._is_joint_training_model


# Global singleton instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def load_model(
    checkpoint_path: str,
    device: str = None,
    config: Any = None,
) -> ModelLoader:
    """
    Load model (convenience function).

    Checkpoint directory should contain:
    - stage1/: Backbone weights
    - cls_head.pt: Classification head
    - pytorch_model.bin: Construct model (parent/sibling heads)

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to use
        config: Optional config object

    Returns:
        ModelLoader instance
    """
    loader = get_model_loader()
    loader.load(checkpoint_path, device, config)
    return loader

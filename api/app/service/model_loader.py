#!/usr/bin/env python
# coding=utf-8
"""
Model Loader - Singleton pattern for model loading

直接使用 comp_hrdoc 的组件：
- StageFeatureExtractor: 加载 Stage1Backbone（backbone + line_pooling + line_enhancer + cls_head）
- ConstructPredictor: TOC 推理

Checkpoint 目录结构：
- stage1/: Backbone (LayoutXLM) weights
- cls_head.pt: Classification head weights
- line_enhancer.pt: Line feature enhancer weights
- pytorch_model.bin: Construct model weights
- config.json: Construct model config
"""

import os
import sys
import logging
from typing import Optional, Any
from threading import Lock

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
EXAMPLES_ROOT = os.path.join(PROJECT_ROOT, "examples")
sys.path.insert(0, EXAMPLES_ROOT)
COMP_HRDOC_ROOT = os.path.join(EXAMPLES_ROOT, "comp_hrdoc")
sys.path.insert(0, COMP_HRDOC_ROOT)

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton model loader for inference.

    使用 comp_hrdoc 的 StageFeatureExtractor 和 ConstructPredictor，
    与训练/评估代码保持一致。
    """

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
        self._feature_extractor = None  # StageFeatureExtractor
        self._construct_model = None  # ConstructFromFeatures
        self._construct_predictor = None  # ConstructPredictor
        self._device = None
        self._checkpoint_path = None
        self._attention_pool_construct = False  # 是否使用 AttentionPooling
        self._initialized = True

    def load(
        self,
        checkpoint_path: str,
        device: str = None,
        config: Any = None,
        max_regions: int = 4096,
    ) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to use ('cuda' or 'cpu')
            config: Optional config object
            max_regions: 最大区域数（默认 4096）
        """
        if self._feature_extractor is not None and self._checkpoint_path == checkpoint_path:
            logger.info(f"Model already loaded from: {checkpoint_path}")
            return

        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model from: {checkpoint_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Max regions: {max_regions}")

        # 使用 comp_hrdoc 的 StageFeatureExtractor
        from comp_hrdoc.utils.stage_feature_extractor import StageFeatureExtractor

        self._feature_extractor = StageFeatureExtractor(
            checkpoint_path=checkpoint_path,
            device=device,
            max_regions=max_regions,
        )
        logger.info("StageFeatureExtractor loaded (backbone + line_enhancer + cls_head)")

        # 加载 Construct 模型
        self._load_construct_model(checkpoint_path, device)

        self._device = device
        self._checkpoint_path = checkpoint_path

        logger.info("Model loaded successfully")

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

        # Build model
        from comp_hrdoc.models.construct_only import build_construct_from_features
        attention_pool = config.get('attention_pool_construct', False)
        if attention_pool:
            logger.info("  AttentionPooling: ENABLED (from config)")
        model = build_construct_from_features(
            hidden_size=config.get('hidden_size', 768),
            num_categories=config.get('num_categories', 14),
            num_heads=config.get('num_heads', 12),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
            attention_pool_construct=attention_pool,
        )
        self._attention_pool_construct = attention_pool

        # Load weights
        weights_path = os.path.join(construct_checkpoint, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            # 过滤掉 RoPE 缓存 buffer
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

        # 创建 ConstructPredictor
        from comp_hrdoc.engines.predictor import ConstructPredictor
        self._construct_predictor = ConstructPredictor(model, device)

        logger.info("Construct model loaded successfully")

    @property
    def feature_extractor(self):
        """StageFeatureExtractor for feature extraction and classification."""
        if self._feature_extractor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._feature_extractor

    @property
    def tokenizer(self):
        """Tokenizer from StageFeatureExtractor."""
        if self._feature_extractor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._feature_extractor.tokenizer

    @property
    def device(self):
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._feature_extractor is not None

    @property
    def construct_model(self):
        """Construct model for TOC generation (may be None)."""
        return self._construct_model

    @property
    def construct_predictor(self):
        """ConstructPredictor for TOC inference (may be None)."""
        return self._construct_predictor

    @property
    def has_construct_model(self) -> bool:
        return self._construct_model is not None

    @property
    def is_joint_training_model(self) -> bool:
        """是否是联合训练模型（现在全部使用 comp_hrdoc，总是返回 True）"""
        return True

    @property
    def attention_pool_construct(self) -> bool:
        """是否使用 AttentionPooling 替代 mean pooling"""
        return self._attention_pool_construct


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
    max_regions: int = 4096,
) -> ModelLoader:
    """
    Load model (convenience function).

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to use
        config: Optional config object
        max_regions: 最大区域数（默认 4096）

    Returns:
        ModelLoader instance
    """
    loader = get_model_loader()
    loader.load(checkpoint_path, device, config, max_regions)
    return loader

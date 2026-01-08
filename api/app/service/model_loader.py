#!/usr/bin/env python
# coding=utf-8
"""
Model Loader - Singleton pattern for model loading

Loads the joint model once at startup and reuses it for all requests.
Optionally loads Construct model for TOC generation.
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
        self._construct_checkpoint_path = None
        self._initialized = True

    def load(
        self,
        checkpoint_path: str,
        device: str = None,
        config: Any = None,
        construct_checkpoint: str = None,
    ) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to use ('cuda' or 'cpu')
            config: Optional config object
            construct_checkpoint: Path to Construct model checkpoint (optional)
        """
        if self._model is not None and self._checkpoint_path == checkpoint_path:
            logger.info(f"Model already loaded from: {checkpoint_path}")
            # Still try to load construct if not loaded
            if construct_checkpoint and self._construct_model is None:
                self._load_construct_model(construct_checkpoint, device)
            return

        import torch
        from models.build import load_joint_model
        from engines.predictor import Predictor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model from: {checkpoint_path}")
        logger.info(f"Device: {device}")

        self._model, self._tokenizer = load_joint_model(
            checkpoint_path, torch.device(device), config
        )
        self._predictor = Predictor(self._model, torch.device(device))
        self._device = device
        self._checkpoint_path = checkpoint_path

        logger.info("Model loaded successfully")

        # Load Construct model if provided
        if construct_checkpoint:
            self._load_construct_model(construct_checkpoint, device)

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
        from models.construct_only import build_construct_from_features
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
            model.load_state_dict(state_dict)
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
        self._construct_checkpoint_path = construct_checkpoint

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
    construct_checkpoint: str = None,
) -> ModelLoader:
    """
    Load model (convenience function).

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to use
        config: Optional config object
        construct_checkpoint: Path to Construct model checkpoint (optional)

    Returns:
        ModelLoader instance
    """
    loader = get_model_loader()
    loader.load(checkpoint_path, device, config, construct_checkpoint)
    return loader

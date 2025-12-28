#!/usr/bin/env python
# coding=utf-8
"""
Model Loader - Singleton pattern for model loading

Loads the joint model once at startup and reuses it for all requests.
"""

import os
import sys
import logging
from typing import Optional, Tuple, Any
from threading import Lock

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
EXAMPLES_ROOT = os.path.join(PROJECT_ROOT, "examples")
sys.path.insert(0, EXAMPLES_ROOT)
STAGE_ROOT = os.path.join(EXAMPLES_ROOT, "stage")
sys.path.insert(0, STAGE_ROOT)

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
        self._device = None
        self._checkpoint_path = None
        self._initialized = True

    def load(
        self,
        checkpoint_path: str,
        device: str = None,
        config: Any = None,
    ) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to use ('cuda' or 'cpu')
            config: Optional config object
        """
        if self._model is not None and self._checkpoint_path == checkpoint_path:
            logger.info(f"Model already loaded from: {checkpoint_path}")
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

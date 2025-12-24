#!/usr/bin/env python
# coding=utf-8
"""
engines/ - 运行骨架

放训练/推理/评估的循环与工程能力
"""

from .predictor import Predictor, PredictionOutput
from .evaluator import Evaluator, EvaluationOutput

__all__ = [
    "Predictor",
    "PredictionOutput",
    "Evaluator",
    "EvaluationOutput",
]

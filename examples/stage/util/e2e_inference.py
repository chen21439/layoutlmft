#!/usr/bin/env python
# coding=utf-8
"""
端到端推理接口 (E2E Inference Wrapper)

=== 重要说明 ===

此模块是 engines/predictor.py 的 wrapper，保持旧接口兼容。
核心逻辑统一在 Predictor 中，避免代码重复。

新代码应直接使用：
    from engines.predictor import Predictor, PredictionOutput
"""

import sys
import os

# 添加路径
STAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STAGE_ROOT not in sys.path:
    sys.path.insert(0, STAGE_ROOT)

from engines.predictor import Predictor, PredictionOutput
from typing import Dict, List, Any
import torch

# 导出 PredictionOutput 作为 E2EInferenceOutput（兼容旧代码）
E2EInferenceOutput = PredictionOutput

# 模块级 Predictor 缓存
_predictor_cache: Dict[int, Predictor] = {}


def _get_or_create_predictor(model, device=None) -> Predictor:
    """获取或创建 Predictor 实例（缓存复用）"""
    model_id = id(model)
    if model_id not in _predictor_cache:
        _predictor_cache[model_id] = Predictor(model=model, device=device)
    return _predictor_cache[model_id]


def run_e2e_inference_single(
    model,
    batch: Dict[str, torch.Tensor],
    batch_idx: int = 0,
    device: torch.device = None,
) -> PredictionOutput:
    """
    对单个样本运行端到端推理（页面级别）

    这是 Predictor.predict_single_from_batch 的 wrapper。

    Args:
        model: JointModel
        batch: 页面级别 batch，input_ids 形状为 [batch_size, seq_len]
        batch_idx: batch 中的样本索引
        device: 设备

    Returns:
        PredictionOutput: 预测结果
    """
    predictor = _get_or_create_predictor(model, device)
    return predictor.predict_single_from_batch(batch, batch_idx=batch_idx)


def run_e2e_inference_batch(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device = None,
) -> List[PredictionOutput]:
    """
    对整个 batch 运行端到端推理

    Args:
        model: JointModel
        batch: batch dict
        device: 设备

    Returns:
        List[PredictionOutput]: 每个样本的预测结果
    """
    predictor = _get_or_create_predictor(model, device)
    batch_size = batch["input_ids"].shape[0]
    return [predictor.predict_single_from_batch(batch, batch_idx=b) for b in range(batch_size)]


def run_e2e_inference_document(
    model,
    batch: Dict[str, torch.Tensor],
    doc_idx: int = 0,
    device: torch.device = None,
) -> PredictionOutput:
    """
    文档级别端到端推理

    这是 Predictor.predict_from_dict 的 wrapper。

    Args:
        model: JointModel
        batch: 文档级别 batch，包含 num_docs, chunks_per_doc 等
        doc_idx: 文档索引
        device: 设备

    Returns:
        PredictionOutput: 预测结果
    """
    predictor = _get_or_create_predictor(model, device)
    return predictor.predict_from_dict(batch, doc_idx=doc_idx)

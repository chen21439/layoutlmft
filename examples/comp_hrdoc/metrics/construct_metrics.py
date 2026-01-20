#!/usr/bin/env python
# coding=utf-8
"""
Construct 指标计算 - 纯指标实现

遵循项目结构规范：
- metrics/ 放纯指标实现，可复用、可单测
- 不放模型推理、数据读取
"""

from typing import Dict


def compute_prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    计算 Precision, Recall, F1

    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives

    Returns:
        Dict 包含 precision, recall, f1
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_construct_metrics(
    parent_preds,
    parent_labels,
    sibling_preds,
    sibling_labels,
    mask,
) -> Dict[str, float]:
    """
    计算 Construct 任务的评估指标

    Args:
        parent_preds: Parent 预测 [batch, num_lines]
        parent_labels: Parent 标签 [batch, num_lines]
        sibling_preds: Sibling 预测 [batch, num_lines]
        sibling_labels: Sibling 标签 [batch, num_lines]
        mask: 有效行掩码 [batch, num_lines]

    Returns:
        Dict 包含各项指标
    """
    import torch

    metrics = {}

    # Parent 准确率
    if parent_preds is not None and parent_labels is not None:
        parent_correct = (parent_preds == parent_labels) & mask
        parent_accuracy = parent_correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
        metrics["parent_accuracy"] = parent_accuracy

    # Sibling 准确率
    if sibling_preds is not None and sibling_labels is not None:
        sibling_correct = (sibling_preds == sibling_labels) & mask
        sibling_accuracy = sibling_correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
        metrics["sibling_accuracy"] = sibling_accuracy

    # Combined 准确率（parent 和 sibling 都正确）
    if parent_preds is not None and parent_labels is not None and sibling_preds is not None and sibling_labels is not None:
        combined_correct = parent_correct & sibling_correct
        combined_accuracy = combined_correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
        metrics["combined_accuracy"] = combined_accuracy

    return metrics

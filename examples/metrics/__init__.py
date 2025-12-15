# coding=utf-8
"""
metrics 模块 - 统一的评估指标计算

所有评估逻辑的 single source of truth，避免重复实现导致的指标不一致。
"""

from .line_eval import (
    # 核心聚合函数
    aggregate_token_to_line_predictions,
    extract_line_labels_from_tokens,
    # 行级别指标计算
    compute_line_level_metrics,
    compute_line_level_metrics_from_tokens,
    compute_line_level_metrics_batch,
    # 数据结构
    LineMetricsResult,
)

__all__ = [
    "aggregate_token_to_line_predictions",
    "extract_line_labels_from_tokens",
    "compute_line_level_metrics",
    "compute_line_level_metrics_from_tokens",
    "compute_line_level_metrics_batch",
    "LineMetricsResult",
]

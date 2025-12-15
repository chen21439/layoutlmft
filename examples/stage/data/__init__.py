#!/usr/bin/env python
# coding=utf-8
"""
统一数据加载模块

提供 Stage 1 和联合训练共用的数据加载逻辑。
"""

from .hrdoc_data_loader import (
    HRDocDataLoader,
    HRDocDataLoaderConfig,
    tokenize_with_line_boundary,
    compute_line_bboxes,
    get_label2id,
    get_id2label,
    NUM_LABELS,
    LABEL_LIST,
)

__all__ = [
    "HRDocDataLoader",
    "HRDocDataLoaderConfig",
    "tokenize_with_line_boundary",
    "compute_line_bboxes",
    "get_label2id",
    "get_id2label",
    "NUM_LABELS",
    "LABEL_LIST",
]

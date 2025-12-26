#!/usr/bin/env python
# coding=utf-8
"""
统一数据加载模块

提供文档级别的数据加载逻辑（支持跨页关系）。
"""

from .hrdoc_data_loader import (
    HRDocDataLoader,
    HRDocDataLoaderConfig,
    load_hrdoc_raw_datasets,
    load_hrdoc_raw_datasets_batched,
    tokenize_page_with_line_boundary,
    compute_line_bboxes,
    get_label2id,
    get_id2label,
    NUM_LABELS,
    LABEL_LIST,
)

from .batch import (
    Sample,
    BatchBase,
    PageLevelBatch,
    DocumentLevelBatch,
    wrap_batch,
)

__all__ = [
    # 数据加载
    "HRDocDataLoader",
    "HRDocDataLoaderConfig",
    "load_hrdoc_raw_datasets",
    "load_hrdoc_raw_datasets_batched",
    "tokenize_page_with_line_boundary",
    "compute_line_bboxes",
    "get_label2id",
    "get_id2label",
    "NUM_LABELS",
    "LABEL_LIST",
    # Batch 抽象
    "Sample",
    "BatchBase",
    "PageLevelBatch",
    "DocumentLevelBatch",
    "wrap_batch",
]

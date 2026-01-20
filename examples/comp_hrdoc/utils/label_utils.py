#!/usr/bin/env python
# coding=utf-8
"""
标签转换工具 - 负责 Stage 标签到 Construct 标签的转换

遵循项目结构规范：
- utils/ 负责横切能力与基础设施
"""

import torch
from typing import Tuple, Optional


def convert_stage_labels_to_construct(
    batch: dict,
    max_lines: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    从 Stage collator 输出转换为 Construct 标签格式

    Args:
        batch: Stage collator 输出的 batch
        max_lines: 最大行数（用于 padding）
        device: 目标设备

    Returns:
        (parent_labels, sibling_labels, line_labels) 元组
        - parent_labels: [num_docs, max_lines]
        - sibling_labels: [num_docs, max_lines, max_lines]
        - line_labels: [num_docs, max_lines]
    """
    # 从 batch 获取标签
    tree_labels = batch.get("tree_labels")
    line_labels_raw = batch.get("line_labels")

    if tree_labels is None:
        return None, None, None

    num_docs = len(tree_labels)

    # 初始化张量
    parent_labels = torch.full((num_docs, max_lines), -1, dtype=torch.long, device=device)
    sibling_labels = torch.full((num_docs, max_lines, max_lines), -1, dtype=torch.long, device=device)
    line_labels = torch.full((num_docs, max_lines), 0, dtype=torch.long, device=device)

    # 填充标签
    for doc_idx, doc_tree in enumerate(tree_labels):
        if doc_tree is None:
            continue

        # Parent labels
        if "parent" in doc_tree:
            parents = doc_tree["parent"]
            num_lines_in_doc = min(len(parents), max_lines)
            parent_labels[doc_idx, :num_lines_in_doc] = torch.tensor(
                parents[:num_lines_in_doc], dtype=torch.long, device=device
            )

        # Sibling labels (adjacency matrix)
        if "sibling" in doc_tree:
            siblings = doc_tree["sibling"]  # List[List[int]]
            num_lines_in_doc = min(len(siblings), max_lines)
            for i in range(num_lines_in_doc):
                sibling_row = siblings[i]
                num_siblings = min(len(sibling_row), max_lines)
                sibling_labels[doc_idx, i, :num_siblings] = torch.tensor(
                    sibling_row[:num_siblings], dtype=torch.long, device=device
                )

    # Line labels
    if line_labels_raw is not None:
        if isinstance(line_labels_raw, torch.Tensor):
            line_labels_raw = line_labels_raw.to(device)
            batch_size, seq_len = line_labels_raw.shape
            copy_len = min(seq_len, max_lines)
            line_labels[:batch_size, :copy_len] = line_labels_raw[:, :copy_len]
        elif isinstance(line_labels_raw, list):
            for doc_idx, doc_labels in enumerate(line_labels_raw):
                if doc_labels is not None:
                    num_lines_in_doc = min(len(doc_labels), max_lines)
                    line_labels[doc_idx, :num_lines_in_doc] = torch.tensor(
                        doc_labels[:num_lines_in_doc], dtype=torch.long, device=device
                    )

    return parent_labels, sibling_labels, line_labels

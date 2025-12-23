#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 联合训练的数据整理器 (Data Collator)

文档级别处理：
- 每个样本是一个文档（包含多个 chunks）
- Stage 1：逐 chunk 处理
- Stage 2/3/4：使用文档级别的 parent_ids 和 relations（全局索引）
"""

import os
import sys
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from layoutlmft.data.labels import NUM_LABELS, LABEL_LIST, LABEL2ID
from layoutlmft.models.relation_classifier import (
    RELATION_LABELS,
    RELATION_NAMES,
    NUM_RELATIONS,
)


@dataclass
class HRDocJointDataCollator:
    """
    文档级别联合训练的 Data Collator

    输入格式（每个样本是一个文档）：
    {
        "document_name": "doc1",
        "chunks": [chunk1, chunk2, ...],  # 每个 chunk 是一页的一部分
        "line_parent_ids": [...],         # 文档级别，全局索引
        "line_relations": [...],          # 文档级别
    }

    输出格式：
    {
        "num_docs": batch_size,
        "chunks_per_doc": [n1, n2, ...],  # 每个文档的 chunk 数量
        "input_ids": [chunk1, chunk2, ...],  # 所有 chunks 展平
        "bbox": [...],
        "attention_mask": [...],
        "labels": [...],
        "line_ids": [...],                # 全局 line_id
        "image": [...],
        "line_parent_ids": [...],         # 每个文档的 parent_ids（全局索引）
        "line_relations": [...],
    }
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    # 关系类型到索引的映射
    relation2id: Dict[str, int] = None

    # 每个文档最多取多少个 chunks（防止显存爆炸）
    # 设为 0 或 None 表示不限制
    max_chunks_per_doc: int = 0

    def __post_init__(self):
        if self.relation2id is None:
            self.relation2id = RELATION_LABELS

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理一个 batch 的文档数据

        Args:
            features: 一个 batch 的文档样本

        Returns:
            batch: 整理后的 batch
        """
        batch_size = len(features)

        # 收集所有 chunks
        all_chunks = []
        chunks_per_doc = []
        all_line_parent_ids = []
        all_line_relations = []
        document_names = []

        for doc in features:
            document_names.append(doc["document_name"])
            chunks = doc["chunks"]
            chunks_per_doc.append(len(chunks))
            all_chunks.extend(chunks)

            # 文档级别的 parent_ids 和 relations（全局索引）
            all_line_parent_ids.append(doc["line_parent_ids"])
            all_line_relations.append(doc["line_relations"])

        # ==================== 1. 处理所有 chunks（用于 Stage 1）====================
        num_chunks = len(all_chunks)

        # 找出最大序列长度
        max_seq_len = max(len(chunk["input_ids"]) for chunk in all_chunks)
        if self.max_length:
            max_seq_len = min(max_seq_len, self.max_length)

        padded_input_ids = []
        padded_bbox = []
        padded_labels = []
        padded_line_ids = []
        attention_masks = []
        all_images = []

        for chunk in all_chunks:
            seq_len = len(chunk["input_ids"])
            padding_len = max_seq_len - seq_len

            # input_ids
            padded_input_ids.append(
                list(chunk["input_ids"]) + [self.tokenizer.pad_token_id] * padding_len
            )

            # bbox
            padded_bbox.append(
                list(chunk["bbox"]) + [[0, 0, 0, 0]] * padding_len
            )

            # labels
            labels = chunk.get("labels", [])
            if labels:
                padded_labels.append(
                    list(labels) + [self.label_pad_token_id] * padding_len
                )

            # line_ids（全局 line_id）
            line_ids = chunk.get("line_ids", [])
            if line_ids:
                padded_line_ids.append(
                    list(line_ids) + [-1] * padding_len
                )
            else:
                padded_line_ids.append([-1] * max_seq_len)

            # attention_mask
            attention_masks.append(
                [1] * seq_len + [0] * padding_len
            )

            # image
            if chunk.get("image") is not None:
                all_images.append(chunk["image"])

        batch = {
            "num_docs": batch_size,
            "chunks_per_doc": chunks_per_doc,
            "document_names": document_names,
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "bbox": torch.tensor(padded_bbox, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "line_ids": torch.tensor(padded_line_ids, dtype=torch.long),
        }

        if padded_labels:
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        if all_images:
            batch["image"] = torch.stack([
                torch.tensor(img) if not isinstance(img, torch.Tensor) else img
                for img in all_images
            ])

        # ==================== 2. 处理文档级别 parent_ids 和 relations ====================
        # 找出最大行数
        max_lines = max(len(pids) for pids in all_line_parent_ids) if all_line_parent_ids else 0

        if max_lines > 0:
            padded_parent_ids = []
            padded_relations = []

            for doc_idx in range(batch_size):
                parent_ids = all_line_parent_ids[doc_idx]
                relations = all_line_relations[doc_idx]
                num_lines = len(parent_ids)
                padding_len = max_lines - num_lines

                # parent_ids（全局索引，不重映射）
                padded_parent_ids.append(
                    list(parent_ids) + [-100] * padding_len
                )

                # relations
                if relations:
                    rel_indices = [
                        self.relation2id.get(str(rel).lower(), -100) for rel in relations
                    ]
                    padded_relations.append(
                        rel_indices + [-100] * padding_len
                    )
                else:
                    padded_relations.append([-100] * max_lines)

            batch["line_parent_ids"] = torch.tensor(padded_parent_ids, dtype=torch.long)
            batch["line_relations"] = torch.tensor(padded_relations, dtype=torch.long)

        return batch


def get_line_semantic_label(token_labels: List[int], line_ids: List[int], target_line_id: int) -> int:
    """
    从 token-level 标签中提取指定行的语义标签

    当前使用 14 类标签（无 BIO 前缀），直接取第一个 token 的标签

    Args:
        token_labels: token-level 标签列表
        line_ids: 每个 token 对应的 line_id
        target_line_id: 目标行 ID

    Returns:
        语义类别索引（0-13）
    """
    for label, line_id in zip(token_labels, line_ids):
        if line_id == target_line_id and label >= 0:
            return label
    return 0  # 默认返回第一个类别


# 使用统一的标签定义
SEMANTIC_CLASSES = LABEL_LIST
SEMANTIC_CLASS2ID = LABEL2ID

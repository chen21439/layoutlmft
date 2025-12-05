#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 联合训练的数据整理器 (Data Collator)
处理三个任务的标签：语义分类、父节点查找、关系分类
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class HRDocJointDataCollator:
    """
    联合训练的 Data Collator

    处理的字段：
    1. input_ids, bbox, image, attention_mask - 模型输入
    2. labels - SubTask1 语义标签（token-level）
    3. line_ids - token到line的映射
    4. line_parent_ids - SubTask2 父节点标签（line-level）
    5. line_relations - SubTask3 关系标签（line-level）
    6. line_semantic_labels - line的语义类别（用于soft-mask）
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    # 关系类型到索引的映射
    relation2id: Dict[str, int] = None

    def __post_init__(self):
        # 默认的关系映射
        if self.relation2id is None:
            self.relation2id = {
                "none": 0,
                "child": 1,
                "sibling": 2,
                "next": 3,
                "other": 4,
            }

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        整理一个batch的数据

        Args:
            features: 一个batch的样本，每个样本是一个字典

        Returns:
            batch: 整理后的batch，所有tensor都padding到相同长度
        """
        # 提取各个字段
        input_ids = [f["input_ids"] for f in features]
        bbox = [f["bbox"] for f in features]
        image = [f["image"] for f in features]
        labels = [f.get("labels", f.get("ner_tags")) for f in features]  # 支持两种命名

        # line相关信息
        line_ids = [f.get("line_ids", []) for f in features]
        line_parent_ids = [f.get("line_parent_ids", []) for f in features]
        line_relations = [f.get("line_relations", []) for f in features]

        batch_size = len(features)

        # ==================== 1. Token-level数据 ====================
        # Padding input_ids, bbox, labels
        max_length = max(len(ids) for ids in input_ids)
        if self.max_length:
            max_length = min(max_length, self.max_length)

        padded_input_ids = []
        padded_bbox = []
        padded_labels = []
        padded_line_ids = []
        attention_mask = []

        for i in range(batch_size):
            seq_len = len(input_ids[i])
            padding_len = max_length - seq_len

            # input_ids
            padded_input_ids.append(
                input_ids[i] + [self.tokenizer.pad_token_id] * padding_len
            )

            # bbox
            padded_bbox.append(
                bbox[i] + [[0, 0, 0, 0]] * padding_len
            )

            # labels
            if labels[i] is not None:
                padded_labels.append(
                    labels[i] + [self.label_pad_token_id] * padding_len
                )

            # line_ids
            if len(line_ids[i]) > 0:
                padded_line_ids.append(
                    line_ids[i] + [-1] * padding_len
                )

            # attention_mask
            attention_mask.append(
                [1] * seq_len + [0] * padding_len
            )

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "bbox": torch.tensor(padded_bbox, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if labels[0] is not None:
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        if len(line_ids[0]) > 0:
            batch["line_ids"] = torch.tensor(padded_line_ids, dtype=torch.long)

        # image
        if image[0] is not None:
            batch["image"] = torch.stack([
                torch.tensor(img) if not isinstance(img, torch.Tensor) else img
                for img in image
            ])

        # ==================== 2. Line-level数据 ====================
        if len(line_parent_ids[0]) > 0:
            # 找出最大line数
            max_lines = max(len(lp) for lp in line_parent_ids)

            # Padding line_parent_ids
            padded_line_parent_ids = []
            padded_line_relations = []
            padded_line_semantic_labels = []

            for i in range(batch_size):
                num_lines = len(line_parent_ids[i])
                padding_len = max_lines - num_lines

                # parent_ids
                padded_line_parent_ids.append(
                    line_parent_ids[i] + [-100] * padding_len  # -100会被忽略
                )

                # relations: 转换为索引
                if len(line_relations[i]) > 0:
                    rel_indices = [
                        self.relation2id.get(rel, 0) for rel in line_relations[i]
                    ]
                    padded_line_relations.append(
                        rel_indices + [0] * padding_len
                    )

                # semantic_labels: 从labels中提取（需要转换BIO -> 语义类别）
                # 这里简化处理，实际需要根据line_ids聚合token的标签
                if "line_semantic_labels" in features[i]:
                    sem_labels = features[i]["line_semantic_labels"]
                    padded_line_semantic_labels.append(
                        sem_labels + [0] * padding_len
                    )

            batch["line_parent_ids"] = torch.tensor(padded_line_parent_ids, dtype=torch.long)

            if len(line_relations[0]) > 0:
                batch["line_relations"] = torch.tensor(padded_line_relations, dtype=torch.long)

            if len(padded_line_semantic_labels) > 0:
                batch["line_semantic_labels"] = torch.tensor(padded_line_semantic_labels, dtype=torch.long)

        return batch


def convert_bio_to_semantic_label(bio_labels: List[int], label_list: List[str]) -> int:
    """
    将BIO标签列表转换为语义类别

    Args:
        bio_labels: BIO标签索引列表
        label_list: 标签名称列表

    Returns:
        semantic_label: 语义类别索引（去掉BIO前缀）

    Example:
        bio_labels = [26, 27]  # B-TITLE, I-TITLE
        label_list = ["O", "B-TITLE", "I-TITLE", ...]
        -> 返回 TITLE 的类别索引
    """
    # 找到第一个非O标签
    for label_idx in bio_labels:
        label_name = label_list[label_idx]
        if label_name != "O":
            # 去掉 B- 或 I- 前缀
            if label_name.startswith("B-") or label_name.startswith("I-"):
                semantic_name = label_name[2:]
            else:
                semantic_name = label_name
            return semantic_name

    return "O"


# 语义类别映射（line-level，去掉BIO前缀）
SEMANTIC_CLASSES = [
    "O",
    "AFFILI",
    "ALG",
    "AUTHOR",
    "EQU",
    "FIG",
    "FIGCAP",
    "FNOTE",
    "FOOT",
    "FSTLINE",
    "MAIL",
    "OPARA",
    "PARA",
    "SEC1",
    "SEC2",
    "SEC3",
    "SEC4",
    "SECX",
    "TAB",
    "TABCAP",
    "TITLE",
]

SEMANTIC_CLASS2ID = {cls: i for i, cls in enumerate(SEMANTIC_CLASSES)}

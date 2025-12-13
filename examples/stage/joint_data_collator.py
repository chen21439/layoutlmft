#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 联合训练的数据整理器 (Data Collator)
处理三个任务的标签：语义分类、父节点查找、关系分类

更新：使用统一的 14 类标签定义（无 BIO 前缀）
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

# 关系类型映射（与 train_multiclass_relation.py 一致）
RELATION_LABELS = {
    "none": 0,
    "connect": 1,
    "contain": 2,
    "equality": 3,
}
RELATION_NAMES = ["none", "connect", "contain", "equality"]


@dataclass
class HRDocJointDataCollator:
    """
    联合训练的 Data Collator

    处理的字段：
    1. input_ids, bbox, image, attention_mask - 模型输入
    2. labels - SubTask1 语义标签（token-level，14 类，无 BIO）
    3. line_ids - token到line的映射
    4. line_parent_ids - SubTask2 父节点标签（line-level）
    5. line_relations - SubTask3 关系标签（line-level）
    6. line_semantic_labels - line的语义类别（用于soft-mask）
    7. line_bboxes - line的边界框坐标（用于几何特征计算）
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    # 关系类型到索引的映射
    relation2id: Dict[str, int] = None

    def __post_init__(self):
        # 使用统一的关系映射
        if self.relation2id is None:
            self.relation2id = RELATION_LABELS

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
        line_bboxes = [f.get("line_bboxes", []) for f in features]

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
                        self.relation2id.get(rel.lower(), 0) for rel in line_relations[i]
                    ]
                    padded_line_relations.append(
                        rel_indices + [0] * padding_len
                    )

                # semantic_labels: 从 features 中直接获取（已经是 14 类索引）
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

        # ==================== 3. Line bboxes ====================
        if len(line_bboxes[0]) > 0:
            # 找出最大line数（与 line_parent_ids 一致）
            max_lines = max(len(lb) for lb in line_bboxes)

            padded_line_bboxes = []
            for i in range(batch_size):
                num_lines = len(line_bboxes[i])
                padding_len = max_lines - num_lines

                # 将每个 bbox 转换为 list（如果是 dict 或其他格式）
                bboxes = []
                for bb in line_bboxes[i]:
                    if isinstance(bb, (list, tuple)):
                        bboxes.append(list(bb))
                    else:
                        bboxes.append([0, 0, 0, 0])

                # padding
                padded_line_bboxes.append(
                    bboxes + [[0, 0, 0, 0]] * padding_len
                )

            batch["line_bboxes"] = torch.tensor(padded_line_bboxes, dtype=torch.float)

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

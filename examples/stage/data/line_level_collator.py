#!/usr/bin/env python
# coding=utf-8
"""
Line-Level Data Collator for Stage 1 Training

用于 Stage 1 line-level 分类训练的数据整理器。
与 HRDocJointDataCollator 保持一致，提供 line_ids 和 line_labels。
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class LineLevelDataCollator:
    """
    Line-Level 训练的 Data Collator

    与 HRDocJointDataCollator 的主要区别：
    - 不需要 line_parent_ids 和 line_relations（Stage 3/4 专用）
    - 需要提供 line_labels（从 token labels 提取）

    每个样本包含：
    - input_ids, bbox, image, attention_mask
    - labels (token-level，用于提取 line_labels)
    - line_ids (每个 token 所属的 line_id)
    - line_labels (每个 line 的分类标签，从 token labels 提取)
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理一个 batch 的数据

        Args:
            features: List of samples, each with:
                - input_ids: token ids
                - bbox: bounding boxes
                - labels: token-level labels
                - line_ids: token → line mapping
                - image: page image (optional)

        Returns:
            Batch dict with:
                - input_ids: [batch, seq_len]
                - bbox: [batch, seq_len, 4]
                - attention_mask: [batch, seq_len]
                - labels: [batch, seq_len] - token-level labels
                - line_ids: [batch, seq_len]
                - line_labels: [batch, max_lines] - line-level labels (extracted)
                - image: [batch, 3, H, W] (if available)
        """
        input_ids = [f["input_ids"] for f in features]
        bbox = [f["bbox"] for f in features]
        image = [f.get("image") for f in features]
        labels = [f.get("labels", f.get("ner_tags")) for f in features]
        line_ids = [f.get("line_ids", []) for f in features]

        batch_size = len(features)

        # ==================== Token-level padding ====================
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

            padded_input_ids.append(
                list(input_ids[i]) + [self.tokenizer.pad_token_id] * padding_len
            )
            padded_bbox.append(
                list(bbox[i]) + [[0, 0, 0, 0]] * padding_len
            )
            if labels[i] is not None:
                padded_labels.append(
                    list(labels[i]) + [self.label_pad_token_id] * padding_len
                )
            if len(line_ids[i]) > 0:
                padded_line_ids.append(
                    list(line_ids[i]) + [-1] * padding_len
                )
            else:
                padded_line_ids.append([-1] * max_length)
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

        has_line_ids = any(len(lid) > 0 for lid in line_ids)
        if has_line_ids:
            batch["line_ids"] = torch.tensor(padded_line_ids, dtype=torch.long)

        # ==================== Image ====================
        if image[0] is not None:
            batch["image"] = torch.stack([
                torch.tensor(img).float() if not isinstance(img, torch.Tensor) else img.float()
                for img in image
            ])

        # ==================== Line-level labels ====================
        # 从 token-level labels 提取 line-level labels
        # 注意：这里我们在 collator 中预先提取，避免在每次 forward 中重复计算
        if has_line_ids and labels[0] is not None:
            # 确定最大 line 数量
            max_lines = 0
            for i in range(batch_size):
                if len(line_ids[i]) > 0:
                    valid_line_ids = [lid for lid in line_ids[i] if lid >= 0]
                    if valid_line_ids:
                        max_lines = max(max_lines, max(valid_line_ids) + 1)

            # 提取每行的标签
            line_labels_list = []
            for i in range(batch_size):
                sample_line_ids = line_ids[i]
                sample_labels = labels[i]
                sample_line_labels = [-100] * max_lines

                # 对每个 line，找到第一个有效的 token 标签
                for line_idx in range(max_lines):
                    for token_idx, (lid, label) in enumerate(zip(sample_line_ids, sample_labels)):
                        if lid == line_idx and label >= 0:
                            sample_line_labels[line_idx] = label
                            break  # 使用第一个有效标签

                line_labels_list.append(sample_line_labels)

            batch["line_labels"] = torch.tensor(line_labels_list, dtype=torch.long)

        return batch


def extract_line_labels(
    token_labels: List[int],
    line_ids: List[int],
) -> Dict[int, int]:
    """
    从 token-level labels 提取 line-level labels

    Args:
        token_labels: Token-level labels
        line_ids: Token → line mapping

    Returns:
        Dict mapping line_id → label
    """
    line_label_map = {}
    for label, line_id in zip(token_labels, line_ids):
        if line_id >= 0 and label >= 0:
            if line_id not in line_label_map:
                line_label_map[line_id] = label
    return line_label_map

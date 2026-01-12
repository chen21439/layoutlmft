#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 联合训练的数据整理器 (Data Collator)

支持两种模式：
- 页面级别（document_level=False）：每个样本是一个 chunk，快速训练
- 文档级别（document_level=True）：每个样本是一个文档，用于推理

=== 训练时 Stage 3/4 处理 ===

    Stage 3 (Parent): 所有行参与
      parent_id=-1 → target=ROOT,  >=0 → target=parent+1

    Stage 4 (Relation): 根据 relation 字段
      "meta" → -100 (loss 忽略)
      其他   → 正常计算 loss

    Meta 类定义见: layoutlmft/data/labels.py

比官方更完整：官方只处理 struct_words=['sec','line','fstline']，
我们处理所有类别（包括 tab/fig/tabcap/figcap 等）。
"""

import os
import sys
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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
    联合训练的 Data Collator（页面级别模式）

    每个样本是一个 chunk，包含：
    - input_ids, bbox, image, attention_mask
    - labels (token-level)
    - line_ids, line_parent_ids, line_relations (line-level，本地索引)
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    relation2id: Dict[str, int] = None

    def __post_init__(self):
        if self.relation2id is None:
            self.relation2id = RELATION_LABELS

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理一个 batch 的 chunk 数据"""
        input_ids = [f["input_ids"] for f in features]
        bbox = [f["bbox"] for f in features]
        image = [f.get("image") for f in features]
        labels = [f.get("labels", f.get("ner_tags")) for f in features]
        doc_ids = [f.get("doc_id", f.get("id", f"unknown_{i}")) for i, f in enumerate(features)]

        line_ids = [f.get("line_ids", []) for f in features]
        line_parent_ids = [f.get("line_parent_ids", []) for f in features]
        line_relations = [f.get("line_relations", []) for f in features]
        line_bboxes = [f.get("line_bboxes", []) for f in features]

        batch_size = len(features)

        # Token-level padding
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

        # Image（确保转换为 float 类型）
        if image[0] is not None:
            batch["image"] = torch.stack([
                torch.tensor(img).float() if not isinstance(img, torch.Tensor) else img.float()
                for img in image
            ])

        # Line-level padding
        has_line_parent_ids = any(len(lp) > 0 for lp in line_parent_ids)
        if has_line_parent_ids:
            max_lines = max(len(lp) for lp in line_parent_ids)

            padded_line_parent_ids = []
            padded_line_relations = []
            padded_line_labels = []

            for i in range(batch_size):
                num_lines = len(line_parent_ids[i])
                padding_len = max_lines - num_lines

                # parent_ids
                padded_line_parent_ids.append(
                    list(line_parent_ids[i]) + [-100] * padding_len
                )

                # relations: 转换为索引
                if len(line_relations[i]) > 0:
                    # Stage 4: 只有 relation!="meta" 的行参与
                    # - "meta" → -100 (loss 忽略)
                    # - "contain/connect/equality" → 0/1/2 (正常计算)
                    # 详见文件头部流程图
                    rel_indices = [
                        self.relation2id.get(rel.lower(), -100) for rel in line_relations[i]
                    ]
                    padded_line_relations.append(
                        rel_indices + [-100] * padding_len  # -100 会被 loss 忽略
                    )
                else:
                    # 空样本用 -100 填充到 max_lines
                    padded_line_relations.append([-100] * max_lines)

                # line_labels: 从 features 中直接获取（已经是 14 类索引），padding 用 -100
                if "line_labels" in features[i]:
                    labels_list = features[i]["line_labels"]
                    padded_line_labels.append(
                        list(labels_list) + [-100] * padding_len
                    )
                else:
                    padded_line_labels.append([-100] * max_lines)

            batch["line_parent_ids"] = torch.tensor(padded_line_parent_ids, dtype=torch.long)

            has_line_relations = any(len(lr) > 0 for lr in line_relations)
            if has_line_relations and len(padded_line_relations) > 0:
                batch["line_relations"] = torch.tensor(padded_line_relations, dtype=torch.long)

            if len(padded_line_labels) > 0:
                batch["line_labels"] = torch.tensor(padded_line_labels, dtype=torch.long)

        # Line bboxes
        has_line_bboxes = any(len(lb) > 0 for lb in line_bboxes)
        if has_line_bboxes:
            max_lines = max(len(lb) for lb in line_bboxes)
            padded_line_bboxes = []
            for i in range(batch_size):
                num_lines = len(line_bboxes[i])
                padding_len = max_lines - num_lines
                bboxes = [list(bb) if isinstance(bb, (list, tuple)) else [0, 0, 0, 0] for bb in line_bboxes[i]]
                padded_line_bboxes.append(bboxes + [[0, 0, 0, 0]] * padding_len)
            batch["line_bboxes"] = torch.tensor(padded_line_bboxes, dtype=torch.float)

        # doc_id 用于调试（不转为 tensor，保持字符串列表）
        batch["doc_id"] = doc_ids

        return batch


@dataclass
class HRDocDocumentLevelCollator:
    """
    文档级别联合训练的 Data Collator

    每个样本是一个文档，包含多个 chunks
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    relation2id: Dict[str, int] = None
    max_chunks_per_doc: int = 0

    def __post_init__(self):
        if self.relation2id is None:
            self.relation2id = RELATION_LABELS

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理一个 batch 的文档数据"""
        batch_size = len(features)

        all_chunks = []
        chunks_per_doc = []
        all_line_parent_ids = []
        all_line_relations = []
        document_names = []
        json_paths = []

        all_line_labels = []  # 从数据源直接获取的行级别标签
        all_line_text_maps = []  # line_id -> text 映射（用于可视化）

        for doc in features:
            document_names.append(doc["document_name"])
            json_paths.append(doc.get("json_path", ""))
            chunks = doc["chunks"]
            chunks_per_doc.append(len(chunks))
            all_chunks.extend(chunks)
            all_line_parent_ids.append(doc["line_parent_ids"])
            all_line_relations.append(doc["line_relations"])
            all_line_labels.append(doc.get("line_labels", []))
            # 从 JSON 字符串解析 text_map（避免 Dataset 序列化问题）
            text_map_json = doc.get("line_text_map_json", "{}")
            text_map = json.loads(text_map_json) if text_map_json else {}
            # 确保 key 是 int 类型
            text_map = {int(k): v for k, v in text_map.items()}
            all_line_text_maps.append(text_map)

        num_chunks = len(all_chunks)
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

            padded_input_ids.append(
                list(chunk["input_ids"]) + [self.tokenizer.pad_token_id] * padding_len
            )
            padded_bbox.append(
                list(chunk["bbox"]) + [[0, 0, 0, 0]] * padding_len
            )

            labels = chunk.get("labels", [])
            if labels:
                padded_labels.append(
                    list(labels) + [self.label_pad_token_id] * padding_len
                )

            line_ids = chunk.get("line_ids", [])
            if line_ids:
                padded_line_ids.append(
                    list(line_ids) + [-1] * padding_len
                )
            else:
                padded_line_ids.append([-1] * max_seq_len)

            attention_masks.append(
                [1] * seq_len + [0] * padding_len
            )

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
            # 保持为 list，不 stack，在 joint_model.py 中按需加载到 GPU
            batch["image"] = all_images

        # 文档级别 parent_ids 和 relations
        max_lines = max(len(pids) for pids in all_line_parent_ids) if all_line_parent_ids else 0

        if max_lines > 0:
            padded_parent_ids = []
            padded_relations = []

            for doc_idx in range(batch_size):
                parent_ids = all_line_parent_ids[doc_idx]
                relations = all_line_relations[doc_idx]
                num_lines = len(parent_ids)
                padding_len = max_lines - num_lines

                padded_parent_ids.append(
                    list(parent_ids) + [-100] * padding_len
                )

                if relations:
                    # relation="meta" → -100 (忽略), 其他 → 正常索引
                    # 详见文件头部流程图
                    # [DEBUG] 打印第一个文档的前几个 relation 值
                    if doc_idx == 0 and len(relations) > 0:
                        import logging
                        logger = logging.getLogger(__name__)
                        doc_name = document_names[doc_idx] if doc_idx < len(document_names) else "unknown"
                        sample_rels = relations[:min(10, len(relations))]
                        # 统计各 relation 类型数量
                        from collections import Counter
                        rel_counts = Counter(relations)
                        logger.info(f"[Collator DEBUG] 文档名: {doc_name}")
                        logger.info(f"[Collator DEBUG] relation 统计: {dict(rel_counts)}")
                        logger.info(f"[Collator DEBUG] relation2id: {self.relation2id}")
                        logger.info(f"[Collator DEBUG] 原始 relations 类型: {type(sample_rels[0])}, 前10个值: {sample_rels}")
                    rel_indices = [
                        self.relation2id.get(rel.lower() if isinstance(rel, str) else rel, -100) for rel in relations
                    ]
                    # [DEBUG] 打印转换后的值
                    if doc_idx == 0:
                        logger.info(f"[Collator DEBUG] 转换后 rel_indices: {rel_indices[:10]}")
                    padded_relations.append(
                        rel_indices + [-100] * padding_len  # -100 会被 loss 忽略
                    )
                else:
                    # 空样本用 -100 填充到 max_lines
                    padded_relations.append([-100] * max_lines)

            batch["line_parent_ids"] = torch.tensor(padded_parent_ids, dtype=torch.long)
            batch["line_relations"] = torch.tensor(padded_relations, dtype=torch.long)

            # 提取 line_labels（用于 section_parent_weight 加权和分类 loss）
            padded_line_labels = []
            chunk_idx = 0
            for doc_idx in range(batch_size):
                num_chunks = chunks_per_doc[doc_idx]
                num_lines = len(all_line_parent_ids[doc_idx])

                # 优先使用数据源提供的 line_labels
                doc_line_labels = all_line_labels[doc_idx] if doc_idx < len(all_line_labels) else []
                if doc_line_labels and len(doc_line_labels) >= num_lines:
                    # 直接使用数据源的 line_labels
                    line_labels = list(doc_line_labels[:num_lines]) + [-100] * (max_lines - num_lines)
                else:
                    # Fallback: 从 token labels 反推
                    doc_labels = []
                    doc_line_ids = []
                    for i in range(chunk_idx, chunk_idx + num_chunks):
                        if i < len(padded_labels):
                            doc_labels.extend(padded_labels[i])
                        if i < len(padded_line_ids):
                            doc_line_ids.extend(padded_line_ids[i])

                    line_labels = [-100] * max_lines
                    for line_idx in range(num_lines):
                        for token_label, token_line_id in zip(doc_labels, doc_line_ids):
                            if token_line_id == line_idx and token_label >= 0:
                                line_labels[line_idx] = token_label
                                break

                chunk_idx += num_chunks
                padded_line_labels.append(line_labels)

            batch["line_labels"] = torch.tensor(padded_line_labels, dtype=torch.long)

        # doc_id 用于调试（不转为 tensor，保持字符串列表）
        batch["doc_id"] = document_names
        batch["json_paths"] = json_paths
        batch["line_text_maps"] = all_line_text_maps  # line_id -> text 映射（用于可视化）

        return batch


def get_line_semantic_label(token_labels: List[int], line_ids: List[int], target_line_id: int) -> int:
    """从 token-level 标签中提取指定行的语义标签"""
    for label, line_id in zip(token_labels, line_ids):
        if line_id == target_line_id and label >= 0:
            return label
    return 0


SEMANTIC_CLASSES = LABEL_LIST
SEMANTIC_CLASS2ID = LABEL2ID

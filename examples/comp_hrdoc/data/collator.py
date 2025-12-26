"""Data Collator

将文档级别的样本组织成 batch。
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class OrderDataCollator:
    """Order 任务的 Data Collator

    输入格式（每个样本是一个文档）：
    {
        "document_name": "doc1",
        "chunks": [chunk1, chunk2, ...],
        "num_lines": 50,
        "sorted_line_ids": [0, 1, 2, ...],
        "line_id_to_order": {0: 0, 1: 1, ...},
    }

    输出格式：
    {
        "num_docs": batch_size,
        "chunks_per_doc": [n1, n2, ...],
        "input_ids": [total_chunks, seq_len],
        "bbox": [total_chunks, seq_len, 4],
        "attention_mask": [total_chunks, seq_len],
        "labels": [total_chunks, seq_len],
        "line_ids": [total_chunks, seq_len],
        "reading_order": [batch_size, max_lines],  # Order 任务所需
        "line_mask": [batch_size, max_lines],
    }
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        # 收集所有 chunks
        all_chunks = []
        chunks_per_doc = []
        document_names = []
        all_num_lines = []
        all_line_id_to_order = []

        for doc in features:
            document_names.append(doc["document_name"])
            chunks = doc["chunks"]
            chunks_per_doc.append(len(chunks))
            all_chunks.extend(chunks)
            all_num_lines.append(doc["num_lines"])
            all_line_id_to_order.append(doc["line_id_to_order"])

        # ==================== 1. 处理所有 chunks ====================
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

            # line_ids
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

        # ==================== 2. 处理 Order 任务所需数据 ====================
        max_lines = max(all_num_lines) if all_num_lines else 0

        if max_lines > 0:
            # reading_order: [batch, max_lines]
            # 在 HRDoc 中，line_id 本身就是阅读顺序
            reading_order = torch.full((batch_size, max_lines), -1, dtype=torch.long)
            line_mask = torch.zeros(batch_size, max_lines, dtype=torch.bool)

            for doc_idx in range(batch_size):
                num_lines = all_num_lines[doc_idx]
                # 阅读顺序就是 0, 1, 2, ..., num_lines-1
                reading_order[doc_idx, :num_lines] = torch.arange(num_lines)
                line_mask[doc_idx, :num_lines] = True

            batch["reading_order"] = reading_order
            batch["line_mask"] = line_mask
            batch["num_lines"] = all_num_lines

        return batch

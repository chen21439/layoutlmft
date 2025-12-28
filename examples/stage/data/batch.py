#!/usr/bin/env python
# coding=utf-8
"""
Batch 抽象层

统一页面级别和文档级别的数据访问接口，下游代码不需要关心具体的数据组织方式。

设计原则：
- 提供统一的 Sample 访问接口
- 隐藏 page-level vs document-level 的差异
- 支持迭代和索引访问
"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator


@dataclass
class Sample:
    """
    单个样本的抽象（可以是一个 chunk 或一个完整文档）

    统一接口，无论来自页面级别还是文档级别
    """
    # 模型输入
    input_ids: torch.Tensor      # [seq_len] or [num_chunks, seq_len]
    bbox: torch.Tensor           # [seq_len, 4] or [num_chunks, seq_len, 4]
    attention_mask: torch.Tensor # [seq_len] or [num_chunks, seq_len]
    image: Optional[torch.Tensor] = None  # [C, H, W] or [num_chunks, C, H, W]

    # 行级别信息
    line_ids: Optional[torch.Tensor] = None  # [seq_len] or [num_chunks, seq_len]

    # Ground Truth（训练/评估时）
    labels: Optional[torch.Tensor] = None
    line_parent_ids: Optional[torch.Tensor] = None  # [num_lines]
    line_relations: Optional[torch.Tensor] = None   # [num_lines]
    line_semantic_labels: Optional[torch.Tensor] = None  # [num_lines]

    # 元信息
    document_name: Optional[str] = None
    json_path: Optional[str] = None  # 原始 JSON 文件路径
    num_chunks: int = 1
    is_document_level: bool = False

    def to(self, device: torch.device) -> 'Sample':
        """移动到指定设备"""
        def move(t):
            return t.to(device) if isinstance(t, torch.Tensor) else t

        return Sample(
            input_ids=move(self.input_ids),
            bbox=move(self.bbox),
            attention_mask=move(self.attention_mask),
            image=move(self.image),
            line_ids=move(self.line_ids),
            labels=move(self.labels),
            line_parent_ids=move(self.line_parent_ids),
            line_relations=move(self.line_relations),
            line_semantic_labels=move(self.line_semantic_labels),
            document_name=self.document_name,
            json_path=self.json_path,
            num_chunks=self.num_chunks,
            is_document_level=self.is_document_level,
        )


class BatchBase(ABC):
    """Batch 抽象基类"""

    @abstractmethod
    def __len__(self) -> int:
        """返回样本数量"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample:
        """获取第 idx 个样本"""
        pass

    def __iter__(self) -> Iterator[Sample]:
        """迭代所有样本"""
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def to(self, device: torch.device) -> 'BatchBase':
        """移动到指定设备"""
        pass

    @property
    @abstractmethod
    def is_document_level(self) -> bool:
        """是否是文档级别"""
        pass


class PageLevelBatch(BatchBase):
    """
    页面级别 Batch

    每个样本是一个独立的 chunk，batch_size = 样本数
    """

    def __init__(self, raw_batch: Dict[str, Any]):
        self.raw = raw_batch
        self._batch_size = raw_batch["input_ids"].shape[0]

    def __len__(self) -> int:
        return self._batch_size

    def __getitem__(self, idx: int) -> Sample:
        if idx >= self._batch_size:
            raise IndexError(f"Index {idx} out of range for batch size {self._batch_size}")

        image = None
        if self.raw.get("image") is not None:
            img = self.raw["image"]
            if isinstance(img, torch.Tensor):
                image = img[idx]
            elif isinstance(img, list) and idx < len(img):
                image = img[idx]
                if not isinstance(image, torch.Tensor):
                    image = torch.tensor(image)

        return Sample(
            input_ids=self.raw["input_ids"][idx],
            bbox=self.raw["bbox"][idx],
            attention_mask=self.raw["attention_mask"][idx],
            image=image,
            line_ids=self.raw.get("line_ids", [None] * self._batch_size)[idx] if "line_ids" in self.raw else None,
            labels=self.raw.get("labels", [None] * self._batch_size)[idx] if "labels" in self.raw else None,
            line_parent_ids=self.raw.get("line_parent_ids", [None] * self._batch_size)[idx] if "line_parent_ids" in self.raw else None,
            line_relations=self.raw.get("line_relations", [None] * self._batch_size)[idx] if "line_relations" in self.raw else None,
            line_semantic_labels=self.raw.get("line_semantic_labels", [None] * self._batch_size)[idx] if "line_semantic_labels" in self.raw else None,
            document_name=self.raw.get("document_names", [None] * self._batch_size)[idx] if "document_names" in self.raw else None,
            num_chunks=1,
            is_document_level=False,
        )

    def to(self, device: torch.device) -> 'PageLevelBatch':
        moved = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.raw.items()
        }
        return PageLevelBatch(moved)

    @property
    def is_document_level(self) -> bool:
        return False

    def get_raw_batch(self) -> Dict[str, Any]:
        """获取原始 batch（用于兼容旧代码）"""
        return self.raw


class DocumentLevelBatch(BatchBase):
    """
    文档级别 Batch

    每个样本是一个完整文档（包含多个 chunks）
    """

    def __init__(self, raw_batch: Dict[str, Any]):
        self.raw = raw_batch
        self._num_docs = raw_batch.get("num_docs", 1)
        self._chunks_per_doc = raw_batch.get("chunks_per_doc", [raw_batch["input_ids"].shape[0]])
        self._document_names = raw_batch.get("document_names", [None] * self._num_docs)
        self._json_paths = raw_batch.get("json_paths", [None] * self._num_docs)

    def __len__(self) -> int:
        return self._num_docs

    def __getitem__(self, idx: int) -> Sample:
        if idx >= self._num_docs:
            raise IndexError(f"Index {idx} out of range for {self._num_docs} documents")

        # 计算该文档对应的 chunk 范围
        chunk_start = sum(self._chunks_per_doc[:idx])
        chunk_end = chunk_start + self._chunks_per_doc[idx]
        num_chunks = self._chunks_per_doc[idx]

        # 提取该文档的所有 chunks
        input_ids = self.raw["input_ids"][chunk_start:chunk_end]
        bbox = self.raw["bbox"][chunk_start:chunk_end]
        attention_mask = self.raw["attention_mask"][chunk_start:chunk_end]

        # 处理 image（必须是 float 类型，cuDNN 要求）
        image = None
        if self.raw.get("image") is not None:
            img = self.raw["image"]
            if isinstance(img, list):
                image = img[chunk_start:chunk_end]
                # 转换为 float tensor（与 JointDataCollator 保持一致）
                image = torch.stack([
                    torch.tensor(im).float() if not isinstance(im, torch.Tensor) else im.float()
                    for im in image
                ])
            elif isinstance(img, torch.Tensor):
                image = img[chunk_start:chunk_end].float()

        # 处理 line_ids
        line_ids = None
        if "line_ids" in self.raw:
            line_ids = self.raw["line_ids"][chunk_start:chunk_end]

        # 处理 labels
        labels = None
        if "labels" in self.raw:
            labels = self.raw["labels"][chunk_start:chunk_end]

        # 文档级别的 parent_ids 和 relations（按文档索引）
        line_parent_ids = None
        if "line_parent_ids" in self.raw:
            line_parent_ids = self.raw["line_parent_ids"][idx]

        line_relations = None
        if "line_relations" in self.raw:
            line_relations = self.raw["line_relations"][idx]

        line_semantic_labels = None
        if "line_semantic_labels" in self.raw:
            line_semantic_labels = self.raw["line_semantic_labels"][idx]

        return Sample(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            line_ids=line_ids,
            labels=labels,
            line_parent_ids=line_parent_ids,
            line_relations=line_relations,
            line_semantic_labels=line_semantic_labels,
            document_name=self._document_names[idx] if idx < len(self._document_names) else None,
            json_path=self._json_paths[idx] if idx < len(self._json_paths) else None,
            num_chunks=num_chunks,
            is_document_level=True,
        )

    def to(self, device: torch.device) -> 'DocumentLevelBatch':
        moved = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.raw.items()
        }
        return DocumentLevelBatch(moved)

    @property
    def is_document_level(self) -> bool:
        return True

    def get_raw_batch(self) -> Dict[str, Any]:
        """获取原始 batch（用于兼容旧代码）"""
        return self.raw


def wrap_batch(raw_batch: Dict[str, Any]) -> BatchBase:
    """
    自动检测并包装 batch

    根据 batch 中是否有 num_docs 字段判断是文档级别还是页面级别
    """
    if "num_docs" in raw_batch:
        return DocumentLevelBatch(raw_batch)
    else:
        return PageLevelBatch(raw_batch)

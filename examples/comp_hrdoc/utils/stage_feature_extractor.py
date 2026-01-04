#!/usr/bin/env python
# coding=utf-8
"""
Stage 模型特征提取器

使用 stage 目录训练好的 JointModel 提取 line-level 特征，
供 comp_hrdoc 的 Construct 模块使用。

功能：
1. 加载训练好的 stage JointModel
2. 使用 backbone + line_pooling 提取 line-level 特征
3. 返回 line_features [num_docs, max_lines, hidden_size]

使用方式：
    from examples.comp_hrdoc.utils.stage_feature_extractor import StageFeatureExtractor

    extractor = StageFeatureExtractor(checkpoint_path="/path/to/joint/checkpoint")
    line_features, line_mask = extractor.extract_features(batch)
"""

import os
import sys
import logging
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 添加 stage 目录到 path
_STAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "stage"))
if _STAGE_ROOT not in sys.path:
    sys.path.insert(0, _STAGE_ROOT)


class StageFeatureExtractor:
    """
    使用 stage 训练好的模型提取 line-level 特征

    只使用 JointModel 的 backbone + line_pooling，不需要 stage3/4。
    输出特征可直接用于 comp_hrdoc 的 Construct 模块。

    Example:
        >>> extractor = StageFeatureExtractor("/path/to/checkpoint")
        >>> features, mask = extractor.extract_features(batch)
        >>> # features: [num_docs, max_lines, 768]
        >>> # mask: [num_docs, max_lines]
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        config: Any = None,
    ):
        """
        Args:
            checkpoint_path: stage JointModel checkpoint 路径
            device: 计算设备 ("cuda" / "cpu")
            config: 配置对象（可选，用于 tokenizer fallback）
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 加载模型
        self.model, self.tokenizer = self._load_model(checkpoint_path, config)
        self.model.eval()

        logger.info(f"StageFeatureExtractor initialized")
        logger.info(f"  checkpoint: {checkpoint_path}")
        logger.info(f"  device: {self.device}")
        logger.info(f"  hidden_size: 768")

    def _load_model(self, checkpoint_path: str, config: Any = None):
        """加载 stage JointModel"""
        from models.build import load_joint_model

        model, tokenizer = load_joint_model(
            model_path=checkpoint_path,
            device=self.device,
            config=config,
        )
        return model, tokenizer

    @torch.no_grad()
    def extract_features(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ids: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        num_docs: Optional[int] = None,
        chunks_per_doc: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取 line-level 特征

        Args:
            input_ids: [total_chunks, seq_len] tokenized input
            bbox: [total_chunks, seq_len, 4] bounding boxes
            attention_mask: [total_chunks, seq_len]
            line_ids: [total_chunks, seq_len] 每个 token 的 line_id
            image: [total_chunks, C, H, W] 可选图像输入
            num_docs: 文档数量（文档级别模式）
            chunks_per_doc: 每个文档的 chunk 数量列表

        Returns:
            line_features: [num_docs, max_lines, hidden_size]
            line_mask: [num_docs, max_lines] bool mask
        """
        # 移动到设备
        input_ids = input_ids.to(self.device)
        bbox = bbox.to(self.device)
        attention_mask = attention_mask.to(self.device)
        line_ids = line_ids.to(self.device)
        if image is not None:
            image = image.to(self.device)

        # Step 1: 获取 backbone hidden states
        hidden_states = self.model.encode_with_micro_batch(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            no_grad=True,
        )

        # 截取文本部分（排除视觉 tokens）
        seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :seq_len, :]  # [total_chunks, seq_len, H]

        # Step 2: Line Pooling 聚合
        total_chunks = input_ids.shape[0]
        is_page_level = (num_docs is None or chunks_per_doc is None)

        if is_page_level:
            # 页面级别：每个 chunk 是一个样本
            return self._aggregate_page_level(text_hidden, line_ids)
        else:
            # 文档级别：多个 chunks 聚合为一个文档
            return self._aggregate_document_level(
                text_hidden, line_ids, num_docs, chunks_per_doc
            )

    def _aggregate_page_level(
        self,
        text_hidden: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """页面级别聚合：每个 chunk 独立处理"""
        batch_size = text_hidden.shape[0]
        device = text_hidden.device

        doc_line_features_list = []
        doc_line_masks_list = []

        for b in range(batch_size):
            sample_hidden = text_hidden[b:b+1]  # [1, seq_len, H]
            sample_line_ids = line_ids[b:b+1]   # [1, seq_len]
            features, mask = self.model.line_pooling(sample_hidden, sample_line_ids)
            doc_line_features_list.append(features)
            doc_line_masks_list.append(mask)

        # 填充到相同长度
        return self._pad_features(doc_line_features_list, doc_line_masks_list, device)

    def _aggregate_document_level(
        self,
        text_hidden: torch.Tensor,
        line_ids: torch.Tensor,
        num_docs: int,
        chunks_per_doc: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """文档级别聚合：多个 chunks 聚合为一个文档"""
        device = text_hidden.device

        doc_line_features_list = []
        doc_line_masks_list = []

        chunk_idx = 0
        for doc_idx in range(num_docs):
            num_chunks = chunks_per_doc[doc_idx]

            # 收集该文档所有 chunks
            doc_hidden = text_hidden[chunk_idx:chunk_idx + num_chunks]
            doc_line_ids = line_ids[chunk_idx:chunk_idx + num_chunks]

            # 使用 line_pooling 聚合
            features, mask = self.model.line_pooling(doc_hidden, doc_line_ids)
            doc_line_features_list.append(features)
            doc_line_masks_list.append(mask)

            chunk_idx += num_chunks

        return self._pad_features(doc_line_features_list, doc_line_masks_list, device)

    def _pad_features(
        self,
        features_list: list,
        masks_list: list,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将不同长度的特征填充到相同长度"""
        num_docs = len(features_list)
        max_lines = max(f.shape[0] for f in features_list)
        hidden_dim = features_list[0].shape[1]

        line_features = torch.zeros(num_docs, max_lines, hidden_dim, device=device)
        line_mask = torch.zeros(num_docs, max_lines, dtype=torch.bool, device=device)

        for b, (features, mask) in enumerate(zip(features_list, masks_list)):
            num_lines = features.shape[0]
            line_features[b, :num_lines] = features
            line_mask[b, :num_lines] = mask

        return line_features, line_mask

    def extract_from_batch(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 batch dict 提取特征（便捷方法）

        Args:
            batch: 包含 input_ids, bbox, attention_mask, line_ids 等的字典

        Returns:
            line_features: [num_docs, max_lines, hidden_size]
            line_mask: [num_docs, max_lines]
        """
        return self.extract_features(
            input_ids=batch["input_ids"],
            bbox=batch["bbox"],
            attention_mask=batch["attention_mask"],
            line_ids=batch.get("line_ids"),
            image=batch.get("image"),
            num_docs=batch.get("num_docs"),
            chunks_per_doc=batch.get("chunks_per_doc"),
        )

    @property
    def hidden_size(self) -> int:
        """返回特征维度"""
        return 768

#!/usr/bin/env python
# coding=utf-8
"""
Stage1 特征提取器

使用 Stage1Backbone 提取 line-level 特征，供 comp_hrdoc 的 Construct 模块使用。

功能：
1. 加载 Stage1 checkpoint（backbone + line_pooling + line_enhancer）
2. 提取 line-level 特征
3. 返回 line_features [num_docs, max_lines, hidden_size]

使用方式：
    from examples.comp_hrdoc.utils.stage_feature_extractor import StageFeatureExtractor

    extractor = StageFeatureExtractor(checkpoint_path="/path/to/checkpoint")
    line_features, line_mask = extractor.extract_features(batch)

注意：此模块不依赖 stage 目录，使用 comp_hrdoc 内部的 Stage1Backbone。
"""

import os
import logging
from typing import Dict, Tuple, Optional, Any

import torch

logger = logging.getLogger(__name__)


class StageFeatureExtractor:
    """
    使用 Stage1Backbone 提取 line-level 特征

    封装 backbone + line_pooling + line_enhancer，不依赖 stage 目录。
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
        max_regions: int = 4096,
    ):
        """
        Args:
            checkpoint_path: Stage1 checkpoint 路径
            device: 计算设备 ("cuda" / "cpu")
            config: 配置对象（可选）
            max_regions: 最大区域数（用于 padding，后续会合并 line 为 region）
        """
        self.checkpoint_path = checkpoint_path
        self.max_regions = max_regions
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 加载模型（使用 comp_hrdoc 内部的 Stage1Backbone）
        self.model, self.tokenizer = self._load_model(checkpoint_path, config)
        self.model.eval()

        logger.info(f"StageFeatureExtractor initialized")
        logger.info(f"  checkpoint: {checkpoint_path}")
        logger.info(f"  device: {self.device}")
        logger.info(f"  hidden_size: 768")

    def _load_model(self, checkpoint_path: str, config: Any = None):
        """加载 Stage1Backbone

        支持两种路径格式：
        1. 联合训练 checkpoint（包含 stage1/ 子目录）
        2. 标准 LayoutXLM checkpoint
        """
        from examples.comp_hrdoc.models.detect import build_stage1_backbone

        model, tokenizer = build_stage1_backbone(
            checkpoint_path=checkpoint_path,
            device=str(self.device),
            use_line_enhancer=True,
            use_cls_head=True,
        )

        return model, tokenizer

    def _extract_features_impl(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ids: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        num_docs: Optional[int] = None,
        chunks_per_doc: Optional[list] = None,
        no_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取 line-level 特征的内部实现

        Args:
            input_ids: [total_chunks, seq_len] tokenized input
            bbox: [total_chunks, seq_len, 4] bounding boxes
            attention_mask: [total_chunks, seq_len]
            line_ids: [total_chunks, seq_len] 每个 token 的 line_id
            image: [total_chunks, C, H, W] 可选图像输入
            num_docs: 文档数量（文档级别模式）
            chunks_per_doc: 每个文档的 chunk 数量列表
            no_grad: 是否禁用梯度（True=推理模式，False=训练模式）

        Returns:
            line_features: [num_docs, max_lines, hidden_size]
            line_mask: [num_docs, max_lines] bool mask
        """
        # 移动到设备
        input_ids = input_ids.to(self.device)
        bbox = bbox.to(self.device)
        attention_mask = attention_mask.to(self.device)
        line_ids = line_ids.to(self.device)
        # image 可能是 list（来自 DocumentLevelCollator），由模型内部处理
        if image is not None and isinstance(image, torch.Tensor):
            image = image.to(self.device)

        # Step 1: 获取 backbone hidden states
        hidden_states = self.model.encode_with_micro_batch(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            no_grad=no_grad,  # 关键：控制梯度
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
        提取 line-level 特征（无梯度，推理/评估用）

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
        return self._extract_features_impl(
            input_ids, bbox, attention_mask, line_ids,
            image, num_docs, chunks_per_doc, no_grad=True
        )

    def extract_features_with_grad(
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
        提取 line-level 特征（允许梯度回传，用于 train_stage1 模式）

        与 extract_features() 参数相同，但梯度从 line_features 回传到 backbone。
        使用前需要调用 set_train_mode() 设置 backbone 为训练模式。

        Returns:
            line_features: [num_docs, max_lines, hidden_size]，梯度流向 backbone
            line_mask: [num_docs, max_lines] bool mask
        """
        return self._extract_features_impl(
            input_ids, bbox, attention_mask, line_ids,
            image, num_docs, chunks_per_doc, no_grad=False
        )

    def set_train_mode(self, freeze_visual: bool = True):
        """
        设置为训练模式（用于 train_stage1）

        Args:
            freeze_visual: 是否冻结视觉编码器
        """
        self.model.train()
        if freeze_visual:
            self.model.freeze_visual()

    def set_eval_mode(self):
        """设置为评估模式"""
        self.model.eval()

    def get_trainable_params(self) -> list:
        """
        获取可训练参数（用于 train_stage1 的优化器）

        Returns:
            可训练参数列表
        """
        return [p for p in self.model.parameters() if p.requires_grad]

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
        """将不同长度的特征填充到固定长度 (self.max_regions)"""
        num_docs = len(features_list)
        max_regions = self.max_regions  # 使用固定长度
        hidden_dim = features_list[0].shape[1]

        line_features = torch.zeros(num_docs, max_regions, hidden_dim, device=device)
        line_mask = torch.zeros(num_docs, max_regions, dtype=torch.bool, device=device)

        for b, (features, mask) in enumerate(zip(features_list, masks_list)):
            num_lines = min(features.shape[0], max_regions)  # 截断超长的
            line_features[b, :num_lines] = features[:num_lines]
            line_mask[b, :num_lines] = mask[:num_lines]

        # Step 3: 行间特征增强（论文 4.2.2）
        # 在 line_pooling 之后应用 Transformer 增强
        if self.model.line_enhancer is not None:
            line_features = self.model.line_enhancer(line_features, line_mask)

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

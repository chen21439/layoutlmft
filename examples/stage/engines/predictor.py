#!/usr/bin/env python
# coding=utf-8
"""
Predictor - 统一推理接口

支持页面级别和文档级别的推理，使用 Batch 抽象层隐藏差异。

设计原则：
- 接收 Sample，返回 PredictionOutput
- 不关心 Sample 来自页面级别还是文档级别
- 内部处理多 chunk 聚合
- 使用 tasks/ 中的 decode 逻辑，确保训练和评估一致
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.batch import Sample, BatchBase, wrap_batch
from tasks import SemanticClassificationTask


@dataclass
class PredictionOutput:
    """单个样本的预测输出"""
    # Stage 1: 分类
    line_classes: Dict[int, int] = field(default_factory=dict)  # {line_id: class_id}

    # Stage 3: 父节点
    line_parents: List[int] = field(default_factory=list)  # [parent_id for each line]

    # Stage 4: 关系
    line_relations: List[int] = field(default_factory=list)  # [relation_id for each line]

    # 元信息
    num_lines: int = 0
    line_ids: List[int] = field(default_factory=list)

    # 置信度（可选）
    line_class_probs: Optional[Dict[int, List[float]]] = None


class Predictor:
    """
    统一推理器

    使用方式：
        predictor = Predictor(model, device)
        for sample in batch:
            output = predictor.predict(sample)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        micro_batch_size: int = 1,
    ):
        """
        Args:
            model: JointModel 或类似结构（需要有 stage1, stage3, stage4, feature_extractor）
            device: 计算设备
            micro_batch_size: Stage 1 推理时的 micro-batch 大小（默认 1，与训练一致）
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.micro_batch_size = micro_batch_size

        # 使用 tasks/ 中的统一 decode 逻辑
        self.cls_task = SemanticClassificationTask(model=model, use_line_level=True)

    def predict(self, sample: Sample) -> PredictionOutput:
        """
        对单个样本进行推理

        Args:
            sample: Sample 对象（可能包含多个 chunks）

        Returns:
            PredictionOutput: 预测结果
        """
        sample = sample.to(self.device)

        if sample.is_document_level:
            return self._predict_document(sample)
        else:
            return self._predict_page(sample)

    def _predict_page(self, sample: Sample) -> PredictionOutput:
        """页面级别推理（单 chunk）"""
        # 添加 batch 维度
        input_ids = sample.input_ids.unsqueeze(0)
        bbox = sample.bbox.unsqueeze(0)
        attention_mask = sample.attention_mask.unsqueeze(0)
        image = sample.image.unsqueeze(0) if sample.image is not None else None
        line_ids = sample.line_ids.unsqueeze(0) if sample.line_ids is not None else None

        return self._run_inference(input_ids, bbox, attention_mask, image, line_ids)

    def _predict_document(self, sample: Sample) -> PredictionOutput:
        """文档级别推理（多 chunk 聚合）"""
        # 文档级别：input_ids 已经是 [num_chunks, seq_len]
        input_ids = sample.input_ids
        bbox = sample.bbox
        attention_mask = sample.attention_mask
        image = sample.image
        line_ids = sample.line_ids

        return self._run_inference(input_ids, bbox, attention_mask, image, line_ids)

    def _run_inference(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: Optional[torch.Tensor],
        line_ids: Optional[torch.Tensor],
    ) -> PredictionOutput:
        """
        核心推理逻辑

        Args:
            input_ids: [num_chunks, seq_len]
            bbox: [num_chunks, seq_len, 4]
            attention_mask: [num_chunks, seq_len]
            image: [num_chunks, C, H, W] or None
            line_ids: [num_chunks, seq_len] or None

        Returns:
            PredictionOutput
        """
        if line_ids is None:
            return PredictionOutput()

        num_chunks = input_ids.shape[0]

        # ==================== Stage 1: Classification ====================
        # 使用 encode_with_micro_batch 复用 micro-batching 逻辑（与训练一致）
        # 推理时强制 no_grad=True 节省显存
        hidden_states = self.model.encode_with_micro_batch(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            micro_batch_size=self.micro_batch_size,
            no_grad=True,
        )

        # 截取文本部分的 hidden states（排除视觉 tokens）
        seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :seq_len, :]  # [num_chunks, seq_len, H]

        # 使用 SemanticClassificationTask 进行分类（与训练时一致）
        # 内部使用 model.line_pooling + model.cls_head（line-level 模式）
        line_classes = self.cls_task.decode(
            hidden_states=text_hidden,
            line_ids=line_ids,
        )

        if not line_classes:
            return PredictionOutput()

        # ==================== Stage 2: Feature Extraction ====================
        # 使用 model.line_pooling 聚合（与训练一致，不再区分页面/文档级别）
        # line_pooling 内部自动处理多 chunk 聚合
        line_features, line_mask = self.model.line_pooling(text_hidden, line_ids)
        # line_features: [num_lines, H], line_mask: [num_lines]

        actual_num_lines = int(line_mask.sum().item())

        if actual_num_lines == 0:
            return PredictionOutput(line_classes=line_classes)

        # ==================== Stage 3: Parent Finding ====================
        pred_parents = [-1] * actual_num_lines
        gru_hidden = None

        use_gru = getattr(self.model, 'use_gru', False)

        # 获取 cls_logits（用于 soft-mask）
        # 如果模型有 cls_head（line-level 分类），使用它来获取 cls_logits
        cls_logits = None
        if hasattr(self.model, 'cls_head') and self.model.cls_head is not None:
            valid_features = line_features[:actual_num_lines]  # [L, H]
            cls_logits = self.model.cls_head(valid_features)  # [L, num_classes]
            cls_logits = cls_logits.unsqueeze(0)  # [1, L, num_classes]

        if use_gru:
            parent_logits, gru_hidden = self.model.stage3(
                line_features.unsqueeze(0),
                line_mask.unsqueeze(0),
                return_gru_hidden=True,
                cls_logits=cls_logits  # 传入分类 logits 用于 soft-mask
            )
            gru_hidden = gru_hidden[0]  # [L+1, gru_hidden_size]

            for child_idx in range(actual_num_lines):
                child_logits = parent_logits[0, child_idx + 1, :child_idx + 2]
                pred_parent_idx = child_logits.argmax().item()
                pred_parents[child_idx] = pred_parent_idx - 1  # -1 means ROOT
        else:
            for child_idx in range(1, actual_num_lines):
                parent_candidates = line_features[:child_idx]
                child_feat = line_features[child_idx]
                scores = self.model.stage3(parent_candidates, child_feat)
                pred_parents[child_idx] = scores.argmax().item()

        # ==================== Stage 4: Relation Classification ====================
        pred_relations = [0] * actual_num_lines  # Default: connect (0)

        for child_idx in range(actual_num_lines):
            parent_idx = pred_parents[child_idx]
            if parent_idx < 0 or parent_idx >= actual_num_lines:
                continue

            if gru_hidden is not None:
                parent_gru_idx = parent_idx + 1
                child_gru_idx = child_idx + 1
                parent_feat = gru_hidden[parent_gru_idx]
                child_feat = gru_hidden[child_gru_idx]
            else:
                parent_feat = line_features[parent_idx]
                child_feat = line_features[child_idx]

            rel_logits = self.model.stage4(
                parent_feat.unsqueeze(0),
                child_feat.unsqueeze(0),
            )
            pred_relations[child_idx] = rel_logits.argmax(dim=1).item()

        # 构建输出
        sorted_line_ids = sorted(line_classes.keys())

        return PredictionOutput(
            line_classes=line_classes,
            line_parents=pred_parents,
            line_relations=pred_relations,
            num_lines=actual_num_lines,
            line_ids=sorted_line_ids,
        )

    def predict_batch(self, batch: BatchBase) -> List[PredictionOutput]:
        """
        对整个 batch 进行推理

        Args:
            batch: BatchBase 对象

        Returns:
            List[PredictionOutput]: 每个样本的预测结果
        """
        results = []
        for sample in batch:
            with torch.no_grad():
                result = self.predict(sample)
            results.append(result)
        return results

    def predict_single_from_batch(
        self,
        batch: Dict[str, Any],
        batch_idx: int = 0,
    ) -> PredictionOutput:
        """
        从页面级别 batch 中提取单个样本进行推理

        Args:
            batch: 页面级别 batch，input_ids 形状为 [batch_size, seq_len]
            batch_idx: 要推理的样本索引

        Returns:
            PredictionOutput: 预测结果
        """
        # 提取单个样本
        input_ids = batch["input_ids"][batch_idx:batch_idx+1]
        bbox = batch["bbox"][batch_idx:batch_idx+1]
        attention_mask = batch["attention_mask"][batch_idx:batch_idx+1]

        line_ids = batch.get("line_ids")
        if line_ids is not None:
            line_ids = line_ids[batch_idx:batch_idx+1]

        # 处理 image
        image = batch.get("image")
        if image is not None:
            if isinstance(image, list):
                image = [image[batch_idx]]
            else:
                image = image[batch_idx:batch_idx+1]

        # 移动到设备
        input_ids = input_ids.to(self.device)
        bbox = bbox.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if line_ids is not None:
            line_ids = line_ids.to(self.device)
        if image is not None and isinstance(image, torch.Tensor):
            image = image.to(self.device)

        return self._run_inference(input_ids, bbox, attention_mask, image, line_ids)

    def predict_from_dict(
        self,
        batch: Dict[str, Any],
        doc_idx: int = 0,
    ) -> PredictionOutput:
        """
        从文档级别 batch dict 进行推理

        Args:
            batch: 包含 input_ids, bbox, attention_mask, image, line_ids 等的字典
                - 文档级别：input_ids 形状为 [total_chunks, seq_len]
                - 包含 num_docs 和 chunks_per_doc 字段
            doc_idx: 文档索引

        Returns:
            PredictionOutput: 预测结果
        """
        # 获取文档范围
        num_docs = batch.get("num_docs", 1)
        chunks_per_doc = batch.get("chunks_per_doc", [batch["input_ids"].shape[0]])

        chunk_start = sum(chunks_per_doc[:doc_idx])
        chunk_end = chunk_start + chunks_per_doc[doc_idx]

        # 提取该文档的数据
        input_ids = batch["input_ids"][chunk_start:chunk_end]
        bbox = batch["bbox"][chunk_start:chunk_end]
        attention_mask = batch["attention_mask"][chunk_start:chunk_end]
        line_ids = batch.get("line_ids")
        if line_ids is not None:
            line_ids = line_ids[chunk_start:chunk_end]

        # 处理 image
        image = batch.get("image")
        if image is not None:
            if isinstance(image, list):
                image = image[chunk_start:chunk_end]
            else:
                image = image[chunk_start:chunk_end]

        # 移动到设备
        input_ids = input_ids.to(self.device)
        bbox = bbox.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if line_ids is not None:
            line_ids = line_ids.to(self.device)
        if image is not None and isinstance(image, torch.Tensor):
            image = image.to(self.device)

        return self._run_inference(input_ids, bbox, attention_mask, image, line_ids)

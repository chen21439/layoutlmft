#!/usr/bin/env python
# coding=utf-8
"""
Predictor - 统一推理接口

支持页面级别和文档级别的推理，使用 Batch 抽象层隐藏差异。

设计原则：
- 接收 Sample，返回 PredictionOutput
- 不关心 Sample 来自页面级别还是文档级别
- 内部处理多 chunk 聚合
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.batch import Sample, BatchBase, wrap_batch


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

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Args:
            model: JointModel 或类似结构（需要有 stage1, stage3, stage4, feature_extractor）
            device: 计算设备
        """
        self.model = model
        self.device = device or next(model.parameters()).device

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
        stage1_outputs = self.model.stage1(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            output_hidden_states=True,
        )

        logits = stage1_outputs.logits  # [num_chunks, seq_len, num_classes]
        hidden_states = stage1_outputs.hidden_states[-1]  # [num_chunks, seq_len, hidden_size]

        # 聚合所有 chunks 的 token predictions
        all_token_preds = []
        all_line_ids = []

        for c in range(num_chunks):
            chunk_logits = logits[c]  # [seq_len, num_classes]
            chunk_line_ids = line_ids[c].cpu().tolist()

            for i in range(chunk_logits.shape[0]):
                all_token_preds.append(chunk_logits[i].argmax().item())
                all_line_ids.append(chunk_line_ids[i])

        # Token -> Line 聚合（多数投票）
        line_classes = self._aggregate_to_lines(all_token_preds, all_line_ids)

        if not line_classes:
            return PredictionOutput()

        # ==================== Stage 2: Feature Extraction ====================
        # 注意：hidden_states 包含视觉 tokens (7x7=49)，需要截取文本部分
        # 与 JointModel.forward() 中的处理保持一致
        seq_len = input_ids.shape[1]  # 文本序列长度 (512)
        text_hidden = hidden_states[:, :seq_len, :]  # [num_chunks, seq_len, H]

        # 根据模式选择特征聚合方法（与训练保持一致）
        if num_chunks > 1:
            # 文档级别：使用 _aggregate_document_line_features（与 JointModel 一致）
            line_features, line_mask = self._aggregate_document_line_features(
                text_hidden, line_ids
            )
            # line_features: [num_lines, H], line_mask: [num_lines]
        else:
            # 页面级别：使用 extract_line_features
            all_hidden = text_hidden.reshape(-1, text_hidden.shape[-1])
            all_line_ids_flat = line_ids.reshape(-1)
            line_features, line_mask = self.model.feature_extractor.extract_line_features(
                all_hidden.unsqueeze(0),
                all_line_ids_flat.unsqueeze(0),
                pooling="mean"
            )
            line_features = line_features[0]  # [max_lines, H]
            line_mask = line_mask[0]

        actual_num_lines = int(line_mask.sum().item())

        if actual_num_lines == 0:
            return PredictionOutput(line_classes=line_classes)

        # ==================== Stage 3: Parent Finding ====================
        pred_parents = [-1] * actual_num_lines
        gru_hidden = None

        use_gru = getattr(self.model, 'use_gru', False)

        if use_gru:
            parent_logits, gru_hidden = self.model.stage3(
                line_features.unsqueeze(0),
                line_mask.unsqueeze(0),
                return_gru_hidden=True
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

    def _aggregate_to_lines(
        self,
        token_preds: List[int],
        line_ids: List[int],
        method: str = "majority"
    ) -> Dict[int, int]:
        """Token-level 预测聚合到 Line-level"""
        line_votes = {}

        for pred, line_id in zip(token_preds, line_ids):
            if line_id < 0:
                continue
            if line_id not in line_votes:
                line_votes[line_id] = []
            line_votes[line_id].append(pred)

        line_classes = {}
        for line_id, votes in line_votes.items():
            if method == "majority":
                counter = Counter(votes)
                line_classes[line_id] = counter.most_common(1)[0][0]
            elif method == "first":
                line_classes[line_id] = votes[0]
            else:
                line_classes[line_id] = votes[0]

        return line_classes

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

    def _aggregate_document_line_features(
        self,
        doc_hidden: torch.Tensor,
        doc_line_ids: torch.Tensor,
    ) -> tuple:
        """
        从文档的所有 chunks 中聚合 line features（与 JointModel 一致）

        Args:
            doc_hidden: [num_chunks, seq_len, hidden_dim]
            doc_line_ids: [num_chunks, seq_len]，每个 token 的全局 line_id

        Returns:
            features: [num_lines, hidden_dim]
            mask: [num_lines]，有效行的 mask
        """
        device = doc_hidden.device
        hidden_dim = doc_hidden.shape[-1]

        # 展平（使用 reshape 兼容非连续 tensor）
        flat_hidden = doc_hidden.reshape(-1, hidden_dim)  # [N, hidden_dim]
        flat_line_ids = doc_line_ids.reshape(-1)  # [N]

        # 获取有效 token（line_id >= 0）
        valid_mask = flat_line_ids >= 0
        valid_line_ids = flat_line_ids[valid_mask]
        valid_hidden = flat_hidden[valid_mask]

        if len(valid_line_ids) == 0:
            return torch.zeros(1, hidden_dim, device=device), torch.zeros(1, dtype=torch.bool, device=device)

        # 获取唯一的 line_id 并排序
        unique_line_ids = valid_line_ids.unique()
        unique_line_ids = unique_line_ids.sort()[0]
        num_lines = len(unique_line_ids)

        # 创建 line_id 到连续索引的映射（向量化）
        # 使用 searchsorted 进行快速映射
        line_indices = torch.searchsorted(unique_line_ids, valid_line_ids)

        # 使用 scatter_add 聚合 features
        line_features = torch.zeros(num_lines, hidden_dim, device=device)
        line_features.scatter_add_(0, line_indices.unsqueeze(1).expand(-1, hidden_dim), valid_hidden)

        # 统计每个 line 的 token 数量
        line_counts = torch.zeros(num_lines, device=device)
        line_counts.scatter_add_(0, line_indices, torch.ones_like(line_indices, dtype=torch.float))

        # 计算平均值
        valid_counts = line_counts.clamp(min=1)
        line_features = line_features / valid_counts.unsqueeze(1)

        # 创建 mask
        line_mask = line_counts > 0

        return line_features, line_mask

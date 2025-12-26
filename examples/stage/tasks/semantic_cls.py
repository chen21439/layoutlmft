#!/usr/bin/env python
# coding=utf-8
"""
Semantic Classification Task (Line-level 分类)

包含：
1. SemanticClassificationTask - 封装分类的 loss/decode/metrics 逻辑

设计原则：
- 统一 decode 逻辑，避免训练和评估时使用不同的预测方式
- 支持两种模式：
  - Line-level: mean pooling + cls_head (推荐，与训练一致)
  - Token-level: token 预测 + 投票 (旧方式，兼容)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class SemanticClassificationTask:
    """
    语义分类任务封装类

    封装分类的核心逻辑：
    - decode: 从模型输出获取 line-level 预测
    - compute_loss: 计算分类损失
    - compute_metrics: 计算分类指标

    使用方式（推理时）：
        task = SemanticClassificationTask(model)
        line_classes = task.decode(
            hidden_states=hidden_states,
            line_ids=line_ids,
        )

    使用方式（评估时）：
        metrics = task.compute_metrics(predictions, labels)
    """

    def __init__(
        self,
        model: nn.Module = None,
        num_classes: int = 14,
        use_line_level: bool = True,
    ):
        """
        Args:
            model: JointModel 或类似结构，需要有 cls_head 和 line_pooling
            num_classes: 分类类别数
            use_line_level: 是否使用 line-level 分类（推荐 True）
        """
        self.model = model
        self.num_classes = num_classes
        self.use_line_level = use_line_level

        # 检查模型是否支持 line-level 分类
        if model is not None:
            self._has_cls_head = hasattr(model, 'cls_head') and model.cls_head is not None
            self._has_line_pooling = hasattr(model, 'line_pooling') and model.line_pooling is not None
        else:
            self._has_cls_head = False
            self._has_line_pooling = False

    def decode(
        self,
        hidden_states: torch.Tensor,
        line_ids: torch.Tensor,
        token_logits: Optional[torch.Tensor] = None,
    ) -> Dict[int, int]:
        """
        从模型输出获取 line-level 预测

        Args:
            hidden_states: [num_chunks, seq_len, hidden_size] - Backbone 输出的 hidden states
            line_ids: [num_chunks, seq_len] - 每个 token 对应的 line_id
            token_logits: [num_chunks, seq_len, num_classes] - Token-level 分类 logits (可选，用于回退)

        Returns:
            Dict[line_id, class_id]: 每行的预测类别
        """
        if self.use_line_level and self._has_cls_head and self._has_line_pooling:
            return self._decode_line_level(hidden_states, line_ids)
        elif token_logits is not None:
            return self._decode_token_level(token_logits, line_ids)
        else:
            logger.warning("No valid decode method available, returning empty predictions")
            return {}

    def _decode_line_level(
        self,
        hidden_states: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> Dict[int, int]:
        """
        Line-level 分类：使用 model.line_pooling + model.cls_head

        与 JointModel.forward() 的 Stage 1 逻辑一致
        """
        device = hidden_states.device
        num_chunks = hidden_states.shape[0]

        # Step 1: 使用 model.line_pooling 聚合 line features（复用模型模块，不重复实现）
        line_features, line_mask = self.model.line_pooling(hidden_states, line_ids)
        actual_num_lines = int(line_mask.sum().item())

        if actual_num_lines == 0:
            return {}

        # Step 2: 使用 cls_head 分类
        valid_features = line_features[:actual_num_lines]  # [L, H]
        with torch.no_grad():
            cls_logits = self.model.cls_head(valid_features)  # [L, num_classes]
            line_preds = cls_logits.argmax(dim=-1).cpu().tolist()  # [L]

        # Step 3: 获取 line_id 映射（使用 line_pooling 的方法）
        line_id_list = self.model.line_pooling.get_line_ids_mapping(line_ids).cpu().tolist()

        # Step 4: 构建结果
        line_classes = {}
        for i, pred in enumerate(line_preds):
            if i < len(line_id_list):
                line_classes[line_id_list[i]] = pred

        return line_classes

    def _decode_token_level(
        self,
        token_logits: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> Dict[int, int]:
        """
        Token-level 分类：token 预测 + 多数投票

        兼容旧方式
        """
        num_chunks = token_logits.shape[0]

        # 收集所有 token 预测
        all_token_preds = []
        all_line_ids = []

        for c in range(num_chunks):
            chunk_logits = token_logits[c]  # [seq_len, num_classes]
            chunk_line_ids = line_ids[c].cpu().tolist()

            for i in range(chunk_logits.shape[0]):
                all_token_preds.append(chunk_logits[i].argmax().item())
                all_line_ids.append(chunk_line_ids[i])

        # 多数投票
        return self._aggregate_by_voting(all_token_preds, all_line_ids)

    # 注：_aggregate_line_features 和 _get_ordered_line_ids 已移除
    # 统一使用 model.line_pooling（在 models/modules/line_pooling.py 中定义）
    # 避免代码重复，保持训练和推理一致

    def _aggregate_by_voting(
        self,
        token_preds: List[int],
        line_ids: List[int],
    ) -> Dict[int, int]:
        """多数投票聚合"""
        line_votes = {}
        for pred, line_id in zip(token_preds, line_ids):
            if line_id < 0:
                continue
            if line_id not in line_votes:
                line_votes[line_id] = []
            line_votes[line_id].append(pred)

        line_classes = {}
        for line_id, votes in line_votes.items():
            line_classes[line_id] = Counter(votes).most_common(1)[0][0]

        return line_classes

    def decode_batch(
        self,
        hidden_states: torch.Tensor,
        line_ids: torch.Tensor,
        line_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        批量解码（用于 batch 推理）

        Args:
            hidden_states: [B, seq_len, H] 或 [num_chunks, seq_len, H]
            line_ids: [B, seq_len] 或 [num_chunks, seq_len]
            line_mask: [B, max_lines]

        Returns:
            List[Tensor]: 每个样本的预测，[num_lines]
        """
        if not self._has_cls_head or not self._has_line_pooling:
            raise RuntimeError("Model does not have cls_head or line_pooling for batch decode")

        batch_size = line_mask.shape[0]
        device = hidden_states.device

        all_predictions = []

        for b in range(batch_size):
            num_lines = int(line_mask[b].sum().item())
            if num_lines == 0:
                all_predictions.append(torch.tensor([], device=device, dtype=torch.long))
                continue

            # 获取该样本的 hidden states 和 line_ids
            sample_hidden = hidden_states[b:b+1]  # [1, seq_len, H]
            sample_line_ids = line_ids[b:b+1]  # [1, seq_len]

            # 使用 model.line_pooling 聚合（复用模型模块）
            line_features, mask = self.model.line_pooling(sample_hidden, sample_line_ids)
            valid_features = line_features[:num_lines]

            # 分类
            with torch.no_grad():
                logits = self.model.cls_head(valid_features)
                preds = logits.argmax(dim=-1)

            all_predictions.append(preds)

        return all_predictions

    def compute_metrics(
        self,
        predictions: List[int],
        labels: List[int],
    ) -> Dict[str, float]:
        """
        计算分类指标

        Args:
            predictions: 预测列表
            labels: 标签列表

        Returns:
            Dict with accuracy, macro_f1, micro_f1
        """
        if not predictions or not labels:
            return {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

        # Accuracy
        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(labels)

        # Macro F1
        f1_scores = []
        for c in range(self.num_classes):
            tp = sum(1 for p, l in zip(predictions, labels) if p == c and l == c)
            fp = sum(1 for p, l in zip(predictions, labels) if p == c and l != c)
            fn = sum(1 for p, l in zip(predictions, labels) if p != c and l == c)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            if tp + fn > 0:  # 只计算有样本的类别
                f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "micro_f1": accuracy,  # micro F1 = accuracy for single-label
        }

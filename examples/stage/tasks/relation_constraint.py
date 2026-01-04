#!/usr/bin/env python
# coding=utf-8
"""
Relation Classification Task with Label-Pair Gating (P0)

提供带约束的关系分类解码逻辑：
1. 使用 label-pair 规则过滤不合理的关系类型
2. 支持置信度回退
3. 统一处理 ROOT 边的关系

规则定义在 layoutlmft/data/labels.py 中的 get_allowed_relation_ids()
"""

import logging
import torch
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


class RelationClassificationTask:
    """
    Relation Classification Task 封装类

    封装关系分类的解码逻辑：
    - decode: 原始贪心解码
    - decode_with_gating: 带 label-pair gating 的解码

    使用方式：
        task = RelationClassificationTask()
        predictions = task.decode_with_gating(
            rel_logits, pred_labels, pred_parents
        )
    """

    def __init__(self, num_relations: int = 3):
        """
        Args:
            num_relations: 关系类型数量（默认 3: connect, contain, equality）
        """
        self.num_relations = num_relations

        # 延迟导入避免循环依赖
        try:
            from layoutlmft.data.labels import (
                RELATION_LIST, RELATION2ID, ID2RELATION,
                get_allowed_relation_ids
            )
            self.relation_list = RELATION_LIST
            self.relation2id = RELATION2ID
            self.id2relation = ID2RELATION
            self.get_allowed_relation_ids = get_allowed_relation_ids
        except ImportError:
            logger.warning("无法导入 labels.py 中的关系定义，使用默认值")
            self.relation_list = ["connect", "contain", "equality"]
            self.relation2id = {r: i for i, r in enumerate(self.relation_list)}
            self.id2relation = {i: r for i, r in enumerate(self.relation_list)}
            self.get_allowed_relation_ids = None

    def decode(
        self,
        rel_logits: torch.Tensor,
    ) -> int:
        """
        原始贪心解码

        Args:
            rel_logits: [1, num_relations] - 单个样本的关系 logits

        Returns:
            预测的关系 ID
        """
        return rel_logits.argmax(dim=-1).item()

    def decode_with_gating(
        self,
        rel_logits: torch.Tensor,
        child_label_id: int,
        parent_label_id: int,
        fallback_to_argmax: bool = True,
    ) -> int:
        """
        带 label-pair gating 的解码

        Args:
            rel_logits: [1, num_relations] 或 [num_relations] - 关系 logits
            child_label_id: 子节点的语义类别 ID
            parent_label_id: 父节点的语义类别 ID
            fallback_to_argmax: 如果所有关系都被过滤，是否回退到 argmax

        Returns:
            预测的关系 ID
        """
        # 确保 logits 是 1D
        if rel_logits.dim() == 2:
            rel_logits = rel_logits.squeeze(0)

        # 获取原始 argmax（用于回退）
        original_pred = rel_logits.argmax().item()

        # 如果没有 gating 函数，直接返回 argmax
        if self.get_allowed_relation_ids is None:
            return original_pred

        # 获取允许的关系 ID 列表
        allowed_ids = self.get_allowed_relation_ids(child_label_id, parent_label_id)

        if not allowed_ids:
            # 没有允许的关系（不应该发生），返回原始 argmax
            return original_pred

        # 如果原始预测在允许列表中，直接返回
        if original_pred in allowed_ids:
            return original_pred

        # 原始预测不在允许列表中，需要在允许的关系中选择
        # 构建 mask
        mask = torch.full_like(rel_logits, float('-inf'))
        for rid in allowed_ids:
            if rid < len(mask):
                mask[rid] = 0.0

        masked_logits = rel_logits + mask

        # 检查是否有有效值
        if torch.isinf(masked_logits).all():
            if fallback_to_argmax:
                return original_pred
            else:
                return allowed_ids[0]  # 返回第一个允许的关系

        return masked_logits.argmax().item()

    def decode_batch(
        self,
        rel_model,
        parent_features: torch.Tensor,
        child_features: torch.Tensor,
        pred_parents: List[int],
        pred_labels: torch.Tensor,
        line_mask: torch.Tensor,
        use_gating: bool = True,
    ) -> List[int]:
        """
        批量解码关系

        Args:
            rel_model: Stage4 关系分类模型
            parent_features: [L, H] 或 [L+1, H] - 父节点特征（可能包含 ROOT）
            child_features: [L, H] 或 [L+1, H] - 子节点特征
            pred_parents: [L] - 预测的父节点 ID 列表（-1 表示 ROOT）
            pred_labels: [L] - 预测的语义类别 ID
            line_mask: [L] - 有效行的 mask
            use_gating: 是否使用 label-pair gating

        Returns:
            pred_relations: [L] - 预测的关系 ID 列表
        """
        num_lines = len(pred_parents)
        pred_relations = [0] * num_lines  # 默认 connect

        # 判断特征是否包含 ROOT（索引偏移）
        has_root_offset = (parent_features.shape[0] == num_lines + 1)

        for child_idx in range(num_lines):
            if not line_mask[child_idx]:
                continue

            parent_idx = pred_parents[child_idx]

            # 跳过 ROOT 子节点（parent_idx < 0）
            # ROOT 边的关系单独处理，不参与模型预测
            if parent_idx < 0 or parent_idx >= num_lines:
                pred_relations[child_idx] = 0  # ROOT 边默认 connect
                continue

            # 获取特征
            if has_root_offset:
                # 特征包含 ROOT，需要偏移
                parent_feat = parent_features[parent_idx + 1]
                child_feat = child_features[child_idx + 1]
            else:
                parent_feat = parent_features[parent_idx]
                child_feat = child_features[child_idx]

            # 前向传播
            rel_logits = rel_model(
                parent_feat.unsqueeze(0),
                child_feat.unsqueeze(0),
            )

            if use_gating:
                # 使用 label-pair gating
                child_label_id = pred_labels[child_idx].item()
                parent_label_id = pred_labels[parent_idx].item()
                pred_rel = self.decode_with_gating(
                    rel_logits, child_label_id, parent_label_id
                )
            else:
                pred_rel = self.decode(rel_logits)

            pred_relations[child_idx] = pred_rel

        return pred_relations


def get_default_relation_for_root_edge() -> int:
    """
    获取 ROOT 边的默认关系

    ROOT 边（parent_id < 0）不参与模型预测，使用固定值。
    返回 0 (connect)，表示直接连接到文档根。

    注意：这个值不会被 Stage4 评估，只是为了保持输出格式一致。
    """
    return 0  # connect

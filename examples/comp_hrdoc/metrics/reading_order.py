"""阅读顺序评估指标

用于评估 Order 模块的区域间阅读顺序预测性能。
包括主体文本阅读顺序和浮动元素阅读顺序。
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

from .teds import (
    Node,
    generate_doc_tree,
    transfer_tree_to_chain,
    split_chain_by_tag,
    sequence_edit_distance,
    min_edit_distance_between_groups,
)


@dataclass
class ReadingOrderResult:
    """阅读顺序评估结果"""
    # 主体文本阅读顺序
    macro_teds: float = 0.0
    micro_teds: float = 0.0
    total_distance: int = 0
    total_gt_nodes: int = 0
    total_pred_nodes: int = 0

    # 浮动元素阅读顺序
    macro_teds_floating: float = 0.0
    micro_teds_floating: float = 0.0
    total_distance_floating: float = 0.0
    total_gt_floating_nodes: int = 0
    total_pred_floating_nodes: int = 0

    num_samples: int = 0
    per_sample: Dict[str, Dict] = field(default_factory=dict)


class ReadingOrderMetric:
    """阅读顺序评估指标类

    评估区域间阅读顺序预测的质量。
    分别计算主体文本和浮动元素的 TEDS。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置指标状态"""
        # 主体文本
        self.teds_list = []
        self.distance_list = []
        self.gt_nodes_list = []
        self.pred_nodes_list = []

        # 浮动元素
        self.teds_floating_list = []
        self.distance_floating_list = []
        self.gt_floating_nodes_list = []
        self.pred_floating_nodes_list = []

        self.sample_ids = []

    def update(
        self,
        pred_texts: List[str],
        pred_parent_ids: List[int],
        pred_relations: List[str],
        gt_texts: List[str],
        gt_parent_ids: List[int],
        gt_relations: List[str],
        sample_id: str = None,
    ):
        """更新指标

        Args:
            pred_*: 预测结果 (格式: "class:text")
            gt_*: 真实标签
            sample_id: 样本ID
        """
        try:
            # 构建文档树
            gt_tree = generate_doc_tree(gt_texts, gt_parent_ids, gt_relations)
            pred_tree = generate_doc_tree(pred_texts, pred_parent_ids, pred_relations)

            # 转换为阅读顺序链
            gt_main_chain, gt_floating_chain = transfer_tree_to_chain(gt_tree)
            pred_main_chain, pred_floating_chain = transfer_tree_to_chain(pred_tree)

            # 主体文本 TEDS
            distance, teds = sequence_edit_distance(pred_main_chain, gt_main_chain)
            self.teds_list.append(teds)
            self.distance_list.append(distance)
            self.gt_nodes_list.append(len(gt_main_chain))
            self.pred_nodes_list.append(len(pred_main_chain))

            # 浮动元素 TEDS
            gt_floating_groups = split_chain_by_tag(gt_floating_chain[1:])  # 跳过 ROOT
            pred_floating_groups = split_chain_by_tag(pred_floating_chain[1:])

            distance_floating, teds_floating = min_edit_distance_between_groups(
                gt_floating_groups, pred_floating_groups
            )
            self.teds_floating_list.append(teds_floating)
            self.distance_floating_list.append(distance_floating)
            self.gt_floating_nodes_list.append(sum(len(g) for g in gt_floating_groups))
            self.pred_floating_nodes_list.append(sum(len(g) for g in pred_floating_groups))

            self.sample_ids.append(sample_id or str(len(self.teds_list)))

        except Exception as e:
            print(f"阅读顺序评估错误 ({sample_id}): {e}")

    def update_from_chains(
        self,
        pred_main_chain: List[str],
        gt_main_chain: List[str],
        pred_floating_groups: List[List[str]] = None,
        gt_floating_groups: List[List[str]] = None,
        sample_id: str = None,
    ):
        """直接从阅读顺序链更新指标

        Args:
            pred_main_chain: 预测的主体文本链
            gt_main_chain: 真实的主体文本链
            pred_floating_groups: 预测的浮动元素组
            gt_floating_groups: 真实的浮动元素组
            sample_id: 样本ID
        """
        try:
            # 主体文本 TEDS
            distance, teds = sequence_edit_distance(pred_main_chain, gt_main_chain)
            self.teds_list.append(teds)
            self.distance_list.append(distance)
            self.gt_nodes_list.append(len(gt_main_chain))
            self.pred_nodes_list.append(len(pred_main_chain))

            # 浮动元素 TEDS
            if pred_floating_groups is not None and gt_floating_groups is not None:
                distance_floating, teds_floating = min_edit_distance_between_groups(
                    gt_floating_groups, pred_floating_groups
                )
                self.teds_floating_list.append(teds_floating)
                self.distance_floating_list.append(distance_floating)
                self.gt_floating_nodes_list.append(sum(len(g) for g in gt_floating_groups))
                self.pred_floating_nodes_list.append(sum(len(g) for g in pred_floating_groups))
            else:
                self.teds_floating_list.append(1.0)
                self.distance_floating_list.append(0)
                self.gt_floating_nodes_list.append(0)
                self.pred_floating_nodes_list.append(0)

            self.sample_ids.append(sample_id or str(len(self.teds_list)))

        except Exception as e:
            print(f"阅读顺序评估错误 ({sample_id}): {e}")

    def compute(self) -> ReadingOrderResult:
        """计算最终指标"""
        result = ReadingOrderResult()

        if len(self.teds_list) == 0:
            return result

        result.num_samples = len(self.teds_list)

        # 主体文本指标
        result.macro_teds = sum(self.teds_list) / len(self.teds_list)

        total_distance = sum(self.distance_list)
        total_max_nodes = sum(
            max(g, p) for g, p in zip(self.gt_nodes_list, self.pred_nodes_list)
        )
        result.micro_teds = 1.0 - total_distance / total_max_nodes if total_max_nodes > 0 else 1.0

        result.total_distance = total_distance
        result.total_gt_nodes = sum(self.gt_nodes_list)
        result.total_pred_nodes = sum(self.pred_nodes_list)

        # 浮动元素指标
        if self.teds_floating_list:
            result.macro_teds_floating = sum(self.teds_floating_list) / len(self.teds_floating_list)

            total_distance_floating = sum(self.distance_floating_list)
            total_max_floating_nodes = sum(
                max(g, p) for g, p in zip(self.gt_floating_nodes_list, self.pred_floating_nodes_list)
            )
            result.micro_teds_floating = (
                1.0 - total_distance_floating / total_max_floating_nodes
                if total_max_floating_nodes > 0 else 1.0
            )

            result.total_distance_floating = total_distance_floating
            result.total_gt_floating_nodes = sum(self.gt_floating_nodes_list)
            result.total_pred_floating_nodes = sum(self.pred_floating_nodes_list)

        # 每样本结果
        for i, sid in enumerate(self.sample_ids):
            result.per_sample[sid] = {
                'teds': self.teds_list[i],
                'distance': self.distance_list[i],
                'gt_nodes': self.gt_nodes_list[i],
                'pred_nodes': self.pred_nodes_list[i],
            }
            if i < len(self.teds_floating_list):
                result.per_sample[sid].update({
                    'teds_floating': self.teds_floating_list[i],
                    'distance_floating': self.distance_floating_list[i],
                    'gt_floating_nodes': self.gt_floating_nodes_list[i],
                    'pred_floating_nodes': self.pred_floating_nodes_list[i],
                })

        return result


class IntraRegionOrderMetric:
    """区域内阅读顺序评估

    评估 Detect 模块中文本行分组的阅读顺序预测。
    使用 successor 准确率和组正确率。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置状态"""
        self.correct_successors = 0
        self.total_successors = 0
        self.correct_groups = 0
        self.total_groups = 0

    def update(
        self,
        pred_successors: List[int],
        gt_successors: List[int],
        line_mask: List[bool] = None,
    ):
        """更新指标

        Args:
            pred_successors: 预测的后继索引 (-1 表示无后继)
            gt_successors: 真实的后继索引
            line_mask: 有效行掩码
        """
        assert len(pred_successors) == len(gt_successors)

        for i, (pred, gt) in enumerate(zip(pred_successors, gt_successors)):
            if line_mask and not line_mask[i]:
                continue
            if gt >= 0:  # 有后继的行
                self.total_successors += 1
                if pred == gt:
                    self.correct_successors += 1

    def update_groups(
        self,
        pred_groups: List[List[int]],
        gt_groups: List[List[int]],
    ):
        """更新分组准确率

        Args:
            pred_groups: 预测的分组 (每组是行索引列表)
            gt_groups: 真实的分组
        """
        self.total_groups += len(gt_groups)

        # 将预测组转换为集合进行比较
        pred_sets = [frozenset(g) for g in pred_groups]
        gt_sets = [frozenset(g) for g in gt_groups]

        for gt_set in gt_sets:
            if gt_set in pred_sets:
                self.correct_groups += 1

    def compute(self) -> Dict[str, float]:
        """计算指标"""
        successor_acc = (
            self.correct_successors / self.total_successors
            if self.total_successors > 0 else 0.0
        )
        group_acc = (
            self.correct_groups / self.total_groups
            if self.total_groups > 0 else 0.0
        )

        return {
            'successor_accuracy': successor_acc,
            'group_accuracy': group_acc,
            'correct_successors': self.correct_successors,
            'total_successors': self.total_successors,
            'correct_groups': self.correct_groups,
            'total_groups': self.total_groups,
        }

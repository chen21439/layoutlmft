#!/usr/bin/env python
# coding=utf-8
"""
Evaluator - 统一评估接口

支持页面级别和文档级别的评估，使用 Batch 抽象层隐藏差异。

设计原则：
- 使用 Predictor 进行推理
- 从 Sample 中提取 GT
- 计算指标并返回 EvaluationOutput
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.batch import Sample, BatchBase, wrap_batch
from .predictor import Predictor, PredictionOutput


# 标签映射（从 layoutlmft.data.labels 导入或定义）
try:
    from layoutlmft.data.labels import LABEL_LIST, LABEL2ID, ID2LABEL
except ImportError:
    LABEL_LIST = [
        "other", "title", "section", "list", "table", "figure",
        "caption", "header", "footer", "equation", "abstract",
        "reference", "paragraph", "toc"
    ]
    LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
    ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# 关系映射
RELATION_LABELS = {"connect": 0, "contain": 1, "equality": 2}
ID2RELATION = {v: k for k, v in RELATION_LABELS.items()}


@dataclass
class EvaluationOutput:
    """评估结果"""
    # Stage 1: 分类指标
    line_accuracy: float = 0.0
    line_macro_f1: float = 0.0
    line_micro_f1: float = 0.0

    # Stage 3: Parent 准确率
    parent_accuracy: float = 0.0

    # Stage 4: Relation 指标
    relation_accuracy: float = 0.0
    relation_macro_f1: float = 0.0

    # 统计信息
    num_samples: int = 0
    num_lines: int = 0
    num_parent_pairs: int = 0
    num_relation_pairs: int = 0

    # 详细指标（可选）
    per_class_f1: Optional[Dict[str, float]] = None
    per_relation_f1: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[Any] = None


class Evaluator:
    """
    统一评估器

    使用方式：
        evaluator = Evaluator(model, device)
        output = evaluator.evaluate(dataloader)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device = None,
        id2label: Dict[int, str] = None,
    ):
        """
        Args:
            model: JointModel
            device: 计算设备
            id2label: 类别 ID 到名称的映射
        """
        self.predictor = Predictor(model, device)
        self.device = device or next(model.parameters()).device
        self.id2label = id2label or ID2LABEL

    def evaluate(
        self,
        dataloader,
        compute_teds: bool = False,
        verbose: bool = True,
        debug: bool = False,
    ) -> EvaluationOutput:
        """
        评估整个数据集

        Args:
            dataloader: DataLoader，返回 raw batch dict
            compute_teds: 是否计算 TEDS（较慢）
            verbose: 是否显示进度条
            debug: 是否打印调试信息

        Returns:
            EvaluationOutput: 评估结果
        """
        self.predictor.model.eval()

        # 收集所有预测和 GT
        all_gt_classes = []
        all_pred_classes = []
        all_gt_parents = []
        all_pred_parents = []
        all_gt_relations = []
        all_pred_relations = []

        num_samples = 0

        # 调试统计
        debug_parent_skipped_padding = 0
        debug_parent_skipped_invalid = 0
        debug_parent_total = 0
        debug_first_samples = []

        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

        with torch.no_grad():
            for raw_batch in iterator:
                # 包装为 Batch 抽象
                batch = wrap_batch(raw_batch)
                batch = batch.to(self.device)

                for sample in batch:
                    num_samples += 1

                    # 提取 GT
                    gt = self._extract_gt(sample)

                    # 预测
                    pred = self.predictor.predict(sample)

                    # 收集分类结果
                    for line_id, gt_class in gt["classes"].items():
                        pred_class = pred.line_classes.get(line_id, 0)
                        all_gt_classes.append(gt_class)
                        all_pred_classes.append(pred_class)

                    # 收集 Parent 结果
                    # 注意：gt_parent = -1 表示 ROOT，也是有效目标
                    # gt_parent = -100 表示 padding，应该跳过
                    for idx, (gt_parent, pred_parent) in enumerate(zip(
                        gt["parents"], pred.line_parents
                    )):
                        debug_parent_total += 1
                        if gt_parent == -100:
                            debug_parent_skipped_padding += 1
                            continue
                        if idx >= len(pred.line_parents):
                            continue
                        # 训练时跳过 gt_parent >= child_idx 的情况
                        # 这里 idx 就是 child_idx
                        if gt_parent >= idx:
                            debug_parent_skipped_invalid += 1
                            continue
                        all_gt_parents.append(gt_parent)
                        all_pred_parents.append(pred_parent)

                        # 调试：收集前几个样本的详情
                        if debug and len(debug_first_samples) < 5 and num_samples <= 2:
                            debug_first_samples.append({
                                "sample": num_samples,
                                "child_idx": idx,
                                "gt_parent": gt_parent,
                                "pred_parent": pred_parent,
                                "num_lines_gt": len(gt["parents"]),
                                "num_lines_pred": len(pred.line_parents),
                            })

                    # 收集 Relation 结果
                    # 注意：relation 只在 parent >= 0 且 parent < child_idx 时有效
                    for idx, (gt_rel, gt_parent, pred_rel) in enumerate(zip(
                        gt["relations"], gt["parents"], pred.line_relations
                    )):
                        if gt_parent == -100 or gt_rel == -100:
                            continue
                        if idx >= len(pred.line_relations):
                            continue
                        if gt_parent < 0 or gt_parent >= idx:
                            continue
                        all_gt_relations.append(gt_rel)
                        all_pred_relations.append(pred_rel)

        # 打印调试信息
        if debug or verbose:
            print(f"\n[Evaluator Debug] Parent: evaluated={len(all_gt_parents)}, skipped_padding={debug_parent_skipped_padding}, skipped_invalid={debug_parent_skipped_invalid}")

            # Relation 统计
            if all_gt_relations:
                from collections import Counter
                gt_rel_counter = Counter(all_gt_relations)
                pred_rel_counter = Counter(all_pred_relations)
                # 转换为英文名称
                gt_rel_named = {ID2RELATION.get(k, f"rel_{k}"): v for k, v in gt_rel_counter.items()}
                pred_rel_named = {ID2RELATION.get(k, f"rel_{k}"): v for k, v in pred_rel_counter.items()}
                print(f"[Evaluator Debug] Relation: evaluated={len(all_gt_relations)}")
                print(f"  GT:   {gt_rel_named}")
                print(f"  Pred: {pred_rel_named}")
                # 计算每类 Recall
                for rel_id in sorted(gt_rel_counter.keys()):
                    gt_count = gt_rel_counter[rel_id]
                    correct = sum(1 for g, p in zip(all_gt_relations, all_pred_relations) if g == rel_id and p == rel_id)
                    rel_name = ID2RELATION.get(rel_id, f"rel_{rel_id}")
                    print(f"  {rel_name}: GT={gt_count}, Correct={correct}, Recall={100*correct/gt_count:.1f}%")

        # 计算指标
        output = self._compute_metrics(
            all_gt_classes, all_pred_classes,
            all_gt_parents, all_pred_parents,
            all_gt_relations, all_pred_relations,
        )

        output.num_samples = num_samples
        output.num_lines = len(all_gt_classes)
        output.num_parent_pairs = len(all_gt_parents)
        output.num_relation_pairs = len(all_gt_relations)

        self.predictor.model.train()
        return output

    def _extract_gt(self, sample: Sample) -> Dict[str, Any]:
        """
        从 Sample 中提取 Ground Truth

        Returns:
            {
                "classes": {line_id: class_id, ...},
                "parents": [parent_id, ...],
                "relations": [relation_id, ...],
            }
        """
        gt = {
            "classes": {},
            "parents": [],
            "relations": [],
        }

        if sample.line_ids is None or sample.labels is None:
            return gt

        # 处理分类标签
        if sample.is_document_level:
            # 多 chunk：展平处理
            all_labels = sample.labels.reshape(-1).cpu().tolist()
            all_line_ids = sample.line_ids.reshape(-1).cpu().tolist()
        else:
            # 单 chunk
            all_labels = sample.labels.cpu().tolist()
            all_line_ids = sample.line_ids.cpu().tolist()

        # Token -> Line 聚合
        line_label_votes = defaultdict(list)
        for label, line_id in zip(all_labels, all_line_ids):
            if line_id >= 0 and label >= 0:
                line_label_votes[line_id].append(label)

        for line_id, votes in line_label_votes.items():
            # 多数投票
            from collections import Counter
            gt["classes"][line_id] = Counter(votes).most_common(1)[0][0]

        # 处理 parent_ids
        if sample.line_parent_ids is not None:
            gt["parents"] = sample.line_parent_ids.cpu().tolist()

        # 处理 relations
        if sample.line_relations is not None:
            gt["relations"] = sample.line_relations.cpu().tolist()

        return gt

    def _compute_metrics(
        self,
        gt_classes: List[int],
        pred_classes: List[int],
        gt_parents: List[int],
        pred_parents: List[int],
        gt_relations: List[int],
        pred_relations: List[int],
    ) -> EvaluationOutput:
        """计算所有指标"""
        output = EvaluationOutput()

        # Stage 1: 分类指标
        if gt_classes:
            output.line_accuracy = self._accuracy(gt_classes, pred_classes)
            output.line_macro_f1 = self._macro_f1(gt_classes, pred_classes)
            output.line_micro_f1 = self._micro_f1(gt_classes, pred_classes)

        # Stage 3: Parent 准确率
        if gt_parents:
            output.parent_accuracy = self._accuracy(gt_parents, pred_parents)

        # Stage 4: Relation 指标
        if gt_relations:
            output.relation_accuracy = self._accuracy(gt_relations, pred_relations)
            output.relation_macro_f1 = self._macro_f1(
                gt_relations, pred_relations, num_classes=3
            )

        return output

    def _accuracy(self, gt: List[int], pred: List[int]) -> float:
        """计算准确率"""
        if not gt:
            return 0.0
        correct = sum(g == p for g, p in zip(gt, pred))
        return correct / len(gt)

    def _macro_f1(
        self,
        gt: List[int],
        pred: List[int],
        num_classes: int = None
    ) -> float:
        """计算 Macro F1"""
        if not gt:
            return 0.0

        if num_classes is None:
            num_classes = max(max(gt), max(pred)) + 1

        f1_scores = []
        for c in range(num_classes):
            tp = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
            fp = sum(1 for g, p in zip(gt, pred) if g != c and p == c)
            fn = sum(1 for g, p in zip(gt, pred) if g == c and p != c)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            if tp + fn > 0:  # 只计算有样本的类别
                f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    def _micro_f1(self, gt: List[int], pred: List[int]) -> float:
        """计算 Micro F1（等于 accuracy）"""
        return self._accuracy(gt, pred)

    def print_results(self, output: EvaluationOutput):
        """打印评估结果"""
        print("=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"  Samples: {output.num_samples}")
        print(f"  Lines:   {output.num_lines}")
        print("-" * 60)
        print(f"  Stage 1 (Classification):")
        print(f"    Accuracy:  {output.line_accuracy * 100:.2f}%")
        print(f"    Macro F1:  {output.line_macro_f1 * 100:.2f}%")
        print("-" * 60)
        print(f"  Stage 3 (Parent):")
        print(f"    Accuracy:  {output.parent_accuracy * 100:.2f}%")
        print(f"    Pairs:     {output.num_parent_pairs}")
        print("-" * 60)
        print(f"  Stage 4 (Relation):")
        print(f"    Accuracy:  {output.relation_accuracy * 100:.2f}%")
        print(f"    Macro F1:  {output.relation_macro_f1 * 100:.2f}%")
        print(f"    Pairs:     {output.num_relation_pairs}")
        print("=" * 60)

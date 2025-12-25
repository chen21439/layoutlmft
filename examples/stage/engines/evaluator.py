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
        self._parent_class_stats = []  # 重置 parent 类别统计

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
                    gt_line_ids = gt.get("line_ids", list(range(len(gt["parents"]))))

                    # 调试：打印前几个样本的对齐信息
                    if debug and num_samples <= 2:
                        print(f"\n[Parent Debug] Sample {num_samples}:")
                        print(f"  gt['parents'][:10] = {gt['parents'][:10]}")
                        print(f"  gt['line_ids'][:10] = {gt['line_ids'][:10]}")
                        print(f"  pred.line_parents[:10] = {pred.line_parents[:10]}")
                        print(f"  pred.line_ids[:10] = {pred.line_ids[:10]}")
                        print(f"  len(gt['parents'])={len(gt['parents'])}, len(pred.line_parents)={len(pred.line_parents)}")

                    for idx, (gt_parent, pred_parent) in enumerate(zip(
                        gt["parents"], pred.line_parents
                    )):
                        debug_parent_total += 1
                        if gt_parent == -100:
                            debug_parent_skipped_padding += 1
                            continue
                        if idx >= len(pred.line_parents):
                            continue
                        # 使用实际 line_id 而不是 idx 来判断父子关系有效性
                        # parent 的 line_id 必须小于 child 的 line_id
                        child_line_id = gt_line_ids[idx] if idx < len(gt_line_ids) else idx
                        if gt_parent >= child_line_id:
                            debug_parent_skipped_invalid += 1
                            continue
                        all_gt_parents.append(gt_parent)
                        all_pred_parents.append(pred_parent)

                        # 收集 parent 类别统计信息
                        child_class = gt["classes"].get(child_line_id, -1)
                        gt_parent_line_id = gt_line_ids[gt_parent] if gt_parent >= 0 and gt_parent < len(gt_line_ids) else None
                        gt_parent_class = gt["classes"].get(gt_parent_line_id, None) if gt_parent_line_id is not None else None
                        pred_parent_line_id = gt_line_ids[pred_parent] if pred_parent >= 0 and pred_parent < len(gt_line_ids) else None
                        pred_parent_class = gt["classes"].get(pred_parent_line_id, None) if pred_parent_line_id is not None else None

                        self._parent_class_stats.append({
                            "child_idx": idx,
                            "child_class": child_class,
                            "gt_parent": gt_parent,
                            "gt_parent_class": gt_parent_class,
                            "pred_parent": pred_parent,
                            "pred_parent_class": pred_parent_class,
                            "is_correct": gt_parent == pred_parent,
                        })

                        # 调试：收集前几个样本的详情
                        if debug and len(debug_first_samples) < 5 and num_samples <= 2:
                            debug_first_samples.append({
                                "sample": num_samples,
                                "child_idx": idx,
                                "child_line_id": child_line_id,
                                "gt_parent": gt_parent,
                                "pred_parent": pred_parent,
                                "num_lines_gt": len(gt["parents"]),
                                "num_lines_pred": len(pred.line_parents),
                            })

                    # 收集 Relation 结果
                    # 注意：relation 只在 parent >= 0 且 parent < child_line_id 时有效
                    for idx, (gt_rel, gt_parent, pred_rel) in enumerate(zip(
                        gt["relations"], gt["parents"], pred.line_relations
                    )):
                        if gt_parent == -100 or gt_rel == -100:
                            continue
                        if idx >= len(pred.line_relations):
                            continue
                        # 使用实际 line_id 进行比较
                        child_line_id = gt_line_ids[idx] if idx < len(gt_line_ids) else idx
                        if gt_parent < 0 or gt_parent >= child_line_id:
                            continue
                        all_gt_relations.append(gt_rel)
                        all_pred_relations.append(pred_rel)

        # 打印调试信息
        if debug or verbose:
            print(f"\n[Evaluator Debug] Parent: evaluated={len(all_gt_parents)}, skipped_padding={debug_parent_skipped_padding}, skipped_invalid={debug_parent_skipped_invalid}")

            # Parent 按类别统计
            if all_gt_parents and hasattr(self, '_parent_class_stats'):
                from collections import Counter
                stats = self._parent_class_stats
                print(f"[Evaluator Debug] Parent by class (child_class -> parent_class):")
                # 按 child class 分组统计
                child_class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
                for item in stats:
                    child_cls = item["child_class"]
                    child_class_stats[child_cls]["total"] += 1
                    if item["is_correct"]:
                        child_class_stats[child_cls]["correct"] += 1

                for child_cls in sorted(child_class_stats.keys()):
                    s = child_class_stats[child_cls]
                    cls_name = self.id2label.get(child_cls, f"cls_{child_cls}")
                    acc = 100 * s["correct"] / s["total"] if s["total"] > 0 else 0
                    print(f"  {cls_name}: {s['correct']}/{s['total']} = {acc:.1f}%")

                # 打印一些错误案例
                errors = [item for item in stats if not item["is_correct"]][:10]
                if errors:
                    print(f"[Evaluator Debug] Parent errors (first 10):")
                    for e in errors:
                        child_name = self.id2label.get(e["child_class"], f"cls_{e['child_class']}")
                        gt_parent_name = self.id2label.get(e["gt_parent_class"], f"cls_{e['gt_parent_class']}") if e["gt_parent_class"] is not None else "ROOT"
                        pred_parent_name = self.id2label.get(e["pred_parent_class"], f"cls_{e['pred_parent_class']}") if e["pred_parent_class"] is not None else "ROOT"
                        print(f"  child[{e['child_idx']}]={child_name}, gt_parent={e['gt_parent']}({gt_parent_name}), pred_parent={e['pred_parent']}({pred_parent_name})")

                # 按 (child_class, gt_parent_class) 分组统计误判情况
                self._print_parent_confusion_matrix(stats)

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

    def _print_parent_confusion_matrix(self, stats: List[Dict]) -> None:
        """
        以表格格式打印 Parent 混淆矩阵

        格式示例：
        +-------------+-------------+----------+-------------------------+
        | Child Class | GT Parent   | Acc      | Mispredictions          |
        +-------------+-------------+----------+-------------------------+
        | fstline     | fstline     | 90% (587/652) | section:54, paraline:11 |
        ...
        """
        confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for item in stats:
            child_cls = item["child_class"]
            gt_p_cls = item["gt_parent_class"]
            pred_p_cls = item["pred_parent_class"]
            confusion[child_cls][gt_p_cls][pred_p_cls] += 1

        # 收集所有需要显示的行（只显示有错误的）
        rows = []
        for child_cls in sorted(confusion.keys()):
            child_name = self.id2label.get(child_cls, f"cls_{child_cls}")

            for gt_p_cls in sorted(confusion[child_cls].keys(), key=lambda x: (x is None, x)):
                gt_p_name = self.id2label.get(gt_p_cls, f"cls_{gt_p_cls}") if gt_p_cls is not None else "ROOT"
                pred_counts = confusion[child_cls][gt_p_cls]
                total = sum(pred_counts.values())
                correct = pred_counts.get(gt_p_cls, 0)

                # 只显示有错误的情况
                if correct < total:
                    error_count = total - correct

                    # 收集错误详情，按数量从大到小排序
                    errors_detail = []
                    for pred_p_cls, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
                        if pred_p_cls != gt_p_cls:
                            pred_p_name = self.id2label.get(pred_p_cls, f"cls_{pred_p_cls}") if pred_p_cls is not None else "ROOT"
                            errors_detail.append(f"{pred_p_name}:{cnt}")

                    acc_pct = 100 * correct / total if total > 0 else 0
                    rows.append({
                        'child_name': child_name,
                        'gt_name': gt_p_name,
                        'acc_pct': acc_pct,
                        'correct': correct,
                        'total': total,
                        'error_count': error_count,
                        'errors_detail': ', '.join(errors_detail),
                    })

        # 按错误数量从大到小排序
        rows.sort(key=lambda x: -x['error_count'])

        if not rows:
            print(f"[Evaluator Debug] Parent Confusion Matrix: No errors found")
            return

        # 计算列宽
        col_widths = {
            'child': max(13, max(len(row['child_name']) for row in rows) + 2) if rows else 13,
            'gt': max(13, max(len(row['gt_name']) for row in rows) + 2) if rows else 13,
            'acc': max(10, 12),  # "90% (587/652)"
            'errors': max(25, max(len(row['errors_detail']) for row in rows) + 2) if rows else 25,
        }

        # 打印表格
        print(f"\n[Evaluator Debug] Parent Confusion Matrix:")

        # 上边框
        total_width = sum(col_widths.values()) + 7  # 3 separators + 2 edges
        print('+' + '-' * (col_widths['child'] + 1) + '+' + '-' * (col_widths['gt'] + 1) + '+' + '-' * (col_widths['acc'] + 1) + '+' + '-' * (col_widths['errors'] + 1) + '+')

        # 表头
        print('| ' + 'Child Class'.ljust(col_widths['child']) + ' | ' + 'GT Parent'.ljust(col_widths['gt']) + ' | ' + 'Accuracy'.ljust(col_widths['acc']) + ' | ' + 'Mispredictions'.ljust(col_widths['errors']) + ' |')

        # 中间分隔线
        print('+' + '-' * (col_widths['child'] + 1) + '+' + '-' * (col_widths['gt'] + 1) + '+' + '-' * (col_widths['acc'] + 1) + '+' + '-' * (col_widths['errors'] + 1) + '+')

        # 数据行
        for row in rows:
            acc_str = f"{row['acc_pct']:.0f}% ({row['correct']}/{row['total']})"
            child_str = row['child_name'].ljust(col_widths['child'])
            gt_str = row['gt_name'].ljust(col_widths['gt'])
            acc_str = acc_str.ljust(col_widths['acc'])
            errors_str = row['errors_detail'].ljust(col_widths['errors'])

            print(f"| {child_str} | {gt_str} | {acc_str} | {errors_str} |")

        # 下边框
        print('+' + '-' * (col_widths['child'] + 1) + '+' + '-' * (col_widths['gt'] + 1) + '+' + '-' * (col_widths['acc'] + 1) + '+' + '-' * (col_widths['errors'] + 1) + '+')

    def _extract_gt(self, sample: Sample) -> Dict[str, Any]:
        """
        从 Sample 中提取 Ground Truth

        Returns:
            {
                "classes": {line_id: class_id, ...},
                "parents": [parent_id, ...],
                "relations": [relation_id, ...],
                "line_ids": [line_id, ...],  # 每个位置对应的实际 line_id
            }
        """
        gt = {
            "classes": {},
            "parents": [],
            "relations": [],
            "line_ids": [],  # 用于正确比较 parent_id 和 child_id
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

        # Token -> Line 聚合，同时保持 line_id 顺序
        line_label_votes = defaultdict(list)
        seen_line_ids = set()
        ordered_line_ids = []
        for label, line_id in zip(all_labels, all_line_ids):
            if line_id >= 0:
                if line_id not in seen_line_ids:
                    seen_line_ids.add(line_id)
                    ordered_line_ids.append(line_id)
                if label >= 0:
                    line_label_votes[line_id].append(label)

        for line_id, votes in line_label_votes.items():
            # 多数投票
            from collections import Counter
            gt["classes"][line_id] = Counter(votes).most_common(1)[0][0]

        # 处理 parent_ids 和 relations
        # 重要：
        # 1. sample.line_parent_ids 按行序号索引（0, 1, 2, ...），不是按 line_id 索引
        # 2. sample.line_parent_ids 的值是 parent 的 line_id，需要转换为行序号
        # 3. pred.line_parents 是行序号，所以 GT 也要用行序号表示

        # 建立 line_id -> 行序号 的映射
        line_id_to_row = {lid: row for row, lid in enumerate(ordered_line_ids)}

        if sample.line_parent_ids is not None:
            raw_parents = sample.line_parent_ids.cpu().tolist()
            # 按行序号顺序提取，并将 parent_line_id 转换为行序号
            for row in range(len(ordered_line_ids)):
                if row < len(raw_parents):
                    parent_line_id = raw_parents[row]
                    if parent_line_id == -1:
                        gt["parents"].append(-1)  # ROOT
                    elif parent_line_id == -100:
                        gt["parents"].append(-100)  # padding
                    elif parent_line_id in line_id_to_row:
                        gt["parents"].append(line_id_to_row[parent_line_id])
                    else:
                        # parent 的 line_id 不在当前文档中（可能是跨页被截断）
                        gt["parents"].append(-1)  # 视为 ROOT
                else:
                    gt["parents"].append(-100)  # padding

        if sample.line_relations is not None:
            raw_relations = sample.line_relations.cpu().tolist()
            # 按行序号顺序提取
            for row in range(len(ordered_line_ids)):
                if row < len(raw_relations):
                    gt["relations"].append(raw_relations[row])
                else:
                    gt["relations"].append(-100)

        # 存储按顺序出现的 line_ids
        gt["line_ids"] = ordered_line_ids

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

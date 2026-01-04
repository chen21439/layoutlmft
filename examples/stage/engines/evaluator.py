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

# 导入 TEDS 计算函数
try:
    from util.hrdoc_eval import compute_teds_score
    TEDS_AVAILABLE = True
except ImportError:
    TEDS_AVAILABLE = False


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
    relation_micro_f1: float = 0.0

    # TEDS 指标
    teds_score: Optional[float] = None

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
        save_predictions: bool = False,
        output_dir: str = None,
    ) -> EvaluationOutput:
        """
        评估整个数据集

        Args:
            dataloader: DataLoader，返回 raw batch dict
            compute_teds: 是否计算 TEDS（较慢）
            verbose: 是否显示进度条
            debug: 是否打印调试信息
            save_predictions: 是否保存预测结果到文件
            output_dir: 预测结果输出目录

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

        # 收集预测结果用于保存
        all_predictions = []

        # 收集用于 TEDS 计算的文档级数据
        teds_gt_docs = []
        teds_pred_docs = []

        num_samples = 0

        # 调试统计
        debug_parent_skipped_padding = 0
        debug_parent_skipped_invalid = 0
        debug_parent_total = 0
        debug_first_samples = []
        self._parent_class_stats = []  # 重置 parent 类别统计
        self._doc_json_paths = {}  # 文档名 -> json_path 映射

        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

        with torch.no_grad():
            for raw_batch in iterator:
                # 包装为 Batch 抽象
                batch = wrap_batch(raw_batch)
                batch = batch.to(self.device)

                for sample in batch:
                    num_samples += 1

                    # 保存 json_path 映射
                    if sample.document_name and sample.json_path:
                        self._doc_json_paths[sample.document_name] = sample.json_path

                    # 提取 GT
                    gt = self._extract_gt(sample)

                    # 预测
                    pred = self.predictor.predict(sample)

                    # 收集预测结果用于保存（按文档分组，同时保存 GT 和预测）
                    if save_predictions:
                        doc_name = sample.document_name or f"doc_{num_samples}"
                        sorted_line_ids = sorted(pred.line_classes.keys())
                        doc_lines = []
                        for idx, line_id in enumerate(sorted_line_ids):
                            # 预测结果
                            pred_class = pred.line_classes.get(line_id, 0)
                            pred_parent = pred.line_parents[idx] if idx < len(pred.line_parents) else -1
                            pred_relation = pred.line_relations[idx] if idx < len(pred.line_relations) else 0
                            # GT 结果
                            gt_class = gt["classes"].get(line_id, -1)
                            gt_parent = gt["parents"][idx] if idx < len(gt["parents"]) else -1
                            gt_relation = gt["relations"][idx] if idx < len(gt["relations"]) else -1
                            doc_lines.append({
                                "line_id": line_id,
                                "gt_class": self.id2label.get(gt_class, f"cls_{gt_class}"),
                                "pred_class": self.id2label.get(pred_class, f"cls_{pred_class}"),
                                "gt_parent_id": gt_parent,
                                "pred_parent_id": pred_parent,
                                "gt_relation": ID2RELATION.get(gt_relation, f"rel_{gt_relation}") if gt_relation >= 0 else "N/A",
                                "pred_relation": ID2RELATION.get(pred_relation, f"rel_{pred_relation}"),
                            })
                        all_predictions.append({
                            "document_name": doc_name,
                            "lines": doc_lines,
                        })

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

                        # 获取实际的 line_id（不是索引）
                        child_line_id = gt_line_ids[idx] if idx < len(gt_line_ids) else idx
                        gt_parent_line_id = gt_line_ids[gt_parent] if gt_parent >= 0 and gt_parent < len(gt_line_ids) else -1
                        pred_parent_line_id_val = gt_line_ids[pred_parent] if pred_parent >= 0 and pred_parent < len(gt_line_ids) else -1

                        self._parent_class_stats.append({
                            "child_idx": idx,
                            "child_class": child_class,
                            "child_line_id": child_line_id,
                            "gt_parent": gt_parent,
                            "gt_parent_class": gt_parent_class,
                            "gt_parent_line_id": gt_parent_line_id,
                            "pred_parent": pred_parent,
                            "pred_parent_class": pred_parent_class,
                            "pred_parent_line_id": pred_parent_line_id_val,
                            "is_correct": gt_parent == pred_parent,
                            "document_name": sample.document_name,
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

                    # 收集用于 TEDS 计算的文档级数据
                    if compute_teds and TEDS_AVAILABLE:
                        sorted_line_ids = sorted(gt["classes"].keys())
                        gt_doc = []
                        pred_doc = []
                        for idx, line_id in enumerate(sorted_line_ids):
                            gt_class_id = gt["classes"].get(line_id, 0)
                            pred_class_id = pred.line_classes.get(line_id, 0)
                            gt_parent_idx = gt["parents"][idx] if idx < len(gt["parents"]) else -1
                            pred_parent_idx = pred.line_parents[idx] if idx < len(pred.line_parents) else -1
                            gt_rel_id = gt["relations"][idx] if idx < len(gt["relations"]) else -100
                            pred_rel_id = pred.line_relations[idx] if idx < len(pred.line_relations) else 0

                            gt_doc.append({
                                "class": self.id2label.get(gt_class_id, f"cls_{gt_class_id}"),
                                "text": f"line_{line_id}",
                                "parent_id": gt_parent_idx,
                                "relation": ID2RELATION.get(gt_rel_id, "none") if gt_rel_id >= 0 else "none",
                            })
                            pred_doc.append({
                                "class": self.id2label.get(pred_class_id, f"cls_{pred_class_id}"),
                                "text": f"line_{line_id}",
                                "parent_id": pred_parent_idx,
                                "relation": ID2RELATION.get(pred_rel_id, "none"),
                            })
                        if gt_doc and pred_doc:
                            teds_gt_docs.append(gt_doc)
                            teds_pred_docs.append(pred_doc)

        # 打印调试信息
        if debug or verbose:
            # 预测类别统计（title/section 等）
            from collections import Counter
            pred_class_counter = Counter(all_pred_classes)
            gt_class_counter = Counter(all_gt_classes)
            print(f"\n[Evaluator Debug] Prediction class distribution:")
            for cls_id in sorted(set(pred_class_counter.keys()) | set(gt_class_counter.keys())):
                cls_name = self.id2label.get(cls_id, f"cls_{cls_id}")
                gt_cnt = gt_class_counter.get(cls_id, 0)
                pred_cnt = pred_class_counter.get(cls_id, 0)
                diff = pred_cnt - gt_cnt
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                print(f"  {cls_name}: GT={gt_cnt}, Pred={pred_cnt} ({diff_str})")

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

                # 打印 Section 详细统计表格
                self._print_section_stats(stats, gt_class_counter, pred_class_counter)

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

        # 计算 TEDS 分数
        if compute_teds and TEDS_AVAILABLE and teds_gt_docs and teds_pred_docs:
            try:
                print(f"\n[Evaluator] Computing TEDS for {len(teds_gt_docs)} documents...")
                teds_score = compute_teds_score(teds_gt_docs, teds_pred_docs)
                if teds_score is not None:
                    output.teds_score = teds_score
                    print(f"[Evaluator] TEDS Score: {teds_score:.4f}")
            except Exception as e:
                print(f"[Evaluator] TEDS computation failed: {e}")
        elif compute_teds and not TEDS_AVAILABLE:
            print("[Evaluator] TEDS computation skipped: util/hrdoc_eval.py not available")

        # 保存预测结果（按文档保存到时间戳目录）
        if save_predictions and all_predictions:
            import json
            from datetime import datetime

            # 创建时间戳目录: runs/{timestamp}/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            runs_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs")
            save_dir = os.path.join(runs_dir, timestamp)
            os.makedirs(save_dir, exist_ok=True)

            # 按文档保存
            for doc_pred in all_predictions:
                doc_name = doc_pred["document_name"]
                doc_lines = doc_pred["lines"]
                output_file = os.path.join(save_dir, f"{doc_name}_infer.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(doc_lines, f, ensure_ascii=False, indent=2)

            print(f"\n[Evaluator] Predictions saved to: {save_dir}/ ({len(all_predictions)} documents)")

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

    def _print_section_stats(self, stats: List[Dict], gt_class_counter: Dict, pred_class_counter: Dict) -> None:
        """
        打印 Section 类别的详细统计表格

        包括：
        1. Section 分类统计
        2. Section Parent 准确率
        3. Section 错误详情
        """
        SECTION_ID = LABEL2ID.get("section", 2)

        # 筛选 section 相关的统计
        section_stats = [s for s in stats if s["child_class"] == SECTION_ID]
        if not section_stats:
            return

        # 计算统计数据
        section_total = len(section_stats)
        section_correct = sum(1 for s in section_stats if s["is_correct"])
        section_acc = 100 * section_correct / section_total if section_total > 0 else 0

        # 错误分析
        section_errors = [s for s in section_stats if not s["is_correct"]]
        error_by_pred_class = defaultdict(int)
        for e in section_errors:
            pred_cls = e["pred_parent_class"]
            pred_name = self.id2label.get(pred_cls, "ROOT") if pred_cls is not None else "ROOT"
            error_by_pred_class[pred_name] += 1

        # GT/Pred 类别统计
        gt_section_count = gt_class_counter.get(SECTION_ID, 0)
        pred_section_count = pred_class_counter.get(SECTION_ID, 0)

        # 按 gt_parent_class 分组统计
        parent_class_stats = defaultdict(lambda: {"correct": 0, "total": 0, "errors": defaultdict(int)})
        for s in section_stats:
            gt_p_cls = s["gt_parent_class"]
            gt_p_name = self.id2label.get(gt_p_cls, "ROOT") if gt_p_cls is not None else "ROOT"
            parent_class_stats[gt_p_name]["total"] += 1
            if s["is_correct"]:
                parent_class_stats[gt_p_name]["correct"] += 1
            else:
                pred_p_cls = s["pred_parent_class"]
                pred_p_name = self.id2label.get(pred_p_cls, "ROOT") if pred_p_cls is not None else "ROOT"
                parent_class_stats[gt_p_name]["errors"][pred_p_name] += 1

        # 打印表格
        print("\n" + "=" * 70)
        print("  📊 SECTION 类别详细统计")
        print("=" * 70)

        # 1. 分类统计
        print("\n┌─────────────────────────────────────────────────────────────────────┐")
        print("│ 1. Section 分类统计                                                 │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        diff = pred_section_count - gt_section_count
        diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "0"
        print(f"│   GT Section 数量:    {gt_section_count:<6}                                       │")
        print(f"│   Pred Section 数量:  {pred_section_count:<6} ({diff_str})                                     │")
        print("└─────────────────────────────────────────────────────────────────────┘")

        # 2. Parent 准确率
        print("\n┌─────────────────────────────────────────────────────────────────────┐")
        print("│ 2. Section Parent 预测准确率                                        │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        bar_len = 30
        filled = int(bar_len * section_acc / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"│   准确率: {section_acc:5.1f}%  [{bar}]  {section_correct}/{section_total}    │")
        print("└─────────────────────────────────────────────────────────────────────┘")

        # 3. 按 GT Parent 类型分组统计
        print("\n┌─────────────────────────────────────────────────────────────────────┐")
        print("│ 3. Section Parent 按 GT Parent 类型分组                             │")
        print("├───────────────┬──────────┬───────────┬───────────────────────────────┤")
        print("│ GT Parent     │ 正确/总数 │ 准确率    │ 误判分布                      │")
        print("├───────────────┼──────────┼───────────┼───────────────────────────────┤")

        for gt_p_name in sorted(parent_class_stats.keys()):
            pstat = parent_class_stats[gt_p_name]
            p_acc = 100 * pstat["correct"] / pstat["total"] if pstat["total"] > 0 else 0
            count_str = f"{pstat['correct']}/{pstat['total']}"

            # 错误分布
            if pstat["errors"]:
                err_list = [f"{k}:{v}" for k, v in sorted(pstat["errors"].items(), key=lambda x: -x[1])]
                err_str = ", ".join(err_list)
            else:
                err_str = "-"

            print(f"│ {gt_p_name:<13} │ {count_str:<8} │ {p_acc:>6.1f}%   │ {err_str:<29} │")

        print("└───────────────┴──────────┴───────────┴───────────────────────────────┘")

        # 4. 错误汇总
        if section_errors:
            print("\n┌─────────────────────────────────────────────────────────────────────┐")
            print("│ 4. Section Parent 错误汇总                                          │")
            print("├─────────────────────────────────────────────────────────────────────┤")
            print(f"│   总错误数: {len(section_errors):<6}                                              │")
            print("│   误判为:                                                           │")
            for pred_name, cnt in sorted(error_by_pred_class.items(), key=lambda x: -x[1]):
                pct = 100 * cnt / len(section_errors)
                print(f"│     - {pred_name:<10}: {cnt:>3} ({pct:>5.1f}%)                                    │")
            print("└─────────────────────────────────────────────────────────────────────┘")

            # 5. 错误详情（带文本）
            self._print_section_error_details(section_errors)

        print("=" * 70 + "\n")

    def _print_section_error_details(self, section_errors: List[Dict]) -> None:
        """
        打印 Section 错误详情（带文本信息）
        """
        import json

        # 加载原始数据缓存
        doc_line_texts = {}  # {doc_name: {line_id: text}}

        def load_doc_texts(doc_name: str) -> Dict[int, str]:
            """从原始 JSON 加载文档的行文本"""
            if doc_name in doc_line_texts:
                return doc_line_texts[doc_name]

            line_texts = {}
            json_path = self._doc_json_paths.get(doc_name)

            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 处理多页面格式（pages 数组）
                    if "pages" in data:
                        for page in data["pages"]:
                            items = page.get("items", page.get("lines", []))
                            for item in items:
                                line_id = item.get("line_id", item.get("id"))
                                if line_id is not None:
                                    words = item.get("words", [])
                                    if words:
                                        text = " ".join(w.get("text", "") for w in words)
                                    else:
                                        text = item.get("text", "")
                                    line_texts[line_id] = text[:50]  # 截断
                    else:
                        # 单页面格式
                        items = data.get("items", data.get("lines", []))
                        for item in items:
                            line_id = item.get("line_id", item.get("id"))
                            if line_id is not None:
                                words = item.get("words", [])
                                if words:
                                    text = " ".join(w.get("text", "") for w in words)
                                else:
                                    text = item.get("text", "")
                                line_texts[line_id] = text[:50]
                except Exception as e:
                    pass  # 忽略加载错误

            doc_line_texts[doc_name] = line_texts
            return line_texts

        print("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────┐")
        print("│ 5. Section Parent 错误详情                                                                      │")
        print("├─────────────────────────────────────────────────────────────────────────────────────────────────┤")

        for i, err in enumerate(section_errors[:10]):  # 只显示前 10 个
            doc_name = err.get("document_name", "unknown")
            json_path = self._doc_json_paths.get(doc_name, "")
            line_texts = load_doc_texts(doc_name)

            child_line_id = err.get("child_line_id", -1)
            gt_parent_line_id = err.get("gt_parent_line_id", -1)
            pred_parent_line_id = err.get("pred_parent_line_id", -1)

            child_text = line_texts.get(child_line_id, "N/A")
            gt_parent_text = line_texts.get(gt_parent_line_id, "ROOT" if gt_parent_line_id == -1 else "N/A")
            pred_parent_text = line_texts.get(pred_parent_line_id, "ROOT" if pred_parent_line_id == -1 else "N/A")

            gt_p_cls = err.get("gt_parent_class")
            pred_p_cls = err.get("pred_parent_class")
            gt_p_name = self.id2label.get(gt_p_cls, "ROOT") if gt_p_cls is not None else "ROOT"
            pred_p_name = self.id2label.get(pred_p_cls, "ROOT") if pred_p_cls is not None else "ROOT"

            print(f"│ [{i+1}] 文档: {doc_name}")
            if json_path:
                print(f"│     文件: {json_path}")
            print(f"│     当前行 (id={child_line_id}): \"{child_text}\"")
            print(f"│     ✓ GT Parent   (id={gt_parent_line_id}, {gt_p_name}): \"{gt_parent_text}\"")
            print(f"│     ✗ Pred Parent (id={pred_parent_line_id}, {pred_p_name}): \"{pred_parent_text}\"")
            if i < len(section_errors) - 1 and i < 9:
                print("│" + "─" * 97 + "│")

        if len(section_errors) > 10:
            print(f"│     ... 还有 {len(section_errors) - 10} 个错误未显示                                                        │")

        print("└─────────────────────────────────────────────────────────────────────────────────────────────────┘")

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

        if sample.line_ids is None:
            return gt

        # 提取 line_ids 和 labels（展平处理多 chunk 情况）
        if sample.is_document_level:
            all_line_ids = sample.line_ids.reshape(-1).cpu().tolist()
            all_labels = sample.labels.reshape(-1).cpu().tolist() if sample.labels is not None else []
        else:
            all_line_ids = sample.line_ids.cpu().tolist()
            all_labels = sample.labels.cpu().tolist() if sample.labels is not None else []

        # 提取 sorted_line_ids（与 predictor 的 LinePooling.get_line_ids_mapping 一致）
        unique_line_ids = sorted(set(lid for lid in all_line_ids if lid >= 0))

        # 提取分类 GT
        if sample.line_labels is not None:
            # 优先使用 line_labels（行索引直接对应）
            labels = sample.line_labels.cpu().tolist()
            for line_idx, label in enumerate(labels):
                if label >= 0 and label != -100:
                    # line_labels 按行序号索引，需要映射到 line_id
                    if line_idx < len(unique_line_ids):
                        line_id = unique_line_ids[line_idx]
                        gt["classes"][line_id] = label
        elif all_labels:
            # Fallback: 从 token labels 提取（首次出现策略）
            for label, line_id in zip(all_labels, all_line_ids):
                if line_id >= 0 and label >= 0 and line_id not in gt["classes"]:
                    gt["classes"][line_id] = label

        # 处理 parent_ids 和 relations
        # 重要：
        # 1. sample.line_parent_ids 按行序号索引（0, 1, 2, ...），不是按 line_id 索引
        # 2. sample.line_parent_ids 的值是 parent 的 line_id，需要转换为行序号
        # 3. pred.line_parents 是行序号，所以 GT 也要用行序号表示

        # 建立 line_id -> 行序号 的映射（使用 sorted 顺序，与 predictor 一致）
        line_id_to_row = {lid: row for row, lid in enumerate(unique_line_ids)}

        if sample.line_parent_ids is not None:
            raw_parents = sample.line_parent_ids.cpu().tolist()
            # 按行序号顺序提取，并将 parent_line_id 转换为行序号
            for row in range(len(unique_line_ids)):
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
            for row in range(len(unique_line_ids)):
                if row < len(raw_relations):
                    gt["relations"].append(raw_relations[row])
                else:
                    gt["relations"].append(-100)

        # 存储排序后的 line_ids（与 predictor 一致）
        gt["line_ids"] = unique_line_ids

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
            output.relation_micro_f1 = self._micro_f1(gt_relations, pred_relations)

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

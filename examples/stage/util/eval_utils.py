#!/usr/bin/env python
# coding=utf-8
"""
评估工具函数

提供训练过程中评估的公共逻辑，供所有 Stage 使用。
"""

import logging
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    """计算准确率"""
    return accuracy_score(labels, predictions)


def compute_macro_f1(
    predictions: List[int],
    labels: List[int],
    num_classes: Optional[int] = None,
    exclude_empty_classes: bool = True
) -> Tuple[float, Dict[int, Dict[str, float]]]:
    """
    计算 Macro F1 和每类指标

    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        num_classes: 类别数量（如果为 None，从数据推断）
        exclude_empty_classes: 是否排除无样本的类别

    Returns:
        (macro_f1, per_class_metrics): Macro F1 值和每类指标字典
    """
    if num_classes is None:
        num_classes = max(max(predictions), max(labels)) + 1

    # 统计每类的 TP, FP, FN
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0})

    for pred, gt in zip(predictions, labels):
        class_stats[gt]["gt_count"] += 1
        class_stats[pred]["pred_count"] += 1

        if pred == gt:
            class_stats[gt]["tp"] += 1
        else:
            class_stats[gt]["fn"] += 1
            class_stats[pred]["fp"] += 1

    # 计算每类的 P/R/F1
    per_class_metrics = {}
    f1_scores = []

    for cls in range(num_classes):
        stats = class_stats[cls]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        gt_count = stats["gt_count"]
        pred_count = stats["pred_count"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gt_count": gt_count,
            "pred_count": pred_count,
        }

        # 是否计入 macro F1
        if exclude_empty_classes:
            if gt_count > 0 or pred_count > 0:
                f1_scores.append(f1)
        else:
            f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0

    return macro_f1, per_class_metrics


def compute_multiclass_metrics(
    predictions: List[int],
    labels: List[int],
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    计算多分类任务的完整指标

    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        class_names: 类别名称列表

    Returns:
        指标字典，包含 accuracy, macro_p, macro_r, macro_f1, 混淆矩阵等
    """
    # 基础指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }

    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    metrics["confusion_matrix"] = cm

    # 分类报告
    if class_names:
        report = classification_report(
            labels, predictions,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )
        metrics["classification_report"] = report

    return metrics


def log_per_class_metrics(
    per_class_metrics: Dict[int, Dict[str, float]],
    class_names: Optional[List[str]] = None,
    title: str = "Per-Class Metrics"
):
    """
    打印每类指标的格式化日志

    Args:
        per_class_metrics: 每类指标字典
        class_names: 类别名称列表
        title: 标题
    """
    logger.info("=" * 60)
    logger.info(f"{title}:")
    logger.info(f"{'Class':<15} {'Prec':>7} {'Recall':>7} {'F1':>7} {'GT':>6} {'Pred':>6}")
    logger.info("-" * 55)

    for cls_id, metrics in sorted(per_class_metrics.items()):
        if metrics["gt_count"] > 0 or metrics["pred_count"] > 0:
            cls_name = class_names[cls_id] if class_names else str(cls_id)
            logger.info(
                f"{cls_name:<15} "
                f"{metrics['precision']:>7.1%} "
                f"{metrics['recall']:>7.1%} "
                f"{metrics['f1']:>7.1%} "
                f"{metrics['gt_count']:>6} "
                f"{metrics['pred_count']:>6}"
            )

    logger.info("=" * 60)


def log_evaluation_summary(
    metrics: Dict[str, float],
    split_name: str = "Eval",
    extra_info: Optional[Dict[str, any]] = None
):
    """
    打印评估摘要日志

    Args:
        metrics: 指标字典
        split_name: 数据集名称（Train/Eval/Test）
        extra_info: 额外信息
    """
    logger.info("-" * 40)
    logger.info(f"{split_name} Results:")

    # 打印主要指标
    main_metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "loss"]
    for key in main_metrics:
        if key in metrics:
            if key == "loss":
                logger.info(f"  {key}: {metrics[key]:.4f}")
            else:
                logger.info(f"  {key}: {metrics[key]:.1%}")

    # 打印额外信息
    if extra_info:
        for key, value in extra_info.items():
            logger.info(f"  {key}: {value}")

    logger.info("-" * 40)


class EvaluationTracker:
    """
    评估追踪器

    在训练过程中追踪指标，支持 early stopping 和 best model 选择。
    """

    def __init__(
        self,
        metric_name: str = "macro_f1",
        greater_is_better: bool = True,
        patience: int = 5
    ):
        """
        Args:
            metric_name: 追踪的指标名称
            greater_is_better: 指标是否越大越好
            patience: early stopping 耐心值
        """
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.patience = patience

        self.best_metric = None
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.history = []

    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        更新追踪器

        Args:
            epoch: 当前 epoch
            metrics: 当前指标

        Returns:
            是否是新的最佳结果
        """
        current_metric = metrics.get(self.metric_name, 0.0)
        self.history.append({"epoch": epoch, **metrics})

        is_best = False

        if self.best_metric is None:
            is_best = True
        elif self.greater_is_better and current_metric > self.best_metric:
            is_best = True
        elif not self.greater_is_better and current_metric < self.best_metric:
            is_best = True

        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return is_best

    def should_stop_early(self) -> bool:
        """检查是否应该 early stopping"""
        return self.epochs_without_improvement >= self.patience

    def get_best_info(self) -> Dict:
        """获取最佳结果信息"""
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "metric_name": self.metric_name,
        }

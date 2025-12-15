# coding=utf-8
"""
行级别评估模块 (Single Source of Truth)

本模块是所有行级别评估的唯一实现，包括：
1. Token → Line 聚合逻辑
2. 行级别指标计算（Accuracy, Macro-F1, Per-class metrics）

设计原则：
- 评估粒度：Line 级别（与 HRDoc 论文一致）
- 聚合方法：多数投票（每个 token 投一票，票数相同时取先出现的类别）
- Macro-F1：包含所有 14 类，无忽略类

使用方式：
    from metrics import compute_line_level_metrics_from_tokens

    result = compute_line_level_metrics_from_tokens(
        token_predictions=preds,  # List[int], shape [seq_len]
        token_labels=labels,      # List[int], shape [seq_len], -100 表示忽略
        line_ids=line_ids,        # List[int], shape [seq_len], -1 表示无效
        num_classes=14,
        class_names=LABEL_LIST,
    )
    print(f"Line Accuracy: {result.accuracy:.2%}")
    print(f"Line Macro-F1: {result.macro_f1:.2%}")
"""

import logging
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class LineMetricsResult:
    """行级别评估结果"""
    # 主要指标
    accuracy: float = 0.0
    macro_f1: float = 0.0
    micro_f1: float = 0.0

    # 统计信息
    num_lines: int = 0
    num_classes: int = 14

    # 每类详细指标
    per_class_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # 原始预测和标签（用于进一步分析）
    line_predictions: List[int] = field(default_factory=list)
    line_labels: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        """转换为字典（用于 logging/wandb）"""
        result = {
            "line_accuracy": self.accuracy,
            "line_macro_f1": self.macro_f1,
            "line_micro_f1": self.micro_f1,
            "num_lines": self.num_lines,
        }
        # 添加每类 F1
        for cls_id, metrics in self.per_class_metrics.items():
            result[f"line_f1_class_{cls_id}"] = metrics.get("f1", 0.0)
        return result

    def log_summary(self, class_names: Optional[List[str]] = None, title: str = "Line-Level Metrics"):
        """打印格式化的评估摘要"""
        logger.info("=" * 65)
        logger.info(f"{title}")
        logger.info("=" * 65)
        logger.info(f"  Accuracy:  {self.accuracy:>7.2%}")
        logger.info(f"  Macro-F1:  {self.macro_f1:>7.2%}")
        logger.info(f"  Micro-F1:  {self.micro_f1:>7.2%}")
        logger.info(f"  Num Lines: {self.num_lines:>7}")
        logger.info("-" * 65)

        if self.per_class_metrics:
            logger.info(f"{'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'GT':>7} {'Pred':>7}")
            logger.info("-" * 65)

            for cls_id in sorted(self.per_class_metrics.keys()):
                m = self.per_class_metrics[cls_id]
                if m["gt_count"] > 0 or m["pred_count"] > 0:
                    cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
                    logger.info(
                        f"{cls_name:<15} "
                        f"{m['precision']:>8.2%} "
                        f"{m['recall']:>8.2%} "
                        f"{m['f1']:>8.2%} "
                        f"{m['gt_count']:>7} "
                        f"{m['pred_count']:>7}"
                    )

        logger.info("=" * 65)


# ==============================================================================
# Token → Line 聚合函数
# ==============================================================================

def aggregate_token_to_line_predictions(
    token_predictions: List[int],
    line_ids: List[int],
    method: str = "majority",
    token_labels: Optional[List[int]] = None,
    debug: bool = False,
) -> Dict[int, int]:
    """
    将 token-level 预测聚合为 line-level 预测

    这是统一的聚合函数，训练评估和推理都应该使用这个函数。

    Args:
        token_predictions: 每个 token 的预测标签
        line_ids: 每个 token 对应的 line_id（-1 表示无效 token，如 [CLS]/[SEP]/PAD）
        method: 聚合方法
            - "majority": 多数投票（推荐，每个 token 一票）
            - "first": 取首个 token（不推荐，仅用于兼容旧代码）
        token_labels: [诊断用] token 级别的 GT 标签，用于统计 label=-100 的 token 参与投票情况
        debug: 是否打印诊断日志

    Returns:
        Dict[line_id, predicted_label]: 每行的预测标签

    Note:
        - 多数投票时，票数相同则取先出现的类别（Counter.most_common 行为）
        - 这与 sklearn 的行为一致，无额外的 tie-breaking 逻辑
    """
    if method == "majority":
        # 收集每行所有 token 的预测
        line_pred_tokens = defaultdict(list)

        # [诊断] 统计投票情况
        total_tokens = len(token_predictions)
        voted_tokens = 0
        ignored_tokens = 0  # line_id < 0 的 token
        label_minus100_voted = 0  # label=-100 但参与了投票的 token

        for i, (pred, line_id) in enumerate(zip(token_predictions, line_ids)):
            if line_id >= 0:
                line_pred_tokens[line_id].append(pred)
                voted_tokens += 1
                # 检查这个 token 的 label 是否是 -100
                if token_labels is not None and i < len(token_labels):
                    if token_labels[i] == -100:
                        label_minus100_voted += 1
            else:
                ignored_tokens += 1

        # 多数投票
        line_predictions = {}
        for line_id, preds in line_pred_tokens.items():
            if preds:
                line_predictions[line_id] = Counter(preds).most_common(1)[0][0]

        # [诊断日志] 打印投票统计
        if debug:
            logger.info(f"[DIAG aggregate_token_to_line] "
                       f"total_tokens={total_tokens}, voted={voted_tokens}, "
                       f"ignored(line_id<0)={ignored_tokens}, "
                       f"label=-100_but_voted={label_minus100_voted}, "
                       f"num_lines={len(line_predictions)}")
            if token_labels is not None and voted_tokens > 0:
                pct = label_minus100_voted / voted_tokens * 100
                logger.info(f"[DIAG] ⚠️  {pct:.1f}% of voted tokens have label=-100 (未被监督但参与投票)")

        return line_predictions

    elif method == "first":
        # 取首个 token（不推荐，仅用于兼容）
        line_predictions = {}
        for pred, line_id in zip(token_predictions, line_ids):
            if line_id >= 0 and line_id not in line_predictions:
                line_predictions[line_id] = pred
        return line_predictions

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def extract_line_labels_from_tokens(
    token_labels: List[int],
    line_ids: List[int]
) -> Dict[int, int]:
    """
    从 token-level GT 标签中提取 line-level 标签

    原始数据本来就是 line-level 的，tokenize 时被展开成 token-level，
    同一行的所有 token 标签相同。这里只是还原回 line-level。

    Args:
        token_labels: 每个 token 的 GT 标签（-100 表示忽略，如 subword 非首 token）
        line_ids: 每个 token 对应的 line_id（-1 表示无效）

    Returns:
        Dict[line_id, gt_label]: 每行的 GT 标签
    """
    line_labels = {}
    for label, line_id in zip(token_labels, line_ids):
        # 只取首个有效标签（同行标签本来就相同）
        if line_id >= 0 and label >= 0 and line_id not in line_labels:
            line_labels[line_id] = label
    return line_labels


# ==============================================================================
# 指标计算函数
# ==============================================================================

def compute_line_level_metrics(
    line_predictions: List[int],
    line_labels: List[int],
    num_classes: int = 14,
    class_names: Optional[List[str]] = None,
) -> LineMetricsResult:
    """
    计算行级别指标

    Args:
        line_predictions: 行级别预测列表
        line_labels: 行级别 GT 列表
        num_classes: 类别数量（默认 14，HRDoc 标准）
        class_names: 类别名称列表（用于日志）

    Returns:
        LineMetricsResult: 包含所有指标的结果对象
    """
    if len(line_predictions) == 0 or len(line_labels) == 0:
        return LineMetricsResult()

    # 基础指标
    accuracy = accuracy_score(line_labels, line_predictions)
    macro_f1 = f1_score(line_labels, line_predictions, average="macro", zero_division=0)
    micro_f1 = f1_score(line_labels, line_predictions, average="micro", zero_division=0)

    # 每类指标
    per_class_metrics = _compute_per_class_metrics(
        line_predictions, line_labels, num_classes
    )

    return LineMetricsResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        num_lines=len(line_labels),
        num_classes=num_classes,
        per_class_metrics=per_class_metrics,
        line_predictions=list(line_predictions),
        line_labels=list(line_labels),
    )


def compute_line_level_metrics_from_tokens(
    token_predictions: List[int],
    token_labels: List[int],
    line_ids: List[int],
    num_classes: int = 14,
    class_names: Optional[List[str]] = None,
    aggregation_method: str = "majority",
) -> LineMetricsResult:
    """
    从 token-level 预测计算行级别指标（主入口函数）

    这是推荐使用的主入口，封装了聚合和指标计算的完整流程。

    Args:
        token_predictions: token 级别预测，shape [seq_len]
        token_labels: token 级别 GT，shape [seq_len]，-100 表示忽略
        line_ids: 每个 token 的 line_id，shape [seq_len]，-1 表示无效
        num_classes: 类别数量
        class_names: 类别名称列表
        aggregation_method: 聚合方法，默认 "majority"

    Returns:
        LineMetricsResult: 包含所有指标的结果对象

    Example:
        >>> from metrics import compute_line_level_metrics_from_tokens
        >>> result = compute_line_level_metrics_from_tokens(
        ...     token_predictions=[0, 0, 1, 1, 1, 2],
        ...     token_labels=[0, -100, 1, 1, -100, 2],
        ...     line_ids=[0, 0, 1, 1, 1, 2],
        ... )
        >>> print(f"Accuracy: {result.accuracy:.2%}")
    """
    # Step 1: 聚合 token → line
    line_pred_dict = aggregate_token_to_line_predictions(
        token_predictions, line_ids, method=aggregation_method
    )
    line_gt_dict = extract_line_labels_from_tokens(token_labels, line_ids)

    # Step 2: 对齐预测和 GT（只保留两者都有的 line_id）
    common_line_ids = sorted(set(line_pred_dict.keys()) & set(line_gt_dict.keys()))

    line_predictions = [line_pred_dict[lid] for lid in common_line_ids]
    line_labels = [line_gt_dict[lid] for lid in common_line_ids]

    # Step 3: 计算指标
    return compute_line_level_metrics(
        line_predictions=line_predictions,
        line_labels=line_labels,
        num_classes=num_classes,
        class_names=class_names,
    )


def compute_line_level_metrics_batch(
    batch_token_predictions: List[List[int]],
    batch_token_labels: List[List[int]],
    batch_line_ids: List[List[int]],
    num_classes: int = 14,
    class_names: Optional[List[str]] = None,
    aggregation_method: str = "majority",
) -> LineMetricsResult:
    """
    批量计算行级别指标

    将多个样本的预测合并后计算指标。

    Args:
        batch_token_predictions: 批量 token 预测，List of [seq_len]
        batch_token_labels: 批量 token GT，List of [seq_len]
        batch_line_ids: 批量 line_ids，List of [seq_len]
        num_classes: 类别数量
        class_names: 类别名称列表
        aggregation_method: 聚合方法

    Returns:
        LineMetricsResult: 合并后的指标结果
    """
    all_line_predictions = []
    all_line_labels = []

    for token_preds, token_labels, line_ids in zip(
        batch_token_predictions, batch_token_labels, batch_line_ids
    ):
        # 聚合每个样本
        line_pred_dict = aggregate_token_to_line_predictions(
            token_preds, line_ids, method=aggregation_method
        )
        line_gt_dict = extract_line_labels_from_tokens(token_labels, line_ids)

        # 对齐
        common_line_ids = sorted(set(line_pred_dict.keys()) & set(line_gt_dict.keys()))

        all_line_predictions.extend([line_pred_dict[lid] for lid in common_line_ids])
        all_line_labels.extend([line_gt_dict[lid] for lid in common_line_ids])

    return compute_line_level_metrics(
        line_predictions=all_line_predictions,
        line_labels=all_line_labels,
        num_classes=num_classes,
        class_names=class_names,
    )


# ==============================================================================
# 内部辅助函数
# ==============================================================================

def _compute_per_class_metrics(
    predictions: List[int],
    labels: List[int],
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    """
    计算每类的 Precision/Recall/F1

    Args:
        predictions: 预测列表
        labels: GT 列表
        num_classes: 类别数量

    Returns:
        Dict[class_id, {"precision", "recall", "f1", "gt_count", "pred_count"}]
    """
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

    return per_class_metrics

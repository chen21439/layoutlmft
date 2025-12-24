"""通用评估工具

包含可复用的评估工具函数。
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import json
import numpy as np


def format_metrics(metrics: Dict[str, Any], precision: int = 4) -> str:
    """格式化指标为可读字符串

    Args:
        metrics: 指标字典
        precision: 浮点数精度

    Returns:
        格式化后的字符串
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.{precision}f}")
        elif isinstance(value, dict):
            lines.append(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.{precision}f}")
                else:
                    lines.append(f"    {k}: {v}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def save_metrics(metrics: Dict[str, Any], path: str):
    """保存指标到 JSON 文件

    Args:
        metrics: 指标字典
        path: 保存路径
    """
    # 转换 numpy 类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(convert(metrics), f, indent=2, ensure_ascii=False)


def load_metrics(path: str) -> Dict[str, Any]:
    """从 JSON 文件加载指标

    Args:
        path: 文件路径

    Returns:
        指标字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class MetricAggregator:
    """指标聚合器

    用于聚合多个批次的指标。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置状态"""
        self.values = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        """更新指标

        Args:
            metrics: 本批次指标
            count: 样本数
        """
        for key, value in metrics.items():
            if key not in self.values:
                self.values[key] = 0.0
                self.counts[key] = 0
            self.values[key] += value * count
            self.counts[key] += count

    def compute(self) -> Dict[str, float]:
        """计算平均指标"""
        result = {}
        for key in self.values:
            if self.counts[key] > 0:
                result[key] = self.values[key] / self.counts[key]
            else:
                result[key] = 0.0
        return result


def compute_pairwise_accuracy(
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
    mask: np.ndarray = None,
) -> Dict[str, float]:
    """计算 pairwise 预测的准确率

    用于评估阅读顺序预测。

    Args:
        pred_matrix: [N, N] 预测的 pairwise 关系矩阵
        gt_matrix: [N, N] 真实的 pairwise 关系矩阵
        mask: [N, N] 有效位置掩码

    Returns:
        准确率指标字典
    """
    if mask is None:
        mask = np.ones_like(pred_matrix, dtype=bool)

    # 排除对角线
    np.fill_diagonal(mask, False)

    valid = mask.sum()
    if valid == 0:
        return {'accuracy': 0.0, 'total': 0}

    correct = ((pred_matrix > 0) == (gt_matrix > 0)) & mask
    accuracy = correct.sum() / valid

    return {
        'accuracy': float(accuracy),
        'correct': int(correct.sum()),
        'total': int(valid),
    }


def compute_successor_accuracy(
    pred_successors: List[int],
    gt_successors: List[int],
    mask: List[bool] = None,
) -> Dict[str, float]:
    """计算后继预测准确率

    Args:
        pred_successors: 预测的后继索引
        gt_successors: 真实的后继索引
        mask: 有效位置掩码

    Returns:
        准确率指标字典
    """
    correct = 0
    total = 0

    for i, (pred, gt) in enumerate(zip(pred_successors, gt_successors)):
        if mask and not mask[i]:
            continue
        if gt >= 0:  # 有后继的位置
            total += 1
            if pred == gt:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }


@dataclass
class EvaluationReport:
    """评估报告"""
    # Detect 模块
    detect_classification_macro_f1: float = 0.0
    detect_classification_micro_f1: float = 0.0
    detect_intra_order_accuracy: float = 0.0

    # Order 模块
    order_main_macro_teds: float = 0.0
    order_main_micro_teds: float = 0.0
    order_floating_macro_teds: float = 0.0
    order_floating_micro_teds: float = 0.0

    # Construct 模块
    construct_macro_teds: float = 0.0
    construct_micro_teds: float = 0.0

    # 汇总
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def __str__(self) -> str:
        """格式化输出"""
        lines = [
            "=" * 60,
            "Detect-Order-Construct 评估报告",
            "=" * 60,
            "",
            "[Detect 模块]",
            f"  分类 Macro F1: {self.detect_classification_macro_f1:.4f}",
            f"  分类 Micro F1: {self.detect_classification_micro_f1:.4f}",
            f"  区域内顺序准确率: {self.detect_intra_order_accuracy:.4f}",
            "",
            "[Order 模块]",
            f"  主体文本 Macro TEDS: {self.order_main_macro_teds:.4f}",
            f"  主体文本 Micro TEDS: {self.order_main_micro_teds:.4f}",
            f"  浮动元素 Macro TEDS: {self.order_floating_macro_teds:.4f}",
            f"  浮动元素 Micro TEDS: {self.order_floating_micro_teds:.4f}",
            "",
            "[Construct 模块]",
            f"  层级结构 Macro TEDS: {self.construct_macro_teds:.4f}",
            f"  层级结构 Micro TEDS: {self.construct_micro_teds:.4f}",
            "",
            f"样本数: {self.num_samples}",
            "=" * 60,
        ]
        return "\n".join(lines)

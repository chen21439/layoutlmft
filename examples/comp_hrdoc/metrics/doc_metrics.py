"""DOC模型评估指标

包含 Detect-Order-Construct 三个模块的评估指标计算。

4.2 Detect: 语义分类 (Accuracy, F1)
4.3 Order: 阅读顺序/后继预测 (Accuracy, F1)
4.4 Construct: 层级结构 (Parent Acc, Sibling Acc, Root Acc, F1)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


@dataclass
class DetectMetrics:
    """4.2 Detect 模块指标"""
    accuracy: float = 0.0
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    weighted_f1: float = 0.0
    per_class_f1: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        result = {
            'cls_accuracy': self.accuracy,
            'cls_macro_f1': self.macro_f1,
            'cls_micro_f1': self.micro_f1,
            'cls_weighted_f1': self.weighted_f1,
        }
        for cls_id, f1 in self.per_class_f1.items():
            result[f'cls_f1_class_{cls_id}'] = f1
        return result


@dataclass
class OrderMetrics:
    """4.3 Order 模块指标"""
    accuracy: float = 0.0
    f1: float = 0.0
    correct: int = 0
    total: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            'order_accuracy': self.accuracy,
            'order_f1': self.f1,
            'order_correct': self.correct,
            'order_total': self.total,
        }


@dataclass
class ConstructMetrics:
    """4.4 Construct 模块指标"""
    parent_accuracy: float = 0.0
    parent_f1: float = 0.0
    sibling_accuracy: float = 0.0
    sibling_f1: float = 0.0
    root_accuracy: float = 0.0
    root_f1: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'parent_accuracy': self.parent_accuracy,
            'parent_f1': self.parent_f1,
            'sibling_accuracy': self.sibling_accuracy,
            'sibling_f1': self.sibling_f1,
            'root_accuracy': self.root_accuracy,
            'root_f1': self.root_f1,
        }


@dataclass
class DOCMetrics:
    """DOC模型完整指标"""
    detect: DetectMetrics = field(default_factory=DetectMetrics)
    order: OrderMetrics = field(default_factory=OrderMetrics)
    construct: ConstructMetrics = field(default_factory=ConstructMetrics)

    def to_dict(self) -> Dict[str, float]:
        result = {}
        result.update(self.detect.to_dict())
        result.update(self.order.to_dict())
        result.update(self.construct.to_dict())
        return result

    def summary(self) -> str:
        """生成摘要字符串"""
        lines = [
            "=" * 60,
            "DOC Model Evaluation Results",
            "=" * 60,
            "",
            "[4.2 Detect - Semantic Classification]",
            f"  Accuracy:    {self.detect.accuracy:.4f}",
            f"  Macro F1:    {self.detect.macro_f1:.4f}",
            f"  Micro F1:    {self.detect.micro_f1:.4f}",
            f"  Weighted F1: {self.detect.weighted_f1:.4f}",
            "",
            "[4.3 Order - Reading Order]",
            f"  Accuracy: {self.order.accuracy:.4f}",
            f"  F1:       {self.order.f1:.4f}",
            f"  Correct:  {self.order.correct} / {self.order.total}",
            "",
            "[4.4 Construct - Hierarchical Structure]",
            f"  Parent Accuracy:  {self.construct.parent_accuracy:.4f}",
            f"  Parent F1:        {self.construct.parent_f1:.4f}",
            f"  Sibling Accuracy: {self.construct.sibling_accuracy:.4f}",
            f"  Sibling F1:       {self.construct.sibling_f1:.4f}",
            f"  Root Accuracy:    {self.construct.root_accuracy:.4f}",
            f"  Root F1:          {self.construct.root_f1:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class DOCMetricsComputer:
    """DOC模型指标计算器

    收集batch预测结果，最终计算汇总指标。
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置所有收集的数据"""
        # Detect (4.2)
        self.cls_preds = []
        self.cls_labels = []

        # Order (4.3)
        self.order_preds = []
        self.order_labels = []

        # Construct (4.4)
        self.parent_preds = []
        self.parent_labels = []
        self.sibling_preds = []
        self.sibling_labels = []
        self.root_preds = []
        self.root_labels = []

    def update(
        self,
        # Detect
        cls_preds: Optional[torch.Tensor] = None,
        cls_labels: Optional[torch.Tensor] = None,
        # Order
        order_preds: Optional[torch.Tensor] = None,
        order_labels: Optional[torch.Tensor] = None,
        # Construct
        parent_preds: Optional[torch.Tensor] = None,
        parent_labels: Optional[torch.Tensor] = None,
        sibling_preds: Optional[torch.Tensor] = None,
        sibling_labels: Optional[torch.Tensor] = None,
        root_preds: Optional[torch.Tensor] = None,
        root_labels: Optional[torch.Tensor] = None,
        # Mask
        mask: Optional[torch.Tensor] = None,
    ):
        """更新一个batch的预测结果

        Args:
            cls_preds: [B, N] 分类预测
            cls_labels: [B, N] 分类标签
            order_preds: [B, N] 后继预测索引
            order_labels: [B, N] 后继标签索引
            parent_preds: [B, N] 父节点预测索引
            parent_labels: [B, N] 父节点标签索引
            sibling_preds: [B, N, N] 兄弟关系预测
            sibling_labels: [B, N, N] 兄弟关系标签
            root_preds: [B, N] 根节点预测
            root_labels: [B, N] 根节点标签
            mask: [B, N] 有效区域掩码
        """
        if mask is None and cls_preds is not None:
            mask = torch.ones(cls_preds.shape[:2], dtype=torch.bool, device=cls_preds.device)

        mask_np = mask.cpu().numpy().astype(bool) if mask is not None else None

        # Detect
        if cls_preds is not None and cls_labels is not None:
            preds = cls_preds.cpu().numpy().flatten()
            labels = cls_labels.cpu().numpy().flatten()
            mask_flat = mask_np.flatten() if mask_np is not None else np.ones_like(preds, dtype=bool)
            self.cls_preds.extend(preds[mask_flat].tolist())
            self.cls_labels.extend(labels[mask_flat].tolist())

        # Order
        if order_preds is not None and order_labels is not None:
            preds = order_preds.cpu().numpy().flatten()
            labels = order_labels.cpu().numpy().flatten()
            mask_flat = mask_np.flatten() if mask_np is not None else np.ones_like(preds, dtype=bool)
            self.order_preds.extend(preds[mask_flat].tolist())
            self.order_labels.extend(labels[mask_flat].tolist())

        # Construct - Parent
        if parent_preds is not None and parent_labels is not None:
            preds = parent_preds.cpu().numpy().flatten()
            labels = parent_labels.cpu().numpy().flatten()
            mask_flat = mask_np.flatten() if mask_np is not None else np.ones_like(preds, dtype=bool)
            # 只统计有父节点的区域 (parent_label >= 0)
            valid = mask_flat & (labels >= 0)
            self.parent_preds.extend(preds[valid].tolist())
            self.parent_labels.extend(labels[valid].tolist())

        # Construct - Root
        # 复用 construct_only.py 的逻辑:
        # is_root_gt = (parent_labels == -1)
        # is_root_pred = (root_logits > 0)
        if root_preds is not None and root_labels is not None:
            preds = root_preds.cpu().numpy().flatten()
            labels = root_labels.cpu().numpy().flatten()
            mask_flat = mask_np.flatten() if mask_np is not None else np.ones_like(preds, dtype=bool)
            self.root_preds.extend(preds[mask_flat].astype(int).tolist())
            self.root_labels.extend(labels[mask_flat].astype(int).tolist())

        # Construct - Sibling
        if sibling_preds is not None and sibling_labels is not None:
            # sibling: [B, N, N] -> flatten valid pairs
            for b in range(sibling_preds.shape[0]):
                n = mask[b].sum().item() if mask is not None else sibling_preds.shape[1]
                pred_mat = sibling_preds[b, :n, :n].cpu().numpy()
                label_mat = sibling_labels[b, :n, :n].cpu().numpy()
                # 上三角（排除对角线）
                triu_idx = np.triu_indices(n, k=1)
                self.sibling_preds.extend(pred_mat[triu_idx].tolist())
                self.sibling_labels.extend(label_mat[triu_idx].tolist())

    def compute(self) -> DOCMetrics:
        """计算所有指标"""
        metrics = DOCMetrics()

        # Detect metrics
        if self.cls_preds and self.cls_labels:
            metrics.detect = self._compute_detect_metrics()

        # Order metrics
        if self.order_preds and self.order_labels:
            metrics.order = self._compute_order_metrics()

        # Construct metrics
        metrics.construct = self._compute_construct_metrics()

        return metrics

    def _compute_detect_metrics(self) -> DetectMetrics:
        """计算 Detect (4.2) 指标"""
        preds = np.array(self.cls_preds)
        labels = np.array(self.cls_labels)

        accuracy = accuracy_score(labels, preds)

        # F1 scores
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
        weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)

        # Per-class F1
        per_class = {}
        for cls_id in range(self.num_classes):
            cls_mask = (labels == cls_id) | (preds == cls_id)
            if cls_mask.any():
                cls_f1 = f1_score(labels == cls_id, preds == cls_id, average='binary', zero_division=0)
                per_class[cls_id] = cls_f1

        return DetectMetrics(
            accuracy=accuracy,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            weighted_f1=weighted_f1,
            per_class_f1=per_class,
        )

    def _compute_order_metrics(self) -> OrderMetrics:
        """计算 Order (4.3) 指标"""
        preds = np.array(self.order_preds)
        labels = np.array(self.order_labels)

        correct = (preds == labels).sum()
        total = len(labels)
        accuracy = correct / total if total > 0 else 0.0

        # F1: 二分类 (正确/错误)
        binary_preds = (preds == labels).astype(int)
        binary_labels = np.ones_like(binary_preds)  # 所有都应该正确
        f1 = f1_score(binary_labels, binary_preds, average='binary', zero_division=0)

        return OrderMetrics(
            accuracy=accuracy,
            f1=f1,
            correct=int(correct),
            total=int(total),
        )

    def _compute_construct_metrics(self) -> ConstructMetrics:
        """计算 Construct (4.4) 指标"""
        result = ConstructMetrics()

        # Parent metrics
        if self.parent_preds and self.parent_labels:
            preds = np.array(self.parent_preds)
            labels = np.array(self.parent_labels)
            result.parent_accuracy = accuracy_score(labels, preds)
            binary_preds = (preds == labels).astype(int)
            binary_labels = np.ones_like(binary_preds)
            result.parent_f1 = f1_score(binary_labels, binary_preds, average='binary', zero_division=0)

        # Sibling metrics
        if self.sibling_preds and self.sibling_labels:
            preds = np.array(self.sibling_preds)
            labels = np.array(self.sibling_labels)
            result.sibling_accuracy = accuracy_score(labels, preds)
            result.sibling_f1 = f1_score(labels, preds, average='binary', zero_division=0)

        # Root metrics
        if self.root_preds and self.root_labels:
            preds = np.array(self.root_preds)
            labels = np.array(self.root_labels)
            result.root_accuracy = accuracy_score(labels, preds)
            result.root_f1 = f1_score(labels, preds, average='binary', zero_division=0)

        return result


def compute_detect_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_classes: int = 5,
) -> DetectMetrics:
    """计算 Detect (4.2) 指标

    Args:
        preds: [B, N] 或 [N] 预测类别
        labels: [B, N] 或 [N] 真实类别
        mask: [B, N] 或 [N] 有效区域掩码
        num_classes: 类别数

    Returns:
        DetectMetrics
    """
    computer = DOCMetricsComputer(num_classes=num_classes)
    computer.update(cls_preds=preds, cls_labels=labels, mask=mask)
    return computer.compute().detect


def compute_order_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> OrderMetrics:
    """计算 Order (4.3) 指标

    Args:
        preds: [B, N] 后继预测索引
        labels: [B, N] 后继标签索引
        mask: [B, N] 有效区域掩码

    Returns:
        OrderMetrics
    """
    computer = DOCMetricsComputer()
    computer.update(order_preds=preds, order_labels=labels, mask=mask)
    return computer.compute().order


def compute_construct_metrics(
    parent_preds: Optional[torch.Tensor] = None,
    parent_labels: Optional[torch.Tensor] = None,
    sibling_preds: Optional[torch.Tensor] = None,
    sibling_labels: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> ConstructMetrics:
    """计算 Construct (4.4) 指标

    Args:
        parent_preds: [B, N] 父节点预测索引
        parent_labels: [B, N] 父节点标签索引
        sibling_preds: [B, N, N] 兄弟关系预测
        sibling_labels: [B, N, N] 兄弟关系标签
        mask: [B, N] 有效区域掩码

    Returns:
        ConstructMetrics
    """
    computer = DOCMetricsComputer()
    computer.update(
        parent_preds=parent_preds,
        parent_labels=parent_labels,
        sibling_preds=sibling_preds,
        sibling_labels=sibling_labels,
        mask=mask,
    )
    return computer.compute().construct

"""逻辑角色分类指标

用于评估 Detect 模块的逻辑角色分类性能。
支持 14 类逻辑角色的 F1 评估。
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import Counter


# 14 类逻辑角色定义
CLASS2ID = {
    "title": 0,
    "author": 1,
    "mail": 2,
    "affili": 3,
    "section": 4,
    "fstline": 5,
    "paraline": 6,
    "table": 7,
    "figure": 8,
    "caption": 9,
    "equation": 10,
    "footer": 11,
    "header": 12,
    "footnote": 13,
}

ID2CLASS = {v: k for k, v in CLASS2ID.items()}

# 类别映射 (将细分类别映射到标准类别)
CLASS_MAPPING = {
    "title": "title",
    "author": "author",
    "mail": "mail",
    "affili": "affili",
    "sec1": "section",
    "sec2": "section",
    "sec3": "section",
    "secx": "section",
    "section": "section",
    "alg": "paraline",
    "fstline": "fstline",
    "para": "paraline",
    "paraline": "paraline",
    "tab": "table",
    "table": "table",
    "fig": "figure",
    "figure": "figure",
    "tabcap": "caption",
    "figcap": "caption",
    "caption": "caption",
    "equ": "equation",
    "equation": "equation",
    "foot": "footer",
    "footer": "footer",
    "header": "header",
    "fnote": "footnote",
    "footnote": "footnote",
    "background": "table",
    "opara": "paraline",  # 特殊处理
}


@dataclass
class ClassificationResult:
    """分类评估结果"""
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    accuracy: float = 0.0
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    num_samples: int = 0


def normalize_class(cls: str, context: List[Dict] = None, unit: Dict = None) -> str:
    """标准化类别名称

    Args:
        cls: 原始类别名称
        context: 上下文信息 (用于 opara 处理)
        unit: 当前单元信息

    Returns:
        标准化后的类别名称
    """
    cls_lower = cls.lower()

    if cls_lower in CLASS_MAPPING:
        return CLASS_MAPPING[cls_lower]

    # 特殊处理 opara: 查找父节点类别
    if cls_lower == "opara" and context and unit:
        parent_id = unit.get('parent_id', -1)
        if parent_id >= 0 and parent_id < len(context):
            parent_cls = context[parent_id].get('class', 'paraline')
            while parent_cls.lower() == 'opara' and parent_id >= 0:
                parent_id = context[parent_id].get('parent_id', -1)
                if parent_id >= 0 and parent_id < len(context):
                    parent_cls = context[parent_id].get('class', 'paraline')
                else:
                    break
            return normalize_class(parent_cls)
        return "paraline"

    # 默认返回 paraline
    return "paraline"


def class_to_id(cls: str) -> int:
    """将类别名称转换为ID"""
    normalized = normalize_class(cls)
    return CLASS2ID.get(normalized, CLASS2ID["paraline"])


class ClassificationMetric:
    """分类评估指标类

    支持 macro/micro F1, 准确率, 每类 F1/precision/recall。
    """

    def __init__(self, num_classes: int = 14):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置指标状态"""
        self.all_preds = []
        self.all_labels = []

    def update(
        self,
        preds: List[int],
        labels: List[int],
    ):
        """更新指标

        Args:
            preds: 预测类别ID列表
            labels: 真实类别ID列表
        """
        assert len(preds) == len(labels), "预测和标签长度不匹配"
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)

    def update_from_classes(
        self,
        pred_classes: List[str],
        gt_classes: List[str],
        pred_context: List[Dict] = None,
        gt_context: List[Dict] = None,
    ):
        """从类别名称更新指标

        Args:
            pred_classes: 预测类别名称列表
            gt_classes: 真实类别名称列表
            pred_context: 预测上下文
            gt_context: 真实上下文
        """
        preds = []
        labels = []

        for i, (pred_cls, gt_cls) in enumerate(zip(pred_classes, gt_classes)):
            pred_unit = pred_context[i] if pred_context else None
            gt_unit = gt_context[i] if gt_context else None

            pred_normalized = normalize_class(pred_cls, pred_context, pred_unit)
            gt_normalized = normalize_class(gt_cls, gt_context, gt_unit)

            preds.append(CLASS2ID.get(pred_normalized, 6))  # 默认 paraline
            labels.append(CLASS2ID.get(gt_normalized, 6))

        self.update(preds, labels)

    def compute(self) -> ClassificationResult:
        """计算最终指标"""
        result = ClassificationResult()

        if len(self.all_preds) == 0:
            return result

        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        result.num_samples = len(preds)

        # 准确率
        result.accuracy = (preds == labels).mean()

        # 混淆矩阵
        result.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for p, l in zip(preds, labels):
            if 0 <= p < self.num_classes and 0 <= l < self.num_classes:
                result.confusion_matrix[l, p] += 1

        # 每类指标
        precisions = []
        recalls = []
        f1s = []

        for c in range(self.num_classes):
            tp = result.confusion_matrix[c, c]
            fp = result.confusion_matrix[:, c].sum() - tp
            fn = result.confusion_matrix[c, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            class_name = ID2CLASS.get(c, f"class_{c}")
            result.per_class_precision[class_name] = precision
            result.per_class_recall[class_name] = recall
            result.per_class_f1[class_name] = f1

            # 只有该类有样本时才计入 macro 平均
            if result.confusion_matrix[c, :].sum() > 0:
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        # Macro F1
        result.macro_f1 = np.mean(f1s) if f1s else 0.0

        # Micro F1
        total_tp = np.diag(result.confusion_matrix).sum()
        total_samples = result.confusion_matrix.sum()
        result.micro_f1 = total_tp / total_samples if total_samples > 0 else 0.0

        return result

    def compute_from_sklearn(self) -> ClassificationResult:
        """使用 sklearn 计算指标"""
        from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

        result = ClassificationResult()

        if len(self.all_preds) == 0:
            return result

        preds = self.all_preds
        labels = self.all_labels
        result.num_samples = len(preds)

        result.accuracy = accuracy_score(labels, preds)
        result.macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        result.micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)

        # 每类指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, labels=list(range(self.num_classes)), zero_division=0
        )

        for c in range(self.num_classes):
            class_name = ID2CLASS.get(c, f"class_{c}")
            result.per_class_precision[class_name] = float(precision[c])
            result.per_class_recall[class_name] = float(recall[c])
            result.per_class_f1[class_name] = float(f1[c])

        return result

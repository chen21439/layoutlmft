"""DOC模型任务定义

包含 Detect-Order-Construct 三个任务的定义：
- DetectTask (4.2): 语义分类
- OrderTask (4.3): 阅读顺序预测
- ConstructTask (4.4): 层级结构构建
"""

from typing import Dict, Any, Optional
import torch
from torch import Tensor
import torch.nn.functional as F

from .base import BaseTask
from ..metrics.doc_metrics import (
    DOCMetricsComputer,
    DetectMetrics,
    OrderMetrics,
    ConstructMetrics,
    compute_detect_metrics,
    compute_order_metrics,
    compute_construct_metrics,
)
from ..utils.tree_utils import tree_insertion_decode


class DetectTask(BaseTask):
    """4.2 Detect 任务 - 语义分类

    预测每个区域的语义类别。
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.num_classes = config.get('num_classes', 5) if config else 5
        self.label_smoothing = config.get('label_smoothing', 0.0) if config else 0.0

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """计算分类损失

        Args:
            outputs: {'category_logits': [B, N, C]}
            targets: {'categories': [B, N]}
            mask: [B, N] 有效区域掩码

        Returns:
            loss: 交叉熵损失
        """
        logits = outputs['category_logits']  # [B, N, C]
        labels = targets['categories']  # [B, N]

        B, N, C = logits.shape

        # Flatten
        logits_flat = logits.view(-1, C)
        labels_flat = labels.view(-1)

        if mask is not None:
            mask_flat = mask.view(-1)
            logits_flat = logits_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]

        if logits_flat.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device)

        loss = F.cross_entropy(
            logits_flat, labels_flat,
            label_smoothing=self.label_smoothing,
        )
        return loss

    def decode(
        self,
        outputs: Dict[str, Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """解码分类预测

        Args:
            outputs: {'category_logits': [B, N, C]}

        Returns:
            {'pred_categories': [B, N], 'category_probs': [B, N, C]}
        """
        logits = outputs['category_logits']
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        return {
            'pred_categories': preds,
            'category_probs': probs,
        }

    def compute_metrics(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        mask: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """计算分类指标

        Args:
            predictions: {'pred_categories': [B, N]}
            targets: {'categories': [B, N]}
            mask: [B, N]

        Returns:
            指标字典
        """
        preds = predictions['pred_categories']
        labels = targets['categories']

        metrics = compute_detect_metrics(preds, labels, mask, self.num_classes)
        return metrics.to_dict()


class OrderTask(BaseTask):
    """4.3 Order 任务 - 阅读顺序预测

    预测每个区域的后继区域（论文 4.2.3 格式）。
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.temperature = config.get('temperature', 1.0) if config else 1.0

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """计算后继预测损失 (Softmax CE per row)

        Args:
            outputs: {'order_logits': [B, N, N]}
            targets: {'successor_labels': [B, N]}
            mask: [B, N] 有效区域掩码

        Returns:
            loss: 交叉熵损失
        """
        logits = outputs['order_logits']  # [B, N, N]
        labels = targets['successor_labels']  # [B, N]

        B, N, _ = logits.shape

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Flatten: each row is a classification problem
        logits_flat = logits.view(B * N, N)
        labels_flat = labels.view(B * N)

        if mask is not None:
            mask_flat = mask.view(B * N)
            logits_flat = logits_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]

        if logits_flat.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device)

        # 排除无效标签 (< 0 表示无后继)
        valid = labels_flat >= 0
        if not valid.any():
            return torch.tensor(0.0, device=logits.device)

        loss = F.cross_entropy(logits_flat[valid], labels_flat[valid])
        return loss

    def decode(
        self,
        outputs: Dict[str, Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """解码后继预测

        Args:
            outputs: {'order_logits': [B, N, N]}

        Returns:
            {'pred_successors': [B, N], 'order_probs': [B, N, N]}
        """
        logits = outputs['order_logits']
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        return {
            'pred_successors': preds,
            'order_probs': probs,
        }

    def compute_metrics(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        mask: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """计算后继预测指标

        Args:
            predictions: {'pred_successors': [B, N]}
            targets: {'successor_labels': [B, N]}
            mask: [B, N]

        Returns:
            指标字典
        """
        preds = predictions['pred_successors']
        labels = targets['successor_labels']

        metrics = compute_order_metrics(preds, labels, mask)
        return metrics.to_dict()


class ConstructTask(BaseTask):
    """4.4 Construct 任务 - 层级结构构建 (Paper Section 4.4.1)

    预测父节点、兄弟关系（均使用 softmax CE，N选1）。
    采用论文自指向方案：Root 节点的 parent_label 指向自己（而不是 -1）。
    这样所有 label 都在 [0, N-1] 范围内，可以直接计算 cross_entropy。
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.parent_weight = config.get('parent_weight', 1.0) if config else 1.0
        self.sibling_weight = config.get('sibling_weight', 0.5) if config else 0.5

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """计算层级结构损失

        Args:
            outputs: {
                'parent_logits': [B, N, N],
                'sibling_logits': [B, N, N]
            }
            targets: {
                'parent_labels': [B, N],  # self-index for root (自指向)
                'sibling_labels': [B, N]  # -1 for no left sibling
            }
            mask: [B, N]

        Returns:
            loss: 加权总损失
        """
        total_loss = torch.tensor(0.0, device=outputs['parent_logits'].device)

        # Parent loss (softmax CE, N选1)
        if 'parent_logits' in outputs and 'parent_labels' in targets:
            parent_loss = self._compute_parent_loss(
                outputs['parent_logits'],
                targets['parent_labels'],
                mask,
            )
            total_loss = total_loss + self.parent_weight * parent_loss

        # Sibling loss (softmax CE, N选1, same as parent)
        if 'sibling_logits' in outputs and 'sibling_labels' in targets:
            sibling_loss = self._compute_sibling_loss(
                outputs['sibling_logits'],
                targets['sibling_labels'],
                mask,
            )
            total_loss = total_loss + self.sibling_weight * sibling_loss

        return total_loss

    def _compute_parent_loss(
        self,
        logits: Tensor,  # [B, N, N]
        labels: Tensor,  # [B, N]
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """计算父节点预测损失 (softmax CE)

        论文自指向方案：所有有效节点都计算 loss，包括 root（自指向）。
        """
        B, N, _ = logits.shape

        logits_flat = logits.view(B * N, N)
        labels_flat = labels.view(B * N)

        if mask is not None:
            mask_flat = mask.view(B * N)
            logits_flat = logits_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]

        # 论文自指向方案：所有 label 都在 [0, N-1] 范围内
        valid = (labels_flat >= 0) & (labels_flat < N)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device)

        return F.cross_entropy(logits_flat[valid], labels_flat[valid])

    def _compute_sibling_loss(
        self,
        logits: Tensor,  # [B, N, N]
        labels: Tensor,  # [B, N] index of left sibling (self-index for none)
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """计算兄弟关系损失 (softmax CE, N选1)

        论文自指向方案：无左兄弟的节点指向自己，所有有效节点都参与训练。
        """
        B, N, _ = logits.shape

        logits_flat = logits.view(B * N, N)
        labels_flat = labels.view(B * N)

        if mask is not None:
            mask_flat = mask.view(B * N)
            logits_flat = logits_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]

        # 论文自指向方案：所有有效节点都参与（包括自指向=无左兄弟）
        valid = (labels_flat >= 0) & (labels_flat < N)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device)

        return F.cross_entropy(logits_flat[valid], labels_flat[valid])

    def decode(
        self,
        outputs: Dict[str, Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """解码层级结构预测（论文 Algorithm 1: Tree Insertion Algorithm）

        使用联合解码保证树结构的合法性（sibling 约束）。

        Args:
            outputs: {
                'parent_logits': [B, N, N],
                'sibling_logits': [B, N, N]
            }

        Returns:
            {
                'pred_parents': [B, N] 层级父节点（自指向方案）,
                'pred_siblings': [B, N] 左兄弟（自指向方案）
            }
        """
        parent_logits = outputs.get('parent_logits')
        sibling_logits = outputs.get('sibling_logits')

        if parent_logits is None:
            return {}

        B = parent_logits.size(0)
        N = parent_logits.size(1)
        device = parent_logits.device

        # 批量联合解码
        all_parents = []
        all_siblings = []

        for b in range(B):
            if sibling_logits is not None:
                parents, siblings = tree_insertion_decode(
                    parent_logits[b], sibling_logits[b]
                )
            else:
                # 无 sibling_logits 时退化为 argmax
                parents = parent_logits[b].argmax(dim=-1).cpu().tolist()
                siblings = list(range(N))  # 全部自指向

            all_parents.append(parents)
            all_siblings.append(siblings)

        result = {
            'pred_parents': torch.tensor(all_parents, dtype=torch.long, device=device),
            'pred_siblings': torch.tensor(all_siblings, dtype=torch.long, device=device),
        }

        return result

    def compute_metrics(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        mask: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """计算层级结构指标

        Args:
            predictions: {'pred_parents': [B, N], 'pred_siblings': [B, N, N]}
            targets: {'parent_labels': [B, N], 'sibling_labels': [B, N, N]}
            mask: [B, N]

        Returns:
            指标字典
        """
        metrics = compute_construct_metrics(
            parent_preds=predictions.get('pred_parents'),
            parent_labels=targets.get('parent_labels'),
            sibling_preds=predictions.get('pred_siblings'),
            sibling_labels=targets.get('sibling_labels'),
            mask=mask,
        )
        return metrics.to_dict()


class DOCTask:
    """DOC 完整任务 - 组合 Detect + Order + Construct

    提供统一的评估接口。
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.detect_task = DetectTask(config.get('detect', {}))
        self.order_task = OrderTask(config.get('order', {}))
        self.construct_task = ConstructTask(config.get('construct', {}))
        self.metrics_computer = DOCMetricsComputer(
            num_classes=config.get('num_classes', 5)
        )

    def reset_metrics(self):
        """重置指标计算器"""
        self.metrics_computer.reset()

    def update_metrics(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
    ):
        """更新指标（累积一个batch）

        Args:
            outputs: 模型输出
            targets: 标签
            mask: 有效区域掩码
        """
        # Detect
        if 'category_logits' in outputs:
            cls_preds = outputs['category_logits'].argmax(dim=-1)
            self.metrics_computer.update(
                cls_preds=cls_preds,
                cls_labels=targets.get('categories'),
                mask=mask,
            )

        # Order
        if 'order_logits' in outputs:
            order_preds = outputs['order_logits'].argmax(dim=-1)
            self.metrics_computer.update(
                order_preds=order_preds,
                order_labels=targets.get('successor_labels'),
                mask=mask,
            )

        # Construct - 使用联合解码
        if 'parent_logits' in outputs:
            # 使用 ConstructTask.decode 进行联合解码
            construct_preds = self.construct_task.decode(outputs)
            parent_preds = construct_preds.get('pred_parents')
            sibling_preds = construct_preds.get('pred_siblings')

            parent_labels = targets.get('parent_labels')
            self.metrics_computer.update(
                parent_preds=parent_preds,
                parent_labels=parent_labels,
                mask=mask,
            )

            if sibling_preds is not None:
                self.metrics_computer.update(
                    sibling_preds=sibling_preds,
                    sibling_labels=targets.get('sibling_labels'),
                    mask=mask,
                )

    def compute_metrics(self) -> Dict[str, float]:
        """计算最终指标

        Returns:
            所有任务的指标字典
        """
        return self.metrics_computer.compute().to_dict()

    def get_metrics_summary(self) -> str:
        """获取指标摘要字符串"""
        return self.metrics_computer.compute().summary()

#!/usr/bin/env python
# coding=utf-8
"""
Parent Finding Task (M_cp 相关代码)

包含：
1. ChildParentDistributionMatrix - Child-Parent Distribution Matrix (M_cp)
2. build_child_parent_matrix_from_dataset() - 从数据集构建 M_cp
3. ParentFindingTask - 封装 Parent Finding 的 loss/decode 逻辑

重构自 train_parent_finder.py 和 train_joint.py
"""

import logging
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

logger = logging.getLogger(__name__)


class ChildParentDistributionMatrix:
    """
    Child-Parent Distribution Matrix (M_cp)
    根据训练数据统计不同语义类别的父子关系分布
    """

    def __init__(self, num_classes=None, pseudo_count=5):
        """
        Args:
            num_classes: 语义类别数（不包含ROOT），默认使用 NUM_LABELS
            pseudo_count: 加性平滑的伪计数
        """
        # 延迟导入避免循环依赖
        if num_classes is None:
            from layoutlmft.data.labels import NUM_LABELS
            num_classes = NUM_LABELS

        self.num_classes = num_classes
        self.pseudo_count = pseudo_count

        # M_cp: [num_classes+1, num_classes]
        # 第i列表示类别i作为子节点时，其父节点的类别分布
        # 行包含ROOT（索引0）和所有语义类别（索引1到num_classes）
        self.matrix = np.zeros((num_classes + 1, num_classes))
        self.counts = np.zeros((num_classes + 1, num_classes))

    def update(self, child_label, parent_label):
        """
        更新统计计数

        Args:
            child_label: 子节点的语义类别 [0, num_classes-1]
            parent_label: 父节点的语义类别 [-1, num_classes-1]
                         -1 表示 ROOT
        """
        if child_label < 0 or child_label >= self.num_classes:
            return

        # 将 parent_label=-1 映射到索引0（ROOT）
        parent_idx = parent_label + 1 if parent_label >= 0 else 0

        if parent_idx < 0 or parent_idx > self.num_classes:
            return

        self.counts[parent_idx, child_label] += 1

    def build(self):
        """
        构建分布矩阵（加性平滑）
        """
        # 加性平滑
        smoothed_counts = self.counts + self.pseudo_count

        # 归一化每一列（每个子类别的父类别分布）
        col_sums = smoothed_counts.sum(axis=0, keepdims=True)
        self.matrix = smoothed_counts / (col_sums + 1e-10)

        logger.info(f"Child-Parent Distribution Matrix 构建完成")
        logger.info(f"  形状: {self.matrix.shape}")
        logger.info(f"  统计样本数: {self.counts.sum():.0f}")

    def get_tensor(self, device='cpu'):
        """返回 torch.Tensor 版本"""
        return torch.tensor(self.matrix, dtype=torch.float32, device=device)

    def save(self, path):
        """保存矩阵"""
        np.save(path, self.matrix)
        logger.info(f"保存 M_cp 到: {path}")

    def load(self, path):
        """加载矩阵"""
        self.matrix = np.load(path)
        logger.info(f"加载 M_cp 从: {path}")


def build_child_parent_matrix_from_dataset(dataset, num_classes=None):
    """
    从 HuggingFace Dataset 构建 Child-Parent Distribution Matrix (M_cp)

    Args:
        dataset: HuggingFace Dataset 对象，需要包含 ner_tags, line_ids, line_parent_ids
        num_classes: 语义类别数（不包含ROOT），默认使用 NUM_LABELS

    Returns:
        ChildParentDistributionMatrix 实例
    """
    # 延迟导入避免循环依赖
    if num_classes is None:
        from layoutlmft.data.labels import NUM_LABELS
        num_classes = NUM_LABELS

    logger.info("从数据集构建 Child-Parent Distribution Matrix...")

    cp_matrix = ChildParentDistributionMatrix(num_classes=num_classes)

    for example in tqdm(dataset, desc="统计父子关系"):
        ner_tags = example.get("ner_tags", [])
        line_ids = example.get("line_ids", [])
        line_parent_ids = example.get("line_parent_ids", [])

        if not line_parent_ids or not ner_tags:
            continue

        # 从 token-level 标签提取 line-level 标签
        line_labels = {}
        for tag, line_id in zip(ner_tags, line_ids):
            if line_id >= 0 and line_id not in line_labels and tag >= 0:
                line_labels[line_id] = tag

        # 统计父子关系
        for child_idx, parent_idx in enumerate(line_parent_ids):
            if child_idx not in line_labels:
                continue

            child_label = line_labels[child_idx]
            parent_label = line_labels.get(parent_idx, -1) if parent_idx >= 0 else -1

            cp_matrix.update(child_label, parent_label)

    cp_matrix.build()
    return cp_matrix


class ParentFindingTask:
    """
    Parent Finding Task 封装类

    封装父节点查找的核心逻辑：
    - Loss 计算（带 Soft-Mask）
    - Decode（贪心解码）
    - 准确率统计

    使用方式：
        task = ParentFindingTask(use_soft_mask=True, M_cp=cp_matrix.get_tensor(device))
        loss, acc = task.compute_loss(parent_logits, line_parent_ids, line_mask)
        predictions = task.decode(parent_logits, line_mask)
    """

    def __init__(
        self,
        use_soft_mask: bool = False,
        M_cp: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Args:
            use_soft_mask: 是否使用 Soft-Mask（需要提供 M_cp）
            M_cp: Child-Parent Distribution Matrix [num_classes+1, num_classes]
            num_classes: 语义类别数（仅在需要时使用）
        """
        self.use_soft_mask = use_soft_mask
        self.M_cp = M_cp

        if num_classes is None:
            try:
                from layoutlmft.data.labels import NUM_LABELS
                num_classes = NUM_LABELS
            except ImportError:
                num_classes = 16  # 默认值

        self.num_classes = num_classes

        if use_soft_mask and M_cp is None:
            logger.warning("use_soft_mask=True but M_cp not provided, soft-mask will be disabled")
            self.use_soft_mask = False

    def compute_loss(
        self,
        parent_logits: torch.Tensor,
        line_parent_ids: torch.Tensor,
        line_mask: torch.Tensor,
    ) -> tuple:
        """
        计算 Parent Finding Loss

        Args:
            parent_logits: [B, L+1, L+1] - 父节点 logits（包括 ROOT）
            line_parent_ids: [B, L] - 每行的父节点索引（-1 表示 ROOT）
            line_mask: [B, L] - 有效行的 mask

        Returns:
            (loss, accuracy): 平均 loss 和准确率
        """
        import torch.nn.functional as F

        batch_size = parent_logits.shape[0]
        num_lines = line_parent_ids.shape[1]
        device = parent_logits.device

        total_loss = 0.0
        correct = 0
        count = 0

        for b in range(batch_size):
            for i in range(num_lines):
                if not line_mask[b, i]:
                    continue

                # 行 i 在新序列中的位置是 i+1（因为 ROOT 在位置 0）
                logits_i = parent_logits[b, i+1, :i+1]  # [0:i+1] 包含 ROOT 和行 0~i-1
                target_i = line_parent_ids[b, i]

                # 映射：line_parent_ids -> logits索引
                # parent_id = -1 (ROOT) -> index = 0
                # parent_id = j (行j)    -> index = j+1
                if target_i == -1:
                    target_idx = 0  # ROOT
                else:
                    target_idx = target_i + 1

                # 检查有效性
                if target_idx < 0 or target_idx > i:
                    continue
                if torch.isinf(logits_i).all() or torch.isnan(logits_i).any():
                    continue
                if target_idx >= len(logits_i):
                    continue

                # 交叉熵损失
                loss_i = F.cross_entropy(
                    logits_i.unsqueeze(0),
                    torch.tensor([target_idx], device=device)
                )

                if torch.isnan(loss_i) or torch.isinf(loss_i):
                    continue

                total_loss += loss_i.item()

                # 统计准确率
                pred_idx = torch.argmax(logits_i).item()
                if pred_idx == target_idx:
                    correct += 1
                count += 1

        # 返回平均 loss 和准确率
        avg_loss = total_loss / count if count > 0 else 0.0
        accuracy = correct / count if count > 0 else 0.0

        return avg_loss, accuracy

    def decode(
        self,
        parent_logits: torch.Tensor,
        line_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        贪心解码：为每一行选择概率最高的父节点

        Args:
            parent_logits: [B, L+1, L+1] - 父节点 logits（包括 ROOT）
            line_mask: [B, L] - 有效行的 mask

        Returns:
            predictions: [B, L] - 预测的父节点索引（-1 表示 ROOT）
        """
        batch_size = parent_logits.shape[0]
        num_lines = line_mask.shape[1]
        device = parent_logits.device

        predictions = torch.full((batch_size, num_lines), -1, dtype=torch.long, device=device)

        for b in range(batch_size):
            for i in range(num_lines):
                if not line_mask[b, i]:
                    continue

                # 行 i 的候选父节点：ROOT (0) + 行 0~i-1 (1~i)
                logits_i = parent_logits[b, i+1, :i+1]

                if torch.isinf(logits_i).all() or torch.isnan(logits_i).any():
                    continue

                # 选择概率最高的父节点
                pred_idx = torch.argmax(logits_i).item()

                # 映射回原始索引
                # index=0 -> parent_id=-1 (ROOT)
                # index=j+1 -> parent_id=j (行j)
                if pred_idx == 0:
                    predictions[b, i] = -1  # ROOT
                else:
                    predictions[b, i] = pred_idx - 1

        return predictions

#!/usr/bin/env python
# coding=utf-8
"""
Line-level 分类头

功能：接收聚合后的 line_features，输出每行的分类 logits

=== 流程 ===

输入:
    line_features: [num_lines, hidden_dim]
    - 来自 LinePooling 的输出
    - 每行一个特征向量

处理:
    1. Dropout（训练时随机丢弃，防止过拟合）
    2. Linear（hidden_dim → num_classes）

输出:
    logits: [num_lines, num_classes]
    - 每行对每个类别的得分（未经 softmax）

=== 损失计算 ===

在模型的 forward 中计算:
    loss = F.cross_entropy(logits, line_labels, ignore_index=-100)

其中 line_labels 是每行的真实类别标签。
"""

import torch
import torch.nn as nn
from typing import Optional


class LineClassificationHead(nn.Module):
    """
    Line-level 分类头

    结构简单：Dropout → Linear

    Example:
        >>> head = LineClassificationHead(hidden_size=768, num_classes=14)
        >>> features = torch.randn(10, 768)  # 10 lines
        >>> logits = head(features)  # [10, 14]
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: 输入特征维度（通常是 768）
            num_classes: 分类类别数
            dropout: Dropout 比例
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        line_features: torch.Tensor,
        line_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对每行进行分类

        Args:
            line_features: [num_lines, hidden_dim]
                - Line-level 特征
            line_mask: [num_lines]（可选）
                - True 表示有效行，False 表示 padding
                - 目前未使用，保留用于未来扩展

        Returns:
            logits: [num_lines, num_classes]
                - 每行对每个类别的得分
        """
        # Dropout: 训练时随机丢弃部分神经元
        x = self.dropout(line_features)

        # Linear: 映射到类别空间
        logits = self.classifier(x)

        return logits

    def get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        从 logits 获取预测类别

        Args:
            logits: [num_lines, num_classes]

        Returns:
            predictions: [num_lines]，每行的预测类别 ID
        """
        return logits.argmax(dim=-1)

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        从 logits 获取概率分布

        Args:
            logits: [num_lines, num_classes]

        Returns:
            probs: [num_lines, num_classes]，经过 softmax 的概率
        """
        return torch.softmax(logits, dim=-1)

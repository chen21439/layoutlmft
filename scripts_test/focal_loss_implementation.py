#!/usr/bin/env python
# coding=utf-8
"""
FocalLoss 实现（参考论文）
用于替换当前的 CrossEntropyLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    论文中使用的损失函数：
    FL(p_t) = -α_t(1 - p_t)^γ * log(p_t)

    Args:
        alpha: 类别权重，shape: [num_classes]
        gamma: focusing parameter，通常取 2
        reduction: 'mean' or 'sum'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] - 模型输出 logits
            targets: [batch_size] - 真实标签（整数）

        Returns:
            loss: scalar
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算预测概率
        p = torch.exp(-ce_loss)  # p_t

        # 计算 focal loss
        focal_weight = (1 - p) ** self.gamma
        loss = focal_weight * ce_loss

        # 应用类别权重
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # 归约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# 使用示例
if __name__ == "__main__":
    # 创建 FocalLoss
    num_classes = 4
    alpha = torch.tensor([1.0, 2.0, 2.0, 2.0])  # none类权重低，其他类权重高
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    # 测试
    batch_size = 8
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss = criterion(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")

    # 对比 CrossEntropyLoss
    ce_criterion = nn.CrossEntropyLoss()
    ce_loss = ce_criterion(logits, targets)
    print(f"CrossEntropy Loss: {ce_loss.item():.4f}")

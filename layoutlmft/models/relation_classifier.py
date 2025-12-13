#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 层级关系分类器 (方案C)
基于行业最佳实践实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import random


class LineFeatureExtractor:
    """
    从 token-level 的 LayoutLMv2 hidden states 提取行级特征
    """

    @staticmethod
    def extract_line_features(
        hidden_states: torch.Tensor,  # [batch_size, seq_len, hidden_size]
        line_ids: torch.Tensor,       # [batch_size, seq_len] - 每个token属于哪个line
        pooling: str = "mean"          # "mean", "max", or "first"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: LayoutLMv2的输出 [B, T, H]
            line_ids: 每个token对应的line_id [B, T]
            pooling: 池化方式

        Returns:
            line_features: [B, max_lines, H] - 每行的特征向量
            line_mask: [B, max_lines] - 有效行的mask
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        # 找出每个batch中的最大line数
        max_lines = line_ids.max().item() + 1

        # 初始化行特征和mask
        line_features = torch.zeros(batch_size, max_lines, hidden_size, device=device)
        line_mask = torch.zeros(batch_size, max_lines, dtype=torch.bool, device=device)

        # 对每个batch单独处理
        for b in range(batch_size):
            unique_lines = torch.unique(line_ids[b])
            unique_lines = unique_lines[unique_lines >= 0]  # 过滤掉无效line_id

            for line_id in unique_lines:
                # 找到属于这个line的所有token
                token_mask = (line_ids[b] == line_id)
                line_tokens = hidden_states[b][token_mask]  # [num_tokens, H]

                # 池化
                if pooling == "mean":
                    line_feat = line_tokens.mean(dim=0)
                elif pooling == "max":
                    line_feat = line_tokens.max(dim=0)[0]
                elif pooling == "first":
                    line_feat = line_tokens[0]
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")

                line_features[b, line_id] = line_feat
                line_mask[b, line_id] = True

        return line_features, line_mask


class NegativeSampler:
    """
    负采样策略：从同页、在child之前的行中采样负样本
    """

    def __init__(
        self,
        neg_ratio: int = 3,  # 每个正样本采样3-5个负样本
        same_page_only: bool = True,
        before_child_only: bool = True
    ):
        self.neg_ratio = neg_ratio
        self.same_page_only = same_page_only
        self.before_child_only = before_child_only

    def sample_pairs(
        self,
        line_parent_ids: List[int],   # 每个line的parent_id
        line_relations: List[str],    # 每个line的relation
        num_lines: int                # 总行数
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        构造训练样本对

        Returns:
            pairs: [(parent_idx, child_idx), ...]
            labels: [0 or 1] - 0=非父子, 1=父子关系
        """
        pairs = []
        labels = []

        # 1. 收集所有正样本
        positive_pairs = []
        for child_idx in range(num_lines):
            parent_idx = line_parent_ids[child_idx]
            if parent_idx >= 0 and parent_idx < num_lines:
                positive_pairs.append((parent_idx, child_idx))
                pairs.append((parent_idx, child_idx))
                labels.append(1)  # 正样本

        # 2. 为每个正样本采样负样本
        for parent_idx, child_idx in positive_pairs:
            # 候选负样本：在child之前的所有行（排除真实parent）
            if self.before_child_only:
                candidates = list(range(child_idx))
            else:
                candidates = list(range(num_lines))

            # 移除真实parent
            if parent_idx in candidates:
                candidates.remove(parent_idx)

            # 采样
            neg_samples = min(self.neg_ratio, len(candidates))
            if neg_samples > 0:
                neg_parents = random.sample(candidates, neg_samples)
                for neg_p in neg_parents:
                    pairs.append((neg_p, child_idx))
                    labels.append(0)  # 负样本

        return pairs, labels


class SimpleRelationClassifier(nn.Module):
    """
    简单的关系分类器（方案C第一版）
    先做二分类：是否为父子关系
    """

    def __init__(
        self,
        hidden_size: int = 768,
        use_geometry: bool = False,  # 是否使用几何特征
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_geometry = use_geometry

        # 输入维度：parent_feat + child_feat (+ optional geometry)
        input_dim = hidden_size * 2
        if use_geometry:
            input_dim += 4  # [vertical_dist, horizontal_overlap, page_diff, type_match]

        # 简单的MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # 二分类：0=非父子, 1=父子
        )

    def forward(
        self,
        parent_features: torch.Tensor,  # [batch_size, hidden_size]
        child_features: torch.Tensor,   # [batch_size, hidden_size]
        geometry_features: Optional[torch.Tensor] = None  # [batch_size, 4]
    ) -> torch.Tensor:
        """
        Args:
            parent_features: 候选父节点的特征
            child_features: 子节点的特征
            geometry_features: 可选的几何特征

        Returns:
            logits: [batch_size, 2] - 二分类logits
        """
        # 拼接特征
        combined = torch.cat([parent_features, child_features], dim=-1)

        if self.use_geometry and geometry_features is not None:
            combined = torch.cat([combined, geometry_features], dim=-1)

        # 分类
        logits = self.classifier(combined)
        return logits


class MultiClassRelationClassifier(nn.Module):
    """
    多类别关系分类器（论文对齐版本）
    严格按照 HRDoc 论文实现：单层线性投影，不使用几何特征
    公式：P_rel_(i,j) = softmax(LinearProj(Concat(h_i, h_j)))

    注意：
    - 论文中只有 3 类关系 (connect, contain, equality)，不含 none
    - 输入是 GRU 隐状态 h_i, h_j，维度是 gru_hidden_size（默认 512）
    """

    def __init__(
        self,
        hidden_size: int = 512,  # GRU hidden size（论文对齐）
        num_relations: int = 3,  # connect=0, contain=1, equality=2（论文对齐）
        use_geometry: bool = False,  # 论文不使用几何特征
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        self.use_geometry = use_geometry

        input_dim = hidden_size * 2
        if use_geometry:
            input_dim += 4

        # 论文版本：单层线性投影（严格对齐）
        self.classifier = nn.Linear(input_dim, num_relations)

    def forward(
        self,
        parent_features: torch.Tensor,
        child_features: torch.Tensor,
        geometry_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns:
            logits: [batch_size, num_relations]
        """
        combined = torch.cat([parent_features, child_features], dim=-1)

        if self.use_geometry and geometry_features is not None:
            combined = torch.cat([combined, geometry_features], dim=-1)

        logits = self.classifier(combined)
        return logits


def compute_geometry_features(
    parent_bbox: torch.Tensor,  # [4] or [batch, 4]
    child_bbox: torch.Tensor,   # [4] or [batch, 4]
    page_diff: Optional[torch.Tensor] = None  # [1] or [batch]
) -> torch.Tensor:
    """
    计算几何特征（可选的增强特征）

    Args:
        parent_bbox: [x0, y0, x1, y1] 归一化坐标
        child_bbox: [x0, y0, x1, y1] 归一化坐标
        page_diff: 页码差

    Returns:
        features: [4] - [vertical_dist, horizontal_overlap, page_diff, 0]
    """
    # 垂直距离（child的顶部 - parent的底部）
    vertical_dist = child_bbox[..., 1] - parent_bbox[..., 3]

    # 水平重叠率
    x_overlap = torch.clamp(
        torch.min(parent_bbox[..., 2], child_bbox[..., 2]) -
        torch.max(parent_bbox[..., 0], child_bbox[..., 0]),
        min=0
    )
    parent_width = parent_bbox[..., 2] - parent_bbox[..., 0]
    child_width = child_bbox[..., 2] - child_bbox[..., 0]
    horizontal_overlap = x_overlap / (parent_width + child_width + 1e-6)

    # 页码差（如果提供）
    if page_diff is None:
        page_diff = torch.zeros_like(vertical_dist)

    # 组合特征
    features = torch.stack([
        vertical_dist,
        horizontal_overlap,
        page_diff,
        torch.zeros_like(vertical_dist)  # 保留位置，未来可加type_match等
    ], dim=-1)

    return features


# 关系类型映射（只有3类，不含none/meta）
RELATION_LABELS = {
    "connect": 0,
    "contain": 1,
    "equality": 2,
}
NUM_RELATIONS = 3
RELATION_NAMES = ["connect", "contain", "equality"]


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    论文公式：FL(p_t) = -α_t(1 - p_t)^γ * log(p_t)

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    HRDoc paper: https://ar5iv.labs.arxiv.org/html/2303.13839

    Args:
        alpha: 类别权重，shape: [num_classes] 或 None
        gamma: focusing parameter，论文中常用 2.0
        reduction: 'mean', 'sum' 或 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha      # 类别权重 [num_classes]
        self.gamma = gamma      # focusing parameter (论文中常用2)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] - 模型输出 logits
            targets: [batch_size] - 真实标签（整数）

        Returns:
            loss: scalar (如果 reduction='mean' 或 'sum')
        """
        # 计算交叉熵（不进行归约）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算预测概率 p_t
        p = torch.exp(-ce_loss)  # p_t

        # 计算 focal weight: (1 - p_t)^γ
        focal_weight = (1 - p) ** self.gamma

        # Focal Loss = focal_weight * ce_loss
        loss = focal_weight * ce_loss

        # 应用类别权重 α_t
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
            else:
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            loss = alpha_t * loss

        # 归约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

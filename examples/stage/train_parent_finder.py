#!/usr/bin/env python
# coding=utf-8
"""
训练父节点查找器（SubTask 2）
基于论文 HRDoc 的方法实现：
- GRU decoder 顺序处理语义单元
- Soft-mask 操作（Child-Parent Distribution Matrix）
- 注意力机制计算父节点概率
- 多分类交叉熵损失
"""

import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import defaultdict
from functools import partial

logger = logging.getLogger(__name__)


class ChildParentDistributionMatrix:
    """
    Child-Parent Distribution Matrix (M_cp)
    根据训练数据统计不同语义类别的父子关系分布
    """

    def __init__(self, num_classes=16, pseudo_count=5):
        """
        Args:
            num_classes: 语义类别数（不包含ROOT）
            pseudo_count: 加性平滑的伪计数
        """
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


class ParentFinderGRU(nn.Module):
    """
    基于GRU的父节点查找器（论文方法）

    对每个语义单元 u_i，预测其父节点索引 P̂_i ∈ {0, 1, ..., i-1}
    其中 0 表示 ROOT，1到i-1表示之前的语义单元
    """

    def __init__(
        self,
        hidden_size=768,
        gru_hidden_size=512,
        num_classes=16,
        dropout=0.1,
        use_soft_mask=True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.gru_hidden_size = gru_hidden_size
        self.num_classes = num_classes
        self.use_soft_mask = use_soft_mask

        # GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0 if dropout == 0 else dropout
        )

        # 查询向量投影（用于注意力计算）
        self.query_proj = nn.Linear(gru_hidden_size, gru_hidden_size)

        # 键向量投影
        self.key_proj = nn.Linear(gru_hidden_size, gru_hidden_size)

        # 类别预测头（用于预测每个单元的语义类别概率）
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Soft-mask 矩阵（可选）
        # M_cp: [num_classes+1, num_classes]
        self.register_buffer('M_cp', torch.ones(num_classes + 1, num_classes))

        self.dropout = nn.Dropout(dropout)

    def set_child_parent_matrix(self, M_cp):
        """设置 Child-Parent Distribution Matrix"""
        self.M_cp = M_cp

    def forward(
        self,
        line_features,     # [batch_size, max_lines, hidden_size]
        line_mask          # [batch_size, max_lines] - 有效行的mask
    ):
        """
        Args:
            line_features: 行级特征 [B, L, H]
            line_mask: 有效行mask [B, L]

        Returns:
            parent_logits: [B, L+1, L+1] - 每个位置i的父节点logits（包括ROOT）
        """
        batch_size, max_lines, hidden_size = line_features.shape
        device = line_features.device

        # 1. 构建 ROOT 节点（论文方法：所有单元表示的平均）
        # 只对有效行求平均
        valid_sum = (line_features * line_mask.unsqueeze(-1)).sum(dim=1)  # [B, H]
        valid_count = line_mask.sum(dim=1, keepdim=True).clamp(min=1)     # [B, 1]
        root_feat = valid_sum / valid_count                                # [B, H]
        root_feat = root_feat.unsqueeze(1)                                 # [B, 1, H]

        # 2. 将 ROOT 节点拼接到序列最前面
        line_features_with_root = torch.cat([root_feat, line_features], dim=1)  # [B, L+1, H]
        line_mask_with_root = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=device), line_mask],
            dim=1
        )  # [B, L+1]

        # 3. 通过 GRU 获取隐藏状态
        # h_i 包含了从开始到第i个单元的上下文信息
        gru_output, _ = self.gru(line_features_with_root)  # [B, L+1, gru_hidden]

        # 2. 预测每个单元的语义类别概率（用于 soft-mask）
        # 注意：line_features 是原始行特征 [B, L, H]，不包括 ROOT
        cls_logits = self.cls_head(line_features)  # [B, L, num_classes]
        cls_probs = F.softmax(cls_logits, dim=-1)  # [B, L, num_classes]

        # 为 ROOT 添加一个虚拟的类别概率分布（全1的均匀分布或特殊处理）
        # ROOT 的类别概率：使用一个特殊的"ROOT类别"（索引 num_classes）
        root_cls_prob = torch.zeros(batch_size, 1, self.num_classes + 1, device=device)
        root_cls_prob[:, :, -1] = 1.0  # ROOT 类别（最后一个）

        # 原始类别概率需要扩展到 num_classes+1（添加 ROOT 类别）
        # 为每行添加一个0的 ROOT 类别概率
        cls_probs_extended = torch.cat([
            cls_probs,
            torch.zeros(batch_size, max_lines, 1, device=device)
        ], dim=-1)  # [B, L, num_classes+1]

        # 将 ROOT 和行的类别概率拼接
        cls_probs_with_root = torch.cat([root_cls_prob, cls_probs_extended], dim=1)  # [B, L+1, num_classes+1]

        # 4. 计算父节点概率
        # 对每个位置 i，计算它与之前所有位置 j (0 <= j < i) 的父子概率
        # 现在序列长度是 L+1（包括 ROOT）

        # 查询向量（当前单元）
        query = self.query_proj(gru_output)  # [B, L+1, gru_hidden]

        # 键向量（候选父节点）
        key = self.key_proj(gru_output)  # [B, L+1, gru_hidden]

        # 注意力分数: alpha(q_i, h_j)
        # 使用高效的 bmm 避免创建巨大的中间张量
        # [B, L+1, L+1] = [B, L+1, gru_hidden] @ [B, gru_hidden, L+1]
        attention_scores = torch.bmm(
            query,  # [B, L+1, gru_hidden]
            key.transpose(1, 2)  # [B, gru_hidden, L+1]
        ) / (self.gru_hidden_size ** 0.5)  # [B, L+1, L+1]

        # 5. Soft-mask 操作（根据语义类别约束）
        if self.use_soft_mask and self.M_cp is not None:
            # M_cp: [num_classes+1, num_classes] = [17, 16]
            # cls_probs: [B, L, num_classes] = [B, L, 16] (原始行的类别概率，不包括 ROOT)
            # cls_probs_with_root: [B, L+1, num_classes+1] (包括 ROOT)

            # 论文公式：P_dom(i,j) = P_cls_j · M_cp · P_cls_i^T
            # 其中 i 是 child（只能是语义单元，不包括 ROOT），j 是 parent（可以是 ROOT 或语义单元）

            # 对于每个 child i (位置 1 到 L+1，对应原始行 0 到 L-1)
            # 和每个 parent j (位置 0 到 i-1)，计算领域先验概率

            # soft_mask[b, i, j] = P_cls_j · M_cp · P_cls_i^T
            # = cls_probs_with_root[b, j] @ M_cp @ cls_probs[b, i-1].T

            # 使用 einsum 计算：
            # cls_probs: [B, L, C]
            # cls_probs_with_root: [B, L+1, C+1] (parent 的类别概率)
            # M_cp: [C+1, C]

            # 步骤1：M_cp @ cls_probs[child].T -> [C+1, B*L]
            # 步骤2：cls_probs_with_root[parent] @ 结果 -> [B, L+1, B*L]

            # 更简单的方法：逐对计算
            # soft_mask[b, i, j] = cls_probs_with_root[b, j, :] @ M_cp @ cls_probs[b, i-1, :]

            # 使用 einsum 一次性计算所有对：
            # 'bic,cp,bjp->bij'
            # b=batch, i=child位置(1~L), j=parent位置(0~i-1), c=parent类别(C+1), p=child类别(C)

            # 注意：位置 0 是 ROOT，它的 child 类别概率需要特殊处理
            # ROOT 不能作为 child，所以我们只需要计算位置 1~L+1 作为 child 的情况

            # 为了简化，我们可以给 ROOT 一个虚拟的 child 类别概率（全0或均匀分布）
            # 但实际上 ROOT 永远不会作为 child，所以这个值不重要

            # cls_probs 只包含原始行（不包括 ROOT），形状是 [B, L, C]
            # 我们需要为 ROOT 添加一个虚拟的 child 类别概率
            cls_probs_for_child = torch.cat([
                torch.ones(batch_size, 1, self.num_classes, device=device) / self.num_classes,  # ROOT 的虚拟分布
                cls_probs  # [B, L, C]
            ], dim=1)  # [B, L+1, C]

            # 计算 soft_mask:
            # soft_mask[b, i, j] = cls_probs_with_root[b, j, :] @ M_cp @ cls_probs_for_child[b, i, :]
            # 使用 einsum: 'bjc,cp,bip->bij'
            soft_mask = torch.einsum('bjc,cp,bip->bij',
                                     cls_probs_with_root,  # [B, L+1, C+1] - parent 类别概率
                                     self.M_cp,            # [C+1, C] - 分布矩阵
                                     cls_probs_for_child)  # [B, L+1, C] - child 类别概率

            # 将 soft-mask 与注意力分数相乘（使用 log 空间相加避免数值下溢）
            attention_scores = attention_scores + torch.log(soft_mask.clamp(min=1e-10))

        # 6. 创建因果mask（只能选择之前的单元作为父节点）
        # 对于位置i，只有位置0到i-1可以作为候选父节点
        # 注意：现在序列长度是 L+1
        seq_len = max_lines + 1
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        causal_mask = causal_mask.bool()

        # 应用 causal mask
        attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # 7. 应用 line_mask（忽略无效位置）
        # ROOT 始终有效，其余行按 line_mask_with_root 判断
        parent_mask = ~line_mask_with_root.unsqueeze(1)  # [B, 1, L+1]
        child_mask = ~line_mask_with_root.unsqueeze(2)   # [B, L+1, 1]
        combined_mask = parent_mask | child_mask

        attention_scores = attention_scores.masked_fill(combined_mask, float('-inf'))

        return attention_scores  # [B, L+1, L+1] - 父节点logits（包括ROOT）


def collate_fn_simple(batch):
    """简化版collate函数：样本对格式"""
    max_candidates = max(item["parent_feats"].shape[0] for item in batch)
    batch_size = len(batch)
    hidden_size = batch[0]["child_feat"].shape[0]

    child_feats = torch.stack([item["child_feat"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    parent_feats = torch.zeros(batch_size, max_candidates, hidden_size)
    geom_feats = torch.zeros(batch_size, max_candidates, 4)  # 几何特征是4维
    masks = torch.zeros(batch_size, max_candidates, dtype=torch.bool)

    for i, item in enumerate(batch):
        num_cands = item["parent_feats"].shape[0]
        parent_feats[i, :num_cands] = item["parent_feats"]
        geom_feats[i, :num_cands] = item["geom_feats"]
        masks[i, :num_cands] = True

    return {
        "child_feat": child_feats,
        "parent_feats": parent_feats,
        "geom_feats": geom_feats,
        "labels": labels,
        "masks": masks
    }


def collate_fn(batch, max_lines_limit=512):
    """
    自定义 collate 函数，处理可变长度的序列

    支持页面级别和文档级别训练：
    - 页面级别: 每个样本是一页，max_lines_limit=256 足够
    - 文档级别: 每个样本是整个文档，max_lines_limit=512 支持跨页关系

    Args:
        batch: batch 数据
        max_lines_limit: 最大行数限制（防止极端情况导致显存爆炸）
                        推荐：页面级别=256, 文档级别=512
    """
    # 找出 batch 中最大的行数，但限制在 max_lines_limit 以内
    max_lines = min(
        max(item["line_features"].shape[0] for item in batch),
        max_lines_limit
    )

    batch_size = len(batch)
    hidden_size = batch[0]["line_features"].shape[1]

    # 初始化 padded tensors
    line_features = torch.zeros(batch_size, max_lines, hidden_size)
    line_mask = torch.zeros(batch_size, max_lines, dtype=torch.bool)
    line_parent_ids = torch.full((batch_size, max_lines), -1, dtype=torch.long)
    line_labels = torch.full((batch_size, max_lines), -1, dtype=torch.long)  # 语义标签

    for i, item in enumerate(batch):
        # 截断到 max_lines
        num_lines = min(item["line_features"].shape[0], max_lines)
        line_features[i, :num_lines] = item["line_features"][:num_lines]
        line_mask[i, :num_lines] = item["line_mask"][:num_lines]

        # line_parent_ids 可能比 line_features 更长，截断或padding
        parent_ids = item["line_parent_ids"]
        actual_len = min(len(parent_ids), num_lines)
        line_parent_ids[i, :actual_len] = parent_ids[:actual_len]

        # line_labels（如果存在）
        if "line_labels" in item:
            labels = item["line_labels"]
            actual_len_labels = min(len(labels), num_lines)
            line_labels[i, :actual_len_labels] = torch.tensor(labels[:actual_len_labels], dtype=torch.long)

    return {
        "line_features": line_features,
        "line_mask": line_mask,
        "line_parent_ids": line_parent_ids,
        "line_labels": line_labels
    }


class SimpleParentFinder(nn.Module):
    """
    简化版父节点查找器（内存友好）
    对每个 child，给候选 parents 打分，选择分数最高的
    """

    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()

        # 特征融合层
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size * 2 + 4, hidden_size),  # 4 是几何特征维度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # 输出一个分数
        )

    def forward(self, child_feat, parent_feats, geom_feats):
        """
        Args:
            child_feat: [batch_size, hidden_size]
            parent_feats: [batch_size, num_candidates, hidden_size]
            geom_feats: [batch_size, num_candidates, 4]

        Returns:
            scores: [batch_size, num_candidates]
        """
        batch_size, num_candidates, hidden_size = parent_feats.shape

        # 扩展 child_feat 到每个候选
        child_feat_expanded = child_feat.unsqueeze(1).expand(batch_size, num_candidates, hidden_size)

        # 拼接特征
        combined = torch.cat([child_feat_expanded, parent_feats, geom_feats], dim=-1)

        # 计算分数
        scores = self.score_head(combined).squeeze(-1)  # [B, num_candidates]

        return scores


class ParentFinderDatasetSimple(torch.utils.data.Dataset):
    """
    简化版数据集：构造样本对
    每个样本是 (child, candidate_parents)
    """

    def __init__(
        self,
        features_dir: str,
        split: str = "train",
        max_chunks: int = None,
        max_samples_per_page: int = 10,
        max_candidates: int = 20
    ):
        self.max_samples_per_page = max_samples_per_page
        self.max_candidates = max_candidates

        # 加载缓存的特征
        import glob
        from layoutlmft.models.relation_classifier import compute_geometry_features

        single_file = os.path.join(features_dir, f"{split}_line_features.pkl")
        if os.path.exists(single_file):
            logger.info(f"加载单个特征文件: {single_file}")
            with open(single_file, "rb") as f:
                self.page_features = pickle.load(f)
        else:
            pattern = os.path.join(features_dir, f"{split}_line_features_chunk_*.pkl")
            chunk_files = sorted(glob.glob(pattern))

            if max_chunks is not None:
                chunk_files = chunk_files[:max_chunks]

            if len(chunk_files) == 0:
                raise ValueError(f"没有找到特征文件: {single_file} 或 {pattern}")

            logger.info(f"找到 {len(chunk_files)} 个chunk文件")
            self.page_features = []
            for chunk_file in chunk_files:
                logger.info(f"  加载 {os.path.basename(chunk_file)}...")
                with open(chunk_file, "rb") as f:
                    chunk_data = pickle.load(f)
                self.page_features.extend(chunk_data)
                logger.info(f"    累计 {len(self.page_features)} 页")

        logger.info(f"总共加载了 {len(self.page_features)} 页")

        # 构造样本
        logger.info("构造训练样本...")
        self.samples = []

        for page_data in tqdm(self.page_features, desc="构造样本"):
            line_features = page_data["line_features"].squeeze(0)
            line_mask = page_data["line_mask"].squeeze(0)
            line_parent_ids = page_data["line_parent_ids"]
            line_bboxes = page_data["line_bboxes"]

            num_lines = line_mask.sum().item()

            # 随机采样一些 child
            valid_child_indices = []
            for child_idx in range(1, num_lines):
                parent_idx = line_parent_ids[child_idx]
                if parent_idx >= 0 and parent_idx < num_lines:
                    valid_child_indices.append(child_idx)

            if len(valid_child_indices) > self.max_samples_per_page:
                sampled_indices = random.sample(valid_child_indices, self.max_samples_per_page)
            else:
                sampled_indices = valid_child_indices

            for child_idx in sampled_indices:
                parent_idx_gt = line_parent_ids[child_idx]

                if parent_idx_gt < 0 or parent_idx_gt >= child_idx:
                    continue

                # 候选父节点
                max_cands = min(self.max_candidates, child_idx)
                candidate_start = max(0, child_idx - max_cands)
                candidate_indices = list(range(candidate_start, child_idx))

                if len(candidate_indices) == 0:
                    continue

                # 提取特征
                child_feat = line_features[child_idx]
                parent_feats = line_features[candidate_indices]

                # 几何特征
                child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                geom_feats = []
                for cand_idx in candidate_indices:
                    cand_bbox = torch.tensor(line_bboxes[cand_idx], dtype=torch.float32)
                    geom_feat = compute_geometry_features(cand_bbox, child_bbox)
                    geom_feats.append(geom_feat)
                geom_feats = torch.stack(geom_feats, dim=0)

                # 找到 ground truth 在候选列表中的索引
                if parent_idx_gt in candidate_indices:
                    label = candidate_indices.index(parent_idx_gt)
                else:
                    continue

                self.samples.append({
                    "child_feat": child_feat,
                    "parent_feats": parent_feats,
                    "geom_feats": geom_feats,
                    "label": label
                })

        logger.info(f"总样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ParentFinderDataset(torch.utils.data.Dataset):
    """
    完整版数据集：支持页面级别和文档级别（全量加载，适合小数据集）

    数据级别：
    - 页面级别: 每个样本是一页，parent_ids 是页内局部索引
    - 文档级别: 每个样本是整个文档（多页），parent_ids 是文档内全局索引，支持跨页关系

    特征文件由 extract_line_features.py (页面级别) 或
    extract_line_features_document_level.py (文档级别) 生成
    """

    def __init__(
        self,
        features_dir: str,
        split: str = "train",
        max_chunks: int = None
    ):
        # 加载缓存的特征
        import glob

        single_file = os.path.join(features_dir, f"{split}_line_features.pkl")
        if os.path.exists(single_file):
            logger.info(f"加载单个特征文件: {single_file}")
            with open(single_file, "rb") as f:
                self.page_features = pickle.load(f)
        else:
            pattern = os.path.join(features_dir, f"{split}_line_features_chunk_*.pkl")
            chunk_files = sorted(glob.glob(pattern))

            if max_chunks is not None:
                chunk_files = chunk_files[:max_chunks]

            if len(chunk_files) == 0:
                raise ValueError(f"没有找到特征文件: {single_file} 或 {pattern}")

            logger.info(f"找到 {len(chunk_files)} 个chunk文件")
            self.page_features = []
            for chunk_file in chunk_files:
                logger.info(f"  加载 {os.path.basename(chunk_file)}...")
                with open(chunk_file, "rb") as f:
                    chunk_data = pickle.load(f)
                self.page_features.extend(chunk_data)
                logger.info(f"    累计 {len(self.page_features)} 页")

        logger.info(f"总共加载了 {len(self.page_features)} 页")

    def __len__(self):
        return len(self.page_features)

    def __getitem__(self, idx):
        page_data = self.page_features[idx]

        line_features = page_data["line_features"].squeeze(0)  # [max_lines, H]
        line_mask = page_data["line_mask"].squeeze(0)  # [max_lines]
        line_parent_ids = torch.tensor(page_data["line_parent_ids"], dtype=torch.long)

        return {
            "line_features": line_features,
            "line_mask": line_mask,
            "line_parent_ids": line_parent_ids
        }


class ChunkIterableDataset(torch.utils.data.IterableDataset):
    """
    基于 chunk 的流式数据集（内存友好，适合大数据集）
    逐个加载 chunk，避免一次性占用大量内存

    支持页面级别和文档级别训练：
    - 页面级别: 每个样本是一页
    - 文档级别: 每个样本是整个文档，支持跨页父子关系
    """

    def __init__(
        self,
        features_dir: str,
        split: str = "train",
        max_chunks: int = None,
        shuffle: bool = True,
        seed: int = 42
    ):
        import glob

        self.features_dir = features_dir
        self.split = split
        self.shuffle = shuffle
        self.seed = seed

        # 查找 chunk 文件
        single_file = os.path.join(features_dir, f"{split}_line_features.pkl")
        if os.path.exists(single_file):
            # 单文件模式，转换为列表
            self.chunk_files = [single_file]
        else:
            pattern = os.path.join(features_dir, f"{split}_line_features_chunk_*.pkl")
            self.chunk_files = sorted(glob.glob(pattern))

            if max_chunks is not None:
                self.chunk_files = self.chunk_files[:max_chunks]

            if len(self.chunk_files) == 0:
                raise ValueError(f"没有找到特征文件: {single_file} 或 {pattern}")

        logger.info(f"[ChunkIterable] 找到 {len(self.chunk_files)} 个chunk文件")
        logger.info(f"[ChunkIterable] 流式加载模式，内存占用低")

        # 统计总页面数（用于显示）
        self.total_pages = 0
        for chunk_file in self.chunk_files:
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)
                self.total_pages += len(chunk_data)

        logger.info(f"[ChunkIterable] 总计 {self.total_pages} 页")

    def process_page(self, page_data):
        """处理单个页面数据"""
        line_features = page_data["line_features"].squeeze(0)
        line_mask = page_data["line_mask"].squeeze(0)
        line_parent_ids = torch.tensor(page_data["line_parent_ids"], dtype=torch.long)

        return {
            "line_features": line_features,
            "line_mask": line_mask,
            "line_parent_ids": line_parent_ids
        }

    def __iter__(self):
        # 获取 worker info（支持多进程）
        worker_info = torch.utils.data.get_worker_info()

        # 打乱 chunk 顺序（每个 epoch）
        chunk_files = self.chunk_files.copy()
        if self.shuffle:
            # 使用固定 seed + epoch 保证可复现
            rng = random.Random(self.seed + torch.initial_seed())
            rng.shuffle(chunk_files)

        # 如果有多个 worker，分配 chunk
        if worker_info is not None:
            # 分配给当前 worker 的 chunk
            per_worker = int(np.ceil(len(chunk_files) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(chunk_files))
            chunk_files = chunk_files[start:end]

        # 逐个 chunk 加载和返回
        for chunk_file in chunk_files:
            # 加载当前 chunk
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)

            # 打乱 chunk 内的页面
            if self.shuffle:
                rng = random.Random(self.seed + torch.initial_seed())
                rng.shuffle(chunk_data)

            # 逐个返回页面
            for page_data in chunk_data:
                yield self.process_page(page_data)

            # chunk_data 离开作用域，内存自动释放


def train_epoch_simple(model, dataloader, optimizer, criterion, device):
    """简化版训练epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="训练"):
        child_feat = batch["child_feat"].to(device)
        parent_feats = batch["parent_feats"].to(device)
        geom_feats = batch["geom_feats"].to(device)
        labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)

        optimizer.zero_grad()

        # 前向传播
        scores = model(child_feat, parent_feats, geom_feats)
        scores = scores.masked_fill(~masks, float('-inf'))

        # 计算损失
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(scores, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate_simple(model, dataloader, device):
    """简化版评估"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            child_feat = batch["child_feat"].to(device)
            parent_feats = batch["parent_feats"].to(device)
            geom_feats = batch["geom_feats"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["masks"].to(device)

            scores = model(child_feat, parent_feats, geom_feats)
            scores = scores.masked_fill(~masks, float('-inf'))

            preds = torch.argmax(scores, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_preds, all_labels)
    return accuracy


def train_epoch(model, dataloader, optimizer, criterion, device):
    """完整版训练epoch（GRU）"""
    model.train()
    total_loss = 0
    total_parent_correct = 0
    total_parent_count = 0
    num_batches = 0  # 手动计数 batch 数量（IterableDataset 不支持 len）

    for batch in tqdm(dataloader, desc="训练"):
        line_features = batch["line_features"].to(device)
        line_mask = batch["line_mask"].to(device)
        line_parent_ids = batch["line_parent_ids"].to(device)

        optimizer.zero_grad()

        # 前向传播
        parent_logits = model(line_features, line_mask)

        # 计算损失
        # parent_logits: [B, L+1, L+1] - 包括ROOT节点
        # line_parent_ids: [B, L] - 每个位置的父节点索引（不包括ROOT）

        batch_size = parent_logits.shape[0]
        num_lines = line_features.shape[1]  # 原始行数（不包括ROOT）

        # 只计算有效位置的损失
        valid_mask = line_mask  # [B, L]

        # 使用累加而非列表收集，避免大量计算图堆积
        batch_loss = 0.0
        total_loss_value = 0.0
        correct = 0
        count = 0

        for b in range(batch_size):
            for i in range(num_lines):
                if not valid_mask[b, i]:
                    continue

                # 行 i 在新序列中的位置是 i+1（因为 ROOT 在位置 0）
                # parent_logits[b, i+1, :i+2] 包含：
                #   - 位置 0: ROOT
                #   - 位置 1 到 i+1: 行 0 到 i
                # 但行 i 只能选择 0 到 i-1 作为父节点（不能选自己），加上 ROOT
                # 所以取 parent_logits[b, i+1, :i+1]
                logits_i = parent_logits[b, i+1, :i+1]  # [0:i+1] 包含 ROOT 和行 0~i-1
                target_i = line_parent_ids[b, i]

                # 映射：line_parent_ids -> logits索引
                # parent_id = -1 (ROOT) -> index = 0
                # parent_id = j (行j)    -> index = j+1
                if target_i == -1:
                    target_idx = 0  # ROOT
                else:
                    target_idx = target_i + 1  # 行j在新序列中的位置是j+1

                # 检查 target_idx 是否在有效范围内
                if target_idx < 0 or target_idx > i:
                    # 父节点无效或在当前位置之后，跳过
                    continue

                # 检查 logits 是否有效（避免全是 -inf 导致 NaN）
                if torch.isinf(logits_i).all() or torch.isnan(logits_i).any():
                    continue

                # 再次检查索引（防御性编程）
                if target_idx >= len(logits_i):
                    continue

                # 交叉熵损失
                loss_i = F.cross_entropy(
                    logits_i.unsqueeze(0),
                    torch.tensor([target_idx], device=device)
                )

                # 检查损失是否有效
                if torch.isnan(loss_i) or torch.isinf(loss_i):
                    continue

                # 累加损失（避免 torch.stack 导致的内存问题）
                batch_loss = batch_loss + loss_i
                total_loss_value += loss_i.item()

                # 统计准确率
                pred_idx = torch.argmax(logits_i).item()
                if pred_idx == target_idx:
                    correct += 1
                count += 1

        # 计算平均损失并反向传播
        if count > 0:
            loss = batch_loss / count
            loss.backward()
        else:
            # 没有有效样本，跳过
            continue

        # 执行优化步骤
        optimizer.step()

        total_loss += total_loss_value / max(1, count)  # 当前批次的平均loss
        total_parent_correct += correct
        total_parent_count += count
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = total_parent_correct / total_parent_count if total_parent_count > 0 else 0

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_parent_correct = 0
    total_parent_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            line_features = batch["line_features"].to(device)
            line_mask = batch["line_mask"].to(device)
            line_parent_ids = batch["line_parent_ids"].to(device)

            # 前向传播
            parent_logits = model(line_features, line_mask)

            # parent_logits: [B, L+1, L+1] - 包括ROOT节点
            # line_parent_ids: [B, L] - 每个位置的父节点索引（不包括ROOT）

            batch_size = parent_logits.shape[0]
            num_lines = line_features.shape[1]  # 原始行数（不包括ROOT）
            valid_mask = line_mask  # [B, L]

            for b in range(batch_size):
                for i in range(num_lines):
                    if not valid_mask[b, i]:
                        continue

                    # 行 i 在新序列中的位置是 i+1（因为 ROOT 在位置 0）
                    # parent_logits[b, i+1, :i+1] 包含 ROOT 和行 0~i-1
                    logits_i = parent_logits[b, i+1, :i+1]
                    target_i = line_parent_ids[b, i]

                    # 映射：line_parent_ids -> logits索引
                    # parent_id = -1 (ROOT) -> index = 0
                    # parent_id = j (行j)    -> index = j+1
                    if target_i == -1:
                        target_idx = 0  # ROOT
                    else:
                        target_idx = target_i + 1  # 行j在新序列中的位置是j+1

                    # 检查 target_idx 是否在有效范围内
                    if target_idx < 0 or target_idx > i:
                        continue

                    # 跳过全是-inf或NaN的情况
                    if torch.isinf(logits_i).all() or torch.isnan(logits_i).any():
                        continue

                    # 再次检查索引（防御性编程）
                    if target_idx >= len(logits_i):
                        continue

                    pred_idx = torch.argmax(logits_i).item()
                    if pred_idx == target_idx:
                        total_parent_correct += 1
                    total_parent_count += 1

    accuracy = total_parent_correct / total_parent_count if total_parent_count > 0 else 0

    return accuracy


def build_child_parent_matrix(features_dir, split="train", num_classes=16):
    """构建 Child-Parent Distribution Matrix"""

    logger.info("构建 Child-Parent Distribution Matrix...")

    cp_matrix = ChildParentDistributionMatrix(num_classes=num_classes)

    # 加载特征文件
    import glob

    single_file = os.path.join(features_dir, f"{split}_line_features.pkl")
    if os.path.exists(single_file):
        with open(single_file, "rb") as f:
            page_features = pickle.load(f)

        # 统计父子关系
        for page_data in tqdm(page_features, desc="统计"):
            line_parent_ids = page_data["line_parent_ids"]

            if "line_labels" not in page_data:
                continue

            line_labels = page_data["line_labels"]

            for child_idx, parent_idx in enumerate(line_parent_ids):
                if child_idx >= len(line_labels):
                    continue

                child_label = line_labels[child_idx]
                parent_label = line_labels[parent_idx] if parent_idx >= 0 and parent_idx < len(line_labels) else -1

                cp_matrix.update(child_label, parent_label)
    else:
        # 流式处理：逐个chunk加载，避免一次性占用大量内存
        pattern = os.path.join(features_dir, f"{split}_line_features_chunk_*.pkl")
        chunk_files = sorted(glob.glob(pattern))

        logger.info(f"流式处理 {len(chunk_files)} 个chunk文件...")
        for chunk_file in tqdm(chunk_files, desc="处理chunk"):
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)

            # 立即处理当前chunk，处理完后chunk_data会被释放
            for page_data in chunk_data:
                line_parent_ids = page_data["line_parent_ids"]

                if "line_labels" not in page_data:
                    continue

                line_labels = page_data["line_labels"]

                for child_idx, parent_idx in enumerate(line_parent_ids):
                    if child_idx >= len(line_labels):
                        continue

                    child_label = line_labels[child_idx]
                    parent_label = line_labels[parent_idx] if parent_idx >= 0 and parent_idx < len(line_labels) else -1

                    cp_matrix.update(child_label, parent_label)

            # chunk_data 离开作用域，内存自动释放

    # 构建矩阵
    cp_matrix.build()

    return cp_matrix


def main():
    import argparse

    parser = argparse.ArgumentParser(description="训练父节点查找器（任务2）")
    parser.add_argument("--mode", type=str, default="simple", choices=["simple", "full"],
                        help="训练模式：simple=简化版（内存友好），full=完整论文方法（需要大显存）")
    parser.add_argument("--features_dir", type=str, default=None,
                        help="特征文件目录（默认：本机=/mnt/e/models/train_data/layoutlmft/line_features）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认：本机=/mnt/e/models/train_data/layoutlmft）")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批大小（simple默认128，full默认2）")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="学习率（simple默认1e-3，full默认1e-4）")
    parser.add_argument("--max_chunks", type=int, default=-1,
                        help="加载的chunk数量（-1=全部，用于测试时可设为1）")
    parser.add_argument("--use_soft_mask", action="store_true",
                        help="是否使用soft-mask（需要语义标签，仅full模式）")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="是否使用梯度检查点（节省显存，稍慢）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数（模拟更大的batch size）")
    parser.add_argument("--level", type=str, default="document", choices=["page", "document"],
                        help="训练级别：page=页面级别，document=文档级别（支持跨页关系）")
    parser.add_argument("--max_lines_limit", type=int, default=None,
                        help="最大行数限制（page级别默认256，document级别默认512）")

    args = parser.parse_args()

    # 配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 路径配置：优先级 命令行参数 > 环境变量 > 默认值
    # 默认使用文档级别特征目录 (line_features_doc)
    if args.features_dir:
        features_dir = args.features_dir
    else:
        features_dir = os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features_doc")

    if args.output_dir:
        output_dir_base = args.output_dir
    else:
        output_dir_base = os.getenv("LAYOUTLMFT_OUTPUT_DIR", "/mnt/e/models/train_data/layoutlmft")

    output_dir = os.path.join(output_dir_base, f"parent_finder_{args.mode}")

    # 根据训练级别设置 max_lines_limit
    if args.max_lines_limit is not None:
        max_lines_limit = args.max_lines_limit
    else:
        max_lines_limit = 512 if args.level == "document" else 256

    # 根据模式设置默认参数
    if args.mode == "simple":
        batch_size = args.batch_size if args.batch_size is not None else 128
        learning_rate = args.learning_rate if args.learning_rate is not None else 1e-3
        use_gru = False
        use_soft_mask = False
        logger.info("=" * 60)
        logger.info(f"模式: 简化版（内存友好，适合本地4GB显存测试）")
        logger.info(f"级别: {args.level} (max_lines={max_lines_limit})")
        logger.info("=" * 60)
    else:  # full
        batch_size = args.batch_size if args.batch_size is not None else 1
        learning_rate = args.learning_rate if args.learning_rate is not None else 1e-4
        use_gru = True
        use_soft_mask = args.use_soft_mask
        logger.info("=" * 60)
        logger.info(f"模式: 完整论文方法（需要24GB显存）")
        logger.info(f"级别: {args.level} (max_lines={max_lines_limit})")
        logger.info("=" * 60)

    num_epochs = args.num_epochs
    num_classes = 16
    max_chunks = args.max_chunks if args.max_chunks > 0 else None
    gradient_checkpointing = args.gradient_checkpointing

    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 构建 Child-Parent Distribution Matrix（已禁用）
    # cp_matrix_path = os.path.join(output_dir, "child_parent_matrix.npy")
    # if os.path.exists(cp_matrix_path):
    #     logger.info(f"加载已有的 M_cp: {cp_matrix_path}")
    #     cp_matrix = ChildParentDistributionMatrix(num_classes=num_classes)
    #     cp_matrix.load(cp_matrix_path)
    # else:
    #     cp_matrix = build_child_parent_matrix(features_dir, split="train", num_classes=num_classes)
    #     cp_matrix.save(cp_matrix_path)

    # 根据模式创建数据集和模型
    if args.mode == "simple":
        # 简化版
        train_dataset = ParentFinderDatasetSimple(features_dir, split="train", max_chunks=max_chunks)
        val_dataset = ParentFinderDatasetSimple(features_dir, split="validation", max_chunks=max_chunks)

        logger.info(f"训练集: {len(train_dataset)} 样本")
        logger.info(f"验证集: {len(val_dataset)} 样本")

        # 创建dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_simple
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn_simple
        )

        # 创建简化模型
        model = SimpleParentFinder(hidden_size=768, dropout=0.1).to(device)

        # 训练和评估函数
        train_fn = train_epoch_simple
        eval_fn = evaluate_simple

    else:  # full
        # 完整版 - 使用流式加载（内存友好）
        train_dataset = ChunkIterableDataset(
            features_dir,
            split="train",
            max_chunks=max_chunks,
            shuffle=True,  # 在 Dataset 内部打乱
            seed=42
        )
        val_dataset = ChunkIterableDataset(
            features_dir,
            split="validation",
            max_chunks=max_chunks,
            shuffle=False,  # 验证集不打乱
            seed=42
        )

        logger.info(f"训练集: {train_dataset.total_pages} 页（流式加载）")
        logger.info(f"验证集: {val_dataset.total_pages} 页（流式加载）")

        # 创建dataloader（IterableDataset 不支持 shuffle 参数）
        # 使用 partial 设置最大行数限制（防止极端情况导致显存爆炸）
        # 文档级别: 512行，页面级别: 256行
        collate_fn_with_limit = partial(collate_fn, max_lines_limit=max_lines_limit)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,  # IterableDataset 建议 num_workers=0
            collate_fn=collate_fn_with_limit
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_fn_with_limit
        )

        # 创建GRU模型
        model = ParentFinderGRU(
            hidden_size=768,
            gru_hidden_size=512,
            num_classes=num_classes,
            dropout=0.1,
            use_soft_mask=use_soft_mask
        ).to(device)

        # 设置 M_cp（如果启用）
        if use_soft_mask:
            logger.info("构建 Child-Parent Distribution Matrix...")
            cp_matrix = build_child_parent_matrix(features_dir, split="train", num_classes=num_classes)
            model.set_child_parent_matrix(cp_matrix.get_tensor(device))

        # 训练和评估函数
        train_fn = train_epoch
        eval_fn = evaluate

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练
    logger.info(f"\n开始训练...")
    best_acc = 0

    for epoch in range(num_epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        # 使用选择的训练函数
        if args.mode == "simple":
            train_loss, train_acc = train_fn(model, train_loader, optimizer, criterion, device)
            logger.info(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            val_acc = eval_fn(model, val_loader, device)
            logger.info(f"验证 - Acc: {val_acc:.4f}")
        else:  # full
            train_loss, train_acc = train_fn(model, train_loader, optimizer, criterion, device)
            logger.info(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            val_acc = eval_fn(model, val_loader, device)
            logger.info(f"验证 - Acc: {val_acc:.4f}")

        # 保存checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }
            best_model_path = os.path.join(output_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            logger.info(f"✓ 保存最佳模型 (Acc: {best_acc:.4f})")

    logger.info(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()

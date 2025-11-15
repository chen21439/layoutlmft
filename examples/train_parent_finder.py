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
            line_labels: 语义类别标签 [B, L]（训练时可提供ground truth）

        Returns:
            parent_logits: [B, L, L] - 每个位置i的父节点logits（对位置0到i-1）
            cls_logits: [B, L, num_classes] - 语义类别预测logits
        """
        batch_size, max_lines, _ = line_features.shape
        device = line_features.device

        # 1. 通过 GRU 获取隐藏状态
        # h_i 包含了从开始到第i个单元的上下文信息
        gru_output, _ = self.gru(line_features)  # [B, L, gru_hidden]

        # 2. 预测每个单元的语义类别概率（仅用于 soft-mask）
        # 简化版本：不使用语义类别，直接基于特征计算父节点
        # cls_logits = self.cls_head(line_features)  # [B, L, num_classes]
        # cls_probs = F.softmax(cls_logits, dim=-1)  # [B, L, num_classes]

        # 3. 计算父节点概率
        # 对每个位置 i，计算它与之前所有位置 j (0 <= j < i) 的父子概率

        # 查询向量（当前单元）
        query = self.query_proj(gru_output)  # [B, L, gru_hidden]

        # 键向量（候选父节点）
        key = self.key_proj(gru_output)  # [B, L, gru_hidden]

        # 注意力分数: alpha(q_i, h_j)
        # 使用高效的 bmm 避免创建巨大的中间张量
        # [B, L, L] = [B, L, gru_hidden] @ [B, gru_hidden, L]
        attention_scores = torch.bmm(
            query,  # [B, L, gru_hidden]
            key.transpose(1, 2)  # [B, gru_hidden, L]
        ) / (self.gru_hidden_size ** 0.5)  # [B, L, L]

        # 4. Soft-mask 操作（已禁用 - 需要语义标签）
        # 简化版本：不使用 soft-mask，直接基于注意力分数预测父节点
        # if self.use_soft_mask and self.M_cp is not None:
        #     ... (soft-mask 代码已省略)

        # 5. 创建因果mask（只能选择之前的单元作为父节点）
        # 对于位置i，只有位置0到i-1可以作为候选父节点
        causal_mask = torch.triu(torch.ones(max_lines, max_lines, device=device), diagonal=1)
        causal_mask = causal_mask.bool()

        # 应用 causal mask
        attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # 6. 应用 line_mask（忽略无效位置）
        # 如果 parent j 或 child i 无效，则mask掉
        parent_mask = ~line_mask.unsqueeze(1)  # [B, 1, L]
        child_mask = ~line_mask.unsqueeze(2)   # [B, L, 1]
        combined_mask = parent_mask | child_mask

        attention_scores = attention_scores.masked_fill(combined_mask, float('-inf'))

        # 7. 对于第一个位置（i=0），强制其父节点为ROOT（假设ROOT在位置0）
        # 这里我们假设ROOT就是位置0，可以调整

        return attention_scores  # [B, L, L] - 父节点logits


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


def collate_fn(batch, max_lines_limit=256):
    """
    自定义 collate 函数，处理可变长度的序列

    Args:
        batch: batch 数据
        max_lines_limit: 最大行数限制（防止极端情况导致显存爆炸）
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

    for i, item in enumerate(batch):
        # 截断到 max_lines
        num_lines = min(item["line_features"].shape[0], max_lines)
        line_features[i, :num_lines] = item["line_features"][:num_lines]
        line_mask[i, :num_lines] = item["line_mask"][:num_lines]

        # line_parent_ids 可能比 line_features 更长，截断或padding
        parent_ids = item["line_parent_ids"]
        actual_len = min(len(parent_ids), num_lines)
        line_parent_ids[i, :actual_len] = parent_ids[:actual_len]

    return {
        "line_features": line_features,
        "line_mask": line_mask,
        "line_parent_ids": line_parent_ids
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
    完整版数据集：页面级别（全量加载，适合小数据集）
    每个样本是一个文档页面，包含多个语义单元（lines）
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
        # parent_logits: [B, L, L]
        # line_parent_ids: [B, L] - 每个位置的父节点索引

        batch_size, max_lines, _ = parent_logits.shape

        # 只计算有效位置的损失
        valid_mask = line_mask  # [B, L]

        losses = []  # 收集所有损失，最后一次性计算
        correct = 0
        count = 0

        for b in range(batch_size):
            for i in range(max_lines):
                if not valid_mask[b, i]:
                    continue

                # 位置i的父节点预测
                # 注意：我们将位置0视为隐式ROOT
                if i == 0:
                    # 第一个位置的父节点应该是ROOT，跳过
                    continue

                logits_i = parent_logits[b, i, :i]  # 只取0到i-1的候选父节点
                target_i = line_parent_ids[b, i]

                # 映射：parent_id直接对应logits索引
                # parent_id=0 -> index=0
                # parent_id=1 -> index=1
                # ...
                # parent_id=-1 (ROOT) 视为position 0
                target_idx = target_i if target_i >= 0 else 0

                # 检查 target_idx 是否在有效范围内
                if target_idx < 0 or target_idx >= i:
                    # 父节点无效或在当前位置之后，跳过
                    continue

                # 检查 logits 是否有效（避免全是 -inf 导致 NaN）
                if torch.isinf(logits_i).all() or torch.isnan(logits_i).any():
                    continue

                # 再次检查索引（防御性编程）
                if target_idx >= len(logits_i):
                    continue

                # 交叉熵损失（收集而不是累加）
                loss_i = F.cross_entropy(
                    logits_i.unsqueeze(0),
                    torch.tensor([target_idx], device=device)
                )

                # 检查损失是否有效
                if torch.isnan(loss_i) or torch.isinf(loss_i):
                    continue

                losses.append(loss_i)

                # 统计准确率
                pred_idx = torch.argmax(logits_i).item()
                if pred_idx == target_idx:
                    correct += 1
                count += 1

        if len(losses) > 0:
            # 一次性计算平均损失，避免巨大计算图
            loss = torch.stack(losses).mean()
        else:
            continue

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
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

            batch_size, max_lines, _ = parent_logits.shape
            valid_mask = line_mask

            for b in range(batch_size):
                for i in range(max_lines):
                    if not valid_mask[b, i]:
                        continue

                    # 跳过第一个位置（隐式ROOT）
                    if i == 0:
                        continue

                    logits_i = parent_logits[b, i, :i]  # 只取0到i-1的候选父节点
                    target_i = line_parent_ids[b, i]
                    target_idx = target_i if target_i >= 0 else 0

                    # 检查有效性
                    if target_idx < 0 or target_idx >= i:
                        continue

                    # 跳过全是-inf或NaN的情况
                    if torch.isinf(logits_i).all() or torch.isnan(logits_i).any():
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

    args = parser.parse_args()

    # 配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 路径配置：优先级 命令行参数 > 环境变量 > 默认值
    if args.features_dir:
        features_dir = args.features_dir
    else:
        features_dir = os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features")

    if args.output_dir:
        output_dir_base = args.output_dir
    else:
        output_dir_base = os.getenv("LAYOUTLMFT_OUTPUT_DIR", "/mnt/e/models/train_data/layoutlmft")

    output_dir = os.path.join(output_dir_base, f"parent_finder_{args.mode}")

    # 根据模式设置默认参数
    if args.mode == "simple":
        batch_size = args.batch_size if args.batch_size is not None else 128
        learning_rate = args.learning_rate if args.learning_rate is not None else 1e-3
        use_gru = False
        use_soft_mask = False
        logger.info("=" * 60)
        logger.info("模式: 简化版（内存友好，适合本地4GB显存测试）")
        logger.info("=" * 60)
    else:  # full
        batch_size = args.batch_size if args.batch_size is not None else 2
        learning_rate = args.learning_rate if args.learning_rate is not None else 1e-4
        use_gru = True
        use_soft_mask = args.use_soft_mask
        logger.info("=" * 60)
        logger.info("模式: 完整论文方法（需要24GB显存）")
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
        # 使用 partial 设置最大行数限制为 256（防止极端情况导致显存爆炸）
        collate_fn_with_limit = partial(collate_fn, max_lines_limit=256)

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

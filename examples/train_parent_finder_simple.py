#!/usr/bin/env python
# coding=utf-8
"""
训练父节点查找器（简化版，参考任务3的成功方法）

与 train_multiclass_relation.py 类似的架构：
- 输入：(child_feat, candidate_parent_feat, geom_feat)
- 输出：parent index (多分类)

不使用 GRU，避免内存问题
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

# 从 relation_classifier 导入已有的工具
from layoutlmft.models.relation_classifier import compute_geometry_features

logger = logging.getLogger(__name__)


class SimpleParentFinder(nn.Module):
    """
    简单的父节点查找器
    对每个 child，给候选 parent 打分，选择分数最高的
    """

    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()

        # 特征融合层
        # child_feat + parent_feat + geom_feat -> score
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size * 2 + 8, hidden_size),  # 8 是几何特征维度
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
            geom_feats: [batch_size, num_candidates, 8]

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


class ParentFinderDataset(torch.utils.data.Dataset):
    """
    父节点查找数据集（样本对格式，类似任务3）
    """

    def __init__(
        self,
        features_dir: str,
        split: str = "train",
        max_chunks: int = None,
        max_samples_per_page: int = 10  # 每页最多采样多少个 child
    ):
        self.max_samples_per_page = max_samples_per_page

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

        # 构造样本
        logger.info("构造训练样本...")
        self.samples = []

        for page_data in tqdm(self.page_features, desc="构造样本"):
            line_features = page_data["line_features"].squeeze(0)  # [max_lines, H]
            line_mask = page_data["line_mask"].squeeze(0)  # [max_lines]
            line_parent_ids = page_data["line_parent_ids"]
            line_bboxes = page_data["line_bboxes"]

            num_lines = line_mask.sum().item()  # 有效行数

            # 随机采样一些 child（避免样本过多）
            valid_child_indices = []
            for child_idx in range(1, num_lines):  # 从1开始，因为0通常是ROOT
                parent_idx = line_parent_ids[child_idx]
                if parent_idx >= 0 and parent_idx < num_lines:
                    valid_child_indices.append(child_idx)

            # 随机采样
            if len(valid_child_indices) > self.max_samples_per_page:
                sampled_indices = random.sample(valid_child_indices, self.max_samples_per_page)
            else:
                sampled_indices = valid_child_indices

            for child_idx in sampled_indices:
                parent_idx_gt = line_parent_ids[child_idx]

                if parent_idx_gt < 0 or parent_idx_gt >= child_idx:
                    continue

                # 候选父节点：0 到 child_idx-1
                # 为了效率，限制候选数量（例如只考虑前面20个）
                max_candidates = min(20, child_idx)
                candidate_start = max(0, child_idx - max_candidates)

                candidate_indices = list(range(candidate_start, child_idx))

                if len(candidate_indices) == 0:
                    continue

                # 提取特征
                child_feat = line_features[child_idx]
                parent_feats = line_features[candidate_indices]  # [num_candidates, H]

                # 几何特征
                child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                geom_feats = []
                for cand_idx in candidate_indices:
                    cand_bbox = torch.tensor(line_bboxes[cand_idx], dtype=torch.float32)
                    geom_feat = compute_geometry_features(cand_bbox, child_bbox)
                    geom_feats.append(geom_feat)
                geom_feats = torch.stack(geom_feats, dim=0)  # [num_candidates, 8]

                # 找到 ground truth 在候选列表中的索引
                if parent_idx_gt in candidate_indices:
                    label = candidate_indices.index(parent_idx_gt)
                else:
                    # ground truth 不在候选列表中，跳过
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


def collate_fn(batch):
    """自定义 collate 函数"""
    # 找出最大候选数
    max_candidates = max(item["parent_feats"].shape[0] for item in batch)
    batch_size = len(batch)
    hidden_size = batch[0]["child_feat"].shape[0]

    child_feats = torch.stack([item["child_feat"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    # Padding parent_feats 和 geom_feats
    parent_feats = torch.zeros(batch_size, max_candidates, hidden_size)
    geom_feats = torch.zeros(batch_size, max_candidates, 8)
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


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
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
        scores = model(child_feat, parent_feats, geom_feats)  # [B, num_candidates]

        # Apply mask
        scores = scores.masked_fill(~masks, float('-inf'))

        # 计算损失
        loss = criterion(scores, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 收集预测
        preds = torch.argmax(scores, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估模型"""
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

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def main():
    # 配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    features_dir = os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features")
    output_dir = os.getenv("LAYOUTLMFT_OUTPUT_DIR", "/mnt/e/models/train_data/layoutlmft") + "/parent_finder_simple"

    num_epochs = 20
    batch_size = 128  # 样本对，可以更大
    learning_rate = 1e-3
    max_chunks = int(os.getenv("MAX_CHUNKS", "-1"))
    if max_chunks == -1:
        max_chunks = None

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

    # 创建数据集
    train_dataset = ParentFinderDataset(features_dir, split="train", max_chunks=max_chunks)
    val_dataset = ParentFinderDataset(features_dir, split="validation", max_chunks=max_chunks)

    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")

    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # 创建模型
    model = SimpleParentFinder(hidden_size=768, dropout=0.1).to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练
    logger.info(f"\n开始训练...")
    best_acc = 0

    for epoch in range(num_epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        val_acc = evaluate(model, val_loader, device)
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
    logger.info(f"模型保存路径: {output_dir}")


if __name__ == "__main__":
    main()

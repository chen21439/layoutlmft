#!/usr/bin/env python
# coding=utf-8
"""
训练关系分类器（方案C）
基于缓存的行级特征进行训练，快速迭代
"""

import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from layoutlmft.models.relation_classifier import (
    SimpleRelationClassifier,
    NegativeSampler,
    RELATION_LABELS,
    compute_geometry_features
)

logger = logging.getLogger(__name__)


class RelationDataset(torch.utils.data.Dataset):
    """
    关系分类数据集（支持chunk文件加载）
    """

    def __init__(
        self,
        features_dir: str,
        split: str = "train",
        neg_sampler: NegativeSampler = None,
        binary_only: bool = True,  # 是否只做二分类
        max_chunks: int = None  # 最多加载多少个chunk（用于测试）
    ):
        self.binary_only = binary_only

        # 加载缓存的特征（支持单个文件或chunk文件）
        import glob

        # 先尝试加载单个pkl文件（兼容旧格式）
        single_file = os.path.join(features_dir, f"{split}_line_features.pkl")
        if os.path.exists(single_file):
            logger.info(f"加载单个特征文件: {single_file}")
            with open(single_file, "rb") as f:
                self.page_features = pickle.load(f)
        else:
            # 加载chunk文件
            pattern = os.path.join(features_dir, f"{split}_line_features_chunk_*.pkl")
            chunk_files = sorted(glob.glob(pattern))

            if max_chunks is not None:
                chunk_files = chunk_files[:max_chunks]
                logger.info(f"限制加载前 {max_chunks} 个chunk文件")

            if len(chunk_files) == 0:
                raise ValueError(f"没有找到特征文件: {single_file} 或 {pattern}")

            logger.info(f"找到 {len(chunk_files)} 个chunk文件，开始加载...")
            self.page_features = []
            for chunk_file in chunk_files:
                logger.info(f"  加载 {os.path.basename(chunk_file)}...")
                with open(chunk_file, "rb") as f:
                    chunk_data = pickle.load(f)
                self.page_features.extend(chunk_data)
                logger.info(f"    已加载 {len(chunk_data)} 页，累计 {len(self.page_features)} 页")

        logger.info(f"总共加载了 {len(self.page_features)} 页的特征")

        # 构造样本对
        logger.info("构造训练样本对...")
        self.samples = []

        for page_data in tqdm(self.page_features, desc="构造样本"):
            line_features = page_data["line_features"].squeeze(0)  # [max_lines, H]
            line_mask = page_data["line_mask"].squeeze(0)  # [max_lines]
            line_parent_ids = page_data["line_parent_ids"]
            line_relations = page_data["line_relations"]
            line_bboxes = page_data["line_bboxes"]  # numpy array [num_lines, 4]

            num_lines = len(line_parent_ids)

            # 使用负采样器生成样本对
            pairs, labels = neg_sampler.sample_pairs(
                line_parent_ids, line_relations, num_lines
            )

            # 保存样本
            for (parent_idx, child_idx), label in zip(pairs, labels):
                # 确保索引有效（parent_idx和child_idx是原始的line_id，可能超出range）
                # 我们需要检查line_mask的长度
                max_idx = line_mask.shape[0]
                if parent_idx >= max_idx or child_idx >= max_idx:
                    continue
                if parent_idx < 0 or child_idx < 0:
                    continue
                if not line_mask[parent_idx] or not line_mask[child_idx]:
                    continue

                # 检查 bbox 是否有效
                if parent_idx >= len(line_bboxes) or child_idx >= len(line_bboxes):
                    continue

                # 计算几何特征
                parent_bbox = torch.tensor(line_bboxes[parent_idx], dtype=torch.float32)
                child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                geom_feat = compute_geometry_features(parent_bbox, child_bbox)

                self.samples.append({
                    "parent_feat": line_features[parent_idx],
                    "child_feat": line_features[child_idx],
                    "geom_feat": geom_feat,  # 新增几何特征
                    "label": label,  # 0=非父子, 1=父子
                    "relation": line_relations[child_idx] if label == 1 else "none"
                })

        logger.info(f"总样本数: {len(self.samples)}")
        pos_count = sum(1 for s in self.samples if s["label"] == 1)
        neg_count = len(self.samples) - pos_count
        logger.info(f"  正样本: {pos_count} ({pos_count/len(self.samples)*100:.1f}%)")
        logger.info(f"  负样本: {neg_count} ({neg_count/len(self.samples)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "parent_feat": sample["parent_feat"],
            "child_feat": sample["child_feat"],
            "geom_feat": sample["geom_feat"],  # 新增几何特征
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def train_epoch(model, dataloader, optimizer, criterion, device, log_interval=100):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # 用于打印中间loss
    running_loss = 0
    batch_count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="训练")):
        parent_feat = batch["parent_feat"].to(device)
        child_feat = batch["child_feat"].to(device)
        geom_feat = batch["geom_feat"].to(device)  # 新增几何特征
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # 前向传播（传入几何特征）
        logits = model(parent_feat, child_feat, geom_feat)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        batch_count += 1

        # 收集预测结果
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # 每log_interval步打印一次loss
        if (batch_idx + 1) % log_interval == 0:
            avg_running_loss = running_loss / batch_count
            logger.info(f"  Step [{batch_idx + 1}/{len(dataloader)}] - Loss: {avg_running_loss:.4f}")
            running_loss = 0
            batch_count = 0

    avg_loss = total_loss / len(dataloader)

    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            parent_feat = batch["parent_feat"].to(device)
            child_feat = batch["child_feat"].to(device)
            geom_feat = batch["geom_feat"].to(device)  # 新增几何特征
            labels = batch["label"].to(device)

            logits = model(parent_feat, child_feat, geom_feat)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    accuracy = accuracy_score(all_labels, all_preds)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, cm


def main():
    # 配置
    # 获取项目根目录（脚本在 examples/ 下，根目录是上一级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 从 E 盘读取特征文件（支持chunk加载），输出也到 E 盘
    features_dir = os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features")
    output_dir = os.getenv("LAYOUTLMFT_OUTPUT_DIR", "/mnt/e/models/train_data/layoutlmft") + "/relation_classifier"
    max_steps = 200  # 增加训练步数
    batch_size = 32
    learning_rate = 5e-4  # 降低学习率
    neg_ratio = 2  # 减少负样本比例：每个正样本2个负样本
    max_chunks = int(os.getenv("MAX_CHUNKS", "-1"))  # 限制chunk数量（-1表示加载全部）
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
    neg_sampler = NegativeSampler(
        neg_ratio=neg_ratio,
        same_page_only=True,
        before_child_only=True
    )

    # 创建训练集（使用train chunk）
    train_dataset = RelationDataset(
        features_dir=features_dir,
        split="train",
        neg_sampler=neg_sampler,
        binary_only=True,
        max_chunks=max_chunks
    )

    # 创建验证集（使用validation chunk，独立数据）
    val_dataset = RelationDataset(
        features_dir=features_dir,
        split="validation",
        neg_sampler=neg_sampler,
        binary_only=True,
        max_chunks=max_chunks
    )

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")

    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 创建模型
    model = SimpleRelationClassifier(
        hidden_size=768,
        use_geometry=True,  # 启用几何特征
        dropout=0.1
    ).to(device)

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 使用加权损失处理类别不平衡（只基于训练集）
    # 计算样本比例并反向加权
    neg_count = sum(1 for s in train_dataset.samples if s["label"] == 0)
    pos_count = sum(1 for s in train_dataset.samples if s["label"] == 1)
    pos_weight = torch.tensor([1.0, neg_count / pos_count]).to(device)
    logger.info(f"类别权重: [1.0, {neg_count / pos_count:.2f}]")
    criterion = nn.CrossEntropyLoss(weight=pos_weight)

    # 训练循环
    logger.info(f"\n开始训练（最多 {max_steps} 步）...")
    best_f1 = 0
    step = 0

    num_epochs = (max_steps // len(train_loader)) + 1

    for epoch in range(num_epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        # 训练（每100步打印一次loss）
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, log_interval=100
        )

        logger.info(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                   f"P: {train_prec:.4f}, R: {train_rec:.4f}, F1: {train_f1:.4f}")

        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, cm = evaluate(
            model, val_loader, criterion, device
        )

        logger.info(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                   f"P: {val_prec:.4f}, R: {val_rec:.4f}, F1: {val_f1:.4f}")
        logger.info(f"混淆矩阵:\n{cm}")

        # 保存checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch{epoch + 1}.pt")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_loss': val_loss,
            'val_metrics': {
                'accuracy': val_acc,
                'precision': val_prec,
                'recall': val_rec,
                'f1': val_f1
            }
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存checkpoint: {checkpoint_path}")

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_path = os.path.join(output_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            logger.info(f"✓ 保存最佳模型 (F1: {best_f1:.4f})")

            # 同时保存一个纯权重文件（兼容旧版加载方式）
            weights_path = os.path.join(output_dir, "best_model_weights.pt")
            torch.save(model.state_dict(), weights_path)

        step += len(train_loader)
        if step >= max_steps:
            break

    logger.info(f"\n训练完成！最佳验证F1: {best_f1:.4f}")
    logger.info(f"模型保存路径: {output_dir}")
    logger.info(f"  - best_model.pt (完整checkpoint)")
    logger.info(f"  - best_model_weights.pt (仅权重)")
    logger.info(f"  - checkpoint-epochN.pt (每个epoch)")


if __name__ == "__main__":
    main()

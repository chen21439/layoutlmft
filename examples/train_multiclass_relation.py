#!/usr/bin/env python
# coding=utf-8
"""
训练多分类关系分类器（方案C第二版）
基于缓存的行级特征，训练4类关系分类：Connect/Contain/Equality/None
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
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from layoutlmft.models.relation_classifier import (
    MultiClassRelationClassifier,
    FocalLoss,
    RELATION_LABELS,
    RELATION_NAMES,
    compute_geometry_features
)

logger = logging.getLogger(__name__)


class MultiClassRelationDataset(torch.utils.data.Dataset):
    """
    多分类关系数据集
    标签：0=none, 1=connect, 2=contain, 3=equality
    """

    def __init__(
        self,
        features_file: str,
        neg_ratio: float = 1.0  # 负样本比例（相对于正样本）
    ):
        self.neg_ratio = neg_ratio

        # 加载缓存的特征
        logger.info(f"加载特征文件: {features_file}")
        with open(features_file, "rb") as f:
            self.page_features = pickle.load(f)

        logger.info(f"加载了 {len(self.page_features)} 页的特征")

        # 构造样本对
        logger.info("构造多分类训练样本...")
        self.samples = []

        for page_data in tqdm(self.page_features, desc="构造样本"):
            line_features = page_data["line_features"].squeeze(0)  # [max_lines, H]
            line_mask = page_data["line_mask"].squeeze(0)  # [max_lines]
            line_parent_ids = page_data["line_parent_ids"]
            line_relations = page_data["line_relations"]
            line_bboxes = page_data["line_bboxes"]  # numpy array [num_lines, 4]

            num_lines = len(line_parent_ids)

            # 1. 收集所有正样本（有标注的关系）
            positive_pairs = []
            for child_idx in range(num_lines):
                parent_idx = line_parent_ids[child_idx]
                relation = line_relations[child_idx]

                # 跳过无效样本
                if parent_idx < 0 or parent_idx >= num_lines:
                    continue
                if relation not in RELATION_LABELS:
                    continue
                if relation == "none" or relation == "meta":
                    continue

                # 检查mask和bbox有效性
                max_idx = line_mask.shape[0]
                if parent_idx >= max_idx or child_idx >= max_idx:
                    continue
                if not line_mask[parent_idx] or not line_mask[child_idx]:
                    continue
                if parent_idx >= len(line_bboxes) or child_idx >= len(line_bboxes):
                    continue

                # 计算几何特征
                parent_bbox = torch.tensor(line_bboxes[parent_idx], dtype=torch.float32)
                child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                geom_feat = compute_geometry_features(parent_bbox, child_bbox)

                # 获取关系标签
                label = RELATION_LABELS[relation]

                self.samples.append({
                    "parent_feat": line_features[parent_idx],
                    "child_feat": line_features[child_idx],
                    "geom_feat": geom_feat,
                    "label": label,
                    "relation": relation
                })

                positive_pairs.append((parent_idx, child_idx))

            # 2. 负采样（标记为none=0）
            if self.neg_ratio > 0:
                num_neg_samples = int(len(positive_pairs) * self.neg_ratio)

                for _ in range(num_neg_samples):
                    # 随机选择两个不同的行
                    child_idx = random.randint(0, num_lines - 1)
                    parent_idx = random.randint(0, child_idx) if child_idx > 0 else 0

                    # 跳过已有的正样本对
                    if (parent_idx, child_idx) in positive_pairs:
                        continue

                    # 检查有效性
                    max_idx = line_mask.shape[0]
                    if parent_idx >= max_idx or child_idx >= max_idx:
                        continue
                    if not line_mask[parent_idx] or not line_mask[child_idx]:
                        continue
                    if parent_idx >= len(line_bboxes) or child_idx >= len(line_bboxes):
                        continue

                    # 计算几何特征
                    parent_bbox = torch.tensor(line_bboxes[parent_idx], dtype=torch.float32)
                    child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                    geom_feat = compute_geometry_features(parent_bbox, child_bbox)

                    self.samples.append({
                        "parent_feat": line_features[parent_idx],
                        "child_feat": line_features[child_idx],
                        "geom_feat": geom_feat,
                        "label": 0,  # none
                        "relation": "none"
                    })

        logger.info(f"总样本数: {len(self.samples)}")

        # 统计各类别样本数
        label_counts = {}
        for s in self.samples:
            label = s["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        for label_id, count in sorted(label_counts.items()):
            label_name = RELATION_NAMES[label_id]
            percentage = count / len(self.samples) * 100
            logger.info(f"  {label_name} (id={label_id}): {count} ({percentage:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "parent_feat": sample["parent_feat"],
            "child_feat": sample["child_feat"],
            "geom_feat": sample["geom_feat"],
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="训练"):
        parent_feat = batch["parent_feat"].to(device)
        child_feat = batch["child_feat"].to(device)
        geom_feat = batch["geom_feat"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # 前向传播
        logits = model(parent_feat, child_feat, geom_feat)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 收集预测结果
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # 计算指标（macro平均）
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
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
            geom_feat = batch["geom_feat"].to(device)
            labels = batch["label"].to(device)

            logits = model(parent_feat, child_feat, geom_feat)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 分类报告
    report = classification_report(
        all_labels, all_preds,
        target_names=RELATION_NAMES,
        zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1, cm, report


def main():
    # 配置
    # 获取项目根目录（脚本在 examples/ 下，根目录是上一级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 从 E 盘读取特征文件，输出也到 E 盘
    features_file = os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features") + "/train_line_features.pkl"
    output_dir = os.getenv("LAYOUTLMFT_OUTPUT_DIR", "/mnt/e/models/train_data/layoutlmft") + "/multiclass_relation"
    max_steps = 300  # 训练步数
    batch_size = 32
    learning_rate = 5e-4
    neg_ratio = 1.5  # 负样本比例

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
    dataset = MultiClassRelationDataset(
        features_file=features_file,
        neg_ratio=neg_ratio
    )

    # 划分训练集和验证集（80/20）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
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
    model = MultiClassRelationClassifier(
        hidden_size=768,
        num_relations=4,  # none, connect, contain, equality
        use_geometry=True,  # 使用几何特征
        dropout=0.1
    ).to(device)

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 计算类别权重（处理类别不平衡）
    label_counts = {}
    for sample in dataset.samples:
        label = sample["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    total_samples = len(dataset.samples)
    class_weights = []
    for i in range(4):
        if i in label_counts:
            weight = total_samples / (4 * label_counts[i])
        else:
            weight = 1.0
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"类别权重: {class_weights.cpu().numpy()}")

    # 使用 FocalLoss（论文对齐）
    # 论文公式：L_rel = Σ FocalLoss(R_i, P_rel_(i,j)) / L
    # gamma=2.0 是论文标准参数
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # 训练循环
    logger.info(f"\n开始训练（最多 {max_steps} 步）...")
    best_f1 = 0
    step = 0

    num_epochs = (max_steps // len(train_loader)) + 1

    for epoch in range(num_epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

        # 训练
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        logger.info(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                   f"P: {train_prec:.4f}, R: {train_rec:.4f}, F1: {train_f1:.4f}")

        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, cm, report = evaluate(
            model, val_loader, criterion, device
        )

        logger.info(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                   f"P: {val_prec:.4f}, R: {val_rec:.4f}, F1: {val_f1:.4f}")
        logger.info(f"混淆矩阵:\n{cm}")
        logger.info(f"分类报告:\n{report}")

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

            # 同时保存一个纯权重文件
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

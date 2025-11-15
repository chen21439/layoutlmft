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
        # [B, L, L] = [B, L, 1, gru_hidden] @ [B, 1, gru_hidden, L]
        attention_scores = torch.matmul(
            query.unsqueeze(2),  # [B, L, 1, gru_hidden]
            key.transpose(1, 2).unsqueeze(1)  # [B, 1, gru_hidden, L]
        ).squeeze(2) / (self.gru_hidden_size ** 0.5)  # [B, L, L]

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


class ParentFinderDataset(torch.utils.data.Dataset):
    """
    父节点查找数据集
    每个样本是一个文档页面，包含多个语义单元（lines）

    注意：不需要 line_labels，直接使用 line_features
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


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_parent_correct = 0
    total_parent_count = 0

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

        loss = 0
        correct = 0
        count = 0

        for b in range(batch_size):
            for i in range(max_lines):
                if not valid_mask[b, i]:
                    continue

                # 位置i的父节点预测
                logits_i = parent_logits[b, i, :i+1]  # [0:i+1] 包含ROOT和之前的单元
                target_i = line_parent_ids[b, i]

                # 将parent_id映射到logits索引
                # parent_id=-1 -> index=0 (ROOT)
                # parent_id=0,1,2,... -> index=1,2,3,...
                target_idx = target_i + 1 if target_i >= 0 else 0

                if target_idx > i:
                    # 父节点在当前位置之后，跳过（数据错误）
                    continue

                # 交叉熵损失
                loss += F.cross_entropy(
                    logits_i.unsqueeze(0),
                    torch.tensor([target_idx], device=device)
                )

                # 统计准确率
                pred_idx = torch.argmax(logits_i).item()
                if pred_idx == target_idx:
                    correct += 1
                count += 1

        if count > 0:
            loss = loss / count
        else:
            continue

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_parent_correct += correct
        total_parent_count += count

    avg_loss = total_loss / len(dataloader)
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

                    logits_i = parent_logits[b, i, :i+1]
                    target_i = line_parent_ids[b, i]
                    target_idx = target_i + 1 if target_i >= 0 else 0

                    if target_idx > i:
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
    else:
        pattern = os.path.join(features_dir, f"{split}_line_features_chunk_*.pkl")
        chunk_files = sorted(glob.glob(pattern))

        page_features = []
        for chunk_file in chunk_files:
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)
            page_features.extend(chunk_data)

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

    # 构建矩阵
    cp_matrix.build()

    return cp_matrix


def main():
    # 配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    features_dir = os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features")
    output_dir = os.getenv("LAYOUTLMFT_OUTPUT_DIR", "/mnt/e/models/train_data/layoutlmft") + "/parent_finder"

    num_epochs = 10
    batch_size = 4  # 每个样本是一个页面，较大
    learning_rate = 1e-4
    num_classes = 16
    use_soft_mask = False  # 简化版本：不使用 soft-mask（需要语义标签）
    max_chunks = int(os.getenv("MAX_CHUNKS", "1"))  # 默认只加载1个chunk用于测试

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

    # 创建数据集
    train_dataset = ParentFinderDataset(features_dir, split="train", max_chunks=max_chunks)
    val_dataset = ParentFinderDataset(features_dir, split="validation", max_chunks=max_chunks)

    logger.info(f"训练集: {len(train_dataset)} 页")
    logger.info(f"验证集: {len(val_dataset)} 页")

    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 创建模型
    model = ParentFinderGRU(
        hidden_size=768,
        gru_hidden_size=512,
        num_classes=num_classes,
        dropout=0.1,
        use_soft_mask=use_soft_mask
    ).to(device)

    # 设置 M_cp（已禁用）
    # if use_soft_mask:
    #     model.set_child_parent_matrix(cp_matrix.get_tensor(device))

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


if __name__ == "__main__":
    main()

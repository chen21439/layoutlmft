"""特征池化模块

提供 Token -> Line 的特征聚合功能。
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List, Optional


class LineFeatureAggregator(nn.Module):
    """行级特征聚合器

    将 Token 级特征按 line_id 聚合为行级特征。
    支持多种聚合方式：mean, max, first, attention。
    """

    def __init__(
        self,
        hidden_size: int = 768,
        aggregation: str = "mean",
    ):
        """
        Args:
            hidden_size: 隐藏层维度
            aggregation: 聚合方式 ("mean", "max", "first", "attention")
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.aggregation = aggregation

        if aggregation == "attention":
            # 注意力聚合 (论文公式10: α = FC1(tanh(FC2(F))))
            # FC2: hidden_size -> 1024, FC1: 1024 -> 1
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 1024),  # FC2: 1024 nodes (per paper)
                nn.Tanh(),
                nn.Linear(1024, 1),  # FC1: 1 node (per paper)
            )

    def forward(
        self,
        token_hidden: Tensor,
        line_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """聚合 token 特征为行特征

        Args:
            token_hidden: [batch, seq_len, hidden_size] token 特征
            line_ids: [batch, seq_len] 每个 token 的 line_id (-1 表示无效)
            attention_mask: [batch, seq_len] 注意力掩码 (可选)

        Returns:
            line_features: [batch, max_lines, hidden_size] 行特征
            line_mask: [batch, max_lines] 有效行掩码
        """
        batch_size, seq_len, hidden_size = token_hidden.shape
        device = token_hidden.device

        # 找出每个 batch 的最大 line_id
        max_line_id = line_ids.max().item()
        if max_line_id < 0:
            # 没有有效行
            return (
                torch.zeros(batch_size, 1, hidden_size, device=device),
                torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            )

        num_lines = int(max_line_id) + 1

        # 初始化输出
        line_features = torch.zeros(batch_size, num_lines, hidden_size, device=device)
        line_counts = torch.zeros(batch_size, num_lines, device=device)

        # 聚合
        if self.aggregation == "mean":
            # Mean pooling
            for b in range(batch_size):
                for t in range(seq_len):
                    lid = line_ids[b, t].item()
                    if lid >= 0:
                        line_features[b, lid] += token_hidden[b, t]
                        line_counts[b, lid] += 1

            # 计算平均
            valid_counts = line_counts.clamp(min=1)
            line_features = line_features / valid_counts.unsqueeze(-1)

        elif self.aggregation == "max":
            # Max pooling
            line_features.fill_(float('-inf'))
            for b in range(batch_size):
                for t in range(seq_len):
                    lid = line_ids[b, t].item()
                    if lid >= 0:
                        line_features[b, lid] = torch.max(
                            line_features[b, lid],
                            token_hidden[b, t]
                        )
                        line_counts[b, lid] = 1
            # 替换未更新的行
            line_features = torch.where(
                line_features == float('-inf'),
                torch.zeros_like(line_features),
                line_features
            )

        elif self.aggregation == "first":
            # 取每行第一个 token
            seen = torch.zeros(batch_size, num_lines, dtype=torch.bool, device=device)
            for b in range(batch_size):
                for t in range(seq_len):
                    lid = line_ids[b, t].item()
                    if lid >= 0 and not seen[b, lid]:
                        line_features[b, lid] = token_hidden[b, t]
                        seen[b, lid] = True
                        line_counts[b, lid] = 1

        elif self.aggregation == "attention":
            # 注意力加权聚合
            attention_scores = self.attention(token_hidden).squeeze(-1)  # [B, seq_len]

            for b in range(batch_size):
                for lid in range(num_lines):
                    # 找出属于该行的 token
                    mask = (line_ids[b] == lid)
                    if mask.sum() == 0:
                        continue

                    tokens = token_hidden[b, mask]  # [num_tokens, hidden]
                    scores = attention_scores[b, mask]  # [num_tokens]
                    weights = torch.softmax(scores, dim=0)  # [num_tokens]

                    line_features[b, lid] = (tokens * weights.unsqueeze(-1)).sum(dim=0)
                    line_counts[b, lid] = 1

        # 创建有效行掩码
        line_mask = line_counts > 0

        return line_features, line_mask


def aggregate_document_line_features(
    doc_hidden: Tensor,
    doc_line_ids: Tensor,
) -> Tuple[Tensor, Tensor]:
    """从文档的所有 chunks 中聚合 line features

    这是一个独立函数，用于处理单个文档的多个 chunks。

    Args:
        doc_hidden: [num_chunks, seq_len, hidden_dim]
        doc_line_ids: [num_chunks, seq_len]，每个 token 的全局 line_id

    Returns:
        features: [num_lines, hidden_dim]
        mask: [num_lines]，有效行的 mask
    """
    device = doc_hidden.device
    hidden_dim = doc_hidden.shape[-1]

    # 收集所有有效的 line_id
    valid_line_ids = doc_line_ids[doc_line_ids >= 0].unique()
    if len(valid_line_ids) == 0:
        return (
            torch.zeros(1, hidden_dim, device=device),
            torch.zeros(1, dtype=torch.bool, device=device),
        )

    # 按 line_id 排序
    valid_line_ids = valid_line_ids.sort()[0]
    num_lines = len(valid_line_ids)

    # 创建 line_id 到索引的映射
    line_id_to_idx = {lid.item(): idx for idx, lid in enumerate(valid_line_ids)}

    # 聚合每个 line 的 features（mean pooling）
    line_features = torch.zeros(num_lines, hidden_dim, device=device)
    line_counts = torch.zeros(num_lines, device=device)

    num_chunks, seq_len, _ = doc_hidden.shape
    for chunk_idx in range(num_chunks):
        for token_idx in range(seq_len):
            lid = doc_line_ids[chunk_idx, token_idx].item()
            if lid >= 0 and lid in line_id_to_idx:
                line_idx = line_id_to_idx[lid]
                line_features[line_idx] += doc_hidden[chunk_idx, token_idx]
                line_counts[line_idx] += 1

    # 计算平均值
    valid_counts = line_counts.clamp(min=1)
    line_features = line_features / valid_counts.unsqueeze(1)

    # 创建 mask（所有收集到的行都是有效的）
    line_mask = line_counts > 0

    return line_features, line_mask

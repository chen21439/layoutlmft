#!/usr/bin/env python
# coding=utf-8
"""
Attention Pooling 模块

功能：将变长的 token 序列聚合成一个固定维度的向量
用途：在 TOC 构建时，将 section 行对应的多个 token 聚合成 section 向量

=== 流程图 ===

输入（Section Token-level）:
    section_tokens: [num_sections, max_tokens, hidden_size]
    section_token_mask: [num_sections, max_tokens]

聚合过程:
    Section 0: tokens = [t0, t1, t2, PAD, PAD]
               attention_weights = softmax([w0, w1, w2, -inf, -inf])
               section_feat_0 = sum(w_i * t_i)

输出（Section-level）:
    section_features: [num_sections, hidden_size]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionPooling(nn.Module):
    """
    Attention-based Pooling 模块

    将变长 token 序列通过可学习的 attention 权重聚合成一个向量

    Example:
        >>> pooling = AttentionPooling(hidden_size=768)
        >>> tokens = torch.randn(10, 20, 768)  # [num_sections, max_tokens, hidden]
        >>> mask = torch.ones(10, 20, dtype=torch.bool)
        >>> output = pooling(tokens, mask)  # [num_sections, hidden]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: 隐藏层维度
            dropout: Dropout rate
        """
        super().__init__()

        # 可学习的 attention 打分网络
        # 使用两层 MLP 增加表达能力
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        return_weights: bool = False,
    ):
        """
        对每个 section 的 tokens 进行 attention pooling

        Args:
            tokens: [num_sections, max_tokens, hidden_size]
                    每个 section 的 token 特征
            mask: [num_sections, max_tokens]
                  True 表示有效 token，False 表示 padding
            return_weights: 是否返回 attention 权重（用于诊断）

        Returns:
            pooled: [num_sections, hidden_size]
                    每个 section 聚合后的特征向量
            weights: (仅当 return_weights=True) [num_sections, max_tokens]
                     attention 权重分布
        """
        # 计算每个 token 的重要性分数
        scores = self.attention(tokens).squeeze(-1)  # [num_sections, max_tokens]

        # 对 padding 位置设置极小值（排除在 softmax 之外）
        scores = scores.masked_fill(~mask, float('-inf'))

        # 处理全 padding 的情况（避免 NaN）
        all_masked = ~mask.any(dim=-1, keepdim=True)  # [num_sections, 1]
        scores = scores.masked_fill(all_masked.expand_as(scores), 0.0)

        # Softmax 得到 attention 权重
        weights = F.softmax(scores, dim=-1)  # [num_sections, max_tokens]

        # 保存 dropout 前的权重用于诊断
        weights_before_dropout = weights.clone().detach() if return_weights else None

        # 处理全 padding：均匀权重（虽然理论上不应该发生）
        weights = weights.masked_fill(all_masked.expand_as(weights), 1.0 / max(1, tokens.shape[1]))

        # Dropout
        weights = self.dropout(weights)

        # 加权求和: [num_sections, max_tokens, 1] * [num_sections, max_tokens, hidden]
        pooled = torch.bmm(
            weights.unsqueeze(1),  # [num_sections, 1, max_tokens]
            tokens,                # [num_sections, max_tokens, hidden]
        ).squeeze(1)  # [num_sections, hidden]

        if return_weights:
            return pooled, weights_before_dropout
        return pooled


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling 模块

    使用多头注意力机制进行池化，每个头可以关注不同的语义特征
    （比如：标题关键词、编号、修饰语等）

    Example:
        >>> pooling = MultiHeadAttentionPooling(hidden_size=768, num_heads=8)
        >>> tokens = torch.randn(10, 20, 768)
        >>> mask = torch.ones(10, 20, dtype=torch.bool)
        >>> output = pooling(tokens, mask)  # [num_sections, hidden]
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout rate
        """
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 每个头一个可学习的 query vector
        self.queries = nn.Parameter(torch.randn(num_heads, self.head_dim))

        # Key 投影
        self.key_proj = nn.Linear(hidden_size, hidden_size)

        # Value 投影
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.queries)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [num_sections, max_tokens, hidden_size]
            mask: [num_sections, max_tokens]

        Returns:
            pooled: [num_sections, hidden_size]
        """
        num_sections, max_tokens, _ = tokens.shape

        # Key, Value 投影
        k = self.key_proj(tokens)    # [S, T, H]
        v = self.value_proj(tokens)  # [S, T, H]

        # 拆分成多头: [S, T, num_heads, head_dim] -> [S, num_heads, T, head_dim]
        k = k.view(num_sections, max_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(num_sections, max_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Query: [num_heads, head_dim] -> [S, num_heads, 1, head_dim]
        q = self.queries.unsqueeze(0).unsqueeze(2).expand(num_sections, -1, 1, -1)

        # Attention: [S, num_heads, 1, T]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用 mask: [S, 1, 1, T]
        attn_mask = ~mask.unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        # 处理全 padding
        all_masked = ~mask.any(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights.masked_fill(all_masked.expand_as(attn_weights), 0.0)

        # Softmax
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 加权求和: [S, num_heads, 1, head_dim]
        context = torch.matmul(attn_probs, v)

        # 合并多头: [S, num_heads, head_dim] -> [S, hidden_size]
        context = context.squeeze(2).view(num_sections, self.hidden_size)

        # 输出投影
        output = self.out_proj(context)

        return output


def extract_section_tokens(
    hidden_states: torch.Tensor,
    line_ids: torch.Tensor,
    section_line_indices: torch.Tensor,
    max_tokens_per_section: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 token-level hidden states 中提取 section 行对应的 tokens

    Args:
        hidden_states: [num_chunks, seq_len, hidden_size]
                       LayoutXLM 输出的 token 特征
        line_ids: [num_chunks, seq_len]
                  每个 token 所属的 line_id（-1 表示特殊 token）
        section_line_indices: [num_sections]
                              被判定为 section 的 line_id 列表
        max_tokens_per_section: 每个 section 保留的最大 token 数

    Returns:
        section_tokens: [num_sections, max_tokens_per_section, hidden_size]
        section_token_mask: [num_sections, max_tokens_per_section]
    """
    device = hidden_states.device
    hidden_size = hidden_states.shape[-1]
    num_sections = len(section_line_indices)

    # 展平
    flat_hidden = hidden_states.reshape(-1, hidden_size)  # [N, H]
    flat_line_ids = line_ids.reshape(-1)  # [N]

    section_tokens_list = []

    for section_lid in section_line_indices:
        # 找到属于这个 line 的所有 token
        token_mask = (flat_line_ids == section_lid.item())
        tokens = flat_hidden[token_mask]  # [?, H]

        # 截断或补齐
        num_tokens = tokens.shape[0]
        if num_tokens > max_tokens_per_section:
            tokens = tokens[:max_tokens_per_section]
            num_tokens = max_tokens_per_section

        section_tokens_list.append((tokens, num_tokens))

    # 确定实际的 max_tokens（避免过度 padding）
    actual_max_tokens = max(t[1] for t in section_tokens_list) if section_tokens_list else 1
    actual_max_tokens = min(actual_max_tokens, max_tokens_per_section)

    # Padding
    section_tokens = torch.zeros(num_sections, actual_max_tokens, hidden_size, device=device)
    section_token_mask = torch.zeros(num_sections, actual_max_tokens, dtype=torch.bool, device=device)

    for i, (tokens, num_tokens) in enumerate(section_tokens_list):
        if num_tokens > 0:
            n = min(num_tokens, actual_max_tokens)
            section_tokens[i, :n] = tokens[:n]
            section_token_mask[i, :n] = True

    return section_tokens, section_token_mask

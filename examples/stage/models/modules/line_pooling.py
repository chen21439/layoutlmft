#!/usr/bin/env python
# coding=utf-8
"""
Line-level 特征聚合模块

功能：将 Token-level hidden states 聚合为 Line-level features
原理：对属于同一行的所有 token 的 hidden state 做 mean pooling

=== 流程图 ===

输入（Token-level）:
    tokens:     [CLS] [t1] [t2] [t3] [t4] [SEP] [t5] [t6] ...
    line_ids:   [-1]  [0]  [0]  [0]  [1]  [-1]  [1]  [1]  ...
    hidden:     [h0]  [h1] [h2] [h3] [h4] [h5]  [h6] [h7] ...

聚合过程:
    Line 0: 找到 line_id=0 的 tokens → [t1, t2, t3]
            mean([h1, h2, h3]) → line_feat_0

    Line 1: 找到 line_id=1 的 tokens → [t4, t5, t6]
            mean([h4, h6, h7]) → line_feat_1

输出（Line-level）:
    line_features: [line_feat_0, line_feat_1, ...]
    line_mask: [True, True, ...]

=== 使用场景 ===

1. 训练时（JointModel.forward）:
   - 页面级别：单个 chunk，line_id 是本地索引
   - 文档级别：多个 chunks，line_id 是全局索引

2. 推理时（Predictor.predict）:
   - 同上，但不计算损失

=== 设计原则 ===

- 统一接口：训练和推理使用相同的聚合逻辑
- 向量化：使用 scatter_add 实现高效聚合，避免 Python 循环
- 紧凑输出：只返回有效的 line features，无 padding
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LinePooling(nn.Module):
    """
    Line-level 特征聚合器

    将多个 token 的 hidden states 按 line_id 分组，做 mean pooling

    Example:
        >>> pooling = LinePooling()
        >>> hidden = torch.randn(2, 512, 768)  # [batch, seq_len, hidden_dim]
        >>> line_ids = torch.randint(-1, 10, (2, 512))  # [batch, seq_len]
        >>> features, mask = pooling(hidden, line_ids)
        >>> # features: [num_lines, 768], mask: [num_lines]
    """

    def __init__(self, pooling_method: str = "mean"):
        """
        Args:
            pooling_method: 聚合方式，目前只支持 "mean"
        """
        super().__init__()
        self.pooling_method = pooling_method
        if pooling_method != "mean":
            raise ValueError(f"Unsupported pooling method: {pooling_method}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        聚合 token hidden states 到 line-level

        Args:
            hidden_states: Token-level 特征
                - 形状: [batch, seq_len, hidden_dim] 或 [num_chunks, seq_len, hidden_dim]
            line_ids: 每个 token 所属的 line_id
                - 形状: [batch, seq_len] 或 [num_chunks, seq_len]
                - 值 >= 0 表示有效 token，-1 表示忽略（CLS, SEP, padding 等）

        Returns:
            line_features: [num_lines, hidden_dim]
                - 紧凑数组，按 line_id 排序
            line_mask: [num_lines]
                - 每个位置是否有效（token count > 0）

        流程:
            1. 展平所有 chunks/batch
            2. 过滤掉 line_id < 0 的 token
            3. 按 line_id 分组，使用 scatter_add 聚合
            4. 计算每行的平均值
        """
        device = hidden_states.device
        hidden_dim = hidden_states.shape[-1]

        # ========== Step 1: 展平 ==========
        # [batch, seq_len, H] → [N, H]
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        # [batch, seq_len] → [N]
        flat_line_ids = line_ids.reshape(-1)

        # ========== Step 2: 过滤无效 token ==========
        # line_id >= 0 的是有效 token（属于某一行）
        # line_id < 0 的是特殊 token（CLS, SEP, padding）
        valid_mask = flat_line_ids >= 0
        valid_line_ids = flat_line_ids[valid_mask]  # [M]
        valid_hidden = flat_hidden[valid_mask]  # [M, H]

        # 无有效 token 的边界情况
        if len(valid_line_ids) == 0:
            return (
                torch.zeros(1, hidden_dim, device=device),
                torch.zeros(1, dtype=torch.bool, device=device),
            )

        # ========== Step 3: 获取唯一 line_id ==========
        # 排序确保输出顺序一致
        unique_line_ids, _ = valid_line_ids.unique(sorted=True, return_inverse=False)
        num_lines = len(unique_line_ids)

        # ========== Step 4: 建立 line_id → 紧凑索引的映射 ==========
        # 使用 searchsorted 进行快速映射
        # 例如: unique_line_ids = [0, 2, 5]
        #       valid_line_ids = [0, 0, 2, 5, 5]
        #       line_indices = [0, 0, 1, 2, 2]
        line_indices = torch.searchsorted(unique_line_ids, valid_line_ids)

        # ========== Step 5: 使用 scatter_add 聚合 ==========
        # 初始化累加器
        line_features = torch.zeros(num_lines, hidden_dim, device=device)
        line_counts = torch.zeros(num_lines, device=device)

        # 累加 hidden states
        # scatter_add_(dim, index, src): 按 index 把 src 累加到对应位置
        line_features.scatter_add_(
            0,  # 在第 0 维（行维度）操作
            line_indices.unsqueeze(1).expand(-1, hidden_dim),  # [M, H]
            valid_hidden,  # [M, H]
        )

        # 统计每行的 token 数量
        line_counts.scatter_add_(
            0,
            line_indices,
            torch.ones_like(line_indices, dtype=torch.float),
        )

        # ========== Step 6: 计算平均值 ==========
        # 防止除零
        valid_counts = line_counts.clamp(min=1)
        line_features = line_features / valid_counts.unsqueeze(1)

        # ========== Step 7: 生成 mask ==========
        line_mask = line_counts > 0

        return line_features, line_mask

    def get_line_ids_mapping(self, line_ids: torch.Tensor) -> torch.Tensor:
        """
        获取紧凑索引到原始 line_id 的映射

        Args:
            line_ids: [batch, seq_len] 或 [num_chunks, seq_len]

        Returns:
            unique_line_ids: [num_lines]，排序后的唯一 line_id 列表

        用途:
            将预测结果（紧凑索引）转换回原始 line_id
        """
        flat_line_ids = line_ids.reshape(-1)
        valid_line_ids = flat_line_ids[flat_line_ids >= 0]
        unique_line_ids, _ = valid_line_ids.unique(sorted=True, return_inverse=False)
        return unique_line_ids


# 为了向后兼容，提供函数式接口
def aggregate_line_features(
    hidden_states: torch.Tensor,
    line_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    函数式接口，用于简单场景

    等价于: LinePooling()(hidden_states, line_ids)
    """
    pooling = LinePooling()
    return pooling(hidden_states, line_ids)

"""任务预测头定义

包含 Order 模块的核心实现：
- OrderHead: 区域间阅读顺序预测
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OrderHead(nn.Module):
    """Order 模块预测头

    基于论文 Detect-Order-Construct 的 Order 模块实现。

    功能:
    - 使用 3 层 Transformer 编码器增强行级特征
    - 预测行间阅读顺序关系（pairwise: 行 i 是否在行 j 之前）

    输入: 行级特征 [batch, num_lines, hidden_size]
    输出: 阅读顺序矩阵 [batch, num_lines, num_lines]
           order_matrix[i,j] = 1 表示行 i 在行 j 之前阅读
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_biaffine: bool = True,
    ):
        """
        Args:
            hidden_size: 输入特征维度
            num_heads: Transformer 注意力头数
            num_layers: Transformer 层数
            dropout: Dropout 比例
            use_biaffine: 是否使用双仿射变换计算关系分数
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.use_biaffine = use_biaffine

        # 3 层 Transformer 编码器（论文架构）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 阅读顺序关系预测
        if use_biaffine:
            # 双仿射变换：更精确的 pairwise 关系建模
            self.head_proj = nn.Linear(hidden_size, hidden_size)
            self.tail_proj = nn.Linear(hidden_size, hidden_size)
            self.biaffine_weight = nn.Parameter(torch.zeros(hidden_size, hidden_size))
            self.biaffine_bias = nn.Parameter(torch.zeros(1))
            nn.init.xavier_uniform_(self.biaffine_weight)
        else:
            # 简单的 MLP 方式
            self.order_head = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

    def forward(
        self,
        line_features: Tensor,
        line_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """前向传播

        Args:
            line_features: [batch, num_lines, hidden_size] 行级特征
            line_mask: [batch, num_lines] 有效行掩码 (True=有效)

        Returns:
            Dict containing:
                - enhanced_features: [batch, num_lines, hidden_size] 增强后的特征
                - order_logits: [batch, num_lines, num_lines] 阅读顺序 logits
                                order_logits[b,i,j] > 0 表示行 i 在行 j 之前
        """
        batch_size, num_lines, _ = line_features.shape
        device = line_features.device

        # Transformer 增强
        if line_mask is not None:
            # TransformerEncoder 需要 src_key_padding_mask，True 表示要 mask 的位置
            src_key_padding_mask = ~line_mask
        else:
            src_key_padding_mask = None

        enhanced = self.transformer(
            line_features,
            src_key_padding_mask=src_key_padding_mask,
        )

        # 计算 pairwise 阅读顺序分数
        if self.use_biaffine:
            # 双仿射变换
            # head: 可能是前面的行
            # tail: 可能是后面的行
            head = self.head_proj(enhanced)  # [B, L, H]
            tail = self.tail_proj(enhanced)  # [B, L, H]

            # order_logits[b,i,j] = head[b,i] @ W @ tail[b,j] + bias
            # 使用 einsum: 'bih,hk,bjk->bij'
            order_logits = torch.einsum(
                'bih,hk,bjk->bij',
                head,
                self.biaffine_weight,
                tail,
            ) + self.biaffine_bias
        else:
            # MLP 方式
            # 构造所有 (i, j) 对的特征
            # head_features: [B, L, 1, H] -> [B, L, L, H]
            head_features = enhanced.unsqueeze(2).expand(-1, -1, num_lines, -1)
            # tail_features: [B, 1, L, H] -> [B, L, L, H]
            tail_features = enhanced.unsqueeze(1).expand(-1, num_lines, -1, -1)

            # 拼接: [B, L, L, 2H]
            pair_features = torch.cat([head_features, tail_features], dim=-1)

            # 预测: [B, L, L, 1] -> [B, L, L]
            order_logits = self.order_head(pair_features).squeeze(-1)

        # 应用掩码：无效位置的分数设为 -inf
        if line_mask is not None:
            # 行掩码：无效行不能参与
            row_mask = ~line_mask.unsqueeze(2)  # [B, L, 1]
            col_mask = ~line_mask.unsqueeze(1)  # [B, 1, L]
            combined_mask = row_mask | col_mask  # [B, L, L]
            order_logits = order_logits.masked_fill(combined_mask, float('-inf'))

        return {
            "enhanced_features": enhanced,
            "order_logits": order_logits,
        }


class OrderLoss(nn.Module):
    """Order 任务损失函数

    基于阅读顺序的标签，计算 pairwise 损失。
    """

    def __init__(self, margin: float = 1.0, use_bce: bool = True):
        """
        Args:
            margin: Margin ranking loss 的 margin 值
            use_bce: 是否使用 BCE loss (否则用 margin ranking loss)
        """
        super().__init__()
        self.margin = margin
        self.use_bce = use_bce

        if use_bce:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(
        self,
        order_logits: Tensor,
        reading_order: Tensor,
        line_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """计算损失

        Args:
            order_logits: [batch, num_lines, num_lines] 预测的顺序分数
            reading_order: [batch, num_lines] 每行的阅读顺序索引 (0, 1, 2, ...)
                          reading_order[i] < reading_order[j] 表示行 i 在行 j 之前
            line_mask: [batch, num_lines] 有效行掩码

        Returns:
            loss: 标量损失值
        """
        batch_size, num_lines = reading_order.shape
        device = order_logits.device

        # 构建 ground truth 顺序矩阵
        # target[i,j] = 1 if i 在 j 之前，否则 0
        order_i = reading_order.unsqueeze(2)  # [B, L, 1]
        order_j = reading_order.unsqueeze(1)  # [B, 1, L]

        # i 在 j 之前: order[i] < order[j]
        target = (order_i < order_j).float()  # [B, L, L]

        if self.use_bce:
            # Binary Cross Entropy
            loss_matrix = self.criterion(order_logits, target)
        else:
            # Margin Ranking Loss
            # 对于 i < j 的对，希望 score[i,j] > score[j,i] + margin
            # 这里简化为：score[i,j] 应该高
            loss_matrix = self.criterion(
                order_logits,
                -order_logits.transpose(1, 2),
                target * 2 - 1  # 转换为 {-1, 1}
            )

        # 应用掩码
        if line_mask is not None:
            # 只计算有效位置的损失
            valid_mask = line_mask.unsqueeze(2) & line_mask.unsqueeze(1)  # [B, L, L]
            # 排除对角线（自己和自己的关系）
            diag_mask = ~torch.eye(num_lines, dtype=torch.bool, device=device).unsqueeze(0)
            valid_mask = valid_mask & diag_mask

            loss_matrix = loss_matrix * valid_mask.float()
            loss = loss_matrix.sum() / valid_mask.sum().clamp(min=1)
        else:
            # 排除对角线
            diag_mask = ~torch.eye(num_lines, dtype=torch.bool, device=device).unsqueeze(0)
            loss_matrix = loss_matrix * diag_mask.float()
            loss = loss_matrix.sum() / (batch_size * num_lines * (num_lines - 1))

        return loss


def compute_reading_order_from_line_ids(line_ids: Tensor) -> Tensor:
    """从 line_id 计算阅读顺序

    假设 line_id 已经按阅读顺序编号（这是 HRDoc 数据的格式）

    Args:
        line_ids: [batch, max_lines] 行 ID

    Returns:
        reading_order: [batch, max_lines] 阅读顺序索引
    """
    # line_id 本身就是阅读顺序
    # 但需要处理无效行（line_id < 0）
    reading_order = line_ids.clone()
    # 无效行设为一个很大的值，表示在最后
    reading_order = torch.where(
        reading_order >= 0,
        reading_order,
        torch.full_like(reading_order, 999999)
    )
    return reading_order

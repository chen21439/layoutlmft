"""行间特征增强模块

参考论文 4.2.2 Multi-modal Feature Enhancement Module

使用轻量级 Transformer Encoder 增强行级特征的上下文信息。
由于 LayoutLM 已经在 token 级别编码了 bbox 位置信息，
LinePooling 后的行特征已隐式包含空间位置，因此不额外加位置编码。

配置（论文参数）：
- 1 层 Transformer Encoder
- 12 heads
- hidden_size = 768
- FFN dim = 2048
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class LineFeatureEnhancer(nn.Module):
    """行间特征增强器

    通过 Transformer self-attention 让行与行之间交互上下文信息。

    支持两种输入格式：
    - 带 batch: [batch_size, num_lines, hidden_size]
    - 不带 batch: [num_lines, hidden_size]

    Example:
        >>> enhancer = LineFeatureEnhancer(hidden_size=768, enabled=True)
        >>> # 页面级（带 batch）
        >>> features = torch.randn(4, 50, 768)  # 4 个页面，每页 50 行
        >>> enhanced = enhancer(features)
        >>> # 文档级（不带 batch）
        >>> features = torch.randn(200, 768)  # 200 行
        >>> enhanced = enhancer(features)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 1,
        enabled: bool = True,
    ):
        """
        Args:
            hidden_size: 隐藏层维度（默认 768，与 LayoutLM 一致）
            num_heads: 注意力头数（论文用 12）
            ffn_dim: FFN 中间层维度（论文用 2048）
            dropout: Dropout 率
            num_layers: Transformer 层数（论文用 1）
            enabled: 是否启用增强，False 时直接返回输入
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.enabled = enabled

        if enabled:
            # PyTorch 原生 Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation='relu',
                batch_first=True,  # 使用 [batch, seq, hidden] 格式
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )

            # Layer Norm（可选，Transformer 内部已有，这里做最终归一化）
            self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        line_features: Tensor,
        line_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """增强行级特征

        Args:
            line_features: 行特征
                - [batch_size, num_lines, hidden_size] 页面级（带 batch）
                - [num_lines, hidden_size] 文档级（不带 batch）
            line_mask: 有效行掩码
                - [batch_size, num_lines] 或 [num_lines]
                - True 表示有效，False 表示 padding

        Returns:
            增强后的行特征，形状与输入一致
        """
        if not self.enabled:
            return line_features

        # 处理输入维度
        input_dim = line_features.dim()
        if input_dim == 2:
            # [L, H] -> [1, L, H]
            x = line_features.unsqueeze(0)
            if line_mask is not None:
                line_mask = line_mask.unsqueeze(0)
        elif input_dim == 3:
            x = line_features
        else:
            raise ValueError(f"Expected 2D or 3D input, got {input_dim}D")

        # 准备 attention mask
        # TransformerEncoder 的 src_key_padding_mask: True 表示忽略该位置
        attn_mask = None
        if line_mask is not None:
            attn_mask = ~line_mask  # 反转：有效位置为 False，padding 位置为 True

        # Transformer 编码
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = self.norm(x)

        # 恢复原始维度
        if input_dim == 2:
            x = x.squeeze(0)

        return x

    def set_enabled(self, enabled: bool):
        """动态启用/禁用增强"""
        self.enabled = enabled

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, enabled={self.enabled}"

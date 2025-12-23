"""LayoutXLM 基座模型封装

复用 layoutlmft 中的 LayoutXLM 实现，提供统一的特征提取接口。
"""

import os
import sys
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from layoutlmft.models.layoutxlm import (
    LayoutXLMForTokenClassification,
    LayoutXLMConfig,
)


class LayoutXLMBackbone(nn.Module):
    """LayoutXLM 多模态编码器封装

    封装 LayoutXLMForTokenClassification，提供统一的特征提取接口。
    支持:
    - Token 级特征提取
    - 可选的分类头（用于 Detect 任务的逻辑角色分类）
    """

    def __init__(
        self,
        model_path: str,
        num_labels: int = 16,
        use_visual: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            model_path: 预训练模型路径
            num_labels: 分类标签数（用于 token classification）
            use_visual: 是否使用视觉特征
            freeze_backbone: 是否冻结 backbone 参数
        """
        super().__init__()
        self.model_path = model_path
        self.use_visual = use_visual
        self.num_labels = num_labels

        # 加载配置
        self.config = LayoutXLMConfig.from_pretrained(model_path)
        self.config.num_labels = num_labels
        self.hidden_size = self.config.hidden_size  # 768

        # 加载 LayoutXLM 模型
        self.encoder = LayoutXLMForTokenClassification.from_pretrained(
            model_path,
            config=self.config,
        )

        if freeze_backbone:
            self._freeze_parameters()

    def _freeze_parameters(self):
        """冻结 backbone 参数（只训练下游任务头）"""
        # 冻结 LayoutXLM 的编码器部分，保留分类头
        for name, param in self.encoder.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        bbox: Tensor,
        attention_mask: Tensor,
        image: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: bool = True,
    ) -> Dict[str, Any]:
        """前向传播

        Args:
            input_ids: [batch, seq_len] token ids
            bbox: [batch, seq_len, 4] 边界框坐标
            attention_mask: [batch, seq_len] 注意力掩码
            image: [batch, 3, H, W] 图像 (可选)
            labels: [batch, seq_len] 分类标签 (可选，训练时使用)
            output_hidden_states: 是否输出隐状态

        Returns:
            Dict containing:
                - loss: 分类损失 (如果提供 labels)
                - logits: [batch, seq_len, num_labels] 分类 logits
                - hidden_states: [batch, seq_len, hidden_size] 最后一层隐状态
                - all_hidden_states: tuple of hidden states (如果 output_hidden_states=True)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

        result = {
            "logits": outputs.logits,
        }

        if outputs.loss is not None:
            result["loss"] = outputs.loss

        if output_hidden_states and outputs.hidden_states is not None:
            result["hidden_states"] = outputs.hidden_states[-1]  # 最后一层
            result["all_hidden_states"] = outputs.hidden_states

        return result

    def get_hidden_size(self) -> int:
        """返回隐藏层维度"""
        return self.hidden_size

    def save_pretrained(self, save_path: str):
        """保存模型"""
        self.encoder.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "LayoutXLMBackbone":
        """从预训练模型加载"""
        return cls(model_path=model_path, **kwargs)

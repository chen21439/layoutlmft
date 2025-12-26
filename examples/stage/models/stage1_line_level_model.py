#!/usr/bin/env python
# coding=utf-8
"""
Stage 1 Line-Level Classification Model

独立的 Stage 1 分类模型，复用共享模块：
- LinePooling: 从 modules/line_pooling.py
- LineClassificationHead: 从 heads/classification_head.py

与 JointModel 的 Stage 1 部分逻辑完全一致，但职责更清晰。

=== 流程 ===

1. LayoutXLM Backbone:
   input_ids [B, seq] → hidden_states [B, seq, 768]

2. LinePooling (共享模块):
   hidden_states + line_ids → line_features [L, 768]

3. LineClassificationHead (共享模块):
   line_features → logits [L, num_classes]

4. Loss:
   CrossEntropy(logits, line_labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass

# 复用共享模块
from .modules import LinePooling
from .heads import LineClassificationHead


@dataclass
class LineLevelClassifierOutput:
    """Line-level 分类模型的输出"""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


class LayoutXLMForLineLevelClassification(nn.Module):
    """
    LayoutXLM Line-Level 分类模型

    复用共享模块，与 JointModel 的 Stage 1 逻辑一致：
    1. 使用 LayoutXLM backbone 获取 token hidden states
    2. 使用 LinePooling（共享）聚合到 line-level
    3. 使用 LineClassificationHead（共享）进行分类

    Args:
        backbone_model: LayoutXLMForTokenClassification 模型
        num_classes: 分类类别数
        hidden_size: Hidden dimension (默认 768)
        cls_dropout: 分类头 dropout (默认 0.1)
    """

    def __init__(
        self,
        backbone_model,
        num_classes: int = 14,
        hidden_size: int = 768,
        cls_dropout: float = 0.1,
    ):
        super().__init__()

        # Backbone: 使用完整的 LayoutXLMForTokenClassification（与 JointModel 一致）
        # 这样可以确保 forward 调用方式完全相同
        self.backbone = backbone_model

        # 共享模块: LinePooling
        self.line_pooling = LinePooling(pooling_method="mean")

        # 共享模块: LineClassificationHead
        self.cls_head = LineClassificationHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=cls_dropout,
        )

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.config = backbone_model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        line_ids: Optional[torch.Tensor] = None,
        line_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, LineLevelClassifierOutput]:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]
            bbox: [batch, seq_len, 4]
            attention_mask: [batch, seq_len]
            image: [batch, 3, H, W] (可选)
            labels: [batch, seq_len] Token-level labels (用于提取 line_labels)
            line_ids: [batch, seq_len] 每个 token 所属的 line_id
            line_labels: [batch, max_lines] Line-level labels

        Returns:
            LineLevelClassifierOutput with loss and logits
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # ========== Step 1: Backbone ==========
        # 与 JointModel 完全一致的调用方式：
        # - 使用完整的 LayoutXLMForTokenClassification
        # - output_hidden_states=True
        # - 从 hidden_states[-1] 获取最后一层特征
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            output_hidden_states=True,
        )
        hidden_states = backbone_outputs.hidden_states[-1]  # [B, seq_len, H]

        # 如果没有 line_ids，返回空输出
        if line_ids is None:
            dummy_logits = torch.zeros(batch_size, 1, self.num_classes, device=device)
            if not return_dict:
                return (None, dummy_logits)
            return LineLevelClassifierOutput(loss=None, logits=dummy_logits)

        # ========== Step 2: LinePooling (共享模块) ==========
        all_logits = []
        all_losses = []
        max_lines = 0

        # 如果没有提供 line_labels，从 token labels 提取
        if line_labels is None and labels is not None:
            line_labels = self._extract_line_labels(labels, line_ids, device)

        for b in range(batch_size):
            sample_hidden = hidden_states[b:b+1]
            sample_line_ids = line_ids[b:b+1]

            line_features, line_mask = self.line_pooling(sample_hidden, sample_line_ids)
            num_lines = line_features.shape[0]
            max_lines = max(max_lines, num_lines)

            # ========== Step 3: Classification (共享模块) ==========
            logits = self.cls_head(line_features)
            all_logits.append(logits)

            # 计算损失
            if line_labels is not None:
                sample_labels = line_labels[b, :num_lines]
                valid_mask = sample_labels != -100
                if valid_mask.any():
                    loss = F.cross_entropy(
                        logits[valid_mask],
                        sample_labels[valid_mask],
                        reduction="mean",
                    )
                    all_losses.append(loss)

        # ========== Step 4: Padding & Loss ==========
        padded_logits = []
        for logits in all_logits:
            num_lines = logits.shape[0]
            if num_lines < max_lines:
                padding = torch.zeros(
                    max_lines - num_lines, self.num_classes,
                    device=device, dtype=logits.dtype
                )
                logits = torch.cat([logits, padding], dim=0)
            padded_logits.append(logits)

        batch_logits = torch.stack(padded_logits, dim=0)

        loss = None
        if all_losses:
            loss = torch.stack(all_losses).mean()

        if not return_dict:
            return (loss, batch_logits)

        return LineLevelClassifierOutput(
            loss=loss,
            logits=batch_logits,
            hidden_states=backbone_outputs.hidden_states if hasattr(backbone_outputs, 'hidden_states') else None,
        )

    def _extract_line_labels(self, labels, line_ids, device):
        """从 token labels 提取 line labels"""
        batch_size = labels.shape[0]

        # 找出最大 line 数量
        max_lines = 0
        for b in range(batch_size):
            valid_ids = line_ids[b][line_ids[b] >= 0]
            if len(valid_ids) > 0:
                max_lines = max(max_lines, valid_ids.max().item() + 1)

        if max_lines == 0:
            max_lines = 1

        line_labels = torch.full((batch_size, max_lines), -100, dtype=torch.long, device=device)

        for b in range(batch_size):
            sample_line_ids = line_ids[b]
            sample_labels = labels[b]

            for line_idx in range(max_lines):
                token_mask = (sample_line_ids == line_idx)
                if token_mask.any():
                    first_idx = token_mask.nonzero(as_tuple=True)[0][0]
                    label = sample_labels[first_idx].item()
                    if label >= 0:
                        line_labels[b, line_idx] = label

        return line_labels

    def save_pretrained(self, save_directory):
        """保存模型"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)

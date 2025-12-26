#!/usr/bin/env python
# coding=utf-8
"""
Stage 1 Line-Level Classification Model

使用 mean pooling 进行 line-level 分类，与联合训练中 Stage 1 的逻辑完全对齐。

Architecture:
    LayoutLM Backbone → LinePooling → LineClassificationHead → Line-level Loss

与 JointModel 的对齐：
- 使用相同的 LinePooling 模块
- 使用相同的 LineClassificationHead
- 使用相同的损失计算方式（line-level cross entropy）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers.modeling_outputs import TokenClassifierOutput

# 导入共享模块
import sys
import os

# 使用绝对路径导入 stage/models 模块
_stage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage"))
if _stage_dir not in sys.path:
    sys.path.insert(0, _stage_dir)

try:
    from models.modules import LinePooling
    from models.heads import LineClassificationHead
except ImportError:
    # 备用导入方式：如果上面失败，尝试从当前目录相对导入
    import importlib.util

    # 加载 line_pooling.py
    lp_path = os.path.join(_stage_dir, "models", "modules", "line_pooling.py")
    spec = importlib.util.spec_from_file_location("line_pooling", lp_path)
    line_pooling_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(line_pooling_module)
    LinePooling = line_pooling_module.LinePooling

    # 加载 classification_head.py
    ch_path = os.path.join(_stage_dir, "models", "heads", "classification_head.py")
    spec = importlib.util.spec_from_file_location("classification_head", ch_path)
    classification_head_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classification_head_module)
    LineClassificationHead = classification_head_module.LineClassificationHead


class LayoutXLMForLineLevelClassification(nn.Module):
    """
    Line-Level 分类模型（Stage 1）

    与 JointModel 的 Stage 1 部分完全一致：
    1. Backbone: LayoutXLM 获取 token-level hidden states
    2. LinePooling: 聚合到 line-level features
    3. LineClassificationHead: 分类每一行
    4. Loss: Line-level cross entropy

    Args:
        backbone_model: LayoutXLMForTokenClassification（只使用其 backbone）
        num_classes: 分类类别数（默认 14）
        hidden_size: Hidden dimension（默认 768）
        cls_dropout: 分类头的 dropout 比例
    """

    def __init__(
        self,
        backbone_model,  # LayoutXLMForTokenClassification
        num_classes: int = 14,
        hidden_size: int = 768,
        cls_dropout: float = 0.1,
    ):
        super().__init__()

        # ========== Backbone ==========
        # 使用 LayoutXLM 的 backbone（不使用其内置的 token-level 分类头）
        self.backbone = backbone_model

        # ========== 共享模块 ==========
        # LinePooling: Token-level → Line-level 特征聚合
        self.line_pooling = LinePooling(pooling_method="mean")

        # ========== Line-level 分类头 ==========
        self.cls_head = LineClassificationHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=cls_dropout,
        )

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # 用于存储准确率（用于 logging）
        self._cls_acc = 0.0

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor = None,
        labels: torch.Tensor = None,  # Token-level labels（用于提取 line_labels）
        line_ids: Optional[torch.Tensor] = None,
        line_labels: Optional[torch.Tensor] = None,  # Line-level labels（优先使用）
        return_dict: bool = True,
        **kwargs,
    ) -> TokenClassifierOutput:
        """
        前向传播

        流程:
        1. Backbone: 获取 token-level hidden states
        2. LinePooling: 聚合到 line-level features
        3. LineClassificationHead: 分类每一行
        4. 计算 line-level loss

        Args:
            input_ids: [batch, seq_len]
            bbox: [batch, seq_len, 4]
            attention_mask: [batch, seq_len]
            image: [batch, 3, H, W] 或 ImageList
            labels: [batch, seq_len] - Token-level labels（用于提取 line_labels）
            line_ids: [batch, seq_len] - 每个 token 的 line_id（-1 表示忽略）
            line_labels: [batch, max_lines] - Line-level labels（如果提供，优先使用）

        Returns:
            TokenClassifierOutput with:
                - loss: Line-level classification loss
                - logits: [batch, max_lines, num_classes]
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # ==================== Step 1: Backbone 获取 hidden states ====================
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            output_hidden_states=True,
        )
        hidden_states = backbone_outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

        # ==================== Step 2: LinePooling 聚合 ====================
        # 截取文本部分的 hidden states（排除视觉 tokens）
        text_seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :text_seq_len, :]

        # 逐样本聚合（因为每个样本的 line 数量不同）
        doc_line_features_list = []
        doc_line_masks_list = []
        for b in range(batch_size):
            sample_hidden = text_hidden[b:b+1]  # [1, seq_len, H]
            sample_line_ids = line_ids[b:b+1]  # [1, seq_len]
            features, mask = self.line_pooling(sample_hidden, sample_line_ids)
            doc_line_features_list.append(features)
            doc_line_masks_list.append(mask)

        # 填充到相同长度
        max_lines = max(f.shape[0] for f in doc_line_features_list)
        line_features = torch.zeros(batch_size, max_lines, self.hidden_size, device=device)
        line_mask = torch.zeros(batch_size, max_lines, dtype=torch.bool, device=device)

        for b, (features, mask) in enumerate(zip(doc_line_features_list, doc_line_masks_list)):
            num_lines_in_doc = features.shape[0]
            line_features[b, :num_lines_in_doc] = features
            line_mask[b, :num_lines_in_doc] = mask

        # ==================== Step 3: 提取 line_labels ====================
        # 优先使用传入的 line_labels，否则从 token labels 提取
        if line_labels is None and labels is not None:
            line_labels = torch.full((batch_size, max_lines), -100, dtype=torch.long, device=device)

            for b in range(batch_size):
                sample_line_ids = line_ids[b]
                sample_labels = labels[b]
                num_lines = int(line_mask[b].sum().item())

                for line_idx in range(num_lines):
                    # 找到该 line 的第一个 token
                    token_mask = (sample_line_ids == line_idx)
                    if token_mask.any():
                        first_token_idx = token_mask.nonzero(as_tuple=True)[0][0]
                        if first_token_idx < len(sample_labels):
                            label = sample_labels[first_token_idx].item()
                            if label >= 0:
                                line_labels[b, line_idx] = label

        # ==================== Step 4: Line-level 分类 ====================
        all_cls_logits = []
        cls_loss = torch.tensor(0.0, device=device)
        cls_correct = 0
        cls_total = 0

        for b in range(batch_size):
            sample_features = line_features[b]
            num_lines = int(line_mask[b].sum().item())

            if num_lines > 0:
                valid_features = sample_features[:num_lines]
                logits = self.cls_head(valid_features)  # [num_lines, num_classes]
                all_cls_logits.append(logits)

                # 计算损失
                if line_labels is not None:
                    sample_labels = line_labels[b, :num_lines]
                    valid_indices = sample_labels != -100
                    if valid_indices.any():
                        valid_logits = logits[valid_indices]
                        valid_targets = sample_labels[valid_indices]
                        loss = F.cross_entropy(valid_logits, valid_targets)
                        cls_loss = cls_loss + loss

                        # 计算准确率
                        preds = valid_logits.argmax(dim=-1)
                        cls_correct += (preds == valid_targets).sum().item()
                        cls_total += valid_targets.numel()

        # 平均损失
        if cls_total > 0:
            cls_loss = cls_loss / batch_size
            self._cls_acc = cls_correct / cls_total

        # 填充 logits 到相同长度
        if all_cls_logits:
            padded_logits = torch.zeros(batch_size, max_lines, self.num_classes, device=device)
            for b, logits in enumerate(all_cls_logits):
                padded_logits[b, :logits.shape[0]] = logits
        else:
            padded_logits = None

        return TokenClassifierOutput(
            loss=cls_loss,
            logits=padded_logits,
        )

    def get_accuracy(self) -> float:
        """获取最近一次 forward 的准确率"""
        return self._cls_acc

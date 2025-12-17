#!/usr/bin/env python
# coding=utf-8
"""
JointModel - HRDoc 联合训练模型

将 Stage 1/2/3/4 组合为一个端到端模型：
1. Stage 1: LayoutXLM 分类 (产生分类 loss + hidden states)
2. Stage 2: 从 hidden states 提取 line-level 特征
3. Stage 3: ParentFinder 训练 (产生 parent loss)
4. Stage 4: RelationClassifier 训练 (产生 relation loss)

总 Loss = λ1 * L_cls + λ2 * L_par + λ3 * L_rel (论文公式)

此文件只包含模型定义，不包含训练循环、数据加载等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers.modeling_outputs import TokenClassifierOutput


class JointModel(nn.Module):
    """
    联合模型：包含 Stage 1/2/3/4 的所有模块

    论文公式: L_total = L_cls + α₁·L_par + α₂·L_rel
    """

    def __init__(
        self,
        stage1_model,  # LayoutXLMForTokenClassification
        stage3_model: nn.Module,  # ParentFinderGRU 或 SimpleParentFinder
        stage4_model: nn.Module,  # MultiClassRelationClassifier
        feature_extractor,  # LineFeatureExtractor
        lambda_cls: float = 1.0,
        lambda_parent: float = 1.0,
        lambda_rel: float = 1.0,
        use_focal_loss: bool = True,
        use_gru: bool = False,
    ):
        super().__init__()

        self.stage1 = stage1_model
        self.stage3 = stage3_model
        self.stage4 = stage4_model
        self.feature_extractor = feature_extractor

        self.lambda_cls = lambda_cls
        self.lambda_parent = lambda_parent
        self.lambda_rel = lambda_rel
        self.use_gru = use_gru

        # 关系分类损失
        if use_focal_loss:
            from layoutlmft.models.relation_classifier import FocalLoss
            self.relation_criterion = FocalLoss(gamma=2.0)
        else:
            self.relation_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor = None,
        labels: torch.Tensor = None,
        line_ids: Optional[torch.Tensor] = None,
        line_parent_ids: Optional[torch.Tensor] = None,
        line_relations: Optional[torch.Tensor] = None,
        line_bboxes: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """前向传播，返回 loss 和各阶段输出"""

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # ==================== Stage 1: Classification ====================
        stage1_outputs = self.stage1(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            labels=labels,
            output_hidden_states=True,
        )

        cls_loss = stage1_outputs.loss
        logits = stage1_outputs.logits
        hidden_states = stage1_outputs.hidden_states[-1]

        outputs = {
            "loss": cls_loss * self.lambda_cls,
            "cls_loss": cls_loss,
            "logits": logits,
        }

        # 如果没有 line 信息，直接返回（使用 TokenClassifierOutput 格式）
        if line_ids is None or line_parent_ids is None:
            return TokenClassifierOutput(
                loss=outputs["loss"],
                logits=logits,
            )

        # ==================== Stage 2: Feature Extraction ====================
        # 保持梯度流，让 Stage 3/4 的 loss 可以回传到 Stage 1
        text_seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :text_seq_len, :]

        line_features, line_mask = self.feature_extractor.extract_line_features(
            text_hidden, line_ids, pooling="mean"
        )

        # ==================== Stage 3: Parent Finding ====================
        parent_loss = torch.tensor(0.0, device=device)
        parent_correct = 0
        parent_total = 0
        gru_hidden = None  # GRU 隐状态，用于 Stage 4

        if self.lambda_parent > 0:
            if self.use_gru:
                # 论文对齐：获取 GRU 隐状态用于 Stage 4
                parent_logits, gru_hidden = self.stage3(
                    line_features, line_mask, return_gru_hidden=True
                )
                # gru_hidden: [B, L+1, gru_hidden_size]，包括 ROOT

                for b in range(batch_size):
                    sample_parent_ids = line_parent_ids[b]
                    sample_mask = line_mask[b]
                    num_lines = int(sample_mask.sum().item())

                    for child_idx in range(num_lines):
                        gt_parent = sample_parent_ids[child_idx].item()

                        if gt_parent == -100:
                            continue
                        if gt_parent >= child_idx:
                            continue

                        target_idx = gt_parent + 1 if gt_parent >= 0 else 0
                        child_logits = parent_logits[b, child_idx + 1, :child_idx + 2]

                        if torch.isinf(child_logits).all():
                            continue

                        child_logits = torch.where(
                            torch.isinf(child_logits),
                            torch.full_like(child_logits, -1e4),
                            child_logits
                        )

                        target = torch.tensor([target_idx], device=device)
                        loss = F.cross_entropy(child_logits.unsqueeze(0), target)

                        if not torch.isnan(loss):
                            parent_loss = parent_loss + loss
                            parent_total += 1

                        pred_parent = child_logits.argmax().item()
                        if pred_parent == target_idx:
                            parent_correct += 1
            else:
                for b in range(batch_size):
                    sample_features = line_features[b]
                    sample_mask = line_mask[b]
                    sample_parent_ids = line_parent_ids[b]

                    num_lines = sample_mask.sum().item()
                    if num_lines <= 1:
                        continue

                    for child_idx in range(1, int(num_lines)):
                        gt_parent = sample_parent_ids[child_idx].item()

                        if gt_parent < 0 or gt_parent >= child_idx:
                            continue

                        parent_candidates = sample_features[:child_idx]
                        child_feat = sample_features[child_idx]

                        scores = self.stage3(parent_candidates, child_feat)

                        target = torch.tensor([gt_parent], device=device)
                        loss = F.cross_entropy(scores.unsqueeze(0), target)
                        parent_loss = parent_loss + loss

                        pred_parent = scores.argmax().item()
                        if pred_parent == gt_parent:
                            parent_correct += 1
                        parent_total += 1

            if parent_total > 0:
                parent_loss = parent_loss / parent_total
                self._parent_acc = parent_correct / parent_total

            outputs["parent_loss"] = parent_loss
            outputs["loss"] = outputs["loss"] + parent_loss * self.lambda_parent

        # ==================== Stage 4: Relation Classification ====================
        rel_loss = torch.tensor(0.0, device=device)
        rel_correct = 0
        rel_total = 0

        if self.lambda_rel > 0 and line_relations is not None:
            if gru_hidden is None:
                gru_hidden = line_features
                use_gru_offset = False
            else:
                use_gru_offset = True

            for b in range(batch_size):
                sample_mask = line_mask[b]
                sample_parent_ids = line_parent_ids[b]
                sample_relations = line_relations[b]

                num_lines = int(sample_mask.sum().item())

                for child_idx in range(num_lines):
                    parent_idx = sample_parent_ids[child_idx].item()
                    rel_label = sample_relations[child_idx].item()

                    if parent_idx < 0 or parent_idx >= num_lines:
                        continue
                    if rel_label == -100:
                        continue

                    if use_gru_offset:
                        parent_gru_idx = parent_idx + 1
                        child_gru_idx = child_idx + 1
                        parent_feat = gru_hidden[b, parent_gru_idx]
                        child_feat = gru_hidden[b, child_gru_idx]
                    else:
                        parent_feat = gru_hidden[b, parent_idx]
                        child_feat = gru_hidden[b, child_idx]

                    rel_logits = self.stage4(
                        parent_feat.unsqueeze(0),
                        child_feat.unsqueeze(0),
                    )

                    target = torch.tensor([rel_label], device=device)
                    loss = F.cross_entropy(rel_logits, target)
                    rel_loss = rel_loss + loss

                    pred_rel = rel_logits.argmax(dim=1).item()
                    if pred_rel == rel_label:
                        rel_correct += 1
                    rel_total += 1

            if rel_total > 0:
                rel_loss = rel_loss / rel_total
                self._rel_acc = rel_correct / rel_total

            outputs["rel_loss"] = rel_loss
            outputs["loss"] = outputs["loss"] + rel_loss * self.lambda_rel

        # 保存完整的 outputs 供 compute_loss 使用
        self._outputs_dict = outputs
        return TokenClassifierOutput(
            loss=outputs["loss"],
            logits=outputs["logits"],
        )

#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 联合训练模型 - 端到端多任务学习
实现论文中的联合训练方法：L_total = L_cls + α₁·L_par + α₂·L_rel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging

from layoutlmft.models.layoutlmv2 import LayoutLMv2Model, LayoutLMv2PreTrainedModel
from train_parent_finder import ParentFinderGRU, ChildParentDistributionMatrix

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    论文中用于 SubTask1 和 SubTask3
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [*, num_classes] - logits
            targets: [*] - class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class RelationClassifier(nn.Module):
    """
    SubTask 3: 关系分类
    预测 parent-child 之间的关系类型
    """
    def __init__(self, hidden_size=768, num_relations=5, dropout=0.1):
        super().__init__()

        self.num_relations = num_relations

        # 关系分类头：拼接 parent 和 child 的特征
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_relations)
        )

    def forward(self, parent_features, child_features):
        """
        Args:
            parent_features: [batch_size, hidden_size]
            child_features: [batch_size, hidden_size]

        Returns:
            relation_logits: [batch_size, num_relations]
        """
        # 拼接 parent 和 child 特征
        combined = torch.cat([parent_features, child_features], dim=-1)
        logits = self.relation_head(combined)
        return logits


class HRDocJointModel(LayoutLMv2PreTrainedModel):
    """
    HRDoc 联合训练模型

    三个子任务：
    1. SubTask1: 语义单元分类（token-level）
    2. SubTask2: 父节点查找（line-level）
    3. SubTask3: 关系分类（edge-level）

    Loss: L_total = L_cls + α₁·L_par + α₂·L_rel
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels  # 语义类别数（token-level BIO标签）

        # 计算line-level的语义类别数（去掉BIO前缀）
        # 例如：B-TITLE, I-TITLE -> TITLE
        # 假设标签格式为 O, B-X, I-X，那么类别数约为 (num_labels - 1) / 2
        self.num_semantic_classes = getattr(config, 'num_semantic_classes', 16)
        self.num_relations = getattr(config, 'num_relations', 5)

        # 共享的 Encoder
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # ==================== SubTask 1: 语义单元分类 ====================
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.focal_loss_cls = FocalLoss(alpha=0.25, gamma=2.0)

        # ==================== SubTask 2: 父节点查找 ====================
        self.parent_finder = ParentFinderGRU(
            hidden_size=config.hidden_size,
            gru_hidden_size=getattr(config, 'gru_hidden_size', 512),
            num_classes=self.num_semantic_classes,
            dropout=config.hidden_dropout_prob,
            use_soft_mask=getattr(config, 'use_soft_mask', True)
        )

        # ==================== SubTask 3: 关系分类 ====================
        self.relation_classifier = RelationClassifier(
            hidden_size=config.hidden_size,
            num_relations=self.num_relations,
            dropout=config.hidden_dropout_prob
        )
        self.focal_loss_rel = FocalLoss(alpha=0.25, gamma=2.0)

        # Loss权重
        self.alpha1 = getattr(config, 'alpha1', 1.0)  # parent loss权重
        self.alpha2 = getattr(config, 'alpha2', 1.0)  # relation loss权重

        # 初始化权重
        self.init_weights()

        logger.info(f"HRDocJointModel initialized:")
        logger.info(f"  num_labels (token-level): {self.num_labels}")
        logger.info(f"  num_semantic_classes (line-level): {self.num_semantic_classes}")
        logger.info(f"  num_relations: {self.num_relations}")
        logger.info(f"  Loss weights: α1={self.alpha1}, α2={self.alpha2}")

    def extract_line_features(
        self,
        hidden_states: torch.Tensor,
        line_ids: torch.Tensor,
        pooling: str = "mean"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 token-level 的 hidden states 提取 line-level 特征

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            line_ids: [batch_size, seq_len] - 每个token属于哪个line
            pooling: "mean", "max", or "first"

        Returns:
            line_features: [batch_size, max_lines, hidden_size]
            line_mask: [batch_size, max_lines]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        # 找出最大line数
        max_lines = line_ids.max().item() + 1 if line_ids.numel() > 0 else 1

        # 初始化
        line_features = torch.zeros(batch_size, max_lines, hidden_size, device=device)
        line_mask = torch.zeros(batch_size, max_lines, dtype=torch.bool, device=device)

        # 对每个batch处理
        for b in range(batch_size):
            unique_lines = torch.unique(line_ids[b])
            unique_lines = unique_lines[unique_lines >= 0]  # 过滤无效line_id

            for line_id in unique_lines:
                token_mask = (line_ids[b] == line_id)
                line_tokens = hidden_states[b][token_mask]

                if len(line_tokens) == 0:
                    continue

                # 池化
                if pooling == "mean":
                    line_feat = line_tokens.mean(dim=0)
                elif pooling == "max":
                    line_feat = line_tokens.max(dim=0)[0]
                elif pooling == "first":
                    line_feat = line_tokens[0]
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")

                line_features[b, line_id] = line_feat
                line_mask[b, line_id] = True

        return line_features, line_mask

    def forward(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,                    # SubTask1: token-level语义标签
        line_ids=None,                  # token到line的映射
        line_parent_ids=None,           # SubTask2: 每个line的parent索引
        line_relations=None,            # SubTask3: 每个line的relation标签
        line_semantic_labels=None,      # 每个line的语义类别（用于soft-mask）
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        联合训练的forward函数

        Returns:
            如果提供了labels，返回字典包含：
            - loss: 总loss
            - loss_cls: SubTask1 loss
            - loss_parent: SubTask2 loss
            - loss_relation: SubTask3 loss
            - logits: SubTask1的预测
            - parent_logits: SubTask2的预测
            - relation_logits: SubTask3的预测
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ==================== 1. 共享Encoder ====================
        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)

        # ==================== 2. SubTask1: 语义分类 ====================
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]

        loss_cls = None
        if labels is not None:
            # Focal Loss for class imbalance
            loss_cls = self.focal_loss_cls(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )

        # ==================== 3. SubTask2: 父节点查找 ====================
        loss_parent = None
        parent_logits = None

        if line_ids is not None:
            # 提取line-level特征
            line_features, line_mask = self.extract_line_features(
                sequence_output, line_ids, pooling="mean"
            )

            # 通过ParentFinderGRU预测
            parent_logits = self.parent_finder(line_features, line_mask)
            # parent_logits: [batch_size, max_lines+1, max_lines+1]

            if line_parent_ids is not None:
                # 计算parent finding loss
                batch_size, max_lines_plus_1, _ = parent_logits.shape
                max_lines = max_lines_plus_1 - 1

                # 准备targets：[batch_size, max_lines]
                # line_parent_ids[i] 表示第i个line的parent索引
                # -1 表示ROOT（在logits中对应索引0）
                # k 表示第k个line（在logits中对应索引k+1）

                loss_parent_list = []
                for b in range(batch_size):
                    for i in range(max_lines):
                        if not line_mask[b, i]:
                            continue

                        # 第i+1个位置（第i个line，因为index 0是ROOT）
                        logits_i = parent_logits[b, i+1, :i+2]  # 只考虑ROOT和前面的lines

                        # 目标parent索引
                        parent_id = line_parent_ids[b, i].item()
                        if parent_id == -1:
                            target = 0  # ROOT
                        else:
                            target = parent_id + 1  # line索引+1

                        # 确保target在有效范围内
                        if target < len(logits_i):
                            loss_parent_list.append(
                                F.cross_entropy(logits_i.unsqueeze(0), torch.tensor([target], device=logits_i.device))
                            )

                if len(loss_parent_list) > 0:
                    loss_parent = torch.stack(loss_parent_list).mean()
                else:
                    loss_parent = torch.tensor(0.0, device=sequence_output.device)

        # ==================== 4. SubTask3: 关系分类 ====================
        loss_relation = None
        relation_logits = None

        if line_ids is not None and line_parent_ids is not None and line_relations is not None:
            # 对于每个有parent的line，预测relation
            batch_size = sequence_output.size(0)
            relation_logits_list = []
            relation_targets_list = []

            for b in range(batch_size):
                for i in range(line_mask[b].sum().item()):
                    parent_id = line_parent_ids[b, i].item()
                    if parent_id == -1:
                        continue  # ROOT没有relation

                    # 获取parent和child的特征
                    child_feat = line_features[b, i]
                    parent_feat = line_features[b, parent_id]

                    # 预测relation
                    rel_logits = self.relation_classifier(
                        parent_feat.unsqueeze(0),
                        child_feat.unsqueeze(0)
                    )
                    relation_logits_list.append(rel_logits)

                    # 关系标签（需要转换为索引）
                    rel_label = line_relations[b, i]
                    relation_targets_list.append(rel_label)

            if len(relation_logits_list) > 0:
                relation_logits = torch.cat(relation_logits_list, dim=0)
                relation_targets = torch.stack(relation_targets_list)

                # Focal Loss
                loss_relation = self.focal_loss_rel(relation_logits, relation_targets)
            else:
                loss_relation = torch.tensor(0.0, device=sequence_output.device)

        # ==================== 5. 总Loss ====================
        loss = None
        if loss_cls is not None:
            loss = loss_cls

            if loss_parent is not None:
                loss = loss + self.alpha1 * loss_parent

            if loss_relation is not None:
                loss = loss + self.alpha2 * loss_relation

        # ==================== 6. 返回结果 ====================
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_parent': loss_parent,
            'loss_relation': loss_relation,
            'logits': logits,
            'parent_logits': parent_logits,
            'relation_logits': relation_logits,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'attentions': outputs.attentions if output_attentions else None,
        }

    def set_child_parent_matrix(self, M_cp):
        """设置 Child-Parent Distribution Matrix（用于SubTask2）"""
        self.parent_finder.set_child_parent_matrix(M_cp)

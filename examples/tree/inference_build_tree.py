#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 完整推理 Pipeline
实现论文的 Overall Task：将三个子任务串联，输出文档结构树

重要说明 - 数据流转：
推理时，输入数据中的 parent_id、relation、class 都是占位符（默认值：-1/none/paragraph）
这些值需要通过三个阶段的模型推理得到，最终输出的是推理结果而不是输入占位符：

输入数据（占位符）:
    - class: "paragraph" (默认值，不使用)
    - parent_id: -1 (默认值，不使用)
    - relation: "none" (默认值，不使用)
    - tokens, bboxes, line_ids, image: 真实数据（使用）

推理流程（生成真实值）:
    1. SubTask 1: 语义单元分类 (LayoutLMv2) → line_labels (预测 class)
    2. SubTask 2: 父节点查找 (ParentFinder) → parent_indices (预测 parent_id)
    3. SubTask 3: 关系分类 (RelationClassifier) → relation_types (预测 relation)
    4. Overall Task: 构建文档树 (DocumentTree)

输出数据（推理结果）:
    - class: 一阶段预测的语义标签
    - parent_id: 二阶段预测的父节点索引
    - relation: 三阶段预测的关系类型
    - text: 从 tokens 聚合得到
    - box: 从 token bbox 聚合得到

使用方法：
    python examples/tree/inference_build_tree.py \\
        --subtask1_model /path/to/layoutlmv2/checkpoint \\
        --subtask2_model /path/to/parent_finder/best_model.pt \\
        --subtask3_model /path/to/relation_classifier/best_model.pt \\
        --data_dir /path/to/hrdoc_data \\
        --output_dir ./outputs/trees \\
        --max_samples 10
"""

import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据加载相关（和训练一致）
from datasets import load_dataset
from transformers import BertTokenizerFast, set_seed
import layoutlmft.data.datasets.hrdoc
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForTokenClassification, LayoutLMv2Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# 关系分类器
from layoutlmft.models.relation_classifier import (
    MultiClassRelationClassifier,
    LineFeatureExtractor,
    compute_geometry_features,
)

# 导入树结构
from document_tree import DocumentTree, LABEL_MAP, RELATION_MAP

CONFIG_MAPPING.update({"layoutlmv2": LayoutLMv2Config})

logger = logging.getLogger(__name__)


# ==================== 模型定义 ====================

class SimpleParentFinder(torch.nn.Module):
    """父节点查找器（从train_parent_finder_simple.py复制）"""

    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2 + 4, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, child_feat, parent_feats, geom_feats):
        batch_size, num_candidates, hidden_size = parent_feats.shape
        child_feat_expanded = child_feat.unsqueeze(1).expand(batch_size, num_candidates, hidden_size)
        combined = torch.cat([child_feat_expanded, parent_feats, geom_feats], dim=-1)
        scores = self.score_head(combined).squeeze(-1)
        return scores


# ==================== ParentFinderGRU 模型定义 ====================

class ParentFinderGRU(nn.Module):
    """
    基于GRU的父节点查找器（论文方法，从训练代码复制）

    对每个语义单元 u_i，预测其父节点索引 P̂_i ∈ {0, 1, ..., i-1}
    其中 0 表示 ROOT，1到i-1表示之前的语义单元
    """

    def __init__(
        self,
        hidden_size=768,
        gru_hidden_size=512,
        num_classes=16,
        dropout=0.1,
        use_soft_mask=True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.gru_hidden_size = gru_hidden_size
        self.num_classes = num_classes
        self.use_soft_mask = use_soft_mask

        # GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0 if dropout == 0 else dropout
        )

        # 查询向量投影（用于注意力计算）
        self.query_proj = nn.Linear(gru_hidden_size, gru_hidden_size)

        # 键向量投影
        self.key_proj = nn.Linear(gru_hidden_size, gru_hidden_size)

        # 类别预测头（用于预测每个单元的语义类别概率）
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Soft-mask 矩阵（可选）
        self.register_buffer('M_cp', torch.ones(num_classes + 1, num_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, line_features, line_mask):
        """
        Args:
            line_features: 行级特征 [B, L, H]
            line_mask: 有效行mask [B, L]

        Returns:
            parent_logits: [B, L+1, L+1] - 每个位置i的父节点logits（包括ROOT）
            cls_logits: [B, L, num_classes] - 类别预测logits
        """
        batch_size, max_lines, hidden_size = line_features.shape
        device = line_features.device

        # 1. 构建 ROOT 节点
        valid_sum = (line_features * line_mask.unsqueeze(-1)).sum(dim=1)
        valid_count = line_mask.sum(dim=1, keepdim=True).clamp(min=1)
        root_feat = (valid_sum / valid_count).unsqueeze(1)

        # 2. 将 ROOT 拼接到序列最前面
        line_features_with_root = torch.cat([root_feat, line_features], dim=1)
        line_mask_with_root = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=device), line_mask],
            dim=1
        )

        # 3. 通过 GRU 获取隐藏状态
        gru_output, _ = self.gru(line_features_with_root)

        # 4. 预测语义类别概率
        cls_logits = self.cls_head(line_features)
        cls_probs = F.softmax(cls_logits, dim=-1)

        # 为 ROOT 添加类别概率
        root_cls_prob = torch.zeros(batch_size, 1, self.num_classes + 1, device=device)
        root_cls_prob[:, :, -1] = 1.0

        cls_probs_extended = torch.cat([
            cls_probs,
            torch.zeros(batch_size, max_lines, 1, device=device)
        ], dim=-1)

        cls_probs_with_root = torch.cat([root_cls_prob, cls_probs_extended], dim=1)

        # 5. 计算父节点概率
        query = self.query_proj(gru_output)
        key = self.key_proj(gru_output)

        attention_scores = torch.bmm(
            query,
            key.transpose(1, 2)
        ) / (self.gru_hidden_size ** 0.5)

        # 6. Soft-mask 操作
        if self.use_soft_mask and self.M_cp is not None:
            cls_probs_for_child = torch.cat([
                torch.ones(batch_size, 1, self.num_classes, device=device) / self.num_classes,
                cls_probs
            ], dim=1)

            soft_mask = torch.einsum('bjc,cp,bip->bij',
                                     cls_probs_with_root,
                                     self.M_cp,
                                     cls_probs_for_child)

            attention_scores = attention_scores + torch.log(soft_mask.clamp(min=1e-10))

        # 7. 因果mask
        seq_len = max_lines + 1
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # 8. 应用 line_mask
        parent_mask = ~line_mask_with_root.unsqueeze(1)
        child_mask = ~line_mask_with_root.unsqueeze(2)
        combined_mask = parent_mask | child_mask
        attention_scores = attention_scores.masked_fill(combined_mask, float('-inf'))

        return attention_scores, cls_logits


# ==================== 工具函数 ====================

def extract_line_texts(tokens, line_ids, line_id_mapping=None):
    """
    从 token-level 数据聚合成 line-level 文本

    和训练时保持一致的逻辑：根据 line_ids 将 tokens 分组

    Args:
        tokens: token 列表 [num_tokens]
        line_ids: 每个 token 对应的 line_id [num_tokens]（可能是全局编号）
        line_id_mapping: 全局 line_id 到本地索引的映射 {global_id: local_idx}

    Returns:
        line_texts: 每个 line 的文本 [num_lines]
    """
    # 找出有效的 line_ids（忽略特殊 token 的 -1）
    valid_line_ids = [lid for lid in line_ids if lid >= 0]
    if len(valid_line_ids) == 0:
        return []

    # 如果提供了映射，使用映射；否则假设从 0 开始连续
    if line_id_mapping is not None:
        # 获取唯一的 line_ids 并排序
        unique_line_ids = sorted(set(valid_line_ids))
        num_lines = len(unique_line_ids)
        line_texts = []

        # 按映射后的顺序收集文本
        for global_lid in unique_line_ids:
            # 收集这个 line 的所有 tokens
            line_tokens = []
            for i in range(len(tokens)):
                if i < len(line_ids) and line_ids[i] == global_lid:
                    line_tokens.append(tokens[i])

            # 拼接成文本
            line_text = " ".join(line_tokens) if line_tokens else ""
            line_texts.append(line_text)
    else:
        # 原始逻辑：假设 line_id 从 0 开始连续
        num_lines = max(valid_line_ids) + 1
        line_texts = []

        # 按 line_id 分组 tokens
        for line_id in range(num_lines):
            # 收集这个 line 的所有 tokens
            line_tokens = []
            for i in range(len(tokens)):
                if i < len(line_ids) and line_ids[i] == line_id:
                    line_tokens.append(tokens[i])

            # 拼接成文本
            line_text = " ".join(line_tokens) if line_tokens else ""
            line_texts.append(line_text)

    return line_texts


# ==================== 推理函数 ====================

def load_models(
    subtask1_path: str,
    subtask2_path: str,
    subtask3_path: str,
    device: torch.device
):
    """
    加载训练好的模型（和训练一致）

    Args:
        subtask1_path: SubTask 1模型路径（LayoutLMv2）
        subtask2_path: SubTask 2模型路径（父节点查找）
        subtask3_path: SubTask 3模型路径（关系分类）
        device: 设备

    Returns:
        subtask1_model, tokenizer, data_collator, feature_extractor, subtask2_model, subtask3_model
    """
    # ==================== SubTask 1: LayoutLMv2 ====================
    logger.info(f"加载SubTask 1模型: {subtask1_path}")
    config = LayoutLMv2Config.from_pretrained(subtask1_path)
    tokenizer = BertTokenizerFast.from_pretrained(subtask1_path)
    subtask1_model = LayoutLMv2ForTokenClassification.from_pretrained(
        subtask1_path, config=config
    )
    subtask1_model = subtask1_model.to(device)
    subtask1_model.eval()
    logger.info(f"✓ SubTask 1模型加载成功")

    # Data collator（和训练一致）
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )

    # Line feature extractor（用于聚合 token → line）
    feature_extractor = LineFeatureExtractor()

    # ==================== SubTask 2: ParentFinder ====================
    logger.info(f"加载SubTask 2模型: {subtask2_path}")
    subtask2_checkpoint = torch.load(subtask2_path, map_location=device)

    # 尝试判断模型类型
    state_dict = subtask2_checkpoint.get("model_state_dict", subtask2_checkpoint)

    # 检查模型类型（参考训练代码）
    if any("gru" in k for k in state_dict.keys()):
        # ParentFinderGRU模型（full模型）
        subtask2_model = ParentFinderGRU(
            hidden_size=768,
            gru_hidden_size=512,
            num_classes=16,
            dropout=0.1,
            use_soft_mask=True
        )
        logger.info("  检测到ParentFinderGRU模型（full）")

        # 加载M_cp矩阵（如果存在）
        if "M_cp" in subtask2_checkpoint:
            subtask2_model.M_cp = subtask2_checkpoint["M_cp"].to(device)
            logger.info("  ✓ 加载M_cp矩阵")

    elif any("score_head" in k for k in state_dict.keys()):
        # SimpleParentFinder模型
        subtask2_model = SimpleParentFinder(hidden_size=768, dropout=0.1)
        logger.info("  检测到SimpleParentFinder模型")
    else:
        raise ValueError("不支持的SubTask 2模型类型")

    subtask2_model.load_state_dict(state_dict)
    subtask2_model = subtask2_model.to(device)
    subtask2_model.eval()
    logger.info(f"✓ SubTask 2模型加载成功")

    logger.info(f"加载SubTask 3模型: {subtask3_path}")
    subtask3_checkpoint = torch.load(subtask3_path, map_location=device)
    subtask3_model = MultiClassRelationClassifier(
        hidden_size=768,
        num_relations=4,
        use_geometry=True,
        dropout=0.1,
    )
    subtask3_model.load_state_dict(subtask3_checkpoint["model_state_dict"])
    subtask3_model = subtask3_model.to(device)
    subtask3_model.eval()
    logger.info(f"✓ SubTask 3模型加载成功")

    return (
        subtask1_model,
        tokenizer,
        data_collator,
        feature_extractor,
        subtask2_model,
        subtask3_model,
    )


def predict_parents(
    subtask2_model,
    line_features: torch.Tensor,
    line_mask: torch.Tensor,
    line_bboxes: list,
    device: torch.device,
) -> list:
    """
    SubTask 2: 预测每个语义单元的父节点

    Args:
        subtask2_model: 父节点查找模型
        line_features: 行级特征 [num_lines, hidden_size]
        line_mask: 有效行mask [num_lines]
        line_bboxes: 行边界框列表 [num_lines, 4]
        device: 设备

    Returns:
        parent_indices: 每个行的父节点索引列表 [num_lines]
    """
    num_lines = line_features.shape[0]
    parent_indices = []

    # 判断模型类型（参考训练代码的推理方式）
    is_gru_model = isinstance(subtask2_model, ParentFinderGRU)

    with torch.no_grad():
        if is_gru_model:
            # ParentFinderGRU: 一次性预测所有父节点
            line_features_batch = line_features.unsqueeze(0)  # [1, L, H]
            line_mask_batch = line_mask.unsqueeze(0)  # [1, L]

            # forward返回 [B, L+1, L+1] 的logits
            parent_logits, _ = subtask2_model(line_features_batch, line_mask_batch)
            parent_logits = parent_logits.squeeze(0)  # [L+1, L+1]

            # 对每个child（位置1到L），选择得分最高的parent（位置0到child-1）
            for child_idx in range(num_lines):
                # child在序列中的位置是child_idx+1（因为位置0是ROOT）
                child_pos = child_idx + 1
                # 候选parent：位置0到child_idx（对应ROOT和前面的行）
                logits = parent_logits[child_pos, :child_pos+1]  # [child_pos+1]
                pred_pos = torch.argmax(logits).item()
                # 转换为parent索引：位置0表示ROOT(-1)，位置1-N表示行0-(N-1)
                pred_parent_idx = -1 if pred_pos == 0 else pred_pos - 1
                parent_indices.append(pred_parent_idx)

            return parent_indices

        # SimpleParentFinder: 逐个预测（原有逻辑）
        for child_idx in range(num_lines):
            # 检查有效性
            if child_idx >= line_mask.shape[0]:
                parent_indices.append(-1)
                continue

            # 安全地获取mask值
            child_mask_val = line_mask[child_idx]
            if hasattr(child_mask_val, 'item'):
                child_mask_val = child_mask_val.item()
            if not child_mask_val:
                parent_indices.append(-1)
                continue

            # 候选父节点：0 到 child_idx-1
            if child_idx == 0:
                parent_indices.append(-1)  # 第一个节点的父节点是ROOT
                continue

            best_score = -float('inf')
            pred_parent_idx = -1

            # 遍历所有候选父节点
            for candidate_idx in range(child_idx):
                if candidate_idx >= line_mask.shape[0]:
                    continue

                # 安全地获取mask值
                cand_mask_val = line_mask[candidate_idx]
                if hasattr(cand_mask_val, 'item'):
                    cand_mask_val = cand_mask_val.item()
                if not cand_mask_val:
                    continue

                if candidate_idx >= len(line_bboxes) or child_idx >= len(line_bboxes):
                    continue

                # 提取特征
                child_feat = line_features[child_idx].unsqueeze(0)
                parent_feat = line_features[candidate_idx].unsqueeze(0)

                parent_bbox = torch.tensor(line_bboxes[candidate_idx], dtype=torch.float32)
                child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

                # 预测得分
                scores = subtask2_model(child_feat, parent_feat.unsqueeze(1), geom_feat.unsqueeze(1))
                score = scores[0, 0].item()

                if score > best_score:
                    best_score = score
                    pred_parent_idx = candidate_idx

            parent_indices.append(pred_parent_idx)

    return parent_indices


def predict_relations(
    subtask3_model,
    line_features: torch.Tensor,
    parent_indices: list,
    line_bboxes: list,
    device: torch.device,
) -> list:
    """
    SubTask 3: 预测每个父子对之间的关系类型

    Args:
        subtask3_model: 关系分类模型
        line_features: 行级特征 [num_lines, hidden_size]
        parent_indices: 父节点索引列表 [num_lines]
        line_bboxes: 行边界框列表 [num_lines, 4]
        device: 设备

    Returns:
        relation_types: 关系类型列表 [num_lines]
        relation_confidences: 关系置信度列表 [num_lines]
    """
    num_lines = line_features.shape[0]
    relation_types = []
    relation_confidences = []

    with torch.no_grad():
        for child_idx in range(num_lines):
            parent_idx = parent_indices[child_idx]

            # 如果没有父节点，关系为none
            if parent_idx < 0 or parent_idx >= num_lines:
                relation_types.append(0)  # none
                relation_confidences.append(1.0)
                continue

            # 检查有效性
            if parent_idx >= len(line_bboxes) or child_idx >= len(line_bboxes):
                relation_types.append(0)
                relation_confidences.append(0.0)
                continue

            # 提取特征
            parent_feat = line_features[parent_idx].unsqueeze(0)
            child_feat = line_features[child_idx].unsqueeze(0)

            parent_bbox = torch.tensor(line_bboxes[parent_idx], dtype=torch.float32)
            child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
            geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

            # 预测关系
            logits = subtask3_model(parent_feat, child_feat, geom_feat)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()

            relation_types.append(pred_label)
            relation_confidences.append(confidence)

    return relation_types, relation_confidences


def tree_to_hrds_format(tree, page_num=0):
    """
    将DocumentTree转换为HRDS平铺格式

    Args:
        tree: DocumentTree实例
        page_num: 页码

    Returns:
        list: HRDS格式的平铺列表
    """
    flat_list = []

    def traverse(node, parent_id=-1):
        if node.idx >= 0:  # 跳过ROOT
            # 转换为HRDS格式
            # 【重要】所有字段都来自推理结果，不是输入数据的占位符：
            # - class: 来自一阶段预测（node.label）
            # - parent_id: 来自二阶段预测（通过树结构传递）
            # - relation: 来自三阶段预测（node.relation_to_parent）
            # - text: 从 tokens 聚合得到
            # - box: 从 token bbox 聚合得到
            hrds_item = {
                "line_id": node.idx,
                "text": node.text if node.text else "",
                "box": node.bbox,
                "class": node.label.lower().replace("-", "_"),  # 一阶段预测
                "page": page_num,
                "parent_id": parent_id,  # 二阶段预测
                "relation": node.relation_to_parent if node.relation_to_parent else "none",  # 三阶段预测
                "is_meta": (node.relation_to_parent == "meta"),
            }
            flat_list.append(hrds_item)

        # 递归遍历子节点
        for child in node.children:
            traverse(child, node.idx)

    traverse(tree.root)
    return flat_list


def inference_single_page(
    raw_sample: dict,
    subtask1_model,
    tokenizer,
    data_collator,
    feature_extractor,
    subtask2_model,
    subtask3_model,
    device: torch.device,
) -> DocumentTree:
    """
    对单个页面进行完整推理（和训练流程一致）

    重要说明：
    - 推理时，输入数据中的 parent_id、relation、class 都是占位符（默认值：-1/none/paragraph）
    - 这些值需要通过三个阶段的模型推理得到：
      * 一阶段（LayoutLMv2）：预测 class（语义标签）
      * 二阶段（ParentFinder）：预测 parent_id（父节点）
      * 三阶段（RelationClassifier）：预测 relation（关系类型）
    - 输出时必须使用推理结果，而不是输入数据的占位符

    Args:
        raw_sample: 原始样本数据（包含 tokens, bboxes, line_ids, image等）
                   注意：raw_sample中的parent_id/relation/class是占位符，不使用
        subtask1_model: LayoutLMv2 模型
        tokenizer: tokenizer
        data_collator: data collator
        feature_extractor: line feature extractor
        subtask2_model: 父节点查找模型
        subtask3_model: 关系分类模型
        device: 设备

    Returns:
        DocumentTree实例（包含推理得到的class、parent_id、relation）
    """
    # ==================== 一阶段：提取行级特征 ====================
    # 和 extract_line_features.py 保持一致的逻辑

    # Tokenize（和训练一样）
    tokenized = tokenizer(
        raw_sample["tokens"],
        padding="max_length",
        truncation=True,
        max_length=512,
        is_split_into_words=True,
    )

    # 对齐 bbox 和 line_ids（推理时不需要 ground truth labels）
    word_ids = tokenized.word_ids()
    bboxes = []
    token_line_ids = []

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            bboxes.append([0, 0, 0, 0])
            token_line_ids.append(-1)
        elif word_idx != previous_word_idx:
            bboxes.append(raw_sample["bboxes"][word_idx])
            token_line_ids.append(raw_sample["line_ids"][word_idx])
        else:
            bboxes.append(raw_sample["bboxes"][word_idx])
            token_line_ids.append(raw_sample["line_ids"][word_idx])
        previous_word_idx = word_idx

    # ==================== 创建 line_id 映射（全局 -> 本地）====================
    # HRDS 数据使用全局 line_id（跨页面连续），需要映射为本地索引（0, 1, 2, ...）
    valid_global_line_ids = [lid for lid in raw_sample["line_ids"] if lid >= 0]
    if len(valid_global_line_ids) > 0:
        unique_global_line_ids = sorted(set(valid_global_line_ids))
        global_to_local = {global_id: local_idx for local_idx, global_id in enumerate(unique_global_line_ids)}
    else:
        global_to_local = {}

    # 提取 line_texts（从 tokens 聚合），使用映射
    line_texts = extract_line_texts(raw_sample["tokens"], raw_sample["line_ids"], line_id_mapping=global_to_local)

    # 准备模型输入（推理时不需要 labels）
    # 注意：image 需要单独处理，不通过 data_collator
    sample_dict = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "bbox": bboxes,
    }

    batch = data_collator([sample_dict])

    # 转移到GPU
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # 单独处理 image（不通过 data_collator）
    # HuggingFace datasets 将 Array3D 序列化为 list，需要转回 tensor
    image = raw_sample["image"]
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(np.array(image), dtype=torch.uint8)

    # 添加batch维度：(3, 224, 224) -> (1, 3, 224, 224)
    if image.dim() == 3:
        image = image.unsqueeze(0)

    if hasattr(image, "to"):
        image = image.to(device)

    # ==================== 一阶段：模型预测语义标签（class）====================
    # 【重要】输入数据的 class 是占位符（paragraph），这里预测真正的 class
    with torch.no_grad():
        outputs = subtask1_model(
            input_ids=batch["input_ids"],
            bbox=batch["bbox"],
            attention_mask=batch["attention_mask"],
            image=image,
            output_hidden_states=True,
        )

    # 【关键】从模型输出中提取预测的标签（而不是使用输入的 ner_tags 占位符）
    # 这些预测的标签将作为最终输出的 class 字段
    logits = outputs.logits  # [1, seq_len, num_labels]
    predicted_labels = torch.argmax(logits, dim=-1)  # [1, seq_len]
    predicted_labels = predicted_labels.squeeze(0).cpu().tolist()  # [seq_len]

    # 保存 logits 用于计算 top5（squeeze 去掉 batch 维度）
    token_logits = logits.squeeze(0).cpu()  # [seq_len, num_labels]

    # ==================== 聚合成 line-level 的 bbox 和 labels ====================
    # 使用预测的标签（而不是 ground truth）
    # 使用本地索引（映射后的）
    valid_line_ids = [lid for lid in token_line_ids if lid >= 0]
    if len(valid_line_ids) > 0:
        # 获取唯一的全局 line_ids 并排序
        unique_global_line_ids = sorted(set(valid_line_ids))
        num_lines = len(unique_global_line_ids)

        line_bboxes = np.zeros((num_lines, 4), dtype=np.float32)
        line_bboxes[:, 0] = 1e9
        line_bboxes[:, 1] = 1e9
        line_bboxes[:, 2] = -1e9
        line_bboxes[:, 3] = -1e9

        # 收集每个 line 的所有预测标签（用于多数投票）
        # 使用本地索引
        line_label_votes = defaultdict(list)

        # 收集每个 line 的所有 token logits（用于计算 top5）
        line_logits_list = defaultdict(list)

        for bbox, pred_label, token_logit, global_lid in zip(bboxes, predicted_labels, token_logits, token_line_ids):
            if global_lid < 0:
                continue

            # 映射为本地索引
            local_idx = global_to_local[global_lid]

            # 更新 bbox
            x1, y1, x2, y2 = bbox
            line_bboxes[local_idx, 0] = min(line_bboxes[local_idx, 0], x1)
            line_bboxes[local_idx, 1] = min(line_bboxes[local_idx, 1], y1)
            line_bboxes[local_idx, 2] = max(line_bboxes[local_idx, 2], x2)
            line_bboxes[local_idx, 3] = max(line_bboxes[local_idx, 3], y2)

            # 收集预测的标签（忽略特殊标签 0=O，对应"其他"类别）
            if pred_label > 0:  # 只收集有意义的标签
                line_label_votes[local_idx].append(pred_label)

            # 收集 logits
            line_logits_list[local_idx].append(token_logit)

        # 计算每个 line 的最终标签（多数投票）和 top5 预测
        line_labels = []
        line_top5_labels = []
        line_top5_scores = []

        for local_idx in range(num_lines):
            if local_idx in line_label_votes and len(line_label_votes[local_idx]) > 0:
                # 使用多数投票
                most_common_label = Counter(line_label_votes[local_idx]).most_common(1)[0][0]
                line_labels.append(most_common_label)
            else:
                # 如果没有有效标签，使用 0（O 标签）
                line_labels.append(0)

            # 计算 top5：对该 line 的所有 token logits 取平均
            if local_idx in line_logits_list and len(line_logits_list[local_idx]) > 0:
                # 平均所有 token 的 logits
                avg_logits = torch.stack(line_logits_list[local_idx]).mean(dim=0)  # [num_labels]

                # Softmax 得到概率
                probs = torch.softmax(avg_logits, dim=0)  # [num_labels]

                # 取 top5
                top5_probs, top5_indices = torch.topk(probs, k=min(5, len(probs)))

                line_top5_labels.append(top5_indices.tolist())
                line_top5_scores.append(top5_probs.tolist())
            else:
                line_top5_labels.append([])
                line_top5_scores.append([])
    else:
        line_bboxes = np.zeros((0, 4), dtype=np.float32)
        line_labels = []
        line_top5_labels = []
        line_top5_scores = []
        num_lines = 0

    # 提取hidden states
    hidden_states = outputs.hidden_states[-1]
    text_seq_len = batch["input_ids"].shape[1]
    hidden_states = hidden_states[:, :text_seq_len, :]

    # 提取行级特征
    line_ids_tensor = torch.tensor(token_line_ids, device=device).unsqueeze(0)
    line_features, line_mask = feature_extractor.extract_line_features(
        hidden_states, line_ids_tensor, pooling="mean"
    )

    # 移除 batch 维度
    line_features = line_features.squeeze(0)
    line_mask = line_mask.squeeze(0)

    # 过滤掉空行（和训练代码一致）
    num_lines = line_mask.sum().item()
    line_features = line_features[:num_lines]
    line_mask = line_mask[:num_lines]
    line_bboxes = line_bboxes[:num_lines]
    line_labels = line_labels[:num_lines]
    line_texts = line_texts[:num_lines]

    # ==================== 二阶段：预测父节点（parent_id）====================
    # 【重要】输入数据的 parent_id 是占位符（-1），这里预测真正的父节点索引
    # parent_indices: 每个 line 的父节点索引（本地索引，范围 0 到 num_lines-1）
    parent_indices = predict_parents(
        subtask2_model, line_features, line_mask, line_bboxes, device
    )

    # ==================== 三阶段：预测关系类型（relation）====================
    # 【重要】输入数据的 relation 是占位符（none），这里预测真正的关系类型
    # relation_types: 每个 line 与其父节点的关系类型（如 meta, continuation 等）
    relation_types, relation_confidences = predict_relations(
        subtask3_model, line_features, parent_indices, line_bboxes, device
    )

    # ==================== 构建树（使用推理结果，不是输入占位符）====================
    # 【关键】所有值都来自推理结果：
    # - line_labels: 一阶段预测的语义标签（class）
    # - parent_indices: 二阶段预测的父节点索引（parent_id）
    # - relation_types: 三阶段预测的关系类型（relation）
    # - line_texts: 从 tokens 聚合的文本
    # - line_bboxes: 从 token bbox 聚合的边界框
    tree = DocumentTree.from_predictions(
        line_labels=line_labels,           # 一阶段预测
        parent_indices=parent_indices,     # 二阶段预测
        relation_types=relation_types,     # 三阶段预测
        line_bboxes=line_bboxes,          # 聚合得到
        line_texts=line_texts,            # 聚合得到
        label_confidences=None,
        relation_confidences=relation_confidences,
    )

    # 返回树和各阶段的中间结果（用于调试）
    stage_results = {
        "stage1": {
            "line_labels": line_labels,
            "line_texts": line_texts,
            "line_bboxes": line_bboxes.tolist(),
            "line_top5_labels": line_top5_labels,  # Top5 预测的标签 ID
            "line_top5_scores": line_top5_scores,  # Top5 对应的置信度
        },
        "stage2": {
            "parent_indices": parent_indices.tolist() if isinstance(parent_indices, np.ndarray) else parent_indices,
        },
        "stage3": {
            "relation_types": relation_types,
            "relation_confidences": relation_confidences if relation_confidences is not None else None,
        },
    }

    return tree, stage_results


def main():
    parser = argparse.ArgumentParser(description="HRDoc完整推理Pipeline（和训练一致）")
    parser.add_argument(
        "--subtask1_model",
        type=str,
        default="/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000",
        help="SubTask 1模型路径（LayoutLMv2）",
    )
    parser.add_argument(
        "--subtask2_model",
        type=str,
        default="/mnt/e/models/train_data/layoutlmft/parent_finder_full/best_model.pt",
        help="SubTask 2模型路径（父节点查找）",
    )
    parser.add_argument(
        "--subtask3_model",
        type=str,
        default="/mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt",
        help="SubTask 3模型路径（关系分类）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/e/programFile/AIProgram/modelTrain/HRDoc/output",
        help="数据目录（包含 JSON 和 images）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/trees",
        help="输出目录",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="数据集分割",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="最多处理多少个样本（用于快速测试）",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="是否保存JSON格式的树",
    )
    parser.add_argument(
        "--save_markdown",
        action="store_true",
        help="是否保存Markdown格式的树",
    )
    parser.add_argument(
        "--save_ascii",
        action="store_true",
        help="是否保存ASCII格式的树",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="hrds",
        choices=["tree", "hrds", "both"],
        help="输出格式：tree=嵌套树，hrds=HRDS平铺格式，both=两种都输出",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # 创建输出目录（包括各阶段子目录）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 注意：stage 目录现在在每个文档的子目录中创建，不再创建全局的 stage 目录

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")

    # 设置数据集路径环境变量（传递给 hrdoc.py）
    os.environ["HRDOC_DATA_DIR"] = args.data_dir

    # 加载模型（和训练一致）
    logger.info("\n" + "="*80)
    logger.info("加载模型")
    logger.info("="*80)
    (
        subtask1_model,
        tokenizer,
        data_collator,
        feature_extractor,
        subtask2_model,
        subtask3_model,
    ) = load_models(
        args.subtask1_model,
        args.subtask2_model,
        args.subtask3_model,
        device
    )

    # 加载数据集（和训练一致）
    logger.info("\n" + "="*80)
    logger.info("加载数据集")
    logger.info("="*80)
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))
    logger.info(f"数据集包含: {list(datasets.keys())}")

    # 根据 split 参数选择数据
    if args.split == "train" and "train" in datasets:
        dataset = datasets["train"]
    elif args.split == "validation" and "validation" in datasets:
        dataset = datasets["validation"]
    elif args.split == "test" and "test" in datasets:
        dataset = datasets["test"]
    else:
        raise ValueError(f"数据集中没有 {args.split} split，可用的有: {list(datasets.keys())}")

    logger.info(f"使用 {args.split} 集，共 {len(dataset)} 个样本")

    # 限制样本数量
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
        logger.info(f"限制处理前 {args.max_samples} 个样本")

    # 推理（和训练一致）
    logger.info("\n" + "="*80)
    logger.info("开始推理")
    logger.info("="*80)

    trees = []
    # 记录当前文档名称和文档内的页面映射
    current_doc_name = None
    doc_page_counter = {}  # {doc_name: page_counter}

    for i, raw_sample in enumerate(tqdm(dataset, desc="推理进度")):
        try:
            # 获取文档名称和页码
            document_name = raw_sample.get("document_name", f"document_{i}")
            page_number = raw_sample.get("page_number", i)

            # 初始化文档计数器
            if document_name not in doc_page_counter:
                doc_page_counter[document_name] = 0

            tree, stage_results = inference_single_page(
                raw_sample,
                subtask1_model,
                tokenizer,
                data_collator,
                feature_extractor,
                subtask2_model,
                subtask3_model,
                device
            )
            trees.append(tree)

            # 创建文档专属的输出目录
            doc_output_dir = output_dir / document_name
            doc_output_dir.mkdir(parents=True, exist_ok=True)

            doc_stage1_dir = doc_output_dir / "stage1"
            doc_stage2_dir = doc_output_dir / "stage2"
            doc_stage3_dir = doc_output_dir / "stage3"
            doc_stage1_dir.mkdir(exist_ok=True)
            doc_stage2_dir.mkdir(exist_ok=True)
            doc_stage3_dir.mkdir(exist_ok=True)

            # 使用原始页码作为文件名
            page_idx = page_number

            # ==================== 保存各阶段的中间结果（用于调试）====================
            # Stage 1: 只有 class, text, bbox, 以及 top5 预测
            stage1_data = []
            for idx in range(len(stage_results["stage1"]["line_labels"])):
                item = {
                    "line_id": idx,
                    "text": stage_results["stage1"]["line_texts"][idx],
                    "box": stage_results["stage1"]["line_bboxes"][idx],
                    "class": LABEL_MAP.get(stage_results["stage1"]["line_labels"][idx], f"unknown_{stage_results['stage1']['line_labels'][idx]}"),
                    "page": page_idx,
                }

                # 添加 top5 预测信息
                if idx < len(stage_results["stage1"]["line_top5_labels"]):
                    top5_labels = stage_results["stage1"]["line_top5_labels"][idx]
                    top5_scores = stage_results["stage1"]["line_top5_scores"][idx]

                    # 转换为标签名称和分数的列表
                    top5_predictions = []
                    for label_id, score in zip(top5_labels, top5_scores):
                        label_name = LABEL_MAP.get(label_id, f"unknown_{label_id}")
                        top5_predictions.append({
                            "label": label_name,
                            "label_id": label_id,
                            "score": float(score)
                        })
                    item["top5_predictions"] = top5_predictions

                stage1_data.append(item)

            stage1_path = doc_stage1_dir / f"page_{page_idx:04d}.json"
            with open(stage1_path, 'w', encoding='utf-8') as f:
                json.dump(stage1_data, f, indent=2, ensure_ascii=False)

            # Stage 2: 添加 parent_id
            stage2_data = []
            for idx in range(len(stage_results["stage1"]["line_labels"])):
                stage2_data.append({
                    "line_id": idx,
                    "text": stage_results["stage1"]["line_texts"][idx],
                    "box": stage_results["stage1"]["line_bboxes"][idx],
                    "class": LABEL_MAP.get(stage_results["stage1"]["line_labels"][idx], f"unknown_{stage_results['stage1']['line_labels'][idx]}"),
                    "page": page_idx,
                    "parent_id": stage_results["stage2"]["parent_indices"][idx],
                })
            stage2_path = doc_stage2_dir / f"page_{page_idx:04d}.json"
            with open(stage2_path, 'w', encoding='utf-8') as f:
                json.dump(stage2_data, f, indent=2, ensure_ascii=False)

            # Stage 3: 添加 relation（最终结果，等同于主输出）
            stage3_data = []
            for idx in range(len(stage_results["stage1"]["line_labels"])):
                stage3_data.append({
                    "line_id": idx,
                    "text": stage_results["stage1"]["line_texts"][idx],
                    "box": stage_results["stage1"]["line_bboxes"][idx],
                    "class": LABEL_MAP.get(stage_results["stage1"]["line_labels"][idx], f"unknown_{stage_results['stage1']['line_labels'][idx]}"),
                    "page": page_idx,
                    "parent_id": stage_results["stage2"]["parent_indices"][idx],
                    "relation": stage_results["stage3"]["relation_types"][idx],
                })
            stage3_path = doc_stage3_dir / f"page_{page_idx:04d}.json"
            with open(stage3_path, 'w', encoding='utf-8') as f:
                json.dump(stage3_data, f, indent=2, ensure_ascii=False)

            # 根据output_format保存最终结果
            if args.output_format in ["hrds", "both"]:
                # 保存HRDS格式（从树转换，包含层级结构）
                hrds_data = tree_to_hrds_format(tree, page_num=page_idx)
                hrds_path = doc_output_dir / f"page_{page_idx:04d}.json"
                with open(hrds_path, 'w', encoding='utf-8') as f:
                    json.dump(hrds_data, f, indent=2, ensure_ascii=False)

            if args.output_format in ["tree", "both"]:
                # 保存树格式
                if args.save_json:
                    json_path = output_dir / f"tree_{page_idx:04d}.json"
                    tree.to_json(str(json_path))

                if args.save_markdown:
                    md_path = output_dir / f"tree_{page_idx:04d}.md"
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(tree.to_markdown())

                if args.save_ascii:
                    ascii_path = output_dir / f"tree_{page_idx:04d}_ascii.txt"
                    with open(ascii_path, 'w', encoding='utf-8') as f:
                        f.write(tree.visualize_ascii())

        except Exception as e:
            import traceback
            logger.error(f"处理页面 {i} 时出错: {str(e)}")
            logger.error(f"详细错误:\n{traceback.format_exc()}")
            continue

    # 统计
    logger.info("\n" + "="*80)
    logger.info("推理完成！")
    logger.info("="*80)
    logger.info(f"成功处理: {len(trees)}/{len(dataset)} 个页面")
    logger.info(f"输出目录: {output_dir}")

    # 汇总统计
    if trees:
        total_nodes = sum(tree.get_statistics()["total_nodes"] for tree in trees)
        avg_nodes = total_nodes / len(trees)
        logger.info(f"\n平均每页节点数: {avg_nodes:.1f}")

        # 保存汇总
        summary = {
            "total_pages": len(trees),
            "total_nodes": total_nodes,
            "average_nodes_per_page": avg_nodes,
            "trees": [tree.to_dict() for tree in trees[:5]],  # 只保存前5个完整树
        }
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ 汇总信息保存到: {summary_path}")


if __name__ == "__main__":
    main()

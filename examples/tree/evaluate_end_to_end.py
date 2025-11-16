#!/usr/bin/env python
# coding=utf-8
"""
End-to-End 三阶段评估脚本（完整论文方法）
评估 SubTask 1 (Semantic Classification) + SubTask 2 (Parent Finding - Full) + SubTask 3 (Relation Classification)
"""

import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from datasets import load_dataset
from transformers import BertTokenizerFast, set_seed

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layoutlmft.data.datasets.hrdoc
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.models.relation_classifier import MultiClassRelationClassifier, compute_geometry_features, LineFeatureExtractor
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForTokenClassification, LayoutLMv2Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

CONFIG_MAPPING.update({"layoutlmv2": LayoutLMv2Config})

logger = logging.getLogger(__name__)


# ParentFinderGRU模型定义（Full模式 - 论文方法）
class ParentFinderGRU(nn.Module):
    """
    基于GRU的父节点查找器（论文完整方法）
    包含：GRU解码器 + 注意力机制 + Soft-mask + 语义分类头
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
        # M_cp: [num_classes+1, num_classes]
        self.register_buffer('M_cp', torch.ones(num_classes + 1, num_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, line_features, line_mask):
        """
        Args:
            line_features: [B, L, H] - 行特征
            line_mask: [B, L] - 有效行mask

        Returns:
            parent_logits: [B, L+1, L+1] - 每个位置对候选父节点的logits
            cls_logits: [B, L, num_classes] - 每个位置的语义类别logits（Task 1）
        """
        batch_size, max_lines, hidden_size = line_features.shape
        device = line_features.device

        # 1. 构建 ROOT 节点（论文方法：所有单元表示的平均）
        valid_sum = (line_features * line_mask.unsqueeze(-1)).sum(dim=1)  # [B, H]
        valid_count = line_mask.sum(dim=1, keepdim=True).clamp(min=1)
        root_feat = valid_sum / valid_count  # [B, H]
        root_feat = root_feat.unsqueeze(1)  # [B, 1, H]

        # 2. 预测每个单元的语义类别概率（用于 soft-mask 和 Task 1评估）
        cls_logits = self.cls_head(line_features)  # [B, L, num_classes]
        cls_probs = F.softmax(cls_logits, dim=-1)  # [B, L, num_classes]

        # 为 ROOT 添加一个虚拟的类别概率分布
        root_cls_prob = torch.zeros(batch_size, 1, self.num_classes + 1, device=device)
        root_cls_prob[:, :, -1] = 1.0  # ROOT 类别（最后一个）

        # 原始类别概率需要扩展到 num_classes+1
        cls_probs_extended = torch.cat([
            cls_probs,
            torch.zeros(batch_size, max_lines, 1, device=device)
        ], dim=-1)  # [B, L, num_classes+1]

        # 将 ROOT 和行的类别概率拼接
        cls_probs_with_root = torch.cat([root_cls_prob, cls_probs_extended], dim=1)  # [B, L+1, num_classes+1]

        # 3. 将 ROOT 节点拼接到序列最前面
        line_features_with_root = torch.cat([root_feat, line_features], dim=1)  # [B, L+1, H]

        # 为 ROOT 创建 mask（ROOT 始终有效）
        root_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        line_mask_with_root = torch.cat([root_mask, line_mask], dim=1)  # [B, L+1]

        # 4. GRU 编码
        gru_out, _ = self.gru(line_features_with_root)  # [B, L+1, gru_hidden]
        gru_out = self.dropout(gru_out)

        # 5. 注意力机制：每个位置 i 计算对之前位置的注意力分数
        queries = self.query_proj(gru_out)  # [B, L+1, gru_hidden]
        keys = self.key_proj(gru_out)  # [B, L+1, gru_hidden]

        # 计算注意力分数 [B, L+1, L+1]
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # [B, L+1, L+1]
        attention_scores = attention_scores / (self.gru_hidden_size ** 0.5)

        # 6. Causal Masking：位置 i 只能看到 0 到 i-1
        seq_len = max_lines + 1
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attention_scores = attention_scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

        # 7. Soft-mask 操作（根据语义类别约束）
        if self.use_soft_mask and self.M_cp is not None:
            # cls_probs 只包含原始行（不包括 ROOT），形状是 [B, L, C]
            cls_probs_for_child = torch.cat([
                torch.ones(batch_size, 1, self.num_classes, device=device) / self.num_classes,
                cls_probs
            ], dim=1)  # [B, L+1, C]

            # 计算 soft_mask: P_dom(i,j) = P_cls_j · M_cp · P_cls_i^T
            soft_mask = torch.einsum('bjc,cp,bip->bij',
                                     cls_probs_with_root,  # [B, L+1, C+1] - parent 类别概率
                                     self.M_cp,            # [C+1, C] - 分布矩阵
                                     cls_probs_for_child)  # [B, L+1, C] - child 类别概率

            # 将 soft-mask 与注意力分数相乘
            attention_scores = attention_scores + torch.log(soft_mask.clamp(min=1e-10))

        # 8. Mask掉无效位置
        attention_scores = attention_scores.masked_fill(~line_mask_with_root.unsqueeze(1), float('-inf'))

        return attention_scores, cls_logits


# 语义标签映射（16个类别）
SEMANTIC_LABELS = {
    "title": 0, "author": 1, "abstract": 2, "keywords": 3,
    "section": 4, "paragraph": 5, "list": 6, "table": 7,
    "figure": 8, "caption": 9, "equation": 10, "algorithm": 11,
    "footer": 12, "header": 13, "reference": 14, "other": 15
}
SEMANTIC_NAMES = list(SEMANTIC_LABELS.keys())

# 关系标签映射（4个类别）
RELATION_LABELS = {
    "none": 0,
    "connect": 1,
    "contain": 2,
    "equality": 3,
    "meta": 0,
}
RELATION_NAMES = ["none", "connect", "contain", "equality"]


def load_models(subtask1_path, subtask2_path, subtask3_path, device):
    """加载所有模型"""
    # 加载 SubTask 1 模型 (LayoutLMv2)
    logger.info(f"加载SubTask 1模型 (LayoutLMv2): {subtask1_path}")
    config = LayoutLMv2Config.from_pretrained(subtask1_path)
    tokenizer = BertTokenizerFast.from_pretrained(subtask1_path)
    subtask1_model = LayoutLMv2ForTokenClassification.from_pretrained(subtask1_path, config=config)
    subtask1_model = subtask1_model.to(device)
    subtask1_model.eval()
    logger.info(f"✓ SubTask 1加载成功 (num_labels: {config.num_labels})")

    # 加载 SubTask 2 模型 (ParentFinderGRU)
    logger.info(f"加载SubTask 2模型 (ParentFinderGRU - Full): {subtask2_path}")
    subtask2_checkpoint = torch.load(subtask2_path, map_location=device)

    subtask2_model = ParentFinderGRU(
        hidden_size=768,
        gru_hidden_size=512,
        num_classes=16,
        dropout=0.1,
        use_soft_mask=True
    )
    subtask2_model.load_state_dict(subtask2_checkpoint["model_state_dict"])

    # 加载 M_cp 矩阵（如果有）
    if "M_cp" in subtask2_checkpoint:
        subtask2_model.M_cp = subtask2_checkpoint["M_cp"].to(device)
        logger.info(f"✓ 加载 M_cp 矩阵: {subtask2_model.M_cp.shape}")

    subtask2_model = subtask2_model.to(device)
    subtask2_model.eval()
    logger.info(f"✓ SubTask 2加载成功 (Acc: {subtask2_checkpoint.get('best_acc', 'N/A')})")

    # 加载 SubTask 3 模型 (MultiClassRelationClassifier)
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
    logger.info(f"✓ SubTask 3加载成功 (F1: {subtask3_checkpoint.get('best_f1', 'N/A')})")

    return subtask1_model, tokenizer, subtask2_model, subtask3_model


def load_validation_data(features_dir, max_chunks=None):
    """加载验证集数据（预处理的特征）"""
    import glob
    pattern = os.path.join(features_dir, "validation_line_features_chunk_*.pkl")
    chunk_files = sorted(glob.glob(pattern))

    if max_chunks is not None:
        chunk_files = chunk_files[:max_chunks]

    if len(chunk_files) == 0:
        raise ValueError(f"没有找到validation特征文件: {pattern}")

    logger.info(f"找到 {len(chunk_files)} 个validation chunk文件")
    page_features = []
    for chunk_file in chunk_files:
        with open(chunk_file, "rb") as f:
            chunk_data = pickle.load(f)
        page_features.extend(chunk_data)

    logger.info(f"总共加载了 {len(page_features)} 页的验证集特征")
    return page_features


def load_raw_validation_data(data_dir, max_samples=None):
    """加载原始验证集数据（用于 SubTask 1 评估）"""
    os.environ["HRDOC_DATA_DIR"] = data_dir

    logger.info("加载原始验证集数据...")
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))

    if "test" not in datasets:
        raise ValueError("数据集中没有test集（验证集）")

    validation_dataset = datasets["test"]

    if max_samples is not None and max_samples > 0:
        logger.info(f"限制验证集样本数: {max_samples}")
        if len(validation_dataset) > max_samples:
            validation_dataset = validation_dataset.select(range(max_samples))

    logger.info(f"加载了 {len(validation_dataset)} 个验证集样本")
    return validation_dataset


def evaluate_subtask1_semantic_classification(subtask1_model, tokenizer, validation_dataset, device):
    """
    评估SubTask 1：语义分类（完整论文方法）

    流程：
    1. Token-level 推理（使用 LayoutLMv2）
    2. Line-level 聚合（多数投票）
    3. 计算 line-level F1-score（Micro + Macro）
    """
    logger.info("\n" + "="*80)
    logger.info("SubTask 1: Semantic Classification (Token→Line)")
    logger.info("="*80)

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )

    # 收集所有预测和真实标签
    all_pred_labels = []
    all_gt_labels = []

    with torch.no_grad():
        for sample_idx, raw_sample in enumerate(tqdm(validation_dataset, desc="SubTask 1评估")):
            # 准备输入数据
            tokenized = raw_sample["tokenized"]
            bboxes = np.array(tokenized["bbox"])
            labels = np.array(tokenized["ner_tags"])
            token_line_ids = raw_sample["token_line_ids"]

            # 准备模型输入
            sample_dict = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "bbox": bboxes,
                "labels": labels,
                "image": raw_sample["image"],
            }

            batch = data_collator([sample_dict])

            # 转移到GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif k == "image" and hasattr(v, "to"):
                    batch[k] = v.to(device)

            # Token-level 推理
            outputs = subtask1_model(
                input_ids=batch["input_ids"],
                bbox=batch["bbox"],
                attention_mask=batch["attention_mask"],
                image=batch["image"],
            )

            # 获取 token-level 预测
            logits = outputs.logits  # [batch_size, seq_len, num_labels]
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]  # [seq_len]

            # Line-level 聚合（多数投票）
            num_lines = raw_sample["num_lines"]
            line_label_votes_pred = defaultdict(list)
            line_label_votes_gt = defaultdict(list)

            for pred_label, gt_label, lid in zip(predictions, labels, token_line_ids):
                if lid < 0:
                    continue
                # 忽略特殊标签
                if gt_label != -100:
                    line_label_votes_pred[lid].append(pred_label)
                    line_label_votes_gt[lid].append(gt_label)

            # 计算每个 line 的最终标签（多数投票）
            for lid in range(num_lines):
                if lid in line_label_votes_gt and len(line_label_votes_gt[lid]) > 0:
                    # 预测标签（多数投票）
                    pred_line_label = Counter(line_label_votes_pred[lid]).most_common(1)[0][0]
                    # GT 标签（多数投票）
                    gt_line_label = Counter(line_label_votes_gt[lid]).most_common(1)[0][0]

                    all_pred_labels.append(pred_line_label)
                    all_gt_labels.append(gt_line_label)

    if len(all_gt_labels) == 0:
        logger.warning("没有找到有效的语义标签数据")
        return None

    # 计算 F1 指标
    accuracy = accuracy_score(all_gt_labels, all_pred_labels)

    # Micro F1 (整体准确率)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_gt_labels, all_pred_labels, average="micro", zero_division=0
    )

    # Macro F1 (各类别平均)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_gt_labels, all_pred_labels, average="macro", zero_division=0
    )

    # 分类报告
    unique_labels = sorted(set(all_gt_labels))
    target_names = [SEMANTIC_NAMES[i] if i < len(SEMANTIC_NAMES) else f"class_{i}" for i in unique_labels]
    report = classification_report(
        all_gt_labels, all_pred_labels,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    )

    logger.info(f"\nSubTask 1 结果 (Line-level):")
    logger.info(f"  总样本数: {len(all_gt_labels)}")
    logger.info(f"  准确率: {accuracy:.4f}")
    logger.info(f"  Micro-F1: {micro_f1:.4f}")
    logger.info(f"  Macro-F1: {macro_f1:.4f}")
    logger.info(f"\n分类报告:\n{report}")

    return {
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


def evaluate_subtask2_parent_finding(subtask2_model, page_features, device):
    """
    评估SubTask 2：父节点查找（使用GRU方法）
    """
    logger.info("\n" + "="*80)
    logger.info("评估SubTask 2: Parent Finding (GRU + Attention + Soft-mask)")
    logger.info("="*80)

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for page_idx, page_data in enumerate(tqdm(page_features, desc="SubTask 2评估")):
            line_features = page_data["line_features"].to(device)  # [1, L, H]
            line_mask = page_data["line_mask"].to(device)  # [1, L]
            line_parent_ids = page_data["line_parent_ids"]

            # 前向传播
            parent_logits, _ = subtask2_model(line_features, line_mask)  # [1, L+1, L+1]
            parent_logits = parent_logits.squeeze(0)  # [L+1, L+1]
            line_mask = line_mask.squeeze(0)  # [L]

            num_lines = len(line_parent_ids)
            mask_len = line_mask.shape[0]
            page_predictions = {}

            for i in range(num_lines):
                # 检查边界
                if i >= mask_len or not line_mask[i]:
                    continue

                parent_id_gt = line_parent_ids[i]
                if parent_id_gt == -1:  # ROOT 不需要预测
                    continue

                # 行 i 在新序列中的位置是 i+1（因为 ROOT 在位置 0）
                logits_i = parent_logits[i+1, :i+1]  # 只看 ROOT 和行 0~i-1

                # 预测
                pred_idx = torch.argmax(logits_i).item()

                # 映射回原始索引
                if pred_idx == 0:
                    pred_parent_idx = -1  # ROOT
                else:
                    pred_parent_idx = pred_idx - 1  # 行索引

                page_predictions[i] = pred_parent_idx

                # 统计
                total += 1
                if pred_parent_idx == parent_id_gt:
                    correct += 1

            predictions.append({
                "page_idx": page_idx,
                "predictions": page_predictions,
            })

    accuracy = correct / total if total > 0 else 0
    logger.info(f"\nSubTask 2结果:")
    logger.info(f"  准确率: {accuracy:.4f} ({correct}/{total})")

    return {"accuracy": accuracy}, predictions


def evaluate_subtask3_with_gt_parents(subtask3_model, page_features, device):
    """评估SubTask 3（使用GT父节点）"""
    logger.info("\n" + "="*80)
    logger.info("评估SubTask 3: Relation Classification (使用GT父节点)")
    logger.info("="*80)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for page_data in tqdm(page_features, desc="SubTask 3 (GT)"):
            line_features = page_data["line_features"].squeeze(0).to(device)
            line_mask = page_data["line_mask"].squeeze(0)
            line_parent_ids = page_data["line_parent_ids"]
            line_relations = page_data["line_relations"]
            line_bboxes = page_data["line_bboxes"]

            num_lines = len(line_parent_ids)

            for child_idx in range(num_lines):
                parent_idx = line_parent_ids[child_idx]
                relation = line_relations[child_idx]

                if parent_idx < 0 or parent_idx >= num_lines:
                    continue
                if relation not in RELATION_LABELS or relation in ["none", "meta"]:
                    continue
                if child_idx >= line_mask.shape[0] or not line_mask[child_idx]:
                    continue
                if parent_idx >= line_mask.shape[0] or not line_mask[parent_idx]:
                    continue

                parent_feat = line_features[parent_idx].unsqueeze(0)
                child_feat = line_features[child_idx].unsqueeze(0)

                parent_bbox = torch.tensor(line_bboxes[parent_idx], dtype=torch.float32)
                child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

                logits = subtask3_model(parent_feat, child_feat, geom_feat)
                pred_label = torch.argmax(logits, dim=1).item()
                gt_label = RELATION_LABELS[relation]

                all_labels.append(gt_label)
                all_preds.append(pred_label)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=RELATION_NAMES, zero_division=0)

    logger.info(f"\nSubTask 3结果 (GT父节点):")
    logger.info(f"  准确率: {accuracy:.4f}")
    logger.info(f"  F1 (macro): {f1:.4f}")
    logger.info(f"  总样本: {len(all_labels)}")
    logger.info(f"\n混淆矩阵:\n{cm}")
    logger.info(f"\n分类报告:\n{report}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate_end_to_end(subtask3_model, page_features, subtask2_predictions, device):
    """End-to-End评估（使用预测的父节点）"""
    logger.info("\n" + "="*80)
    logger.info("评估End-to-End: 预测父节点 → 预测关系")
    logger.info("="*80)

    all_labels = []
    all_preds = []
    parent_correct = 0
    parent_total = 0
    both_correct = 0
    total_samples = 0

    with torch.no_grad():
        for pred_info in tqdm(subtask2_predictions, desc="End-to-End"):
            page_idx = pred_info["page_idx"]
            pred_parents = pred_info["predictions"]

            page_data = page_features[page_idx]
            line_features = page_data["line_features"].squeeze(0).to(device)
            line_mask = page_data["line_mask"].squeeze(0)
            line_parent_ids = page_data["line_parent_ids"]
            line_relations = page_data["line_relations"]
            line_bboxes = page_data["line_bboxes"]

            num_lines = len(line_parent_ids)

            for child_idx, pred_parent_idx in pred_parents.items():
                gt_parent_idx = line_parent_ids[child_idx]
                gt_relation = line_relations[child_idx]

                if gt_parent_idx < 0 or gt_parent_idx >= num_lines:
                    continue
                if gt_relation not in RELATION_LABELS or gt_relation in ["none", "meta"]:
                    continue
                if child_idx >= line_mask.shape[0] or not line_mask[child_idx]:
                    continue

                total_samples += 1
                parent_is_correct = (pred_parent_idx == gt_parent_idx)
                if parent_is_correct:
                    parent_correct += 1
                parent_total += 1

                # 使用预测的父节点
                if pred_parent_idx >= 0 and pred_parent_idx < num_lines:
                    if pred_parent_idx >= line_mask.shape[0] or not line_mask[pred_parent_idx]:
                        pred_relation_label = 0
                    else:
                        parent_feat = line_features[pred_parent_idx].unsqueeze(0)
                        child_feat = line_features[child_idx].unsqueeze(0)

                        parent_bbox = torch.tensor(line_bboxes[pred_parent_idx], dtype=torch.float32)
                        child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                        geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

                        logits = subtask3_model(parent_feat, child_feat, geom_feat)
                        pred_relation_label = torch.argmax(logits, dim=1).item()
                else:
                    pred_relation_label = 0

                gt_relation_label = RELATION_LABELS[gt_relation]
                relation_is_correct = (pred_relation_label == gt_relation_label)

                if parent_is_correct and relation_is_correct:
                    both_correct += 1

                all_labels.append(gt_relation_label)
                all_preds.append(pred_relation_label)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=RELATION_NAMES, zero_division=0)

    end_to_end_acc = both_correct / total_samples if total_samples > 0 else 0
    parent_acc = parent_correct / parent_total if parent_total > 0 else 0

    logger.info(f"\nEnd-to-End结果:")
    logger.info(f"  总样本: {total_samples}")
    logger.info(f"  父节点准确率: {parent_acc:.4f} ({parent_correct}/{parent_total})")
    logger.info(f"  关系分类准确率: {accuracy:.4f}")
    logger.info(f"  关系分类F1 (macro): {f1:.4f}")
    logger.info(f"  End-to-End准确率 (父+关系都对): {end_to_end_acc:.4f} ({both_correct}/{total_samples})")
    logger.info(f"\n混淆矩阵:\n{cm}")
    logger.info(f"\n分类报告:\n{report}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "end_to_end_accuracy": end_to_end_acc,
        "parent_accuracy": parent_acc,
    }


def main():
    # 配置
    data_dir = os.getenv("HRDOC_DATA_DIR", "/mnt/e/models/data/Section/HRDS")

    features_dir = os.getenv(
        "LAYOUTLMFT_FEATURES_DIR",
        "/mnt/e/models/train_data/layoutlmft/line_features"
    )

    subtask1_model_path = os.getenv(
        "SUBTASK1_MODEL_PATH",
        "/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000"
    )

    subtask2_model_path = os.getenv(
        "SUBTASK2_MODEL_PATH",
        "/mnt/e/models/train_data/layoutlmft/parent_finder_full/best_model.pt"
    )

    subtask3_model_path = os.getenv(
        "SUBTASK3_MODEL_PATH",
        "/mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt"
    )

    max_chunks = int(os.getenv("MAX_CHUNKS", "-1"))
    if max_chunks == -1:
        max_chunks = None

    max_samples = int(os.getenv("MAX_SAMPLES", "-1"))
    if max_samples == -1:
        max_samples = None

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("="*80)
    logger.info("HRDoc End-to-End 评估（完整论文方法 - Full模式）")
    logger.info("="*80)
    logger.info(f"使用设备: {device}")
    logger.info(f"数据集目录: {data_dir}")
    logger.info(f"特征目录: {features_dir}")
    logger.info(f"SubTask 1模型: {subtask1_model_path}")
    logger.info(f"SubTask 2模型 (Full): {subtask2_model_path}")
    logger.info(f"SubTask 3模型: {subtask3_model_path}")

    # 加载所有模型
    subtask1_model, tokenizer, subtask2_model, subtask3_model = load_models(
        subtask1_model_path,
        subtask2_model_path,
        subtask3_model_path,
        device
    )

    # 加载原始验证数据（用于 SubTask 1）
    validation_dataset = load_raw_validation_data(data_dir, max_samples)

    # 加载预处理特征（用于 SubTask 2/3）
    page_features = load_validation_data(features_dir, max_chunks)

    # 1. 评估SubTask 1: 语义分类（Token→Line）
    subtask1_metrics = evaluate_subtask1_semantic_classification(
        subtask1_model, tokenizer, validation_dataset, device
    )

    # 2. 评估SubTask 2: 父节点查找
    subtask2_metrics, subtask2_predictions = evaluate_subtask2_parent_finding(
        subtask2_model, page_features, device
    )

    # 3. 评估SubTask 3（GT父节点）
    subtask3_gt_metrics = evaluate_subtask3_with_gt_parents(
        subtask3_model, page_features, device
    )

    # 4. 评估End-to-End
    end_to_end_metrics = evaluate_end_to_end(
        subtask3_model, page_features, subtask2_predictions, device
    )

    # 总结
    logger.info("\n" + "="*80)
    logger.info("评估总结（完整3阶段 - Full模式）")
    logger.info("="*80)

    if subtask1_metrics:
        logger.info(f"\n【SubTask 1: Semantic Classification (Token→Line)】")
        logger.info(f"  准确率: {subtask1_metrics['accuracy']:.4f}")
        logger.info(f"  Micro-F1: {subtask1_metrics['micro_f1']:.4f}")
        logger.info(f"  Macro-F1: {subtask1_metrics['macro_f1']:.4f}")

    logger.info(f"\n【SubTask 2: Parent Finding (GRU + Attention + Soft-mask)】")
    logger.info(f"  准确率: {subtask2_metrics['accuracy']:.4f}")

    logger.info(f"\n【SubTask 3: Relation Classification (GT父节点)】")
    logger.info(f"  F1 (macro): {subtask3_gt_metrics['f1']:.4f}")

    logger.info(f"\n【SubTask 3: Relation Classification (预测父节点)】")
    logger.info(f"  F1 (macro): {end_to_end_metrics['f1']:.4f}")
    logger.info(f"  性能下降: {subtask3_gt_metrics['f1'] - end_to_end_metrics['f1']:.4f}")

    logger.info(f"\n【End-to-End: 父节点+关系都对】")
    logger.info(f"  准确率: {end_to_end_metrics['end_to_end_accuracy']:.4f}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding=utf-8
"""
评估 SimpleParentFinder + MultiClassRelationClassifier 的End-to-End性能
"""

import logging
import os
import sys
import torch
import torch.nn as nn
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layoutlmft.models.relation_classifier import MultiClassRelationClassifier, compute_geometry_features

logger = logging.getLogger(__name__)


# Simple Parent Finder模型定义
class SimpleParentFinder(nn.Module):
    """简单的父节点查找器"""

    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size * 2 + 8, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, child_feat, parent_feats, geom_feats):
        batch_size, num_candidates, hidden_size = parent_feats.shape
        child_feat_expanded = child_feat.unsqueeze(1).expand(batch_size, num_candidates, hidden_size)
        combined = torch.cat([child_feat_expanded, parent_feats, geom_feats], dim=-1)
        scores = self.score_head(combined).squeeze(-1)
        return scores


# 关系标签映射
RELATION_LABELS = {
    "none": 0,
    "connect": 1,
    "contain": 2,
    "equality": 3,
    "meta": 0,
}
RELATION_NAMES = ["none", "connect", "contain", "equality"]


def load_models(subtask2_path, subtask3_path, device):
    """加载模型"""
    logger.info(f"加载SubTask 2模型: {subtask2_path}")
    subtask2_checkpoint = torch.load(subtask2_path, map_location=device)
    subtask2_model = SimpleParentFinder(hidden_size=768, dropout=0.1)
    subtask2_model.load_state_dict(subtask2_checkpoint["model_state_dict"])
    subtask2_model = subtask2_model.to(device)
    subtask2_model.eval()
    logger.info(f"✓ SubTask 2加载成功 (Acc: {subtask2_checkpoint.get('best_acc', 'N/A')})")

    logger.info(f"加载SubTask 3模型: {subtask3_path}")
    subtask3_checkpoint = torch.load(subtask3_path, map_location=device)
    subtask3_model = MultiClassRelationClassifier(
        input_dim=768,
        hidden_dim=256,
        num_relations=4,
        use_geometry=True,
        geometry_dim=4,
    )
    subtask3_model.load_state_dict(subtask3_checkpoint["model_state_dict"])
    subtask3_model = subtask3_model.to(device)
    subtask3_model.eval()
    logger.info(f"✓ SubTask 3加载成功 (F1: {subtask3_checkpoint.get('best_f1', 'N/A')})")

    return subtask2_model, subtask3_model


def load_validation_data(features_dir, max_chunks=None):
    """加载验证集数据"""
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


def evaluate_subtask2_only(subtask2_model, page_features, device, max_candidates=20):
    """
    单独评估SubTask 2（父节点查找）
    返回：metrics, predictions
    """
    logger.info("\n" + "="*80)
    logger.info("评估SubTask 2 (Parent Finding)")
    logger.info("="*80)

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for page_idx, page_data in enumerate(tqdm(page_features, desc="SubTask 2评估")):
            line_features = page_data["line_features"].squeeze(0).to(device)
            line_mask = page_data["line_mask"].squeeze(0)
            line_parent_ids = page_data["line_parent_ids"]
            line_bboxes = page_data["line_bboxes"]

            num_lines = len(line_parent_ids)
            page_predictions = {}

            for child_idx in range(num_lines):
                parent_idx_gt = line_parent_ids[child_idx]

                if parent_idx_gt < 0:  # ROOT
                    continue
                if child_idx >= line_mask.shape[0] or not line_mask[child_idx]:
                    continue

                # 候选父节点：child之前的max_candidates个行
                max_cands = min(max_candidates, child_idx)
                candidate_start = max(0, child_idx - max_cands)
                candidate_indices = list(range(candidate_start, child_idx))

                if len(candidate_indices) == 0:
                    continue

                # 提取特征
                child_feat = line_features[child_idx].unsqueeze(0)  # [1, H]
                parent_feats = line_features[candidate_indices].unsqueeze(0)  # [1, num_cands, H]

                # 几何特征
                child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                geom_feats = []
                for cand_idx in candidate_indices:
                    cand_bbox = torch.tensor(line_bboxes[cand_idx], dtype=torch.float32)
                    geom_feat = compute_geometry_features(cand_bbox, child_bbox)
                    geom_feats.append(geom_feat)
                geom_feats = torch.stack(geom_feats, dim=0).unsqueeze(0).to(device)  # [1, num_cands, 8]

                # 预测
                scores = subtask2_model(child_feat, parent_feats, geom_feats)  # [1, num_cands]
                pred_idx_in_candidates = torch.argmax(scores, dim=1).item()
                pred_parent_idx = candidate_indices[pred_idx_in_candidates]

                page_predictions[child_idx] = pred_parent_idx

                # 统计
                total += 1
                if pred_parent_idx == parent_idx_gt:
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
    logger.info("评估SubTask 3 (使用GT父节点)")
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
    logger.info("评估End-to-End (预测父节点 → 预测关系)")
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
    features_dir = os.getenv(
        "LAYOUTLMFT_FEATURES_DIR",
        "/mnt/e/models/train_data/layoutlmft/line_features"
    )

    subtask2_model_path = os.getenv(
        "SUBTASK2_MODEL_PATH",
        "/mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt"
    )

    subtask3_model_path = os.getenv(
        "SUBTASK3_MODEL_PATH",
        "/mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt"
    )

    max_chunks = int(os.getenv("MAX_CHUNKS", "-1"))
    if max_chunks == -1:
        max_chunks = None

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"特征目录: {features_dir}")
    logger.info(f"SubTask 2模型: {subtask2_model_path}")
    logger.info(f"SubTask 3模型: {subtask3_model_path}")

    # 加载模型
    subtask2_model, subtask3_model = load_models(
        subtask2_model_path,
        subtask3_model_path,
        device
    )

    # 加载验证数据
    page_features = load_validation_data(features_dir, max_chunks)

    # 1. 评估SubTask 2
    subtask2_metrics, subtask2_predictions = evaluate_subtask2_only(
        subtask2_model, page_features, device
    )

    # 2. 评估SubTask 3（GT父节点）
    subtask3_gt_metrics = evaluate_subtask3_with_gt_parents(
        subtask3_model, page_features, device
    )

    # 3. 评估End-to-End
    end_to_end_metrics = evaluate_end_to_end(
        subtask3_model, page_features, subtask2_predictions, device
    )

    # 总结
    logger.info("\n" + "="*80)
    logger.info("评估总结")
    logger.info("="*80)
    logger.info(f"\n【SubTask 2: Parent Finding】")
    logger.info(f"  准确率: {subtask2_metrics['accuracy']:.4f}")
    logger.info(f"\n【SubTask 3: Relation Classification (GT父节点)】")
    logger.info(f"  F1 (macro): {subtask3_gt_metrics['f1']:.4f}")
    logger.info(f"\n【SubTask 3: Relation Classification (预测父节点)】")
    logger.info(f"  F1 (macro): {end_to_end_metrics['f1']:.4f}")
    logger.info(f"  性能下降: {subtask3_gt_metrics['f1'] - end_to_end_metrics['f1']:.4f}")
    logger.info(f"\n【End-to-End: 父节点+关系都对】")
    logger.info(f"  准确率: {end_to_end_metrics['end_to_end_accuracy']:.4f}")


if __name__ == "__main__":
    main()

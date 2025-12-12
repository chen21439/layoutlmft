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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from transformers import set_seed

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from layoutlmft.models.relation_classifier import MultiClassRelationClassifier, compute_geometry_features
from layoutlmft.data.labels import NUM_LABELS, LABEL_LIST

# 从 train_parent_finder 导入模型类（避免代码重复）
STAGE_ROOT = os.path.join(PROJECT_ROOT, "examples", "stage")
sys.path.insert(0, STAGE_ROOT)
from train_parent_finder import ParentFinderGRU

logger = logging.getLogger(__name__)

# 使用统一的标签定义
SEMANTIC_LABELS = {label: i for i, label in enumerate(LABEL_LIST)}
SEMANTIC_NAMES = LABEL_LIST

# 关系标签映射（4个类别）
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
    # 加载 SubTask 2 模型 (ParentFinderGRU)
    logger.info(f"加载SubTask 2模型 (ParentFinderGRU - Full): {subtask2_path}")
    subtask2_checkpoint = torch.load(subtask2_path, map_location=device)

    subtask2_model = ParentFinderGRU(
        hidden_size=768,
        gru_hidden_size=512,
        num_classes=NUM_LABELS,  # 使用统一标签定义
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

    return subtask2_model, subtask3_model


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


def evaluate_subtask1_semantic_classification(page_features):
    """
    评估SubTask 1：语义分类（Line-level）

    说明：
    - 根据论文，使用预先提取的语义单元特征
    - SubTask 1 的 token-level 评估已在 run_hrdoc.py 中完成
    - 这里跳过重复评估，建议查看 run_hrdoc.py 的训练日志获取 F1 结果
    """
    logger.info("\n" + "="*80)
    logger.info("SubTask 1: Semantic Classification")
    logger.info("="*80)
    logger.info("说明：")
    logger.info("  1. 论文使用预先提取的语义单元（离线特征提取）")
    logger.info("  2. SubTask 1 的评估已在 run_hrdoc.py 中完成（token-level F1）")
    logger.info("  3. 当前使用 extract_line_features.py 预提取的特征")
    logger.info("  4. 建议：查看 hrdoc_train 训练日志获取 token-level F1 结果")
    logger.info("")
    logger.info("⏭  跳过 SubTask 1 重复评估，专注于 SubTask 2/3")

    return None


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
    features_dir = os.getenv(
        "LAYOUTLMFT_FEATURES_DIR",
        "/mnt/e/models/train_data/layoutlmft/line_features"
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
    logger.info("HRDoc End-to-End 评估（论文方法 - 使用预提取特征）")
    logger.info("="*80)
    logger.info(f"使用设备: {device}")
    logger.info(f"特征目录: {features_dir}")
    logger.info(f"SubTask 2模型 (Full): {subtask2_model_path}")
    logger.info(f"SubTask 3模型: {subtask3_model_path}")

    # 加载模型
    subtask2_model, subtask3_model = load_models(
        subtask2_model_path,
        subtask3_model_path,
        device
    )

    # 加载预处理特征
    page_features = load_validation_data(features_dir, max_chunks)

    # 1. 说明SubTask 1评估
    subtask1_metrics = evaluate_subtask1_semantic_classification(page_features)

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

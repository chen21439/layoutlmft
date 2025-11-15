#!/usr/bin/env python
# coding=utf-8
"""
End-to-End 评估脚本
串行执行 SubTask 2 (Parent Finding) + SubTask 3 (Relation Classification)
评估真实的综合性能
"""

import logging
import os
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from collections import defaultdict

# 导入模型
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layoutlmft.models.relation_classifier import (
    SimpleRelationClassifier,
    MultiClassRelationClassifier,
    compute_geometry_features,
)

logger = logging.getLogger(__name__)


# SimpleParentFinder模型定义（从train_parent_finder_simple.py复制）
class SimpleParentFinder(torch.nn.Module):
    """
    简单的父节点查找器
    对每个 child，给候选 parent 打分，选择分数最高的
    """

    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()

        # 特征融合层
        # child_feat + parent_feat + geom_feat -> score
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2 + 8, hidden_size),  # 8 是几何特征维度
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, 1)  # 输出一个分数
        )

    def forward(self, child_feat, parent_feats, geom_feats):
        """
        Args:
            child_feat: [batch_size, hidden_size]
            parent_feats: [batch_size, num_candidates, hidden_size]
            geom_feats: [batch_size, num_candidates, 8]

        Returns:
            scores: [batch_size, num_candidates]
        """
        batch_size, num_candidates, hidden_size = parent_feats.shape

        # 扩展 child_feat 到每个候选
        child_feat_expanded = child_feat.unsqueeze(1).expand(batch_size, num_candidates, hidden_size)

        # 拼接特征
        combined = torch.cat([child_feat_expanded, parent_feats, geom_feats], dim=-1)

        # 计算分数
        scores = self.score_head(combined).squeeze(-1)  # [B, num_candidates]

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
    """
    加载训练好的SubTask 2和SubTask 3模型

    Args:
        subtask2_path: SubTask 2模型路径（parent finder）
        subtask3_path: SubTask 3模型路径（relation classifier）
        device: 设备

    Returns:
        subtask2_model, subtask3_model
    """
    logger.info(f"加载SubTask 2模型: {subtask2_path}")
    subtask2_checkpoint = torch.load(subtask2_path, map_location=device)
    subtask2_model = SimpleRelationClassifier(
        input_dim=768,
        hidden_dim=256,
        num_relations=2,
        use_geometry=True,
        geometry_dim=4,
    )
    subtask2_model.load_state_dict(subtask2_checkpoint["model_state_dict"])
    subtask2_model = subtask2_model.to(device)
    subtask2_model.eval()
    logger.info(f"✓ SubTask 2模型加载成功 (F1: {subtask2_checkpoint.get('best_f1', 'N/A')})")

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
    logger.info(f"✓ SubTask 3模型加载成功 (F1: {subtask3_checkpoint.get('best_f1', 'N/A')})")

    return subtask2_model, subtask3_model


def load_validation_data(features_dir, max_chunks=None):
    """
    加载验证集数据

    Args:
        features_dir: 特征文件目录
        max_chunks: 最多加载多少个chunk

    Returns:
        page_features: 页面特征列表
    """
    import glob

    pattern = os.path.join(features_dir, "validation_line_features_chunk_*.pkl")
    chunk_files = sorted(glob.glob(pattern))

    if max_chunks is not None:
        chunk_files = chunk_files[:max_chunks]
        logger.info(f"限制加载前 {max_chunks} 个validation chunk")

    if len(chunk_files) == 0:
        raise ValueError(f"没有找到validation特征文件: {pattern}")

    logger.info(f"找到 {len(chunk_files)} 个validation chunk文件，开始加载...")
    page_features = []
    for chunk_file in chunk_files:
        logger.info(f"  加载 {os.path.basename(chunk_file)}...")
        with open(chunk_file, "rb") as f:
            chunk_data = pickle.load(f)
        page_features.extend(chunk_data)
        logger.info(f"    已加载 {len(chunk_data)} 页，累计 {len(page_features)} 页")

    logger.info(f"总共加载了 {len(page_features)} 页的验证集特征")
    return page_features


def evaluate_subtask2_only(subtask2_model, page_features, device):
    """
    单独评估SubTask 2（父节点查找）的性能

    Returns:
        metrics: {accuracy, precision, recall, f1}
        predictions: 每个样本的预测结果
    """
    logger.info("\n" + "="*80)
    logger.info("评估SubTask 2 (Parent Finding) - 单独性能")
    logger.info("="*80)

    all_labels = []
    all_preds = []
    predictions = []  # 保存预测结果供SubTask 3使用

    with torch.no_grad():
        for page_idx, page_data in enumerate(tqdm(page_features, desc="SubTask 2评估")):
            line_features = page_data["line_features"].squeeze(0).to(device)
            line_mask = page_data["line_mask"].squeeze(0)
            line_parent_ids = page_data["line_parent_ids"]
            line_bboxes = page_data["line_bboxes"]

            num_lines = len(line_parent_ids)
            page_predictions = {}

            # 对每个child，预测其父节点
            for child_idx in range(num_lines):
                parent_idx_gt = line_parent_ids[child_idx]

                # 跳过ROOT（没有父节点）
                if parent_idx_gt < 0:
                    continue

                # 检查有效性
                if child_idx >= line_mask.shape[0] or not line_mask[child_idx]:
                    continue

                # 对所有可能的父节点（在child之前的行）进行预测
                best_score = -float('inf')
                pred_parent_idx = -1

                for candidate_parent_idx in range(child_idx):  # 只考虑在child之前的行
                    if candidate_parent_idx >= line_mask.shape[0] or not line_mask[candidate_parent_idx]:
                        continue
                    if candidate_parent_idx >= len(line_bboxes) or child_idx >= len(line_bboxes):
                        continue

                    # 提取特征
                    parent_feat = line_features[candidate_parent_idx].unsqueeze(0)
                    child_feat = line_features[child_idx].unsqueeze(0)

                    parent_bbox = torch.tensor(line_bboxes[candidate_parent_idx], dtype=torch.float32)
                    child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                    geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

                    # 预测
                    logits = subtask2_model(parent_feat, child_feat, geom_feat)
                    score = logits[0, 1].item()  # 是父子关系的得分

                    if score > best_score:
                        best_score = score
                        pred_parent_idx = candidate_parent_idx

                # 记录预测结果
                page_predictions[child_idx] = pred_parent_idx

                # 评估：预测的父节点是否正确
                label = 1 if pred_parent_idx == parent_idx_gt else 0
                all_labels.append(1)  # ground truth是有父节点
                all_preds.append(label)

            predictions.append({
                "page_idx": page_idx,
                "predictions": page_predictions,
            })

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    logger.info(f"\nSubTask 2结果:")
    logger.info(f"  准确率: {accuracy:.4f}")
    logger.info(f"  精确率: {precision:.4f}")
    logger.info(f"  召回率: {recall:.4f}")
    logger.info(f"  F1分数: {f1:.4f}")
    logger.info(f"  总样本数: {len(all_labels)}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }, predictions


def evaluate_subtask3_with_gt_parents(subtask3_model, page_features, device):
    """
    评估SubTask 3（使用Ground Truth父节点）
    这是SubTask 3训练时的设置
    """
    logger.info("\n" + "="*80)
    logger.info("评估SubTask 3 (Relation Classification) - 使用GT父节点")
    logger.info("="*80)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for page_data in tqdm(page_features, desc="SubTask 3评估(GT父节点)"):
            line_features = page_data["line_features"].squeeze(0).to(device)
            line_mask = page_data["line_mask"].squeeze(0)
            line_parent_ids = page_data["line_parent_ids"]
            line_relations = page_data["line_relations"]
            line_bboxes = page_data["line_bboxes"]

            num_lines = len(line_parent_ids)

            for child_idx in range(num_lines):
                parent_idx = line_parent_ids[child_idx]  # Ground Truth父节点
                relation = line_relations[child_idx]

                # 跳过无效样本
                if parent_idx < 0 or parent_idx >= num_lines:
                    continue
                if relation not in RELATION_LABELS:
                    continue
                if relation == "none" or relation == "meta":
                    continue

                # 检查有效性
                max_idx = line_mask.shape[0]
                if parent_idx >= max_idx or child_idx >= max_idx:
                    continue
                if not line_mask[parent_idx] or not line_mask[child_idx]:
                    continue
                if parent_idx >= len(line_bboxes) or child_idx >= len(line_bboxes):
                    continue

                # 提取特征
                parent_feat = line_features[parent_idx].unsqueeze(0)
                child_feat = line_features[child_idx].unsqueeze(0)

                parent_bbox = torch.tensor(line_bboxes[parent_idx], dtype=torch.float32)
                child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

                # 预测关系
                logits = subtask3_model(parent_feat, child_feat, geom_feat)
                pred_label = torch.argmax(logits, dim=1).item()

                gt_label = RELATION_LABELS[relation]

                all_labels.append(gt_label)
                all_preds.append(pred_label)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=RELATION_NAMES,
        zero_division=0
    )

    logger.info(f"\nSubTask 3结果 (使用GT父节点):")
    logger.info(f"  准确率: {accuracy:.4f}")
    logger.info(f"  精确率 (macro): {precision:.4f}")
    logger.info(f"  召回率 (macro): {recall:.4f}")
    logger.info(f"  F1分数 (macro): {f1:.4f}")
    logger.info(f"  总样本数: {len(all_labels)}")
    logger.info(f"\n混淆矩阵:\n{cm}")
    logger.info(f"\n分类报告:\n{report}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_end_to_end(subtask3_model, page_features, subtask2_predictions, device):
    """
    End-to-End评估：使用SubTask 2预测的父节点进行SubTask 3预测
    """
    logger.info("\n" + "="*80)
    logger.info("评估End-to-End性能 (SubTask 2预测 → SubTask 3预测)")
    logger.info("="*80)

    all_labels = []
    all_preds = []

    # 统计
    parent_correct = 0
    parent_total = 0
    relation_correct_with_gt_parent = 0
    relation_correct_with_pred_parent = 0
    both_correct = 0
    total_samples = 0

    with torch.no_grad():
        for pred_info in tqdm(subtask2_predictions, desc="End-to-End评估"):
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

                # 跳过无效样本
                if gt_parent_idx < 0 or gt_parent_idx >= num_lines:
                    continue
                if gt_relation not in RELATION_LABELS:
                    continue
                if gt_relation == "none" or gt_relation == "meta":
                    continue

                # 检查有效性
                max_idx = line_mask.shape[0]
                if child_idx >= max_idx or not line_mask[child_idx]:
                    continue
                if child_idx >= len(line_bboxes):
                    continue

                total_samples += 1

                # 检查父节点预测是否正确
                parent_is_correct = (pred_parent_idx == gt_parent_idx)
                if parent_is_correct:
                    parent_correct += 1
                parent_total += 1

                # 使用预测的父节点进行关系分类
                if pred_parent_idx >= 0 and pred_parent_idx < num_lines:
                    if pred_parent_idx >= max_idx or not line_mask[pred_parent_idx]:
                        continue
                    if pred_parent_idx >= len(line_bboxes):
                        continue

                    # 提取特征（使用预测的父节点）
                    parent_feat = line_features[pred_parent_idx].unsqueeze(0)
                    child_feat = line_features[child_idx].unsqueeze(0)

                    parent_bbox = torch.tensor(line_bboxes[pred_parent_idx], dtype=torch.float32)
                    child_bbox = torch.tensor(line_bboxes[child_idx], dtype=torch.float32)
                    geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

                    # 预测关系
                    logits = subtask3_model(parent_feat, child_feat, geom_feat)
                    pred_relation_label = torch.argmax(logits, dim=1).item()
                else:
                    # 如果没有预测父节点，默认为none
                    pred_relation_label = 0

                gt_relation_label = RELATION_LABELS[gt_relation]

                # 统计
                relation_is_correct = (pred_relation_label == gt_relation_label)
                if relation_is_correct:
                    relation_correct_with_pred_parent += 1

                if parent_is_correct and relation_is_correct:
                    both_correct += 1

                all_labels.append(gt_relation_label)
                all_preds.append(pred_relation_label)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=RELATION_NAMES,
        zero_division=0
    )

    # End-to-End F1（父节点和关系都正确）
    end_to_end_f1 = both_correct / total_samples if total_samples > 0 else 0
    parent_accuracy = parent_correct / parent_total if parent_total > 0 else 0

    logger.info(f"\nEnd-to-End结果:")
    logger.info(f"  总样本数: {total_samples}")
    logger.info(f"  父节点准确率: {parent_accuracy:.4f} ({parent_correct}/{parent_total})")
    logger.info(f"  关系分类准确率 (使用预测父节点): {accuracy:.4f}")
    logger.info(f"  关系分类F1 (macro): {f1:.4f}")
    logger.info(f"  End-to-End准确率 (父节点+关系都正确): {end_to_end_f1:.4f} ({both_correct}/{total_samples})")
    logger.info(f"\n混淆矩阵:\n{cm}")
    logger.info(f"\n分类报告:\n{report}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "end_to_end_accuracy": end_to_end_f1,
        "parent_accuracy": parent_accuracy,
    }


def main():
    # 配置
    features_dir = os.getenv(
        "LAYOUTLMFT_FEATURES_DIR",
        "/mnt/e/models/train_data/layoutlmft/line_features"
    )

    # SubTask 2模型路径（父节点查找）
    # 选项1: relation_classifier (二分类方式，已训练 F1=0.9562)
    # 选项2: parent_finder (GRU-based论文方法，需要先训练 train_parent_finder.py)
    subtask2_model_path = os.getenv(
        "SUBTASK2_MODEL_PATH",
        "/mnt/e/models/train_data/layoutlmft/relation_classifier/best_model.pt"
    )

    # SubTask 3模型路径（关系分类，已训练 F1=0.9081）
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

    # 1. 评估SubTask 2单独性能
    subtask2_metrics, subtask2_predictions = evaluate_subtask2_only(
        subtask2_model, page_features, device
    )

    # 2. 评估SubTask 3（使用GT父节点）
    subtask3_gt_metrics = evaluate_subtask3_with_gt_parents(
        subtask3_model, page_features, device
    )

    # 3. 评估End-to-End（使用预测父节点）
    end_to_end_metrics = evaluate_end_to_end(
        subtask3_model, page_features, subtask2_predictions, device
    )

    # 总结
    logger.info("\n" + "="*80)
    logger.info("评估总结")
    logger.info("="*80)
    logger.info(f"\n【SubTask 2: Parent Finding】")
    logger.info(f"  F1: {subtask2_metrics['f1']:.4f}")
    logger.info(f"\n【SubTask 3: Relation Classification (使用GT父节点)】")
    logger.info(f"  F1 (macro): {subtask3_gt_metrics['f1']:.4f}")
    logger.info(f"\n【SubTask 3: Relation Classification (使用预测父节点)】")
    logger.info(f"  F1 (macro): {end_to_end_metrics['f1']:.4f}")
    logger.info(f"  性能下降: {subtask3_gt_metrics['f1'] - end_to_end_metrics['f1']:.4f}")
    logger.info(f"\n【End-to-End: 父节点 + 关系都正确】")
    logger.info(f"  准确率: {end_to_end_metrics['end_to_end_accuracy']:.4f}")
    logger.info(f"\n【误差累积分析】")
    logger.info(f"  父节点准确率: {end_to_end_metrics['parent_accuracy']:.4f}")
    logger.info(f"  理论最大End-to-End: {subtask2_metrics['f1'] * subtask3_gt_metrics['f1']:.4f}")
    logger.info(f"  实际End-to-End: {end_to_end_metrics['end_to_end_accuracy']:.4f}")


if __name__ == "__main__":
    main()

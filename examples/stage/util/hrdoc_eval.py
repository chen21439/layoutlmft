#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 评估工具

提供两种评估模式：
1. Stage1 评估：只评估语义分类（line-level F1）
2. 端到端评估：评估分类 + Parent准确率 + TEDS

使用 HRDoc 官方评估方法
"""

import os
import sys
import json
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

# 添加 HRDoc 工具路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HRDOC_UTILS_PATH = os.path.join(PROJECT_ROOT, "HRDoc", "utils")
sys.path.insert(0, HRDOC_UTILS_PATH)

logger = logging.getLogger(__name__)

# 类别映射（与 HRDoc 一致）
CLASS2ID = {
    "title": 0, "author": 1, "mail": 2, "affili": 3, "section": 4,
    "fstline": 5, "paraline": 6, "table": 7, "figure": 8, "caption": 9,
    "equation": 10, "footer": 11, "header": 12, "footnote": 13,
}
ID2CLASS = {v: k for k, v in CLASS2ID.items()}

# 关系映射（论文对齐：只有3类，不含none）
# connect=0, contain=1, equality=2
RELATION2ID = {"connect": 0, "contain": 1, "equality": 2}
ID2RELATION = {v: k for k, v in RELATION2ID.items()}


def compute_geometry_features(parent_bbox, child_bbox):
    """计算几何特征"""
    import torch

    # 确保是 tensor
    if not isinstance(parent_bbox, torch.Tensor):
        parent_bbox = torch.tensor(parent_bbox, dtype=torch.float)
    if not isinstance(child_bbox, torch.Tensor):
        child_bbox = torch.tensor(child_bbox, dtype=torch.float)

    # 计算相对位置特征
    px1, py1, px2, py2 = parent_bbox
    cx1, cy1, cx2, cy2 = child_bbox

    # 宽度和高度
    pw = px2 - px1 + 1
    ph = py2 - py1 + 1
    cw = cx2 - cx1 + 1
    ch = cy2 - cy1 + 1

    # 中心点
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    ccx, ccy = (cx1 + cx2) / 2, (cy1 + cy2) / 2

    # 相对位置
    dx = (ccx - pcx) / (pw + 1e-6)
    dy = (ccy - pcy) / (ph + 1e-6)

    # 尺寸比例
    wr = cw / (pw + 1e-6)
    hr = ch / (ph + 1e-6)

    # 重叠比例
    ox = max(0, min(px2, cx2) - max(px1, cx1)) / (cw + 1e-6)
    oy = max(0, min(py2, cy2) - max(py1, cy1)) / (ch + 1e-6)

    return torch.tensor([dx, dy, wr, hr, ox, oy], dtype=torch.float)


def extract_gt_from_batch(batch, id2label=None) -> List[Dict]:
    """
    从 batch 中提取 ground truth，转换为 HRDoc JSON 格式

    Returns:
        每个文档的 GT 列表，每个元素是 [{"class": ..., "parent_id": ..., "relation": ...}, ...]
    """
    if id2label is None:
        id2label = ID2CLASS

    batch_size = batch["input_ids"].shape[0]
    results = []

    for b in range(batch_size):
        labels = batch["labels"][b]  # [seq_len]
        line_ids = batch.get("line_ids")
        line_parent_ids = batch.get("line_parent_ids")
        line_relations = batch.get("line_relations")

        if line_ids is None:
            results.append([])
            continue

        line_ids_b = line_ids[b]  # [seq_len]

        # 获取每行的标签（取该行第一个 token 的标签）
        line_labels = {}
        for token_idx, (label, line_id) in enumerate(zip(labels.cpu().tolist(), line_ids_b.cpu().tolist())):
            if line_id >= 0 and line_id not in line_labels and label >= 0:
                line_labels[line_id] = label

        num_lines = len(line_labels)
        if num_lines == 0:
            results.append([])
            continue

        # 获取 parent_ids 和 relations
        parent_ids = [-1] * num_lines
        relations = ["none"] * num_lines

        if line_parent_ids is not None:
            parent_ids_b = line_parent_ids[b].cpu().tolist()
            for i in range(min(num_lines, len(parent_ids_b))):
                parent_ids[i] = parent_ids_b[i]

        if line_relations is not None:
            relations_b = line_relations[b].cpu().tolist()
            for i in range(min(num_lines, len(relations_b))):
                relations[i] = ID2RELATION.get(relations_b[i], "none")

        # 构建 GT
        doc_gt = []
        for line_idx in sorted(line_labels.keys()):
            class_id = line_labels[line_idx]
            class_name = id2label.get(class_id, f"class_{class_id}")

            doc_gt.append({
                "class": class_name,
                "text": f"line_{line_idx}",
                "parent_id": parent_ids[line_idx] if line_idx < len(parent_ids) else -1,
                "relation": relations[line_idx] if line_idx < len(relations) else "none",
            })

        results.append(doc_gt)

    return results


def evaluate_stage1(
    model,
    eval_loader: DataLoader,
    device,
    id2label: Dict[int, str] = None,
) -> Dict[str, float]:
    """
    Stage1 评估：只评估语义分类

    使用 line-level 的 Macro F1 和 Micro F1（与 HRDoc 论文一致）

    Args:
        model: JointModel 或 LayoutXLM 模型
        eval_loader: 评估数据加载器
        device: 设备
        id2label: 类别映射

    Returns:
        评估指标字典
    """
    if id2label is None:
        id2label = ID2CLASS

    model.eval()

    all_gt_classes = []
    all_pred_classes = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Stage1 Eval"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 获取 Stage1 输出
            if hasattr(model, 'stage1'):
                # JointModel
                outputs = model.stage1(
                    input_ids=batch["input_ids"],
                    bbox=batch["bbox"],
                    attention_mask=batch["attention_mask"],
                    image=batch.get("image"),
                )
                logits = outputs.logits
            else:
                # 直接是 LayoutXLM
                outputs = model(
                    input_ids=batch["input_ids"],
                    bbox=batch["bbox"],
                    attention_mask=batch["attention_mask"],
                    image=batch.get("image"),
                )
                logits = outputs.logits

            batch_size = batch["input_ids"].shape[0]

            for b in range(batch_size):
                labels = batch["labels"][b]
                line_ids = batch.get("line_ids")

                if line_ids is None:
                    continue

                line_ids_b = line_ids[b]
                sample_logits = logits[b]

                # 提取每行的 GT 和预测
                line_gt = {}
                line_pred = {}

                for token_idx, (label, line_id) in enumerate(zip(labels.cpu().tolist(), line_ids_b.cpu().tolist())):
                    if line_id >= 0 and label >= 0:
                        if line_id not in line_gt:
                            line_gt[line_id] = label
                        if line_id not in line_pred:
                            line_pred[line_id] = sample_logits[token_idx].argmax().item()

                # 收集结果
                for line_id in sorted(line_gt.keys()):
                    if line_id in line_pred:
                        all_gt_classes.append(line_gt[line_id])
                        all_pred_classes.append(line_pred[line_id])

    # 计算指标
    results = {}

    if all_gt_classes:
        results["line_macro_f1"] = f1_score(all_gt_classes, all_pred_classes, average="macro")
        results["line_micro_f1"] = f1_score(all_gt_classes, all_pred_classes, average="micro")
        results["line_accuracy"] = accuracy_score(all_gt_classes, all_pred_classes)
        results["num_lines"] = len(all_gt_classes)

        # 每类 F1
        per_class_f1 = f1_score(all_gt_classes, all_pred_classes, average=None, labels=list(range(14)))
        for i, f1 in enumerate(per_class_f1):
            class_name = id2label.get(i, f"class_{i}")
            results[f"f1_{class_name}"] = f1

    return results


def evaluate_e2e(
    model,
    eval_loader: DataLoader,
    device,
    args=None,
    global_step: int = 0,
    id2label: Dict[int, str] = None,
) -> Dict[str, float]:
    """
    端到端评估：分类 + Parent准确率 + TEDS

    Args:
        model: JointModel
        eval_loader: 评估数据加载器
        device: 设备
        args: 训练参数（用于判断是否计算 TEDS）
        global_step: 当前训练步数
        id2label: 类别映射

    Returns:
        评估指标字典
    """
    # 根据 args.quick 决定是否计算 TEDS
    compute_teds = True
    if args is not None and hasattr(args, 'quick') and args.quick:
        compute_teds = False

    if id2label is None:
        id2label = ID2CLASS

    model.eval()

    # 收集所有预测和 GT
    all_gt_classes = []
    all_pred_classes = []
    all_gt_parents = []
    all_pred_parents = []
    all_gt_relations = []
    all_pred_relations = []

    # 用于 TEDS
    all_gt_docs = []
    all_pred_docs = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="E2E Eval"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Stage 1: 分类
            stage1_outputs = model.stage1(
                input_ids=batch["input_ids"],
                bbox=batch["bbox"],
                attention_mask=batch["attention_mask"],
                image=batch.get("image"),
                output_hidden_states=True,
            )

            logits = stage1_outputs.logits
            hidden_states = stage1_outputs.hidden_states[-1]

            batch_size = batch["input_ids"].shape[0]

            # 提取 GT
            gt_docs = extract_gt_from_batch(batch, id2label)

            for b in range(batch_size):
                gt_doc = gt_docs[b]
                if not gt_doc:
                    continue

                labels = batch["labels"][b]
                line_ids = batch.get("line_ids")

                if line_ids is None:
                    continue

                line_ids_b = line_ids[b]
                sample_logits = logits[b]

                # 提取每行的预测类别
                line_preds = {}
                for token_idx, line_id in enumerate(line_ids_b.cpu().tolist()):
                    if line_id >= 0 and line_id not in line_preds:
                        line_preds[line_id] = sample_logits[token_idx].argmax().item()

                num_lines = len(line_preds)
                if num_lines == 0:
                    continue

                # Stage 2: 提取特征
                text_seq_len = batch["input_ids"].shape[1]
                text_hidden = hidden_states[b:b+1, :text_seq_len, :]

                line_features, line_mask = model.feature_extractor.extract_line_features(
                    text_hidden, line_ids_b.unsqueeze(0), pooling="mean"
                )

                line_features = line_features[0]  # [max_lines, H]
                line_mask = line_mask[0]
                actual_num_lines = int(line_mask.sum().item())

                # Stage 3: 预测父节点
                pred_parents = [-1] * actual_num_lines
                gru_hidden = None  # 用于 Stage 4

                if hasattr(model, 'use_gru') and model.use_gru:
                    # 论文对齐：获取 GRU 隐状态用于 Stage 4
                    parent_logits, gru_hidden = model.stage3(
                        line_features.unsqueeze(0),
                        line_mask.unsqueeze(0),
                        return_gru_hidden=True
                    )
                    # gru_hidden: [1, L+1, gru_hidden_size]，包括 ROOT
                    gru_hidden = gru_hidden[0]  # [L+1, gru_hidden_size]

                    for child_idx in range(actual_num_lines):
                        child_logits = parent_logits[0, child_idx + 1, :child_idx + 2]
                        pred_parent_idx = child_logits.argmax().item()
                        pred_parents[child_idx] = pred_parent_idx - 1
                else:
                    # SimpleParentFinder
                    for child_idx in range(1, actual_num_lines):
                        parent_candidates = line_features[:child_idx]
                        child_feat = line_features[child_idx]
                        scores = model.stage3(parent_candidates, child_feat)
                        pred_parents[child_idx] = scores.argmax().item()

                # Stage 4: 预测关系（论文对齐：使用 GRU 隐状态，不使用几何特征）
                pred_relations = [0] * actual_num_lines

                for child_idx in range(actual_num_lines):
                    parent_idx = pred_parents[child_idx]
                    if parent_idx < 0 or parent_idx >= actual_num_lines:
                        continue

                    if gru_hidden is not None:
                        # 论文对齐：使用 GRU 隐状态
                        # gru_hidden 包含 ROOT，所以需要 +1 偏移
                        parent_gru_idx = parent_idx + 1
                        child_gru_idx = child_idx + 1
                        parent_feat = gru_hidden[parent_gru_idx]
                        child_feat = gru_hidden[child_gru_idx]
                    else:
                        # 非 GRU 模式：使用 encoder 输出
                        parent_feat = line_features[parent_idx]
                        child_feat = line_features[child_idx]

                    # 论文对齐：不使用几何特征
                    rel_logits = model.stage4(
                        parent_feat.unsqueeze(0),
                        child_feat.unsqueeze(0),
                    )
                    pred_relations[child_idx] = rel_logits.argmax(dim=1).item()

                # 收集分类结果
                for line_idx in range(min(actual_num_lines, len(gt_doc))):
                    gt_class = CLASS2ID.get(gt_doc[line_idx]["class"], 0)
                    pred_class = line_preds.get(line_idx, 0)

                    all_gt_classes.append(gt_class)
                    all_pred_classes.append(pred_class)

                    all_gt_parents.append(gt_doc[line_idx]["parent_id"])
                    all_pred_parents.append(pred_parents[line_idx] if line_idx < len(pred_parents) else -1)

                    gt_rel = RELATION2ID.get(gt_doc[line_idx].get("relation", "none"), 0)
                    all_gt_relations.append(gt_rel)
                    all_pred_relations.append(pred_relations[line_idx] if line_idx < len(pred_relations) else 0)

                # 保存用于 TEDS
                if compute_teds:
                    pred_doc = []
                    for line_idx in range(actual_num_lines):
                        class_id = line_preds.get(line_idx, 0)
                        class_name = id2label.get(class_id, f"class_{class_id}")
                        pred_doc.append({
                            "class": class_name,
                            "text": f"line_{line_idx}",
                            "parent_id": pred_parents[line_idx],
                            "relation": ID2RELATION.get(pred_relations[line_idx], "none"),
                        })

                    all_gt_docs.append(gt_doc)
                    all_pred_docs.append(pred_doc)

    # 计算指标
    results = {}

    if all_gt_classes:
        # 分类指标
        results["line_macro_f1"] = f1_score(all_gt_classes, all_pred_classes, average="macro")
        results["line_micro_f1"] = f1_score(all_gt_classes, all_pred_classes, average="micro")
        results["line_accuracy"] = accuracy_score(all_gt_classes, all_pred_classes)

        # Parent 准确率
        parent_correct = sum(1 for g, p in zip(all_gt_parents, all_pred_parents) if g == p)
        results["parent_accuracy"] = parent_correct / len(all_gt_parents)

        # Relation 准确率（只计算有父节点的）
        rel_pairs = [(g, p) for g, p, gp in zip(all_gt_relations, all_pred_relations, all_gt_parents) if gp >= 0]
        if rel_pairs:
            rel_correct = sum(1 for g, p in rel_pairs if g == p)
            results["relation_accuracy"] = rel_correct / len(rel_pairs)

        results["num_lines"] = len(all_gt_classes)

    # TEDS
    if compute_teds and all_gt_docs:
        try:
            teds_score = compute_teds_score(all_gt_docs, all_pred_docs)
            if teds_score is not None:
                results["macro_teds"] = teds_score
        except Exception as e:
            logger.warning(f"TEDS computation failed: {e}")

    return results


def compute_teds_score(gt_docs: List[List[Dict]], pred_docs: List[List[Dict]]) -> Optional[float]:
    """
    计算 TEDS 分数

    Args:
        gt_docs: GT 文档列表
        pred_docs: 预测文档列表

    Returns:
        Macro TEDS 分数
    """
    try:
        from doc_utils import generate_doc_tree_from_log_line_level, tree_edit_distance
    except ImportError:
        logger.warning("Cannot import doc_utils for TEDS computation")
        return None

    teds_list = []

    for gt_doc, pred_doc in zip(gt_docs, pred_docs):
        if len(gt_doc) != len(pred_doc):
            continue

        try:
            gt_texts = [f"{t['class']}:{t.get('text', '')}" for t in gt_doc]
            gt_parents = [t["parent_id"] for t in gt_doc]
            gt_relations = [t.get("relation", "none") for t in gt_doc]

            pred_texts = [f"{t['class']}:{t.get('text', '')}" for t in pred_doc]
            pred_parents = [t["parent_id"] for t in pred_doc]
            pred_relations = [t.get("relation", "none") for t in pred_doc]

            gt_tree = generate_doc_tree_from_log_line_level(gt_texts, gt_parents, gt_relations)
            pred_tree = generate_doc_tree_from_log_line_level(pred_texts, pred_parents, pred_relations)

            _, teds = tree_edit_distance(pred_tree, gt_tree)
            teds_list.append(teds)
        except Exception as e:
            continue

    if teds_list:
        return sum(teds_list) / len(teds_list)
    return None


def log_eval_results(results: Dict[str, float], prefix: str = ""):
    """打印评估结果"""
    logger.info("=" * 60)
    logger.info(f"{prefix} Evaluation Results")
    logger.info("-" * 60)

    # 主要指标
    if "line_macro_f1" in results:
        logger.info(f"Line-level Macro F1:  {results['line_macro_f1']:.4f}")
    if "line_micro_f1" in results:
        logger.info(f"Line-level Micro F1:  {results['line_micro_f1']:.4f}")
    if "line_accuracy" in results:
        logger.info(f"Line-level Accuracy:  {results['line_accuracy']:.4f}")

    if "parent_accuracy" in results:
        logger.info(f"Parent Accuracy:      {results['parent_accuracy']:.4f}")
    if "relation_accuracy" in results:
        logger.info(f"Relation Accuracy:    {results['relation_accuracy']:.4f}")
    if "macro_teds" in results:
        logger.info(f"Macro TEDS:           {results['macro_teds']:.4f}")

    if "num_lines" in results:
        logger.info(f"Total Lines:          {results['num_lines']}")

    logger.info("=" * 60)

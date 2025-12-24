#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 评估工具

提供两种评估模式：
1. Stage1 评估：只评估语义分类（line-level F1）
2. 端到端评估：评估分类 + Parent准确率 + TEDS

使用 HRDoc 官方评估方法

重构说明：
- 端到端推理逻辑已抽离到 e2e_inference.py
- 本模块只负责评估指标计算
- 行级评估使用 metrics.line_eval 模块（Single Source of Truth）
"""

import os
import sys
import json
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===========================================================================
# 导入行级评估模块（Single Source of Truth）
# ===========================================================================
# 添加 examples 目录到路径（用于导入统一的 metrics 模块）
STAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_ROOT = os.path.dirname(STAGE_ROOT)  # examples/ 目录
if EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT)

from metrics.line_eval import (
    aggregate_token_to_line_predictions,
    extract_line_labels_from_tokens,
    compute_line_level_metrics,
    LineMetricsResult,
)

# 兼容旧代码
aggregate_token_to_line_labels = extract_line_labels_from_tokens

# 导入共享的推理模块
from util.e2e_inference import run_e2e_inference_single, E2EInferenceOutput

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

    文档级别 batch 结构：
    - num_docs: 文档数量
    - chunks_per_doc: 每个文档的 chunk 数量
    - input_ids: [total_chunks, seq_len]
    - line_parent_ids: [num_docs, max_lines]  # 按文档组织

    Returns:
        每个文档的 GT 列表，每个元素是 [{"class": ..., "parent_id": ..., "relation": ...}, ...]
    """
    if id2label is None:
        id2label = ID2CLASS

    # 文档级别：使用 num_docs；chunk 级别：使用 input_ids.shape[0]
    num_docs = batch.get("num_docs", batch["input_ids"].shape[0])
    chunks_per_doc = batch.get("chunks_per_doc", [1] * num_docs)

    line_parent_ids = batch.get("line_parent_ids")
    line_relations = batch.get("line_relations")

    results = []
    chunk_idx = 0

    for doc_idx in range(num_docs):
        num_chunks = chunks_per_doc[doc_idx]

        # 收集该文档所有 chunks 的 labels 和 line_ids
        all_labels = []
        all_line_ids = []
        for c in range(num_chunks):
            labels = batch["labels"][chunk_idx + c].cpu().tolist()
            line_ids = batch["line_ids"][chunk_idx + c].cpu().tolist()
            all_labels.extend(labels)
            all_line_ids.extend(line_ids)

        chunk_idx += num_chunks

        # 使用统一的聚合函数获取每行的 GT 标签
        line_labels = aggregate_token_to_line_labels(all_labels, all_line_ids)

        num_lines = len(line_labels)
        if num_lines == 0:
            results.append([])
            continue

        # 获取 parent_ids 和 relations（按文档索引）
        parent_ids = [-1] * num_lines
        relations = ["none"] * num_lines

        if line_parent_ids is not None and doc_idx < line_parent_ids.shape[0]:
            parent_ids_b = line_parent_ids[doc_idx].cpu().tolist()
            for i in range(min(num_lines, len(parent_ids_b))):
                parent_ids[i] = parent_ids_b[i]

        if line_relations is not None and doc_idx < line_relations.shape[0]:
            relations_b = line_relations[doc_idx].cpu().tolist()
            for i in range(min(num_lines, len(relations_b))):
                relations[i] = ID2RELATION.get(relations_b[i], "none")

        # 构建 GT（保存实际的 line_id 以便后续对齐）
        doc_gt = []
        for line_id in sorted(line_labels.keys()):
            class_id = line_labels[line_id]
            class_name = id2label.get(class_id, f"class_{class_id}")

            doc_gt.append({
                "line_id": line_id,  # 保存实际的 line_id
                "class": class_name,
                "text": f"line_{line_id}",
                "parent_id": parent_ids[line_id] if line_id < len(parent_ids) else -1,
                "relation": relations[line_id] if line_id < len(relations) else "none",
            })

        results.append(doc_gt)

    return results


def evaluate_stage1(
    model,
    eval_loader: DataLoader,
    device,
    id2label: Dict[int, str] = None,
    class_names: List[str] = None,
    log_details: bool = True,
) -> Dict[str, float]:
    """
    Stage1 评估：只评估语义分类（LINE 级别）

    使用 line-level 的 Macro F1 和 Micro F1（与 HRDoc 论文一致）
    评估逻辑使用 metrics.line_eval 模块（Single Source of Truth）

    Args:
        model: JointModel 或 LayoutXLM 模型
        eval_loader: 评估数据加载器
        device: 设备
        id2label: 类别 ID -> 名称映射
        class_names: 类别名称列表（用于日志）
        log_details: 是否打印详细的每类指标

    Returns:
        评估指标字典
    """
    if id2label is None:
        id2label = ID2CLASS
    if class_names is None:
        class_names = [id2label.get(i, f"class_{i}") for i in range(14)]

    model.eval()

    # 收集所有行的预测和标签
    all_line_predictions = []
    all_line_labels = []

    # [诊断] 累积统计
    diag_total_tokens = 0
    diag_voted_tokens = 0
    diag_label_minus100_voted = 0  # label=-100 但参与了投票的 token
    diag_total_lines_gt = 0
    diag_total_lines_pred = 0
    diag_aligned_lines = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Stage1 Eval [LINE-LEVEL]"):
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

                # 使用 metrics.line_eval 的聚合函数（Single Source of Truth）
                labels_list = labels.cpu().tolist()
                line_ids_list = line_ids_b.cpu().tolist()
                token_preds = [sample_logits[i].argmax().item() for i in range(sample_logits.shape[0])]

                # [诊断] 统计投票情况（只统计文本 token，不含 CLS/SEP/PAD）
                for i, (pred, line_id, label) in enumerate(zip(token_preds, line_ids_list, labels_list)):
                    if line_id >= 0:  # 只统计真正的文本 token
                        diag_total_tokens += 1
                        diag_voted_tokens += 1
                        if label == -100:
                            diag_label_minus100_voted += 1

                # Token → Line 聚合
                line_gt = extract_line_labels_from_tokens(labels_list, line_ids_list)
                line_pred = aggregate_token_to_line_predictions(token_preds, line_ids_list, method="majority")

                diag_total_lines_gt += len(line_gt)
                diag_total_lines_pred += len(line_pred)

                # 对齐并收集结果
                for line_id in sorted(line_gt.keys()):
                    if line_id in line_pred:
                        all_line_labels.append(line_gt[line_id])
                        all_line_predictions.append(line_pred[line_id])
                        diag_aligned_lines += 1

    # [诊断日志] 评估结束后打印一次汇总
    logger.info("=" * 65)
    logger.info("[DIAG Stage1] 投票诊断汇总 (不含 special tokens):")
    logger.info(f"  文本 token: {diag_total_tokens}")
    if diag_voted_tokens > 0:
        pct = diag_label_minus100_voted / diag_voted_tokens * 100
        logger.info(f"  未监督 (label=-100): {diag_label_minus100_voted} ({pct:.1f}%)")
        if pct > 30:
            logger.warning(f"  ⚠️  {pct:.1f}% 的文本 token 未被监督，可能影响 LINE 级别指标！")
    logger.info(f"  GT 行数: {diag_total_lines_gt}, Pred 行数: {diag_total_lines_pred}, 对齐行数: {diag_aligned_lines}")
    logger.info("=" * 65)

    # 使用 metrics.line_eval 计算指标（Single Source of Truth）
    if not all_line_labels:
        return {}

    metrics_result: LineMetricsResult = compute_line_level_metrics(
        line_predictions=all_line_predictions,
        line_labels=all_line_labels,
        num_classes=14,
        class_names=class_names,
    )

    # 打印详细指标
    if log_details:
        metrics_result.log_summary(class_names=class_names, title="Stage1 Line-Level Metrics")

    # 转换为旧格式的字典（保持向后兼容）
    results = {
        "line_macro_f1": metrics_result.macro_f1,
        "line_micro_f1": metrics_result.micro_f1,
        "line_accuracy": metrics_result.accuracy,
        "num_lines": metrics_result.num_lines,
    }

    # 每类 F1（保持旧键名格式）
    for cls_id, cls_metrics in metrics_result.per_class_metrics.items():
        class_name = id2label.get(cls_id, f"class_{cls_id}")
        results[f"f1_{class_name}"] = cls_metrics["f1"]

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

    使用共享的 e2e_inference 模块进行推理，本函数只负责评估指标计算。

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

    # [诊断] line_id 对齐统计
    diag_total_lines = 0
    diag_hit_lines = 0  # GT line_id 在 pred 字典中命中
    diag_miss_lines = 0  # GT line_id 在 pred 字典中未命中（默认给 0）
    diag_pred_line_id_ranges = []  # 记录每个样本的 pred line_id 范围

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="E2E Eval"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 提取 GT（文档级别返回 num_docs 个元素，页面级别返回 batch_size 个元素）
            gt_docs = extract_gt_from_batch(batch, id2label)
            num_samples = len(gt_docs)  # 使用 gt_docs 长度，适配两种模式

            for b in range(num_samples):
                gt_doc = gt_docs[b]
                if not gt_doc:
                    continue

                # 使用共享推理模块进行端到端推理
                pred_output = run_e2e_inference_single(model, batch, batch_idx=b, device=device)

                if pred_output.num_lines == 0:
                    continue

                line_preds = pred_output.line_classes
                pred_parents = pred_output.line_parents
                pred_relations = pred_output.line_relations
                actual_num_lines = pred_output.num_lines

                # [诊断] 记录 pred line_id 范围
                if line_preds:
                    pred_line_ids = list(line_preds.keys())
                    diag_pred_line_id_ranges.append((min(pred_line_ids), max(pred_line_ids), len(pred_line_ids)))

                # 收集分类结果（使用 line_id 对齐）
                for idx, gt_item in enumerate(gt_doc):
                    if idx >= actual_num_lines:
                        break
                    line_id = gt_item["line_id"]  # 使用实际的 line_id

                    gt_class = CLASS2ID.get(gt_item["class"], 0)

                    # [诊断] 统计命中率
                    diag_total_lines += 1
                    if line_id in line_preds:
                        pred_class = line_preds[line_id]
                        diag_hit_lines += 1
                    else:
                        pred_class = 0  # 默认给 0
                        diag_miss_lines += 1

                    all_gt_classes.append(gt_class)
                    all_pred_classes.append(pred_class)

                    all_gt_parents.append(gt_item["parent_id"])
                    all_pred_parents.append(pred_parents[idx] if idx < len(pred_parents) else -1)

                    # 保留原始 relation: "none" -> -100 (与训练一致)
                    gt_rel_str = gt_item.get("relation", "none")
                    if gt_rel_str == "none":
                        gt_rel = -100  # 与训练一致，后续过滤
                    else:
                        gt_rel = RELATION2ID.get(gt_rel_str, -100)
                    all_gt_relations.append(gt_rel)
                    all_pred_relations.append(pred_relations[idx] if idx < len(pred_relations) else 0)

                # 保存用于 TEDS（使用 line_id 对齐）
                if compute_teds:
                    pred_doc = []
                    for idx, gt_item in enumerate(gt_doc):
                        if idx >= actual_num_lines:
                            break
                        line_id = gt_item["line_id"]
                        class_id = line_preds.get(line_id, 0)  # 用 line_id
                        class_name = id2label.get(class_id, f"class_{class_id}")
                        pred_doc.append({
                            "class": class_name,
                            "text": f"line_{line_id}",
                            "parent_id": pred_parents[idx],
                            "relation": ID2RELATION.get(pred_relations[idx], "none"),
                        })

                    all_gt_docs.append(gt_doc)
                    all_pred_docs.append(pred_doc)

    # [诊断日志] line_id 对齐统计
    logger.info("=" * 65)
    logger.info("[DIAG E2E] line_id 对齐诊断:")
    logger.info(f"  总行数: {diag_total_lines}")
    logger.info(f"  命中数: {diag_hit_lines} ({diag_hit_lines/diag_total_lines*100:.1f}%)" if diag_total_lines > 0 else "  命中数: 0")
    logger.info(f"  未命中(默认0): {diag_miss_lines} ({diag_miss_lines/diag_total_lines*100:.1f}%)" if diag_total_lines > 0 else "  未命中: 0")
    if diag_miss_lines > 0:
        logger.warning(f"  ⚠️  {diag_miss_lines} 行因 line_id 对不上被默认预测为 class=0，这会严重影响 macro-F1！")
    if diag_pred_line_id_ranges:
        # 检查 line_id 是否从 0 开始连续
        non_zero_start = sum(1 for r in diag_pred_line_id_ranges if r[0] != 0)
        if non_zero_start > 0:
            logger.warning(f"  ⚠️  {non_zero_start}/{len(diag_pred_line_id_ranges)} 个样本的 pred line_id 不是从 0 开始")
        # 打印几个样本的范围
        logger.info(f"  pred line_id 范围示例 (min, max, count): {diag_pred_line_id_ranges[:5]}")
    logger.info("=" * 65)

    # 计算指标
    results = {}

    if all_gt_classes:
        # 分类指标
        from sklearn.metrics import f1_score, accuracy_score
        results["line_macro_f1"] = f1_score(all_gt_classes, all_pred_classes, average="macro", zero_division=0)
        results["line_micro_f1"] = f1_score(all_gt_classes, all_pred_classes, average="micro", zero_division=0)
        results["line_accuracy"] = accuracy_score(all_gt_classes, all_pred_classes)

        # [诊断] 打印各类别的 GT/Pred 统计
        from collections import Counter
        gt_counter = Counter(all_gt_classes)
        pred_counter = Counter(all_pred_classes)

        # 类别名称（与 labels.py 中的 LABEL_LIST 对应）
        from layoutlmft.data.labels import LABEL_LIST
        class_names = LABEL_LIST

        logger.info("[DIAG E2E] 各类别 GT/Pred 统计:")
        for cls_id in sorted(set(gt_counter.keys()) | set(pred_counter.keys())):
            gt_cnt = gt_counter.get(cls_id, 0)
            pred_cnt = pred_counter.get(cls_id, 0)
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            diff_str = "" if gt_cnt == pred_cnt else f" (diff={pred_cnt - gt_cnt:+d})"
            logger.info(f"  {cls_id:2d} ({cls_name:8s}): GT={gt_cnt:3d}, Pred={pred_cnt:3d}{diff_str}")

        # Parent 准确率
        parent_correct = sum(1 for g, p in zip(all_gt_parents, all_pred_parents) if g == p)
        results["parent_accuracy"] = parent_correct / len(all_gt_parents)

        # Relation 准确率和 F1（只计算有父节点且有有效关系标签的）
        # 与训练一致：跳过 gt_rel == -100 的样本
        rel_pairs = [(g, p) for g, p, gp in zip(all_gt_relations, all_pred_relations, all_gt_parents) if gp >= 0 and g >= 0]

        # [诊断] 统计 relation 样本
        total_rel_candidates = len(all_gt_relations)
        skipped_no_parent = sum(1 for gp in all_gt_parents if gp < 0)
        skipped_invalid_rel = sum(1 for g, gp in zip(all_gt_relations, all_gt_parents) if gp >= 0 and g < 0)
        logger.info(f"[DIAG E2E] Relation统计: 总行数={total_rel_candidates}, 无父节点跳过={skipped_no_parent}, 无效关系跳过={skipped_invalid_rel}, 有效样本={len(rel_pairs)}")

        if rel_pairs:
            gt_rels = [g for g, p in rel_pairs]
            pred_rels = [p for g, p in rel_pairs]
            rel_correct = sum(1 for g, p in rel_pairs if g == p)
            results["relation_accuracy"] = rel_correct / len(rel_pairs)
            results["relation_macro_f1"] = f1_score(gt_rels, pred_rels, average="macro", zero_division=0)
            results["relation_micro_f1"] = f1_score(gt_rels, pred_rels, average="micro", zero_division=0)

            # [诊断] 打印 relation 分布
            from collections import Counter
            gt_rel_counter = Counter(gt_rels)
            pred_rel_counter = Counter(pred_rels)
            logger.info(f"[DIAG E2E] Relation GT分布: connect={gt_rel_counter.get(0, 0)}, contain={gt_rel_counter.get(1, 0)}, equality={gt_rel_counter.get(2, 0)}")
            logger.info(f"[DIAG E2E] Relation Pred分布: connect={pred_rel_counter.get(0, 0)}, contain={pred_rel_counter.get(1, 0)}, equality={pred_rel_counter.get(2, 0)}")
        else:
            logger.warning("[DIAG E2E] 没有有效的 relation 样本用于评估！")

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


def compute_teds_score(
    gt_docs: List[List[Dict]],
    pred_docs: List[List[Dict]],
    max_lines_per_doc: int = 500,  # 超过此行数的文档跳过 TEDS 计算
    timeout_per_doc: float = 60.0,  # 单个文档超时时间（秒）
) -> Optional[float]:
    """
    计算 TEDS 分数（带进度日志和超时保护）

    Args:
        gt_docs: GT 文档列表
        pred_docs: 预测文档列表
        max_lines_per_doc: 单个文档最大行数，超过则跳过
        timeout_per_doc: 单个文档计算超时时间

    Returns:
        Macro TEDS 分数
    """
    import time
    import signal

    try:
        from HRDoc.utils.doc_utils import generate_doc_tree_from_log_line_level, tree_edit_distance
    except ImportError as e:
        logger.warning(f"Cannot import doc_utils for TEDS computation: {e}")
        return None

    teds_list = []
    skipped_docs = 0
    total_docs = len(gt_docs)

    logger.info(f"[TEDS] Starting TEDS computation for {total_docs} documents...")

    for doc_idx, (gt_doc, pred_doc) in enumerate(zip(gt_docs, pred_docs)):
        doc_lines = len(gt_doc)

        if len(gt_doc) != len(pred_doc):
            logger.warning(f"[TEDS] Doc {doc_idx+1}: length mismatch (GT={len(gt_doc)}, Pred={len(pred_doc)}), skipping")
            skipped_docs += 1
            continue

        # 跳过超大文档
        if doc_lines > max_lines_per_doc:
            logger.warning(f"[TEDS] Doc {doc_idx+1}: {doc_lines} lines > {max_lines_per_doc}, skipping (too large)")
            skipped_docs += 1
            continue

        try:
            start_time = time.time()

            gt_texts = [f"{t['class']}:{t.get('text', '')}" for t in gt_doc]
            gt_parents = [t["parent_id"] for t in gt_doc]
            gt_relations = [t.get("relation", "none") for t in gt_doc]

            pred_texts = [f"{t['class']}:{t.get('text', '')}" for t in pred_doc]
            pred_parents = [t["parent_id"] for t in pred_doc]
            pred_relations = [t.get("relation", "none") for t in pred_doc]

            gt_tree = generate_doc_tree_from_log_line_level(gt_texts, gt_parents, gt_relations)
            pred_tree = generate_doc_tree_from_log_line_level(pred_texts, pred_parents, pred_relations)

            _, teds = tree_edit_distance(pred_tree, gt_tree)

            elapsed = time.time() - start_time
            logger.info(f"[TEDS] Doc {doc_idx+1}/{total_docs}: {doc_lines} lines, TEDS={teds:.4f}, time={elapsed:.2f}s")

            teds_list.append(teds)
        except Exception as e:
            logger.warning(f"[TEDS] Doc {doc_idx+1}: computation failed: {e}")
            skipped_docs += 1
            continue

    if teds_list:
        avg_teds = sum(teds_list) / len(teds_list)
        logger.info(f"[TEDS] Completed: {len(teds_list)} docs computed, {skipped_docs} skipped, avg TEDS={avg_teds:.4f}")
        return avg_teds

    logger.warning(f"[TEDS] No valid TEDS scores computed ({skipped_docs} docs skipped)")
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

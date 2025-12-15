#!/usr/bin/env python
# coding=utf-8
"""
端到端评估脚本

功能：
1. 加载 Stage 1/3/4 的模型
2. 对 test 数据进行推理
3. 组装成 HRDoc 格式的 JSON
4. 调用 classify_eval.py 和 teds_eval.py 进行评估

Usage:
    # 基本用法
    python examples/stage/scripts/eval_end2end.py --env test --dataset hrds

    # 指定实验
    python examples/stage/scripts/eval_end2end.py --env test --dataset hrds --exp exp_001

    # 只做分类评估（跳过 TEDS）
    python examples/stage/scripts/eval_end2end.py --env test --dataset hrds --skip_teds

    # 只生成 JSON，不评估
    python examples/stage/scripts/eval_end2end.py --env test --dataset hrds --generate_only

    # Dry run
    python examples/stage/scripts/eval_end2end.py --env test --dataset hrds --dry_run
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import get_config, load_config

# Add examples/stage to path
STAGE_ROOT = os.path.join(PROJECT_ROOT, "examples", "stage")
sys.path.insert(0, STAGE_ROOT)

from util.checkpoint_utils import get_latest_checkpoint, get_best_model
from util.experiment_manager import ensure_experiment

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation")

    # Environment selection
    parser.add_argument("--env", type=str, default=None,
                        help="Environment: dev, test, or auto-detect")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh"],
                        help="Dataset to evaluate: hrds or hrdh")

    # Experiment management
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment ID (default: current or latest)")

    # Override paths
    parser.add_argument("--stage1_model", type=str, default=None,
                        help="Override Stage 1 model path")
    parser.add_argument("--stage3_model", type=str, default=None,
                        help="Override Stage 3 model path (ParentFinder)")
    parser.add_argument("--stage4_model", type=str, default=None,
                        help="Override Stage 4 model path (RelationClassifier)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory for predictions")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Override ground truth directory")

    # Evaluation options
    parser.add_argument("--skip_teds", action="store_true",
                        help="Skip TEDS evaluation (only do classification)")
    parser.add_argument("--skip_classify", action="store_true",
                        help="Skip classification evaluation (only do TEDS)")
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate prediction JSONs, skip evaluation")

    # Other options
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to evaluate (-1 for all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print config and exit without evaluation")

    return parser.parse_args()


def load_stage1_model(model_path: str, device: str):
    """加载 Stage 1 模型（LayoutXLM）"""
    import torch
    from transformers import AutoConfig

    # Import LayoutXLM
    from layoutlmft.models.layoutxlm import (
        LayoutXLMForTokenClassification,
        LayoutXLMConfig,
        LayoutXLMTokenizerFast,
    )
    from layoutlmft.data.labels import NUM_LABELS, get_id2label, get_label2id

    logger.info(f"Loading Stage 1 model from: {model_path}")

    # Load config
    config = LayoutXLMConfig.from_pretrained(model_path)
    config.num_labels = NUM_LABELS
    config.id2label = get_id2label()
    config.label2id = get_label2id()

    # Load model
    model = LayoutXLMForTokenClassification.from_pretrained(
        model_path,
        config=config,
    )
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(model_path)

    logger.info(f"  Loaded model with {NUM_LABELS} labels")

    return model, tokenizer


def load_stage3_model(model_path: str, device: str, use_gru: bool = True, num_classes: int = 14):
    """加载 Stage 3 模型（ParentFinder）"""
    import torch

    logger.info(f"Loading Stage 3 model from: {model_path}")

    # Load checkpoint or state_dict
    checkpoint = torch.load(model_path, map_location=device)

    # 根据模型类型创建
    if use_gru:
        from train_parent_finder import ParentFinderGRU
        model = ParentFinderGRU(
            hidden_size=768,
            gru_hidden_size=512,
            num_classes=num_classes,
            dropout=0.1,
            use_soft_mask=False,  # 推理时不需要 soft mask
        )
    else:
        from train_parent_finder import SimpleParentFinder
        model = SimpleParentFinder(hidden_size=768, dropout=0.1)

    # 兼容两种保存格式：checkpoint (dict with 'model_state_dict') 或直接 state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        val_acc = checkpoint.get('val_acc', checkpoint.get('parent_acc', 'N/A'))
    else:
        model.load_state_dict(checkpoint)
        val_acc = 'N/A'

    model.to(device)
    model.eval()

    logger.info(f"  Loaded model with val_acc: {val_acc}")

    return model


def load_stage4_model(model_path: str, device: str):
    """加载 Stage 4 模型（RelationClassifier）"""
    import torch
    from layoutlmft.models.relation_classifier import MultiClassRelationClassifier

    logger.info(f"Loading Stage 4 model from: {model_path}")

    # Load checkpoint or state_dict
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model = MultiClassRelationClassifier(
        hidden_size=768,
        num_relations=4,
        use_geometry=True,
        dropout=0.1
    )

    # 兼容两种保存格式：checkpoint (dict with 'model_state_dict') 或直接 state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        val_f1 = checkpoint.get('val_f1', 'N/A')
    else:
        model.load_state_dict(checkpoint)
        val_f1 = 'N/A'

    model.to(device)
    model.eval()

    logger.info(f"  Loaded model with val_f1: {val_f1}")

    return model


def run_inference(
    stage1_model,
    stage3_model,
    stage4_model,
    tokenizer,
    data_dir: str,
    output_dir: str,
    device: str,
    max_samples: int = -1,
    batch_size: int = 1,
):
    """
    运行端到端推理

    Args:
        stage1_model: LayoutXLM 分类模型
        stage3_model: ParentFinder 模型
        stage4_model: RelationClassifier 模型
        tokenizer: LayoutXLM tokenizer
        data_dir: 数据目录（包含 test 子目录）
        output_dir: 输出目录
        device: 设备
        max_samples: 最大样本数
        batch_size: 批大小

    Returns:
        pred_dir: 预测结果目录
    """
    import torch
    import numpy as np
    from datasets import load_dataset
    from layoutlmft.data.labels import ID2LABEL, LABEL2ID
    from layoutlmft.models.relation_classifier import compute_geometry_features, RELATION_NAMES

    logger.info("Running end-to-end inference...")

    # Set environment for dataset loading
    os.environ["HRDOC_DATA_DIR"] = data_dir

    # Load test dataset
    import layoutlmft.data.datasets.hrdoc
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))

    if "test" not in datasets:
        raise ValueError("Test split not found in dataset")

    test_dataset = datasets["test"]
    logger.info(f"Test dataset size: {len(test_dataset)}")

    if max_samples > 0:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
        logger.info(f"Limited to {len(test_dataset)} samples")

    # Create output directory
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Process each sample
    results = []

    for idx, sample in enumerate(tqdm(test_dataset, desc="Inference")):
        try:
            # Get sample info
            doc_name = sample.get("document_name", f"doc_{idx}")
            page_num = sample.get("page_number", 0)
            tokens = sample["tokens"]
            bboxes = sample["bboxes"]
            line_ids = sample.get("line_ids", list(range(len(tokens))))
            image = sample["image"]

            # Skip empty samples
            if len(tokens) == 0:
                continue

            # === Stage 1: Classification ===
            # Tokenize input
            encoding = tokenizer(
                tokens,
                boxes=bboxes,
                is_split_into_words=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )

            # Move to device
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            bbox = encoding["bbox"].to(device)

            # Handle image
            if image is not None:
                image_tensor = torch.tensor(image).unsqueeze(0).to(device)
            else:
                image_tensor = torch.zeros(1, 3, 224, 224).to(device)

            # Forward pass (需要获取隐藏状态用于 Stage 3/4)
            with torch.no_grad():
                outputs = stage1_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    image=image_tensor,
                    output_hidden_states=True,
                )

            # Get predictions
            logits = outputs.logits  # [1, seq_len, num_labels]
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

            # Map predictions back to lines
            # Get word_ids from tokenizer
            word_ids = encoding.word_ids(batch_index=0)

            # Aggregate token predictions to line predictions
            line_predictions = {}
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id >= len(line_ids):
                    continue
                line_id = line_ids[word_id]
                if line_id not in line_predictions:
                    line_predictions[line_id] = []
                line_predictions[line_id].append(predictions[token_idx])

            # Get majority vote for each line
            line_labels = {}
            for line_id, preds in line_predictions.items():
                # Use majority vote
                from collections import Counter
                counter = Counter(preds)
                line_labels[line_id] = counter.most_common(1)[0][0]

            # === Build line-level features for Stage 3/4 ===
            # For now, use simple features (in production, use Stage 1 hidden states)
            # This is a simplified version - ideally we'd extract features from Stage 1

            # Group tokens by line
            line_info = {}
            for i, (token, bbox_item, line_id) in enumerate(zip(tokens, bboxes, line_ids)):
                if line_id not in line_info:
                    line_info[line_id] = {
                        "tokens": [],
                        "bboxes": [],
                        "text": "",
                        "box": None,
                    }
                line_info[line_id]["tokens"].append(token)
                line_info[line_id]["bboxes"].append(bbox_item)

            # Compute line text and merged box
            for line_id, info in line_info.items():
                info["text"] = " ".join(info["tokens"])
                # Merge bboxes
                all_bboxes = info["bboxes"]
                info["box"] = [
                    min(b[0] for b in all_bboxes),
                    min(b[1] for b in all_bboxes),
                    max(b[2] for b in all_bboxes),
                    max(b[3] for b in all_bboxes),
                ]

            # === Stage 3 & 4: Parent and Relation prediction ===
            # 使用 Stage 3/4 模型进行预测（而不是启发式规则）

            sorted_line_ids = sorted(line_info.keys())
            num_lines = len(sorted_line_ids)

            # 提取行级特征（从 Stage 1 隐藏状态）
            hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
            text_seq_len = input_ids.shape[1]
            text_hidden = hidden_states[:, :text_seq_len, :]  # [1, seq_len, H]

            # 构建 line_ids tensor 用于特征聚合
            line_ids_tensor = torch.full((1, text_seq_len), -1, dtype=torch.long, device=device)
            for token_idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id < len(line_ids):
                    # 将原始 line_id 映射到排序后的索引
                    orig_line_id = line_ids[word_id]
                    if orig_line_id in sorted_line_ids:
                        local_idx = sorted_line_ids.index(orig_line_id)
                        line_ids_tensor[0, token_idx] = local_idx

            # 使用 LineFeatureExtractor 提取行级特征
            from layoutlmft.models.relation_classifier import LineFeatureExtractor
            feature_extractor = LineFeatureExtractor()
            line_features, line_mask = feature_extractor.extract_line_features(
                text_hidden, line_ids_tensor, pooling="mean"
            )  # [1, max_lines, H], [1, max_lines]

            # === Stage 3: Parent prediction ===
            parent_predictions = [-1] * num_lines  # 默认指向 ROOT

            if stage3_model is not None and num_lines > 0:
                with torch.no_grad():
                    # GRU 方法: parent_logits [1, L+1, L+1]
                    parent_logits = stage3_model(line_features, line_mask)

                    for child_idx in range(num_lines):
                        # child_idx 在 logits 中的位置是 child_idx + 1
                        child_logits = parent_logits[0, child_idx + 1, :child_idx + 2]  # [child_idx + 2]

                        # 处理 -inf
                        child_logits = torch.where(
                            torch.isinf(child_logits),
                            torch.full_like(child_logits, -1e4),
                            child_logits
                        )

                        # 预测 parent (index 0 = ROOT, index 1+ = 实际行)
                        pred_idx = child_logits.argmax().item()
                        if pred_idx == 0:
                            parent_predictions[child_idx] = -1  # ROOT
                        else:
                            parent_predictions[child_idx] = pred_idx - 1  # 实际行索引

            # === Stage 4: Relation prediction ===
            relation_predictions = ["meta"] * num_lines  # 默认 meta

            # 构建行级 bbox
            line_bboxes = []
            for line_id in sorted_line_ids:
                info = line_info[line_id]
                line_bboxes.append(info["box"])
            line_bboxes_tensor = torch.tensor(line_bboxes, dtype=torch.float32, device=device)

            if stage4_model is not None:
                with torch.no_grad():
                    for child_idx in range(num_lines):
                        parent_idx = parent_predictions[child_idx]

                        if parent_idx < 0:
                            # 指向 ROOT 的元素标记为 meta
                            relation_predictions[child_idx] = "meta"
                        else:
                            # 使用 Stage 4 预测 relation
                            parent_feat = line_features[0, parent_idx].unsqueeze(0)  # [1, H]
                            child_feat = line_features[0, child_idx].unsqueeze(0)  # [1, H]

                            parent_bbox = line_bboxes_tensor[parent_idx]
                            child_bbox = line_bboxes_tensor[child_idx]
                            geom_feat = compute_geometry_features(parent_bbox, child_bbox).unsqueeze(0).to(device)

                            rel_logits = stage4_model(parent_feat, child_feat, geom_feat)  # [1, 4]
                            pred_rel = rel_logits.argmax(dim=1).item()

                            # 映射回关系名称 (0=none, 1=connect, 2=contain, 3=equality)
                            if pred_rel == 0:
                                relation_predictions[child_idx] = "none"
                            else:
                                relation_predictions[child_idx] = RELATION_NAMES[pred_rel]

            # Build output JSON
            output_json = []

            for line_idx, line_id in enumerate(sorted_line_ids):
                info = line_info[line_id]
                label_id = line_labels.get(line_id, 0)
                label_name = ID2LABEL.get(label_id, "paraline")

                parent_id = parent_predictions[line_idx]
                relation = relation_predictions[line_idx]

                output_json.append({
                    "text": info["text"],
                    "box": info["box"],
                    "class": label_name,
                    "page": page_num,
                    "line_id": line_idx,
                    "parent_id": parent_id,
                    "relation": relation,
                })

            # Save to file
            output_file = os.path.join(pred_dir, f"{doc_name}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)

            results.append({
                "doc_name": doc_name,
                "page_num": page_num,
                "num_lines": len(output_json),
            })

        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue

    logger.info(f"Inference completed. {len(results)} documents processed.")
    logger.info(f"Predictions saved to: {pred_dir}")

    return pred_dir


def run_classify_eval(gt_dir: str, pred_dir: str, use_subprocess: bool = False):
    """
    运行分类评估

    Args:
        gt_dir: Ground truth 目录
        pred_dir: 预测结果目录
        use_subprocess: 是否使用 subprocess 调用（True=子进程，False=直接调用）

    Returns:
        dict: 评估结果（如果直接调用）或 returncode（如果 subprocess）
    """
    logger.info("=" * 60)
    logger.info("Running Classification Evaluation")
    logger.info("=" * 60)
    logger.info(f"  GT Dir:   {gt_dir}")
    logger.info(f"  Pred Dir: {pred_dir}")

    if use_subprocess:
        # 使用 subprocess 调用，日志会直接打印到终端
        import subprocess

        eval_script = os.path.join(PROJECT_ROOT, "HRDoc", "utils", "classify_eval.py")

        if not os.path.exists(eval_script):
            logger.error(f"classify_eval.py not found: {eval_script}")
            return None

        cmd = [
            sys.executable, eval_script,
            "--gt_folder", gt_dir,
            "--pred_folder", pred_dir,
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        # 使用实时输出（不捕获，直接打印到终端）
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        return result.returncode
    else:
        # 直接调用函数，日志通过 logging 模块输出
        from sklearn.metrics import f1_score
        import json as json_module

        # 添加 HRDoc/utils 到路径
        hrdoc_utils_path = os.path.join(PROJECT_ROOT, "HRDoc", "utils")
        if hrdoc_utils_path not in sys.path:
            sys.path.insert(0, hrdoc_utils_path)

        try:
            from utils import trans_class
        except ImportError:
            # 如果导入失败，定义一个简单的替代函数
            def trans_class(jdata, unit):
                return unit.get("class", "paraline")

        # 14 类标签映射
        class2id_dict = {
            "title": 0, "author": 1, "mail": 2, "affili": 3,
            "section": 4, "fstline": 5, "paraline": 6,
            "table": 7, "figure": 8, "caption": 9, "equation": 10,
            "footer": 11, "header": 12, "footnote": 13,
        }
        id2class_dict = {v: k for k, v in class2id_dict.items()}

        def class2id(jdata, unit):
            class_ = unit.get('class', 'paraline')
            if class_ not in class2id_dict:
                class_ = trans_class(jdata, unit)
            return class2id_dict.get(class_, 6)  # 默认 paraline

        # 检查文件一致性
        gt_files = set(f for f in os.listdir(gt_dir) if f.endswith('.json'))
        pred_files = set(f for f in os.listdir(pred_dir) if f.endswith('.json'))

        common_files = gt_files & pred_files
        if len(common_files) == 0:
            logger.error("No common files found between GT and Pred directories!")
            return None

        if gt_files != pred_files:
            logger.warning(f"File mismatch: {len(gt_files)} GT files, {len(pred_files)} Pred files")
            logger.warning(f"Only evaluating {len(common_files)} common files")

        gt_class = []
        pred_class = []

        for pdf_file in tqdm(sorted(common_files), desc="Evaluating"):
            try:
                gt_file = os.path.join(gt_dir, pdf_file)
                pred_file = os.path.join(pred_dir, pdf_file)

                with open(gt_file, 'r', encoding='utf-8') as f:
                    gt_json = json_module.load(f)
                with open(pred_file, 'r', encoding='utf-8') as f:
                    pred_json = json_module.load(f)

                if len(gt_json) != len(pred_json):
                    logger.warning(f"{pdf_file}: line count mismatch (GT={len(gt_json)}, Pred={len(pred_json)})")
                    # 取最小长度
                    min_len = min(len(gt_json), len(pred_json))
                    gt_json = gt_json[:min_len]
                    pred_json = pred_json[:min_len]

                gt_class.extend([class2id(gt_json, x) for x in gt_json])
                pred_class.extend([class2id(pred_json, x) for x in pred_json])
            except Exception as e:
                logger.warning(f"Error processing {pdf_file}: {e}")
                continue

        if len(gt_class) == 0:
            logger.error("No valid samples found for evaluation!")
            return None

        # 计算 F1
        detailed_f1 = f1_score(gt_class, pred_class, average=None)
        macro_f1 = f1_score(gt_class, pred_class, average='macro')
        micro_f1 = f1_score(gt_class, pred_class, average='micro')

        # 打印结果
        unique_classes = sorted(set(gt_class) | set(pred_class))

        logger.info(f"Evaluated {len(gt_class)} lines from {len(common_files)} documents")
        logger.info("=" * 55)
        logger.info("Per-class F1 scores:")
        logger.info("-" * 55)
        logger.info(f"{'Rank':<6}{'Class':<12}{'ID':<6}{'F1 Score':<10}")
        logger.info("-" * 55)

        class_results = []
        for i, class_id in enumerate(unique_classes):
            if i < len(detailed_f1):
                class_name = id2class_dict.get(class_id, f"unknown_{class_id}")
                class_results.append((class_name, class_id, detailed_f1[i]))

        class_results_sorted = sorted(class_results, key=lambda x: x[2], reverse=True)
        for rank, (class_name, class_id, f1) in enumerate(class_results_sorted, 1):
            logger.info(f"{rank:<6}{class_name:<12}{class_id:<6}{f1:.4f}")

        logger.info("-" * 55)
        logger.info(f"{'Macro F1:':<20}{macro_f1:.4f}")
        logger.info(f"{'Micro F1:':<20}{micro_f1:.4f}")
        logger.info("=" * 55)

        return {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "detailed_f1": detailed_f1.tolist(),
            "num_samples": len(gt_class),
            "num_documents": len(common_files),
        }


def run_teds_eval(gt_dir: str, pred_dir: str, use_subprocess: bool = True):
    """
    运行 TEDS 评估

    Args:
        gt_dir: Ground truth 目录
        pred_dir: 预测结果目录
        use_subprocess: 是否使用 subprocess 调用（TEDS 计算复杂，建议用 subprocess）

    Returns:
        dict: 评估结果
    """
    logger.info("=" * 60)
    logger.info("Running TEDS Evaluation")
    logger.info("=" * 60)
    logger.info(f"  GT Dir:   {gt_dir}")
    logger.info(f"  Pred Dir: {pred_dir}")

    if use_subprocess:
        import subprocess

        eval_script = os.path.join(PROJECT_ROOT, "HRDoc", "utils", "teds_eval.py")

        if not os.path.exists(eval_script):
            logger.error(f"teds_eval.py not found: {eval_script}")
            return None

        cmd = [
            sys.executable, eval_script,
            "--gt_folder", gt_dir,
            "--pred_folder", pred_dir,
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        # 使用实时输出
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        return result.returncode
    else:
        # 直接调用 TEDS 评估函数
        # 添加 HRDoc/utils 到路径
        hrdoc_utils_path = os.path.join(PROJECT_ROOT, "HRDoc", "utils")
        if hrdoc_utils_path not in sys.path:
            sys.path.insert(0, hrdoc_utils_path)

        try:
            from doc_utils import generate_doc_tree_from_log_line_level, tree_edit_distance
        except ImportError as e:
            logger.error(f"Failed to import doc_utils: {e}")
            logger.error("TEDS evaluation requires apted library. Install with: pip install apted")
            return None

        import json as json_module

        # 检查文件
        gt_files = set(f for f in os.listdir(gt_dir) if f.endswith('.json'))
        pred_files = set(f for f in os.listdir(pred_dir) if f.endswith('.json'))
        common_files = gt_files & pred_files

        if len(common_files) == 0:
            logger.error("No common files found!")
            return None

        all_teds = []
        all_distances = []
        all_gt_nodes = []
        all_pred_nodes = []

        for pdf_file in tqdm(sorted(common_files), desc="TEDS"):
            try:
                gt_file = os.path.join(gt_dir, pdf_file)
                pred_file = os.path.join(pred_dir, pdf_file)

                with open(gt_file, 'r', encoding='utf-8') as f:
                    gt_info = json_module.load(f)
                with open(pred_file, 'r', encoding='utf-8') as f:
                    pred_info = json_module.load(f)

                if len(gt_info) != len(pred_info):
                    continue

                gt_texts = [t['class'] + ":" + t.get('text', '') for t in gt_info]
                gt_parent_idx = [t.get('parent_id', -1) for t in gt_info]
                gt_relation = [t.get('relation', 'meta') for t in gt_info]

                pred_texts = [t['class'] + ":" + t.get('text', '') for t in pred_info]
                pred_parent_idx = [t.get('parent_id', -1) for t in pred_info]
                pred_relation = [t.get('relation', 'meta') for t in pred_info]

                gt_tree = generate_doc_tree_from_log_line_level(gt_texts, gt_parent_idx, gt_relation)
                pred_tree = generate_doc_tree_from_log_line_level(pred_texts, pred_parent_idx, pred_relation)

                distance, teds = tree_edit_distance(pred_tree, gt_tree)

                all_teds.append(teds)
                all_distances.append(distance)
                all_gt_nodes.append(len(gt_tree))
                all_pred_nodes.append(len(pred_tree))

            except Exception as e:
                logger.warning(f"Error processing {pdf_file}: {e}")
                continue

        if len(all_teds) == 0:
            logger.error("No valid documents for TEDS evaluation!")
            return None

        import numpy as np
        macro_teds = np.mean(all_teds)
        micro_teds = 1.0 - float(sum(all_distances)) / sum(
            [max(all_gt_nodes[i], all_pred_nodes[i]) for i in range(len(all_teds))]
        )

        logger.info(f"Evaluated {len(all_teds)} documents")
        logger.info("=" * 40)
        logger.info(f"{'Macro TEDS:':<20}{macro_teds:.4f}")
        logger.info(f"{'Micro TEDS:':<20}{micro_teds:.4f}")
        logger.info("=" * 40)

        return {
            "macro_teds": macro_teds,
            "micro_teds": micro_teds,
            "num_documents": len(all_teds),
        }


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # Load configuration
    if args.env:
        config = load_config(args.env)
        config = config.get_effective_config()
    else:
        config = get_config()

    # Set GPU
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    # Initialize experiment manager
    exp_manager, exp_dir = ensure_experiment(
        config,
        exp=args.exp,
        new_exp=False,
        name=f"Eval {args.dataset.upper()}",
    )

    # Determine paths
    # Stage 1 model
    if args.stage1_model:
        stage1_model_path = args.stage1_model
    else:
        stage1_dir = exp_manager.get_stage_dir(args.exp, "stage1", args.dataset)
        stage1_model_path = get_latest_checkpoint(stage1_dir)

    # Stage 3 model
    if args.stage3_model:
        stage3_model_path = args.stage3_model
    else:
        # 优先从单独的 stage3 目录查找
        stage3_dir = exp_manager.get_stage_dir(args.exp, "stage3", args.dataset)
        stage3_model_path = get_best_model(stage3_dir)

        # 如果没找到，尝试从 joint 训练目录查找
        if not stage3_model_path:
            joint_dir = exp_manager.get_stage_dir(args.exp, "joint", args.dataset)
            joint_checkpoint = get_latest_checkpoint(joint_dir)
            if joint_checkpoint:
                joint_stage3_path = os.path.join(joint_checkpoint, "stage3.pt")
                if os.path.exists(joint_stage3_path):
                    stage3_model_path = joint_stage3_path
                    logger.info(f"Using Stage 3 model from joint training: {stage3_model_path}")

    # Stage 4 model
    if args.stage4_model:
        stage4_model_path = args.stage4_model
    else:
        # 优先从单独的 stage4 目录查找
        stage4_dir = exp_manager.get_stage_dir(args.exp, "stage4", args.dataset)
        stage4_model_path = get_best_model(stage4_dir)

        # 如果没找到，尝试从 joint 训练目录查找
        if not stage4_model_path:
            joint_dir = exp_manager.get_stage_dir(args.exp, "joint", args.dataset)
            joint_checkpoint = get_latest_checkpoint(joint_dir)
            if joint_checkpoint:
                joint_stage4_path = os.path.join(joint_checkpoint, "stage4.pt")
                if os.path.exists(joint_stage4_path):
                    stage4_model_path = joint_stage4_path
                    logger.info(f"Using Stage 4 model from joint training: {stage4_model_path}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = exp_manager.get_stage_dir(args.exp, "eval", args.dataset)

    # Data and GT directories
    data_dir = config.dataset.get_data_dir(args.dataset)
    if args.gt_dir:
        gt_dir = args.gt_dir
    else:
        gt_dir = os.path.join(data_dir, "test")

    # GPU info
    import torch
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    # Print configuration
    print("=" * 60)
    print("End-to-End Evaluation")
    print("=" * 60)
    print(f"Environment:    {config.env}")
    print(f"Dataset:        {args.dataset.upper()}")
    print(f"Device:         {device}")
    print(f"Experiment:     {os.path.basename(exp_dir)}")
    print("-" * 60)
    print("Model Paths:")
    print(f"  Stage 1:      {stage1_model_path or 'NOT FOUND'}")
    print(f"  Stage 3:      {stage3_model_path or 'NOT FOUND'}")
    print(f"  Stage 4:      {stage4_model_path or 'NOT FOUND'}")
    print("-" * 60)
    print("Data Paths:")
    print(f"  Data Dir:     {data_dir}")
    print(f"  GT Dir:       {gt_dir}")
    print(f"  Output Dir:   {output_dir}")
    print("-" * 60)
    print("Evaluation Options:")
    print(f"  Skip Classify: {args.skip_classify}")
    print(f"  Skip TEDS:     {args.skip_teds}")
    print(f"  Generate Only: {args.generate_only}")
    print(f"  Max Samples:   {args.max_samples}")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run mode - exiting without evaluation]")
        return

    # Check required paths
    if not stage1_model_path or not os.path.exists(stage1_model_path):
        logger.error(f"Stage 1 model not found: {stage1_model_path}")
        logger.error("Please run Stage 1 training first.")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    logger.info("\nLoading models...")
    stage1_model, tokenizer = load_stage1_model(stage1_model_path, device)

    # Stage 3/4 models
    stage3_model = None
    stage4_model = None
    if stage3_model_path and os.path.exists(stage3_model_path):
        try:
            stage3_model = load_stage3_model(stage3_model_path, device, use_gru=True)
        except Exception as e:
            logger.warning(f"Failed to load Stage 3 model: {e}")
            logger.warning("Stage 3 model not loaded - parent prediction will default to ROOT")

    if stage4_model_path and os.path.exists(stage4_model_path):
        try:
            stage4_model = load_stage4_model(stage4_model_path, device)
        except Exception as e:
            logger.warning(f"Failed to load Stage 4 model: {e}")
            logger.warning("Will use heuristic rules for relation prediction")

    # Run inference
    logger.info("\nRunning inference...")
    pred_dir = run_inference(
        stage1_model=stage1_model,
        stage3_model=stage3_model,
        stage4_model=stage4_model,
        tokenizer=tokenizer,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    if args.generate_only:
        logger.info("\n[Generate only mode - skipping evaluation]")
        logger.info(f"Predictions saved to: {pred_dir}")
        return

    # Run evaluations
    eval_results = {}

    if not args.skip_classify:
        classify_result = run_classify_eval(gt_dir, pred_dir, use_subprocess=False)
        if classify_result:
            eval_results["classify"] = classify_result

    if not args.skip_teds:
        teds_result = run_teds_eval(gt_dir, pred_dir, use_subprocess=False)
        if teds_result:
            eval_results["teds"] = teds_result

    # Save evaluation results
    if eval_results:
        results_file = os.path.join(output_dir, "eval_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(eval_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        logger.info(f"Evaluation results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 60)
    print("End-to-End Evaluation Completed!")
    print("=" * 60)
    print(f"Predictions saved to: {pred_dir}")
    print("\nTo view detailed results, check the log output above.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Joint Model End-to-End Inference - 使用联合训练模型进行端到端推理

复用共享模块：
- e2e_inference.py: 端到端推理逻辑
- hrdoc_eval.py: 评估逻辑

Usage:
    python run_joint_infer.py --env test --dataset hrds
    python run_joint_infer.py --env test --dataset hrdh --quick
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace

# Add project paths (before GPU setup)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE_DIR = PROJECT_ROOT / "examples" / "stage"
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "configs"))
sys.path.insert(0, str(EXAMPLES_ROOT))
sys.path.insert(0, str(STAGE_DIR))
sys.path.insert(0, str(STAGE_DIR / "util"))

# ==================== GPU 设置（必须在 import torch 之前）====================
from utils.gpu import setup_gpu_early
setup_gpu_early()

import torch
from tqdm import tqdm

# 切换工作目录以支持 train_parent_finder.py 中的相对导入
os.chdir(STAGE_DIR)

from layoutlmft.data.labels import LABEL_LIST, NUM_LABELS, id2label, get_id2label, get_label2id
from layoutlmft.models.layoutxlm import LayoutXLMTokenizerFast

# 从共享模块导入
from models.build import load_joint_model, get_latest_joint_checkpoint

from e2e_inference import run_e2e_inference_single
from joint_data_collator import HRDocJointDataCollator

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Relation labels
RELATION_LABELS = ["connect", "contain", "equality"]
ID2RELATION = {i: label for i, label in enumerate(RELATION_LABELS)}


def load_config(env: str):
    """Load configuration from YAML file."""
    from config_loader import load_config as _load_config
    return _load_config(env)


def get_data_dir(config, dataset: str) -> str:
    """Get data directory for dataset."""
    return config.dataset.get_data_dir(dataset)



def run_inference(model_path: str, data_dir: str, output_dir: str = None, config=None, max_test_samples: int = None):
    """Run end-to-end inference using shared e2e_inference module.

    Args:
        model_path: Joint model checkpoint path
        data_dir: Data directory containing test/ subdirectory
        output_dir: Deprecated, ignored (runs are saved to data_dir/runs/)
        config: Configuration object
        max_test_samples: Limit number of test samples

    Returns:
        runs_dir: Path to the run directory containing results
    """
    from datasets import load_dataset
    import layoutlmft.data.datasets.hrdoc

    logger.info("=" * 60)
    logger.info("Joint Model End-to-End Inference")
    logger.info("=" * 60)
    logger.info(f"Model:      {model_path}")
    logger.info(f"Data Dir:   {data_dir}")

    # Set environment
    os.environ["HRDOC_DATA_DIR"] = data_dir
    if config and hasattr(config, 'paths') and config.paths.hf_cache_dir:
        os.environ["HF_HOME"] = config.paths.hf_cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(config.paths.hf_cache_dir, "datasets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model, tokenizer = load_joint_model(model_path, device, config=config)

    # Load and tokenize dataset (只加载 test split，避免验证 train 数据的标签)
    test_dataset = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__), split="test")
    if max_test_samples and max_test_samples > 0:
        test_dataset = test_dataset.select(range(min(len(test_dataset), max_test_samples)))

    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Tokenize
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"], padding="max_length", truncation=True,
            max_length=512, is_split_into_words=True, return_overflowing_tokens=True,
        )

        all_labels, all_bboxes, all_images = [], [], []
        all_line_ids, all_line_parent_ids, all_line_relations = [], [], []
        all_doc_names = []
        all_page_numbers = []  # 新增：保留 page_number
        label2id = get_label2id()

        for batch_idx in range(len(tokenized["input_ids"])):
            word_ids = tokenized.word_ids(batch_index=batch_idx)
            org_idx = tokenized["overflow_to_sample_mapping"][batch_idx]

            label = examples["ner_tags"][org_idx]
            bbox = examples["bboxes"][org_idx]
            image = examples["image"][org_idx]
            line_ids_raw = examples.get("line_ids", [[]])[org_idx]
            doc_name = examples.get("document_name", ["unknown"])[org_idx]
            page_number = examples.get("page_number", [0])[org_idx]  # 新增

            token_labels, token_bboxes, token_line_ids = [], [], []
            prev_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    token_labels.append(-100)
                    token_bboxes.append([0, 0, 0, 0])
                    token_line_ids.append(-1)
                elif word_idx != prev_word_idx:
                    lbl = label[word_idx]
                    token_labels.append(lbl if isinstance(lbl, int) else label2id.get(lbl, 0))
                    token_bboxes.append(bbox[word_idx])
                    token_line_ids.append(line_ids_raw[word_idx] if word_idx < len(line_ids_raw) else -1)
                else:
                    token_labels.append(-100)
                    token_bboxes.append(bbox[word_idx])
                    token_line_ids.append(line_ids_raw[word_idx] if word_idx < len(line_ids_raw) else -1)
                prev_word_idx = word_idx

            all_labels.append(token_labels)
            all_bboxes.append(token_bboxes)
            all_images.append(image)
            all_line_ids.append(token_line_ids)
            all_line_parent_ids.append(examples.get("line_parent_ids", [[]])[org_idx])
            all_line_relations.append(examples.get("line_relations", [[]])[org_idx])
            all_doc_names.append(doc_name)
            all_page_numbers.append(page_number)  # 新增

        tokenized["labels"] = all_labels
        tokenized["bbox"] = all_bboxes
        tokenized["image"] = all_images
        tokenized["line_ids"] = all_line_ids
        tokenized["line_parent_ids"] = all_line_parent_ids
        tokenized["line_relations"] = all_line_relations
        tokenized["document_name"] = all_doc_names
        tokenized["page_number"] = all_page_numbers  # 新增
        return tokenized

    test_dataset_tokenized = test_dataset.map(
        tokenize_and_align, batched=True,
        remove_columns=test_dataset.column_names, num_proc=1,
    )

    # DataLoader
    data_collator = HRDocJointDataCollator(tokenizer=tokenizer, padding=True, max_length=512)

    def custom_collator(features):
        doc_names = [f.pop("document_name", "unknown") for f in features]
        page_numbers = [f.pop("page_number", 0) for f in features]  # 新增
        batch = data_collator(features)
        batch["document_name"] = doc_names
        batch["page_number"] = page_numbers  # 新增
        return batch

    test_loader = torch.utils.data.DataLoader(
        test_dataset_tokenized, batch_size=1, shuffle=False,
        collate_fn=custom_collator, num_workers=0,
    )

    # Run inference using shared module
    # 使用 (doc_name, page_number) 作为 key，避免多页 line_id 互相覆盖
    doc_page_predictions = defaultdict(lambda: defaultdict(dict))

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            doc_name = batch["document_name"][0]
            page_number = batch["page_number"][0]  # 获取页码

            # 使用共享推理模块
            pred_output = run_e2e_inference_single(model, batch, batch_idx=0, device=device)

            if pred_output.num_lines == 0:
                continue

            # 存储预测结果：用 (doc_name, page_number, line_id) 作为 key
            for idx, line_id in enumerate(pred_output.line_ids):
                if idx >= pred_output.num_lines:
                    break
                doc_page_predictions[doc_name][page_number][line_id] = {
                    "class": id2label(pred_output.line_classes.get(line_id, 0)),
                    "parent_id": pred_output.line_parents[idx] if idx < len(pred_output.line_parents) else -1,
                    "relation": ID2RELATION.get(pred_output.line_relations[idx], "connect") if idx < len(pred_output.line_relations) else "connect",
                }

    # ==================== 创建 runs 目录结构 ====================
    from datetime import datetime
    import hashlib

    # 生成 run_id: YYYYMMDD_HHMMSS_modelname_hash
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path)
    run_hash = hashlib.md5(f"{model_path}{timestamp}".encode()).hexdigest()[:6]
    run_id = f"{timestamp}_{model_name}_{run_hash}"

    runs_dir = os.path.join(data_dir, "runs", run_id)
    enriched_dir = os.path.join(runs_dir, "enriched")
    os.makedirs(enriched_dir, exist_ok=True)

    logger.info(f"Writing results to: {runs_dir}")

    # ==================== 保存 manifest.json ====================
    total_lines = sum(
        len(preds) for page_preds in doc_page_predictions.values() for preds in page_preds.values()
    )
    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_path": model_path,
        "data_dir": data_dir,
        "device": str(device),
        "num_documents": len(doc_page_predictions),
        "total_lines": total_lines,
    }
    with open(os.path.join(runs_dir, "manifest.json"), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ==================== 保存 predictions.jsonl ====================
    # 纯预测输出，每行一个预测记录
    predictions_path = os.path.join(runs_dir, "predictions.jsonl")
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for doc_name, page_preds in doc_page_predictions.items():
            for page_number, preds in page_preds.items():
                for line_id, pred_info in preds.items():
                    record = {
                        "doc_name": doc_name,
                        "page": page_number,
                        "line_id": line_id,
                        "class": pred_info["class"],
                        "parent_id": pred_info["parent_id"],
                        "relation": pred_info["relation"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ==================== 保存 enriched/ 融合后 JSON ====================
    # 非元信息的类别列表（用于判断 is_meta）
    struct_classes = {"title", "section", "list", "table", "figure", "equation", "paraline"}

    test_gt_dir = os.path.join(data_dir, "test")
    json_files = sorted([f for f in os.listdir(test_gt_dir) if f.endswith('.json')])
    converted_count = 0

    for json_file in json_files:
        doc_name = os.path.splitext(json_file)[0]
        gt_path = os.path.join(test_gt_dir, json_file)
        enriched_path = os.path.join(enriched_dir, json_file)

        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        # ==================== DEBUG: 打印原始 GT 数据信息 ====================
        logger.info(f"\n{'='*60}")
        logger.info(f"[DEBUG] Processing: {json_file}")
        logger.info(f"[DEBUG] GT path: {gt_path}")
        logger.info(f"[DEBUG] GT records count: {len(gt_data)}")
        if len(gt_data) > 0:
            first_item = gt_data[0]
            logger.info(f"[DEBUG] GT first item keys: {list(first_item.keys())}")
            logger.info(f"[DEBUG] GT first item bbox: {first_item.get('box', first_item.get('bbox', 'NO_BBOX'))}")
            logger.info(f"[DEBUG] GT first item text: {first_item.get('text', 'NO_TEXT')[:50]}...")
            logger.info(f"[DEBUG] GT first item page: {first_item.get('page', 'NO_PAGE')}")
            logger.info(f"[DEBUG] GT first item line_id: {first_item.get('line_id', 'NO_LINE_ID')}")
        # ==================== END DEBUG ====================

        # 按页分组原始数据，计算每页内的索引
        from collections import defaultdict as ddict
        page_items = ddict(list)
        for idx, item in enumerate(gt_data):
            page = item.get("page", 0)
            # 转换 page 为整数（可能是字符串）
            page_int = int(page) if isinstance(page, str) else page
            page_items[page_int].append((idx, item))

        enriched_data = [None] * len(gt_data)  # 预分配，保持原始顺序

        for page_int, items in page_items.items():
            page_preds = doc_page_predictions.get(doc_name, {}).get(page_int, {})

            # 建立页内本地索引 -> 全局 line_id 的映射
            local_to_global_line_id = {}
            for local_idx, (global_idx, item) in enumerate(items):
                local_to_global_line_id[local_idx] = item.get("line_id", global_idx)

            for local_idx, (global_idx, item) in enumerate(items):
                enriched_item = item.copy()
                line_id = item.get("line_id", global_idx)

                # 使用 local_idx 查找预测结果（hrdoc.py 将 line_id 重映射为页内索引）
                if local_idx in page_preds:
                    pred_info = page_preds[local_idx]
                    enriched_item["class"] = pred_info["class"]
                    # 将 parent_id 从页内本地索引转换为全局 line_id
                    local_parent_id = pred_info["parent_id"]
                    if local_parent_id == -1:
                        enriched_item["parent_id"] = -1  # ROOT
                    else:
                        enriched_item["parent_id"] = local_to_global_line_id.get(local_parent_id, -1)
                    enriched_item["relation"] = pred_info["relation"]
                else:
                    from layoutlmft.data.labels import trans_class
                    enriched_item["class"] = trans_class(item.get("class", "paraline"))
                    enriched_item["parent_id"] = item.get("parent_id", -1)
                    enriched_item["relation"] = item.get("relation", "connect")

                # 添加 line_id 和 is_meta 字段（参考 HRDoc 格式）
                enriched_item["line_id"] = line_id
                enriched_item["is_meta"] = enriched_item["class"] not in struct_classes

                enriched_data[global_idx] = enriched_item

        # 统计预测匹配情况
        predicted_count = sum(1 for item in enriched_data if item and item.get("class") not in ["", None])
        logger.info(f"[DEBUG] Predicted/Total: {predicted_count}/{len(enriched_data)}")

        # ==================== DEBUG: 打印 enriched 数据信息 ====================
        logger.info(f"[DEBUG] Enriched records count: {len(enriched_data)}")
        if len(enriched_data) > 0:
            first_enriched = enriched_data[0]
            logger.info(f"[DEBUG] Enriched first item bbox: {first_enriched.get('box', first_enriched.get('bbox', 'NO_BBOX'))}")
            logger.info(f"[DEBUG] Enriched first item class: {first_enriched.get('class')}")
            logger.info(f"[DEBUG] Enriched first item parent_id: {first_enriched.get('parent_id')}")
            logger.info(f"[DEBUG] Enriched first item relation: {first_enriched.get('relation')}")
            # 对比 bbox 是否一致
            gt_bbox = gt_data[0].get('box', gt_data[0].get('bbox'))
            enriched_bbox = first_enriched.get('box', first_enriched.get('bbox'))
            if gt_bbox != enriched_bbox:
                logger.warning(f"[DEBUG] !!! BBOX MISMATCH !!! GT: {gt_bbox} vs Enriched: {enriched_bbox}")
            else:
                logger.info(f"[DEBUG] BBOX consistent: {gt_bbox}")
        logger.info(f"[DEBUG] Saving to: {enriched_path}")
        # ==================== END DEBUG ====================

        with open(enriched_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
        converted_count += 1

    logger.info(f"Saved {converted_count} enriched files to {enriched_dir}")
    logger.info(f"Run completed: {run_id}")

    return runs_dir


def run_evaluation(gt_folder: str, pred_folder: str, eval_type: str = "all"):
    """Run HRDoc evaluation using official tools."""
    logger.info("\n" + "=" * 60)
    logger.info("Running HRDoc Evaluation...")
    logger.info("=" * 60)

    hrdoc_utils_path = os.path.join(PROJECT_ROOT, "HRDoc", "utils")
    sys.path.insert(0, str(hrdoc_utils_path))

    if eval_type in ["all", "classify"]:
        try:
            from classify_eval import main as classify_eval_main
            original_argv = sys.argv
            sys.argv = ['classify_eval.py', '--gt_folder', gt_folder, '--pred_folder', pred_folder]
            logger.info("\n--- Classification Evaluation ---")
            classify_eval_main()
            sys.argv = original_argv
        except Exception as e:
            logger.error(f"Classification evaluation failed: {e}")

    if eval_type in ["all", "teds"]:
        try:
            from teds_eval import main as teds_eval_main
            original_argv = sys.argv
            sys.argv = ['teds_eval.py', '--gt_folder', gt_folder, '--pred_folder', pred_folder]
            logger.info("\n--- TEDS Evaluation ---")
            teds_eval_main()
            sys.argv = original_argv
        except Exception as e:
            logger.error(f"TEDS evaluation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Joint Model End-to-End Inference")
    parser.add_argument("--env", type=str, default=None, help="Environment config (dev/test)")
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh", "tender"])
    parser.add_argument("--exp", type=str, default=None, help="Experiment ID")
    parser.add_argument("--model_path", type=str, default=None, help="Joint model checkpoint path")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 10 samples")
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--eval_type", type=str, default="all", choices=["all", "classify", "teds"])

    args = parser.parse_args()

    config = None
    if args.env:
        config = load_config(args.env)
        if config and config.gpu.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    data_dir = args.data_dir or (get_data_dir(config, args.dataset) if config else None)
    if not data_dir:
        parser.error("Must specify --data_dir or --env")

    model_path = args.model_path or (get_latest_joint_checkpoint(config, args.exp, args.dataset) if config else None)
    if not model_path:
        parser.error("Must specify --model_path or --env with trained joint model")
    logger.info(f"Using model: {model_path}")

    max_test_samples = 10 if args.quick else args.max_test_samples

    runs_dir = run_inference(model_path, data_dir, None, config, max_test_samples)

    if runs_dir and not args.skip_eval:
        enriched_dir = os.path.join(runs_dir, "enriched")
        run_evaluation(os.path.join(data_dir, "test"), enriched_dir, args.eval_type)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Stage 1 Inference - Predict semantic class using trained LayoutXLM model

使用论文定义的 14 个语义类别（Line 级别标注，不使用 BIO）。
标签定义统一使用 layoutlmft.data.labels 模块。

Calls run_hrdoc.py with --do_predict, then converts output to HRDoc JSON format.

Usage:
    python run_stage1_infer.py --env test --dataset hrds
    python run_stage1_infer.py --env test --dataset hrdh
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import logging
from pathlib import Path
from collections import Counter

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "configs"))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "stage"))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "stage" / "util"))

# Import unified label definitions
from layoutlmft.data.labels import LABEL_LIST, id2label

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(env: str):
    """Load configuration from YAML file."""
    from config_loader import load_config as _load_config
    return _load_config(env)


def get_data_dir(config, dataset: str) -> str:
    """Get data directory for dataset."""
    return config.dataset.get_data_dir(dataset)


def get_latest_model(config, exp: str = None):
    """Auto-detect latest Stage 1 model (global, not dataset-specific)."""
    from checkpoint_utils import get_latest_checkpoint
    from experiment_manager import get_experiment_manager
    import glob

    exp_manager = get_experiment_manager(config)
    exp_dir = exp_manager.get_experiment_dir(exp)

    # Find all stage1_* directories in the experiment
    stage1_dirs = glob.glob(os.path.join(exp_dir, "stage1_*"))

    latest_model = None
    latest_mtime = 0

    for stage1_dir in stage1_dirs:
        checkpoint = get_latest_checkpoint(stage1_dir)
        if checkpoint and os.path.exists(checkpoint):
            mtime = os.path.getmtime(checkpoint)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_model = checkpoint

    # Fallback to legacy paths if no experiment model found
    if not latest_model:
        base_model_path = config.paths.stage1_model_path
        if base_model_path:
            for suffix in ["_hrds", "_hrdh", ""]:
                legacy_dir = f"{base_model_path}{suffix}"
                if os.path.exists(legacy_dir):
                    checkpoint = get_latest_checkpoint(legacy_dir)
                    if checkpoint:
                        mtime = os.path.getmtime(checkpoint)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_model = checkpoint

    return latest_model


def convert_predictions_to_json(predictions_file: str, data_dir: str, output_dir: str, model_path: str):
    """
    Convert test_predictions.txt to HRDoc JSON format.

    Key insight:
    - test_predictions.txt has one line per TOKENIZED CHUNK (not per original page)
    - Due to 512 token limit, one page may be split into multiple chunks (overflow)
    - We need to re-tokenize to get overflow_to_sample_mapping
    - Then aggregate predictions back to original pages and lines

    Args:
        predictions_file: Path to test_predictions.txt from run_hrdoc.py
        data_dir: Dataset directory (contains test/ folder with GT JSON)
        output_dir: Output directory for predicted JSON files
        model_path: Model path for loading tokenizer
    """
    logger.info("Converting predictions to JSON format...")

    # 使用统一的标签列表（从 labels.py 导入）
    # 模型直接输出论文14类（小写），无需额外映射
    logger.info(f"Using {len(LABEL_LIST)} labels: {LABEL_LIST}")

    # Read predictions (one line per TOKENIZED CHUNK, space-separated token labels)
    with open(predictions_file, 'r') as f:
        all_chunk_predictions = [line.strip().split() for line in f.readlines()]

    logger.info(f"Loaded {len(all_chunk_predictions)} chunk predictions")

    # Load test dataset to get document_name, page_number, line_ids mapping
    # Note: Environment variables (HRDOC_DATA_DIR, HF_HOME, HF_DATASETS_CACHE)
    # should be set by the caller (run_inference) before calling this function
    from datasets import load_dataset
    import layoutlmft.data.datasets.hrdoc

    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))
    test_dataset = datasets["test"]

    logger.info(f"Test dataset has {len(test_dataset)} original pages")

    # Load tokenizer to re-tokenize and get overflow_to_sample_mapping
    from layoutlmft.models.layoutxlm import LayoutXLMTokenizer, LayoutXLMTokenizerFast

    tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(model_path)
        logger.info("Using LayoutXLM fast tokenizer")
    else:
        tokenizer = LayoutXLMTokenizer.from_pretrained(model_path)
        logger.info("Using LayoutXLM slow tokenizer")

    # Re-tokenize test dataset to get overflow_to_sample_mapping
    # This mirrors the tokenization in run_hrdoc.py
    def tokenize_for_mapping(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        # Build mapping: chunk_idx -> (original_sample_idx, word_ids, line_ids)
        chunk_info = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            chunk_info.append({
                "org_sample_idx": org_batch_index,
                "word_ids": word_ids,
            })

        return {"chunk_info": [chunk_info]}  # Wrap in list for batched processing

    # Process in batches to get all chunk mappings
    logger.info("Re-tokenizing to get overflow mapping...")

    # Collect all chunk info
    all_chunk_info = []
    batch_size = 100

    for start_idx in range(0, len(test_dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(test_dataset))
        batch = test_dataset.select(range(start_idx, end_idx))

        # Tokenize batch
        tokenized = tokenizer(
            batch["tokens"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        # Extract chunk info
        for batch_index in range(len(tokenized["input_ids"])):
            org_batch_index = tokenized["overflow_to_sample_mapping"][batch_index]
            word_ids = tokenized.word_ids(batch_index=batch_index)
            all_chunk_info.append({
                "org_sample_idx": start_idx + org_batch_index,  # Global index
                "word_ids": word_ids,
            })

    logger.info(f"Total chunks after tokenization: {len(all_chunk_info)}")

    if len(all_chunk_info) != len(all_chunk_predictions):
        logger.warning(f"Chunk count mismatch! Tokenized: {len(all_chunk_info)}, Predictions: {len(all_chunk_predictions)}")

    # Clear and create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Organize predictions by document
    # {doc_name: {line_id: [labels]}} - collect all votes per line
    doc_line_votes = {}

    for chunk_idx, chunk_info in enumerate(all_chunk_info):
        if chunk_idx >= len(all_chunk_predictions):
            logger.warning(f"No prediction for chunk {chunk_idx}")
            continue

        org_sample_idx = chunk_info["org_sample_idx"]
        word_ids = chunk_info["word_ids"]

        # Get original sample info
        example = test_dataset[org_sample_idx]
        doc_name = example["document_name"]
        line_ids = example["line_ids"]  # word -> line_id mapping

        if doc_name not in doc_line_votes:
            doc_line_votes[doc_name] = {}

        # Get predictions for this chunk
        chunk_preds = all_chunk_predictions[chunk_idx]

        # Map token predictions to line predictions
        # IMPORTANT: chunk_preds only contains predictions for positions where label != -100
        # These are the first subword of each word (where word_idx changes)
        # We need to track pred_idx separately from token_idx
        pred_idx = 0
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                continue  # Skip special tokens (CLS, SEP, PAD)

            # Only the first occurrence of each word has a prediction
            if word_idx != prev_word_idx:
                if pred_idx >= len(chunk_preds):
                    break  # No more predictions for this chunk
                if word_idx >= len(line_ids):
                    pred_idx += 1
                    prev_word_idx = word_idx
                    continue

                line_id = line_ids[word_idx]
                label = chunk_preds[pred_idx]

                # 直接使用标签（无 BIO 前缀，论文14类）
                # 确保是小写格式
                label = label.lower()

                # 验证是否是有效标签
                if label not in LABEL_LIST:
                    logger.warning(f"Unknown label: {label}, skipping")
                    pred_idx += 1
                    prev_word_idx = word_idx
                    continue

                if line_id not in doc_line_votes[doc_name]:
                    doc_line_votes[doc_name][line_id] = []
                doc_line_votes[doc_name][line_id].append(label)

                pred_idx += 1

            prev_word_idx = word_idx

    # Aggregate votes to final predictions
    doc_predictions = {}  # {doc_name: {line_id: predicted_class}}

    for doc_name, line_votes in doc_line_votes.items():
        doc_predictions[doc_name] = {}
        for line_id, votes in line_votes.items():
            if votes:
                vote_counts = Counter(votes)
                predicted_class = vote_counts.most_common(1)[0][0]
                # 直接使用标签（已经是论文14类）
                doc_predictions[doc_name][line_id] = predicted_class

    # Write output JSON files
    test_dir = os.path.join(data_dir, "test")
    json_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.json')])

    converted_count = 0
    for json_file in json_files:
        doc_name = os.path.splitext(json_file)[0]
        gt_path = os.path.join(test_dir, json_file)
        pred_path = os.path.join(output_dir, json_file)

        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        # Create prediction data
        pred_data = []
        for item in gt_data:
            pred_item = item.copy()
            line_id = item.get("line_id")

            if doc_name in doc_predictions and line_id in doc_predictions[doc_name]:
                pred_item["class"] = doc_predictions[doc_name][line_id]
            else:
                # Keep original class if no prediction (shouldn't happen normally)
                # 使用 trans_class 转换原始标签
                from layoutlmft.data.labels import trans_class
                gt_class = item.get("class", "paraline")
                pred_item["class"] = trans_class(gt_class)

            pred_data.append(pred_item)

        with open(pred_path, 'w', encoding='utf-8') as f:
            json.dump(pred_data, f, indent=2, ensure_ascii=False)

        converted_count += 1

    logger.info(f"Converted {converted_count} files to {output_dir}")


def run_inference(model_path: str, data_dir: str, output_dir: str, config, max_test_samples: int = None):
    """
    Run Stage 1 inference using run_hrdoc.py --do_predict.
    """
    logger.info("=" * 60)
    logger.info("Stage 1 Inference")
    logger.info("=" * 60)
    logger.info(f"Model:      {model_path}")
    logger.info(f"Data Dir:   {data_dir}")
    logger.info(f"Output Dir: {output_dir}")
    if max_test_samples:
        logger.info(f"Max Samples: {max_test_samples}")

    # Create temp output directory for run_hrdoc.py
    temp_output = os.path.join(os.path.dirname(output_dir), "temp_predict")
    os.makedirs(temp_output, exist_ok=True)

    # Set environment variables
    env = os.environ.copy()
    env["HRDOC_DATA_DIR"] = data_dir
    # Set HuggingFace cache (same as training scripts, critical for avoiding re-download)
    if config and hasattr(config, 'paths') and hasattr(config.paths, 'hf_cache_dir'):
        if config.paths.hf_cache_dir:
            env["HF_HOME"] = config.paths.hf_cache_dir
            env["TRANSFORMERS_CACHE"] = config.paths.hf_cache_dir
            env["HF_DATASETS_CACHE"] = os.path.join(config.paths.hf_cache_dir, "datasets")
            logger.info(f"HF Cache: {config.paths.hf_cache_dir}")
    # Set GPU from config (same as training scripts)
    if config and hasattr(config, 'gpu') and hasattr(config.gpu, 'cuda_visible_devices'):
        if config.gpu.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices
            logger.info(f"GPU Config: CUDA_VISIBLE_DEVICES={config.gpu.cuda_visible_devices}")
    # Add project root to PYTHONPATH for subprocess
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{pythonpath}"
    else:
        env["PYTHONPATH"] = str(PROJECT_ROOT)

    # Build command to call run_hrdoc.py
    run_hrdoc_script = os.path.join(PROJECT_ROOT, "examples", "stage", "run_hrdoc.py")

    cmd = [
        sys.executable, run_hrdoc_script,
        "--model_name_or_path", model_path,
        "--output_dir", temp_output,
        "--do_predict",
        "--per_device_eval_batch_size", "1",
        "--report_to", "none",  # Disable TensorBoard to avoid distutils.version issue
    ]

    # Add max_test_samples if specified
    if max_test_samples:
        cmd.extend(["--max_test_samples", str(max_test_samples)])

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")
    logger.info(f"HRDOC_DATA_DIR: {env['HRDOC_DATA_DIR']}")

    # Run prediction
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)

    if result.returncode != 0:
        logger.error("Prediction failed!")
        return None

    # Convert predictions to JSON format
    predictions_file = os.path.join(temp_output, "test_predictions.txt")
    if os.path.exists(predictions_file):
        # Apply environment variables to current process for convert_predictions_to_json
        os.environ["HRDOC_DATA_DIR"] = data_dir
        if config and hasattr(config, 'paths') and hasattr(config.paths, 'hf_cache_dir'):
            if config.paths.hf_cache_dir:
                os.environ["HF_HOME"] = config.paths.hf_cache_dir
                os.environ["HF_DATASETS_CACHE"] = os.path.join(config.paths.hf_cache_dir, "datasets")

        convert_predictions_to_json(predictions_file, data_dir, output_dir, model_path)
    else:
        logger.error(f"Predictions file not found: {predictions_file}")
        return None

    # Cleanup temp directory (commented for debugging)
    # shutil.rmtree(temp_output, ignore_errors=True)
    logger.info(f"Temp output kept at: {temp_output}")

    logger.info(f"Inference complete. Predictions saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Inference")

    parser.add_argument("--env", type=str, default=None,
                        help="Environment config (dev/test)")
    parser.add_argument("--dataset", type=str, default="hrds",
                        choices=["hrds", "hrdh", "tender"],
                        help="Dataset name")
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment ID")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model checkpoint path (auto-detect if not specified)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only process 10 test samples")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Max number of test samples to process")

    args = parser.parse_args()

    # Load config
    config = None
    if args.env:
        config = load_config(args.env)

    # Resolve data_dir
    data_dir = args.data_dir
    if not data_dir and config:
        data_dir = get_data_dir(config, args.dataset)

    if not data_dir:
        parser.error("Must specify --data_dir or --env")

    # Resolve model_path (global latest, not dataset-specific)
    model_path = args.model_path
    if not model_path and config:
        model_path = get_latest_model(config, args.exp)
        if model_path:
            logger.info(f"Auto-detected model (global latest): {model_path}")

    if not model_path:
        parser.error("Must specify --model_path or --env with trained model")

    # Determine output directory
    infer_folder = "test_infer_stage1"
    if config and hasattr(config, 'evaluation') and hasattr(config.evaluation, 'stage1_infer_folder'):
        infer_folder = config.evaluation.stage1_infer_folder
    output_dir = os.path.join(data_dir, infer_folder)

    # Determine max_test_samples
    max_test_samples = args.max_test_samples
    if args.quick:
        max_test_samples = 10
        logger.info("Quick mode enabled: processing only 10 samples")

    # Run inference
    result = run_inference(model_path, data_dir, output_dir, config, max_test_samples)

    if result is None:
        logger.error("Inference failed, skipping evaluation")
        return

    # Auto-run HRDoc evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Running HRDoc Classification Evaluation...")
    logger.info("=" * 60)

    gt_folder = os.path.join(data_dir, "test")
    pred_folder = output_dir

    # Import and run HRDoc's classify_eval
    try:
        hrdoc_utils_path = os.path.join(PROJECT_ROOT, "HRDoc", "utils")
        sys.path.insert(0, str(hrdoc_utils_path))
        from classify_eval import main as hrdoc_classify_eval

        # Set sys.argv for classify_eval
        original_argv = sys.argv
        sys.argv = ['classify_eval.py', '--gt_folder', gt_folder, '--pred_folder', pred_folder]

        logger.info(f"GT Folder:   {gt_folder}")
        logger.info(f"Pred Folder: {pred_folder}")

        hrdoc_classify_eval()

        sys.argv = original_argv
    except Exception as e:
        logger.error(f"HRDoc evaluation failed: {e}")
        logger.info("\nYou can run evaluation manually:")
        logger.info(f"  python examples/evaluate/run_classify_eval.py --env {args.env} --dataset {args.dataset}")


if __name__ == "__main__":
    main()

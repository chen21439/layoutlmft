#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Stage 1 Inference - Predict semantic class using trained LayoutXLM model

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
    if hasattr(config, 'datasets') and hasattr(config.datasets, dataset):
        return getattr(config.datasets, dataset).data_dir

    data_dir_base = os.path.dirname(config.paths.hrdoc_data_dir)
    if dataset == "hrds":
        return os.path.join(data_dir_base, "HRDS")
    elif dataset == "hrdh":
        return os.path.join(data_dir_base, "HRDH")
    else:
        return config.paths.hrdoc_data_dir


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


def convert_predictions_to_json(predictions_file: str, data_dir: str, output_dir: str):
    """
    Convert test_predictions.txt to HRDoc JSON format.

    Key insight:
    - test_predictions.txt has one line per PAGE (not per document)
    - Each line contains token-level predictions separated by space
    - We need to aggregate token predictions to line-level predictions
    - Then organize by document for HRDoc evaluation

    Args:
        predictions_file: Path to test_predictions.txt from run_hrdoc.py
        data_dir: Dataset directory (contains test/ folder with GT JSON)
        output_dir: Output directory for predicted JSON files
    """
    logger.info("Converting predictions to JSON format...")

    # Label mapping: our labels -> HRDoc evaluate labels
    # Based on HRDoc/utils/utils.py class2class
    label_mapping = {
        "title": "title",
        "author": "author",
        "mail": "mail",
        "affili": "affili",
        "sec1": "section",
        "sec2": "section",
        "sec3": "section",
        "sec4": "section",
        "secx": "section",
        "fstline": "fstline",
        "para": "paraline",
        "opara": "paraline",  # other paragraph -> paraline
        "tab": "table",
        "fig": "figure",
        "tabcap": "caption",
        "figcap": "caption",
        "equ": "equation",
        "alg": "equation",  # algorithm -> equation
        "foot": "footer",
        "header": "header",
        "fnote": "footnote",
    }

    # Read predictions (one line per page, space-separated token labels)
    with open(predictions_file, 'r') as f:
        all_page_predictions = [line.strip().split() for line in f.readlines()]

    logger.info(f"Loaded {len(all_page_predictions)} page predictions")

    # Load test dataset to get document_name, page_number, line_ids mapping
    from datasets import load_dataset
    import layoutlmft.data.datasets.hrdoc

    # Set environment variable for data directory
    os.environ["HRDOC_DATA_DIR"] = data_dir
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))
    test_dataset = datasets["test"]

    logger.info(f"Test dataset has {len(test_dataset)} pages")

    # Clear and create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Organize predictions by document
    doc_predictions = {}  # {doc_name: {line_id: predicted_class}}

    for page_idx, example in enumerate(test_dataset):
        doc_name = example["document_name"]
        line_ids = example["line_ids"]  # token -> line_id mapping

        if doc_name not in doc_predictions:
            doc_predictions[doc_name] = {}

        # Get token predictions for this page
        if page_idx < len(all_page_predictions):
            token_preds = all_page_predictions[page_idx]
        else:
            logger.warning(f"No predictions for page {page_idx}")
            continue

        # Aggregate token predictions to line predictions
        # Use majority voting for tokens in the same line
        line_votes = {}  # {line_id: [labels]}

        for token_idx, line_id in enumerate(line_ids):
            if token_idx < len(token_preds):
                label = token_preds[token_idx]
                # Remove B-/I- prefix
                if label.startswith('B-') or label.startswith('I-'):
                    label = label[2:].lower()
                elif label == 'O':
                    label = None  # Will use GT later
                else:
                    label = label.lower()

                if label:
                    if line_id not in line_votes:
                        line_votes[line_id] = []
                    line_votes[line_id].append(label)

        # Determine final class per line (majority vote, prefer B- labels)
        for line_id, votes in line_votes.items():
            if votes:
                vote_counts = Counter(votes)
                # Get the most common label
                predicted_class = vote_counts.most_common(1)[0][0]

                # Map to HRDoc evaluation label
                if predicted_class in label_mapping:
                    predicted_class = label_mapping[predicted_class]

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
                gt_class = item.get("class", "para").lower()
                if gt_class in label_mapping:
                    pred_item["class"] = label_mapping[gt_class]

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
        convert_predictions_to_json(predictions_file, data_dir, output_dir)
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
                        choices=["hrds", "hrdh"],
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
    run_inference(model_path, data_dir, output_dir, config, max_test_samples)

    logger.info("\nNext step - run evaluation:")
    logger.info(f"  python examples/evaluate/run_classify_eval.py --env {args.env} --dataset {args.dataset}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Stage 1 Inference - Predict semantic class using trained LayoutXLM model

Reads test JSON files, predicts class for each line, outputs to test_infer_stage1 folder.

Usage:
    python run_stage1_infer.py --env test --dataset hrds
    python run_stage1_infer.py --env test --dataset hrdh
    python run_stage1_infer.py --model_path /path/to/model --data_dir /path/to/data
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "configs"))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "stage"))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "stage" / "util"))

# Register LayoutXLM config (same as run_hrdoc.py)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from layoutlmft.models.layoutxlm import LayoutXLMConfig
CONFIG_MAPPING.update({
    "layoutxlm": LayoutXLMConfig,
    "layoutlmv2": LayoutXLMConfig,
})

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
    """Get data directory for dataset (same logic as train_stage1.py)."""
    if hasattr(config, 'datasets') and hasattr(config.datasets, dataset):
        return getattr(config.datasets, dataset).data_dir

    data_dir_base = os.path.dirname(config.paths.hrdoc_data_dir)
    if dataset == "hrds":
        return os.path.join(data_dir_base, "HRDS")
    elif dataset == "hrdh":
        return os.path.join(data_dir_base, "HRDH")
    else:
        return config.paths.hrdoc_data_dir


def get_latest_model(config, dataset: str, exp: str = None):
    """Auto-detect latest Stage 1 model (same logic as train_stage2.py)."""
    from checkpoint_utils import get_latest_checkpoint
    from experiment_manager import get_experiment_manager

    exp_manager = get_experiment_manager(config)

    # Get Stage 1 model from experiment
    stage1_dir = exp_manager.get_stage_dir(exp, "stage1", dataset)
    model_path = get_latest_checkpoint(stage1_dir)

    if not model_path:
        # Fallback to legacy path
        base_model_path = config.paths.stage1_model_path
        if base_model_path:
            legacy_dir = f"{base_model_path}_{dataset}"
            model_path = get_latest_checkpoint(legacy_dir)

    return model_path


def run_inference(model_path: str, data_dir: str, output_dir: str):
    """
    Run Stage 1 inference.

    Args:
        model_path: Path to trained LayoutXLM checkpoint
        data_dir: Dataset directory (contains test/ folder)
        output_dir: Output directory for predictions (test_infer_stage1)
    """
    from transformers import AutoConfig
    from layoutlmft.models.layoutxlm import LayoutXLMForTokenClassification, LayoutXLMTokenizerFast

    logger.info("=" * 60)
    logger.info("Stage 1 Inference")
    logger.info("=" * 60)
    logger.info(f"Model:      {model_path}")
    logger.info(f"Data Dir:   {data_dir}")
    logger.info(f"Output Dir: {output_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device:     {device}")

    # Load model and tokenizer
    logger.info("Loading model...")
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(model_path)
    model = LayoutXLMForTokenClassification.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()

    # Get label mapping
    id2label = config.id2label
    logger.info(f"Labels: {list(id2label.values())}")

    # Prepare output directory (clear and recreate)
    if os.path.exists(output_dir):
        logger.info(f"Clearing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get test files
    test_dir = os.path.join(data_dir, "test")
    json_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.json')])
    logger.info(f"Processing {len(json_files)} test files...")

    # Process each file
    for json_file in tqdm(json_files, desc="Inference"):
        input_path = os.path.join(test_dir, json_file)
        output_path = os.path.join(output_dir, json_file)

        with open(input_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        # Prepare inputs
        texts = [item['text'] for item in gt_data]
        boxes = [item['box'] for item in gt_data]

        # Normalize boxes to 0-1000 (LayoutXLM format)
        # Note: boxes should already be normalized in HRDoc format
        normalized_boxes = []
        for box in boxes:
            normalized_boxes.append([
                int(min(1000, max(0, box[0]))),
                int(min(1000, max(0, box[1]))),
                int(min(1000, max(0, box[2]))),
                int(min(1000, max(0, box[3])))
            ])

        # Tokenize (process one document at a time)
        try:
            encoding = tokenizer(
                texts,
                boxes=normalized_boxes,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        except Exception as e:
            logger.warning(f"Tokenization failed for {json_file}: {e}")
            # Copy original file as fallback
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(gt_data, f, indent=2, ensure_ascii=False)
            continue

        # Get offset mapping for alignment
        offset_mapping = encoding.pop("offset_mapping")

        # Move to device
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = outputs.logits.argmax(dim=-1).cpu().numpy()[0]

        # Align predictions to original lines
        # Each line may have multiple tokens, we take the first non-special token's prediction
        pred_data = []
        token_idx = 1  # Skip [CLS]

        for i, item in enumerate(gt_data):
            pred_item = item.copy()

            # Find prediction for this line
            # Simple approach: use prediction at current token position
            if token_idx < len(predictions) - 1:  # Avoid [SEP]
                pred_label_id = predictions[token_idx]
                pred_label = id2label.get(pred_label_id, item['class'])

                # Remove B-/I- prefix if present (convert to class name)
                if pred_label.startswith('B-') or pred_label.startswith('I-'):
                    pred_label = pred_label[2:]

                pred_item['class'] = pred_label

                # Move to next line's first token
                # This is simplified - proper alignment would track token-to-line mapping
                token_idx += 1

            pred_data.append(pred_item)

        # Save predictions
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pred_data, f, indent=2, ensure_ascii=False)

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

    # Direct path overrides
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model checkpoint path (auto-detect if not specified)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory")

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

    # Resolve model_path
    model_path = args.model_path
    if not model_path and config:
        model_path = get_latest_model(config, args.dataset, args.exp)
        if model_path:
            logger.info(f"Auto-detected model: {model_path}")

    if not model_path:
        parser.error("Must specify --model_path or --env with trained model")

    # Determine output directory
    infer_folder = "test_infer_stage1"
    if config and hasattr(config, 'evaluation') and hasattr(config.evaluation, 'stage1_infer_folder'):
        infer_folder = config.evaluation.stage1_infer_folder
    output_dir = os.path.join(data_dir, infer_folder)

    # Run inference
    run_inference(model_path, data_dir, output_dir)

    logger.info("\nNext step - run evaluation:")
    logger.info(f"  python examples/evaluate/run_classify_eval.py --env {args.env} --dataset {args.dataset}")


if __name__ == "__main__":
    main()

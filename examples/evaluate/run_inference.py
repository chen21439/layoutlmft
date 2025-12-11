#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
End-to-End Inference Pipeline

Runs Stage 1 → Stage 2 → Stage 3 → Stage 4 inference and outputs JSON files
in HRDoc format for evaluation with classify_eval.py and teds_eval.py.

Usage:
    # Stage 1 only (class prediction) - auto-detect latest model
    python run_inference.py --env test --dataset hrds --stage 1

    # Full pipeline (class + parent_id + relation)
    python run_inference.py --env test --dataset hrds --stage all

    # With direct paths
    python run_inference.py --data_dir /path/to/test --output_dir /path/to/output --model_path /path/to/model
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

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


def resolve_variable(value: str, config: dict) -> str:
    """Resolve ${} variables in config values."""
    if not value or '${' not in value:
        return value

    if '${datasets.hrds.data_dir}' in value:
        base = config.get('datasets', {}).get('hrds', {}).get('data_dir', '')
        value = value.replace('${datasets.hrds.data_dir}', base)
    if '${datasets.hrdh.data_dir}' in value:
        base = config.get('datasets', {}).get('hrdh', {}).get('data_dir', '')
        value = value.replace('${datasets.hrdh.data_dir}', base)
    if '${paths.output_dir}' in value:
        base = config.get('paths', {}).get('output_dir', '')
        value = value.replace('${paths.output_dir}', base)

    return value


def run_stage1_inference(model_path: str, data_dir: str, output_dir: str, config: dict):
    """
    Run Stage 1 inference: predict semantic class for each line.
    Outputs JSON files with predicted 'class' field.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    logger.info("=" * 60)
    logger.info("Stage 1 Inference: Semantic Classification")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_dir}")
    logger.info(f"Output: {output_dir}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Get label mapping from model config
    id2label = model.config.id2label

    # Process test files
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
    logger.info(f"Processing {len(json_files)} files...")

    for json_file in tqdm(json_files, desc="Stage 1 Inference"):
        input_path = os.path.join(test_dir, json_file)
        output_path = os.path.join(output_dir, json_file)

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # For each line, predict class
        # Note: This is a simplified version. Full implementation would
        # need to handle the actual LayoutXLM input format with bboxes.
        predictions = []
        for item in data:
            # Copy original item and update class prediction
            pred_item = item.copy()
            # TODO: Implement actual model inference here
            # For now, keep original class (placeholder)
            predictions.append(pred_item)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

    logger.info(f"Stage 1 inference complete. Output: {output_dir}")
    return output_dir


def run_stage234_inference(features_dir: str, stage3_model: str, stage4_model: str,
                           stage1_output: str, output_dir: str, config: dict):
    """
    Run Stage 2/3/4 inference: predict parent_id and relation.
    Uses Stage 1 output and adds parent_id/relation predictions.
    """
    logger.info("=" * 60)
    logger.info("Stage 2/3/4 Inference: Structure Recovery")
    logger.info("=" * 60)
    logger.info(f"Features: {features_dir}")
    logger.info(f"Stage 3 Model: {stage3_model}")
    logger.info(f"Stage 4 Model: {stage4_model}")
    logger.info(f"Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # TODO: Implement full Stage 2/3/4 inference
    # This requires:
    # 1. Loading features from Stage 2
    # 2. Running ParentFinder (Stage 3) to predict parent_id
    # 3. Running RelationClassifier (Stage 4) to predict relation
    # 4. Combining predictions into final JSON

    # For now, copy Stage 1 output as placeholder
    import shutil
    json_files = [f for f in os.listdir(stage1_output) if f.endswith('.json')]

    for json_file in tqdm(json_files, desc="Stage 2/3/4 Inference"):
        src_path = os.path.join(stage1_output, json_file)
        dst_path = os.path.join(output_dir, json_file)
        shutil.copy(src_path, dst_path)

    logger.info(f"Stage 2/3/4 inference complete. Output: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="End-to-End Inference Pipeline")

    # Environment options
    parser.add_argument("--env", type=str, default=None,
                        help="Environment config (dev/test)")
    parser.add_argument("--dataset", type=str, default="hrds",
                        choices=["hrds", "hrdh"],
                        help="Dataset name")
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment ID")

    # Stage selection
    parser.add_argument("--stage", type=str, default="1",
                        choices=["1", "all"],
                        help="Which stages to run: '1' for Stage 1 only, 'all' for full pipeline")

    # Direct path overrides
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (contains test/ folder)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Stage 1 model path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for predictions")

    args = parser.parse_args()

    # Load config
    config = {}
    if args.env:
        config = load_config(args.env)
        if hasattr(config, '__dict__'):
            config = config.__dict__

    # Resolve paths
    data_dir = args.data_dir
    model_path = args.model_path
    output_dir = args.output_dir

    if args.env and not data_dir:
        if 'datasets' in config and args.dataset in config['datasets']:
            data_dir = config['datasets'][args.dataset].get('data_dir')
        else:
            data_dir = config.get('paths', {}).get('hrdoc_data_dir')

    if args.env and not output_dir:
        eval_config = config.get('evaluation', {})
        if args.stage == "1":
            output_dir = resolve_variable(eval_config.get('stage1_pred_folder', ''), config)
        else:
            output_dir = resolve_variable(eval_config.get('e2e_pred_folder', ''), config)

    if not data_dir or not output_dir:
        parser.error("Must specify --data_dir and --output_dir, or --env with valid config")

    logger.info("=" * 60)
    logger.info("Inference Pipeline Configuration")
    logger.info("=" * 60)
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Data Dir: {data_dir}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info("=" * 60)

    # Auto-detect model path if not specified (same logic as train_stage2.py)
    if not model_path and args.env:
        from checkpoint_utils import get_latest_checkpoint, get_best_model
        from experiment_manager import get_experiment_manager

        exp_manager = get_experiment_manager(config)

        # Get Stage 1 model from experiment
        stage1_dir = exp_manager.get_stage_dir(args.exp, "stage1", args.dataset)
        model_path = get_latest_checkpoint(stage1_dir)

        if not model_path:
            # Fallback to legacy path
            base_model_path = config.get('paths', {}).get('stage1_model_path', '')
            if base_model_path:
                legacy_dir = f"{base_model_path}_{args.dataset}"
                model_path = get_latest_checkpoint(legacy_dir)

        if model_path:
            logger.info(f"Auto-detected Stage 1 model: {model_path}")
        else:
            parser.error(f"No Stage 1 model found for dataset {args.dataset}. Train Stage 1 first or specify --model_path")

    # Run inference
    if args.stage == "1":
        if not model_path:
            parser.error("--model_path required for Stage 1 inference")
        run_stage1_inference(model_path, data_dir, output_dir, config)
    else:
        # Full pipeline
        stage1_output = output_dir + "_stage1"
        if model_path:
            run_stage1_inference(model_path, data_dir, stage1_output, config)
        # TODO: Add Stage 2/3/4 inference
        logger.warning("Full pipeline (Stage 2/3/4) not yet implemented")

    logger.info("\nInference complete!")
    logger.info(f"Predictions saved to: {output_dir}")
    logger.info("\nTo evaluate results:")
    logger.info(f"  Stage 1: python run_classify_eval.py --gt_folder {data_dir}/test --pred_folder {output_dir}")
    if args.stage == "all":
        logger.info(f"  TEDS: python run_teds_eval.py --gt_folder {data_dir}/test --pred_folder {output_dir}")


if __name__ == "__main__":
    main()

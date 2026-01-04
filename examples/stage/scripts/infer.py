#!/usr/bin/env python
# coding=utf-8
"""
Inference Script - Direct JSON Loading

Features:
1. Load trained model from checkpoint
2. Load data directly from JSON files (no HuggingFace Datasets)
3. Run inference and save predictions

Usage:
    python examples/stage/scripts/infer.py \
        --data_dir /path/to/data \
        --checkpoint /path/to/checkpoint \
        --output_dir /path/to/output

Data directory structure:
    data_dir/
    |-- test/           # JSON files
    |   |-- doc1.json
    |   +-- doc2.json
    +-- images/         # Image files
        |-- doc1/
        |   |-- 0.png
        |   +-- 1.png
        +-- doc2/
            +-- 0.png
"""

import os
import sys
import argparse
import logging

# Add project paths
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)
STAGE_ROOT = os.path.join(PROJECT_ROOT, "examples", "stage")
sys.path.insert(0, STAGE_ROOT)

from configs.config_loader import load_config

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Script")

    parser.add_argument("--env", type=str, default="test",
                        help="Environment: dev or test")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory (must have test/ and images/ subdirs)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for predictions")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples (-1 for all)")

    # P0 约束解码参数
    parser.add_argument("--use_parent_constraint", action="store_true",
                        help="Enable M_cp parent constraint filtering (P0)")
    parser.add_argument("--use_relation_gating", action="store_true",
                        help="Enable label-pair relation gating (P0)")
    parser.add_argument("--mcp_threshold", type=float, default=0.02,
                        help="M_cp filtering threshold")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Only enable hard filtering when cls confidence >= this value")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # Load config
    config = load_config(args.env)
    config = config.get_effective_config()

    # Set GPU
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model
    logger.info("Loading model...")
    from models.build import load_joint_model
    model, tokenizer = load_joint_model(args.checkpoint, device, config)

    # 2. Load data using InferenceDataLoader
    logger.info(f"Loading data from: {args.data_dir}")
    from data import InferenceDataLoader
    from joint_data_collator import HRDocDocumentLevelCollator

    data_loader = InferenceDataLoader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=512,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )

    eval_dataset = data_loader.load()
    if not eval_dataset:
        logger.error("No data loaded!")
        sys.exit(1)

    data_collator = HRDocDocumentLevelCollator(
        tokenizer=tokenizer,
        max_length=512,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
    )

    logger.info(f"Dataset loaded: {len(eval_dataset)} samples")

    # 3. Run inference
    logger.info("Running inference...")
    from engines.predictor import Predictor

    # P0: 打印约束解码配置
    if args.use_parent_constraint or args.use_relation_gating:
        logger.info("P0 Constrained Decoding enabled:")
        logger.info(f"  use_parent_constraint: {args.use_parent_constraint}")
        logger.info(f"  use_relation_gating: {args.use_relation_gating}")
        logger.info(f"  mcp_threshold: {args.mcp_threshold}")
        logger.info(f"  confidence_threshold: {args.confidence_threshold}")

    predictor = Predictor(
        model,
        device,
        use_parent_constraint=args.use_parent_constraint,
        use_relation_gating=args.use_relation_gating,
        mcp_threshold=args.mcp_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    predictor.predict_and_save(
        dataloader=eval_dataloader,
        output_dir=args.output_dir,
    )

    logger.info(f"Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

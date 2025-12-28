#!/usr/bin/env python
# coding=utf-8
"""
推理脚本 - 极薄入口

功能：
1. 加载训练好的模型
2. 对输入数据进行推理
3. 保存预测结果

Usage:
    python examples/stage/scripts/infer.py \
        --checkpoint /path/to/checkpoint \
        --input_dir /path/to/input \
        --output_dir /path/to/output
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
EXAMPLES_ROOT = os.path.dirname(STAGE_ROOT)
sys.path.insert(0, EXAMPLES_ROOT)

from configs.config_loader import load_config

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Script")

    parser.add_argument("--env", type=str, default="test",
                        help="Environment: dev or test")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint directory")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (hrds, hrdh, tender) - uses config file paths")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Custom data directory (must have test/ and images/ subdirs)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for predictions")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples (-1 for all)")

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

    # 2. Load data and run inference
    from engines.predictor import Predictor
    predictor = Predictor(model, device)

    # 确定数据目录
    if args.dataset:
        data_dir = config.dataset.get_data_dir(args.dataset)
        dataset_name = args.dataset
        logger.info(f"Using dataset '{args.dataset}' from config: {data_dir}")
    elif args.data_dir:
        data_dir = args.data_dir
        dataset_name = "infer"
        logger.info(f"Using custom data directory: {data_dir}")
    else:
        logger.error("Please specify --dataset or --data_dir")
        sys.exit(1)

    # 统一的数据加载逻辑
    from data import HRDocDataLoader, HRDocDataLoaderConfig
    from joint_data_collator import HRDocDocumentLevelCollator

    os.environ["HRDOC_DATA_DIR"] = data_dir

    loader_config = HRDocDataLoaderConfig(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_length=512,
        preprocessing_num_workers=1,
        max_val_samples=args.max_samples if args.max_samples > 0 else None,
        document_level=True,
    )

    data_loader = HRDocDataLoader(
        tokenizer=tokenizer,
        config=loader_config,
        include_line_info=True,
    )
    data_loader.load_raw_datasets()
    tokenized_datasets = data_loader.prepare_datasets()

    eval_dataset = tokenized_datasets.get("test") or tokenized_datasets.get("validation")
    if eval_dataset is None:
        logger.error("No evaluation dataset found!")
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

    # Run inference
    logger.info("Running inference...")
    predictor.predict_and_save(
        dataloader=eval_dataloader,
        output_dir=args.output_dir,
    )

    logger.info(f"Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

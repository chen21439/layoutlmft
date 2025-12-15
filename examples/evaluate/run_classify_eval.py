#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Stage 1 Evaluation - Wrapper for HRDoc classify_eval.py

Usage:
    python run_classify_eval.py --env test --dataset hrds
    python run_classify_eval.py --env test --dataset hrdh
    python run_classify_eval.py --gt_folder /path/to/gt --pred_folder /path/to/pred
"""

import os
import sys
import argparse
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "configs"))
sys.path.insert(0, str(PROJECT_ROOT / "HRDoc" / "utils"))

# Import HRDoc evaluation module directly
from classify_eval import main as hrdoc_classify_eval


def load_config(env: str):
    """Load configuration from YAML file."""
    from config_loader import load_config as _load_config
    return _load_config(env)


def get_data_dir(config, dataset: str) -> str:
    """Get data directory for dataset."""
    return config.dataset.get_data_dir(dataset)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Classification Evaluation (HRDoc)")

    parser.add_argument("--env", type=str, default=None,
                        help="Environment config (dev/test)")
    parser.add_argument("--dataset", type=str, default="hrds",
                        choices=["hrds", "hrdh", "tender"],
                        help="Dataset name (hrds/hrdh/tender)")
    parser.add_argument("--gt_folder", type=str, default=None,
                        help="Ground truth folder path (overrides config)")
    parser.add_argument("--pred_folder", type=str, default=None,
                        help="Predictions folder path (overrides config)")

    args = parser.parse_args()

    gt_folder = args.gt_folder
    pred_folder = args.pred_folder

    # Load from config if env specified
    if args.env and (not gt_folder or not pred_folder):
        config = load_config(args.env)
        data_dir = get_data_dir(config, args.dataset)

        if not gt_folder:
            gt_folder = os.path.join(data_dir, "test")

        if not pred_folder:
            infer_folder = "test_infer_stage1"
            if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'stage1_infer_folder'):
                infer_folder = config.evaluation.stage1_infer_folder
            pred_folder = os.path.join(data_dir, infer_folder)

    if not gt_folder or not pred_folder:
        parser.error("Must specify --gt_folder and --pred_folder, or --env with --dataset")

    print("=" * 60)
    print("Stage 1 Classification Evaluation")
    print("=" * 60)
    print(f"GT Folder:   {gt_folder}")
    print(f"Pred Folder: {pred_folder}")
    print("=" * 60)

    # Check folders exist
    if not os.path.exists(gt_folder):
        print(f"Error: GT folder not found: {gt_folder}")
        sys.exit(1)
    if not os.path.exists(pred_folder):
        print(f"Error: Pred folder not found: {pred_folder}")
        print(f"Run inference first to generate predictions.")
        sys.exit(1)

    # Call HRDoc's classify_eval directly
    sys.argv = ['classify_eval.py', '--gt_folder', gt_folder, '--pred_folder', pred_folder]
    hrdoc_classify_eval()


if __name__ == "__main__":
    main()

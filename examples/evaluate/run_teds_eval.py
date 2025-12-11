#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
End-to-End Evaluation - Wrapper for HRDoc teds_eval.py

Evaluates tree structure using Tree Edit Distance Similarity (TEDS).

Usage:
    python run_teds_eval.py --env test --dataset hrds
    python run_teds_eval.py --env test --dataset hrdh
    python run_teds_eval.py --gt_folder /path/to/gt --pred_folder /path/to/pred
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
from teds_eval import main as hrdoc_teds_eval


def load_config(env: str):
    """Load configuration from YAML file."""
    from config_loader import load_config as _load_config
    return _load_config(env)


def get_data_dir(config, dataset: str) -> str:
    """Get data directory for dataset (same logic as train_stage1.py)."""
    # Check if datasets config exists
    if hasattr(config, 'datasets') and hasattr(config.datasets, dataset):
        return getattr(config.datasets, dataset).data_dir

    # Fallback to legacy path
    data_dir_base = os.path.dirname(config.paths.hrdoc_data_dir)
    if dataset == "hrds":
        return os.path.join(data_dir_base, "HRDS")
    elif dataset == "hrdh":
        return os.path.join(data_dir_base, "HRDH")
    else:
        return config.paths.hrdoc_data_dir


def main():
    parser = argparse.ArgumentParser(description="End-to-End TEDS Evaluation (HRDoc)")

    parser.add_argument("--env", type=str, default=None,
                        help="Environment config (dev/test)")
    parser.add_argument("--dataset", type=str, default="hrds",
                        choices=["hrds", "hrdh"],
                        help="Dataset name (hrds/hrdh)")
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
            # Get infer folder name from config
            infer_folder = "test_infer_e2e"
            if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'e2e_infer_folder'):
                infer_folder = config.evaluation.e2e_infer_folder
            pred_folder = os.path.join(data_dir, infer_folder)

    if not gt_folder or not pred_folder:
        parser.error("Must specify --gt_folder and --pred_folder, or --env with --dataset")

    print("=" * 60)
    print("End-to-End TEDS Evaluation")
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
        print(f"Run end-to-end inference first to generate predictions.")
        sys.exit(1)

    # Call HRDoc's teds_eval directly
    sys.argv = ['teds_eval.py', '--gt_folder', gt_folder, '--pred_folder', pred_folder]
    hrdoc_teds_eval()


if __name__ == "__main__":
    main()

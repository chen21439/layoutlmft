#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Stage 1 Evaluation - Wrapper for HRDoc classify_eval.py

Usage:
    python run_classify_eval.py --env dev
    python run_classify_eval.py --env test --dataset hrds
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


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Classification Evaluation (HRDoc)")

    parser.add_argument("--env", type=str, default=None,
                        help="Environment config (dev/test)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (hrds/hrdh)")
    parser.add_argument("--gt_folder", type=str, default=None,
                        help="Ground truth folder path")
    parser.add_argument("--pred_folder", type=str, default=None,
                        help="Predictions folder path")

    args = parser.parse_args()

    gt_folder = args.gt_folder
    pred_folder = args.pred_folder

    # Load from config if env specified
    if args.env and (not gt_folder or not pred_folder):
        config = load_config(args.env)

        if not gt_folder:
            if args.dataset and 'datasets' in config:
                gt_folder = os.path.join(config['datasets'][args.dataset]['data_dir'], 'test')
            else:
                gt_folder = resolve_variable(
                    config.get('evaluation', {}).get('gt_folder', ''), config
                )

        if not pred_folder:
            pred_folder = resolve_variable(
                config.get('evaluation', {}).get('stage1_pred_folder', ''), config
            )

    if not gt_folder or not pred_folder:
        parser.error("Must specify --gt_folder and --pred_folder, or --env with valid config")

    # Call HRDoc's classify_eval directly
    sys.argv = ['classify_eval.py', '--gt_folder', gt_folder, '--pred_folder', pred_folder]
    hrdoc_classify_eval()


if __name__ == "__main__":
    main()

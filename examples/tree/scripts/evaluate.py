#!/usr/bin/env python
# coding=utf-8
"""
Evaluation Script: End-to-End Document Structure Evaluation

Usage:
    # Auto-detect environment
    python examples/tree/scripts/evaluate.py

    # Specify environment
    python examples/tree/scripts/evaluate.py --env test

    # Quick test mode
    python examples/tree/scripts/evaluate.py --env test --quick
"""

import os
import sys
import argparse

# Add project root to path (use current working directory)
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import get_config, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation: End-to-End Document Structure")

    # Environment selection
    parser.add_argument("--env", type=str, default=None,
                        help="Environment: dev, test, or auto-detect if not specified")
    parser.add_argument("--quick", action="store_true",
                        help="Force quick test mode (overrides config)")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh"],
                        help="Dataset to use: hrds (HRDoc-Simple) or hrdh (HRDoc-Hard)")

    # Experiment directory
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment directory name (e.g., exp_20251210_201220). If not specified, uses output_dir directly.")

    # Override parameters
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="Override max chunks to load (-1 for all)")
    parser.add_argument("--features_dir", type=str, default=None,
                        help="Override features directory")

    # Flags
    parser.add_argument("--dry_run", action="store_true",
                        help="Print config and exit without evaluation")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    if args.env:
        config = load_config(args.env)
        config = config.get_effective_config()
    else:
        config = get_config()

    # Force quick test if --quick flag
    if args.quick:
        config.quick_test.enabled = True
        config = config.get_effective_config()

    # Determine paths based on experiment and dataset
    experiment_dir = config.paths.output_dir
    if args.experiment:
        experiment_dir = os.path.join(config.paths.output_dir, args.experiment)

    # Features directory (dataset-specific, inside experiment dir)
    features_dir = args.features_dir or os.path.join(experiment_dir, f"features_{args.dataset}")

    # Model paths
    subtask2_model = os.path.join(experiment_dir, f"parent_finder_{args.dataset}", "best_model.pt")
    subtask3_model = os.path.join(experiment_dir, "multiclass_relation", "best_model.pt")

    # Max chunks
    max_chunks = args.max_chunks if args.max_chunks is not None else config.parent_finder.max_chunks

    # Set GPU
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    # Detect GPU info
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else "N/A"

    # Print configuration
    print("=" * 60)
    print("Evaluation: End-to-End Document Structure")
    print("=" * 60)
    print(f"Environment:    {config.env}")
    print(f"Dataset:        {args.dataset.upper()}")
    print(f"Quick Test:     {config.quick_test.enabled}")
    print(f"GPU Config:     CUDA_VISIBLE_DEVICES={config.gpu.cuda_visible_devices or 'all available'}")
    print(f"GPU Status:")
    print(f"  - CUDA available:    {cuda_available}")
    print(f"  - Device count:      {cuda_device_count}")
    print(f"  - Device name:       {cuda_device_name}")
    if args.experiment:
        print(f"Experiment:     {args.experiment}")
    print("-" * 60)
    print(f"Paths:")
    print(f"  Features Dir: {features_dir}")
    print(f"  SubTask 2:    {subtask2_model}")
    print(f"  SubTask 3:    {subtask3_model}")
    print("-" * 60)
    print(f"Parameters:")
    print(f"  Max Chunks:   {max_chunks} (-1 = all)")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run mode - exiting without evaluation]")
        return

    # Check paths
    if not os.path.exists(features_dir):
        print(f"\nError: Features directory does not exist: {features_dir}")
        print("Please run Stage 2 feature extraction first.")
        sys.exit(1)

    if not os.path.exists(subtask2_model):
        print(f"\nError: SubTask 2 model does not exist: {subtask2_model}")
        print("Please run Stage 3 training first.")
        sys.exit(1)

    if not os.path.exists(subtask3_model):
        print(f"\nError: SubTask 3 model does not exist: {subtask3_model}")
        print("Please run Stage 4 training first.")
        sys.exit(1)

    # Build evaluation command
    eval_script = os.path.join(PROJECT_ROOT, "examples", "tree", "evaluate_end_to_end.py")

    # Set environment variables
    os.environ["LAYOUTLMFT_FEATURES_DIR"] = features_dir
    os.environ["SUBTASK2_MODEL_PATH"] = subtask2_model
    os.environ["SUBTASK3_MODEL_PATH"] = subtask3_model
    os.environ["MAX_CHUNKS"] = str(max_chunks)

    cmd_args = [
        sys.executable, eval_script,
    ]

    # Print command
    print("\nRunning command:")
    print(" ".join(cmd_args))
    print(f"Environment variables:")
    print(f"  LAYOUTLMFT_FEATURES_DIR={features_dir}")
    print(f"  SUBTASK2_MODEL_PATH={subtask2_model}")
    print(f"  SUBTASK3_MODEL_PATH={subtask3_model}")
    print(f"  MAX_CHUNKS={max_chunks}")
    print()

    # Execute evaluation
    import subprocess

    # Set PYTHONPATH
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{pythonpath}"
    else:
        env["PYTHONPATH"] = PROJECT_ROOT

    result = subprocess.run(cmd_args, cwd=PROJECT_ROOT, env=env)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Evaluation completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Evaluation failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

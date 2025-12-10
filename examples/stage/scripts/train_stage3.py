#!/usr/bin/env python
# coding=utf-8
"""
Stage 3: ParentFinder Training Script

Usage:
    # Auto-detect environment
    python examples/stage/scripts/train_stage3.py

    # Specify environment
    python examples/stage/scripts/train_stage3.py --env test

    # Quick test mode
    python examples/stage/scripts/train_stage3.py --env test --quick

    # Override specific parameters
    python examples/stage/scripts/train_stage3.py --num_epochs 5 --batch_size 2
"""

import os
import sys
import argparse

# Add project root to path (use current working directory)
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import get_config, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: ParentFinder Training")

    # Environment selection
    parser.add_argument("--env", type=str, default=None,
                        help="Environment: dev, test, or auto-detect if not specified")
    parser.add_argument("--quick", action="store_true",
                        help="Force quick test mode (overrides config)")

    # Override parameters
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="Override max chunks to load (-1 for all)")
    parser.add_argument("--features_dir", type=str, default=None,
                        help="Override features directory (Stage 2 output)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")

    # Flags
    parser.add_argument("--dry_run", action="store_true",
                        help="Print config and exit without training")

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

    # Apply overrides
    if args.num_epochs is not None:
        config.parent_finder.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.parent_finder.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.parent_finder.learning_rate = args.learning_rate
    if args.max_chunks is not None:
        config.parent_finder.max_chunks = args.max_chunks

    # Determine paths
    features_dir = args.features_dir or config.paths.features_dir
    output_dir = args.output_dir or os.path.join(config.paths.output_dir, "parent_finder")

    # Set GPU (CUDA_VISIBLE_DEVICES must be set before importing torch)
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    # Detect GPU info
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else "N/A"

    # Print configuration
    print("=" * 60)
    print("Stage 3: ParentFinder Training")
    print("=" * 60)
    print(f"Environment:    {config.env}")
    print(f"Quick Test:     {config.quick_test.enabled}")
    print(f"GPU Config:     CUDA_VISIBLE_DEVICES={config.gpu.cuda_visible_devices or 'all available'}")
    print(f"GPU Status:")
    print(f"  - CUDA available:    {cuda_available}")
    print(f"  - Device count:      {cuda_device_count}")
    print(f"  - Device name:       {cuda_device_name}")
    print(f"Features Dir:   {features_dir}")
    print(f"Output Dir:     {output_dir}")
    print("-" * 60)
    print(f"Mode:           {config.parent_finder.mode}")
    print(f"Level:          {config.parent_finder.level}")
    print(f"Num Epochs:     {config.parent_finder.num_epochs}")
    print(f"Batch Size:     {config.parent_finder.batch_size}")
    print(f"Learning Rate:  {config.parent_finder.learning_rate}")
    print(f"Max Lines:      {config.parent_finder.max_lines_limit}")
    print(f"Max Chunks:     {config.parent_finder.max_chunks} (-1 = all)")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run mode - exiting without training]")
        return

    # Check if features directory exists
    if not os.path.exists(features_dir):
        print(f"\nError: Features directory does not exist: {features_dir}")
        print("Please run Stage 2 feature extraction first.")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build training command
    train_script = os.path.join(PROJECT_ROOT, "examples", "stage", "train_parent_finder.py")

    cmd_args = [
        sys.executable, train_script,
        "--mode", config.parent_finder.mode,
        "--level", config.parent_finder.level,
        "--features_dir", features_dir,
        "--output_dir", output_dir,
        "--batch_size", str(config.parent_finder.batch_size),
        "--num_epochs", str(config.parent_finder.num_epochs),
        "--learning_rate", str(config.parent_finder.learning_rate),
        "--max_lines_limit", str(config.parent_finder.max_lines_limit),
        "--max_chunks", str(config.parent_finder.max_chunks),
    ]

    # Print command
    print("\nRunning command:")
    print(" ".join(cmd_args))
    print()

    # Execute training
    import subprocess

    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{pythonpath}"
    else:
        env["PYTHONPATH"] = PROJECT_ROOT

    result = subprocess.run(cmd_args, cwd=PROJECT_ROOT, env=env)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("ParentFinder training completed successfully!")
        print(f"Model saved to: {output_dir}")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Check training results in output directory")
        print("  2. Train relation classifier: python examples/stage/scripts/train_stage4.py")
    else:
        print("\n" + "=" * 60)
        print("ParentFinder training failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

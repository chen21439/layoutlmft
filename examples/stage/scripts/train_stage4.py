#!/usr/bin/env python
# coding=utf-8
"""
Stage 4: Relation Classifier Training Script

Usage:
    # Auto-detect environment
    python examples/stage/scripts/train_stage4.py

    # Specify environment
    python examples/stage/scripts/train_stage4.py --env test

    # Quick test mode
    python examples/stage/scripts/train_stage4.py --env test --quick

    # Override specific parameters
    python examples/stage/scripts/train_stage4.py --max_steps 100 --batch_size 64
"""

import os
import sys
import argparse

# Add project root to path (use current working directory)
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import get_config, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4: Relation Classifier Training")

    # Environment selection
    parser.add_argument("--env", type=str, default=None,
                        help="Environment: dev, test, or auto-detect if not specified")
    parser.add_argument("--quick", action="store_true",
                        help="Force quick test mode (overrides config)")

    # Override parameters
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps")
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
    if args.max_steps is not None:
        config.relation_classifier.max_steps = args.max_steps
    if args.batch_size is not None:
        config.relation_classifier.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.relation_classifier.learning_rate = args.learning_rate
    if args.max_chunks is not None:
        config.relation_classifier.max_chunks = args.max_chunks

    # Determine paths
    features_dir = args.features_dir or config.paths.features_dir
    output_dir = args.output_dir or os.path.join(config.paths.output_dir, "relation_classifier")

    # Set GPU (CUDA_VISIBLE_DEVICES must be set before importing torch)
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    # Set environment variables for the training script
    os.environ["LAYOUTLMFT_FEATURES_DIR"] = features_dir
    os.environ["LAYOUTLMFT_OUTPUT_DIR"] = os.path.dirname(output_dir)  # Parent dir
    os.environ["MAX_STEPS"] = str(config.relation_classifier.max_steps)
    os.environ["MAX_CHUNKS"] = str(config.relation_classifier.max_chunks)

    # Detect GPU info
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else "N/A"

    # Print configuration
    print("=" * 60)
    print("Stage 4: Relation Classifier Training")
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
    print(f"Max Steps:      {config.relation_classifier.max_steps}")
    print(f"Batch Size:     {config.relation_classifier.batch_size}")
    print(f"Learning Rate:  {config.relation_classifier.learning_rate}")
    print(f"Neg Ratio:      {config.relation_classifier.neg_ratio}")
    print(f"Max Chunks:     {config.relation_classifier.max_chunks} (-1 = all)")
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
    train_script = os.path.join(PROJECT_ROOT, "examples", "stage", "train_multiclass_relation.py")

    cmd_args = [
        sys.executable, train_script,
    ]

    # Print command
    print("\nRunning command:")
    print(" ".join(cmd_args))
    print(f"Environment variables:")
    print(f"  LAYOUTLMFT_FEATURES_DIR={features_dir}")
    print(f"  LAYOUTLMFT_OUTPUT_DIR={os.path.dirname(output_dir)}")
    print(f"  MAX_STEPS={config.relation_classifier.max_steps}")
    print(f"  MAX_CHUNKS={config.relation_classifier.max_chunks}")
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
        print("Relation classifier training completed successfully!")
        print(f"Model saved to: {output_dir}")
        print("=" * 60)
        print("\nAll stages completed! You can now run inference.")
    else:
        print("\n" + "=" * 60)
        print("Relation classifier training failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

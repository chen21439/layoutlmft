#!/usr/bin/env python
# coding=utf-8
"""
Stage 4: Relation Classifier Training Script

Usage:
    # Auto-detect environment, use default dataset (hrds)
    python examples/stage/scripts/train_stage4.py --env test

    # Specify dataset
    python examples/stage/scripts/train_stage4.py --env test --dataset hrds
    python examples/stage/scripts/train_stage4.py --env test --dataset hrdh

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

# Add examples/stage to path for util imports
STAGE_ROOT = os.path.join(PROJECT_ROOT, "examples", "stage")
sys.path.insert(0, STAGE_ROOT)

from util.checkpoint_utils import (
    get_best_model,
)
from util.experiment_manager import (
    ExperimentManager,
    get_experiment_manager,
    ensure_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4: Relation Classifier Training")

    # Environment selection
    parser.add_argument("--env", type=str, default=None,
                        help="Environment: dev, test, or auto-detect if not specified")
    parser.add_argument("--quick", action="store_true",
                        help="Force quick test mode (overrides config)")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh"],
                        help="Dataset to use: hrds (HRDoc-Simple) or hrdh (HRDoc-Hard)")

    # Experiment management
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment ID (default: current or latest)")
    parser.add_argument("--new_exp", action="store_true",
                        help="Create a new experiment")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name for new experiment")

    # Checkpoint control
    parser.add_argument("--restart", action="store_true",
                        help="Restart training from scratch (ignore existing checkpoints)")

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

    # Initialize experiment manager
    exp_manager, exp_dir = ensure_experiment(
        config,
        exp=args.exp,
        new_exp=args.new_exp,
        name=args.exp_name or f"Stage4 {args.dataset.upper()}",
    )

    # Determine paths based on experiment and dataset
    # Features directory (from Stage 2)
    if args.features_dir:
        features_dir = args.features_dir
    else:
        features_dir = exp_manager.get_stage_dir(args.exp, "stage2", args.dataset)
        if not os.path.exists(features_dir):
            # Fallback to legacy path
            base_features_dir = config.paths.features_dir
            features_dir = f"{base_features_dir}_{args.dataset}"

    # Output directory (experiment-based)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = exp_manager.get_stage_dir(args.exp, "stage4", args.dataset)

    # Check for existing checkpoint
    existing_checkpoint = None
    if not args.restart:
        existing_checkpoint = get_best_model(output_dir)

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

    # Get relation classifier config
    rc_cfg = config.relation_classifier

    # Print configuration
    print("=" * 60)
    print("Stage 4: Relation Classifier Training")
    print("=" * 60)
    print(f"Environment:    {config.env}")
    print(f"Dataset:        {args.dataset.upper()}")
    print(f"Quick Test:     {config.quick_test.enabled}")
    print(f"GPU Config:     CUDA_VISIBLE_DEVICES={config.gpu.cuda_visible_devices or 'all available'}")
    print(f"GPU Status:")
    print(f"  - CUDA available:    {cuda_available}")
    print(f"  - Device count:      {cuda_device_count}")
    print(f"  - Device name:       {cuda_device_name}")
    print(f"Experiment:     {os.path.basename(exp_dir)}")
    print("-" * 60)
    print("Paths:")
    print(f"  Features Dir:   {features_dir}")
    print(f"  Output Dir:     {output_dir}")
    print("-" * 60)
    print("Training Parameters:")
    print(f"  Max Steps:      {rc_cfg.max_steps}")
    print(f"  Batch Size:     {rc_cfg.batch_size}")
    print(f"  Learning Rate:  {rc_cfg.learning_rate}")
    print(f"  Neg Ratio:      {rc_cfg.neg_ratio}")
    print(f"  Max Chunks:     {rc_cfg.max_chunks} (-1 = all)")
    print("-" * 60)
    print("Checkpoint Status:")
    if existing_checkpoint:
        print(f"  Existing Model: {existing_checkpoint}")
        print(f"  Note:           Will overwrite if new model is better")
    elif args.restart:
        print(f"  Mode:           RESTART (training from scratch)")
    else:
        print(f"  Mode:           NEW (no existing checkpoint found)")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run mode - exiting without training]")
        return

    # Check if features directory exists
    if not os.path.exists(features_dir):
        print(f"\nError: Features directory does not exist: {features_dir}")
        print(f"Please run Stage 2 feature extraction first:")
        print(f"  python scripts/train_stage2.py --env {args.env or 'test'} --dataset {args.dataset}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get covmatch directory for train/validation split
    covmatch_dir = config.dataset.get_covmatch_dir(args.dataset)
    if os.path.exists(covmatch_dir):
        print(f"Using covmatch split: {covmatch_dir}")
    else:
        covmatch_dir = None
        print(f"Warning: Covmatch dir not found, using original train/test split")

    # Build training command
    train_script = os.path.join(PROJECT_ROOT, "examples", "stage", "train_multiclass_relation.py")

    cmd_args = [
        sys.executable, train_script,
        "--max_steps", str(config.relation_classifier.max_steps),
        "--max_chunks", str(config.relation_classifier.max_chunks),
        "--batch_size", str(config.relation_classifier.batch_size),
        "--learning_rate", str(config.relation_classifier.learning_rate),
        "--neg_ratio", str(config.relation_classifier.neg_ratio),
    ]

    # Print command
    print("\nRunning command:")
    print(" ".join(cmd_args))
    print()

    # Mark stage as started
    exp_manager.mark_stage_started(args.exp, "stage4", args.dataset)

    # Execute training
    import subprocess

    # Set PYTHONPATH and environment variables
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{pythonpath}"
    else:
        env["PYTHONPATH"] = PROJECT_ROOT

    # Set paths via environment variables (consistent with Stage 1/2/3)
    env["LAYOUTLMFT_FEATURES_DIR"] = features_dir
    env["LAYOUTLMFT_OUTPUT_DIR"] = output_dir
    if covmatch_dir:
        env["HRDOC_SPLIT_DIR"] = covmatch_dir

    result = subprocess.run(cmd_args, cwd=PROJECT_ROOT, env=env)

    if result.returncode == 0:
        # Get best model path and update experiment state
        best_model = get_best_model(output_dir)
        exp_manager.mark_stage_completed(
            args.exp, "stage4", args.dataset,
            best_checkpoint=os.path.basename(best_model) if best_model else None,
        )

        print("\n" + "=" * 60)
        print("Relation classifier training completed successfully!")
        print(f"Model saved to: {output_dir}")
        print("=" * 60)
        print(f"\nAll stages for {args.dataset.upper()} completed!")
        print("\nTo train on another dataset:")
        other_dataset = 'hrdh' if args.dataset == 'hrds' else 'hrds'
        print(f"  python examples/stage/scripts/train_stage1.py --env {args.env or 'test'} --dataset {other_dataset}")
    else:
        # Mark stage as failed
        exp_manager.mark_stage_failed(args.exp, "stage4", args.dataset)

        print("\n" + "=" * 60)
        print("Relation classifier training failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

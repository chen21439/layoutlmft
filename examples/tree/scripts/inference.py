#!/usr/bin/env python
# coding=utf-8
"""
Inference Script: Build Document Tree from trained models

Usage:
    # Auto-detect environment (uses current/latest experiment)
    python examples/tree/scripts/inference.py --env test

    # Specify experiment
    python examples/tree/scripts/inference.py --env test --exp exp_20251210_201220

    # Quick test mode
    python examples/tree/scripts/inference.py --env test --quick

    # Override specific parameters
    python examples/tree/scripts/inference.py --max_samples 20 --level page
"""

import os
import sys
import argparse

# Add project root to path (use current working directory)
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

# Add examples/stage to path for util imports
STAGE_ROOT = os.path.join(PROJECT_ROOT, "examples", "stage")
sys.path.insert(0, STAGE_ROOT)

from configs.config_loader import get_config, load_config
from util.experiment_manager import ExperimentManager, get_experiment_manager


def parse_args():
    parser = argparse.ArgumentParser(description="Inference: Build Document Tree")

    # Environment selection
    parser.add_argument("--env", type=str, default=None,
                        help="Environment: dev, test, or auto-detect if not specified")
    parser.add_argument("--quick", action="store_true",
                        help="Force quick test mode (overrides config)")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh"],
                        help="Dataset to use: hrds (HRDoc-Simple) or hrdh (HRDoc-Hard)")

    # Experiment selection (same as training scripts)
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment ID (e.g., exp_20251210_201220). Default: current or latest.")

    # Override parameters
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Override max samples to process (-1 for all)")
    parser.add_argument("--level", type=str, default=None, choices=["page", "document"],
                        help="Override inference level: page or document")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")

    # Manual model path overrides
    parser.add_argument("--subtask1_model", type=str, default=None,
                        help="Override SubTask 1 (LayoutXLM) model path")
    parser.add_argument("--subtask2_model", type=str, default=None,
                        help="Override SubTask 2 (ParentFinder) model path")
    parser.add_argument("--subtask3_model", type=str, default=None,
                        help="Override SubTask 3 (RelationClassifier) model path")

    # Output options
    parser.add_argument("--save_json", action="store_true", default=True,
                        help="Save JSON format tree")
    parser.add_argument("--save_ascii", action="store_true", default=True,
                        help="Save ASCII format tree")
    parser.add_argument("--save_markdown", action="store_true",
                        help="Save Markdown format tree")

    # Flags
    parser.add_argument("--dry_run", action="store_true",
                        help="Print config and exit without inference")

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

    # Get experiment manager and experiment directory
    exp_manager = get_experiment_manager(config)
    exp_dir = exp_manager.get_experiment_dir(args.exp)

    if not exp_dir:
        print(f"\nError: No experiment found.")
        print(f"Please run training first or specify --exp <experiment_id>")
        print(f"\nAvailable experiments:")
        for exp_info in exp_manager.list_experiments():
            current_marker = " (current)" if exp_info.get('is_current') else ""
            print(f"  - {exp_info['dirname']}: {exp_info.get('name', 'unnamed')}{current_marker}")
        sys.exit(1)

    exp_name = os.path.basename(exp_dir)

    # Model paths using ExperimentManager (same as training scripts)
    # Allow manual overrides via command line
    subtask1_model = args.subtask1_model or exp_manager.get_stage_dir(args.exp, "stage1", args.dataset)
    subtask2_model = args.subtask2_model or os.path.join(exp_manager.get_stage_dir(args.exp, "stage3", args.dataset), "best_model.pt")
    subtask3_model = args.subtask3_model or os.path.join(exp_dir, "multiclass_relation", "best_model.pt")

    # Data directory
    data_dir = config.paths.hrdoc_data_dir

    # Output directory
    output_dir = args.output_dir or os.path.join(exp_dir, f"inference_{args.dataset}")

    # Inference level
    level = args.level or config.parent_finder.level

    # Max samples
    if args.quick or config.quick_test.enabled:
        max_samples = args.max_samples if args.max_samples is not None else 10
    else:
        max_samples = args.max_samples if args.max_samples is not None else -1

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
    print("Inference: Build Document Tree")
    print("=" * 60)
    print(f"Environment:    {config.env}")
    print(f"Dataset:        {args.dataset.upper()}")
    print(f"Quick Test:     {config.quick_test.enabled}")
    print(f"GPU Config:     CUDA_VISIBLE_DEVICES={config.gpu.cuda_visible_devices or 'all available'}")
    print(f"GPU Status:")
    print(f"  - CUDA available:    {cuda_available}")
    print(f"  - Device count:      {cuda_device_count}")
    print(f"  - Device name:       {cuda_device_name}")
    print(f"Experiment:     {exp_name}")
    print("-" * 60)
    print(f"Model Paths:")
    print(f"  SubTask 1:    {subtask1_model}")
    print(f"  SubTask 2:    {subtask2_model}")
    print(f"  SubTask 3:    {subtask3_model}")
    print(f"Data Dir:       {data_dir}")
    print(f"Output Dir:     {output_dir}")
    print("-" * 60)
    print(f"Inference Parameters:")
    print(f"  Split:        {args.split}")
    print(f"  Level:        {level}")
    print(f"  Max Samples:  {max_samples} (-1 = all)")
    print(f"  Save JSON:    {args.save_json}")
    print(f"  Save ASCII:   {args.save_ascii}")
    print(f"  Save MD:      {args.save_markdown}")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run mode - exiting without inference]")
        return

    # Check if model paths exist
    missing_models = []
    if not os.path.exists(subtask1_model):
        missing_models.append(f"SubTask 1: {subtask1_model}")
    if not os.path.exists(subtask2_model):
        missing_models.append(f"SubTask 2: {subtask2_model}")
    if not os.path.exists(subtask3_model):
        missing_models.append(f"SubTask 3: {subtask3_model}")

    if missing_models:
        print(f"\nError: Some model paths do not exist:")
        for m in missing_models:
            print(f"  - {m}")
        print("\nPlease run training stages first:")
        print(f"  python examples/stage/scripts/train_stage1.py --env {args.env or 'test'} --dataset {args.dataset}")
        print(f"  python examples/stage/scripts/train_stage2.py --env {args.env or 'test'} --dataset {args.dataset}")
        print(f"  python examples/stage/scripts/train_stage3.py --env {args.env or 'test'} --dataset {args.dataset}")
        print(f"  python examples/stage/scripts/train_stage4.py --env {args.env or 'test'} --dataset {args.dataset}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build inference command
    inference_script = os.path.join(PROJECT_ROOT, "examples", "tree", "inference_build_tree.py")

    cmd_args = [
        sys.executable, inference_script,
        "--subtask1_model", subtask1_model,
        "--subtask2_model", subtask2_model,
        "--subtask3_model", subtask3_model,
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--split", args.split,
        "--level", level,
        "--max_samples", str(max_samples),
    ]

    if args.save_json:
        cmd_args.append("--save_json")
    if args.save_ascii:
        cmd_args.append("--save_ascii")
    if args.save_markdown:
        cmd_args.append("--save_markdown")

    # Print command
    print("\nRunning command:")
    print(" ".join(cmd_args))
    print()

    # Execute inference
    import subprocess

    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{pythonpath}"
    else:
        env["PYTHONPATH"] = PROJECT_ROOT

    # Set data directory environment variable
    env["HRDOC_DATA_DIR"] = data_dir

    result = subprocess.run(cmd_args, cwd=PROJECT_ROOT, env=env)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Inference completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Inference failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

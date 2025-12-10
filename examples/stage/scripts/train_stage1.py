#!/usr/bin/env python
# coding=utf-8
"""
Stage 1: LayoutXLM Fine-tuning Training Script

Usage:
    # Auto-detect environment, use default dataset (hrds)
    python scripts/train_stage1.py --env test

    # Specify dataset
    python scripts/train_stage1.py --env test --dataset hrds
    python scripts/train_stage1.py --env test --dataset hrdh

    # Resume from checkpoint (auto-detect latest)
    python scripts/train_stage1.py --env test --dataset hrds
    # If checkpoint exists, will auto-resume

    # Restart training from scratch
    python scripts/train_stage1.py --env test --dataset hrds --restart

    # Load weights from another dataset's checkpoint, train on current dataset
    python scripts/train_stage1.py --env test --dataset hrdh --init_from hrds

    # Quick test mode
    python scripts/train_stage1.py --env test --quick

    # Override specific parameters
    python scripts/train_stage1.py --max_steps 100 --batch_size 2
"""

import os
import sys
import argparse
import glob

# Add project root to path (use current working directory)
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import get_config, load_config, print_config


def get_latest_checkpoint(output_dir):
    """Get the latest checkpoint directory from output_dir"""
    if not os.path.isdir(output_dir):
        return None

    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None

    # Sort by step number
    def get_step(path):
        try:
            return int(os.path.basename(path).split("-")[1])
        except:
            return 0

    checkpoint_dirs.sort(key=get_step, reverse=True)
    return checkpoint_dirs[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: LayoutXLM Fine-tuning")

    # Environment selection
    parser.add_argument("--env", type=str, default=None,
                        help="Environment: dev, test, or auto-detect if not specified")
    parser.add_argument("--quick", action="store_true",
                        help="Force quick test mode (overrides config)")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh"],
                        help="Dataset to use: hrds (HRDoc-Simple) or hrdh (HRDoc-Hard)")

    # Checkpoint control
    parser.add_argument("--restart", action="store_true",
                        help="Restart training from scratch (ignore existing checkpoints)")
    parser.add_argument("--init_from", type=str, default=None, choices=["hrds", "hrdh"],
                        help="Initialize model weights from another dataset's checkpoint (e.g., --init_from hrds)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from specific checkpoint path")

    # Override parameters
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path")

    # Flags
    parser.add_argument("--dry_run", action="store_true",
                        help="Print config and exit without training")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    if args.env:
        config = load_config(args.env)
        # Apply quick_test if enabled in config
        config = config.get_effective_config()
    else:
        config = get_config()

    # Force quick test if --quick flag
    if args.quick:
        config.quick_test.enabled = True
        config = config.get_effective_config()

    # Apply overrides
    if args.max_steps is not None:
        config.stage1_training.max_steps = args.max_steps
    if args.batch_size is not None:
        config.stage1_training.per_device_train_batch_size = args.batch_size

    # Determine output directory based on dataset
    # Each dataset has its own output directory for independent checkpoint management
    base_output_dir = args.output_dir or config.paths.stage1_model_path
    output_dir = f"{base_output_dir}_{args.dataset}"

    # Determine model path (for initial weights)
    if args.init_from:
        # Load weights from another dataset's checkpoint
        init_output_dir = f"{base_output_dir}_{args.init_from}"
        init_checkpoint = get_latest_checkpoint(init_output_dir)
        if init_checkpoint:
            model_path = init_checkpoint
            print(f"Initializing weights from {args.init_from} checkpoint: {init_checkpoint}")
        else:
            print(f"Warning: No checkpoint found for {args.init_from}, using base model")
            model_path = args.model_path or config.model.local_path or config.model.name_or_path
    else:
        model_path = args.model_path or config.model.local_path or config.model.name_or_path

    # Determine data directory based on dataset
    data_dir_base = os.path.dirname(config.paths.hrdoc_data_dir)
    if args.dataset == "hrds":
        data_dir = os.path.join(data_dir_base, "HRDS")
    else:  # hrdh
        data_dir = os.path.join(data_dir_base, "HRDH")

    # Fallback to config path if dataset-specific path doesn't exist
    if not os.path.exists(data_dir):
        data_dir = config.paths.hrdoc_data_dir
        print(f"Warning: Dataset-specific path not found, using: {data_dir}")

    # Check for existing checkpoint (for resume)
    resume_checkpoint = None
    if not args.restart:
        if args.resume_from:
            resume_checkpoint = args.resume_from
        else:
            resume_checkpoint = get_latest_checkpoint(output_dir)

    # Set GPU (CUDA_VISIBLE_DEVICES must be set before importing torch)
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    # Set HuggingFace cache
    if config.paths.hf_cache_dir:
        os.environ["HF_HOME"] = config.paths.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = config.paths.hf_cache_dir

    # Set data directory
    os.environ["HRDOC_DATA_DIR"] = data_dir

    # Set seqeval metric path (for offline mode)
    if config.metrics.seqeval_path:
        os.environ["SEQEVAL_PATH"] = config.metrics.seqeval_path

    # Detect GPU info (import torch after setting CUDA_VISIBLE_DEVICES)
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_current_device = torch.cuda.current_device() if cuda_available else None
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else "N/A"

    # Print configuration
    print("=" * 60)
    print("Stage 1: LayoutXLM Fine-tuning")
    print("=" * 60)
    print(f"Environment:  {config.env}")
    print(f"Dataset:      {args.dataset.upper()}")
    print(f"Quick Test:   {config.quick_test.enabled}")
    print(f"GPU Config:   CUDA_VISIBLE_DEVICES={config.gpu.cuda_visible_devices or 'all available'}")
    print(f"GPU Status:")
    print(f"  - CUDA available:    {cuda_available}")
    print(f"  - Device count:      {cuda_device_count}")
    print(f"  - Current device:    {cuda_current_device}")
    print(f"  - Device name:       {cuda_device_name}")
    print(f"Model Path:   {model_path}")
    print(f"Output Dir:   {output_dir}")
    print(f"Data Dir:     {data_dir}")
    print("-" * 60)
    print(f"Max Steps:    {config.stage1_training.max_steps}")
    print(f"Batch Size:   {config.stage1_training.per_device_train_batch_size}")
    print(f"Learning Rate:{config.stage1_training.learning_rate}")
    print(f"FP16:         {config.stage1_training.fp16}")
    print("-" * 60)
    if resume_checkpoint:
        print(f"Resume From:  {resume_checkpoint}")
    elif args.restart:
        print(f"Mode:         RESTART (overwrite existing checkpoints)")
    else:
        print(f"Mode:         NEW (no existing checkpoint found)")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run mode - exiting without training]")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build training command
    train_script = os.path.join(PROJECT_ROOT, "examples", "stage", "run_hrdoc.py")

    cmd_args = [
        sys.executable, train_script,
        "--model_name_or_path", model_path,
        "--output_dir", output_dir,
        "--do_train",
        "--max_steps", str(config.stage1_training.max_steps),
        "--per_device_train_batch_size", str(config.stage1_training.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(config.stage1_training.per_device_eval_batch_size),
        "--gradient_accumulation_steps", str(config.stage1_training.gradient_accumulation_steps),
        "--learning_rate", str(config.stage1_training.learning_rate),
        "--warmup_ratio", str(config.stage1_training.warmup_ratio),
        "--weight_decay", str(config.stage1_training.weight_decay),
        "--logging_steps", str(config.stage1_training.logging_steps),
        "--save_steps", str(config.stage1_training.save_steps),
        "--save_total_limit", str(config.stage1_training.save_total_limit),
        "--seed", str(config.stage1_training.seed),
        "--report_to", "none",  # Disable TensorBoard to avoid distutils.version issue
    ]

    # Handle checkpoint resume/restart
    if args.restart:
        cmd_args.append("--overwrite_output_dir")
    elif resume_checkpoint:
        cmd_args.extend(["--resume_from_checkpoint", resume_checkpoint])

    if config.stage1_training.fp16:
        cmd_args.append("--fp16")

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
        print("Training completed successfully!")
        print(f"Model saved to: {output_dir}")
        print("=" * 60)
        print("\nNext steps:")
        print(f"  1. Check training logs: tensorboard --logdir {output_dir}")
        print(f"  2. Extract features: python scripts/train_stage2.py --dataset {args.dataset}")
        print(f"\nTo continue training on another dataset:")
        print(f"  python scripts/train_stage1.py --env {args.env or 'test'} --dataset {'hrdh' if args.dataset == 'hrds' else 'hrds'} --init_from {args.dataset}")
    else:
        print("\n" + "=" * 60)
        print("Training failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

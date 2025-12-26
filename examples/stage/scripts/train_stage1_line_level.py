#!/usr/bin/env python
# coding=utf-8
"""
Stage 1: Line-Level LayoutXLM Training Script

使用 mean pooling 进行 line-level 分类，与联合训练的 Stage 1 逻辑完全对齐。

Usage:
    # Auto-detect environment, use default dataset (hrds)
    python scripts/train_stage1_line_level.py --env test

    # Specify dataset
    python scripts/train_stage1_line_level.py --env test --dataset hrds
    python scripts/train_stage1_line_level.py --env test --dataset hrdh

    # Resume from checkpoint (auto-detect latest)
    python scripts/train_stage1_line_level.py --env test --dataset hrds

    # Restart training from scratch
    python scripts/train_stage1_line_level.py --env test --dataset hrds --restart

    # Quick test mode
    python scripts/train_stage1_line_level.py --env test --quick
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import get_config, load_config

# Add examples/stage to path
STAGE_ROOT = os.path.join(PROJECT_ROOT, "examples", "stage")
sys.path.insert(0, STAGE_ROOT)

from util.checkpoint_utils import get_latest_checkpoint
from util.experiment_manager import ensure_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Line-Level LayoutXLM Training")

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
                        help="Initialize model weights from another dataset's checkpoint")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from specific checkpoint path")

    # Experiment management
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment ID (default: current or latest)")
    parser.add_argument("--new_exp", action="store_true",
                        help="Create a new experiment")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name for new experiment")

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

    # Initialize experiment manager
    exp_manager, exp_dir = ensure_experiment(
        config,
        exp=args.exp,
        new_exp=args.new_exp,
        name=args.exp_name or f"Stage1-LineLevel {args.dataset.upper()}",
    )

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use experiment-based path with "line_level" suffix
        base_dir = exp_manager.get_stage_dir(args.exp, "stage1", args.dataset)
        output_dir = base_dir + "_line_level"

    # Determine model path
    if args.init_from:
        init_stage_dir = exp_manager.get_stage_dir(args.exp, "stage1", args.init_from)
        init_checkpoint = get_latest_checkpoint(init_stage_dir)
        if init_checkpoint:
            model_path = init_checkpoint
            print(f"Initializing weights from {args.init_from} checkpoint: {init_checkpoint}")
        else:
            print(f"Warning: No checkpoint found for {args.init_from}, using base model")
            model_path = args.model_path or config.model.local_path or config.model.name_or_path
    else:
        model_path = args.model_path or config.model.local_path or config.model.name_or_path

    # Determine data directory
    data_dir = config.dataset.get_data_dir(args.dataset)
    if not data_dir or not os.path.exists(data_dir):
        data_dir = config.paths.hrdoc_data_dir

    # Check for existing checkpoint
    resume_checkpoint = None
    if not args.restart:
        if args.resume_from:
            resume_checkpoint = args.resume_from
        else:
            resume_checkpoint = get_latest_checkpoint(output_dir)

    # Set environment variables
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    if config.paths.hf_cache_dir:
        os.environ["HF_HOME"] = config.paths.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = config.paths.hf_cache_dir

    os.environ["HRDOC_DATA_DIR"] = data_dir

    # Set covmatch directory
    covmatch_dir = config.dataset.get_covmatch_dir(args.dataset)
    if os.path.exists(covmatch_dir):
        os.environ["HRDOC_SPLIT_DIR"] = covmatch_dir

    if config.metrics.seqeval_path:
        os.environ["SEQEVAL_PATH"] = config.metrics.seqeval_path

    # Detect GPU info
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_current_device = torch.cuda.current_device() if cuda_available else None
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else "N/A"

    # Get training config
    train_cfg = config.stage1_training
    effective_batch = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps

    # Print configuration
    print("=" * 60)
    print("Stage 1: Line-Level LayoutXLM Training")
    print("=" * 60)
    print(f"Environment:  {config.env}")
    print(f"Dataset:      {args.dataset.upper()}")
    print(f"Quick Test:   {config.quick_test.enabled}")
    print(f"Mode:         LINE-LEVEL (mean pooling)")
    print(f"GPU Config:   CUDA_VISIBLE_DEVICES={config.gpu.cuda_visible_devices or 'all available'}")
    print(f"GPU Status:")
    print(f"  - CUDA available:    {cuda_available}")
    print(f"  - Device count:      {cuda_device_count}")
    print(f"  - Current device:    {cuda_current_device}")
    print(f"  - Device name:       {cuda_device_name}")
    print(f"Experiment:   {os.path.basename(exp_dir)}")
    print(f"Model Path:   {model_path}")
    print(f"Output Dir:   {output_dir}")
    print(f"Data Dir:     {data_dir}")
    print("-" * 60)
    print("Training Loop:")
    print(f"  Max Steps:            {train_cfg.max_steps}")
    print(f"  Batch Size:           {train_cfg.per_device_train_batch_size}")
    print(f"  Grad Accum Steps:     {train_cfg.gradient_accumulation_steps}")
    print(f"  Effective Batch:      {effective_batch}")
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

    # Build training command - use run_hrdoc_line_level.py
    train_script = os.path.join(PROJECT_ROOT, "examples", "stage", "run_hrdoc_line_level.py")

    cmd_args = [
        sys.executable, train_script,
        "--model_name_or_path", model_path,
        "--output_dir", output_dir,
        "--do_train",
        # Training loop
        "--max_steps", str(train_cfg.max_steps),
        "--per_device_train_batch_size", str(train_cfg.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(train_cfg.per_device_eval_batch_size),
        "--gradient_accumulation_steps", str(train_cfg.gradient_accumulation_steps),
        # Optimizer
        "--learning_rate", str(train_cfg.learning_rate),
        "--weight_decay", str(train_cfg.weight_decay),
        "--max_grad_norm", str(getattr(train_cfg, 'max_grad_norm', 1.0)),
        "--lr_scheduler_type", str(getattr(train_cfg, 'lr_scheduler_type', 'linear')),
        "--warmup_steps", str(getattr(train_cfg, 'warmup_steps', 0)),
        # Evaluation & checkpointing
        "--evaluation_strategy", str(getattr(train_cfg, 'evaluation_strategy', 'no')),
        "--save_strategy", str(getattr(train_cfg, 'save_strategy', 'steps')),
        "--save_steps", str(getattr(train_cfg, 'save_steps', 500)),
        "--save_total_limit", str(train_cfg.save_total_limit),
        # Logging
        "--logging_steps", str(train_cfg.logging_steps),
        # Other
        "--seed", str(train_cfg.seed),
        "--report_to", "none",
        "--dataset_name", args.dataset,
    ]

    # Add eval_steps if evaluation is enabled
    eval_strategy = getattr(train_cfg, 'evaluation_strategy', 'no')
    if eval_strategy != 'no':
        cmd_args.extend(["--do_eval"])
        cmd_args.extend(["--eval_steps", str(getattr(train_cfg, 'eval_steps', 500))])
        if getattr(train_cfg, 'load_best_model_at_end', False):
            cmd_args.extend(["--load_best_model_at_end"])
            metric = getattr(train_cfg, 'metric_for_best_model', None)
            if metric:
                cmd_args.extend(["--metric_for_best_model", metric])
            if getattr(train_cfg, 'greater_is_better', True):
                cmd_args.extend(["--greater_is_better", "True"])

    if args.restart:
        cmd_args.append("--overwrite_output_dir")

    if config.stage1_training.fp16:
        cmd_args.append("--fp16")

    # Print command
    print("\nRunning command:")
    print(" ".join(cmd_args))
    print()

    # Mark stage as started
    exp_manager.mark_stage_started(args.exp, "stage1_line_level", args.dataset)

    # Execute training
    import subprocess

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{PROJECT_ROOT}:{pythonpath}"
    else:
        env["PYTHONPATH"] = PROJECT_ROOT

    result = subprocess.run(cmd_args, cwd=PROJECT_ROOT, env=env)

    if result.returncode == 0:
        best_checkpoint = get_latest_checkpoint(output_dir)
        exp_manager.mark_stage_completed(
            args.exp, "stage1_line_level", args.dataset,
            best_checkpoint=os.path.basename(best_checkpoint) if best_checkpoint else None,
        )

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Model saved to: {output_dir}")
        if best_checkpoint:
            print(f"Best checkpoint: {best_checkpoint}")
        print("=" * 60)
        print("\nNext steps:")
        print(f"  1. Extract features for Stage 3/4 training")
        print(f"  2. Or continue with joint training using this checkpoint")
    else:
        exp_manager.mark_stage_failed(args.exp, "stage1_line_level", args.dataset)
        print("\n" + "=" * 60)
        print("Training failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

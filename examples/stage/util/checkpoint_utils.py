#!/usr/bin/env python
# coding=utf-8
"""
Checkpoint and path utilities for multi-dataset training pipeline.

This module provides shared utilities for:
- Finding latest checkpoints across different training stages
- Managing dataset-specific paths
- Validating model and feature directories
- Integration with experiment management
"""

import os
import glob
from typing import Optional, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment_manager import ExperimentManager


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Get the latest checkpoint from a training output directory.

    Searches for checkpoint-* subdirectories and returns the one with
    the highest step number. If no checkpoints exist but the directory
    itself contains a valid model (config.json), returns the directory.

    Args:
        output_dir: Path to the training output directory

    Returns:
        Path to the latest checkpoint, or None if not found

    Examples:
        >>> get_latest_checkpoint("/artifact/stage1_hrds")
        "/artifact/stage1_hrds/checkpoint-5000"

        >>> get_latest_checkpoint("/artifact/stage1_hrds/checkpoint-5000")
        "/artifact/stage1_hrds/checkpoint-5000"  # Returns self if valid
    """
    if not os.path.isdir(output_dir):
        return None

    # Look for checkpoint-* subdirectories
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))

    if checkpoint_dirs:
        # Sort by step number (checkpoint-1000, checkpoint-2000, ...)
        def get_step(path):
            try:
                return int(os.path.basename(path).split("-")[1])
            except (IndexError, ValueError):
                return 0

        checkpoint_dirs.sort(key=get_step, reverse=True)
        return checkpoint_dirs[0]

    # No checkpoint subdirs, check if output_dir itself is a valid model dir
    if os.path.exists(os.path.join(output_dir, "config.json")):
        return output_dir

    return None


def get_best_model(output_dir: str) -> Optional[str]:
    """Get the best model from a training output directory.

    For Stage 3/4 which save best_model.pt, this returns that path.
    For Stage 1 which uses HuggingFace checkpoints, returns latest checkpoint.

    Args:
        output_dir: Path to the training output directory

    Returns:
        Path to the best model, or None if not found
    """
    if not os.path.isdir(output_dir):
        return None

    # Check for best_model.pt (Stage 3/4)
    best_model = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_model):
        return best_model

    # Fall back to latest checkpoint (Stage 1)
    return get_latest_checkpoint(output_dir)


def get_dataset_path(base_path: str, dataset: str, suffix: str = "") -> str:
    """Generate dataset-specific path.

    Args:
        base_path: Base path from config (e.g., "/artifact/stage1")
        dataset: Dataset name ("hrds", "hrdh", etc.)
        suffix: Optional suffix to append

    Returns:
        Dataset-specific path (e.g., "/artifact/stage1_hrds")
    """
    path = f"{base_path}_{dataset}"
    if suffix:
        path = f"{path}_{suffix}"
    return path


def list_available_checkpoints(output_dir: str) -> List[Dict]:
    """List all available checkpoints with their info.

    Args:
        output_dir: Path to the training output directory

    Returns:
        List of dicts with checkpoint info: [{"path": ..., "step": ..., "size_mb": ...}]
    """
    if not os.path.isdir(output_dir):
        return []

    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    checkpoints = []

    for ckpt_dir in checkpoint_dirs:
        try:
            step = int(os.path.basename(ckpt_dir).split("-")[1])
        except (IndexError, ValueError):
            step = 0

        # Calculate directory size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(ckpt_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

        checkpoints.append({
            "path": ckpt_dir,
            "step": step,
            "size_mb": round(total_size / (1024 * 1024), 2)
        })

    # Sort by step
    checkpoints.sort(key=lambda x: x["step"], reverse=True)
    return checkpoints


def validate_model_path(model_path: str, require_tokenizer: bool = True) -> Dict:
    """Validate a model path and return info about what's available.

    Args:
        model_path: Path to validate
        require_tokenizer: Whether to check for tokenizer files

    Returns:
        Dict with validation results:
        {
            "valid": bool,
            "has_config": bool,
            "has_model": bool,
            "has_tokenizer": bool,
            "is_layoutxlm": bool,
            "errors": List[str]
        }
    """
    result = {
        "valid": False,
        "has_config": False,
        "has_model": False,
        "has_tokenizer": False,
        "is_layoutxlm": False,
        "errors": []
    }

    if not os.path.isdir(model_path):
        result["errors"].append(f"Directory does not exist: {model_path}")
        return result

    # Check for config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        result["has_config"] = True
    else:
        result["errors"].append("Missing config.json")

    # Check for model files
    model_files = ["pytorch_model.bin", "model.safetensors"]
    for mf in model_files:
        if os.path.exists(os.path.join(model_path, mf)):
            result["has_model"] = True
            break
    if not result["has_model"]:
        result["errors"].append("Missing model weights (pytorch_model.bin or model.safetensors)")

    # Check for tokenizer (LayoutXLM uses sentencepiece, LayoutLMv2 uses vocab.txt)
    sentencepiece_path = os.path.join(model_path, "sentencepiece.bpe.model")
    vocab_path = os.path.join(model_path, "vocab.txt")
    tokenizer_json = os.path.join(model_path, "tokenizer.json")

    if os.path.exists(sentencepiece_path):
        result["has_tokenizer"] = True
        result["is_layoutxlm"] = True
    elif os.path.exists(vocab_path):
        result["has_tokenizer"] = True
        result["is_layoutxlm"] = False
    elif os.path.exists(tokenizer_json):
        result["has_tokenizer"] = True
        # Check tokenizer.json content or path for layoutxlm
        result["is_layoutxlm"] = "layoutxlm" in model_path.lower()
    else:
        if require_tokenizer:
            result["errors"].append("Missing tokenizer files")

    # Final validation
    result["valid"] = result["has_config"] and result["has_model"]
    if require_tokenizer:
        result["valid"] = result["valid"] and result["has_tokenizer"]

    return result


def get_stage_output_dir(config, stage: str, dataset: str) -> str:
    """Get the output directory for a specific stage and dataset.

    Args:
        config: Config object with paths
        stage: Stage name ("stage1", "stage2", "stage3", "stage4")
        dataset: Dataset name ("hrds", "hrdh")

    Returns:
        Full path to the stage output directory
    """
    stage_map = {
        "stage1": config.paths.stage1_model_path,
        "stage2": config.paths.features_dir,
        "stage3": os.path.join(config.paths.output_dir, "parent_finder"),
        "stage4": os.path.join(config.paths.output_dir, "relation_classifier"),
    }

    base_path = stage_map.get(stage)
    if not base_path:
        raise ValueError(f"Unknown stage: {stage}")

    return f"{base_path}_{dataset}"


def print_checkpoint_status(output_dir: str, stage_name: str = ""):
    """Print status of checkpoints in a directory.

    Args:
        output_dir: Path to check
        stage_name: Optional name for display
    """
    header = f"{stage_name} Checkpoints" if stage_name else "Checkpoints"
    print(f"\n{header}:")
    print("-" * 40)

    if not os.path.isdir(output_dir):
        print(f"  Directory not found: {output_dir}")
        return

    checkpoints = list_available_checkpoints(output_dir)

    if not checkpoints:
        # Check for best_model.pt
        best_model = os.path.join(output_dir, "best_model.pt")
        if os.path.exists(best_model):
            size_mb = os.path.getsize(best_model) / (1024 * 1024)
            print(f"  best_model.pt ({size_mb:.2f} MB)")
        else:
            print("  No checkpoints found")
        return

    latest = checkpoints[0]
    for ckpt in checkpoints:
        marker = " <- latest" if ckpt == latest else ""
        print(f"  checkpoint-{ckpt['step']} ({ckpt['size_mb']:.2f} MB){marker}")


# ============================================================
# Experiment-aware utilities
# ============================================================

def get_stage_model_path(
    exp_manager: "ExperimentManager",
    exp: Optional[str],
    source_stage: str,
    dataset: str,
) -> Optional[str]:
    """
    Get the best model path from a previous stage within an experiment.

    Args:
        exp_manager: ExperimentManager instance
        exp: Experiment identifier
        source_stage: Source stage name (e.g., "stage1")
        dataset: Dataset name

    Returns:
        Path to the best checkpoint/model from the source stage
    """
    stage_dir = exp_manager.get_stage_dir(exp, source_stage, dataset)
    return get_best_model(stage_dir)


def get_stage_features_dir(
    exp_manager: "ExperimentManager",
    exp: Optional[str],
    dataset: str,
) -> str:
    """
    Get features directory path for a dataset within an experiment.

    Args:
        exp_manager: ExperimentManager instance
        exp: Experiment identifier
        dataset: Dataset name

    Returns:
        Path to features directory
    """
    return exp_manager.get_stage_dir(exp, "stage2", dataset)


def print_experiment_status(exp_manager: "ExperimentManager", exp: Optional[str] = None):
    """
    Print comprehensive status of an experiment.

    Args:
        exp_manager: ExperimentManager instance
        exp: Experiment identifier
    """
    status = exp_manager.get_experiment_status(exp)
    if not status:
        print("No experiment found")
        return

    exp_info = status.get('experiment', {})
    print("=" * 60)
    print(f"Experiment: {exp_info.get('name', 'Unknown')}")
    print("=" * 60)
    print(f"  ID:          {exp_info.get('id', 'N/A')}")
    print(f"  Created:     {exp_info.get('created', 'N/A')}")
    print(f"  Last Run:    {exp_info.get('last_run', 'N/A')}")
    print(f"  Environment: {exp_info.get('env', 'N/A')}")

    stages = status.get('stages', {})
    if stages:
        print("-" * 60)
        print("Stages:")
        for stage_key, stage_state in stages.items():
            status_icon = {
                'pending': '⏳',
                'in_progress': '🔄',
                'completed': '✅',
                'failed': '❌',
            }.get(stage_state.get('status', 'pending'), '❓')

            print(f"  {status_icon} {stage_key}:")
            print(f"      Status: {stage_state.get('status', 'pending')}")
            if stage_state.get('current_step'):
                print(f"      Step:   {stage_state.get('current_step')}")
            if stage_state.get('best_checkpoint'):
                print(f"      Best:   {stage_state.get('best_checkpoint')}")
            if stage_state.get('metrics'):
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in stage_state['metrics'].items())
                print(f"      Metrics: {metrics_str}")
    else:
        print("-" * 60)
        print("No stages started yet")

    print("=" * 60)

#!/usr/bin/env python
# coding=utf-8
"""
Stage 2: Feature Extraction Script

Usage:
    # Auto-detect environment, use default dataset (hrds)
    python scripts/train_stage2.py --env test

    # Specify dataset
    python scripts/train_stage2.py --env test --dataset hrds
    python scripts/train_stage2.py --env test --dataset hrdh

    # Quick test mode
    python scripts/train_stage2.py --env test --quick

    # Override specific parameters
    python scripts/train_stage2.py --num_samples 10 --docs_per_chunk 5
"""

import os
import sys
import argparse

# Add project root to path (use current working directory)
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import get_config, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Feature Extraction")

    # Environment selection
    parser.add_argument("--env", type=str, default=None,
                        help="Environment: dev, test, or auto-detect if not specified")
    parser.add_argument("--quick", action="store_true",
                        help="Force quick test mode (overrides config)")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh"],
                        help="Dataset to use: hrds (HRDoc-Simple) or hrdh (HRDoc-Hard)")

    # Override parameters
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Override number of samples to process (-1 for all)")
    parser.add_argument("--docs_per_chunk", type=int, default=None,
                        help="Override documents per chunk")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override Stage 1 model path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override features output directory")

    # Flags
    parser.add_argument("--dry_run", action="store_true",
                        help="Print config and exit without extraction")

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
    if args.num_samples is not None:
        config.feature_extraction.num_samples = args.num_samples
    if args.docs_per_chunk is not None:
        config.feature_extraction.docs_per_chunk = args.docs_per_chunk
    if args.batch_size is not None:
        config.feature_extraction.batch_size = args.batch_size

    # Determine paths based on dataset
    # Stage 1 model path (dataset-specific)
    base_model_path = config.paths.stage1_model_path
    model_path = args.model_path or f"{base_model_path}_{args.dataset}"

    # Output directory (dataset-specific)
    base_features_dir = config.paths.features_dir
    output_dir = args.output_dir or f"{base_features_dir}_{args.dataset}"

    # Data directory (dataset-specific)
    data_dir_base = os.path.dirname(config.paths.hrdoc_data_dir)
    if args.dataset == "hrds":
        data_dir = os.path.join(data_dir_base, "HRDS")
    else:  # hrdh
        data_dir = os.path.join(data_dir_base, "HRDH")

    # Fallback to config path if dataset-specific path doesn't exist
    if not os.path.exists(data_dir):
        data_dir = config.paths.hrdoc_data_dir
        print(f"Warning: Dataset-specific path not found, using: {data_dir}")

    # Set GPU (CUDA_VISIBLE_DEVICES must be set before importing torch)
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    # Set HuggingFace cache
    if config.paths.hf_cache_dir:
        os.environ["HF_HOME"] = config.paths.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = config.paths.hf_cache_dir

    # Set environment variables for the extraction script
    os.environ["HRDOC_DATA_DIR"] = data_dir
    os.environ["LAYOUTLMFT_MODEL_PATH"] = model_path
    os.environ["LAYOUTLMFT_FEATURES_DIR"] = output_dir
    os.environ["LAYOUTLMFT_NUM_SAMPLES"] = str(config.feature_extraction.num_samples)
    os.environ["LAYOUTLMFT_DOCS_PER_CHUNK"] = str(config.feature_extraction.docs_per_chunk)
    os.environ["LAYOUTLMFT_BATCH_SIZE"] = str(config.feature_extraction.batch_size)

    # Print configuration
    print("=" * 60)
    print("Stage 2: Feature Extraction (Document Level)")
    print("=" * 60)
    print(f"Environment:    {config.env}")
    print(f"Dataset:        {args.dataset.upper()}")
    print(f"Quick Test:     {config.quick_test.enabled}")
    print(f"GPU:            {config.gpu.cuda_visible_devices or 'all available'}")
    print(f"Model Path:     {model_path}")
    print(f"Output Dir:     {output_dir}")
    print(f"Data Dir:       {data_dir}")
    print("-" * 60)
    print(f"Num Samples:    {config.feature_extraction.num_samples} (-1 = all)")
    print(f"Docs per Chunk: {config.feature_extraction.docs_per_chunk}")
    print(f"Batch Size:     {config.feature_extraction.batch_size}")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry run mode - exiting without extraction]")
        return

    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"\nError: Model path does not exist: {model_path}")
        print(f"Please run Stage 1 training first:")
        print(f"  python scripts/train_stage1.py --env {args.env or 'test'} --dataset {args.dataset}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build extraction command
    extract_script = os.path.join(PROJECT_ROOT, "examples", "stage", "util", "extract_line_features_document_level.py")

    cmd_args = [
        sys.executable, extract_script,
        "--data_dir", data_dir,
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--num_samples", str(config.feature_extraction.num_samples),
        "--docs_per_chunk", str(config.feature_extraction.docs_per_chunk),
    ]

    # Print command
    print("\nRunning command:")
    print(" ".join(cmd_args))
    print()

    # Execute extraction
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
        print("Feature extraction completed successfully!")
        print(f"Features saved to: {output_dir}")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Check extracted features in output directory")
        print(f"  2. Train ParentFinder: python scripts/train_stage3.py --env {args.env or 'test'} --dataset {args.dataset}")
    else:
        print("\n" + "=" * 60)
        print("Feature extraction failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

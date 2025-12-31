#!/usr/bin/env python
# coding=utf-8
"""
[DEPRECATED] Stage 1 Training Script

This script is deprecated. Please use train_joint.py with --mode stage1 instead.

Old command:
    python scripts/train_stage1.py --env test --dataset hrds

New command:
    python train_joint.py --env test --dataset hrds --mode stage1

For more options, see:
    python train_joint.py --help
"""

import os
import sys
import warnings

# Show deprecation warning
warnings.warn(
    "\n"
    "=" * 70 + "\n"
    "DEPRECATION WARNING: train_stage1.py is deprecated.\n"
    "Please use: python train_joint.py --mode stage1\n"
    "=" * 70 + "\n",
    DeprecationWarning,
    stacklevel=2
)

print("=" * 70)
print("DEPRECATION WARNING")
print("=" * 70)
print("train_stage1.py is deprecated and will be removed in a future version.")
print()
print("Please use the unified training script instead:")
print("    python train_joint.py --env test --dataset hrds --mode stage1")
print()
print("The new script supports all training modes:")
print("    --mode stage1   : Only train Stage 1 (classification)")
print("    --mode stage34  : Freeze Stage 1, train Stage 3/4")
print("    --mode joint    : End-to-end joint training (default)")
print("=" * 70)
print()

# Convert old arguments to new format and forward to train_joint.py
def convert_args():
    """Convert old train_stage1.py arguments to train_joint.py format"""
    import argparse

    parser = argparse.ArgumentParser(description="[DEPRECATED] Stage 1 Training - Use train_joint.py instead")
    parser.add_argument("--env", type=str, default="test")
    parser.add_argument("--dataset", type=str, default="hrds")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--restart", action="store_true", help="[IGNORED] Use --new_exp instead")
    parser.add_argument("--init_from", type=str, default=None, help="[IGNORED] Use --model_name_or_path instead")
    parser.add_argument("--token_level", action="store_true", help="[NOT SUPPORTED] Line-level is now default")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--new_exp", type=str, default="")

    args, unknown = parser.parse_known_args()

    # Warn about unsupported options
    if args.restart:
        print("[WARNING] --restart is ignored. Use --new_exp to start a new experiment.")
    if args.init_from:
        print(f"[WARNING] --init_from is ignored. Use --model_name_or_path to specify initial weights.")
    if args.token_level:
        print("[WARNING] --token_level is not supported. Line-level classification is now the default.")

    # Build new command
    new_args = ["--mode", "stage1", "--env", args.env, "--dataset", args.dataset]

    if args.quick:
        new_args.append("--quick")
    if args.max_steps:
        new_args.extend(["--max_steps", str(args.max_steps)])
    if args.batch_size:
        new_args.extend(["--per_device_train_batch_size", str(args.batch_size)])
    if args.exp:
        new_args.extend(["--exp", args.exp])
    if args.new_exp:
        new_args.extend(["--new_exp", args.new_exp])

    # Add any unknown arguments
    new_args.extend(unknown)

    return new_args


def main():
    # Get the directory containing train_joint.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stage_dir = os.path.dirname(script_dir)
    train_joint_path = os.path.join(stage_dir, "train_joint.py")

    if not os.path.exists(train_joint_path):
        print(f"[ERROR] train_joint.py not found at: {train_joint_path}")
        sys.exit(1)

    # Convert arguments
    new_args = convert_args()

    print(f"Forwarding to: python train_joint.py {' '.join(new_args)}")
    print()

    # Execute train_joint.py with converted arguments
    sys.argv = [train_joint_path] + new_args

    # Import and run train_joint
    sys.path.insert(0, stage_dir)
    from train_joint import main as train_joint_main
    train_joint_main()


if __name__ == "__main__":
    main()

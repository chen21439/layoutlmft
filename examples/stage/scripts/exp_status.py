#!/usr/bin/env python
# coding=utf-8
"""
Experiment Status Utility

View and manage experiments in the training pipeline.

Usage:
    # List all experiments
    python examples/stage/scripts/exp_status.py --env test --list

    # Show current experiment status
    python examples/stage/scripts/exp_status.py --env test

    # Show specific experiment
    python examples/stage/scripts/exp_status.py --env test --exp exp_20251210_103000

    # Set current experiment
    python examples/stage/scripts/exp_status.py --env test --set-current exp_20251210_103000

    # Create new experiment
    python examples/stage/scripts/exp_status.py --env test --new --name "My Experiment"
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from configs.config_loader import get_config, load_config

# Add examples/stage to path for util imports
STAGE_ROOT = os.path.join(PROJECT_ROOT, "examples", "stage")
sys.path.insert(0, STAGE_ROOT)

from util.experiment_manager import ExperimentManager, get_experiment_manager
from util.checkpoint_utils import print_experiment_status


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Status Utility")

    # Environment
    parser.add_argument("--env", type=str, default=None,
                        help="Environment: dev, test, or auto-detect")

    # Actions
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all experiments")
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment to show (default: current)")
    parser.add_argument("--set-current", type=str, default=None,
                        help="Set current experiment")
    parser.add_argument("--new", action="store_true",
                        help="Create new experiment")
    parser.add_argument("--name", type=str, default="",
                        help="Name for new experiment")
    parser.add_argument("--description", type=str, default="",
                        help="Description for new experiment")

    return parser.parse_args()


def list_experiments(manager: ExperimentManager):
    """List all experiments."""
    experiments = manager.list_experiments()

    if not experiments:
        print("No experiments found.")
        print(f"  Output root: {manager.output_root}")
        return

    print("=" * 70)
    print("Experiments")
    print("=" * 70)
    print(f"{'ID':<5} {'Name':<25} {'Created':<20} {'Stages':<15} {'Current'}")
    print("-" * 70)

    for exp in experiments:
        exp_id = exp.get('id', '?')
        name = exp.get('name', exp['dirname'])[:24]
        created = exp.get('created', 'N/A')[:19]
        stages = ', '.join(exp.get('stages', []))[:14] or 'none'
        current = '*' if exp.get('is_current') else ''

        print(f"{exp_id:<5} {name:<25} {created:<20} {stages:<15} {current}")

    print("=" * 70)
    print(f"Total: {len(experiments)} experiments")


def main():
    args = parse_args()

    # Load configuration
    if args.env:
        config = load_config(args.env)
        config = config.get_effective_config()
    else:
        config = get_config()

    manager = get_experiment_manager(config)

    # Handle actions
    if args.list:
        list_experiments(manager)
        return

    if args.set_current:
        manager.set_current(args.set_current)
        return

    if args.new:
        exp_dir = manager.create_experiment(
            config,
            name=args.name,
            description=args.description,
        )
        print(f"\nExperiment directory: {exp_dir}")
        return

    # Default: show experiment status
    print_experiment_status(manager, args.exp)

    # Also show checkpoint details for each stage
    exp_dir = manager.get_experiment_dir(args.exp)
    if exp_dir:
        print(f"\nExperiment directory: {exp_dir}")

        # List stage directories
        stage_prefixes = ['stage1', 'features', 'parent_finder', 'relation_classifier']
        for prefix in stage_prefixes:
            import glob
            stage_dirs = glob.glob(os.path.join(exp_dir, f"{prefix}_*"))
            for stage_dir in sorted(stage_dirs):
                if os.path.isdir(stage_dir):
                    stage_name = os.path.basename(stage_dir)
                    from util.checkpoint_utils import print_checkpoint_status
                    print_checkpoint_status(stage_dir, stage_name)


if __name__ == "__main__":
    main()

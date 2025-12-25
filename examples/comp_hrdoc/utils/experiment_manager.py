#!/usr/bin/env python
# coding=utf-8
"""
Experiment Manager for Comp_HRDoc training pipeline.

Provides experiment directory management with artifact tracking.

Directory Structure:
    artifact/
    ├── exp_20251210_103000/          # Experiment by timestamp
    │   ├── config.yml                # Config snapshot + state
    │   ├── intra_comp_hrdoc/         # Intra-region stage
    │   ├── order_comp_hrdoc/         # Order stage
    │   └── construct_comp_hrdoc/     # Construct stage
    └── current -> exp_20251210_103000/  # Symlink to active experiment
"""

import os
import yaml
import glob
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path


def convert_to_python_types(obj):
    """Convert numpy types to Python native types for YAML serialization."""
    try:
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    except ImportError:
        return obj


class ExperimentManager:
    """Manages experiments for Comp_HRDoc training pipeline."""

    # Stage name mapping
    STAGE_MAP = {
        "intra": "intra",
        "order": "order",
        "construct": "construct",
        "joint": "joint",
    }

    def __init__(self, artifact_root: str):
        """
        Initialize experiment manager.

        Args:
            artifact_root: Root directory for all experiments
        """
        self.artifact_root = artifact_root
        self.current_link = os.path.join(artifact_root, "current")
        os.makedirs(artifact_root, exist_ok=True)

    def _get_next_exp_id(self) -> int:
        """Get next experiment ID by scanning existing experiments."""
        exp_dirs = glob.glob(os.path.join(self.artifact_root, "exp_*"))
        if not exp_dirs:
            return 1

        max_id = 0
        for exp_dir in exp_dirs:
            config_path = os.path.join(exp_dir, "config.yml")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        exp_id = config.get('experiment', {}).get('id', 0)
                        max_id = max(max_id, exp_id)
                except:
                    pass
        return max_id + 1

    def _generate_exp_dirname(self) -> str:
        """Generate experiment directory name using timestamp."""
        return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def create_experiment(
        self,
        name: str = "",
        description: str = "",
        config: Optional[Dict] = None,
    ) -> str:
        """
        Create a new experiment directory.

        Args:
            name: Human-readable experiment name
            description: Experiment description
            config: Optional config dict to snapshot

        Returns:
            Path to the experiment directory
        """
        exp_dirname = self._generate_exp_dirname()
        exp_dir = os.path.join(self.artifact_root, exp_dirname)
        os.makedirs(exp_dir, exist_ok=True)

        exp_id = self._get_next_exp_id()

        exp_config = {
            'experiment': {
                'id': exp_id,
                'name': name or f"Experiment {exp_id}",
                'description': description,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            'config': config or {},
            'stages': {},
        }

        config_path = os.path.join(exp_dir, "config.yml")
        self._write_config(config_path, exp_config)
        self._update_current_link(exp_dir)

        print(f"Created experiment: {exp_dirname} (ID: {exp_id})")
        print(f"  Path: {exp_dir}")

        return exp_dir

    def get_experiment_dir(self, exp: Optional[str] = None) -> Optional[str]:
        """
        Get experiment directory path.

        Args:
            exp: Experiment identifier. Can be:
                - None or "current": Use current symlink
                - "exp_20251210_103000": Full directory name
                - "latest": Most recent experiment by timestamp

        Returns:
            Path to experiment directory, or None if not found
        """
        if exp is None or exp == "current":
            if os.path.islink(self.current_link):
                return os.path.realpath(self.current_link)
            exp = "latest"

        if exp == "latest":
            exp_dirs = sorted(glob.glob(os.path.join(self.artifact_root, "exp_*")))
            if exp_dirs:
                return exp_dirs[-1]
            return None

        exp_dir = os.path.join(self.artifact_root, exp)
        if os.path.isdir(exp_dir):
            return exp_dir

        return None

    def get_stage_dir(self, exp: Optional[str], stage: str, dataset: str) -> str:
        """
        Get stage output directory.

        Args:
            exp: Experiment identifier
            stage: Stage name ("intra", "order", "construct", "joint")
            dataset: Dataset name

        Returns:
            Full path to stage output directory
        """
        exp_dir = self.get_experiment_dir(exp)
        if not exp_dir:
            raise ValueError(f"Experiment not found: {exp}")

        stage_prefix = self.STAGE_MAP.get(stage, stage)
        stage_dir = os.path.join(exp_dir, f"{stage_prefix}_{dataset}")
        os.makedirs(stage_dir, exist_ok=True)
        return stage_dir

    def update_stage_state(
        self,
        exp: Optional[str],
        stage: str,
        dataset: str,
        status: Optional[str] = None,
        current_step: Optional[int] = None,
        best_checkpoint: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Update dynamic state for a training stage."""
        exp_dir = self.get_experiment_dir(exp)
        if not exp_dir:
            print(f"Warning: Experiment not found: {exp}")
            return

        config_path = os.path.join(exp_dir, "config.yml")
        if not os.path.exists(config_path):
            print(f"Warning: Config not found: {config_path}")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'stages' not in config:
            config['stages'] = {}

        stage_key = f"{stage}_{dataset}"
        if stage_key not in config['stages']:
            config['stages'][stage_key] = {
                'status': 'pending',
                'current_step': 0,
                'best_checkpoint': None,
                'metrics': {},
                'started_at': None,
                'completed_at': None,
            }

        stage_state = config['stages'][stage_key]

        if status is not None:
            stage_state['status'] = status
            if status == 'in_progress' and stage_state.get('started_at') is None:
                stage_state['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            elif status == 'completed':
                stage_state['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if current_step is not None:
            stage_state['current_step'] = current_step

        if best_checkpoint is not None:
            stage_state['best_checkpoint'] = best_checkpoint

        if metrics is not None:
            metrics = convert_to_python_types(metrics)
            stage_state['metrics'].update(metrics)

        config['experiment']['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._write_config(config_path, config)

    def mark_stage_started(self, exp: Optional[str], stage: str, dataset: str):
        """Mark a stage as started."""
        self.update_stage_state(exp, stage, dataset, status='in_progress')

    def mark_stage_completed(
        self,
        exp: Optional[str],
        stage: str,
        dataset: str,
        best_checkpoint: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Mark a stage as completed with final metrics."""
        self.update_stage_state(
            exp, stage, dataset,
            status='completed',
            best_checkpoint=best_checkpoint,
            metrics=metrics,
        )

    def mark_stage_failed(self, exp: Optional[str], stage: str, dataset: str):
        """Mark a stage as failed."""
        self.update_stage_state(exp, stage, dataset, status='failed')

    def list_experiments(self) -> List[Dict]:
        """List all experiments with basic info."""
        experiments = []
        exp_dirs = sorted(glob.glob(os.path.join(self.artifact_root, "exp_*")))

        current_exp = None
        if os.path.islink(self.current_link):
            current_exp = os.path.basename(os.path.realpath(self.current_link))

        for exp_dir in exp_dirs:
            exp_name = os.path.basename(exp_dir)
            config_path = os.path.join(exp_dir, "config.yml")

            info = {
                'dirname': exp_name,
                'path': exp_dir,
                'is_current': exp_name == current_exp,
            }

            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        info['id'] = config.get('experiment', {}).get('id', 0)
                        info['name'] = config.get('experiment', {}).get('name', '')
                        info['created'] = config.get('experiment', {}).get('created', '')
                        info['stages'] = list(config.get('stages', {}).keys())
                except:
                    pass

            experiments.append(info)

        return experiments

    def _update_current_link(self, exp_dir: str):
        """Update the 'current' symlink."""
        try:
            if os.path.islink(self.current_link):
                os.remove(self.current_link)
            elif os.path.exists(self.current_link):
                return
            os.symlink(exp_dir, self.current_link)
        except OSError as e:
            print(f"Warning: Could not create 'current' symlink: {e}")

    def set_current(self, exp: str):
        """Set the current experiment symlink."""
        exp_dir = self.get_experiment_dir(exp)
        if exp_dir:
            self._update_current_link(exp_dir)
            print(f"Set current experiment to: {os.path.basename(exp_dir)}")
        else:
            print(f"Experiment not found: {exp}")

    def _write_config(self, config_path: str, config: Dict):
        """Write config with atomic write."""
        import tempfile
        dir_path = os.path.dirname(config_path)

        fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix='.yml')
        try:
            with os.fdopen(fd, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            os.replace(temp_path, config_path)
        except:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


def get_artifact_path(env: str) -> str:
    """Get artifact directory path based on environment."""
    paths = {
        "dev": "/mnt/e/models/data/layoutlmft/artifact_comp_hrdoc",
        "test": "/data/LLM_group/layoutlmft/artifact_comp_hrdoc",
    }
    return paths.get(env, paths["dev"])


def ensure_experiment(
    artifact_root: str,
    exp: Optional[str] = None,
    new_exp: bool = False,
    name: str = "",
    description: str = "",
    config: Optional[Dict] = None,
) -> tuple:
    """
    Ensure experiment exists, create if needed.

    Args:
        artifact_root: Root directory for artifacts
        exp: Experiment identifier (None for current/latest)
        new_exp: Force create new experiment
        name: Experiment name (for new experiments)
        description: Experiment description
        config: Config to snapshot

    Returns:
        Tuple of (ExperimentManager, experiment_dir)
    """
    manager = ExperimentManager(artifact_root)

    if new_exp:
        exp_dir = manager.create_experiment(name=name, description=description, config=config)
    else:
        exp_dir = manager.get_experiment_dir(exp)
        if not exp_dir:
            print("No existing experiment found, creating new one...")
            exp_dir = manager.create_experiment(name=name, description=description, config=config)

    return manager, exp_dir

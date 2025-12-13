#!/usr/bin/env python
# coding=utf-8
"""
Experiment Manager for multi-stage training pipeline.

This module provides:
- Experiment creation and management
- Configuration snapshot (static config at experiment creation)
- Dynamic state tracking (stage status, checkpoints, metrics)
- Experiment directory structure management

Directory Structure:
    artifact/
    ├── exp_20251210_103000/          # Experiment by timestamp
    │   ├── config.yml                # Static config + dynamic state
    │   ├── stage1_hrds/
    │   ├── stage1_hrdh/
    │   ├── features_hrds/
    │   └── ...
    └── current -> exp_20251210_103000/  # Symlink to active experiment
"""

import os
import yaml
import glob
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path


def convert_to_python_types(obj):
    """Convert numpy types to Python native types for YAML serialization."""
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


@dataclass
class StageState:
    """Dynamic state for a training stage."""
    status: str = "pending"  # pending, in_progress, completed, failed
    current_step: int = 0
    best_checkpoint: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Experiment configuration and state."""
    # Experiment metadata
    id: int = 0
    name: str = ""
    description: str = ""
    created: str = ""
    last_run: str = ""

    # Static config snapshot (set at creation, not modified)
    model: Dict[str, Any] = field(default_factory=dict)
    stage1_training: Dict[str, Any] = field(default_factory=dict)
    feature_extraction: Dict[str, Any] = field(default_factory=dict)
    parent_finder: Dict[str, Any] = field(default_factory=dict)
    relation_classifier: Dict[str, Any] = field(default_factory=dict)
    datasets: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Dynamic state (updated during training)
    stages: Dict[str, Dict] = field(default_factory=dict)


class ExperimentManager:
    """Manages experiments for multi-stage training pipeline."""

    def __init__(self, output_root: str):
        """
        Initialize experiment manager.

        Args:
            output_root: Root directory for all experiments (e.g., /data/artifact)
        """
        self.output_root = output_root
        self.current_link = os.path.join(output_root, "current")

    def _get_next_exp_id(self) -> int:
        """Get next experiment ID by scanning existing experiments."""
        exp_dirs = glob.glob(os.path.join(self.output_root, "exp_*"))
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
        config: Any,  # Config object from config_loader
        name: str = "",
        description: str = "",
    ) -> str:
        """
        Create a new experiment directory with config snapshot.

        Args:
            config: Config object containing all training parameters
            name: Human-readable experiment name
            description: Experiment description

        Returns:
            Path to the experiment directory
        """
        # Generate experiment directory
        exp_dirname = self._generate_exp_dirname()
        exp_dir = os.path.join(self.output_root, exp_dirname)
        os.makedirs(exp_dir, exist_ok=True)

        # Get next experiment ID
        exp_id = self._get_next_exp_id()

        # Build static config snapshot
        exp_config = {
            'experiment': {
                'id': exp_id,
                'name': name or f"Experiment {exp_id}",
                'description': description,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_run': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'env': getattr(config, 'env', 'unknown'),
            },
            'model': self._extract_dict(config, 'model'),
            'stage1_training': self._extract_dict(config, 'stage1_training'),
            'feature_extraction': self._extract_dict(config, 'feature_extraction'),
            'parent_finder': self._extract_dict(config, 'parent_finder'),
            'relation_classifier': self._extract_dict(config, 'relation_classifier'),
            'datasets': getattr(config, 'datasets', {
                'hrds': {'description': 'HRDoc-Simple'},
                'hrdh': {'description': 'HRDoc-Hard'},
            }),
            'stages': {},  # Dynamic state, initially empty
        }

        # Write config snapshot
        config_path = os.path.join(exp_dir, "config.yml")
        self._write_config(config_path, exp_config)

        # Update current symlink
        self._update_current_link(exp_dir)

        print(f"Created experiment: {exp_dirname} (ID: {exp_id})")
        print(f"  Path: {exp_dir}")

        return exp_dir

    def _extract_dict(self, config: Any, attr_name: str) -> Dict:
        """Extract attribute as dict from config object."""
        attr = getattr(config, attr_name, None)
        if attr is None:
            return {}
        if hasattr(attr, '__dict__'):
            return {k: v for k, v in attr.__dict__.items() if not k.startswith('_')}
        if isinstance(attr, dict):
            return attr
        return {}

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
            # Use current symlink
            if os.path.islink(self.current_link):
                return os.path.realpath(self.current_link)
            # Fallback to latest
            exp = "latest"

        if exp == "latest":
            # Find most recent experiment
            exp_dirs = sorted(glob.glob(os.path.join(self.output_root, "exp_*")))
            if exp_dirs:
                return exp_dirs[-1]
            return None

        # Direct experiment name
        exp_dir = os.path.join(self.output_root, exp)
        if os.path.isdir(exp_dir):
            return exp_dir

        return None

    def get_stage_dir(self, exp: Optional[str], stage: str, dataset: str) -> str:
        """
        Get stage output directory for a specific dataset.

        Args:
            exp: Experiment identifier
            stage: Stage name ("stage1", "stage2", "stage3", "stage4")
            dataset: Dataset name ("hrds", "hrdh")

        Returns:
            Full path to stage output directory
        """
        exp_dir = self.get_experiment_dir(exp)
        if not exp_dir:
            raise ValueError(f"Experiment not found: {exp}")

        stage_map = {
            "stage1": "stage1",
            "stage2": "features",
            "stage3": "parent_finder",
            "stage4": "relation_classifier",
        }

        stage_prefix = stage_map.get(stage, stage)
        return os.path.join(exp_dir, f"{stage_prefix}_{dataset}")

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
        """
        Update dynamic state for a training stage.

        This is called at savepoints (eval + checkpoint save).
        Only updates provided fields, preserves others.

        Args:
            exp: Experiment identifier
            stage: Stage name
            dataset: Dataset name
            status: New status (pending, in_progress, completed, failed)
            current_step: Current training step
            best_checkpoint: Path to best checkpoint
            metrics: Latest metrics dict
        """
        exp_dir = self.get_experiment_dir(exp)
        if not exp_dir:
            print(f"Warning: Experiment not found: {exp}")
            return

        config_path = os.path.join(exp_dir, "config.yml")
        if not os.path.exists(config_path):
            print(f"Warning: Config not found: {config_path}")
            return

        # Read current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize stages dict if needed
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

        # Update provided fields
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
            # Convert numpy types to Python native types for YAML serialization
            metrics = convert_to_python_types(metrics)
            stage_state['metrics'].update(metrics)

        # Update last_run timestamp
        config['experiment']['last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write back (atomic write)
        self._write_config(config_path, config)

    def mark_stage_started(self, exp: Optional[str], stage: str, dataset: str):
        """Mark a stage as started (in_progress)."""
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

    def get_experiment_status(self, exp: Optional[str] = None) -> Dict:
        """
        Get full experiment status.

        Args:
            exp: Experiment identifier

        Returns:
            Dict with experiment config and all stage states
        """
        exp_dir = self.get_experiment_dir(exp)
        if not exp_dir:
            return {}

        config_path = os.path.join(exp_dir, "config.yml")
        if not os.path.exists(config_path):
            return {}

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def list_experiments(self) -> List[Dict]:
        """
        List all experiments with basic info.

        Returns:
            List of dicts with experiment info
        """
        experiments = []
        exp_dirs = sorted(glob.glob(os.path.join(self.output_root, "exp_*")))

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
        """Update the 'current' symlink to point to given experiment."""
        try:
            if os.path.islink(self.current_link):
                os.remove(self.current_link)
            elif os.path.exists(self.current_link):
                # Not a symlink but exists (shouldn't happen)
                return
            os.symlink(exp_dir, self.current_link)
        except OSError as e:
            # Symlink creation may fail on some systems
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
        """Write config with atomic write (temp file + rename)."""
        import tempfile

        dir_path = os.path.dirname(config_path)

        # Write to temp file first
        fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix='.yml')
        try:
            with os.fdopen(fd, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            # Atomic rename
            os.replace(temp_path, config_path)
        except:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


def get_experiment_manager(config: Any) -> ExperimentManager:
    """
    Create ExperimentManager from config object.

    Args:
        config: Config object with paths.output_dir

    Returns:
        ExperimentManager instance
    """
    output_root = getattr(config.paths, 'output_dir', '/tmp/artifact')
    return ExperimentManager(output_root)


# Convenience functions for use in training scripts

def ensure_experiment(
    config: Any,
    exp: Optional[str] = None,
    new_exp: bool = False,
    name: str = "",
    description: str = "",
) -> tuple:
    """
    Ensure experiment exists, create if needed.

    Args:
        config: Config object
        exp: Experiment identifier (None for current/latest)
        new_exp: Force create new experiment
        name: Experiment name (for new experiments)
        description: Experiment description

    Returns:
        Tuple of (ExperimentManager, experiment_dir)
    """
    manager = get_experiment_manager(config)

    if new_exp:
        exp_dir = manager.create_experiment(config, name=name, description=description)
    else:
        exp_dir = manager.get_experiment_dir(exp)
        if not exp_dir:
            # No existing experiment, create one
            print("No existing experiment found, creating new one...")
            exp_dir = manager.create_experiment(config, name=name, description=description)

    return manager, exp_dir

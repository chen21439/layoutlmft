#!/usr/bin/env python
# coding=utf-8
"""
Configuration Loader
Load and manage YAML configuration files for different environments.

Usage:
    from configs.config_loader import load_config, get_config

    # Load specific config
    config = load_config("test")  # loads configs/test.yml

    # Or auto-detect environment
    config = get_config()  # auto-detect based on hostname/env var

    # Access config values
    data_dir = config.paths.hrdoc_data_dir
    model_name = config.model.name_or_path
"""

import os
import sys
import yaml
import socket
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Get the configs directory path
CONFIGS_DIR = Path(__file__).parent


@dataclass
class GpuConfig:
    """GPU configuration"""
    cuda_visible_devices: Optional[str] = None  # e.g., "0", "0,1", "2"


@dataclass
class PathsConfig:
    """Data and model paths configuration"""
    hrdoc_data_dir: str = ""
    stage1_model_path: str = ""
    features_dir: str = ""
    output_dir: str = ""
    hf_cache_dir: str = ""


@dataclass
class ModelConfig:
    """Model configuration"""
    name_or_path: str = "microsoft/layoutxlm-base"
    local_path: Optional[str] = None


@dataclass
class MetricsConfig:
    """Metrics configuration"""
    seqeval_path: Optional[str] = None  # Local path to seqeval.py for offline mode


@dataclass
class Stage1TrainingConfig:
    """Stage 1 LayoutXLM fine-tuning parameters"""
    # Training loop
    max_steps: int = 500
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2

    # Optimizer
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"

    # Evaluation & checkpointing (默认开启评估)
    evaluation_strategy: str = "steps"  # "steps", "epoch", or "no"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True

    # Logging
    logging_steps: int = 100
    logging_first_step: bool = True

    # Stability & other
    fp16: bool = False
    dataloader_num_workers: int = 2
    seed: int = 42


@dataclass
class FeatureExtractionConfig:
    """Stage 2 feature extraction parameters"""
    batch_size: int = 50
    docs_per_chunk: int = 100
    num_samples: int = -1  # -1 means all


@dataclass
class ParentFinderConfig:
    """Stage 3 ParentFinder training parameters"""
    mode: str = "full"  # simple or full
    level: str = "document"  # page or document
    max_lines_limit: int = 512
    batch_size: int = 1
    num_epochs: int = 20
    learning_rate: float = 1e-4
    max_chunks: int = -1


@dataclass
class RelationClassifierConfig:
    """Stage 4 relation classifier training parameters"""
    max_steps: int = 300
    batch_size: int = 32
    learning_rate: float = 5e-4
    neg_ratio: float = 1.5
    max_chunks: int = -1


@dataclass
class QuickTestConfig:
    """Quick test configuration (overrides other settings)"""
    enabled: bool = False
    num_samples: int = 10
    docs_per_chunk: int = 5
    stage1_max_steps: int = 50
    parent_finder_epochs: int = 1
    relation_max_steps: int = 50


@dataclass
class InferenceConfig:
    """Inference configuration (FastAPI service)"""
    checkpoint_path: Optional[str] = None
    data_dir_base: Optional[str] = None
    construct_checkpoint: Optional[str] = None  # Construct 模型路径（可选）


# Import dataset configuration from examples/dataset_config
from examples.dataset_config.dataset_config import DATASET_DIR_MAP, DatasetConfig


@dataclass
class Config:
    """Main configuration class"""
    env: str = "dev"
    description: str = ""
    gpu: GpuConfig = field(default_factory=GpuConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    stage1_training: Stage1TrainingConfig = field(default_factory=Stage1TrainingConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    parent_finder: ParentFinderConfig = field(default_factory=ParentFinderConfig)
    relation_classifier: RelationClassifierConfig = field(default_factory=RelationClassifierConfig)
    quick_test: QuickTestConfig = field(default_factory=QuickTestConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def get_effective_config(self) -> 'Config':
        """Apply quick_test overrides if enabled"""
        if self.quick_test.enabled:
            # Override with quick test values
            self.feature_extraction.num_samples = self.quick_test.num_samples
            self.feature_extraction.docs_per_chunk = self.quick_test.docs_per_chunk
            self.stage1_training.max_steps = self.quick_test.stage1_max_steps
            self.parent_finder.num_epochs = self.quick_test.parent_finder_epochs
            self.relation_classifier.max_steps = self.quick_test.relation_max_steps
        return self

    def to_env_vars(self) -> Dict[str, str]:
        """Export config as environment variables (for backward compatibility)"""
        return {
            "HRDOC_DATA_DIR": self.paths.hrdoc_data_dir,
            "LAYOUTLMFT_MODEL_PATH": self.paths.stage1_model_path,
            "LAYOUTLMFT_FEATURES_DIR": self.paths.features_dir,
            "LAYOUTLMFT_OUTPUT_DIR": self.paths.output_dir,
            "LAYOUTLMFT_NUM_SAMPLES": str(self.feature_extraction.num_samples),
            "LAYOUTLMFT_DOCS_PER_CHUNK": str(self.feature_extraction.docs_per_chunk),
            "LAYOUTLMFT_BATCH_SIZE": str(self.feature_extraction.batch_size),
            "MAX_STEPS": str(self.relation_classifier.max_steps),
            "MAX_CHUNKS": str(self.relation_classifier.max_chunks),
        }

    def apply_to_env(self):
        """Apply config values to environment variables"""
        for key, value in self.to_env_vars().items():
            os.environ[key] = value

        # Set HuggingFace cache directory
        if self.paths.hf_cache_dir:
            os.environ["HF_HOME"] = self.paths.hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = self.paths.hf_cache_dir


def _dict_to_dataclass(data: Dict[str, Any], cls):
    """Convert dict to dataclass, handling nested structures"""
    if data is None:
        return cls()

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(env: str = "dev") -> Config:
    """
    Load configuration from YAML file.

    Args:
        env: Environment name (dev, test, prod, etc.)
             Loads configs/{env}.yml

    Returns:
        Config object with all settings
    """
    config_path = CONFIGS_DIR / f"{env}.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Parse dataset config
    dataset_data = data.get('dataset', {})
    dataset_config = DatasetConfig(
        name=dataset_data.get('name', 'hrds'),
        base_dir=dataset_data.get('base_dir', ''),
        covmatch=dataset_data.get('covmatch', 'doc_covmatch_dev10_seed42'),
    )

    # Build Config object
    config = Config(
        env=data.get('env', env),
        description=data.get('description', ''),
        gpu=_dict_to_dataclass(data.get('gpu'), GpuConfig),
        dataset=dataset_config,
        paths=_dict_to_dataclass(data.get('paths'), PathsConfig),
        model=_dict_to_dataclass(data.get('model'), ModelConfig),
        metrics=_dict_to_dataclass(data.get('metrics'), MetricsConfig),
        stage1_training=_dict_to_dataclass(data.get('stage1_training'), Stage1TrainingConfig),
        feature_extraction=_dict_to_dataclass(data.get('feature_extraction'), FeatureExtractionConfig),
        parent_finder=_dict_to_dataclass(data.get('parent_finder'), ParentFinderConfig),
        relation_classifier=_dict_to_dataclass(data.get('relation_classifier'), RelationClassifierConfig),
        quick_test=_dict_to_dataclass(data.get('quick_test'), QuickTestConfig),
        inference=_dict_to_dataclass(data.get('inference'), InferenceConfig),
    )

    return config


def detect_environment() -> str:
    """
    Auto-detect environment based on hostname and paths.

    Returns:
        Environment name: 'dev' or 'test'
    """
    # Check environment variable first
    env = os.getenv("LAYOUTLMFT_ENV")
    if env:
        return env

    hostname = socket.gethostname().lower()

    # Local development indicators
    if hostname == "mi" or os.path.exists("/mnt/e/models"):
        return "dev"

    # Cloud server indicators
    if "ubuntu" in hostname or os.path.exists("/data/LLM_group"):
        return "test"

    # Default to dev
    return "dev"


def get_config(env: Optional[str] = None, apply_quick_test: bool = True) -> Config:
    """
    Get configuration, auto-detecting environment if not specified.

    Args:
        env: Environment name, or None for auto-detect
        apply_quick_test: Whether to apply quick_test overrides if enabled

    Returns:
        Config object
    """
    if env is None:
        env = detect_environment()

    config = load_config(env)

    if apply_quick_test:
        config = config.get_effective_config()

    return config


def print_config(config: Config):
    """Print configuration summary"""
    print("=" * 60)
    print(f"Environment: {config.env}")
    print(f"Description: {config.description}")
    print("=" * 60)
    print("\nPaths:")
    print(f"  hrdoc_data_dir:    {config.paths.hrdoc_data_dir}")
    print(f"  stage1_model_path: {config.paths.stage1_model_path}")
    print(f"  features_dir:      {config.paths.features_dir}")
    print(f"  output_dir:        {config.paths.output_dir}")
    print(f"  hf_cache_dir:      {config.paths.hf_cache_dir}")
    print("\nModel:")
    print(f"  name_or_path: {config.model.name_or_path}")
    print(f"  local_path:   {config.model.local_path}")
    print("\nQuick Test:")
    print(f"  enabled: {config.quick_test.enabled}")
    if config.quick_test.enabled:
        print(f"  num_samples:    {config.quick_test.num_samples}")
        print(f"  stage1_steps:   {config.quick_test.stage1_max_steps}")
        print(f"  pf_epochs:      {config.quick_test.parent_finder_epochs}")
        print(f"  relation_steps: {config.quick_test.relation_max_steps}")
    print("=" * 60)


if __name__ == "__main__":
    # Test loading configs
    print("Testing config loader...\n")

    for env in ["dev", "test"]:
        try:
            config = load_config(env)
            print_config(config)
            print()
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    # Test auto-detection
    print("\nAuto-detected environment:", detect_environment())

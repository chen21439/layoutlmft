"""
Data loading utilities for HRDoc and other datasets.

This module provides:
- Dataset path configuration and resolution
- Data directory mapping for different datasets
- Covmatch split directory utilities
"""

from .dataset_config import (
    DATASET_DIR_MAP,
    DatasetConfig,
    get_data_dir,
    get_covmatch_dir,
)

__all__ = [
    "DATASET_DIR_MAP",
    "DatasetConfig",
    "get_data_dir",
    "get_covmatch_dir",
]

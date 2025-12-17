"""
Dataset configuration and path utilities.

This module centralizes all dataset path logic for HRDoc and other datasets.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# Dataset name to directory name mapping
# Keys are the short names used in CLI (--dataset hrds)
# Values are the actual directory names on disk
DATASET_DIR_MAP = {
    "hrds": "HRDS",
    "hrdh": "HRDH",
    "tender": "tender_document",
}


@dataclass
class DatasetConfig:
    """
    Dataset configuration with path resolution.

    Attributes:
        name: Default dataset to use (hrds, hrdh, tender)
        base_dir: Base directory containing all dataset folders
        covmatch: Covmatch split directory name for train/dev split

    Example:
        config = DatasetConfig(
            name="hrds",
            base_dir="/data/datasets",
            covmatch="doc_covmatch_dev10_seed42"
        )

        # Get data directory
        config.get_data_dir("hrds")  # -> "/data/datasets/HRDS"
        config.get_data_dir("hrdh")  # -> "/data/datasets/HRDH"

        # Get covmatch directory
        config.get_covmatch_dir("hrds")  # -> "/data/datasets/HRDS/covmatch/doc_covmatch_dev10_seed42"
    """
    name: str = "hrds"  # Which dataset to use by default
    base_dir: str = ""  # Base directory for all datasets
    covmatch: str = "doc_covmatch_dev10_seed42"  # Which covmatch split to use

    def get_data_dir(self, dataset_name: Optional[str] = None) -> str:
        """
        Get the data directory path for a dataset.

        Used by:
            - Stage 1: load raw document data
            - Stage 2: load raw document data for feature extraction
            - Joint training: load raw document data
            - get_covmatch_dir(): base path for covmatch splits

        Args:
            dataset_name: Dataset name (hrds, hrdh, tender).
                         If None, uses self.name (default dataset).

        Returns:
            Absolute path to the dataset directory.
            E.g., "/data/datasets/HRDS" for hrds dataset.
        """
        name = dataset_name or self.name
        dir_name = DATASET_DIR_MAP.get(name, name)
        return os.path.join(self.base_dir, dir_name)

    def get_covmatch_dir(self, dataset_name: Optional[str] = None) -> str:
        """
        Get the covmatch split directory path for a dataset.

        Covmatch directories contain train/dev splits created by the
        create_dev_covmatch.py script.

        Note: This method internally calls get_data_dir(), so any fix to
        get_data_dir() will automatically apply here.

        Used by:
            - Stage 1: train/dev split for training
            - Stage 3: train/dev split for ParentFinder
            - Stage 4: train/dev split for RelationClassifier
            - Joint training: train/dev split

        Args:
            dataset_name: Dataset name. If None, uses default.

        Returns:
            Path to covmatch directory.
            E.g., "/data/datasets/HRDS/covmatch/doc_covmatch_dev10_seed42"
        """
        data_dir = self.get_data_dir(dataset_name)
        return os.path.join(data_dir, "covmatch", self.covmatch)

    def validate(self, dataset_name: Optional[str] = None) -> bool:
        """
        Check if the dataset directory exists.

        Args:
            dataset_name: Dataset to validate. If None, uses default.

        Returns:
            True if directory exists, False otherwise.
        """
        data_dir = self.get_data_dir(dataset_name)
        return os.path.isdir(data_dir)

    def get_available_datasets(self) -> list:
        """
        Get list of available datasets (directories that exist).

        Returns:
            List of dataset names that have existing directories.
        """
        available = []
        for name in DATASET_DIR_MAP.keys():
            if self.validate(name):
                available.append(name)
        return available


# Convenience functions for use without DatasetConfig instance

def get_data_dir(base_dir: str, dataset_name: str) -> str:
    """
    Get data directory path for a dataset.

    Args:
        base_dir: Base directory containing all datasets
        dataset_name: Dataset name (hrds, hrdh, tender)

    Returns:
        Full path to dataset directory
    """
    dir_name = DATASET_DIR_MAP.get(dataset_name, dataset_name)
    return os.path.join(base_dir, dir_name)


def get_covmatch_dir(base_dir: str, dataset_name: str,
                     covmatch: str = "doc_covmatch_dev10_seed42") -> str:
    """
    Get covmatch split directory path for a dataset.

    Args:
        base_dir: Base directory containing all datasets
        dataset_name: Dataset name (hrds, hrdh, tender)
        covmatch: Covmatch split name

    Returns:
        Full path to covmatch directory
    """
    data_dir = get_data_dir(base_dir, dataset_name)
    return os.path.join(data_dir, "covmatch", covmatch)

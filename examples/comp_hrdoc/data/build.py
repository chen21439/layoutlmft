#!/usr/bin/env python
# coding=utf-8
"""
数据加载器构建工厂函数 - 遵循项目结构规范

data/ 负责数据加载与预处理
"""

import logging
from pathlib import Path
from torch.utils.data import DataLoader

from examples.stage.data.hrdoc_data_loader import HRDocDataLoader, HRDocDataLoaderConfig
from examples.stage.joint_data_collator import HRDocDocumentLevelCollator
from examples.comp_hrdoc.utils.label_utils import convert_stage_labels_to_construct

logger = logging.getLogger(__name__)


def build_dataloader(
    config: dict,
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    max_samples: int = None,
    document_level: bool = True,
    max_lines: int = 1024,
    force_rebuild: bool = False,
):
    """
    构建数据加载器（复用 stage 的 HRDocDataLoader）

    Args:
        config: 环境配置字典（包含 data_dir）
        tokenizer: LayoutXLM tokenizer
        split: "train" 或 "val"
        batch_size: 批次大小
        max_samples: 最大样本数（用于调试）
        document_level: 是否使用文档级别模式（支持多 chunk）
        max_lines: 最大行数
        force_rebuild: 是否强制重建缓存

    Returns:
        DataLoader
    """
    # 从 config 获取数据目录
    data_dirs = config.get("data", {}).get("data_dirs", {})
    dataset_name = config.get("data", {}).get("dataset_name", "hrds")

    # 选择数据集目录
    if dataset_name in data_dirs:
        data_dir = data_dirs[dataset_name]
    else:
        raise ValueError(f"Dataset {dataset_name} not found in config")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    logger.info(f"Building {split} dataloader from: {data_dir}")

    # 构建 HRDocDataLoader config
    data_loader_config = HRDocDataLoaderConfig(
        data_dir=str(data_dir),
        dataset_name=dataset_name,
        document_level=document_level,
        max_length=512,
        max_train_samples=max_samples if split == "train" else None,
        max_val_samples=max_samples if split == "val" else None,
        force_rebuild=force_rebuild,
    )

    # 创建数据加载器
    data_loader = HRDocDataLoader(tokenizer, data_loader_config)
    datasets = data_loader.prepare_datasets()

    # 获取对应 split 的数据集
    if split == "train":
        dataset = datasets.get("train", [])
    elif split == "val" or split == "validation":
        dataset = datasets.get("validation", [])
    else:
        raise ValueError(f"Unknown split: {split}")

    logger.info(f"{split.capitalize()} dataset: {len(dataset)} documents")

    # 创建 collator
    collator = HRDocDocumentLevelCollator(tokenizer)

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collator,
        num_workers=0,
    )

    return dataloader

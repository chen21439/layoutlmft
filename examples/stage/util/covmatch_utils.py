#!/usr/bin/env python
# coding=utf-8
"""
Covmatch 工具函数

提供 train/validation 数据分割的公共逻辑，供所有 Stage 使用：
- Stage 1: 通过环境变量 HRDOC_SPLIT_DIR 使用 covmatch 分割训练数据
- Stage 2: 全量转换所有训练数据的特征
- Stage 3/4: 根据 covmatch doc_ids 过滤特征，分为 train/validation

Covmatch 是一种基于文档级别的数据分割方式，确保同一文档的所有页面在同一个 split 中。
"""

import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def load_covmatch_doc_ids(covmatch_dir: str) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
    """
    从 covmatch 目录加载 train/dev 文档 ID 集合

    Args:
        covmatch_dir: covmatch 分割目录路径，包含 train_doc_ids.json 和 dev_doc_ids.json

    Returns:
        (train_doc_ids, dev_doc_ids): 两个集合，如果文件不存在则返回 (None, None)
    """
    if not covmatch_dir or not os.path.exists(covmatch_dir):
        logger.warning(f"Covmatch 目录不存在: {covmatch_dir}")
        return None, None

    train_doc_ids = None
    dev_doc_ids = None

    train_ids_file = os.path.join(covmatch_dir, "train_doc_ids.json")
    dev_ids_file = os.path.join(covmatch_dir, "dev_doc_ids.json")

    if os.path.exists(train_ids_file):
        with open(train_ids_file, 'r') as f:
            train_doc_ids = set(json.load(f))
        logger.info(f"Loaded {len(train_doc_ids)} train doc IDs from {train_ids_file}")

    if os.path.exists(dev_ids_file):
        with open(dev_ids_file, 'r') as f:
            dev_doc_ids = set(json.load(f))
        logger.info(f"Loaded {len(dev_doc_ids)} dev doc IDs from {dev_ids_file}")

    return train_doc_ids, dev_doc_ids


def filter_features_by_doc_ids(
    features_list: List[Dict],
    doc_ids: Set[str],
    doc_name_key: str = "document_name"
) -> List[Dict]:
    """
    根据文档 ID 集合过滤特征列表

    Args:
        features_list: 特征字典列表，每个字典代表一个文档
        doc_ids: 要保留的文档 ID 集合
        doc_name_key: 特征字典中文档名称的键名

    Returns:
        过滤后的特征列表
    """
    if doc_ids is None:
        return features_list

    filtered = [f for f in features_list if f.get(doc_name_key) in doc_ids]
    logger.info(f"Filtered {len(features_list)} -> {len(filtered)} documents")
    return filtered


def load_features_with_covmatch(
    features_dir: str,
    covmatch_dir: Optional[str] = None,
    split: str = "train",
    max_chunks: Optional[int] = None,
    doc_name_key: str = "document_name"
) -> List[Dict]:
    """
    加载特征文件并根据 covmatch 分割过滤

    Stage 2 生成的特征文件是全量的 train 数据。
    此函数根据 covmatch 的 doc_ids 过滤出对应 split 的数据。

    Args:
        features_dir: 特征文件目录
        covmatch_dir: covmatch 分割目录（如果为 None，则不过滤）
        split: 数据分割，"train" 或 "validation"
        max_chunks: 最大加载的 chunk 数量（用于测试）
        doc_name_key: 特征字典中文档名称的键名

    Returns:
        特征字典列表
    """
    import glob
    import pickle

    # 加载 covmatch doc_ids
    train_doc_ids, dev_doc_ids = None, None
    if covmatch_dir:
        train_doc_ids, dev_doc_ids = load_covmatch_doc_ids(covmatch_dir)

    # 确定要使用的 doc_ids
    if split == "train":
        target_doc_ids = train_doc_ids
    elif split == "validation":
        target_doc_ids = dev_doc_ids
    else:
        target_doc_ids = None

    # 加载特征文件（始终加载 train 的特征文件，因为 Stage 2 只生成 train）
    # 如果 covmatch 未启用，尝试加载对应 split 的文件
    if target_doc_ids is not None:
        # covmatch 模式：从 train 特征中过滤
        feature_split = "train"
    else:
        # 非 covmatch 模式：直接加载对应 split
        feature_split = split

    single_file = os.path.join(features_dir, f"{feature_split}_line_features.pkl")

    features_list = []

    if os.path.exists(single_file):
        logger.info(f"加载单个特征文件: {single_file}")
        with open(single_file, "rb") as f:
            features_list = pickle.load(f)
    else:
        pattern = os.path.join(features_dir, f"{feature_split}_line_features_chunk_*.pkl")
        chunk_files = sorted(glob.glob(pattern))

        if max_chunks is not None and max_chunks > 0:
            chunk_files = chunk_files[:max_chunks]

        if len(chunk_files) == 0:
            raise ValueError(f"没有找到特征文件: {single_file} 或 {pattern}")

        logger.info(f"找到 {len(chunk_files)} 个 chunk 文件")
        for chunk_file in chunk_files:
            logger.info(f"  加载 {os.path.basename(chunk_file)}...")
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)
            features_list.extend(chunk_data)
            logger.info(f"    累计 {len(features_list)} 个文档")

    logger.info(f"总共加载了 {len(features_list)} 个文档的特征")

    # 根据 covmatch 过滤
    if target_doc_ids is not None:
        features_list = filter_features_by_doc_ids(features_list, target_doc_ids, doc_name_key)
        logger.info(f"Covmatch 过滤后: {len(features_list)} 个 {split} 文档")

    return features_list


class CovmatchFeatureLoader:
    """
    支持 Covmatch 的特征加载器

    封装了特征加载和 covmatch 过滤的逻辑，提供统一的接口。
    """

    def __init__(
        self,
        features_dir: str,
        covmatch_dir: Optional[str] = None,
        max_chunks: Optional[int] = None,
        doc_name_key: str = "document_name"
    ):
        """
        Args:
            features_dir: 特征文件目录
            covmatch_dir: covmatch 分割目录
            max_chunks: 最大加载的 chunk 数量
            doc_name_key: 特征字典中文档名称的键名
        """
        self.features_dir = features_dir
        self.covmatch_dir = covmatch_dir
        self.max_chunks = max_chunks
        self.doc_name_key = doc_name_key

        # 加载 covmatch doc_ids
        self.train_doc_ids, self.dev_doc_ids = None, None
        if covmatch_dir:
            self.train_doc_ids, self.dev_doc_ids = load_covmatch_doc_ids(covmatch_dir)

        self.use_covmatch = self.train_doc_ids is not None and self.dev_doc_ids is not None

        if self.use_covmatch:
            logger.info(f"Covmatch 已启用: train={len(self.train_doc_ids)}, dev={len(self.dev_doc_ids)}")
        else:
            logger.info("Covmatch 未启用，使用原始数据分割")

    def load_split(self, split: str) -> List[Dict]:
        """
        加载指定 split 的特征

        Args:
            split: "train" 或 "validation"

        Returns:
            特征字典列表
        """
        return load_features_with_covmatch(
            features_dir=self.features_dir,
            covmatch_dir=self.covmatch_dir,
            split=split,
            max_chunks=self.max_chunks,
            doc_name_key=self.doc_name_key
        )

    def get_train_features(self) -> List[Dict]:
        """加载训练集特征"""
        return self.load_split("train")

    def get_validation_features(self) -> List[Dict]:
        """加载验证集特征"""
        return self.load_split("validation")

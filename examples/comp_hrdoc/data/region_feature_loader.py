"""预提取区域特征数据加载器

加载由 extract_region_features.py 提取的区域级特征
用于 Order 模块 (4.3) 的独立训练
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ==================== 路径配置 ====================

def get_features_path(env: str) -> str:
    """获取特征目录路径"""
    paths = {
        "dev": "/mnt/e/models/data/Section/Comp_HRDoc/region_features",
        "test": "/data/LLM_group/layoutlmft/data/Comp_HRDoc/region_features",
    }
    return paths.get(env, paths["dev"])


# ==================== 数据集配置 ====================

@dataclass
class RegionFeatureConfig:
    """区域特征数据集配置"""
    env: str = "dev"
    features_dir: Optional[str] = None
    max_regions: int = 128
    max_samples: Optional[int] = None


# ==================== 数据集 ====================

class RegionFeatureDataset(Dataset):
    """预提取区域特征数据集

    加载 pickle 格式的预提取特征文件
    """

    def __init__(
        self,
        config: RegionFeatureConfig,
        split: str = "train",
    ):
        """
        Args:
            config: 数据集配置
            split: 数据划分 ("train" 或 "validation")
        """
        self.config = config
        self.split = split

        # 确定特征目录
        self.features_dir = config.features_dir or get_features_path(config.env)

        # 加载元数据
        metadata_path = os.path.join(self.features_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.hidden_size = self.metadata.get("hidden_size", 768)
        else:
            self.metadata = {}
            self.hidden_size = 768

        # 加载所有 chunk 文件
        self.samples = self._load_chunks()

        logger.info(f"Loaded {len(self.samples)} samples for {split}")
        logger.info(f"Features dir: {self.features_dir}")
        logger.info(f"Hidden size: {self.hidden_size}")

    def _load_chunks(self) -> List[Dict]:
        """加载所有 chunk 文件"""
        samples = []

        # 查找所有 chunk 文件
        prefix = f"{self.split}_region_features_chunk_"
        chunk_files = sorted([
            f for f in os.listdir(self.features_dir)
            if f.startswith(prefix) and f.endswith(".pkl")
        ])

        if not chunk_files:
            logger.warning(f"No chunk files found for {self.split} in {self.features_dir}")
            return samples

        logger.info(f"Found {len(chunk_files)} chunk files for {self.split}")

        for chunk_file in chunk_files:
            chunk_path = os.path.join(self.features_dir, chunk_file)
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)
            samples.extend(chunk_data)

            # 限制样本数
            if self.config.max_samples and len(samples) >= self.config.max_samples:
                samples = samples[:self.config.max_samples]
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        return {
            'region_features': sample['region_features'],  # [num_regions, hidden_size]
            'region_mask': sample['region_mask'],  # [num_regions]
            'bboxes': sample['bboxes'],  # [num_regions, 4]
            'categories': sample['categories'],  # [num_regions]
            'reading_orders': sample['reading_orders'],  # [num_regions]
            'parent_ids': sample['parent_ids'],  # [num_regions]
            'relations': sample['relations'],  # [num_regions]
            'num_regions': sample['num_regions'],
            'image_id': sample.get('image_id', idx),
            'image_file': sample.get('image_file', ''),
        }


# ==================== Collator ====================

class RegionFeatureCollator:
    """区域特征数据 Collator

    将变长的区域特征 padding 到相同长度
    """

    def __init__(self, max_regions: int = 128, hidden_size: int = 768):
        self.max_regions = max_regions
        self.hidden_size = hidden_size

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        # 找出最大区域数
        max_regions = min(
            max(f['num_regions'] for f in features),
            self.max_regions
        )

        # 初始化 batch tensors
        region_features = torch.zeros(batch_size, max_regions, self.hidden_size)
        region_mask = torch.zeros(batch_size, max_regions, dtype=torch.bool)
        bboxes = torch.zeros(batch_size, max_regions, 4)
        categories = torch.zeros(batch_size, max_regions, dtype=torch.long)
        reading_orders = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        parent_ids = torch.full((batch_size, max_regions), -1, dtype=torch.long)
        relations = torch.full((batch_size, max_regions), -1, dtype=torch.long)

        num_regions_list = []
        image_files = []

        for i, f in enumerate(features):
            n = min(f['num_regions'], max_regions)
            num_regions_list.append(n)
            image_files.append(f['image_file'])

            region_features[i, :n] = f['region_features'][:n]
            region_mask[i, :n] = f['region_mask'][:n]
            bboxes[i, :n] = f['bboxes'][:n]
            categories[i, :n] = f['categories'][:n]
            reading_orders[i, :n] = f['reading_orders'][:n]
            relations[i, :n] = f['relations'][:n]

            # 论文自指向方案：-1（root）转换为 self-index
            sample_parent_ids = f['parent_ids'][:n]
            for j in range(n):
                p = sample_parent_ids[j].item() if hasattr(sample_parent_ids[j], 'item') else sample_parent_ids[j]
                if p == -1:
                    parent_ids[i, j] = j  # root 自指向
                else:
                    parent_ids[i, j] = p

        return {
            'region_features': region_features,  # [batch, max_regions, hidden_size]
            'region_mask': region_mask,  # [batch, max_regions]
            'bboxes': bboxes,  # [batch, max_regions, 4]
            'categories': categories,  # [batch, max_regions]
            'reading_orders': reading_orders,  # [batch, max_regions]
            'parent_ids': parent_ids,  # [batch, max_regions]
            'relations': relations,  # [batch, max_regions]
            'num_regions': num_regions_list,
            'image_files': image_files,
            'batch_size': batch_size,
        }


# ==================== 工具函数 ====================

def create_region_feature_dataloaders(
    config: RegionFeatureConfig,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader

    Args:
        config: 数据集配置
        batch_size: 批次大小
        num_workers: 数据加载进程数

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = RegionFeatureDataset(config, split="train")
    val_dataset = RegionFeatureDataset(config, split="validation")

    hidden_size = train_dataset.hidden_size

    collator = RegionFeatureCollator(
        max_regions=config.max_regions,
        hidden_size=hidden_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ==================== 测试 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试 dev 环境
    config = RegionFeatureConfig(env="dev", max_samples=10)

    print("=== 测试数据加载 ===")
    try:
        dataset = RegionFeatureDataset(config, split="train")
        print(f"Train samples: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n=== 第一个样本 ===")
            print(f"region_features shape: {sample['region_features'].shape}")
            print(f"num_regions: {sample['num_regions']}")
            print(f"reading_orders: {sample['reading_orders'][:5]}...")
            print(f"categories: {sample['categories'][:5]}...")

        # 测试 collator
        print("\n=== 测试 Collator ===")
        collator = RegionFeatureCollator(hidden_size=dataset.hidden_size)
        batch = collator([dataset[i] for i in range(min(2, len(dataset)))])
        print(f"Batch keys: {list(batch.keys())}")
        print(f"region_features shape: {batch['region_features'].shape}")
        print(f"reading_orders shape: {batch['reading_orders'].shape}")

    except Exception as e:
        print(f"Error: {e}")
        print("Features may not be extracted yet. Run extract_region_features.py first.")

#!/usr/bin/env python
# coding=utf-8
"""
示例：如何使用分块保存的特征文件进行训练
支持按需加载chunk，降低内存占用
"""

import os
import pickle
import glob
import torch
from torch.utils.data import Dataset, DataLoader


class ChunkedLineFeatureDataset(Dataset):
    """从多个chunk文件按需加载特征的Dataset"""

    def __init__(self, chunk_dir, split="train"):
        """
        Args:
            chunk_dir: chunk文件所在目录
            split: train 或 validation
        """
        self.chunk_dir = chunk_dir
        self.split = split

        # 找到所有chunk文件
        pattern = os.path.join(chunk_dir, f"{split}_line_features_chunk_*.pkl")
        self.chunk_files = sorted(glob.glob(pattern))

        if len(self.chunk_files) == 0:
            raise ValueError(f"没有找到chunk文件: {pattern}")

        print(f"找到 {len(self.chunk_files)} 个{split}集chunk文件")

        # 扫描所有chunk，记录每个样本在哪个chunk的哪个位置
        self.sample_index = []  # [(chunk_idx, sample_idx_in_chunk), ...]
        self.current_chunk_idx = None
        self.current_chunk_data = None

        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)
            num_samples = len(chunk_data)
            print(f"  Chunk {chunk_idx}: {num_samples} 个样本")

            for sample_idx in range(num_samples):
                self.sample_index.append((chunk_idx, sample_idx))

        print(f"总共 {len(self.sample_index)} 个样本")

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        """按需加载样本，只在需要时加载对应的chunk"""
        chunk_idx, sample_idx = self.sample_index[idx]

        # 如果需要的chunk不是当前加载的，则重新加载
        if self.current_chunk_idx != chunk_idx:
            chunk_file = self.chunk_files[chunk_idx]
            with open(chunk_file, "rb") as f:
                self.current_chunk_data = pickle.load(f)
            self.current_chunk_idx = chunk_idx

        # 从当前chunk中获取样本
        page_data = self.current_chunk_data[sample_idx]

        return {
            "line_features": page_data["line_features"],  # [num_lines, 768]
            "line_mask": page_data["line_mask"],  # [num_lines]
            "line_parent_ids": page_data["line_parent_ids"],  # [num_lines]
            "line_relations": page_data["line_relations"],  # [num_lines]
            "line_bboxes": page_data["line_bboxes"],  # [num_lines, 4]
            "page_idx": page_data["page_idx"],
        }


def collate_fn(batch):
    """将batch中的样本collate到一起（处理不同长度）"""
    # 这里简单返回list，实际训练时需要根据模型需求处理
    return batch


def main():
    """使用示例"""
    # 特征文件目录
    chunk_dir = "/mnt/e/models/train_data/layoutlmft/line_features"

    # 创建训练集Dataset
    train_dataset = ChunkedLineFeatureDataset(chunk_dir, split="train")

    # 创建DataLoader（可以多进程）
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,  # 可以并行加载
        collate_fn=collate_fn,
    )

    # 测试加载
    print("\n测试加载前3个batch:")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        print(f"\nBatch {i}:")
        print(f"  样本数: {len(batch)}")
        for j, sample in enumerate(batch[:2]):  # 只打印前2个样本
            print(f"  样本{j}: {sample['line_features'].shape}, "
                  f"page_idx={sample['page_idx']}")


if __name__ == "__main__":
    main()

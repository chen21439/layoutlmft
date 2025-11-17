#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 HRDoc FUNSD 格式数据中的所有标签
"""

import json
import os
from collections import Counter
from pathlib import Path

def check_labels(data_dir):
    """检查数据目录中的所有标签"""
    ann_dir = Path(data_dir) / "train" / "annotations"

    if not ann_dir.exists():
        print(f"❌ 目录不存在: {ann_dir}")
        return

    labels = Counter()
    files_with_issues = []

    print(f"扫描目录: {ann_dir}")
    print("=" * 60)

    json_files = list(ann_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个标注文件\n")

    for idx, filepath in enumerate(json_files, 1):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for item in data.get('form', []):
                label = item.get('label', 'UNKNOWN')
                labels[label.upper()] += 1

        except Exception as e:
            files_with_issues.append((filepath.name, str(e)))

        # 进度显示
        if idx % 100 == 0:
            print(f"已处理: {idx}/{len(json_files)}")

    print("\n" + "=" * 60)
    print(f"扫描完成！共处理 {len(json_files)} 个文件")
    print("=" * 60)

    # 显示标签统计
    print(f"\n发现 {len(labels)} 个不同的标签:")
    print("-" * 60)
    for label in sorted(labels.keys()):
        count = labels[label]
        print(f"  {label:20s} : {count:6d} 次")

    # 生成 B-/I- 标签对
    print("\n" + "=" * 60)
    print("需要的 BIO 标签对:")
    print("-" * 60)

    base_labels = set()
    for label in labels.keys():
        if label != 'O' and label != 'UNKNOWN':
            base_labels.add(label)

    bio_labels = ["O"]
    for base in sorted(base_labels):
        bio_labels.append(f"B-{base}")
        bio_labels.append(f"I-{base}")

    print("_LABELS = [")
    for label in bio_labels:
        print(f'    "{label}",')
    print("]")

    # 显示问题文件
    if files_with_issues:
        print("\n" + "=" * 60)
        print(f"有问题的文件 ({len(files_with_issues)} 个):")
        print("-" * 60)
        for filename, error in files_with_issues[:10]:
            print(f"  {filename}: {error}")
        if len(files_with_issues) > 10:
            print(f"  ... 还有 {len(files_with_issues) - 10} 个")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="检查 HRDoc 数据标签")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/linux/code/layoutlmft/data/hrdoc_funsd_format",
        help="数据目录路径"
    )

    args = parser.parse_args()
    check_labels(args.data_dir)

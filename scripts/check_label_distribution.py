#!/usr/bin/env python
# coding=utf-8
"""
检查 HRDoc 数据集中的标签分布

Usage:
    python scripts/check_label_distribution.py /path/to/HRDH
    python scripts/check_label_distribution.py /path/to/HRDS
"""

import os
import sys
import json
from collections import Counter

def check_dataset(data_dir):
    """统计数据集中各标签的数量"""
    print(f"\n{'='*60}")
    print(f"Dataset: {data_dir}")
    print(f"{'='*60}")

    for split in ["train", "test", "validation"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"\n[{split}] Directory not found: {split_dir}")
            continue

        label_counter = Counter()
        file_count = 0
        line_count = 0

        # 遍历所有 JSON 文件
        for filename in os.listdir(split_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(split_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                file_count += 1

                # 处理不同的数据格式
                lines = []
                if isinstance(data, list):
                    # 直接是 line 列表
                    lines = data
                elif isinstance(data, dict):
                    # 可能有 pages 结构
                    if "pages" in data:
                        for page in data["pages"]:
                            if "lines" in page:
                                lines.extend(page["lines"])
                            elif "units" in page:
                                lines.extend(page["units"])
                    elif "lines" in data:
                        lines = data["lines"]
                    elif "units" in data:
                        lines = data["units"]

                # 统计标签
                for line in lines:
                    line_count += 1
                    # 尝试不同的标签字段名
                    label = line.get("class") or line.get("label") or line.get("category") or "unknown"
                    label = label.lower().strip()
                    label_counter[label] += 1

            except Exception as e:
                print(f"  Error reading {filename}: {e}")

        print(f"\n[{split}] Files: {file_count}, Lines: {line_count}")
        print(f"{'-'*40}")
        print(f"{'Label':<15} {'Count':>8} {'Percent':>10}")
        print(f"{'-'*40}")

        # 按数量排序输出
        for label, count in sorted(label_counter.items(), key=lambda x: -x[1]):
            percent = count / line_count * 100 if line_count > 0 else 0
            print(f"{label:<15} {count:>8} {percent:>9.2f}%")

        print(f"{'-'*40}")
        print(f"Total unique labels: {len(label_counter)}")

        # 检查是否有 header
        if "header" in label_counter:
            print(f">>> 'header' found: {label_counter['header']} samples")
        else:
            print(f">>> 'header' NOT found in this split")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_label_distribution.py <data_dir>")
        print("\nExamples:")
        print("  python scripts/check_label_distribution.py /data/LLM_group/layoutlmft/data/HRDH")
        print("  python scripts/check_label_distribution.py /data/LLM_group/layoutlmft/data/HRDS")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    check_dataset(data_dir)

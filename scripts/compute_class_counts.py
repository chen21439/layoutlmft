#!/usr/bin/env python
# coding=utf-8
"""
统计训练数据的类别分布（line-level）

用法:
    python scripts/compute_class_counts.py --dataset hrdh
    python scripts/compute_class_counts.py --dataset hrds
    python scripts/compute_class_counts.py --dataset tender
    python scripts/compute_class_counts.py --all

输出到: configs/class_counts.yml
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter
from argparse import ArgumentParser

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from layoutlmft.data.labels import LABEL_LIST, trans_class

# 数据集目录映射
DATASET_DIRS = {
    "hrds": "HRDS",
    "hrdh": "HRDH",
    "tender": "tender_document",
}


def get_data_base_dir():
    """获取数据根目录（从环境变量或默认路径）"""
    # 尝试从配置加载
    try:
        from configs.config_loader import load_config
        config = load_config("test")
        return config.dataset.base_dir
    except Exception:
        pass

    # 默认路径
    default_paths = [
        "/data/LLM_group/layoutlmft/data",  # 服务器
        "/mnt/e/models/data/Section",        # 本地 WSL
    ]
    for path in default_paths:
        if os.path.exists(path):
            return path

    raise RuntimeError("Cannot find data base directory. Set HRDOC_DATA_DIR or check paths.")


def count_classes_in_file(filepath: Path) -> Counter:
    """统计单个 JSON 文件中的类别分布（使用标准化标签）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    counter = Counter()
    for item in data:
        raw_cls = item.get('class', item.get('label', 'unknown'))
        # 使用 trans_class 转换为标准 14 类（opara 需要整个文档数据）
        std_cls = trans_class(raw_cls, all_lines=data, unit=item)
        counter[std_cls] += 1

    return counter


def count_classes_in_dataset(dataset: str, base_dir: str) -> dict:
    """统计数据集的类别分布"""
    dataset_dir = DATASET_DIRS.get(dataset)
    if not dataset_dir:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASET_DIRS.keys())}")

    # 优先使用 train，如果不存在则用 test
    data_root = Path(base_dir) / dataset_dir
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if train_dir.exists() and list(train_dir.glob("*.json")):
        data_dir = train_dir
        split_name = "train"
    elif test_dir.exists() and list(test_dir.glob("*.json")):
        data_dir = test_dir
        split_name = "test (no train available)"
    else:
        raise FileNotFoundError(f"No JSON files found in train or test: {data_root}")

    json_files = list(data_dir.glob("*.json"))

    print(f"[{dataset}] Found {len(json_files)} JSON files in {data_dir} ({split_name})")

    total_counter = Counter()
    for filepath in json_files:
        counter = count_classes_in_file(filepath)
        total_counter.update(counter)

    # 按 LABEL_LIST 顺序整理
    counts = []
    for label in LABEL_LIST:
        counts.append(total_counter.get(label, 0))

    # 检查是否有未知类别
    unknown_classes = set(total_counter.keys()) - set(LABEL_LIST)
    if unknown_classes:
        print(f"  Warning: Unknown classes found: {unknown_classes}")

    total = sum(counts)
    print(f"  Total lines: {total}")
    print(f"  Class distribution: {dict(zip(LABEL_LIST, counts))}")

    return {
        "counts": counts,
        "total": total,
    }


def write_yaml(results: dict, output_path: Path):
    """写入 YAML 配置文件"""
    lines = [
        "# 类别分布统计（line-level）",
        "# 由 scripts/compute_class_counts.py 自动生成",
        "#",
        "# 类别顺序（LABEL_LIST）:",
        f"# {LABEL_LIST}",
        "",
    ]

    for dataset, data in results.items():
        lines.append(f"{dataset}:")
        lines.append(f"  counts: {data['counts']}")
        lines.append(f"  total: {data['total']}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nWritten to: {output_path}")


def main():
    parser = ArgumentParser(description="统计训练数据的类别分布")
    parser.add_argument("--dataset", type=str, choices=["hrds", "hrdh", "tender"],
                        help="指定数据集")
    parser.add_argument("--all", action="store_true", help="统计所有数据集")
    parser.add_argument("--output", type=str, default="configs/class_counts.yml",
                        help="输出文件路径")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据根目录（可选）")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        print("\nError: Please specify --dataset or --all")
        sys.exit(1)

    # 获取数据目录
    base_dir = args.data_dir or get_data_base_dir()
    print(f"Data base directory: {base_dir}")

    # 确定要统计的数据集
    if args.all:
        datasets = list(DATASET_DIRS.keys())
    else:
        datasets = [args.dataset]

    # 统计
    results = {}
    for dataset in datasets:
        try:
            results[dataset] = count_classes_in_dataset(dataset, base_dir)
        except FileNotFoundError as e:
            print(f"[{dataset}] Skipped: {e}")

    if not results:
        print("No results to write.")
        sys.exit(1)

    # 写入配置文件
    output_path = PROJECT_ROOT / args.output
    write_yaml(results, output_path)

    # 打印摘要
    print("\n" + "=" * 60)
    print("Summary (for copy-paste):")
    print("=" * 60)
    for dataset, data in results.items():
        print(f"\n{dataset}:")
        print(f"  counts = {data['counts']}")


if __name__ == "__main__":
    main()

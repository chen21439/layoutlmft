#!/usr/bin/env python
# coding=utf-8
"""
为 HRDH 数据集添加 line_id 字段

问题背景：
- HRDS 数据有 line_id 字段，HRDH 没有
- parent_id 是全局索引，需要 line_id 也是全局索引才能正确映射
- 如果没有 line_id，代码会用页内索引，导致 parent 关系完全错误

用法:
    python scripts/add_line_id_to_hrdh.py --data_dir /data/LLM_group/layoutlmft/data/HRDH
    python scripts/add_line_id_to_hrdh.py --data_dir /data/LLM_group/layoutlmft/data/HRDH --dry_run
"""

import os
import json
import argparse
from pathlib import Path


def add_line_id_to_file(filepath: str, dry_run: bool = False) -> dict:
    """为单个 JSON 文件添加 line_id"""
    with open(filepath, 'r', encoding='utf8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        return {"status": "skipped", "reason": "not a list"}

    if len(data) == 0:
        return {"status": "skipped", "reason": "empty"}

    # 检查是否已有 line_id
    if "line_id" in data[0]:
        return {"status": "skipped", "reason": "already has line_id"}

    # 添加全局 line_id（按文档顺序，从 0 开始）
    for idx, item in enumerate(data):
        item["line_id"] = idx

    if not dry_run:
        with open(filepath, 'w', encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    return {"status": "updated", "count": len(data)}


def main():
    parser = argparse.ArgumentParser(description="为 HRDH 数据集添加 line_id 字段")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="HRDH 数据目录")
    parser.add_argument("--dry_run", action="store_true",
                        help="只检查，不实际修改文件")
    args = parser.parse_args()

    # 处理 train 和 test 目录
    dirs_to_process = []
    for subdir in ["train", "test", "val"]:
        path = os.path.join(args.data_dir, subdir)
        if os.path.exists(path):
            dirs_to_process.append(path)

    if not dirs_to_process:
        print(f"Error: No train/test/val directory found in {args.data_dir}")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing directories: {dirs_to_process}")
    print("=" * 60)

    total_updated = 0
    total_skipped = 0
    total_files = 0

    for dir_path in dirs_to_process:
        json_files = sorted(Path(dir_path).glob("*.json"))
        print(f"\nProcessing {dir_path}: {len(json_files)} files")

        for filepath in json_files:
            total_files += 1
            result = add_line_id_to_file(str(filepath), dry_run=args.dry_run)

            if result["status"] == "updated":
                total_updated += 1
                if total_updated <= 5:  # 只显示前5个更新的文件
                    print(f"  [UPDATED] {filepath.name}: {result['count']} lines")
            else:
                total_skipped += 1
                if total_skipped <= 3:  # 只显示前3个跳过的文件
                    print(f"  [SKIPPED] {filepath.name}: {result['reason']}")

    print("=" * 60)
    print(f"Total files: {total_files}")
    print(f"Updated: {total_updated}")
    print(f"Skipped: {total_skipped}")

    if args.dry_run:
        print("\n[DRY RUN] No files were actually modified.")
        print("Run without --dry_run to apply changes.")


if __name__ == "__main__":
    main()

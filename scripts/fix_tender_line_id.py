#!/usr/bin/env python
# coding=utf-8
"""
将 tender 数据的 line_id 改为 id 值，使其与 parent_id 格式一致。

用法：
    python scripts/fix_tender_line_id.py <input_dir> [--output_dir <output_dir>]

示例：
    python scripts/fix_tender_line_id.py "E:\models\data\Section\tender_document\runs\20251218_141436_checkpoint-3000_096b7b\enriched"
"""

import json
import os
import argparse
from pathlib import Path


def fix_line_ids(input_dir: str, output_dir: str = None):
    """
    将 JSON 文件中的 line_id 改为 id 值。

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（默认覆盖原文件）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path

    if output_dir:
        output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            lines = json.load(f)

        # 统计修改
        modified = 0
        for line in lines:
            if 'id' in line and 'line_id' in line:
                old_line_id = line['line_id']
                new_line_id = int(line['id'])  # 确保转换为数字
                if old_line_id != new_line_id:
                    line['line_id'] = new_line_id
                    modified += 1

        # 保存
        output_file = output_path / json_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(lines, f, ensure_ascii=False, indent=2)

        print(f"  {json_file.name}: {len(lines)} lines, {modified} modified")

    print(f"\nDone! Files saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fix tender line_id to match id field")
    parser.add_argument("input_dir", help="Input directory containing JSON files")
    parser.add_argument("--output_dir", "-o", help="Output directory (default: overwrite input files)")
    args = parser.parse_args()

    fix_line_ids(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

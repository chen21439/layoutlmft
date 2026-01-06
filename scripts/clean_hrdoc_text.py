#!/usr/bin/env python
# coding=utf-8
"""
清洗 HRDoc 数据集的 text 字段

清洗策略：
1. \x00：删掉
2. 其他控制字符：替换为空格
3. �（U+FFFD）：替换为空格
4. 如果文本被修改过，做空白折叠（多个空格变一个）

用法：
    python scripts/clean_hrdoc_text.py --data_dir /path/to/HRDH/train
    python scripts/clean_hrdoc_text.py --data_dir /mnt/e/models/data/Section/HRDH/train
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from argparse import ArgumentParser


def clean_text(text: str) -> str:
    """清洗文本"""
    if not text:
        return text

    original = text

    # 1. 删除 \x00
    text = text.replace('\x00', '')

    # 2. 替换控制字符为空格
    def replace_control_char(char):
        if char in '\t\n\r':
            return char
        cat = unicodedata.category(char)
        if cat.startswith('C'):
            return ' '
        return char

    text = ''.join(replace_control_char(c) for c in text)

    # 3. 替换 replacement character
    text = text.replace('\ufffd', ' ')

    # 4. 如果被修改过，做空白折叠
    if text != original:
        text = re.sub(r' +', ' ', text)
        text = text.strip()

    return text


def process_file(filepath: Path) -> tuple:
    """处理单个文件，返回 (修改的行数, 总行数)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified_count = 0
    for item in data:
        if 'text' in item:
            original = item['text']
            cleaned = clean_text(original)
            if cleaned != original:
                item['text'] = cleaned
                modified_count += 1

    if modified_count > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return modified_count, len(data)


def main():
    parser = ArgumentParser(description="清洗 HRDoc 数据集的 text 字段")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录（包含 JSON 文件）")
    parser.add_argument("--dry_run", action="store_true", help="只统计，不修改")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"目录不存在: {data_dir}")
        return

    json_files = list(data_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件")

    total_modified = 0
    total_lines = 0
    modified_files = 0

    for filepath in json_files:
        if args.dry_run:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            count = sum(1 for item in data if 'text' in item and clean_text(item['text']) != item['text'])
            if count > 0:
                print(f"  {filepath.name}: {count} 行需要清洗")
                total_modified += count
                modified_files += 1
            total_lines += len(data)
        else:
            modified, total = process_file(filepath)
            if modified > 0:
                modified_files += 1
            total_modified += modified
            total_lines += total

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}统计:")
    print(f"  文件数: {len(json_files)}")
    print(f"  总行数: {total_lines}")
    print(f"  修改文件数: {modified_files}")
    print(f"  修改行数: {total_modified}")


if __name__ == "__main__":
    main()

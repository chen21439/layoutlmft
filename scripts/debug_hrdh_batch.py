#!/usr/bin/env python
# coding=utf-8
"""
分批测试 HRDH 数据集，定位有问题的文件

主要用途：
1. 检测数据加载时会导致阻塞/卡住的文件
2. 检测 parent_id 循环引用问题（opara 的 parent_id 指向自己会导致 trans_class 无限循环）
3. 检测图片加载问题

用法:
    python scripts/debug_hrdh_batch.py --start 0 --end 100
    python scripts/debug_hrdh_batch.py --start 100 --end 200
    ...

找到有问题的批次后，缩小范围继续排查

已知问题文件（已删除）：
- 1607.03341.json: page 1 中有 5 个 opara 的 parent_id 指向自己（自环），
  导致 layoutlmft/data/labels.py 中 trans_class 函数的 while 循环无限执行

HRDH数据集需要自己根据数组索引添加line_id
"""

import os
import sys
import json
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layoutlmft.data.utils import load_image


def test_batch(data_dir: str, start: int, end: int):
    """测试指定范围的文件"""
    ann_dir = os.path.join(data_dir, "train")
    img_dir = os.path.join(data_dir, "images")

    all_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
    batch_files = all_files[start:end]

    print(f"Testing files {start}-{end} (total {len(batch_files)} files)")
    print("=" * 60)

    success = 0
    failed = []

    for i, filename in enumerate(batch_files):
        file_idx = start + i
        filepath = os.path.join(ann_dir, filename)
        doc_name = filename.replace('.json', '')

        try:
            # 1. 加载 JSON
            with open(filepath, 'r', encoding='utf8') as f:
                data = json.load(f)

            # 2. 按页分组
            if isinstance(data, list):
                pages = {}
                for item in data:
                    page_num = item.get("page", 0)
                    if isinstance(page_num, str):
                        page_num = int(page_num) if page_num.isdigit() else 0
                    if page_num not in pages:
                        pages[page_num] = []
                    pages[page_num].append(item)
            else:
                pages = {0: data.get("form", [])}

            # 3. 测试每页的图片加载
            for page_num in sorted(pages.keys()):
                # HRDH 格式: images/{doc}/{page}.png
                img_path = os.path.join(img_dir, doc_name, f"{page_num}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(img_dir, doc_name, f"{page_num}.jpg")

                if os.path.exists(img_path):
                    image, size = load_image(img_path)
                else:
                    print(f"  [{file_idx}] {filename} page {page_num}: image not found", flush=True)

            success += 1
            print(f"[{file_idx}] {filename}: OK ({len(pages)} pages)", flush=True)

        except Exception as e:
            failed.append((file_idx, filename, str(e)))
            print(f"[{file_idx}] {filename}: FAILED - {e}", flush=True)

    print("=" * 60)
    print(f"Results: {success} success, {len(failed)} failed")

    if failed:
        print("\nFailed files:")
        for idx, name, err in failed:
            print(f"  [{idx}] {name}: {err}")

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="分批测试 HRDH 数据集")
    parser.add_argument("--data_dir", default="/data/LLM_group/layoutlmft/data/HRDH",
                        help="HRDH 数据目录")
    parser.add_argument("--start", type=int, default=0, help="起始文件索引")
    parser.add_argument("--end", type=int, default=100, help="结束文件索引")
    args = parser.parse_args()

    success = test_batch(args.data_dir, args.start, args.end)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

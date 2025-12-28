#!/usr/bin/env python
# coding=utf-8
"""
分析 section 类型的 parent 预测错误

用法:
    python analyze_section_errors.py <infer_dir> <data_dir>

示例:
    python analyze_section_errors.py runs/20241228_150000 /data/HRDoc/dev3

输出:
    筛选 gt_class=section 且 gt_parent_id != pred_parent_id 的行
    从原始 json 读取 text 信息并打印
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def load_infer_results(infer_dir: str) -> Dict[str, List[dict]]:
    """加载推理结果"""
    results = {}
    infer_path = Path(infer_dir)

    for json_file in infer_path.glob("*_infer.json"):
        doc_name = json_file.stem.replace("_infer", "")
        with open(json_file, "r", encoding="utf-8") as f:
            results[doc_name] = json.load(f)

    return results


def load_original_data(data_dir: str, doc_name: str) -> Dict[int, dict]:
    """从原始数据目录加载文档，返回 line_id -> line_info 映射"""
    line_map = {}

    # 尝试多种可能的路径
    possible_paths = [
        Path(data_dir) / doc_name,
        Path(data_dir) / f"{doc_name}.json",
    ]

    doc_path = None
    for p in possible_paths:
        if p.exists():
            doc_path = p
            break

    if doc_path is None:
        return line_map

    # 如果是目录，查找 json 文件
    if doc_path.is_dir():
        json_files = list(doc_path.glob("*.json"))
        if not json_files:
            return line_map

        # 合并所有页面的数据
        for json_file in sorted(json_files):
            with open(json_file, "r", encoding="utf-8") as f:
                page_data = json.load(f)

            items = page_data.get("items", page_data.get("lines", []))
            for item in items:
                line_id = item.get("line_id", item.get("id"))
                if line_id is not None:
                    # 拼接 words 的 text
                    words = item.get("words", [])
                    if words:
                        text = " ".join(w.get("text", "") for w in words)
                    else:
                        text = item.get("text", "")

                    line_map[line_id] = {
                        "text": text[:100],  # 截断过长文本
                        "label": item.get("label", item.get("class", "")),
                        "parent_id": item.get("parent_id", -1),
                    }
    else:
        # 单个 json 文件
        with open(doc_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("items", data.get("lines", []))
        for item in items:
            line_id = item.get("line_id", item.get("id"))
            if line_id is not None:
                words = item.get("words", [])
                if words:
                    text = " ".join(w.get("text", "") for w in words)
                else:
                    text = item.get("text", "")

                line_map[line_id] = {
                    "text": text[:100],
                    "label": item.get("label", item.get("class", "")),
                    "parent_id": item.get("parent_id", -1),
                }

    return line_map


def analyze_section_errors(infer_dir: str, data_dir: str):
    """分析 section 类型的 parent 预测错误"""

    print(f"加载推理结果: {infer_dir}")
    infer_results = load_infer_results(infer_dir)
    print(f"找到 {len(infer_results)} 个文档\n")

    total_sections = 0
    total_errors = 0

    for doc_name, lines in infer_results.items():
        # 加载原始数据获取 text
        original_data = load_original_data(data_dir, doc_name)

        # 筛选 section 类型且 parent 预测错误的
        errors = []
        for line in lines:
            if line.get("gt_class") == "section":
                total_sections += 1
                gt_parent = line.get("gt_parent_id")
                pred_parent = line.get("pred_parent_id")

                if gt_parent != pred_parent:
                    total_errors += 1
                    errors.append(line)

        if errors:
            print(f"=" * 60)
            print(f"文档: {doc_name}")
            print(f"=" * 60)

            for err in errors:
                line_id = err["line_id"]
                gt_parent_id = err["gt_parent_id"]
                pred_parent_id = err["pred_parent_id"]

                # 获取 text 信息
                line_info = original_data.get(line_id, {})
                gt_parent_info = original_data.get(gt_parent_id, {})
                pred_parent_info = original_data.get(pred_parent_id, {})

                print(f"\n[错误] line_id={line_id}")
                print(f"  当前行: \"{line_info.get('text', 'N/A')}\"")
                print(f"  正确父元素 (id={gt_parent_id}): \"{gt_parent_info.get('text', 'N/A')}\"")
                print(f"  预测父元素 (id={pred_parent_id}): \"{pred_parent_info.get('text', 'N/A')}\"")

            print()

    # 汇总统计
    print("=" * 60)
    print("汇总统计")
    print("=" * 60)
    print(f"Section 总数: {total_sections}")
    print(f"Parent 预测错误数: {total_errors}")
    if total_sections > 0:
        print(f"错误率: {total_errors / total_sections * 100:.1f}%")
        print(f"准确率: {(total_sections - total_errors) / total_sections * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="分析 section 类型的 parent 预测错误")
    parser.add_argument("infer_dir", help="推理结果目录 (包含 *_infer.json 文件)")
    parser.add_argument("data_dir", help="原始数据目录 (用于读取 text)")

    args = parser.parse_args()

    analyze_section_errors(args.infer_dir, args.data_dir)


if __name__ == "__main__":
    main()

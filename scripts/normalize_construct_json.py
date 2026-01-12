#!/usr/bin/env python
# coding=utf-8
"""
归一化 construct.json 中的 class 字段

将 HRDoc 19 类映射到标准 14 类:
- para, opara, alg -> paraline
- sec1, sec2, sec3, secx -> section
- fig -> figure
- tab -> table
- tabcap, figcap -> caption
- equ -> equation
- foot -> footer
- fnote -> footnote
- background -> table

Usage:
    python scripts/normalize_construct_json.py /path/to/xxx_construct.json
    python scripts/normalize_construct_json.py /path/to/xxx_construct.json -o /path/to/output.json
"""

import argparse
import json
import sys
import os

# 类别映射 (将细分类别映射到标准类别)
CLASS_MAPPING = {
    "title": "title",
    "author": "author",
    "mail": "mail",
    "affili": "affili",
    "sec1": "section",
    "sec2": "section",
    "sec3": "section",
    "secx": "section",
    "section": "section",
    "alg": "paraline",
    "fstline": "fstline",
    "para": "paraline",
    "paraline": "paraline",
    "tab": "table",
    "table": "table",
    "fig": "figure",
    "figure": "figure",
    "tabcap": "caption",
    "figcap": "caption",
    "caption": "caption",
    "equ": "equation",
    "equation": "equation",
    "foot": "footer",
    "footer": "footer",
    "header": "header",
    "fnote": "footnote",
    "footnote": "footnote",
    "background": "table",
    "opara": "paraline",
}


def normalize_class(cls: str) -> str:
    """标准化类别名称"""
    if not cls:
        return "paraline"
    cls_lower = cls.lower()
    return CLASS_MAPPING.get(cls_lower, "paraline")


def normalize_item(item: dict) -> dict:
    """归一化单个 item 的 class 字段"""
    if "class" in item:
        item["class"] = normalize_class(item["class"])
    return item


def normalize_tree(nodes: list) -> list:
    """递归归一化嵌套树结构"""
    for node in nodes:
        normalize_item(node)
        if "children" in node and node["children"]:
            normalize_tree(node["children"])
        if "content" in node and node["content"]:
            for content_item in node["content"]:
                normalize_item(content_item)
    return nodes


def normalize_construct_json(data: dict) -> dict:
    """归一化整个 construct.json"""
    # 归一化 predictions (扁平格式)
    if "predictions" in data:
        for pred in data["predictions"]:
            normalize_item(pred)

    # 归一化 toc_tree (嵌套格式)
    if "toc_tree" in data:
        normalize_tree(data["toc_tree"])

    return data


def main():
    parser = argparse.ArgumentParser(description="归一化 construct.json 中的 class 字段")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径 (默认覆盖原文件)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: 文件不存在: {args.input}")
        sys.exit(1)

    # 读取
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计原始类别
    original_classes = set()
    if "predictions" in data:
        for pred in data["predictions"]:
            if "class" in pred:
                original_classes.add(pred["class"])

    print(f"原始类别: {sorted(original_classes)}")

    # 归一化
    data = normalize_construct_json(data)

    # 统计归一化后类别
    normalized_classes = set()
    if "predictions" in data:
        for pred in data["predictions"]:
            if "class" in pred:
                normalized_classes.add(pred["class"])

    print(f"归一化后: {sorted(normalized_classes)}")

    # 写入
    output_path = args.output or args.input
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    main()

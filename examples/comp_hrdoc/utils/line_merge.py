#!/usr/bin/env python
# coding=utf-8
"""
Line Merge Utilities

合并连续段落行的工具函数。

主要用于将 fstline 和其后续 connect 关系的 paraline 合并为一个段落元素。
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def merge_connected_lines(
    items: List[Dict],
    start_class: str = "fstline",
    continue_class: str = "paraline",
    continue_relation: str = "connect",
    text_separator: str = " ",
) -> List[Dict]:
    """
    合并连续的段落行

    规则: fstline -connect-> paraline -connect-> paraline -> 合并为一个元素
    停止条件: 遇到非 continue_class 或非 continue_relation

    合并方式:
    - text: 拼接（使用 text_separator 分隔）
    - location: 列表合并（保留所有 bbox）

    Args:
        items: 按阅读顺序排列的元素列表，每个元素需包含:
            - class: 元素类别 (fstline, paraline, section, etc.)
            - text: 文本内容
            - relation: 与前一元素的关系 (connect, contain, equality)
            - location: 坐标列表 (可选)
        start_class: 合并起始的类别，默认 "fstline"
        continue_class: 可继续合并的类别，默认 "paraline"
        continue_relation: 触发合并的关系类型，默认 "connect"
        text_separator: 文本拼接的分隔符，默认空格

    Returns:
        合并后的元素列表，合并的元素会添加 _merged_count 字段

    Example:
        输入:
        [
            {"line_id": 1, "class": "fstline", "text": "段落开头"},
            {"line_id": 2, "class": "paraline", "relation": "connect", "text": "接续内容"},
            {"line_id": 3, "class": "paraline", "relation": "connect", "text": "更多内容"},
            {"line_id": 4, "class": "fstline", "text": "新段落"},
        ]

        输出:
        [
            {"line_id": 1, "class": "fstline", "text": "段落开头 接续内容 更多内容", "_merged_count": 3},
            {"line_id": 4, "class": "fstline", "text": "新段落"},
        ]
    """
    if not items:
        return items

    merged = []
    i = 0
    start_class_lower = start_class.lower()
    continue_class_lower = continue_class.lower()
    continue_relation_lower = continue_relation.lower()

    while i < len(items):
        item = items[i]
        item_class = (item.get("class", "") or "").lower()

        # 检查是否是起始类别 (fstline)
        if item_class == start_class_lower:
            # 开始收集后续 connect 的 paraline
            merged_item = dict(item)
            texts = [item.get("text", "") or ""]
            locations = list(item.get("location", []) or [])
            merged_line_ids = [item.get("line_id")]

            j = i + 1
            while j < len(items):
                next_item = items[j]
                next_class = (next_item.get("class", "") or "").lower()
                next_relation = (next_item.get("relation", "") or "").lower()

                # 检查是否是 paraline 且 relation=connect
                if next_class == continue_class_lower and next_relation == continue_relation_lower:
                    texts.append(next_item.get("text", "") or "")
                    next_loc = next_item.get("location", []) or []
                    locations.extend(next_loc)
                    merged_line_ids.append(next_item.get("line_id"))
                    j += 1
                else:
                    # 遇到非 paraline 或非 connect，停止合并
                    break

            # 如果有合并发生
            if j > i + 1:
                merged_item["text"] = text_separator.join(t for t in texts if t)
                merged_item["location"] = locations
                merged_item["_merged_count"] = j - i
                merged_item["_merged_line_ids"] = merged_line_ids
                logger.debug(
                    f"[MergeLines] Merged {j - i} lines: line_ids={merged_line_ids}"
                )

            merged.append(merged_item)
            i = j
        else:
            merged.append(item)
            i += 1

    original_count = len(items)
    merged_count = len(merged)
    if original_count != merged_count:
        logger.info(f"[MergeLines] Merged {original_count} -> {merged_count} items")

    return merged


def merge_connected_lines_bidirectional(
    items: List[Dict],
    text_separator: str = " ",
) -> List[Dict]:
    """
    双向合并连续段落行

    除了 fstline -> paraline(connect) 的正向合并外，
    还处理 paraline -> paraline(connect) 的情况（当 paraline 是段落的开头时）

    Args:
        items: 按阅读顺序排列的元素列表
        text_separator: 文本拼接的分隔符

    Returns:
        合并后的元素列表
    """
    if not items:
        return items

    merged = []
    i = 0

    while i < len(items):
        item = items[i]
        item_class = (item.get("class", "") or "").lower()

        # fstline 或 paraline 都可以作为段落开头
        if item_class in ("fstline", "paraline"):
            # 检查当前元素是否是段落的开头
            # 如果是 paraline 且 relation=connect，说明它不是段落开头，跳过
            item_relation = (item.get("relation", "") or "").lower()
            if item_class == "paraline" and item_relation == "connect":
                # 这个 paraline 应该被前面的元素合并，但如果到这里说明前面不是 fstline
                # 直接添加不合并
                merged.append(item)
                i += 1
                continue

            # 开始收集后续 connect 的 paraline
            merged_item = dict(item)
            texts = [item.get("text", "") or ""]
            locations = list(item.get("location", []) or [])
            merged_line_ids = [item.get("line_id")]

            j = i + 1
            while j < len(items):
                next_item = items[j]
                next_class = (next_item.get("class", "") or "").lower()
                next_relation = (next_item.get("relation", "") or "").lower()

                # 检查是否是 paraline 且 relation=connect
                if next_class == "paraline" and next_relation == "connect":
                    texts.append(next_item.get("text", "") or "")
                    next_loc = next_item.get("location", []) or []
                    locations.extend(next_loc)
                    merged_line_ids.append(next_item.get("line_id"))
                    j += 1
                else:
                    break

            # 如果有合并发生
            if j > i + 1:
                merged_item["text"] = text_separator.join(t for t in texts if t)
                merged_item["location"] = locations
                merged_item["_merged_count"] = j - i
                merged_item["_merged_line_ids"] = merged_line_ids

            merged.append(merged_item)
            i = j
        else:
            merged.append(item)
            i += 1

    return merged


__all__ = [
    'merge_connected_lines',
    'merge_connected_lines_bidirectional',
]

#!/usr/bin/env python
# coding=utf-8
"""
文本清洗工具

清洗策略（训练友好）：
1. \x00：删掉
2. 其他控制字符：替换为空格
3. �（U+FFFD，replacement character）：替换为空格
4. 如果文本被修改过，做空白折叠（多个空格变一个）
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    清洗文本，处理乱码和控制字符

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not text:
        return text

    original = text

    # 1. 删除 \x00（null character）
    text = text.replace('\x00', '')

    # 2. 替换其他控制字符为空格（保留换行、制表符等常见空白）
    # 控制字符范围：U+0000-U+001F（除了 \t \n \r）和 U+007F-U+009F
    def replace_control_char(char):
        if char in '\t\n\r':
            return char
        cat = unicodedata.category(char)
        if cat.startswith('C'):  # C = Control characters
            return ' '
        return char

    text = ''.join(replace_control_char(c) for c in text)

    # 3. 替换 replacement character（�，U+FFFD）为空格
    text = text.replace('\ufffd', ' ')

    # 4. 如果文本被修改过，做空白折叠（多个空格变一个）
    if text != original:
        text = re.sub(r' +', ' ', text)
        text = text.strip()

    return text


def is_valid_text(text: str) -> bool:
    """
    检查文本是否有效（清洗后非空且有字母数字）

    Args:
        text: 原始文本

    Returns:
        True 如果文本有效
    """
    cleaned = clean_text(text)
    if not cleaned:
        return False
    # 检查是否有字母或数字
    return any(c.isalnum() for c in cleaned)

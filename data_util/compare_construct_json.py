"""
对比两个 construct.json 文件
检查 line 数量和前缀变化

用法：
  python compare_construct_json.py <file1.json> <file2.json>
"""

import json
import sys
from pathlib import Path
from difflib import SequenceMatcher


def convert_windows_path_to_wsl(path_str: str) -> str:
    """将 Windows 路径转换为 WSL 路径"""
    if not path_str:
        return path_str
    if len(path_str) >= 2 and path_str[1] == ':':
        drive_letter = path_str[0].lower()
        rest_path = path_str[2:].replace('\\', '/').lstrip('/')
        return f"/mnt/{drive_letter}/{rest_path}"
    return path_str


def load_json(file_path: Path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_lines(data):
    """提取所有 line 对象"""
    # 处理两种可能的结构：
    # 1. {"success":true, "data":{"dataList":[...]}} - 有API包装
    # 2. 直接的数组 [...] - 无包装

    if isinstance(data, dict):
        if 'data' in data and 'dataList' in data['data']:
            # 第一种结构：从 data.dataList 取数组
            return data['data']['dataList']
        else:
            # 其他字典结构，返回空
            return []
    elif isinstance(data, list):
        # 第二种结构：直接是数组
        return data
    else:
        return []


def find_longest_common_suffix(s1: str, s2: str) -> str:
    """找到两个字符串的最长公共后缀"""
    matcher = SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))

    # 从匹配位置找最长的后缀
    common = ""
    for i in range(min(len(s1), len(s2))):
        if s1[-(i+1)] == s2[-(i+1)]:
            common = s1[-(i+1)] + common
        else:
            break
    return common


def compare_lines(lines1, lines2):
    """对比两组 line 对象"""

    print("=" * 80)
    print("Line 数量统计")
    print("=" * 80)
    print(f"文件1 (style): {len(lines1)} 个 line")
    print(f"文件2 (fulltext): {len(lines2)} 个 line")
    print(f"数量差异: {abs(len(lines1) - len(lines2))}")
    print()

    # 按 line_id 对齐（而不是id字段）
    lines1_by_id = {line.get('line_id'): line for line in lines1 if 'line_id' in line}
    lines2_by_id = {line.get('line_id'): line for line in lines2 if 'line_id' in line}

    common_ids = set(lines1_by_id.keys()) & set(lines2_by_id.keys())
    only_in_1 = set(lines1_by_id.keys()) - set(lines2_by_id.keys())
    only_in_2 = set(lines2_by_id.keys()) - set(lines1_by_id.keys())

    print("=" * 80)
    print("line_id 对齐统计")
    print("=" * 80)
    print(f"共同的 line_id: {len(common_ids)}")
    print(f"只在文件1中: {len(only_in_1)}")
    if only_in_1:
        print(f"  示例: {sorted(only_in_1)[:10]}")
    print(f"只在文件2中: {len(only_in_2)}")
    if only_in_2:
        print(f"  示例: {sorted(only_in_2)[:10]}")
    print()

    # 对比各个字段
    print("=" * 80)
    print("字段变化统计")
    print("=" * 80)

    text_identical = 0
    text_prefix_changed = 0
    text_completely_different = 0

    parent_id_changed = 0
    relation_changed = 0
    class_changed = 0

    prefix_changes = []
    other_field_changes = []

    for line_id in sorted(common_ids):
        line1 = lines1_by_id[line_id]
        line2 = lines2_by_id[line_id]

        # 对比 text
        text1 = line1.get('text', '')
        text2 = line2.get('text', '')

        if text1 == text2:
            text_identical += 1
        else:
            # 检查是否只是前缀变化
            common_suffix = find_longest_common_suffix(text1, text2)

            if len(common_suffix) > 0 and len(common_suffix) >= min(len(text1), len(text2)) * 0.5:
                # 后半部分相同，前缀变化
                prefix1 = text1[:-len(common_suffix)] if common_suffix else text1
                prefix2 = text2[:-len(common_suffix)] if common_suffix else text2

                text_prefix_changed += 1
                prefix_changes.append({
                    'line_id': line_id,
                    'prefix1': prefix1,
                    'prefix2': prefix2,
                    'common_suffix': common_suffix[:50] + ('...' if len(common_suffix) > 50 else ''),
                    'text1': text1,
                    'text2': text2,
                    'class1': line1.get('class'),
                    'class2': line2.get('class')
                })
            else:
                text_completely_different += 1

        # 对比其他字段
        field_change = {}
        if line1.get('parent_id') != line2.get('parent_id'):
            parent_id_changed += 1
            field_change['parent_id'] = (line1.get('parent_id'), line2.get('parent_id'))

        if line1.get('relation') != line2.get('relation'):
            relation_changed += 1
            field_change['relation'] = (line1.get('relation'), line2.get('relation'))

        if line1.get('class') != line2.get('class'):
            class_changed += 1
            field_change['class'] = (line1.get('class'), line2.get('class'))

        if field_change:
            other_field_changes.append({
                'line_id': line_id,
                'text1': text1[:60] + ('...' if len(text1) > 60 else ''),
                'text2': text2[:60] + ('...' if len(text2) > 60 else ''),
                'changes': field_change
            })

    print(f"\ntext 字段:")
    print(f"  完全相同: {text_identical}")
    print(f"  前缀变化（后半部分相同）: {text_prefix_changed}")
    print(f"  完全不同: {text_completely_different}")

    print(f"\n其他字段变化:")
    print(f"  parent_id 变化: {parent_id_changed}")
    print(f"  relation 变化: {relation_changed}")
    print(f"  class 变化: {class_changed}")
    print()

    # 显示前缀变化的详细信息
    if prefix_changes:
        print("=" * 80)
        print("Text 前缀变化详情（前30个）")
        print("=" * 80)

        for i, change in enumerate(prefix_changes[:30], 1):
            print(f"\n[{i}] line_id: {change['line_id']} | class: {change['class1']} -> {change['class2']}")
            print(f"  文件1: '{change['text1']}'")
            print(f"  文件2: '{change['text2']}'")
            print(f"  前缀1: '{change['prefix1']}'")
            print(f"  前缀2: '{change['prefix2']}'")
            print(f"  公共后缀: '{change['common_suffix']}'")

        if len(prefix_changes) > 30:
            print(f"\n... 还有 {len(prefix_changes) - 30} 个前缀变化的条目")

    # 显示其他字段变化
    if other_field_changes:
        print("\n" + "=" * 80)
        print("其他字段变化详情（前30个）")
        print("=" * 80)

        for i, change in enumerate(other_field_changes[:30], 1):
            print(f"\n[{i}] line_id: {change['line_id']}")
            print(f"  text1: {change['text1']}")
            print(f"  text2: {change['text2']}")
            print(f"  变化: {change['changes']}")

        if len(other_field_changes) > 30:
            print(f"\n... 还有 {len(other_field_changes) - 30} 个字段变化的条目")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\n[错误] 请提供两个JSON文件路径")
        sys.exit(1)

    path1 = convert_windows_path_to_wsl(sys.argv[1])
    path2 = convert_windows_path_to_wsl(sys.argv[2])

    file1 = Path(path1)
    file2 = Path(path2)

    if not file1.exists():
        print(f"[错误] 文件不存在: {file1}")
        sys.exit(1)
    if not file2.exists():
        print(f"[错误] 文件不存在: {file2}")
        sys.exit(1)

    print("=" * 80)
    print("对比 Construct JSON 文件")
    print("=" * 80)
    print(f"文件1 (style): {file1.name}")
    print(f"文件2 (fulltext): {file2.name}")
    print()

    print("加载文件...")
    data1 = load_json(file1)
    data2 = load_json(file2)

    print("提取 line 对象...")
    lines1 = extract_lines(data1)
    lines2 = extract_lines(data2)

    compare_lines(lines1, lines2)


if __name__ == "__main__":
    main()

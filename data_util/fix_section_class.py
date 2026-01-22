"""
修复construct.json文件的section class和parent_id

用法：
  python fix_section_class.py <reference.json> <target.json> <output.json>

参数：
  reference.json - 参考文件（253），以其section为基准
  target.json - 目标文件（257），需要修复的文件
  output.json - 输出文件
"""

import json
import sys
from pathlib import Path
import shutil
import re


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


def save_json(data, file_path: Path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_lines(data):
    """提取所有 line 对象"""
    if isinstance(data, dict):
        if 'data' in data and 'dataList' in data['data']:
            return data['data']['dataList']
        else:
            return []
    elif isinstance(data, list):
        return data
    else:
        return []


def remove_prefix(text):
    """去掉编号前缀"""
    patterns = [
        r'^第[一二三四五六七八九十百千]+[章节册]\s*',
        r'^[一二三四五六七八九十百千]+、\s*',
        r'^（[一二三四五六七八九十百千]+）\s*',
        r'^\([一二三四五六七八九十百千]+\)\s*',
        r'^\d+、\s*',
        r'^\d+\.\s*',
        r'^\(\d+\)\s*',
        r'^\d+\.\d+\s*',
        r'^\d+\.\d+\.\d+\s*',
        r'^\d+\.\d+\.\d+\.\d+\s*',
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result)
    return result.strip()


def build_mapping(ref_lines, target_lines):
    """
    建立参考文件section到目标文件的映射

    Returns:
        dict: {ref_line_id: target_line_id}
    """
    # 提取参考文件的section
    ref_sections = []
    for line in ref_lines:
        if line.get('class') == 'section':
            text = line.get('text', '')
            suffix = remove_prefix(text)
            ref_sections.append({
                'line_id': line.get('line_id'),
                'text': text,
                'suffix': suffix,
                'parent_id': line.get('parent_id')
            })

    # 建立目标文件的后缀索引（所有非table行）
    target_index = {}
    for line in target_lines:
        if line.get('class') == 'table':
            continue
        text = line.get('text', '')
        suffix = remove_prefix(text)
        line_id = line.get('line_id')

        # 精确匹配
        if suffix not in target_index:
            target_index[suffix] = []
        target_index[suffix].append(line_id)

    # 建立映射
    mapping = {}

    for ref_sec in ref_sections:
        ref_id = ref_sec['line_id']
        suffix = ref_sec['suffix']

        # 1. 尝试精确匹配
        if suffix in target_index and len(target_index[suffix]) > 0:
            # 如果有多个匹配，选第一个
            target_id = target_index[suffix][0]
            mapping[ref_id] = target_id
        else:
            # 2. 尝试子串匹配
            found = False
            for line in target_lines:
                if line.get('class') == 'table':
                    continue
                target_text = line.get('text', '')
                target_suffix = remove_prefix(target_text)

                # 子串包含
                if suffix in target_suffix or target_suffix in suffix:
                    mapping[ref_id] = line.get('line_id')
                    found = True
                    break

            if not found:
                print(f"  警告: 参考文件 line_id={ref_id} 在目标文件中未找到匹配")
                print(f"    text: {ref_sec['text'][:60]}")

    return mapping, ref_sections


def fix_target_file(target_lines, mapping, ref_sections):
    """
    修复目标文件

    1. 将映射到的行的class改为section
    2. 修复这些section行的parent_id
    """
    # 建立目标文件的line_id索引
    target_by_id = {line.get('line_id'): line for line in target_lines}

    print(f"\n开始修复...")
    print(f"  需要修复的section数量: {len(mapping)}")

    # 统计
    class_changed = 0
    parent_changed = 0

    # 第一步：修改class
    for ref_id, target_id in mapping.items():
        if target_id in target_by_id:
            target_line = target_by_id[target_id]
            old_class = target_line.get('class')
            if old_class != 'section':
                target_line['class'] = 'section'
                class_changed += 1

    print(f"  修改class为section: {class_changed} 个")

    # 第二步：修复parent_id
    ref_sections_by_id = {sec['line_id']: sec for sec in ref_sections}

    for ref_id, target_id in mapping.items():
        if target_id not in target_by_id:
            continue

        target_line = target_by_id[target_id]
        ref_sec = ref_sections_by_id.get(ref_id)

        if not ref_sec:
            continue

        ref_parent_id = ref_sec['parent_id']

        # 如果parent_id是-1，保持不变
        if ref_parent_id == -1:
            if target_line.get('parent_id') != -1:
                target_line['parent_id'] = -1
                parent_changed += 1
        else:
            # 找到parent在目标文件中的对应line_id
            if ref_parent_id in mapping:
                target_parent_id = mapping[ref_parent_id]
                old_parent = target_line.get('parent_id')
                if old_parent != target_parent_id:
                    target_line['parent_id'] = target_parent_id
                    parent_changed += 1
            else:
                print(f"  警告: line_id={target_id} 的parent在映射中未找到 (ref_parent_id={ref_parent_id})")

    print(f"  修改parent_id: {parent_changed} 个")

    return target_lines


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("\n[错误] 请提供三个文件路径")
        sys.exit(1)

    ref_path = convert_windows_path_to_wsl(sys.argv[1])
    target_path = convert_windows_path_to_wsl(sys.argv[2])
    output_path = convert_windows_path_to_wsl(sys.argv[3])

    ref_file = Path(ref_path)
    target_file = Path(target_path)
    output_file = Path(output_path)

    if not ref_file.exists():
        print(f"[错误] 参考文件不存在: {ref_file}")
        sys.exit(1)
    if not target_file.exists():
        print(f"[错误] 目标文件不存在: {target_file}")
        sys.exit(1)

    print("=" * 80)
    print("修复 section class 和 parent_id")
    print("=" * 80)
    print(f"参考文件: {ref_file.name}")
    print(f"目标文件: {target_file.name}")
    print(f"输出文件: {output_file.name}")
    print()

    # 备份目标文件
    backup_file = target_file.parent / f"{target_file.stem}_backup{target_file.suffix}"
    if not backup_file.exists():
        print(f"备份目标文件 -> {backup_file.name}")
        shutil.copy2(target_file, backup_file)
        print()

    # 加载文件
    print("加载文件...")
    ref_data = load_json(ref_file)
    target_data = load_json(target_file)

    ref_lines = extract_lines(ref_data)
    target_lines = extract_lines(target_data)

    print(f"  参考文件: {len(ref_lines)} 行")
    print(f"  目标文件: {len(target_lines)} 行")
    print()

    # 建立映射
    print("建立映射关系...")
    mapping, ref_sections = build_mapping(ref_lines, target_lines)
    print(f"  参考文件section数量: {len(ref_sections)}")
    print(f"  成功映射: {len(mapping)} 个")
    print()

    # 修复目标文件
    fixed_lines = fix_target_file(target_lines, mapping, ref_sections)

    # 保存（保持原格式）
    if isinstance(target_data, dict):
        # 格式: {"success":true, "data":{"dataList":[...]}}
        target_data['data']['dataList'] = fixed_lines
        output_data = target_data
    else:
        # 格式: [...]
        output_data = fixed_lines

    print(f"\n保存到: {output_file}")
    save_json(output_data, output_file)

    print("\n✓ 修复完成")


if __name__ == "__main__":
    main()

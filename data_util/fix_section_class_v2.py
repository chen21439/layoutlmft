"""
使用树结构修复construct.json文件的section class、parent_id和relation

用法：
  python fix_section_class_v2.py <reference.json> <target.json> <output.json>

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
from typing import List, Dict, Optional


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
    """
    去掉编号前缀

    处理各种编号格式：
    - 中文：第一章、一、（一）、(一)
    - 阿拉伯数字：1、1.、1.2.3、1.2.3.4.5
    - 括号：(1)、（1）、1)、1）
    """
    patterns = [
        # 中文章节
        r'^第[一二三四五六七八九十百千]+[章节册条]\s*',
        r'^[一二三四五六七八九十百千]+、\s*',
        r'^（[一二三四五六七八九十百千]+）\s*',
        r'^\([一二三四五六七八九十百千]+\)\s*',
        # 多级数字编号（最多5级）
        r'^\d+\.\d+\.\d+\.\d+\.\d+\s+',
        r'^\d+\.\d+\.\d+\.\d+\s+',
        r'^\d+\.\d+\.\d+\s+',
        r'^\d+\.\d+\s+',
        # 简单数字编号
        r'^\d+、\s*',
        r'^\d+\.\s+',
        r'^\d+．\s+',  # 全角点
        # 括号编号
        r'^\(\d+\)\s*',
        r'^\（\d+\）\s*',  # 全角括号
        r'^\d+\)\s*',
        r'^\d+）\s*',  # 全角括号
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result)
    return result.strip()


class TreeNode:
    """树节点"""
    def __init__(self, line_id: int, text: str, suffix: str, relation: str, parent_id: int):
        self.line_id = line_id
        self.text = text
        self.suffix = suffix
        self.relation = relation
        self.parent_id = parent_id
        self.children: List[TreeNode] = []
        self.matched_target_id: Optional[int] = None

    def add_child(self, child: 'TreeNode'):
        """添加子节点"""
        self.children.append(child)

    def __repr__(self):
        return f"TreeNode(id={self.line_id}, text='{self.text[:20]}...', children={len(self.children)})"


def build_tree(lines: List[dict]) -> Optional[TreeNode]:
    """
    从section列表构建树结构

    Returns:
        根节点（parent_id=-1的节点）
    """
    # 提取所有section
    sections = [line for line in lines if line.get('class') == 'section']

    if not sections:
        return None

    # 创建节点字典
    nodes = {}
    for sec in sections:
        line_id = sec.get('line_id')
        text = sec.get('text', '')
        suffix = remove_prefix(text)
        relation = sec.get('relation', 'equality')
        parent_id = sec.get('parent_id', -1)

        node = TreeNode(line_id, text, suffix, relation, parent_id)
        nodes[line_id] = node

    # 构建父子关系
    root = None
    for node in nodes.values():
        if node.parent_id == -1:
            root = node
        else:
            parent = nodes.get(node.parent_id)
            if parent:
                parent.add_child(node)

    return root


def match_trees(ref_root: TreeNode, target_lines: List[dict]) -> Dict[int, int]:
    """
    匹配两棵树，考虑树的上下文关系

    Args:
        ref_root: 参考树的根节点
        target_lines: 目标文件的所有行（包括非section）

    Returns:
        映射字典 {ref_line_id: target_line_id}
    """
    # 建立目标文件的后缀索引（排除table）
    target_index_by_suffix = {}
    target_index_by_text = {}

    for line in target_lines:
        if line.get('class') == 'table':
            continue

        line_id = line.get('line_id')
        text = line.get('text', '')
        suffix = remove_prefix(text)

        # 完全匹配索引
        if text not in target_index_by_text:
            target_index_by_text[text] = []
        target_index_by_text[text].append(line_id)

        # 后缀匹配索引
        if suffix not in target_index_by_suffix:
            target_index_by_suffix[suffix] = []
        target_index_by_suffix[suffix].append(line_id)

    # 用于记录已匹配的target_id（避免重复匹配）
    used_target_ids = set()
    mapping = {}

    def match_node_recursive(ref_node: TreeNode, parent_target_id: Optional[int] = None):
        """递归匹配节点，考虑父节点上下文"""

        # 候选匹配列表
        candidates = []

        # 1. 优先尝试完全匹配
        if ref_node.text in target_index_by_text:
            for target_id in target_index_by_text[ref_node.text]:
                if target_id not in used_target_ids:
                    candidates.append((target_id, 'exact'))

        # 2. 尝试后缀匹配
        if ref_node.suffix in target_index_by_suffix:
            for target_id in target_index_by_suffix[ref_node.suffix]:
                if target_id not in used_target_ids:
                    # 避免重复
                    if not any(c[0] == target_id for c in candidates):
                        candidates.append((target_id, 'suffix'))

        # 3. 如果还没找到，尝试子串包含匹配
        if not candidates and len(ref_node.suffix) > 2:  # 降低阈值，短文本也可以匹配
            for line in target_lines:
                if line.get('class') == 'table':
                    continue
                target_id = line.get('line_id')
                if target_id in used_target_ids:
                    continue

                target_text = line.get('text', '')
                target_suffix = remove_prefix(target_text)

                # 子串包含（双向），但要求长度差异不能太大
                if ref_node.suffix and target_suffix:
                    len_ref = len(ref_node.suffix)
                    len_tgt = len(target_suffix)

                    # 如果是包含关系
                    if ref_node.suffix in target_suffix or target_suffix in ref_node.suffix:
                        # 长度差异不超过30个字符，避免误匹配
                        if abs(len_ref - len_tgt) < 30:
                            candidates.append((target_id, 'substring'))

        # 4. 如果有候选，选择最佳匹配
        if candidates:
            # 如果只有一个候选，直接使用
            if len(candidates) == 1:
                best_target_id = candidates[0][0]
            else:
                # 多个候选时，优先选择exact，然后suffix，最后substring
                exact_matches = [c for c in candidates if c[1] == 'exact']
                suffix_matches = [c for c in candidates if c[1] == 'suffix']
                substring_matches = [c for c in candidates if c[1] == 'substring']

                if exact_matches:
                    best_target_id = exact_matches[0][0]
                elif suffix_matches:
                    best_target_id = suffix_matches[0][0]
                else:
                    best_target_id = substring_matches[0][0]

            mapping[ref_node.line_id] = best_target_id
            used_target_ids.add(best_target_id)
            ref_node.matched_target_id = best_target_id
        else:
            print(f"  警告: 未找到匹配 ref_id={ref_node.line_id}, text='{ref_node.text[:40]}'")

        # 递归匹配子节点
        for child in ref_node.children:
            match_node_recursive(child, ref_node.matched_target_id)

    # 从根节点开始递归匹配
    match_node_recursive(ref_root)

    return mapping


def collect_all_nodes(root: TreeNode) -> Dict[int, TreeNode]:
    """收集树中所有节点"""
    nodes = {}

    def collect_recursive(node: TreeNode):
        nodes[node.line_id] = node
        for child in node.children:
            collect_recursive(child)

    collect_recursive(root)
    return nodes


def fix_target_file(target_lines: List[dict], mapping: Dict[int, int], ref_nodes: Dict[int, TreeNode]):
    """
    修复目标文件，只修改 class、parent_id、relation 三个字段
    """
    target_by_id = {line.get('line_id'): line for line in target_lines}

    print(f"\n开始修复...")
    print(f"  需要修复的section数量: {len(mapping)}")

    class_changed = 0
    parent_changed = 0
    relation_changed = 0

    # 修改三个字段
    for ref_id, target_id in mapping.items():
        if target_id not in target_by_id:
            continue

        target_line = target_by_id[target_id]
        ref_node = ref_nodes.get(ref_id)

        if not ref_node:
            continue

        # 1. 修改 class
        if target_line.get('class') != 'section':
            target_line['class'] = 'section'
            class_changed += 1

        # 2. 修改 relation
        if target_line.get('relation') != ref_node.relation:
            target_line['relation'] = ref_node.relation
            relation_changed += 1

        # 3. 修改 parent_id
        ref_parent_id = ref_node.parent_id

        if ref_parent_id == -1:
            if target_line.get('parent_id') != -1:
                target_line['parent_id'] = -1
                parent_changed += 1
        else:
            # 找到parent在目标文件中的对应line_id
            if ref_parent_id in mapping:
                target_parent_id = mapping[ref_parent_id]
                if target_line.get('parent_id') != target_parent_id:
                    target_line['parent_id'] = target_parent_id
                    parent_changed += 1
            else:
                print(f"  警告: line_id={target_id} 的parent在映射中未找到 (ref_parent_id={ref_parent_id})")

    print(f"  修改class: {class_changed} 个")
    print(f"  修改relation: {relation_changed} 个")
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
    print("修复 section class、parent_id 和 relation（基于树结构匹配）")
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

    # 构建参考树
    print("构建参考文件的section树...")
    ref_root = build_tree(ref_lines)
    if not ref_root:
        print("[错误] 参考文件没有section节点")
        sys.exit(1)

    ref_nodes = collect_all_nodes(ref_root)
    print(f"  参考树节点数: {len(ref_nodes)}")
    print()

    # 匹配树
    print("匹配参考树到目标文件...")
    mapping = match_trees(ref_root, target_lines)
    print(f"  成功映射: {len(mapping)} 个")
    print()

    # 修复目标文件
    fixed_lines = fix_target_file(target_lines, mapping, ref_nodes)

    # 保存（保持原格式）
    if isinstance(target_data, dict):
        target_data['data']['dataList'] = fixed_lines
        output_data = target_data
    else:
        output_data = fixed_lines

    print(f"\n保存到: {output_file}")
    save_json(output_data, output_file)

    print("\n✓ 修复完成")


if __name__ == "__main__":
    main()

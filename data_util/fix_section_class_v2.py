"""
使用树结构修复construct.json文件的section class、parent_id和relation

核心逻辑：
- 253是标注正确的（class、parent_id、relation都正确）
- 其他文件需要基于253进行修复
- **只有class为section的节点才构建树**

匹配策略（深度优先）：
1. 从253中提取class=section的节点，构建完整的section树
2. 深度优先遍历253树，逐个在259中寻找对应节点
3. 利用树的结构约束搜索范围：
   - 父节点约束：子节点必须在父节点line_id之后
   - 兄弟约束：后一个兄弟必须在前一个兄弟line_id之后
4. 边遍历边构建259树，用于后续节点的搜索

用法：
  python fix_section_class_v2.py <reference.json> <target.json> <output.json>

参数：
  reference.json - 参考文件（253），以其section为基准
  target.json - 目标文件（257/259），需要修复的文件
  output.json - 输出文件
"""

import json
import sys
from pathlib import Path
import shutil
import re
from typing import List, Dict, Optional, Tuple


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
    """树节点（只用于class=section的节点）"""
    def __init__(self, line_id: int, text: str, suffix: str, relation: str, parent_id: int):
        self.line_id = line_id
        self.text = text
        self.suffix = suffix
        self.relation = relation
        self.parent_id = parent_id  # 原始的parent_id（253的）
        self.parent: Optional[TreeNode] = None  # 树中的父节点引用
        self.children: List[TreeNode] = []
        self.matched_target_id: Optional[int] = None

    def add_child(self, child: 'TreeNode'):
        """添加子节点"""
        self.children.append(child)
        child.parent = self  # 设置父节点引用

    def __repr__(self):
        return f"TreeNode(id={self.line_id}, text='{self.text[:20]}...', children={len(self.children)})"


def build_tree(lines: List[dict]) -> Optional[TreeNode]:
    """
    从section列表构建树结构（只构建class=section的节点）

    Returns:
        根节点（parent_id=-1的节点）
    """
    # 只提取class=section的节点
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


def build_target_index(target_lines: List[dict]) -> Tuple[Dict, Dict, Dict]:
    """
    建立目标文件的索引

    Returns:
        (text_index, suffix_index, line_by_id)
    """
    text_index = {}
    suffix_index = {}
    line_by_id = {}

    for line in target_lines:
        if line.get('class') == 'table':
            continue

        line_id = line.get('line_id')
        text = line.get('text', '')
        suffix = remove_prefix(text)

        # 完全匹配索引
        if text not in text_index:
            text_index[text] = []
        text_index[text].append(line_id)

        # 后缀匹配索引
        if suffix not in suffix_index:
            suffix_index[suffix] = []
        suffix_index[suffix].append(line_id)

        # line_id索引
        line_by_id[line_id] = line

    return text_index, suffix_index, line_by_id


def match_trees(ref_root: TreeNode, target_lines: List[dict], ref_nodes: Dict[int, TreeNode]) -> Tuple[Dict[int, int], List[Dict], Optional[TreeNode]]:
    """
    深度优先匹配两棵树，边遍历253树边构建259树

    核心策略：
    1. 从253根节点开始深度优先遍历
    2. 对每个253节点，在259中找对应节点时：
       - 如果有父节点，在父节点的子树范围内找
       - 如果有前一个兄弟节点，从兄弟节点之后找
    3. 找到后记录映射，避免重复使用

    Args:
        ref_root: 参考树（253）的根节点
        target_lines: 目标文件（259）的所有行

    Returns:
        mapping: 映射字典 {ref_line_id: target_line_id}
        unmatched_nodes: 未匹配的节点列表
        target_root: 259的根节点（已构建好的树）
    """
    # 建立目标文件索引
    text_index, suffix_index, line_by_id = build_target_index(target_lines)

    # 已使用的target_id集合
    used_target_ids = set()

    # 映射结果
    mapping = {}

    # 未匹配的节点
    unmatched_nodes = []

    # 259的根节点（边遍历边构建）
    target_root = None

    def find_candidates_in_range(ref_node: TreeNode,
                                  search_start: int = 0,
                                  search_end: int = 999999) -> List[Tuple[int, str]]:
        """
        在指定范围内查找候选节点

        Args:
            ref_node: 参考节点（253）
            search_start: 搜索起始line_id（含）
            search_end: 搜索结束line_id（含）

        Returns:
            候选列表 [(target_line_id, match_type), ...]
            match_type: 'exact', 'suffix', 'substring'
        """
        candidates = []

        # 1. 完全匹配
        if ref_node.text in text_index:
            for target_id in text_index[ref_node.text]:
                if (target_id not in used_target_ids and
                    search_start <= target_id <= search_end):
                    candidates.append((target_id, 'exact'))

        # 2. 后缀匹配
        if ref_node.suffix in suffix_index:
            for target_id in suffix_index[ref_node.suffix]:
                if (target_id not in used_target_ids and
                    search_start <= target_id <= search_end):
                    # 避免重复
                    if not any(c[0] == target_id for c in candidates):
                        candidates.append((target_id, 'suffix'))

        # 3. 子串匹配（如果前两种都没找到）
        if not candidates and len(ref_node.suffix) > 2:
            for line_id in range(search_start, min(search_end + 1, len(target_lines))):
                if line_id not in line_by_id or line_id in used_target_ids:
                    continue

                target_line = line_by_id[line_id]
                if target_line.get('class') == 'table':
                    continue

                target_text = target_line.get('text', '')
                target_suffix = remove_prefix(target_text)

                if ref_node.suffix and target_suffix:
                    len_ref = len(ref_node.suffix)
                    len_tgt = len(target_suffix)

                    # 子串包含，且长度差异不太大
                    if (ref_node.suffix in target_suffix or target_suffix in ref_node.suffix):
                        if abs(len_ref - len_tgt) < 30:
                            candidates.append((line_id, 'substring'))

        return candidates

    def match_node_recursive(ref_node: TreeNode,
                            parent_target_node: Optional[TreeNode] = None,
                            prev_sibling_target_id: Optional[int] = None,
                            nearest_ancestor_target_node: Optional[TreeNode] = None):
        """
        递归匹配节点，边遍历边构建目标树

        Args:
            ref_node: 当前要匹配的参考节点（253）
            parent_target_node: 父节点在目标树中的对应节点（259）
            prev_sibling_target_id: 前一个兄弟节点在目标中的line_id
            nearest_ancestor_target_node: 最近的已匹配祖先节点（用于父节点未匹配时）
        """
        # 确定搜索范围（利用树的位置约束）
        if parent_target_node:
            # 子节点必须在父节点之后
            search_start = parent_target_node.line_id
            search_end = 999999

            # 如果有前一个兄弟，从兄弟之后开始搜索（兄弟约束更强）
            if prev_sibling_target_id is not None:
                search_start = prev_sibling_target_id + 1
        else:
            # 根节点
            search_start = 0
            search_end = 999999

            # 如果有前一个根兄弟（理论上根只有一个）
            if prev_sibling_target_id is not None:
                search_start = prev_sibling_target_id + 1

        # 在范围内查找候选
        candidates = find_candidates_in_range(ref_node, search_start, search_end)

        # 选择最佳候选
        best_target_id = None
        if candidates:
            # 优先级：exact > suffix > substring
            exact_matches = [c for c in candidates if c[1] == 'exact']
            suffix_matches = [c for c in candidates if c[1] == 'suffix']
            substring_matches = [c for c in candidates if c[1] == 'substring']

            if exact_matches:
                best_target_id = exact_matches[0][0]
            elif suffix_matches:
                best_target_id = suffix_matches[0][0]
            elif substring_matches:
                best_target_id = substring_matches[0][0]

        if best_target_id is not None:
            # 记录映射
            mapping[ref_node.line_id] = best_target_id
            used_target_ids.add(best_target_id)
            ref_node.matched_target_id = best_target_id

            # 创建259的树节点（边匹配边构建259树）
            target_line = line_by_id[best_target_id]
            target_node = TreeNode(
                line_id=best_target_id,
                text=target_line.get('text', ''),
                suffix=remove_prefix(target_line.get('text', '')),
                relation=ref_node.relation,  # 使用253的relation
                parent_id=ref_node.parent_id  # 暂时存储253的parent_id（后面会转换）
            )

            # 挂载到259树上
            nonlocal target_root
            if parent_target_node is not None:
                # 父节点存在，挂载到父节点
                parent_target_node.add_child(target_node)
            elif nearest_ancestor_target_node is not None:
                # 父节点未匹配，但有祖先节点，挂载到祖先节点
                nearest_ancestor_target_node.add_child(target_node)
            else:
                # 真正的根节点
                if target_root is None:
                    target_root = target_node
        else:
            # 未找到匹配
            print(f"  警告: 未找到匹配 ref_id={ref_node.line_id}, text='{ref_node.text[:40]}'")
            target_node = None

            # 记录未匹配的节点
            ref_info = ref_nodes.get(ref_node.line_id)
            if ref_info:
                unmatched_nodes.append({
                    'line_id': ref_node.line_id,
                    'text': ref_node.text,
                    'parent_id': ref_node.parent_id,
                    'relation': ref_node.relation,
                })

        # 递归处理子节点
        prev_child_target_id = None
        for child in ref_node.children:
            # 确定nearest_ancestor：如果当前节点匹配上了，就是它；否则继承
            next_nearest_ancestor = target_node if target_node is not None else nearest_ancestor_target_node

            match_node_recursive(child,
                               parent_target_node=target_node,
                               prev_sibling_target_id=prev_child_target_id,
                               nearest_ancestor_target_node=next_nearest_ancestor)

            # 更新前一个兄弟节点的ID
            if child.matched_target_id is not None:
                prev_child_target_id = child.matched_target_id

    # 从根节点开始递归匹配
    match_node_recursive(ref_root)

    return mapping, unmatched_nodes, target_root


def collect_all_nodes(root: TreeNode) -> Dict[int, TreeNode]:
    """收集树中所有节点"""
    nodes = {}

    def collect_recursive(node: TreeNode):
        nodes[node.line_id] = node
        for child in node.children:
            collect_recursive(child)

    collect_recursive(root)
    return nodes


def print_tree(root: TreeNode, max_nodes: int = 50, max_text_len: int = 50):
    """打印树结构"""
    lines = []
    count = [0]  # 使用列表来在闭包中修改

    def format_node(node: TreeNode, indent: int = 0):
        if count[0] >= max_nodes:
            if count[0] == max_nodes:
                lines.append("  " * indent + "... (省略更多节点)")
                count[0] += 1
            return

        text = node.text
        if len(text) > max_text_len:
            text = text[:max_text_len - 3] + "..."

        lines.append("  " * indent + f"├── [line_id={node.line_id}] {text}")
        count[0] += 1

        for child in node.children:
            format_node(child, indent + 1)

    format_node(root)
    print("\n".join(lines))


def build_target_tree_from_mapping(mapping: Dict[int, int], ref_nodes: Dict[int, TreeNode], target_lines: List[dict]) -> Optional[TreeNode]:
    """从映射关系构建目标树（259）

    注意：这里使用修复**后**的parent_id来构建树，
    因为fix_target_file已经修改了target_lines中的parent_id
    """
    # 建立target_line_id索引（已修复）
    target_by_id = {line.get('line_id'): line for line in target_lines if line.get('class') == 'section'}

    # 创建目标树节点（只包含匹配上的section）
    target_nodes = {}
    for ref_id, target_id in mapping.items():
        if target_id not in target_by_id:
            continue

        target_line = target_by_id[target_id]

        # 创建节点（使用修复后的值）
        node = TreeNode(
            line_id=target_id,
            text=target_line.get('text', ''),
            suffix=remove_prefix(target_line.get('text', '')),
            relation=target_line.get('relation', 'contain'),  # 修复后的relation
            parent_id=target_line.get('parent_id', -1),  # 修复后的parent_id
        )
        target_nodes[target_id] = node

    # 建立父子关系（使用修复后的parent_id）
    root = None
    for target_id, node in target_nodes.items():
        if node.parent_id == -1:
            # 根节点
            root = node
        else:
            # 找到父节点
            if node.parent_id in target_nodes:
                target_nodes[node.parent_id].add_child(node)

    return root


def fix_target_file(target_lines: List[dict], target_root: Optional[TreeNode], mapping: Dict[int, int]):
    """
    从259的树提取格式A数据（parent_id + relation），写回target_lines
    只修改 class、parent_id、relation 三个字段
    """
    target_by_id = {line.get('line_id'): line for line in target_lines}

    print(f"\n开始修复...")
    print(f"  需要修复的section数量: {len(mapping)}")

    # 从259的树中提取所有节点
    target_nodes_dict = {}
    def collect_target_nodes(node: TreeNode):
        target_nodes_dict[node.line_id] = node
        for child in node.children:
            collect_target_nodes(child)

    if target_root:
        collect_target_nodes(target_root)

    class_changed = 0
    parent_changed = 0
    relation_changed = 0

    # 修改三个字段
    for target_id, target_node in target_nodes_dict.items():
        if target_id not in target_by_id:
            continue

        target_line = target_by_id[target_id]

        # 1. 修改 class
        if target_line.get('class') != 'section':
            target_line['class'] = 'section'
            class_changed += 1

        # 2. 修改 relation（从259树节点获取，即253的relation）
        if target_line.get('relation') != target_node.relation:
            target_line['relation'] = target_node.relation
            relation_changed += 1

        # 3. 修改 parent_id（从259树结构提取）
        # 根据树的parent关系转换为格式A的parent_id
        if target_node.parent is None:
            # 根节点
            new_parent_id = -1
        else:
            # 父节点的line_id
            new_parent_id = target_node.parent.line_id

        if target_line.get('parent_id') != new_parent_id:
            target_line['parent_id'] = new_parent_id
            parent_changed += 1

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
    print("修复 section class、parent_id 和 relation（基于双树深度优先匹配）")
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

    # 构建参考树（253）
    print("构建参考文件的section树...")
    ref_root = build_tree(ref_lines)
    if not ref_root:
        print("[错误] 参考文件没有section节点")
        sys.exit(1)

    ref_nodes = collect_all_nodes(ref_root)
    print(f"  参考树节点数: {len(ref_nodes)}")
    print()

    # 深度优先匹配，边遍历边构建目标树
    print("深度优先匹配参考树到目标文件...")
    mapping, unmatched_nodes, target_root = match_trees(ref_root, target_lines, ref_nodes)
    print(f"  成功映射: {len(mapping)} / {len(ref_nodes)} 个")
    print()

    # 打印未匹配的元素详情
    if unmatched_nodes:
        print(f"未匹配的section节点详情 (共{len(unmatched_nodes)}个):")
        print("=" * 80)
        for node_info in unmatched_nodes:
            print(f"  line_id={node_info['line_id']}")
            print(f"    text: {node_info['text']}")
            print(f"    parent_id: {node_info['parent_id']}")
            print(f"    relation: {node_info['relation']}")
            print()

    # 打印253树结构
    print("253参考树结构:")
    print("=" * 80)
    print_tree(ref_root, max_nodes=50)
    print()

    # 打印259匹配后的树结构
    print("259匹配后的树结构:")
    print("=" * 80)
    if target_root:
        print_tree(target_root, max_nodes=50)
    else:
        print("  (无法构建259树)")
    print()

    # 修复目标文件（从259树提取格式A数据）
    fixed_lines = fix_target_file(target_lines, target_root, mapping)

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

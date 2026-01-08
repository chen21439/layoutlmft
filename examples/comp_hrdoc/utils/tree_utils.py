"""树构建工具函数

用于构建层级文档结构树，被 metrics/teds.py 和 data/hrdoc_loader.py 复用。

数据处理流程
============

    平铺 JSON (parent_id + relation)
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │  build_doc_tree_with_nodes()        │
    │                                     │
    │  1. 创建 Node 对象                   │
    │  2. contain/connect → 直接继承父节点 │
    │  3. equality → 回溯找最老兄弟        │
    │  4. 建立 node.parent 层级关系       │
    └─────────────────────────────────────┘
                  │
                  ▼
           Node 树 + 节点列表
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │  extract_parents_and_siblings()     │
    │                                     │
    │  从树中提取:                         │
    │  - hierarchical_parents (从 node.parent) │
    │  - sibling_groups (按 parent 分组)  │
    └─────────────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │  Collator 输出 (Tensor)             │
    │  ├── parent_ids:     [B, N]         │ → 训练 parent_head
    │  └── sibling_labels: [B, N, N]      │ → 训练 sibling_head
    └─────────────────────────────────────┘

关键逻辑（来自 CompHRDoc/evaluation/hrdoc_tool/doc_utils.py）
==========================================================

关系类型:
- contain: 父子包含关系
- connect: 阅读顺序延续 (line → next line)
- equality: 兄弟关系 (section1 ←→ section2)

层级父节点计算:
- contain/connect 关系：节点直接成为 ref_parent 的子节点
- equality 关系：沿着 ref_parent 链回溯找到最老的兄弟，
  然后成为最老兄弟的父节点的子节点（即与最老兄弟同级）

示例
====

原始数据:
    A --contain--> B --equality--> C --equality--> D

    parent_ids = [-1, 0, 1, 2]   # A→ROOT, B→A, C→B, D→C
    relations  = [contain, contain, equality, equality]

处理后:
    树结构：A -> [B, C, D]  (B, C, D 都是 A 的子节点)

    hierarchical_parents = [-1, 0, 0, 0]  # A→ROOT, B/C/D→A
    sibling_groups = [[1, 2, 3]]          # B, C, D 互为兄弟

复用说明
=======

- metrics/teds.py: 使用 generate_doc_tree() 构建完整树用于 TEDS 计算
- data/*_loader.py: 使用 build_doc_tree_with_nodes() + extract_parents_and_siblings() 计算训练标签
"""

from typing import List, Dict, Tuple, Optional, Any, Union

# relation 映射从核心库导入，避免多处定义不一致
try:
    from layoutlmft.models.relation_classifier import RELATION_LABELS
except ImportError:
    RELATION_LABELS = {'connect': 0, 'contain': 1, 'equality': 2}

ID2RELATION = {v: k for k, v in RELATION_LABELS.items()}

# 兼容旧名称
RELATION_STR_TO_INT = RELATION_LABELS
RELATION_INT_TO_STR = ID2RELATION


class Node:
    """文档树节点

    来自 CompHRDoc/evaluation/hrdoc_tool/doc_utils.py

    Attributes:
        name: 节点名称/文本
        info: 附加信息（可存储原始索引等）
        children: 子节点列表（实际的层级子节点）
        parent: 父节点（实际的层级父节点）
        ref_children: 引用子节点列表（原始数据中的直接引用）
        ref_parent: 引用父节点（原始数据中的直接引用）
        ref_parent_relation: 与引用父节点的关系类型
        depth: 树的深度
    """

    def __init__(self, name: str, info: Any = None):
        self.name = name
        self.info = info
        self.children: List['Node'] = []
        self.parent: Optional['Node'] = None
        self.ref_children: List['Node'] = []
        self.ref_parent: Optional['Node'] = None
        self.ref_parent_relation: Optional[str] = None
        self.depth: int = 0

    def _set_parent(self, node: 'Node'):
        self.parent = node

    def _set_ref_parent(self, node: 'Node', relation: str):
        self.ref_parent = node
        self.ref_parent_relation = relation

    def add_child(self, node: 'Node'):
        self.children.append(node)
        node._set_parent(self)

    def add_ref_child(self, node: 'Node', relation: str):
        self.ref_children.append(node)
        node._set_ref_parent(self, relation)

    def set_depth(self, cur_depth: int):
        self.depth = cur_depth
        for child in self.children:
            child.set_depth(cur_depth + 1)

    def __repr__(self):
        return self.name

    def __len__(self):
        length = 1
        for child in self.children:
            length += len(child)
        return length


def normalize_relation(relation: Union[str, int]) -> str:
    """将 relation 统一转换为字符串格式

    Args:
        relation: 关系类型，可以是字符串或整数

    Returns:
        关系字符串 ('contain', 'equality', 'connect')
    """
    if isinstance(relation, str):
        return relation
    elif isinstance(relation, int):
        return RELATION_INT_TO_STR.get(relation, 'contain')
    else:
        return 'contain'


def _build_tree_relations(nodes: List[Node], root: Node,
                          parent_ids: List[int],
                          relations: List[Union[str, int]]) -> None:
    """建立树的父子关系（内部函数）

    来自 CompHRDoc/evaluation/hrdoc_tool/doc_utils.py 的核心逻辑

    Args:
        nodes: 节点列表（不含 ROOT）
        root: ROOT 节点
        parent_ids: 父节点索引列表
        relations: 关系类型列表
    """
    n = len(nodes)
    all_nodes = [root] + nodes  # index 0 = ROOT, index i+1 = nodes[i]

    for i in range(n):
        node = nodes[i]
        ref_parent_idx = parent_ids[i]
        relation = normalize_relation(relations[i])

        # 获取 ref_parent（-1 表示 ROOT）
        if ref_parent_idx == -1:
            ref_parent = root
        else:
            ref_parent = nodes[ref_parent_idx]

        # 建立引用关系
        ref_parent.add_ref_child(node, relation)

        # 建立层级关系
        if relation == 'contain':
            # contain: 直接成为 ref_parent 的子节点（层级包含）
            ref_parent.add_child(node)
        elif relation == 'connect':
            # connect: 阅读顺序延续，应该与 ref_parent 是兄弟（相同的层级父节点）
            if ref_parent.parent:
                ref_parent.parent.add_child(node)
            else:
                # ref_parent 是 root 下的顶层节点，当前节点也是
                root.add_child(node)
        elif relation == 'equality':
            # equality 关系：沿着 ref_parent 链回溯找到最老的兄弟
            # 然后成为最老兄弟的父节点的子节点
            oldest_bro = node.ref_parent
            while oldest_bro.ref_parent_relation == 'equality':
                oldest_bro = oldest_bro.ref_parent
            if oldest_bro.parent:
                oldest_bro.parent.add_child(node)


def build_doc_tree_with_nodes(
    parent_ids: List[int],
    relations: List[Union[str, int]],
) -> Tuple[Node, List[Node]]:
    """构建文档树并返回节点列表

    Args:
        parent_ids: 父节点索引列表（-1 表示 ROOT）
        relations: 关系类型列表

    Returns:
        root: ROOT 节点
        nodes: 节点列表，nodes[i] 对应原始数据第 i 个元素，
               每个节点的 node.parent 是其层级父节点
    """
    n = len(parent_ids)
    assert len(relations) == n

    # 创建节点，info 中存储原始索引
    root = Node(name='ROOT')
    nodes = [Node(name=f"node_{i}", info={'index': i}) for i in range(n)]

    # 建立父子关系
    _build_tree_relations(nodes, root, parent_ids, relations)

    root.set_depth(cur_depth=0)
    return root, nodes


def generate_doc_tree(
    texts: List[str],
    parent_ids: List[int],
    relations: List[Union[str, int]],
) -> Node:
    """从模型输出构建文档树（用于 TEDS 评估）

    Args:
        texts: 节点文本列表 (格式: "class:text" 或纯文本)
        parent_ids: 父节点索引列表（-1 表示 ROOT）
        relations: 关系类型列表

    Returns:
        文档树根节点
    """
    assert len(texts) == len(parent_ids) == len(relations)

    n = len(texts)
    root = Node(name='ROOT')
    nodes = [Node(name=text) for text in texts]

    _build_tree_relations(nodes, root, parent_ids, relations)

    root.set_depth(cur_depth=0)
    return root


def extract_parents_and_siblings(
    nodes: List[Node],
) -> Tuple[List[int], List[List[int]]]:
    """从树节点列表中提取层级父节点和兄弟分组

    Args:
        nodes: 节点列表，每个节点的 node.parent 已设置为层级父节点

    Returns:
        hierarchical_parents: 每个节点的层级父节点索引，-1 表示 ROOT
        sibling_groups: 兄弟节点分组列表
    """
    n = len(nodes)

    # 从 node.parent 提取层级父节点索引
    hierarchical_parents = []
    for node in nodes:
        if node.parent is None or node.parent.name == 'ROOT':
            hierarchical_parents.append(-1)
        else:
            # parent 的原始索引存储在 info['index'] 中
            parent_idx = node.parent.info['index']
            hierarchical_parents.append(parent_idx)

    # 按 hierarchical_parent 分组得到兄弟
    parent_to_children: Dict[int, List[int]] = {}
    for i in range(n):
        hp = hierarchical_parents[i]
        if hp not in parent_to_children:
            parent_to_children[hp] = []
        parent_to_children[hp].append(i)

    # 兄弟分组：同一 parent 下有多个子节点
    sibling_groups = [group for group in parent_to_children.values() if len(group) > 1]

    return hierarchical_parents, sibling_groups


def resolve_hierarchical_parents_and_siblings(
    parent_ids: List[int],
    relations: List[Union[str, int]],
) -> Tuple[List[int], List[List[int]]]:
    """从 parent_ids 和 relations 解析出层级父节点和兄弟分组

    这是方案 B 的实现：先构建树，再从树中提取。

    Args:
        parent_ids: 父节点索引列表（-1 表示 ROOT）
        relations: 关系类型列表

    Returns:
        hierarchical_parents: 每个节点的层级父节点索引，-1 表示 ROOT
        sibling_groups: 兄弟节点分组列表
    """
    # 1. 构建树
    root, nodes = build_doc_tree_with_nodes(parent_ids, relations)

    # 2. 从树中提取 parent 和 sibling
    return extract_parents_and_siblings(nodes)


# 保持向后兼容的别名
resolve_parent_and_sibling_from_tree = resolve_hierarchical_parents_and_siblings


def build_sibling_matrix(
    num_regions: int,
    sibling_groups: List[List[int]],
) -> List[List[int]]:
    """根据兄弟分组构建兄弟关系矩阵

    Args:
        num_regions: 区域数量
        sibling_groups: 兄弟节点分组

    Returns:
        sibling_matrix: num_regions x num_regions 的矩阵，
            sibling_matrix[i][j] = 1 表示 i 和 j 是兄弟
    """
    matrix = [[0] * num_regions for _ in range(num_regions)]

    for group in sibling_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                idx_i, idx_j = group[i], group[j]
                if idx_i < num_regions and idx_j < num_regions:
                    matrix[idx_i][idx_j] = 1
                    matrix[idx_j][idx_i] = 1

    return matrix


__all__ = [
    'Node',
    'RELATION_STR_TO_INT',
    'RELATION_INT_TO_STR',
    'normalize_relation',
    'build_doc_tree_with_nodes',
    'generate_doc_tree',
    'extract_parents_and_siblings',
    'resolve_hierarchical_parents_and_siblings',
    'resolve_parent_and_sibling_from_tree',  # 向后兼容别名
    'build_sibling_matrix',
]

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
            visited = {id(oldest_bro)}  # 防止循环引用导致死循环
            while oldest_bro.ref_parent_relation == 'equality':
                oldest_bro = oldest_bro.ref_parent
                if id(oldest_bro) in visited:
                    # 检测到循环引用，跳出
                    break
                visited.add(id(oldest_bro))
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


def build_tree_from_parents(
    predictions: List[Dict],
    id_key: str = "line_id",
    parent_key: str = "toc_parent",
) -> List[Dict]:
    """从 parent 预测结果构建嵌套树结构（用于可视化/日志）

    支持两种方案判断顶层节点：
    - 格式A: parent_id == -1 表示顶层节点
    - 格式B: parent_id == node_id（自指向）表示顶层节点

    注意：此函数只构建嵌套树用于可视化。
    如需完整的扁平格式A（包含所有行），请使用 flatten_full_tree_to_format_a()

    Args:
        predictions: 预测结果列表，每个元素包含 id_key 和 parent_key
        id_key: 节点 ID 的字段名
        parent_key: 父节点 ID 的字段名

    Returns:
        嵌套树结构列表，每个节点包含 children 数组

    Example:
        格式A输入:
        [
            {"line_id": 0, "parent_id": -1, "text": "第一章"},  # 顶层 (parent=-1)
            {"line_id": 5, "parent_id": 0, "text": "1.1 节"},
        ]

        格式B输入:
        [
            {"line_id": 0, "toc_parent": 0, "text": "第一章"},  # 顶层 (自指向)
            {"line_id": 5, "toc_parent": 0, "text": "1.1 节"},
        ]
    """
    if not predictions:
        return []

    # 构建 id -> node 映射，每个 node 添加 children 字段
    id_to_node = {}
    for pred in predictions:
        node_id = pred[id_key]
        node = {**pred, "children": []}
        id_to_node[node_id] = node

    # 构建父子关系
    roots = []
    for pred in predictions:
        node_id = pred[id_key]
        parent_id = pred[parent_key]
        node = id_to_node[node_id]

        if parent_id == -1:
            # 格式A: parent_id == -1 表示顶层节点
            roots.append(node)
        elif parent_id == node_id:
            # 格式B: 自指向 = 顶层节点
            roots.append(node)
        elif parent_id in id_to_node:
            # 添加到父节点的 children
            id_to_node[parent_id]["children"].append(node)
        else:
            # 父节点不存在，作为顶层处理
            roots.append(node)

    return roots


def format_tree_from_parents(
    parents: List[int],
    texts: List[str],
    mask: Optional[List[bool]] = None,
    max_text_len: int = 40,
) -> List[str]:
    """从 parent 列表构建树形可视化字符串

    复用自 train_doc.py 的 build_tree_str 逻辑。
    注意：parents 的值应该是 0-based 索引，不是原始 line_id。

    Args:
        parents: 父节点索引列表，自指向(parent==self)表示 root
        texts: 节点文本列表
        mask: 可选的掩码，True 表示包含该节点
        max_text_len: 文本最大显示长度

    Returns:
        树形字符串列表

    Example:
        ├── [0] 第一章 总则
          ├── [1] 一、适用范围
          ├── [2] 二、定义
        ├── [3] 第二章 招标内容
    """
    n = len(parents)
    children = {i: [] for i in range(-1, n)}

    for i, p in enumerate(parents):
        if mask is None or mask[i]:
            # 自指向方案：parent == self 表示 root，映射到 -1
            if p == i:
                children[-1].append(i)
            else:
                children[p].append(i)

    def _format(node_idx, indent=0):
        result = []
        for child in children.get(node_idx, []):
            text = texts[child] if child < len(texts) else f"[{child}]"
            if len(text) > max_text_len:
                text = text[:max_text_len - 3] + "..."
            result.append("  " * indent + f"├── [{child}] {text}")
            result.extend(_format(child, indent + 1))
        return result

    return _format(-1)  # 从 root (-1) 开始


def format_toc_tree(
    toc_tree: List[Dict],
    id_key: str = "line_id",
    text_key: str = "text",
    max_text_len: int = 40,
) -> List[str]:
    """从嵌套树结构格式化为可视化字符串

    用于 API 推理输出，直接从 build_tree_from_parents 生成的嵌套结构格式化。

    Args:
        toc_tree: 嵌套树结构（由 build_tree_from_parents 生成）
        id_key: 节点 ID 字段名
        text_key: 文本字段名
        max_text_len: 文本最大显示长度

    Returns:
        树形字符串列表

    Example:
        ├── [5] 第一章 总则
          ├── [12] 一、适用范围
          ├── [25] 二、定义
        ├── [30] 第二章 招标内容
    """
    lines = []

    def _format(node: Dict, indent: int = 0):
        node_id = node.get(id_key, "?")
        text = node.get(text_key, "")
        if len(text) > max_text_len:
            text = text[:max_text_len - 3] + "..."
        lines.append("  " * indent + f"├── [{node_id}] {text}")

        for child in node.get("children", []):
            _format(child, indent + 1)

    for root in toc_tree:
        _format(root, 0)

    return lines


def build_complete_tree(
    line_ids: List[int],
    categories: List[int],
    toc_parents: List[int],
    texts: Optional[List[str]] = None,
    section_category: int = 0,
) -> Tuple[Node, List[Node]]:
    """从 TOC + Reading Order + 类别构建完整文档树

    根据论文 Figure 2 和 Section 3 的描述：
    - TOC 树只包含 Section 标题（层级骨架）
    - 非 Section 节点（Paragraph、Caption、Table 等）根据 Reading Order
      挂载到其前面最近的 Section 节点下

    算法流程（单遍遍历，保持阅读顺序）：
    1. 按 line_id 排序所有节点（line_id 暂代 Reading Order）
    2. 遍历每个节点：
       - Section 节点：根据 toc_parent 挂载到 TOC 父节点
       - 非 Section 节点：挂载到当前最近的 Section 节点

    Args:
        line_ids: 所有行的 ID（用于排序，暂代 Reading Order）
        categories: 每行的语义类别
        toc_parents: 每行的 TOC parent（自指向表示 root Section，非 Section 可为 -1）
        texts: 可选的节点文本列表
        section_category: Section 类别的 ID，默认 0

    Returns:
        root: ROOT 节点
        nodes: 所有节点列表，按原始顺序（与 line_ids 对应）

    Example:
        输入:
          line_ids:   [0,       1,         2,          3,       4        ]
          categories: [0,       1,         1,          0,       1        ]  # 0=Section, 1=Paragraph
          toc_parents:[0,       -1,        -1,         0,       -1       ]

        输出树结构:
          ROOT
          └── [0] Section
                ├── [1] Paragraph
                ├── [2] Paragraph
                └── [3] Section
                      └── [4] Paragraph

    TODO: 当 Order 模块训练完成后，使用预测的阅读顺序替代 line_id
    """
    n = len(line_ids)
    assert len(categories) == n and len(toc_parents) == n

    # 创建 ROOT 和所有节点
    root = Node(name='ROOT')
    nodes = []
    for i in range(n):
        name = texts[i] if texts else f"node_{i}"
        node = Node(name=name, info={
            'index': i,
            'line_id': line_ids[i],
            'category': categories[i],
            'toc_parent': toc_parents[i],
        })
        nodes.append(node)

    # 按 line_id 排序得到阅读顺序
    sorted_indices = sorted(range(n), key=lambda i: line_ids[i])

    # 构建 line_id -> index 映射
    line_id_to_idx = {line_ids[i]: i for i in range(n)}

    # 单遍遍历：按阅读顺序处理所有节点
    current_section: Optional[Node] = None

    for idx in sorted_indices:
        cat = categories[idx]
        toc_parent_id = toc_parents[idx]
        node = nodes[idx]

        if cat == section_category:
            # Section 节点：根据 TOC parent 挂载
            if toc_parent_id == line_ids[idx]:
                # 自指向 = root Section，挂载到 ROOT
                root.add_child(node)
            elif toc_parent_id in line_id_to_idx:
                # 挂载到其 TOC parent
                parent_idx = line_id_to_idx[toc_parent_id]
                nodes[parent_idx].add_child(node)
            else:
                # TOC parent 不存在，作为顶层 Section
                root.add_child(node)
            # 更新当前 Section
            current_section = node
        else:
            # 非 Section 节点：挂载到当前最近的 Section
            if current_section is not None:
                current_section.add_child(node)
            else:
                # 还没有遇到任何 Section，挂载到 ROOT
                root.add_child(node)

    root.set_depth(cur_depth=0)
    return root, nodes


def format_complete_tree(
    root: Node,
    max_text_len: int = 40,
    show_category: bool = True,
) -> List[str]:
    """格式化完整文档树为可视化字符串

    Args:
        root: 完整树的 ROOT 节点
        max_text_len: 文本最大显示长度
        show_category: 是否显示类别

    Returns:
        树形字符串列表
    """
    lines = []

    def _format(node: Node, indent: int = 0):
        if node.name == 'ROOT':
            for child in node.children:
                _format(child, indent)
            return

        info = node.info or {}
        line_id = info.get('line_id', '?')
        cat = info.get('category', '?')

        text = node.name
        if len(text) > max_text_len:
            text = text[:max_text_len - 3] + "..."

        if show_category:
            lines.append("  " * indent + f"├── [{line_id}|cat={cat}] {text}")
        else:
            lines.append("  " * indent + f"├── [{line_id}] {text}")

        for child in node.children:
            _format(child, indent + 1)

    _format(root, 0)
    return lines


# ============================================================================
# 反向转换：从 hierarchical_parent + sibling 恢复 ref_parent + relation
# ============================================================================


def build_tree_from_hierarchical_parents(
    hierarchical_parents: List[int],
    texts: Optional[List[str]] = None,
) -> Tuple[Node, List[Node]]:
    """从层级父节点列表构建树（反向：用于推理输出）

    Args:
        hierarchical_parents: 每个节点的层级父节点索引，-1 表示 ROOT
                              或自指向表示 ROOT（自指向方案）
        texts: 可选的节点文本列表

    Returns:
        root: ROOT 节点
        nodes: 节点列表
    """
    n = len(hierarchical_parents)

    root = Node(name='ROOT')
    nodes = []
    for i in range(n):
        name = texts[i] if texts else f"node_{i}"
        node = Node(name=name, info={'index': i})
        nodes.append(node)

    for i in range(n):
        hp = hierarchical_parents[i]
        if hp == -1 or hp == i:  # ROOT（-1 或自指向）
            root.add_child(nodes[i])
        elif 0 <= hp < n:
            nodes[hp].add_child(nodes[i])
        else:
            # 无效的 parent，作为 ROOT 子节点
            root.add_child(nodes[i])

    root.set_depth(cur_depth=0)
    return root, nodes


def extract_ref_parents_and_relations(
    nodes: List[Node],
    left_siblings: Optional[List[int]] = None,
) -> Tuple[List[int], List[str]]:
    """从树中提取 ref_parent 和 relation（反向转换）

    根据论文定义：
    - contain: 父子包含关系
    - equality: 兄弟关系（有左兄弟时）
    - connect: 阅读顺序延续（无左兄弟的后续兄弟 - 简化为 contain）

    Args:
        nodes: 节点列表，node.parent 已设置
        left_siblings: 可选的左兄弟索引列表，-1 表示无左兄弟

    Returns:
        ref_parents: 引用父节点索引列表（-1 表示 ROOT）
        relations: 关系类型列表 ('contain', 'equality')
    """
    n = len(nodes)
    ref_parents = []
    relations = []

    for i in range(n):
        node = nodes[i]

        # 检查是否有左兄弟
        has_left_sibling = (
            left_siblings is not None and
            i < len(left_siblings) and
            left_siblings[i] >= 0
        )

        if has_left_sibling:
            # equality: 指向左兄弟
            ref_parents.append(left_siblings[i])
            relations.append('equality')
        else:
            # contain: 指向层级父节点
            if node.parent is None or node.parent.name == 'ROOT':
                ref_parents.append(-1)
            else:
                parent_idx = node.parent.info.get('index', -1)
                ref_parents.append(parent_idx)
            relations.append('contain')

    return ref_parents, relations


def resolve_ref_parents_and_relations(
    hierarchical_parents: List[int],
    left_siblings: Optional[List[int]] = None,
) -> Tuple[List[int], List[str]]:
    """从 hierarchical_parents 和 left_siblings 解析出 ref_parent 和 relation

    反向转换：格式B → 树 → 格式A

    两种格式：
        格式A: ref_parent + relation (原始标注/输出)
               - 顶层节点 parent = -1
        格式B: hierarchical_parent + sibling (训练标签/预测)
               - 顶层节点 parent = 自己的索引（自指向）
               - 无左兄弟时 sibling = 自己的索引（自指向）
               - 原因：softmax + cross_entropy 不能用 -1 作为目标

    Args:
        hierarchical_parents: 层级父节点索引列表（格式B：顶层节点自指向）
        left_siblings: 左兄弟索引列表（格式B：无左兄弟时自指向）

    Returns:
        ref_parents: 引用父节点索引列表（格式A：顶层节点为 -1）
        relations: 关系类型列表

    Example:
        输入（格式B）:
            hierarchical_parents = [0, 0, 0, 0]  # 节点0自指向=顶层, 节点1/2/3指向0
            left_siblings = [0, 1, 1, 2]         # 节点0/1自指向=无左兄弟, 节点2左兄弟是1, 节点3左兄弟是2

        输出（格式A）:
            ref_parents = [-1, 0, 1, 2]          # 节点0→ROOT, 节点1→0, 节点2→1, 节点3→2
            relations = ['contain', 'contain', 'equality', 'equality']
    """
    n = len(hierarchical_parents)

    # 格式B使用自指向，转换为内部表示（-1 表示 ROOT/无左兄弟）
    hp_converted = [
        -1 if hierarchical_parents[i] == i else hierarchical_parents[i]
        for i in range(n)
    ]

    ls_converted = None
    if left_siblings is not None:
        ls_converted = [
            -1 if left_siblings[i] == i else left_siblings[i]
            for i in range(len(left_siblings))
        ]

    # 1. 构建树（使用转换后的值）
    root, nodes = build_tree_from_hierarchical_parents(hp_converted)

    # 2. 提取 ref_parent 和 relation（使用转换后的值）
    return extract_ref_parents_and_relations(nodes, ls_converted)


def flatten_full_tree_to_format_a(
    section_predictions: List[Dict],
    all_lines: List[Dict],
    section_ids: set,
    id_key: str = "line_id",
) -> List[Dict]:
    """将完整树转换为扁平的格式A

    根据论文 Figure 2：非 section 节点按阅读顺序挂载到其前面最近的 section 下。

    Args:
        section_predictions: section 的预测结果（格式A）
            每个: {"line_id": int, "parent_id": int, "relation": str, ...}
        all_lines: 全量数据（按阅读顺序）
            每行: {"line_id": int, "text": str, "class": ..., ...}
        section_ids: section 的 line_id 集合
        id_key: 节点 ID 字段名

    Returns:
        完整的格式A列表，包含所有行
            section: 保持原有的 parent_id 和 relation
            非 section: parent_id 指向所属 section，relation 为 "contain"
    """
    # 构建 section line_id -> prediction 映射
    section_pred_map = {pred[id_key]: pred for pred in section_predictions}

    # 按阅读顺序遍历，确定每行的 parent
    results = []
    current_section_id = None  # 当前最近的 section 的 line_id

    for line in all_lines:
        line_id = line.get(id_key)
        if line_id is None:
            continue

        if line_id in section_ids:
            # section 节点：使用模型预测的 parent_id 和 relation
            pred = section_pred_map.get(line_id, {})
            results.append({
                id_key: line_id,
                "parent_id": pred.get("parent_id", -1),
                "relation": pred.get("relation", "contain"),
                "text": line.get("text", pred.get("text", "")),
                "class": line.get("class", line.get("category", "")),
                "is_section": True,
            })
            current_section_id = line_id
        else:
            # 非 section 节点：parent 指向当前最近的 section
            results.append({
                id_key: line_id,
                "parent_id": current_section_id if current_section_id is not None else -1,
                "relation": "contain",
                "text": line.get("text", ""),
                "class": line.get("class", line.get("category", "")),
                "is_section": False,
            })

    return results


def insert_content_to_toc_tree(
    toc_tree: List[Dict],
    all_lines: List[Dict],
    section_line_ids: set,
) -> List[Dict]:
    """将非 section 内容插入到 TOC 树中

    按阅读顺序，非 section 节点属于它前面最近的 section。

    Args:
        toc_tree: section-only 的 TOC 树（嵌套结构）
            每个节点: {"line_id": int, "text": str, "children": [...]}
        all_lines: 所有行的列表，按阅读顺序
            每行: {"line_id": int, "text": str, "class": int/str, ...}
        section_line_ids: section 的 line_id 集合

    Returns:
        完整的文档树，每个 section 节点增加 "content" 字段存放非 section 内容
    """
    import copy
    toc_tree = copy.deepcopy(toc_tree)

    # 1. 构建 line_id -> section 节点的映射
    section_nodes = {}

    def collect_sections(nodes):
        for node in nodes:
            section_nodes[node["line_id"]] = node
            if "content" not in node:
                node["content"] = []
            collect_sections(node.get("children", []))

    collect_sections(toc_tree)

    # 2. 按阅读顺序遍历，将非 section 插入到对应 section
    current_section = None
    preamble = []  # 第一个 section 之前的内容

    for line in all_lines:
        line_id = line.get("line_id")
        if line_id is None:
            continue

        if line_id in section_line_ids:
            # 是 section，更新当前 section
            current_section = section_nodes.get(line_id)
        else:
            # 非 section，插入到当前 section
            content_item = {
                "line_id": line_id,
                "text": line.get("text", ""),
                "class": line.get("class", ""),
            }
            if current_section is not None:
                current_section["content"].append(content_item)
            else:
                # 在第一个 section 之前
                preamble.append(content_item)

    # 3. 处理 preamble（放到结果的开头）
    if preamble:
        # 作为一个虚拟的 preamble 节点，或者直接放到第一个 section
        if toc_tree and len(toc_tree) > 0:
            # 放到第一个顶层 section 的 content 开头
            if "content" not in toc_tree[0]:
                toc_tree[0]["content"] = []
            toc_tree[0]["content"] = preamble + toc_tree[0]["content"]
        else:
            # 没有 section，创建一个 preamble 节点
            toc_tree = [{
                "line_id": -1,
                "text": "[Preamble]",
                "children": [],
                "content": preamble,
            }]

    return toc_tree


__all__ = [
    'Node',
    'RELATION_STR_TO_INT',
    'RELATION_INT_TO_STR',
    'normalize_relation',
    # 双向转换
    #   格式A: ref_parent + relation (原始标注/输出)
    #          - 顶层节点 parent = -1
    #   格式B: hierarchical_parent + sibling (训练标签/预测)
    #          - 顶层节点 parent = 自己的索引（自指向）
    #          - 无左兄弟时 sibling = 自己的索引（自指向）
    #          - 原因：softmax + cross_entropy 不能用 -1 作为目标
    # 正向：A → 树 → B
    'build_doc_tree_with_nodes',
    'generate_doc_tree',
    'extract_parents_and_siblings',
    'resolve_hierarchical_parents_and_siblings',
    'resolve_parent_and_sibling_from_tree',  # 向后兼容别名
    'build_sibling_matrix',
    # 反向：B → 树 → A
    'build_tree_from_hierarchical_parents',
    'extract_ref_parents_and_relations',
    'resolve_ref_parents_and_relations',
    # 工具函数
    'build_tree_from_parents',
    'format_tree_from_parents',
    'format_toc_tree',
    'build_complete_tree',
    'format_complete_tree',
    # 非 section 内容插入
    'insert_content_to_toc_tree',
    # 完整树转扁平格式A
    'flatten_full_tree_to_format_a',
]

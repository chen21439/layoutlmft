#!/usr/bin/env python
# coding=utf-8
"""
文档结构树定义和构建工具
实现HRDoc论文的Overall Task：将三个子任务的输出组合成树结构
"""

import json
import sys
import os
from typing import List, Dict, Optional, Any
from collections import defaultdict

# 【重要】HRDS BIO标签定义（和训练时完全一致）
# 数据来源：layoutlmft/data/datasets/hrdoc.py 中的 _LABELS
# 注意：这是唯一的标签定义来源，训练和推理必须使用相同的标签
_LABELS = [
    "O",
    "B-AFFILI", "I-AFFILI",
    "B-ALG", "I-ALG",
    "B-AUTHOR", "I-AUTHOR",
    "B-EQU", "I-EQU",                # equation
    "B-FIG", "I-FIG",                # figure
    "B-FIGCAP", "I-FIGCAP",          # figure caption
    "B-FNOTE", "I-FNOTE",            # footnote
    "B-FOOT", "I-FOOT",              # footer
    "B-FSTLINE", "I-FSTLINE",        # first line
    "B-MAIL", "I-MAIL",
    "B-OPARA", "I-OPARA",            # other paragraph
    "B-PARA", "I-PARA",              # paragraph
    "B-SEC1", "I-SEC1",              # section level 1
    "B-SEC2", "I-SEC2",              # section level 2
    "B-SEC3", "I-SEC3",              # section level 3
    "B-SEC4", "I-SEC4",              # section level 4
    "B-SECX", "I-SECX",              # section level X (additional)
    "B-TAB", "I-TAB",                # table
    "B-TABCAP", "I-TABCAP",          # table caption
    "B-TITLE", "I-TITLE",
]

# 将标签列表转换为字典映射（label_id → label_name）
# 模型预测的是数字 ID（0-42），我们用这个映射转换成标签名称
LABEL_MAP = {i: label for i, label in enumerate(_LABELS)}

# 注释：
# - 训练时：run_hrdoc.py 从数据集 features 读取 label_list = _LABELS
# - 推理时：我们直接从同一个源导入 _LABELS
# - 这样保证了标签映射的一致性：模型预测的数字 ID 可以正确映射回标签名称

# 关系类型映射
RELATION_MAP = {
    0: "none",
    1: "connect",
    2: "contain",
    3: "equality",
}


class DocumentNode:
    """
    文档结构树的节点

    每个节点代表一个语义单元（semantic unit / line）
    """

    def __init__(
        self,
        idx: int,
        label: str,
        bbox: Optional[List[float]] = None,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
    ):
        """
        Args:
            idx: 节点在文档中的索引（行号）
            label: 语义类别标签（如 "Title", "Section", "Para-Line"）
            bbox: 边界框 [x1, y1, x2, y2]
            text: 文本内容（可选）
            confidence: 分类置信度（可选）
        """
        self.idx = idx
        self.label = label
        # 安全处理bbox（可能是numpy数组或list）
        if bbox is not None:
            # 如果是numpy数组，转换为list
            if hasattr(bbox, 'tolist'):
                self.bbox = bbox.tolist()
            else:
                self.bbox = list(bbox) if bbox is not None else [0, 0, 0, 0]
        else:
            self.bbox = [0, 0, 0, 0]
        self.text = text
        self.confidence = confidence

        # 树结构
        self.parent: Optional[DocumentNode] = None
        self.children: List[DocumentNode] = []

        # 关系信息
        self.relation_to_parent: Optional[str] = None  # connect/contain/equality
        self.relation_confidence: Optional[float] = None

    def add_child(self, child: 'DocumentNode', relation: str = "contain", confidence: float = None):
        """添加子节点"""
        child.parent = self
        child.relation_to_parent = relation
        child.relation_confidence = confidence
        self.children.append(child)

    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """转换为字典格式（用于JSON序列化）"""
        result = {
            "idx": self.idx,
            "label": self.label,
            "bbox": self.bbox,
            "text": self.text,
            "confidence": self.confidence,
            "relation_to_parent": self.relation_to_parent,
            "relation_confidence": self.relation_confidence,
        }

        if include_children and self.children:
            result["children"] = [child.to_dict(include_children=True) for child in self.children]

        return result

    def __repr__(self):
        return f"Node(idx={self.idx}, label={self.label}, children={len(self.children)})"


class DocumentTree:
    """
    文档结构树

    根据三个子任务的输出构建：
    1. SubTask 1: 语义分类 (line_labels)
    2. SubTask 2: 父节点查找 (parent_indices)
    3. SubTask 3: 关系分类 (relation_types)
    """

    def __init__(self):
        self.root = DocumentNode(idx=-1, label="ROOT")
        self.nodes: List[DocumentNode] = []

    @classmethod
    def from_predictions(
        cls,
        line_labels: List[int],  # SubTask 1 输出
        parent_indices: List[int],  # SubTask 2 输出
        relation_types: List[int],  # SubTask 3 输出
        line_bboxes: Optional[List[List[float]]] = None,
        line_texts: Optional[List[str]] = None,
        label_confidences: Optional[List[float]] = None,
        relation_confidences: Optional[List[float]] = None,
    ) -> 'DocumentTree':
        """
        从三个子任务的预测结果构建文档树

        Args:
            line_labels: 每个行的语义类别标签 [N]
            parent_indices: 每个行的父节点索引 [N]，-1表示ROOT
            relation_types: 每个行与其父节点的关系类型 [N]
            line_bboxes: 每个行的边界框 [N, 4]（可选）
            line_texts: 每个行的文本内容 [N]（可选）
            label_confidences: 语义分类的置信度 [N]（可选）
            relation_confidences: 关系分类的置信度 [N]（可选）

        Returns:
            DocumentTree实例
        """
        tree = cls()
        num_lines = len(line_labels)

        # 验证输入
        assert len(parent_indices) == num_lines, "parent_indices长度不匹配"
        assert len(relation_types) == num_lines, "relation_types长度不匹配"

        # 创建所有节点
        for i in range(num_lines):
            label_id = line_labels[i]
            label_name = LABEL_MAP.get(label_id, f"Unknown-{label_id}")

            bbox = line_bboxes[i] if line_bboxes is not None and i < len(line_bboxes) else None
            text = line_texts[i] if line_texts is not None and i < len(line_texts) else None
            label_conf = label_confidences[i] if label_confidences is not None and i < len(label_confidences) else None

            node = DocumentNode(
                idx=i,
                label=label_name,
                bbox=bbox,
                text=text,
                confidence=label_conf,
            )
            tree.nodes.append(node)

        # 构建树结构
        for i in range(num_lines):
            child_node = tree.nodes[i]
            parent_idx = parent_indices[i]
            relation_id = relation_types[i]
            relation_name = RELATION_MAP.get(relation_id, "none")
            relation_conf = relation_confidences[i] if relation_confidences is not None and i < len(relation_confidences) else None

            # 确定父节点
            if parent_idx < 0:
                # 挂在ROOT下
                tree.root.add_child(child_node, relation=relation_name, confidence=relation_conf)
            elif parent_idx < num_lines:
                # 挂在其他节点下
                parent_node = tree.nodes[parent_idx]
                parent_node.add_child(child_node, relation=relation_name, confidence=relation_conf)
            else:
                # 无效的父节点索引，挂在ROOT下
                tree.root.add_child(child_node, relation="none", confidence=relation_conf)

        return tree

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "root": self.root.to_dict(),
            "num_nodes": len(self.nodes),
            "statistics": self.get_statistics(),
        }

    def to_json(self, filepath: str, indent: int = 2):
        """保存为JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    def to_markdown(self, indent: str = "  ") -> str:
        """
        转换为Markdown格式的层次结构文本

        Returns:
            Markdown格式的字符串
        """
        lines = ["# Document Structure Tree\n"]

        def _traverse(node: DocumentNode, level: int):
            if node.idx >= 0:  # 跳过ROOT
                prefix = indent * level
                marker = "-" if level > 0 else "#"

                # 节点信息
                info = f"{prefix}{marker} [{node.label}] "
                if node.text:
                    info += f"`{node.text[:50]}...`" if len(node.text) > 50 else f"`{node.text}`"
                else:
                    info += f"Line {node.idx}"

                # 关系信息
                if node.relation_to_parent and node.relation_to_parent != "none":
                    info += f" ({node.relation_to_parent})"

                lines.append(info)

            # 递归遍历子节点
            for child in node.children:
                _traverse(child, level + 1)

        _traverse(self.root, -1)
        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """获取树的统计信息"""
        stats = {
            "total_nodes": len(self.nodes),
            "max_depth": 0,
            "label_distribution": defaultdict(int),
            "relation_distribution": defaultdict(int),
        }

        def _traverse(node: DocumentNode, depth: int):
            stats["max_depth"] = max(stats["max_depth"], depth)

            if node.idx >= 0:  # 跳过ROOT
                stats["label_distribution"][node.label] += 1
                if node.relation_to_parent:
                    stats["relation_distribution"][node.relation_to_parent] += 1

            for child in node.children:
                _traverse(child, depth + 1)

        _traverse(self.root, 0)

        # 转换为普通dict
        stats["label_distribution"] = dict(stats["label_distribution"])
        stats["relation_distribution"] = dict(stats["relation_distribution"])

        return stats

    def visualize_ascii(self, max_depth: int = 10) -> str:
        """
        ASCII艺术风格的树可视化

        Returns:
            ASCII树的字符串表示
        """
        lines = []

        def _traverse(node: DocumentNode, prefix: str = "", is_last: bool = True, depth: int = 0):
            if depth > max_depth:
                return

            # 绘制当前节点
            if node.idx >= 0:  # 跳过ROOT
                connector = "└── " if is_last else "├── "
                label_info = f"{node.label}"
                if node.text:
                    label_info += f": {node.text[:30]}..."

                lines.append(f"{prefix}{connector}{label_info}")

                # 更新prefix
                new_prefix = prefix + ("    " if is_last else "│   ")
            else:
                new_prefix = ""

            # 递归遍历子节点
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                _traverse(child, new_prefix, is_last_child, depth + 1)

        _traverse(self.root)
        return "\n".join(lines)

    def get_node_by_idx(self, idx: int) -> Optional[DocumentNode]:
        """根据索引获取节点"""
        if idx < 0 or idx >= len(self.nodes):
            return None
        return self.nodes[idx]

    def get_path_to_root(self, node: DocumentNode) -> List[DocumentNode]:
        """获取从节点到根的路径"""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]  # 反转，从根到节点

    def prune_none_relations(self):
        """
        剪枝：移除relation="none"的边（可选的后处理步骤）

        这些边可能是预测错误，或者不构成实际的层次关系
        """
        def _prune(node: DocumentNode):
            # 过滤掉relation为none的子节点
            valid_children = [
                child for child in node.children
                if child.relation_to_parent and child.relation_to_parent != "none"
            ]

            # 将被剪枝的子节点提升到当前节点的父节点
            pruned_children = [
                child for child in node.children
                if not child.relation_to_parent or child.relation_to_parent == "none"
            ]

            node.children = valid_children

            # 递归处理
            for child in node.children:
                _prune(child)

            # 处理被剪枝的节点
            if node.parent:
                for pruned in pruned_children:
                    node.parent.add_child(pruned, relation="pruned", confidence=0.0)

        _prune(self.root)


def demo():
    """演示如何使用DocumentTree"""

    # 模拟三个子任务的输出
    line_labels = [0, 1, 3, 3, 1, 3, 3]  # Title, Section, Para-Line, ...
    parent_indices = [-1, 0, 1, 2, 0, 4, 5]  # ROOT, Title, Section, Para-Line, ...
    relation_types = [0, 2, 2, 1, 2, 2, 1]  # none, contain, contain, connect, ...

    line_bboxes = [
        [100, 100, 500, 150],
        [100, 200, 500, 250],
        [100, 300, 500, 350],
        [100, 400, 500, 450],
        [100, 500, 500, 550],
        [100, 600, 500, 650],
        [100, 700, 500, 750],
    ]

    line_texts = [
        "Document Title",
        "Section 1",
        "This is a paragraph.",
        "This is another sentence.",
        "Section 2",
        "Another paragraph here.",
        "Final sentence.",
    ]

    # 构建树
    tree = DocumentTree.from_predictions(
        line_labels=line_labels,
        parent_indices=parent_indices,
        relation_types=relation_types,
        line_bboxes=line_bboxes,
        line_texts=line_texts,
    )

    # 输出
    print("=== ASCII Tree ===")
    print(tree.visualize_ascii())

    print("\n=== Markdown ===")
    print(tree.to_markdown())

    print("\n=== Statistics ===")
    print(json.dumps(tree.get_statistics(), indent=2))

    # 保存JSON
    tree.to_json("demo_tree.json")
    print("\n✓ Tree saved to demo_tree.json")


if __name__ == "__main__":
    demo()

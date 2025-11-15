#!/usr/bin/env python
# coding=utf-8
"""
使用示例：展示如何使用DocumentTree和推理Pipeline

包含以下示例：
1. 基本树构建
2. 从模型预测结果构建树
3. 树的遍历和查询
4. 树的可视化和导出
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_tree import DocumentTree, DocumentNode, LABEL_MAP, RELATION_MAP
import json


def example1_basic_tree():
    """示例1: 基本树构建"""
    print("="*80)
    print("示例1: 基本树构建")
    print("="*80)

    # 模拟文档：一个标题，两个章节，每个章节有段落
    line_labels = [
        0,  # Title
        1,  # Section 1
        2,  # Para-Title (段落标题)
        3,  # Para-Line
        3,  # Para-Line
        1,  # Section 2
        3,  # Para-Line
    ]

    parent_indices = [
        -1,  # Title -> ROOT
        0,   # Section 1 -> Title
        1,   # Para-Title -> Section 1
        2,   # Para-Line -> Para-Title
        3,   # Para-Line -> Para-Line (连接前一行)
        0,   # Section 2 -> Title
        5,   # Para-Line -> Section 2
    ]

    relation_types = [
        0,  # none (ROOT的子节点)
        2,  # contain (Title包含Section)
        2,  # contain (Section包含段落)
        2,  # contain (段落标题包含段落行)
        1,  # connect (段落行连接段落行)
        2,  # contain (Title包含Section)
        2,  # contain (Section包含段落)
    ]

    line_texts = [
        "Research Paper: HRDoc Implementation",
        "1. Introduction",
        "Background",
        "Document structure reconstruction is an important task.",
        "It involves multiple subtasks and complex relationships.",
        "2. Methodology",
        "We propose a three-stage approach for this problem.",
    ]

    # 构建树
    tree = DocumentTree.from_predictions(
        line_labels=line_labels,
        parent_indices=parent_indices,
        relation_types=relation_types,
        line_texts=line_texts,
    )

    # 输出
    print("\nASCII可视化:")
    print(tree.visualize_ascii())

    print("\n统计信息:")
    stats = tree.get_statistics()
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  最大深度: {stats['max_depth']}")
    print(f"  标签分布: {stats['label_distribution']}")
    print(f"  关系分布: {stats['relation_distribution']}")


def example2_tree_traversal():
    """示例2: 树的遍历和查询"""
    print("\n" + "="*80)
    print("示例2: 树的遍历和查询")
    print("="*80)

    # 构建一个简单的树
    line_labels = [0, 1, 3, 3, 1, 3]
    parent_indices = [-1, 0, 1, 2, 0, 4]
    relation_types = [0, 2, 2, 1, 2, 2]
    line_texts = ["Title", "Section 1", "Line 1", "Line 2", "Section 2", "Line 3"]

    tree = DocumentTree.from_predictions(
        line_labels=line_labels,
        parent_indices=parent_indices,
        relation_types=relation_types,
        line_texts=line_texts,
    )

    # 获取特定节点
    node_2 = tree.get_node_by_idx(2)
    print(f"\n节点2: {node_2}")
    print(f"  标签: {node_2.label}")
    print(f"  文本: {node_2.text}")
    print(f"  父节点: {node_2.parent.label if node_2.parent else 'None'}")
    print(f"  子节点数: {len(node_2.children)}")
    print(f"  与父节点关系: {node_2.relation_to_parent}")

    # 获取路径
    path = tree.get_path_to_root(node_2)
    print(f"\n从节点2到根的路径:")
    for i, node in enumerate(path):
        indent = "  " * i
        print(f"{indent}└─ {node.label}: {node.text}")

    # 遍历所有叶子节点
    print(f"\n所有叶子节点:")
    for node in tree.nodes:
        if len(node.children) == 0:
            print(f"  - {node.label}: {node.text}")


def example3_export_formats():
    """示例3: 树的导出格式"""
    print("\n" + "="*80)
    print("示例3: 树的导出格式")
    print("="*80)

    # 构建树
    line_labels = [0, 1, 3]
    parent_indices = [-1, 0, 1]
    relation_types = [0, 2, 2]
    line_texts = ["Document Title", "Section Header", "Paragraph text here."]

    tree = DocumentTree.from_predictions(
        line_labels=line_labels,
        parent_indices=parent_indices,
        relation_types=relation_types,
        line_texts=line_texts,
    )

    # 1. JSON格式
    print("\n1. JSON格式:")
    tree_dict = tree.to_dict()
    print(json.dumps(tree_dict, indent=2, ensure_ascii=False)[:500] + "...")

    # 2. Markdown格式
    print("\n2. Markdown格式:")
    markdown = tree.to_markdown()
    print(markdown)

    # 3. ASCII格式
    print("\n3. ASCII格式:")
    ascii_tree = tree.visualize_ascii()
    print(ascii_tree)

    # 保存到文件
    tree.to_json("example_tree.json")
    with open("example_tree.md", 'w', encoding='utf-8') as f:
        f.write(markdown)
    with open("example_tree_ascii.txt", 'w', encoding='utf-8') as f:
        f.write(ascii_tree)

    print("\n✓ 文件已保存:")
    print("  - example_tree.json")
    print("  - example_tree.md")
    print("  - example_tree_ascii.txt")


def example4_complex_document():
    """示例4: 复杂文档结构"""
    print("\n" + "="*80)
    print("示例4: 复杂文档结构（包含列表、表格）")
    print("="*80)

    # 模拟包含多种元素的文档
    line_labels = [
        0,   # 0: Title
        1,   # 1: Section
        3,   # 2: Para-Line
        4,   # 3: List-Title
        5,   # 4: List-Item
        5,   # 5: List-Item
        6,   # 6: Table-Title
        7,   # 7: Table-Column-Header
        10,  # 8: Figure-Title
        11,  # 9: Figure-Caption
    ]

    parent_indices = [
        -1,  # 0: Title -> ROOT
        0,   # 1: Section -> Title
        1,   # 2: Para -> Section
        1,   # 3: List-Title -> Section
        3,   # 4: List-Item -> List-Title
        3,   # 5: List-Item -> List-Title
        1,   # 6: Table-Title -> Section
        6,   # 7: Table-Header -> Table-Title
        1,   # 8: Figure-Title -> Section
        8,   # 9: Figure-Caption -> Figure-Title
    ]

    relation_types = [
        0,  # none
        2,  # contain
        2,  # contain
        2,  # contain
        2,  # contain
        3,  # equality (并列的list item)
        2,  # contain
        2,  # contain
        2,  # contain
        2,  # contain
    ]

    line_texts = [
        "Machine Learning Survey",
        "1. Deep Learning Methods",
        "Recent advances in neural networks have shown promising results.",
        "Key approaches include:",
        "- Convolutional Neural Networks (CNNs)",
        "- Recurrent Neural Networks (RNNs)",
        "Table 1: Performance Comparison",
        "Model | Accuracy | Speed",
        "Figure 1: Network Architecture",
        "The proposed architecture consists of three main components.",
    ]

    tree = DocumentTree.from_predictions(
        line_labels=line_labels,
        parent_indices=parent_indices,
        relation_types=relation_types,
        line_texts=line_texts,
    )

    print("\n完整树结构:")
    print(tree.visualize_ascii(max_depth=10))

    print("\n统计信息:")
    stats = tree.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def example5_post_processing():
    """示例5: 后处理（剪枝）"""
    print("\n" + "="*80)
    print("示例5: 后处理 - 剪枝none关系")
    print("="*80)

    # 构建包含一些错误预测的树
    line_labels = [0, 1, 3, 3, 3]
    parent_indices = [-1, 0, 1, 1, 3]  # 节点4错误地连接到节点3
    relation_types = [0, 2, 2, 0, 1]   # 节点3的关系被预测为none

    tree = DocumentTree.from_predictions(
        line_labels=line_labels,
        parent_indices=parent_indices,
        relation_types=relation_types,
        line_texts=["Title", "Section", "Para 1", "Para 2", "Para 3"],
    )

    print("\n剪枝前:")
    print(tree.visualize_ascii())
    print(f"关系分布: {tree.get_statistics()['relation_distribution']}")

    # 执行剪枝
    tree.prune_none_relations()

    print("\n剪枝后:")
    print(tree.visualize_ascii())
    print(f"关系分布: {tree.get_statistics()['relation_distribution']}")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("DocumentTree 使用示例集合")
    print("="*80)

    example1_basic_tree()
    example2_tree_traversal()
    example3_export_formats()
    example4_complex_document()
    example5_post_processing()

    print("\n" + "="*80)
    print("所有示例运行完成！")
    print("="*80)


if __name__ == "__main__":
    main()

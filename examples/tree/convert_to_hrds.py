#!/usr/bin/env python
# coding=utf-8
"""
将DocumentTree转换为HRDS平铺格式
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_tree import DocumentTree


def tree_to_hrds_format(tree, page_num=0):
    """
    将DocumentTree转换为HRDS平铺格式

    Args:
        tree: DocumentTree实例
        page_num: 页码

    Returns:
        list: HRDS格式的平铺列表
    """
    flat_list = []

    def traverse(node, parent_id=-1):
        if node.idx >= 0:  # 跳过ROOT
            # 转换为HRDS格式
            hrds_item = {
                "line_id": node.idx,
                "text": node.text if node.text else "",
                "box": node.bbox,
                "class": node.label.lower().replace("-", "_"),  # Para-Line -> para_line
                "page": page_num,
                "parent_id": parent_id,
                "relation": node.relation_to_parent if node.relation_to_parent else "none",
            }

            # 添加is_meta标记（根据relation判断）
            hrds_item["is_meta"] = (node.relation_to_parent == "meta")

            flat_list.append(hrds_item)

        # 递归遍历子节点
        for child in node.children:
            traverse(child, node.idx)

    traverse(tree.root)
    return flat_list


def convert_tree_file(tree_json_path, output_path=None):
    """
    转换单个树文件为HRDS格式

    Args:
        tree_json_path: 树JSON文件路径
        output_path: 输出路径（默认为同名_hrds.json）
    """
    # 读取树
    with open(tree_json_path, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)

    # 重建DocumentTree（从dict）
    # 这里简化处理，直接从JSON重建
    from document_tree import DocumentNode

    def rebuild_tree(node_dict):
        node = DocumentNode(
            idx=node_dict["idx"],
            label=node_dict["label"],
            bbox=node_dict["bbox"],
            text=node_dict["text"],
            confidence=node_dict.get("confidence"),
        )
        node.relation_to_parent = node_dict.get("relation_to_parent")

        for child_dict in node_dict.get("children", []):
            child = rebuild_tree(child_dict)
            child.parent = node
            node.children.append(child)

        return node

    tree = DocumentTree()
    tree.root = rebuild_tree(tree_data["root"])

    # 转换为HRDS格式
    hrds_data = tree_to_hrds_format(tree)

    # 保存
    if output_path is None:
        output_path = tree_json_path.replace('.json', '_hrds.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hrds_data, f, indent=2, ensure_ascii=False)

    print(f"✓ 转换完成: {output_path}")
    print(f"  节点数: {len(hrds_data)}")

    return hrds_data


def demo():
    """演示转换"""
    from document_tree import DocumentTree

    # 创建示例树
    tree = DocumentTree.from_predictions(
        line_labels=[0, 1, 3, 3],
        parent_indices=[-1, 0, 1, 2],
        relation_types=[0, 2, 2, 1],
        line_bboxes=[[100, 100, 500, 150], [100, 200, 500, 250],
                     [100, 300, 500, 350], [100, 400, 500, 450]],
        line_texts=["Document Title", "Section 1", "First paragraph", "Second paragraph"],
    )

    # 转换为HRDS格式
    hrds_data = tree_to_hrds_format(tree, page_num=0)

    # 打印
    print("HRDS平铺格式:")
    print(json.dumps(hrds_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="转换DocumentTree为HRDS格式")
    parser.add_argument("--input", type=str, help="输入树JSON文件")
    parser.add_argument("--output", type=str, help="输出HRDS JSON文件")
    parser.add_argument("--demo", action="store_true", help="运行演示")

    args = parser.parse_args()

    if args.demo or (not args.input):
        demo()
    else:
        convert_tree_file(args.input, args.output)

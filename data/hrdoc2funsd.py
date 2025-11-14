#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 HRDoc 格式转换为 FUNSD 格式，用于 layoutlmft 训练
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# HRDoc 的类别映射字典（来自 utils/utils.py）
class2class = {
    "title": "title",
    "author": "author",
    "mail": "mail",
    "affili": "affili",
    "sec1": "section",
    "sec2": "section",
    "sec3": "section",
    "fstline": "fstline",
    "para": "paraline",
    "tab": "table",
    "fig": "figure",
    "tabcap": "caption",
    "figcap": "caption",
    "equ": "equation",
    "foot": "footer",
    "header": "header",
    "fnote": "footnote",
}

def trans_class(all_pg_lines, unit):
    """
    HRDoc 类别映射函数
    如果 class 不是 'opara'，直接查表映射
    如果是 'opara'，递归查找父节点的类别
    """
    if unit["class"] != "opara":
        return class2class.get(unit["class"], unit["class"])
    else:
        parent_id = unit.get('parent_id', -1)
        if parent_id == -1:
            return class2class.get(unit["class"], unit["class"])
        parent_cl = all_pg_lines[parent_id]
        while parent_cl.get("class") == 'opara':
            parent_id = parent_cl.get('parent_id', -1)
            if parent_id == -1:
                break
            parent_cl = all_pg_lines[parent_id]
        return class2class.get(parent_cl.get("class", "para"), "paraline")


def convert_hrdoc_to_funsd(hrdoc_dir, output_dir, split_name="train"):
    """
    将 HRDoc 格式转换为 FUNSD 格式

    Args:
        hrdoc_dir: HRDoc 数据目录（包含 .json 和对应的图片文件夹）
        output_dir: 输出目录
        split_name: train/val/test
    """
    hrdoc_path = Path(hrdoc_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    img_out = output_path / split_name / "images"
    ann_out = output_path / split_name / "annotations"
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    # 查找所有 JSON 标注文件
    json_files = sorted(hrdoc_path.glob("*.json"))

    print(f"找到 {len(json_files)} 个标注文件")

    for json_file in json_files:
        paper_name = json_file.stem  # 例如 ACL_2020.acl-main.1
        print(f"\n处理: {paper_name}")

        # 读取标注
        with open(json_file, 'r', encoding='utf-8') as f:
            all_lines = json.load(f)

        print(f"  总行数: {len(all_lines)}")

        # 按页分组
        pages = defaultdict(list)
        for line in all_lines:
            page_id = line['page']
            pages[page_id].append(line)

        print(f"  页数: {len(pages)}")

        # 处理每一页
        for page_id in sorted(pages.keys()):
            page_lines = pages[page_id]
            page_name = f"{paper_name}_{page_id}"

            # 构建 FUNSD 格式的实体列表
            entities = []
            for idx, line in enumerate(page_lines):
                text = line.get('text', '').strip()
                if not text:  # 跳过空文本
                    continue

                box = line.get('box', [0, 0, 0, 0])
                raw_class = line.get('class', 'para')

                # 使用 trans_class 映射类别
                # trans_class 需要所有行和当前单元作为参数
                try:
                    label = trans_class(page_lines, line)
                except Exception as e:
                    # 如果 trans_class 失败，直接使用 class2class 映射
                    label = class2class.get(raw_class, raw_class)

                # 简化处理：把整行文本作为一个 word
                # 如果需要更细粒度，可以分词后为每个词分配 bbox
                words = [{"text": text, "box": box}]

                # 保留层级关系信息
                line_id = line.get('line_id', idx)
                parent_id = line.get('parent_id', -1)
                relation = line.get('relation', 'none')

                entity = {
                    "id": line_id,  # 使用原始 line_id
                    "text": text,
                    "box": box,
                    "label": label,
                    "words": words,
                    "linking": [],  # FUNSD 需要这个字段，但我们暂时不用
                    "parent_id": parent_id,  # 新增：父节点ID
                    "relation": relation     # 新增：关系类型
                }
                entities.append(entity)

            # 写入 FUNSD 格式的 JSON
            funsd_data = {"form": entities}
            out_json_path = ann_out / f"{page_name}.json"
            with open(out_json_path, 'w', encoding='utf-8') as f:
                json.dump(funsd_data, f, ensure_ascii=False, indent=2)

            # 复制对应的图片
            img_dir = hrdoc_path / paper_name
            if img_dir.exists():
                for ext in ['.jpg', '.png', '.jpeg']:
                    src_img = img_dir / f"{page_name}{ext}"
                    if src_img.exists():
                        dst_img = img_out / f"{page_name}{ext}"
                        shutil.copy(src_img, dst_img)
                        break
                else:
                    print(f"  ⚠️  未找到图片: {page_name}")

            print(f"  ✓ 页 {page_id}: {len(entities)} 个实体")


def collect_labels(hrdoc_dir):
    """
    收集所有标签，生成 labels.txt
    """
    hrdoc_path = Path(hrdoc_dir)
    labels = set()

    for json_file in hrdoc_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            all_lines = json.load(f)

        for line in all_lines:
            raw_class = line.get('class', 'para')
            # 使用映射后的类别
            mapped_class = class2class.get(raw_class, raw_class)
            labels.add(mapped_class)

    return sorted(labels)


if __name__ == "__main__":
    # 获取脚本所在目录，计算默认输出路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # data -> project_root
    default_output = project_root / "data" / "hrdoc_funsd_format"

    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="将 HRDoc 格式转换为 FUNSD 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数
  python hrdoc2funsd.py

  # 指定输入输出路径
  python hrdoc2funsd.py --input /path/to/HRDS --output /path/to/output

  # 只指定输入路径（使用默认输出）
  python hrdoc2funsd.py --input /path/to/HRDS

  # 指定数据集划分
  python hrdoc2funsd.py --input /path/to/HRDS --split train
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/mnt/e/programFile/AIProgram/modelTrain/HRDoc/examples/HRDS",
        help="HRDoc 原始数据目录 (默认: /mnt/e/programFile/AIProgram/modelTrain/HRDoc/examples/HRDS)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(default_output),
        help=f"输出目录 (默认: {default_output})"
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="数据集划分名称 (默认: train)"
    )

    args = parser.parse_args()

    HRDOC_EXAMPLES = args.input
    OUTPUT_DIR = args.output
    SPLIT_NAME = args.split

    print("=" * 60)
    print("HRDoc → FUNSD 格式转换")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  输入目录: {HRDOC_EXAMPLES}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  数据划分: {SPLIT_NAME}")

    # 检查输入目录
    if not os.path.exists(HRDOC_EXAMPLES):
        print(f"\n❌ 错误: 输入目录不存在: {HRDOC_EXAMPLES}")
        exit(1)

    # 转换数据
    print(f"\n[1/2] 转换 {SPLIT_NAME} 数据...")
    convert_hrdoc_to_funsd(HRDOC_EXAMPLES, OUTPUT_DIR, split_name=SPLIT_NAME)

    # 收集所有标签
    print("\n[2/2] 收集标签...")
    labels = collect_labels(HRDOC_EXAMPLES)
    print(f"找到 {len(labels)} 个标签:")
    for label in labels:
        print(f"  - {label}")

    # 保存标签文件
    labels_file = Path(OUTPUT_DIR) / "labels.txt"
    with open(labels_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + '\n')

    print(f"\n✓ 标签已保存到: {labels_file}")
    print(f"\n✓ 转换完成！输出目录: {OUTPUT_DIR}")
    print("\n目录结构:")
    print(f"  {SPLIT_NAME}/")
    print("    ├── images/       (页面图片)")
    print("    └── annotations/  (FUNSD 格式标注)")
    print("  labels.txt          (所有类别)")

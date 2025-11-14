#!/usr/bin/env python
# coding=utf-8
"""快速转换HRDS测试集到FUNSD格式"""

import json
import os
import shutil
from collections import defaultdict

# HRDoc 类别映射
class2class = {
    "title": "title",
    "author": "author",
    "mail": "mail",
    "affili": "affili",
    "sec1": "section",
    "sec2": "section",
    "sec3": "section",
    "secx": "section",  # 额外的 section 类型
    "fstline": "fstline",
    "para": "paraline",
    "tab": "table",
    "fig": "figure",
    "tabcap": "caption",
    "figcap": "caption",
    "equ": "equation",
    "alg": "opara",  # 算法 -> 其他段落
    "foot": "footer",
    "header": "header",
    "fnote": "footnote",
}

def trans_class(line_dict, unit):
    """处理 opara 类别"""
    if unit["class"] != "opara":
        return class2class.get(unit["class"], unit["class"])
    else:
        parent_id = unit.get('parent_id', -1)
        if parent_id == -1 or parent_id not in line_dict:
            return "opara"
        parent_cl = line_dict[parent_id]
        while parent_cl.get("class") == 'opara':
            parent_id = parent_cl.get('parent_id', -1)
            if parent_id == -1 or parent_id not in line_dict:
                return "opara"
            parent_cl = line_dict[parent_id]
        return class2class.get(parent_cl.get("class", "para"), "paraline")

# 路径配置
hrds_root = "/mnt/e/models/data/Section/HRDS"
test_json_dir = f"{hrds_root}/test"
images_dir = f"{hrds_root}/images"
output_dir = "/root/code/layoutlmft/data/hrdoc_test"

# 创建输出目录
os.makedirs(f"{output_dir}/annotations", exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

print(f"开始转换测试集...")
processed = 0

for json_file in os.listdir(test_json_dir):
    if not json_file.endswith('.json'):
        continue

    paper_name = json_file.replace('.json', '')
    json_path = os.path.join(test_json_dir, json_file)

    # 读取JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        hrdoc_data = json.load(f)

    # 按页分组
    pages = defaultdict(list)
    for line in hrdoc_data:
        pages[line['page']].append(line)

    # 转换每一页
    for page_num, page_lines in pages.items():
        # 复制图片
        img_name = f"{paper_name}_{page_num}.jpg"
        src_img = os.path.join(images_dir, paper_name, img_name)
        if not os.path.exists(src_img):
            print(f"  警告: 图片不存在 {src_img}")
            continue

        dst_img = os.path.join(output_dir, "images", img_name)
        shutil.copy2(src_img, dst_img)

        # 创建 line_id 到 line 的映射
        line_dict = {line["line_id"]: line for line in page_lines}

        # 转换为FUNSD格式
        funsd_data = {"form": []}

        for line in page_lines:
            mapped_class = trans_class(line_dict, line)

            funsd_item = {
                "text": line["text"],
                "box": line["box"],
                "label": mapped_class,
                "words": [{
                    "text": line["text"],
                    "box": line["box"]
                }],
                "linking": [],
                "id": line["line_id"],
                "parent_id": line.get("parent_id", -1),  # 新增：父节点ID
                "relation": line.get("relation", "none")  # 新增：关系类型
            }
            funsd_data["form"].append(funsd_item)

        # 保存JSON
        output_json = os.path.join(output_dir, "annotations", f"{img_name.replace('.jpg', '.json')}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(funsd_data, f, ensure_ascii=False, indent=2)

        processed += 1

print(f"✅ 转换完成！处理了 {processed} 页")

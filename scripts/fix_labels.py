#!/usr/bin/env python
# coding=utf-8
"""
修复类别标签：将自定义标签映射到模型期望的标签
"""
import json
import os
import sys
from pathlib import Path

# 标签映射
LABEL_MAPPING = {
    "ContentLine": "PARA",      # 内容行 → 段落
    "Equation": "EQU",          # 公式
    "FirstLine": "FSTLINE",     # 首行
}

def fix_json_labels(input_dir, output_dir=None):
    """修复JSON文件中的标签

    Args:
        input_dir: 输入目录（包含test/和images/）
        output_dir: 输出目录（默认为input_dir，会覆盖原文件）
    """
    if output_dir is None:
        output_dir = input_dir

    input_test_dir = Path(input_dir) / "test"
    output_test_dir = Path(output_dir) / "test"
    output_test_dir.mkdir(parents=True, exist_ok=True)

    # 如果有images目录，也复制过去
    input_images_dir = Path(input_dir) / "images"
    output_images_dir = Path(output_dir) / "images"
    if input_images_dir.exists() and input_dir != output_dir:
        import shutil
        if output_images_dir.exists():
            shutil.rmtree(output_images_dir)
        shutil.copytree(input_images_dir, output_images_dir)
        print(f"✓ 复制图片目录: {output_images_dir}")

    # 处理所有JSON文件
    json_files = list(input_test_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")

    total_lines = 0
    total_changed = 0

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        changed = 0
        for item in data:
            old_class = item.get("class", "O")
            if old_class in LABEL_MAPPING:
                item["class"] = LABEL_MAPPING[old_class]
                changed += 1
            total_lines += 1

        # 保存修复后的文件
        output_file = output_test_dir / json_file.name
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        total_changed += changed
        print(f"  {json_file.name}: {changed}/{len(data)} 行标签已修改")

    print(f"\n总计: {total_changed}/{total_lines} 行标签已修改")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python fix_labels.py <input_dir> [output_dir]")
        print("示例: python fix_labels.py /mnt/e/programFile/AIProgram/modelTrain/HRDoc/output")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    fix_json_labels(input_dir, output_dir)

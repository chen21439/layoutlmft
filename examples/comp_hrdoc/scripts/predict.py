#!/usr/bin/env python
"""推理入口脚本"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def parse_args():
    parser = argparse.ArgumentParser(description="Detect-Order-Construct 推理")

    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--input", type=str, required=True, help="输入文件 (PDF/图像)")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--format", type=str, default="json", help="输出格式: json, html")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[comp_hrdoc] 推理脚本启动")
    print(f"  模型: {args.model_path}")
    print(f"  输入: {args.input}")

    # TODO: 实现推理逻辑
    print("[comp_hrdoc] 推理脚本待实现")


if __name__ == "__main__":
    main()

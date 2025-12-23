#!/usr/bin/env python
"""评估入口脚本"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def parse_args():
    parser = argparse.ArgumentParser(description="Detect-Order-Construct 评估")

    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--dataset", type=str, default="hrds", help="数据集: hrds, hrdh")
    parser.add_argument("--split", type=str, default="test", help="数据集划分: dev, test")
    parser.add_argument("--output", type=str, help="评估结果输出路径")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[comp_hrdoc] 评估脚本启动")
    print(f"  模型: {args.model_path}")
    print(f"  数据集: {args.dataset}/{args.split}")

    # TODO: 实现评估逻辑
    print("[comp_hrdoc] 评估脚本待实现")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Detect-Order-Construct 评估入口脚本

用法:
    # 从文件夹评估
    python evaluate.py --gt_folder <gt_path> --pred_folder <pred_path>

    # 指定输出目录
    python evaluate.py --gt_folder <gt> --pred_folder <pred> --output_dir ./results

    # 并行评估
    python evaluate.py --gt_folder <gt> --pred_folder <pred> --num_workers 8
"""

import argparse
import os
import sys
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.comp_hrdoc.engines.evaluator import (
    DOCEvaluator,
    EvaluatorConfig,
    evaluate_doc,
)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect-Order-Construct 模型评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 评估阅读顺序预测
    python evaluate.py \\
        --gt_folder datasets/Comp-HRDoc/test_eval \\
        --pred_folder outputs/predictions

    # 仅评估分类
    python evaluate.py \\
        --gt_folder datasets/Comp-HRDoc/test_eval \\
        --pred_folder outputs/predictions \\
        --eval_tasks classification

    # 并行评估
    python evaluate.py \\
        --gt_folder datasets/Comp-HRDoc/test_eval \\
        --pred_folder outputs/predictions \\
        --num_workers 8
        """
    )

    # 数据路径
    parser.add_argument(
        "--gt_folder",
        type=str,
        required=True,
        help="真实标签文件夹路径 (包含 JSON 文件)"
    )
    parser.add_argument(
        "--pred_folder",
        type=str,
        required=True,
        help="预测结果文件夹路径 (包含 JSON 文件)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_output",
        help="评估结果输出目录 (默认: ./eval_output)"
    )

    # 评估选项
    parser.add_argument(
        "--eval_tasks",
        type=str,
        nargs="+",
        default=["classification", "reading_order", "structure"],
        choices=["classification", "reading_order", "structure", "all"],
        help="要评估的任务 (默认: 全部)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行工作进程数 (默认: 4)"
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="禁用并行评估"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Detect-Order-Construct 评估")
    print("=" * 60)
    print(f"GT 文件夹: {args.gt_folder}")
    print(f"预测文件夹: {args.pred_folder}")
    print(f"输出目录: {args.output_dir}")
    print(f"评估任务: {args.eval_tasks}")
    print(f"并行进程: {args.num_workers}")
    print("=" * 60)

    # 检查路径
    if not os.path.exists(args.gt_folder):
        logger.error(f"GT 文件夹不存在: {args.gt_folder}")
        sys.exit(1)
    if not os.path.exists(args.pred_folder):
        logger.error(f"预测文件夹不存在: {args.pred_folder}")
        sys.exit(1)

    # 配置评估选项
    eval_tasks = args.eval_tasks
    if "all" in eval_tasks:
        eval_tasks = ["classification", "reading_order", "structure"]

    config = EvaluatorConfig(
        gt_folder=args.gt_folder,
        pred_folder=args.pred_folder,
        output_dir=args.output_dir,
        eval_classification="classification" in eval_tasks,
        eval_reading_order="reading_order" in eval_tasks,
        eval_structure="structure" in eval_tasks,
        num_workers=args.num_workers,
    )

    # 运行评估
    evaluator = DOCEvaluator(config)

    if args.no_parallel or args.num_workers <= 1:
        report = evaluator.evaluate_from_folders(
            args.gt_folder,
            args.pred_folder,
            args.output_dir,
        )
    else:
        report = evaluator.evaluate_from_folders_parallel(
            args.gt_folder,
            args.pred_folder,
            args.output_dir,
            args.num_workers,
        )

    print("\n评估完成!")
    print(f"详细结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

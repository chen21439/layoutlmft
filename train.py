#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 训练入口脚本（支持多环境）

用法:
    python train.py --env auto          # 自动检测环境
    python train.py --env local         # 本机测试
    python train.py --env cloud         # 云服务器完整训练
    python train.py --env quick         # 快速测试
    python train.py --config my.json    # 自定义配置文件
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from configs.env_config import get_config, EnvironmentDetector


def main():
    parser = argparse.ArgumentParser(description="HRDoc 版面识别训练（多环境支持）")
    parser.add_argument(
        "--env",
        type=str,
        default="auto",
        choices=["auto", "local", "cloud", "quick"],
        help="训练环境 (default: auto)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="自定义配置文件路径 (JSON)"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="仅显示配置而不训练"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/layoutlmv2-base-uncased",
        help="预训练模型名称"
    )

    args = parser.parse_args()

    # 打印环境信息
    print("=" * 60)
    print("HRDoc 版面识别训练")
    print("=" * 60)
    EnvironmentDetector.print_environment_info()

    # 加载配置
    if args.config:
        print(f"\n使用自定义配置: {args.config}")
        from configs.env_config import TrainingConfig
        config = TrainingConfig.from_json(args.config)
    else:
        config = get_config(args.env)

    # 显示配置
    print("\n训练配置:")
    print("-" * 60)
    print(f"  环境: {args.env}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  训练步数: {config.max_steps}")
    print(f"  Batch Size: {config.per_device_train_batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  Warmup比例: {config.warmup_ratio}")
    print(f"  FP16: {config.fp16}")
    print("-" * 60)

    # 预估训练时长
    if config.max_steps >= 30000:
        print("\n⚠️  完整训练预计耗时: 4-6 小时 (V100/A100)")
    elif config.max_steps >= 1000:
        print(f"\n⏱️  预计训练时长: ~{config.max_steps // 60} 分钟")
    else:
        print(f"\n⏱️  快速测试模式: ~{config.max_steps // 10} 分钟")

    if args.show_config:
        print("\n配置详情:")
        print(config.to_dict())
        return

    # 确认开始训练
    if config.max_steps >= 10000:
        response = input("\n是否开始训练？(y/n): ")
        if response.lower() != 'y':
            print("已取消训练")
            return

    # 构建训练命令
    train_script = Path(__file__).parent / "examples" / "run_hrdoc.py"

    cmd = [
        sys.executable,
        str(train_script),
        f"--model_name_or_path={config.model_name_or_path}",
        f"--output_dir={config.output_dir}",
        "--do_train",
        "--do_eval",
        f"--max_steps={config.max_steps}",
        f"--per_device_train_batch_size={config.per_device_train_batch_size}",
        f"--per_device_eval_batch_size={config.per_device_eval_batch_size}",
        f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
        f"--learning_rate={config.learning_rate}",
        f"--warmup_ratio={config.warmup_ratio}",
        f"--weight_decay={config.weight_decay}",
        f"--logging_steps={config.logging_steps}",
        f"--eval_steps={config.eval_steps}",
        f"--save_steps={config.save_steps}",
        f"--save_total_limit={config.save_total_limit}",
        f"--evaluation_strategy={config.evaluation_strategy}",
        f"--seed={config.seed}",
        "--overwrite_output_dir",
        "--task_name=ner",
        "--return_entity_level_metrics",
    ]

    if config.fp16:
        cmd.append("--fp16")

    # 运行训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    print(f"命令: {' '.join(cmd[:3])} ...\n")

    try:
        result = subprocess.run(cmd, check=True)

        print("\n" + "=" * 60)
        print("✓ 训练完成！")
        print("=" * 60)
        print(f"模型保存在: {config.output_dir}")
        print("\n下一步:")
        print("  1. 提取行级特征: python examples/extract_line_features.py")
        print("  2. 训练关系分类:")
        print("     - 二分类: python examples/train_relation_classifier.py")
        print("     - 多分类: python examples/train_multiclass_relation.py")

    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("✗ 训练失败")
        print("=" * 60)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n训练已中断")
        sys.exit(0)


if __name__ == "__main__":
    main()

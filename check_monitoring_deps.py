#!/usr/bin/env python
# coding=utf-8
"""
检查监控依赖是否已安装
"""

import sys
import subprocess


def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安装")
        return False


def main():
    print("=" * 60)
    print("检查监控依赖")
    print("=" * 60)

    deps = {
        # 核心依赖（必需）
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",

        # 监控依赖（推荐）
        "tensorboard": "tensorboard",
        "psutil": "psutil",
        "GPUtil": "GPUtil",

        # 可选依赖
        "wandb": "wandb",
    }

    print("\n核心依赖:")
    core_ok = all([
        check_package("torch"),
        check_package("transformers"),
        check_package("datasets"),
    ])

    print("\n监控依赖（推荐）:")
    monitoring_ok = all([
        check_package("tensorboard"),
        check_package("psutil"),
        check_package("GPUtil"),
    ])

    print("\n可选依赖:")
    check_package("wandb")

    print("\n" + "=" * 60)

    if not core_ok:
        print("❌ 缺少核心依赖，请先安装:")
        print("   pip install torch transformers datasets")
        return False

    if not monitoring_ok:
        print("⚠️  缺少监控依赖，建议安装:")
        print("   pip install tensorboard psutil gputil")
        print("\n没有这些依赖，监控功能将受限")

    # 检查CUDA
    print("\nCUDA检查:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练（很慢）")
    except:
        print("❌ 无法检查CUDA状态")

    # 检查HuggingFace Trainer配置
    print("\nHuggingFace Trainer配置:")
    try:
        from transformers import TrainingArguments

        # 创建一个测试配置
        args = TrainingArguments(
            output_dir="./test",
            logging_dir="./test/runs",
            logging_steps=10,
        )

        print(f"✅ Trainer已配置:")
        print(f"   - TensorBoard日志: {args.logging_dir}")
        print(f"   - 日志频率: 每{args.logging_steps}步")
        print(f"   - 报告工具: {args.report_to}")

    except Exception as e:
        print(f"❌ Trainer配置检查失败: {e}")

    print("\n" + "=" * 60)
    print("总结:")

    if core_ok and monitoring_ok:
        print("✅ 所有依赖已就绪，可以开始训练")
        return True
    elif core_ok:
        print("⚠️  可以训练，但建议安装监控依赖")
        return True
    else:
        print("❌ 缺少必要依赖，请先安装")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

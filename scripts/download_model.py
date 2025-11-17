#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 LayoutLMv2 预训练模型
"""

import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModel

def download_model(model_name, save_dir):
    """
    下载模型到指定目录

    Args:
        model_name: 模型名称，如 'microsoft/layoutlmv2-base-uncased'
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"开始下载模型: {model_name}")
    print("=" * 60)
    print(f"保存路径: {save_path.absolute()}")
    print()

    try:
        # 下载 tokenizer
        print("[1/3] 下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("✓ Tokenizer 下载完成")
        print()

        # 下载 config
        print("[2/3] 下载 Config...")
        config = AutoConfig.from_pretrained(model_name)
        config.save_pretrained(save_path)
        print("✓ Config 下载完成")
        print()

        # 下载 model
        print("[3/3] 下载 Model (约 766MB, 可能需要几分钟)...")
        # 使用 AutoModel 而不是特定的模型类，避免依赖问题
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        model.save_pretrained(save_path)
        print("✓ Model 下载完成")
        print()

        # 验证文件
        print("=" * 60)
        print("验证下载的文件:")
        print("=" * 60)
        for file in save_path.iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                print(f"  ✓ {file.name:30s} ({size_mb:>8.2f} MB)")

        print()
        print("=" * 60)
        print("✓ 模型下载成功！")
        print("=" * 60)
        print(f"模型位置: {save_path.absolute()}")
        print()
        print("下一步:")
        print("  1. 打包模型: tar -czf layoutlmv2-base.tar.gz -C <parent_dir> <model_dir_name>")
        print("  2. 上传到云服务器")

        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("✗ 下载失败！")
        print("=" * 60)
        print(f"错误信息: {e}")
        print()
        print("可能的解决方案:")
        print("  1. 检查网络连接")
        print("  2. 使用代理: export HF_ENDPOINT=https://hf-mirror.com")
        print("  3. 手动从 HuggingFace 镜像站下载")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 LayoutLMv2 预训练模型")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="microsoft/layoutlmv2-base-uncased",
        help="模型名称 (默认: microsoft/layoutlmv2-base-uncased)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./models/layoutlmv2-base-uncased",
        help="保存目录 (默认: ./models/layoutlmv2-base-uncased)"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用 HuggingFace 镜像站 (适用于国内网络)"
    )

    args = parser.parse_args()

    # 设置镜像站（如果需要）
    if args.mirror:
        print("使用 HuggingFace 镜像站...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT')}")
        print()

    # 下载模型
    success = download_model(args.model, args.output)

    exit(0 if success else 1)

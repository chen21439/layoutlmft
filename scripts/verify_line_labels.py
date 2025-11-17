#!/usr/bin/env python
# coding=utf-8
"""验证新提取的特征是否包含 line_labels"""

import pickle
import sys
import os

def verify_features(feature_file):
    """验证特征文件是否包含 line_labels"""
    print(f"检查文件: {feature_file}")

    if not os.path.exists(feature_file):
        print(f"❌ 文件不存在: {feature_file}")
        return False

    try:
        with open(feature_file, "rb") as f:
            data = pickle.load(f)

        print(f"✓ 成功加载，包含 {len(data)} 个页面")

        # 检查第一个页面
        if len(data) > 0:
            sample = data[0]
            print(f"\n第一个页面的键:")
            for key in sample.keys():
                print(f"  - {key}")

            # 检查是否有 line_labels
            if "line_labels" in sample:
                print(f"\n✅ 包含 line_labels!")
                print(f"   line_labels 类型: {type(sample['line_labels'])}")
                print(f"   line_labels 长度: {len(sample['line_labels'])}")
                print(f"   line_labels 示例: {sample['line_labels'][:10]}")

                # 统计标签分布
                from collections import Counter
                label_counts = Counter(sample['line_labels'])
                print(f"\n   标签分布:")
                for label, count in label_counts.most_common():
                    print(f"     标签 {label}: {count} 行")

                return True
            else:
                print(f"\n❌ 不包含 line_labels")
                return False
        else:
            print("⚠️  数据为空")
            return False

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


if __name__ == "__main__":
    import glob

    # 检查特征目录
    features_dir = os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features")

    print("=" * 60)
    print("验证 line_labels 是否正确生成")
    print("=" * 60)

    # 查找所有 chunk 文件
    pattern = os.path.join(features_dir, "*_line_features_chunk_*.pkl")
    chunk_files = sorted(glob.glob(pattern))

    if len(chunk_files) == 0:
        print(f"❌ 未找到特征文件: {pattern}")
        sys.exit(1)

    print(f"\n找到 {len(chunk_files)} 个特征文件")

    # 检查第一个文件
    print(f"\n检查第一个文件...")
    success = verify_features(chunk_files[0])

    if success:
        print("\n" + "=" * 60)
        print("✅ 验证成功！line_labels 已正确生成")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 验证失败！需要重新提取特征")
        print("=" * 60)
        sys.exit(1)

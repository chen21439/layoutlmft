#!/bin/bash

# ========================================
# 阶段一：完整特征提取（文档级别）
# ========================================

set -e  # 遇到错误立即退出

echo "======================================================"
echo "阶段一：文档级别特征提取（完整数据集）"
echo "======================================================"
echo ""

# 激活conda环境
source /root/miniforge3/etc/profile.d/conda.sh
conda activate layoutlmv2

# 环境变量配置
export HRDOC_DATA_DIR="/mnt/e/models/data/Section/HRDS"
export LAYOUTLMFT_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000"
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc"
export LAYOUTLMFT_DOCS_PER_CHUNK="10"  # 每10个文档保存一个chunk

# 不设置LAYOUTLMFT_NUM_SAMPLES，使用全部数据
# 完整数据集：train ~450个文档(900页), test ~50个文档(100页)

echo "配置信息："
echo "  数据集路径: $HRDOC_DATA_DIR"
echo "  模型路径: $LAYOUTLMFT_MODEL_PATH"
echo "  输出目录: $LAYOUTLMFT_FEATURES_DIR"
echo "  每chunk文档数: $LAYOUTLMFT_DOCS_PER_CHUNK"
echo ""
echo "预计处理时间: ~30-60分钟（取决于GPU性能）"
echo ""

# 确认继续
read -p "是否继续？[y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 开始提取
echo ""
echo "开始特征提取..."
echo "------------------------------------------------------"

python examples/extract_line_features_document_level.py

echo ""
echo "======================================================"
echo "✓ 特征提取完成！"
echo "======================================================"
echo ""
echo "输出目录: $LAYOUTLMFT_FEATURES_DIR"
echo ""
echo "生成的文件："
echo "  - train_line_features_chunk_*.pkl (训练集特征)"
echo "  - validation_line_features_chunk_*.pkl (验证集特征)"
echo ""
echo "下一步："
echo "  运行阶段二：bash scripts/train_parent_finder_full.sh"
echo ""

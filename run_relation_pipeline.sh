#!/bin/bash
# 运行完整的关系分类pipeline

echo "=========================================="
echo "HRDoc 层级关系分类 Pipeline (方案C)"
echo "=========================================="

export HF_HOME=/mnt/e/models/HuggingFace
export PYTHONPATH=/root/code/layoutlmft:$PYTHONPATH

cd /root/code/layoutlmft

# Step 1: 提取行级特征
echo ""
echo "[Step 1/2] 从LayoutLMv2提取行级特征..."
/root/miniforge3/envs/layoutlmv2/bin/python examples/extract_line_features.py

if [ $? -ne 0 ]; then
    echo "✗ 特征提取失败"
    exit 1
fi

# Step 2: 训练关系分类器
echo ""
echo "[Step 2/2] 训练二分类关系分类器..."
/root/miniforge3/envs/layoutlmv2/bin/python examples/train_relation_classifier.py

if [ $? -ne 0 ]; then
    echo "✗ 训练失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Pipeline 完成!"
echo "=========================================="

#!/bin/bash
# 快速测试脚本：验证树构建pipeline

set -e

echo "=========================================="
echo "HRDoc 树构建 Pipeline 快速测试"
echo "=========================================="

# 1. 测试树结构定义
echo ""
echo "1. 测试树结构定义 (document_tree.py)..."
python document_tree.py
echo "✓ 树结构定义测试通过"

# 2. 测试完整推理（只处理1个样本）
echo ""
echo "2. 测试完整推理 pipeline (inference_build_tree.py)..."
python inference_build_tree.py \
    --subtask2_model /mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt \
    --subtask3_model /mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt \
    --features_dir /mnt/e/models/train_data/layoutlmft/line_features \
    --output_dir ./test_outputs \
    --max_samples 2 \
    --max_chunks 1 \
    --save_json \
    --save_markdown \
    --save_ascii

echo ""
echo "✓ 完整推理测试通过"

# 3. 显示输出
echo ""
echo "3. 查看生成的树..."
echo ""
echo "--- JSON 格式 ---"
cat ./test_outputs/tree_0000.json | head -30
echo "..."

echo ""
echo "--- Markdown 格式 ---"
cat ./test_outputs/tree_0000.md

echo ""
echo "--- ASCII 格式 ---"
cat ./test_outputs/tree_0000_ascii.txt

echo ""
echo "=========================================="
echo "测试完成！"
echo "输出目录: ./test_outputs/"
echo "=========================================="

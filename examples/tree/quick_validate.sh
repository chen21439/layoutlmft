#!/bin/bash
# 快速验证脚本 - 验证阶段4能否正常工作

set -e

echo "================================================================"
echo "阶段4快速验证 - 使用validation数据测试"
echo "================================================================"

# 检查输入文件
echo -e "\n[1/4] 检查输入文件..."
echo "  ✓ line_features:"
ls -lh /mnt/e/models/train_data/layoutlmft/line_features/validation_*.pkl | head -1

echo "  ✓ SubTask 2模型:"
ls -lh /mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt

echo "  ✓ SubTask 3模型:"
ls -lh /mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt

# 快速推理（只处理1个样本）
echo -e "\n[2/4] 快速推理（处理1个validation样本）..."
cd /root/code/layoutlmft/examples/tree

python inference_build_tree.py \
    --subtask2_model /mnt/e/models/train_data/layoutlmft/parent_finder_simple/best_model.pt \
    --subtask3_model /mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt \
    --features_dir /mnt/e/models/train_data/layoutlmft/line_features \
    --split validation \
    --max_samples 1 \
    --max_chunks 1 \
    --output_dir ./quick_test \
    --save_json \
    --save_markdown \
    --save_ascii

# 查看结果
echo -e "\n[3/4] 查看生成的树..."
echo "================================================================"
echo "Markdown格式:"
echo "================================================================"
cat ./quick_test/tree_0000.md

echo -e "\n================================================================"
echo "ASCII格式:"
echo "================================================================"
cat ./quick_test/tree_0000_ascii.txt

# 统计信息
echo -e "\n[4/4] 统计信息..."
echo "================================================================"
cat ./quick_test/summary.json | python3 -m json.tool

echo -e "\n================================================================"
echo "✅ 验证完成！"
echo "================================================================"
echo "生成的文件："
ls -lh ./quick_test/tree_* ./quick_test/summary.json
echo ""
echo "下一步："
echo "  - 查看更多样本: python inference_build_tree.py --max_samples 10"
echo "  - 完整评估: python evaluate_end_to_end.py"
echo "================================================================"

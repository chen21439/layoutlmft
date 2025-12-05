#!/bin/bash
# 文档级别 ParentFinder 训练脚本
# 与页面级别的区别：使用文档级别的特征文件，支持跨页父子关系

source /root/miniforge3/etc/profile.d/conda.sh
conda activate layoutlmv2

# 配置路径
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc"
export LAYOUTLMFT_OUTPUT_DIR="/mnt/e/models/train_data/layoutlmft"

cd /root/code/layoutlmft

# 文档级别训练参数
# - level=document: 使用文档级别模式（max_lines=512）
# - mode=full: 使用完整的GRU模型（论文方法）
# - batch_size=1: 文档级别batch较大，降低batch_size
# - num_epochs=10: 小规模测试使用10个epoch
# - max_chunks=1: 只用1个chunk测试（约100个文档）

python examples/train_parent_finder.py \
    --mode full \
    --level document \
    --batch_size 1 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --max_chunks 1 \
    --features_dir "$LAYOUTLMFT_FEATURES_DIR" \
    --output_dir "$LAYOUTLMFT_OUTPUT_DIR"

echo ""
echo "训练完成！模型保存在: $LAYOUTLMFT_OUTPUT_DIR/parent_finder_full/best_model.pt"

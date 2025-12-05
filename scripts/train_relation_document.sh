#!/bin/bash
# 文档级别关系分类器训练脚本
# 与页面级别的区别：使用文档级别的特征文件，支持跨页关系

source /root/miniforge3/etc/profile.d/conda.sh
conda activate layoutlmv2

# 配置路径 - 使用文档级别特征目录
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_doc"
export LAYOUTLMFT_OUTPUT_DIR="/mnt/e/models/train_data/layoutlmft"
export MAX_CHUNKS="1"  # 小规模测试：只用1个chunk（约100个文档）

cd /root/code/layoutlmft

python examples/train_multiclass_relation.py

echo ""
echo "训练完成！模型保存在: $LAYOUTLMFT_OUTPUT_DIR/multiclass_relation/best_model.pt"

#!/bin/bash
# 使用自定义数据运行完整评估流程

set -e

echo "========================================"
echo "使用自定义数据进行评估"
echo "========================================"

# ====== 配置路径 ======

# 1. 数据目录（你的PDF提取的数据）
export HRDOC_DATA_DIR="/mnt/e/programFile/AIProgram/modelTrain/HRDoc/output"

# 2. 一阶段模型路径（已训练的LayoutLMv2模型）
export LAYOUTLMFT_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000"

# 3. 特征输出/读取目录
export LAYOUTLMFT_FEATURES_DIR="/mnt/e/models/train_data/layoutlmft/line_features_custom"

# 4. 二阶段模型路径（父节点查找）
export SUBTASK2_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/parent_finder_full/best_model.pt"

# 5. 三阶段模型路径（关系分类）
export SUBTASK3_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt"

# ====== 其他配置 ======
export LAYOUTLMFT_BATCH_SIZE="10"           # 批次大小（根据显存调整）
export LAYOUTLMFT_SAMPLES_PER_CHUNK="100"   # 每个chunk的样本数
export MAX_CHUNKS="-1"                       # 使用所有chunk（-1表示不限制）

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
PYTHON=${PYTHON:-python}

echo ""
echo "配置信息："
echo "  数据目录: $HRDOC_DATA_DIR"
echo "  模型路径: $LAYOUTLMFT_MODEL_PATH"
echo "  特征目录: $LAYOUTLMFT_FEATURES_DIR"
echo "  SubTask2模型: $SUBTASK2_MODEL_PATH"
echo "  SubTask3模型: $SUBTASK3_MODEL_PATH"
echo ""

# ====== 步骤1: 提取行级特征 ======
echo "========================================"
echo "步骤1: 提取行级特征"
echo "========================================"

if [ -d "$LAYOUTLMFT_FEATURES_DIR" ]; then
    echo "⚠️  特征目录已存在: $LAYOUTLMFT_FEATURES_DIR"
    read -p "是否删除并重新提取？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$LAYOUTLMFT_FEATURES_DIR"
        echo "✓ 已删除旧特征"
    else
        echo "跳过特征提取，使用现有特征"
    fi
fi

if [ ! -d "$LAYOUTLMFT_FEATURES_DIR" ]; then
    $PYTHON examples/extract_line_features.py

    if [ $? -ne 0 ]; then
        echo "✗ 特征提取失败"
        exit 1
    fi
    echo "✓ 特征提取完成"
fi

# ====== 步骤2: 运行评估 ======
echo ""
echo "========================================"
echo "步骤2: 运行End-to-End评估"
echo "========================================"

$PYTHON examples/tree/evaluate_end_to_end.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 评估完成！"
    echo "========================================"
else
    echo ""
    echo "✗ 评估失败"
    exit 1
fi

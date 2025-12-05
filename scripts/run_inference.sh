#!/bin/bash
# PDF数据推理：完整三阶段推理（和训练一致）

set -e

echo "========================================"
echo "HRDoc 推理模式 - 构建文档层级树"
echo "========================================"

# ====== 配置路径 ======
export HRDOC_DATA_DIR="/mnt/e/programFile/AIProgram/modelTrain/HRDoc/output"
export SUBTASK1_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000"
export SUBTASK2_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/parent_finder_full/best_model.pt"
export SUBTASK3_MODEL_PATH="/mnt/e/models/train_data/layoutlmft/multiclass_relation/best_model.pt"
OUTPUT_DIR="/root/code/layoutlmft/outputs/trees"

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
PYTHON=${PYTHON:-python}

echo ""
echo "配置信息："
echo "  数据目录: $HRDOC_DATA_DIR"
echo "  一阶段模型: $SUBTASK1_MODEL_PATH"
echo "  二阶段模型: $SUBTASK2_MODEL_PATH"
echo "  三阶段模型: $SUBTASK3_MODEL_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# ====== 完整推理（和训练一致） ======
echo "========================================"
echo "开始完整推理（三阶段）"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

$PYTHON examples/tree/inference_build_tree.py \
    --subtask1_model "$SUBTASK1_MODEL_PATH" \
    --subtask2_model "$SUBTASK2_MODEL_PATH" \
    --subtask3_model "$SUBTASK3_MODEL_PATH" \
    --data_dir "$HRDOC_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split test \
    --max_samples 10 \
    --output_format hrds

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 推理完成！"
    echo "========================================"
    echo "输出目录: $OUTPUT_DIR"
    echo ""
    echo "输出文件："
    ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | head -10
    echo ""
    echo "查看结果："
    echo "  查看page_*.json: cat $OUTPUT_DIR/page_0000.json | jq ."
    echo "  查看summary.json: cat $OUTPUT_DIR/summary.json | jq .trees[0]"
else
    echo ""
    echo "✗ 推理失败"
    exit 1
fi

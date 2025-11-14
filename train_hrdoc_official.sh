#!/usr/bin/env bash
# HRDoc 论文对齐训练脚本
# 严格按照论文参数：30000步(Simple) 或 40000步(Hard)
# 论文引用: "batch size of 3 (page-level) for 30,000 steps" (HRDoc README)

set -e

# 选择数据集版本
DATASET=${1:-simple}  # simple 或 hard

if [ "$DATASET" == "simple" ]; then
    MAX_STEPS=30000
    OUTPUT_DIR="./output/hrdoc_simple_full"
    EXPECTED_TIME="~4.5小时"
elif [ "$DATASET" == "hard" ]; then
    MAX_STEPS=40000
    OUTPUT_DIR="./output/hrdoc_hard_full"
    EXPECTED_TIME="~6小时"
else
    echo "错误: 未知数据集版本 '$DATASET'"
    echo "用法: ./train_hrdoc_official.sh [simple|hard]"
    exit 1
fi

echo "=========================================="
echo "HRDoc 完整训练（论文对齐）"
echo "=========================================="
echo "数据集: HRDoc-$DATASET"
echo "训练步数: $MAX_STEPS"
echo "Batch Size: 3 (page-level)"
echo "预计时长: $EXPECTED_TIME (V100/A100)"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""

# 环境变量
export HF_HOME=${HF_HOME:-/mnt/e/models/HuggingFace}
export PYTHONPATH=/root/code/layoutlmft:$PYTHONPATH

# Python解释器
PYTHON=${PYTHON:-/root/miniforge3/envs/layoutlmv2/bin/python}

# 确认是否继续
read -p "⚠️  训练将耗时 $EXPECTED_TIME，是否继续？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消训练"
    exit 0
fi

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="

# 训练命令（严格对齐论文）
$PYTHON examples/run_hrdoc.py \
  --model_name_or_path microsoft/layoutlmv2-base-uncased \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --logging_steps 100 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --evaluation_strategy steps \
  --save_total_limit 3 \
  --overwrite_output_dir \
  --task_name ner \
  --return_entity_level_metrics \
  --fp16

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 训练完成！"
    echo "=========================================="
    echo "模型保存在: $OUTPUT_DIR"
    echo ""

    # 验证训练配置
    echo "验证训练参数..."
    $PYTHON - << PY
import torch
import json

# 读取训练状态
with open("$OUTPUT_DIR/trainer_state.json") as f:
    state = json.load(f)

# 读取训练参数
args = torch.load("$OUTPUT_DIR/training_args.bin")

print("\n训练参数验证:")
print(f"  ✓ max_steps: {state['max_steps']} (目标: $MAX_STEPS)")
print(f"  ✓ global_step: {state['global_step']}")
print(f"  ✓ batch_size: {args.per_device_train_batch_size} (论文: 3)")
print(f"  ✓ learning_rate: {args.learning_rate} (设置: 5e-5)")
print(f"  ✓ warmup_ratio: {args.warmup_ratio} (设置: 0.1)")
print(f"  ✓ weight_decay: {args.weight_decay} (设置: 0.01)")

# 检查是否达到目标步数
if state['global_step'] >= $MAX_STEPS * 0.95:
    print("\n✓ 训练步数符合预期")
else:
    print(f"\n⚠️  训练可能提前停止 (完成 {state['global_step']}/$MAX_STEPS 步)")
PY

    echo ""
    echo "下一步："
    echo "  1. 提取行级特征: python examples/extract_line_features.py"
    echo "  2. 训练关系分类:"
    echo "     - 二分类: python examples/train_relation_classifier.py"
    echo "     - 多分类: python examples/train_multiclass_relation.py"
else
    echo ""
    echo "=========================================="
    echo "✗ 训练失败 (退出码: $EXIT_CODE)"
    echo "=========================================="
    exit $EXIT_CODE
fi

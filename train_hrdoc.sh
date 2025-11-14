#!/bin/bash
# HRDoc 版面识别训练脚本（支持多环境）
# 用法: ./train_hrdoc.sh [local|cloud|quick|auto]

set -e  # 遇到错误立即退出

# 默认环境为 auto（自动检测）
ENV=${1:-auto}

echo "=========================================="
echo "HRDoc 版面识别训练"
echo "=========================================="
echo "环境: $ENV"
echo ""

# 设置环境变量
export PYTHONPATH=/root/code/layoutlmft:$PYTHONPATH

cd /root/code/layoutlmft

# Python环境
PYTHON=/root/miniforge3/envs/layoutlmv2/bin/python

# 加载对应环境的配置
CONFIG_FILE="./configs/${ENV}_config.json"

if [ "$ENV" == "auto" ]; then
    echo "自动检测环境..."
    # 运行Python脚本自动检测并获取环境
    DETECTED_ENV=$($PYTHON -c "import sys; sys.path.insert(0, '.'); from configs.env_config import EnvironmentDetector; print(EnvironmentDetector.detect_environment())")
    ENV=$DETECTED_ENV
    CONFIG_FILE="./configs/${ENV}_config.json"
    echo "✓ 检测到环境: $ENV"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ 配置文件不存在: $CONFIG_FILE"
    echo "请先运行: python configs/env_config.py"
    exit 1
fi

echo "使用配置: $CONFIG_FILE"
echo ""

# 读取配置参数
MAX_STEPS=$(cat $CONFIG_FILE | grep -oP '"max_steps":\s*\K\d+')
BATCH_SIZE=$(cat $CONFIG_FILE | grep -oP '"per_device_train_batch_size":\s*\K\d+')
OUTPUT_DIR=$(cat $CONFIG_FILE | grep -oP '"output_dir":\s*"\K[^"]+')
LOCAL_MODEL=$(cat $CONFIG_FILE | grep -oP '"local_model_path":\s*"\K[^"]+')

echo "训练参数预览:"
echo "  - Max Steps: $MAX_STEPS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Output Dir: $OUTPUT_DIR"
echo "  - Model Path: $LOCAL_MODEL"
echo ""

# 确认是否继续
if [ "$ENV" == "cloud" ]; then
    echo "⚠️  云环境训练将运行 $MAX_STEPS 步，预计耗时 4-6 小时"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消训练"
        exit 0
    fi
fi

# 开始训练
echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="

$PYTHON examples/run_hrdoc.py \
  --model_name_or_path $LOCAL_MODEL \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --logging_steps $(cat $CONFIG_FILE | grep -oP '"logging_steps":\s*\K\d+') \
  --save_steps $(cat $CONFIG_FILE | grep -oP '"save_steps":\s*\K\d+') \
  --save_total_limit 3 \
  --fp16 \
  --overwrite_output_dir \
  --task_name ner \
  --return_entity_level_metrics

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 训练完成！"
    echo "=========================================="
    echo "模型保存在: $OUTPUT_DIR"
    echo ""
    echo "下一步："
    echo "  1. 查看训练日志: tensorboard --logdir $OUTPUT_DIR"
    echo "  2. 提取行级特征: python examples/extract_line_features.py"
    echo "  3. 训练关系分类: python examples/train_relation_classifier.py"
else
    echo ""
    echo "✗ 训练失败"
    exit 1
fi

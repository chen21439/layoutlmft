#!/bin/bash
# 安装监控依赖

echo "=========================================="
echo "安装训练监控依赖"
echo "=========================================="

PYTHON=${PYTHON:-/root/miniforge3/envs/layoutlmv2/bin/python}
PIP=${PYTHON} -m pip

echo ""
echo "当前Python: $PYTHON"
$PYTHON --version

echo ""
echo "1. 安装监控工具..."
$PIP install psutil gputil -q

echo ""
echo "2. 验证安装..."
$PYTHON check_monitoring_deps.py

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "可选：安装 Weights & Biases (云端监控)"
echo "  pip install wandb"
echo "  wandb login"

#!/bin/bash
# 测试多环境配置系统

echo "=========================================="
echo "测试环境配置系统"
echo "=========================================="

cd /root/code/layoutlmft
PYTHON=/root/miniforge3/envs/layoutlmv2/bin/python

echo ""
echo "1. 检测环境信息..."
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from configs.env_config import EnvironmentDetector
EnvironmentDetector.print_environment_info()
"

echo ""
echo "2. 检查配置文件..."
for env in local cloud quick; do
    if [ -f "configs/${env}_config.json" ]; then
        echo "  ✓ configs/${env}_config.json 存在"
    else
        echo "  ✗ configs/${env}_config.json 不存在"
    fi
done

echo ""
echo "3. 加载配置测试..."
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from configs.env_config import get_config

for env in ['local', 'cloud', 'quick']:
    config = get_config(env)
    print(f'  {env}: max_steps={config.max_steps}, batch_size={config.per_device_train_batch_size}')
"

echo ""
echo "4. 检查训练脚本..."
for script in train.py train_hrdoc.sh; do
    if [ -f "$script" ]; then
        echo "  ✓ $script 存在"
    else
        echo "  ✗ $script 不存在"
    fi
done

echo ""
echo "=========================================="
echo "✓ 环境配置系统测试完成"
echo "=========================================="
echo ""
echo "可用命令:"
echo "  ./train_hrdoc.sh auto        # Bash脚本（自动检测）"
echo "  python train.py --env auto   # Python脚本（自动检测）"
echo "  python train.py --show-config --env cloud  # 显示配置"

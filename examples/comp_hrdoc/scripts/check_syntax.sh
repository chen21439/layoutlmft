#!/bin/bash
# 语法和导入检查脚本 - 在提交前运行

set -e

echo "=== Python 语法检查 ==="
python3 -m py_compile examples/comp_hrdoc/scripts/train_doc.py
python3 -m py_compile examples/comp_hrdoc/engines/construct_trainer.py
python3 -m py_compile examples/comp_hrdoc/utils/stage_feature_extractor.py
python3 -m py_compile examples/comp_hrdoc/utils/label_utils.py
python3 -m py_compile examples/comp_hrdoc/metrics/construct_metrics.py
echo "✓ 语法检查通过"

echo ""
echo "=== 检查导入路径 ==="

# 检查关键导入是否存在
check_import() {
    module=$1
    file=$2

    if grep -q "from $module import\|import $module" "$file"; then
        path="${module//./\/}"
        if [ -f "$path.py" ] || [ -f "$path/__init__.py" ]; then
            echo "✓ $module"
        else
            echo "✗ $module (文件不存在: $path.py)"
            exit 1
        fi
    fi
}

# train_doc.py 的导入
check_import "examples.comp_hrdoc.utils.stage_feature_extractor" "examples/comp_hrdoc/scripts/train_doc.py"
check_import "examples.comp_hrdoc.models.build" "examples/comp_hrdoc/scripts/train_doc.py"
check_import "examples.comp_hrdoc.engines.construct_trainer" "examples/comp_hrdoc/scripts/train_doc.py"
check_import "examples.stage.data.hrdoc_data_loader" "examples/comp_hrdoc/scripts/train_doc.py"
check_import "examples.stage.joint_data_collator" "examples/comp_hrdoc/scripts/train_doc.py"

echo "✓ 导入路径检查通过"

echo ""
echo "=== 检查配置文件 ==="
for env in dev test; do
    if [ -f "examples/comp_hrdoc/configs/$env.yaml" ]; then
        echo "✓ $env.yaml"
    else
        echo "✗ $env.yaml 不存在"
        exit 1
    fi
done
echo "✓ 配置文件检查通过"

echo ""
echo "🎉 所有检查通过！可以提交代码"

#!/usr/bin/env python
# coding=utf-8
"""
测试 parent_finding 模块重构的向后兼容性
"""

import sys
import os

# 添加路径
STAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE_ROOT)

def test_imports():
    """测试所有导入路径"""
    print("Testing imports...")

    # 测试从 tasks.parent_finding 导入
    try:
        from tasks.parent_finding import (
            ChildParentDistributionMatrix,
            build_child_parent_matrix_from_dataset,
            ParentFindingTask,
        )
        print("✓ Direct import from tasks.parent_finding: OK")
    except ImportError as e:
        print(f"✗ Direct import from tasks.parent_finding: FAILED - {e}")
        return False

    # 测试从 tasks 包导入
    try:
        from tasks import (
            ChildParentDistributionMatrix as CPDM,
            build_child_parent_matrix_from_dataset as build_mcp,
            ParentFindingTask as PFTask,
        )
        print("✓ Import from tasks package: OK")
    except ImportError as e:
        print(f"✗ Import from tasks package: FAILED - {e}")
        return False

    return True


def test_basic_functionality():
    """测试基本功能"""
    print("\nTesting basic functionality...")

    from tasks.parent_finding import ChildParentDistributionMatrix
    import numpy as np

    # 创建矩阵
    cp_matrix = ChildParentDistributionMatrix(num_classes=16, pseudo_count=5)
    print("✓ ChildParentDistributionMatrix instantiation: OK")

    # 更新统计
    cp_matrix.update(child_label=0, parent_label=-1)  # ROOT
    cp_matrix.update(child_label=1, parent_label=0)
    cp_matrix.update(child_label=2, parent_label=1)
    print("✓ Update statistics: OK")

    # 构建矩阵
    cp_matrix.build()
    print("✓ Build matrix: OK")

    # 检查矩阵形状
    expected_shape = (17, 16)  # (num_classes+1, num_classes)
    if cp_matrix.matrix.shape == expected_shape:
        print(f"✓ Matrix shape: OK ({cp_matrix.matrix.shape})")
    else:
        print(f"✗ Matrix shape: FAILED (expected {expected_shape}, got {cp_matrix.matrix.shape})")
        return False

    # 获取 tensor
    try:
        import torch
        tensor = cp_matrix.get_tensor(device='cpu')
        print(f"✓ Get tensor: OK (shape={tensor.shape})")
    except Exception as e:
        print(f"✗ Get tensor: FAILED - {e}")
        return False

    return True


def test_parent_finding_task():
    """测试 ParentFindingTask 类"""
    print("\nTesting ParentFindingTask...")

    from tasks.parent_finding import ParentFindingTask
    import torch

    # 创建任务实例
    task = ParentFindingTask(use_soft_mask=False)
    print("✓ ParentFindingTask instantiation: OK")

    # 创建模拟数据
    batch_size = 2
    num_lines = 5
    device = 'cpu'

    # 模拟 parent_logits: [B, L+1, L+1]
    parent_logits = torch.randn(batch_size, num_lines+1, num_lines+1, device=device)

    # 模拟 line_parent_ids: [B, L]
    line_parent_ids = torch.tensor([
        [-1, 0, 1, 2, 3],  # 第一个文档
        [-1, -1, 0, 1, 2],  # 第二个文档
    ], dtype=torch.long, device=device)

    # 模拟 line_mask: [B, L]
    line_mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, True, False],  # 最后一行无效
    ], dtype=torch.bool, device=device)

    # 测试 decode
    try:
        predictions = task.decode(parent_logits, line_mask)
        print(f"✓ ParentFindingTask.decode: OK (shape={predictions.shape})")
    except Exception as e:
        print(f"✗ ParentFindingTask.decode: FAILED - {e}")
        return False

    # 测试 compute_loss
    try:
        loss, acc = task.compute_loss(parent_logits, line_parent_ids, line_mask)
        print(f"✓ ParentFindingTask.compute_loss: OK (loss={loss:.4f}, acc={acc:.4f})")
    except Exception as e:
        print(f"✗ ParentFindingTask.compute_loss: FAILED - {e}")
        return False

    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Parent Finding Refactor - Backward Compatibility Test")
    print("=" * 60)

    all_passed = True

    # 测试导入
    if not test_imports():
        all_passed = False

    # 测试基本功能
    if not test_basic_functionality():
        all_passed = False

    # 测试 ParentFindingTask
    if not test_parent_finding_task():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

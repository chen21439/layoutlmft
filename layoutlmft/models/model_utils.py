#!/usr/bin/env python
# coding=utf-8
"""
模型加载和保存的工具函数
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def load_best_model(
    model,
    checkpoint_dir="./output/relation_classifier",
    device="cuda",
    return_metrics=False
):
    """
    加载最佳模型权重

    Args:
        model: 要加载权重的模型实例
        checkpoint_dir: checkpoint目录
        device: 设备 (cuda/cpu)
        return_metrics: 是否返回训练指标

    Returns:
        model: 加载了权重的模型
        metrics: (可选) 训练指标字典

    Example:
        >>> from layoutlmft.models.relation_classifier import SimpleRelationClassifier
        >>> model = SimpleRelationClassifier()
        >>> model = load_best_model(model, device="cuda")
        >>> # 或者同时获取指标
        >>> model, metrics = load_best_model(model, return_metrics=True)
        >>> print(f"Best F1: {metrics['val_f1']:.4f}")
    """
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(best_model_path):
        # 尝试加载旧版权重文件
        weights_path = os.path.join(checkpoint_dir, "best_model_weights.pt")
        if os.path.exists(weights_path):
            logger.warning(f"未找到 {best_model_path}, 加载 {weights_path}")
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            logger.info(f"✓ 已加载模型权重 (无指标信息)")
            if return_metrics:
                return model, {}
            return model
        else:
            raise FileNotFoundError(
                f"未找到模型文件: {best_model_path} 或 {weights_path}"
            )

    # 加载完整checkpoint
    logger.info(f"从 {best_model_path} 加载模型...")
    checkpoint = torch.load(best_model_path, map_location=device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 提取指标
    metrics = {
        'epoch': checkpoint.get('epoch', -1),
        'val_f1': checkpoint.get('val_f1', 0.0),
        'val_loss': checkpoint.get('val_loss', 0.0),
    }
    if 'val_metrics' in checkpoint:
        metrics.update(checkpoint['val_metrics'])

    logger.info(f"✓ 已加载模型 (Epoch {metrics['epoch']}, F1: {metrics['val_f1']:.4f})")

    if return_metrics:
        return model, metrics
    return model


def load_checkpoint(
    model,
    optimizer=None,
    checkpoint_path=None,
    device="cuda"
):
    """
    从指定checkpoint恢复训练

    Args:
        model: 模型实例
        optimizer: (可选) 优化器实例
        checkpoint_path: checkpoint文件路径
        device: 设备

    Returns:
        model: 加载了权重的模型
        optimizer: (可选) 恢复了状态的优化器
        epoch: 已训练的epoch数
        metrics: 训练指标

    Example:
        >>> model = SimpleRelationClassifier()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> model, optimizer, epoch, metrics = load_checkpoint(
        ...     model, optimizer,
        ...     checkpoint_path="./output/relation_classifier/checkpoint-epoch3.pt"
        ... )
        >>> print(f"从Epoch {epoch} 恢复训练")
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到checkpoint: {checkpoint_path}")

    logger.info(f"从 {checkpoint_path} 恢复训练...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 加载优化器
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('val_metrics', {})
    metrics['val_f1'] = checkpoint.get('val_f1', 0.0)
    metrics['val_loss'] = checkpoint.get('val_loss', 0.0)

    logger.info(f"✓ 已恢复到Epoch {epoch} (F1: {metrics['val_f1']:.4f})")

    if optimizer is not None:
        return model, optimizer, epoch, metrics
    return model, epoch, metrics


def list_checkpoints(checkpoint_dir="./output/relation_classifier"):
    """
    列出所有可用的checkpoint

    Args:
        checkpoint_dir: checkpoint目录

    Returns:
        checkpoints: checkpoint文件列表

    Example:
        >>> checkpoints = list_checkpoints()
        >>> for ckpt in checkpoints:
        ...     print(ckpt)
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"目录不存在: {checkpoint_dir}")
        return []

    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith('.pt'):
            full_path = os.path.join(checkpoint_dir, fname)
            # 获取文件信息
            try:
                ckpt = torch.load(full_path, map_location='cpu')
                info = {
                    'path': full_path,
                    'filename': fname,
                    'epoch': ckpt.get('epoch', -1),
                    'val_f1': ckpt.get('val_f1', 0.0),
                }
                checkpoints.append(info)
            except Exception as e:
                logger.warning(f"无法加载 {fname}: {e}")
                continue

    # 按epoch排序
    checkpoints.sort(key=lambda x: x['epoch'])

    return checkpoints


def print_checkpoint_info(checkpoint_dir="./output/relation_classifier"):
    """
    打印所有checkpoint的详细信息

    Example:
        >>> print_checkpoint_info()
        Available checkpoints in ./output/relation_classifier:
        ├─ checkpoint-epoch1.pt  | Epoch 1 | F1: 0.3245
        ├─ checkpoint-epoch2.pt  | Epoch 2 | F1: 0.4512
        ├─ checkpoint-epoch3.pt  | Epoch 3 | F1: 0.5704 ← Best
        └─ best_model.pt         | Epoch 3 | F1: 0.5704
    """
    checkpoints = list_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"未找到checkpoint在: {checkpoint_dir}")
        return

    print(f"\nAvailable checkpoints in {checkpoint_dir}:")

    # 找出最佳F1
    best_f1 = max(c['val_f1'] for c in checkpoints)

    for i, ckpt in enumerate(checkpoints):
        is_last = (i == len(checkpoints) - 1)
        prefix = "└─" if is_last else "├─"

        is_best = (ckpt['val_f1'] == best_f1 and ckpt['val_f1'] > 0)
        best_marker = " ← Best" if is_best else ""

        print(f"{prefix} {ckpt['filename']:<25} | "
              f"Epoch {ckpt['epoch']} | "
              f"F1: {ckpt['val_f1']:.4f}{best_marker}")

    print()

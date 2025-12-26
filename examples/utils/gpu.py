#!/usr/bin/env python
# coding=utf-8
"""
GPU 设置工具

CUDA_VISIBLE_DEVICES 必须在 import torch 之前设置，
否则 torch 会初始化所有可见 GPU，导致 DataParallel 等问题。

Usage:
    # 在脚本最开始，import torch 之前调用
    from utils.gpu import setup_gpu_early
    setup_gpu_early()

    import torch  # 现在 torch 只会看到配置指定的 GPU
"""

import os
import sys


def setup_gpu_early(env: str = None):
    """
    在 import torch 之前设置 CUDA_VISIBLE_DEVICES

    优先级：命令行 --gpu > 配置文件 > 系统默认

    Args:
        env: 环境名称（dev/test）。如果为 None，会尝试从命令行参数 --env 获取
    """
    # 优先检查命令行 --gpu 参数
    gpu_id = _get_gpu_from_argv()
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return

    # 如果没有指定 env，尝试从命令行参数获取
    if env is None:
        env = _get_env_from_argv()

    if env is None:
        return  # 没有指定环境，使用默认 GPU 设置

    try:
        from configs.config_loader import load_config
        config = load_config(env)
        if hasattr(config, 'gpu') and config.gpu.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices
    except Exception:
        pass  # 如果加载失败，使用默认 GPU 设置


def _get_gpu_from_argv():
    """从命令行参数中提取 --gpu 值"""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        elif arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
    return None


def _get_env_from_argv():
    """从命令行参数中提取 --env 值"""
    for i, arg in enumerate(sys.argv):
        if arg == "--env" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        elif arg.startswith("--env="):
            return arg.split("=", 1)[1]
    return None

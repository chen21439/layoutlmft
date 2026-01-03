#!/usr/bin/env python
# coding=utf-8
"""
配置加载和环境设置

提供统一的配置加载、环境变量设置、实验管理初始化等功能。
"""

import logging
import os
import sys
from typing import Tuple, Any, Optional

logger = logging.getLogger(__name__)


def load_config_and_setup(
    data_args,
    training_args,
    model_args,
    stage_name: str = "joint",
) -> Tuple[Any, str, Any]:
    """
    加载配置并设置环境

    Args:
        data_args: 数据参数（包含 env, dataset, covmatch 等）
        training_args: 训练参数（包含 exp, new_exp, output_dir 等）
        model_args: 模型参数（包含 model_name_or_path 等）
        stage_name: 阶段名称，用于实验管理 ("stage1", "joint" 等)

    Returns:
        config: 加载的配置对象
        data_dir: 数据目录路径
        exp_manager: 实验管理器
    """
    from configs.config_loader import load_config
    from .checkpoint_utils import get_latest_checkpoint
    from .experiment_manager import ensure_experiment

    config = load_config(data_args.env)
    if training_args.quick:
        config.quick_test.enabled = True
    config = config.get_effective_config()

    # 设置 HuggingFace 缓存目录
    if config.paths.hf_cache_dir:
        os.environ["HF_HOME"] = config.paths.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = config.paths.hf_cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(config.paths.hf_cache_dir, "datasets")

    # 命令行 artifact_dir：直接作为输出目录，跳过实验管理的自动目录创建
    artifact_dir = getattr(training_args, 'artifact_dir', '') or ''
    if artifact_dir:
        # 直接使用 artifact_dir 作为 output_dir
        training_args.output_dir = artifact_dir
        logger.info(f"Using artifact_dir as output_dir: {artifact_dir}")

    # 初始化实验管理器（即使指定了 artifact_dir，其他功能可能仍需要）
    exp_manager, exp_dir = ensure_experiment(
        config,
        exp=training_args.exp,
        new_exp=training_args.new_exp if not artifact_dir else "",  # artifact_dir 模式下不创建新实验
        name=training_args.exp_name or f"{stage_name.capitalize()} {data_args.dataset.upper()}",
    )

    # 设置模型路径（优先级：model_name_or_path > 自动检测）
    if model_args.model_name_or_path:
        logger.info(f"Using manually specified model: {model_args.model_name_or_path}")
    else:
        # 尝试从实验目录加载 Stage 1 模型
        stage1_dir = exp_manager.get_stage_dir(training_args.exp, "stage1", data_args.dataset)
        stage1_model = get_latest_checkpoint(stage1_dir)

        if stage1_model:
            model_args.model_name_or_path = stage1_model
            logger.info(f"Using Stage 1 model from experiment: {stage1_model}")
        else:
            # 尝试 legacy 路径
            if hasattr(config.paths, 'stage1_model_path') and config.paths.stage1_model_path:
                legacy_dir = f"{config.paths.stage1_model_path}_{data_args.dataset}"
                stage1_model = get_latest_checkpoint(legacy_dir)
                if stage1_model:
                    model_args.model_name_or_path = stage1_model

            if model_args.model_name_or_path is None:
                model_args.model_name_or_path = config.model.local_path or config.model.name_or_path
                logger.warning(f"No Stage 1 model found, using pretrained: {model_args.model_name_or_path}")

    # 设置输出目录（仅当未指定 artifact_dir 且使用默认值时，才替换为实验目录）
    if not artifact_dir:
        default_output_dirs = ["./output/joint", "./output/stage1", "./output/stage34"]
        if training_args.output_dir in default_output_dirs:
            training_args.output_dir = exp_manager.get_stage_dir(training_args.exp, stage_name, data_args.dataset)

    # 数据目录
    data_dir = config.dataset.get_data_dir(data_args.dataset)
    os.environ["HRDOC_DATA_DIR"] = data_dir

    # tender 数据集默认不使用缓存（数据量小，避免缓存问题）
    if data_args.dataset == "tender" and not data_args.force_rebuild:
        data_args.force_rebuild = True
        logger.info("tender dataset: force_rebuild enabled by default")

    # Covmatch 目录 (命令行参数优先于配置文件)
    covmatch_from_cli = data_args.covmatch is not None
    if covmatch_from_cli:
        config.dataset.covmatch = data_args.covmatch
        logger.info(f"Using covmatch from command line: {data_args.covmatch}")
    covmatch_dir = config.dataset.get_covmatch_dir(data_args.dataset)
    if os.path.exists(covmatch_dir):
        os.environ["HRDOC_SPLIT_DIR"] = covmatch_dir
        logger.info(f"Covmatch directory: {covmatch_dir}")
    else:
        if covmatch_from_cli:
            # 命令行明确指定了 covmatch，但目录不存在，退出
            logger.error(f"Covmatch directory not found: {covmatch_dir}")
            logger.error(f"Specified covmatch '{data_args.covmatch}' does not exist.")
            logger.error(f"Available covmatch directories can be found in: {os.path.dirname(covmatch_dir)}")
            # 列出可用的 covmatch 目录
            parent_dir = os.path.dirname(covmatch_dir)
            if os.path.exists(parent_dir):
                available = [d for d in os.listdir(parent_dir) if d.startswith("doc_covmatch")]
                if available:
                    logger.error(f"Available covmatch options: {', '.join(sorted(available))}")
                else:
                    logger.error(f"No covmatch directories found in {parent_dir}")
            sys.exit(1)
        else:
            logger.warning(f"Covmatch directory not found: {covmatch_dir}, using default directory structure")

    # GPU 设置
    if config.gpu.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices

    return config, data_dir, exp_manager


def setup_gpu_early():
    """
    在 import torch 之前设置 GPU，避免 DataParallel 问题

    必须在脚本最开始调用，在任何 torch import 之前。
    """
    # 优先从命令行参数中提取 --gpu
    gpu_id = None
    env = "test"  # 默认值
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
        if arg == "--env" and i + 1 < len(sys.argv):
            env = sys.argv[i + 1]

    # 如果命令行指定了 GPU，直接使用
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return

    # 否则从配置文件加载
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        sys.path.insert(0, PROJECT_ROOT)
        from configs.config_loader import load_config
        config = load_config(env)
        if config.gpu.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.cuda_visible_devices
    except Exception:
        pass  # 如果加载失败，使用默认 GPU 设置

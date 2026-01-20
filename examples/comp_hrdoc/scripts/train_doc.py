#!/usr/bin/env python
# coding=utf-8
"""
Construct 训练入口脚本 - Stage1 + Construct 联合训练

遵循项目结构规范：
- scripts/ 只放极薄入口脚本
- 业务逻辑在 engines/, models/, tasks/ 中
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.comp_hrdoc.utils.stage_feature_extractor import StageFeatureExtractor
from examples.comp_hrdoc.models.build import build_construct_from_features
from examples.comp_hrdoc.engines.construct_trainer import train_epoch, evaluate, save_model
from examples.comp_hrdoc.utils.label_utils import convert_stage_labels_to_construct
from examples.stage.data.hrdoc_data_loader import HRDocDataLoader, HRDocDataLoaderConfig
from examples.stage.joint_data_collator import HRDocDocumentLevelCollator

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Construct (Stage1 + Construct joint training)")

    # ==================== Environment ====================
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--seed", type=int, default=42)

    # ==================== Model ====================
    parser.add_argument("--model-name-or-path", type=str, required=True,
                        help="Path to pretrained model checkpoint (Stage1 + Construct joint training)")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--construct-num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to save memory")

    # ==================== Data ====================
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh", "tender"])
    parser.add_argument("--covmatch", type=str, default=None,
                        help="Covmatch split name (e.g., doc_random_dev2_section)")
    parser.add_argument("--max-regions", type=int, default=1024,
                        help="Max lines per document")
    parser.add_argument("--document-level", action="store_true", default=True,
                        help="Enable document-level training")
    parser.add_argument("--toc-only", action="store_true", default=True,
                        help="Only train on section headings")
    parser.add_argument("--section-label-id", type=int, default=4)

    # ==================== Training ====================
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate for Construct module")
    parser.add_argument("--stage1-lr", type=float, default=2e-5,
                        help="Learning rate for Stage1 components (backbone + cls_head + line_enhancer)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=False)

    # ==================== Stage1 Options ====================
    parser.add_argument("--freeze-visual", action="store_true", default=True,
                        help="Freeze visual encoder in backbone (save memory)")
    parser.add_argument("--compute-cls-loss", action="store_true", default=False,
                        help="Compute Stage1 classification loss")
    parser.add_argument("--lambda-cls", type=float, default=1.0,
                        help="Weight for Stage1 classification loss")

    # ==================== Logging & Checkpointing ====================
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Output directory (if None, uses experiment manager)")

    # ==================== Experiment Management ====================
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--new-exp", action="store_true")
    parser.add_argument("--exp-name", type=str, default=None)

    # ==================== Quick Test ====================
    parser.add_argument("--quick", action="store_true", help="Quick test with small data")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    # ==================== GPU ====================
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID(s), e.g., '0' or '0,1'")

    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(env: str, config_path: str = None) -> dict:
    """加载配置文件"""
    import yaml

    if config_path:
        config_file = Path(config_path)
    else:
        config_file = PROJECT_ROOT / "examples" / "comp_hrdoc" / "configs" / f"{env}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def main():
    args = parse_args()

    # ==================== GPU 设置 ====================
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ==================== Logging 设置 ====================
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    # ==================== 设置随机种子 ====================
    set_seed(args.seed)
    logger.info(f"Set random seed to: {args.seed}")

    # ==================== 加载配置 ====================
    config = load_config(args.env, args.config)
    logger.info(f"Loaded config from env: {args.env}")

    # ==================== 获取数据目录（使用全局 config）====================
    from configs.config_loader import load_config as load_global_config

    global_config = load_global_config(args.env).get_effective_config()
    data_dir = global_config.dataset.get_data_dir(args.dataset)
    logger.info(f"Data directory: {data_dir}")

    # ==================== 处理 covmatch ====================
    covmatch = args.covmatch or global_config.dataset.covmatch
    if not covmatch:
        logger.warning("No covmatch specified, using default from config")
    logger.info(f"Covmatch: {covmatch}")

    # 设置 covmatch 环境变量
    if covmatch:
        # 如果命令行指定了 covmatch，更新 config
        if args.covmatch:
            global_config.dataset.covmatch = args.covmatch

        covmatch_dir = global_config.dataset.get_covmatch_dir(args.dataset)

        if os.path.exists(covmatch_dir):
            os.environ["HRDOC_SPLIT_DIR"] = covmatch_dir
            logger.info(f"Covmatch directory: {covmatch_dir}")
        else:
            logger.error(f"Covmatch directory not found: {covmatch_dir}")
            parent_dir = os.path.dirname(covmatch_dir)
            if os.path.exists(parent_dir):
                available = [d for d in os.listdir(parent_dir) if d.startswith("doc_")]
                if available:
                    logger.error(f"Available splits: {', '.join(sorted(available)[:5])}")
            raise FileNotFoundError(f"Covmatch directory not found: {covmatch_dir}")

    # tender 数据集默认强制 rebuild
    force_rebuild = args.dataset == "tender"
    if force_rebuild:
        logger.info("tender dataset: force_rebuild enabled by default")

    # ==================== 设备 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ==================== 初始化 StageFeatureExtractor ====================
    logger.info("="*80)
    logger.info("Initializing StageFeatureExtractor")
    logger.info(f"  checkpoint: {args.model_name_or_path}")

    stage_feature_extractor = StageFeatureExtractor(
        checkpoint_path=args.model_name_or_path,
        device=str(device),
        config=config,
        max_lines=args.max_regions,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # 设置训练模式
    stage_feature_extractor.set_train_mode(freeze_visual=args.freeze_visual)
    logger.info(f"  freeze_visual: {args.freeze_visual}")

    # 获取 tokenizer
    tokenizer = stage_feature_extractor.tokenizer

    # ==================== 构建 Construct 模型 ====================
    logger.info("="*80)
    logger.info("Building Construct model")

    construct_model = build_construct_from_features(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.construct_num_layers,
        dropout=args.dropout,
        num_categories=14,  # Line-level classification categories
    )
    construct_model = construct_model.to(device)
    logger.info(f"  num_layers: {args.construct_num_layers}")
    logger.info(f"  hidden_size: {args.hidden_size}")

    # ==================== 构建数据加载器 ====================
    logger.info("="*80)
    logger.info("Building dataloaders")

    # 构建 HRDocDataLoader config
    data_loader_config = HRDocDataLoaderConfig(
        data_dir=data_dir,
        dataset_name=args.dataset,
        document_level=args.document_level,
        max_length=512,
        max_train_samples=args.max_train_samples if args.quick or args.max_train_samples else None,
        max_val_samples=args.max_val_samples if args.quick or args.max_val_samples else None,
        force_rebuild=force_rebuild,
    )

    # 创建数据加载器（只调用一次 prepare_datasets）
    data_loader = HRDocDataLoader(tokenizer, data_loader_config)
    datasets = data_loader.prepare_datasets()

    train_dataset = datasets.get("train", [])
    val_dataset = datasets.get("validation", [])

    logger.info(f"Train dataset: {len(train_dataset)} documents")
    logger.info(f"Val dataset: {len(val_dataset)} documents")

    # 复用 stage 的 collator
    collator = HRDocDocumentLevelCollator(tokenizer)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ==================== 配置优化器（分层学习率）====================
    logger.info("="*80)
    logger.info("Configuring optimizer")

    # Stage1 参数（backbone + cls_head + line_enhancer）
    backbone_params = stage_feature_extractor.get_trainable_params()

    # Construct 参数
    construct_params = list(construct_model.parameters())

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.stage1_lr, "name": "stage1"},
        {"params": construct_params, "lr": args.learning_rate, "name": "construct"},
    ], weight_decay=args.weight_decay)

    logger.info(f"  stage1_lr: {args.stage1_lr}")
    logger.info(f"  construct_lr: {args.learning_rate}")
    logger.info(f"  weight_decay: {args.weight_decay}")

    # ==================== 学习率调度器 ====================
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(f"  total_steps: {total_steps}")
    logger.info(f"  warmup_steps: {warmup_steps}")

    # ==================== AMP Scaler ====================
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    if args.fp16:
        logger.info("  Using mixed precision (FP16)")

    # ==================== 输出目录 ====================
    if args.artifact_dir:
        output_dir = Path(args.artifact_dir)
    else:
        # TODO: 使用 experiment_manager
        output_dir = Path(f"./outputs/exp_{args.exp or 'default'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(exist_ok=True)

    logger.info(f"  output_dir: {output_dir}")

    # ==================== 训练循环 ====================
    logger.info("="*80)
    logger.info("Starting training")
    logger.info(f"  num_epochs: {args.num_epochs}")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    logger.info("="*80)

    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*80}")

        # 训练
        train_metrics = train_epoch(
            model=construct_model,
            dataloader=train_loader,
            feature_extractor=stage_feature_extractor,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            scaler=scaler,
            use_amp=args.fp16,
            compute_cls_loss=args.compute_cls_loss,
            lambda_cls=args.lambda_cls,
        )

        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  - Construct Loss: {train_metrics['construct_loss']:.4f}")
        if args.compute_cls_loss:
            logger.info(f"  - Cls Loss: {train_metrics['cls_loss']:.4f}")

        # 评估
        val_metrics = evaluate(
            model=construct_model,
            dataloader=val_loader,
            feature_extractor=stage_feature_extractor,
            device=device,
        )

        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  - Parent Accuracy: {val_metrics['parent_accuracy']:.4f}")
        logger.info(f"  - Sibling Accuracy: {val_metrics['sibling_accuracy']:.4f}")

        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            logger.info(f"New best model! Saving to: {best_model_dir}")

            save_model(
                construct_model=construct_model,
                save_path=str(best_model_dir),
                feature_extractor=stage_feature_extractor,
                tokenizer=tokenizer,
            )

    logger.info("="*80)
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {best_model_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

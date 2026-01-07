#!/usr/bin/env python
"""Construct 模块训练脚本 (4.4)

使用预训练的 4.3 Order 模型提取的特征来训练 4.4 Construct 模块。
需要先运行 train_order.py 训练 Order 模型。

Usage:
    # 快速测试
    python train_construct.py --env test --quick

    # 完整训练（使用默认 Order 模型路径）
    python train_construct.py --env test --num-epochs 20

    # 指定 Order 模型路径
    python train_construct.py --env test --order-model-path /path/to/order/best_model

    # 创建新实验
    python train_construct.py --env test --new-exp --exp-name "Construct Training"
"""

# ==================== GPU 设置（必须在 import torch 之前）====================
import os
import sys

# 添加项目路径（必须在导入其他模块之前）
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from examples.comp_hrdoc.utils.config import setup_environment
setup_environment()
# ==================== GPU 设置结束 ====================

import argparse
import logging
from typing import Dict

import torch
from torch.optim import AdamW
from tqdm import tqdm

from examples.comp_hrdoc.data.region_feature_loader import (
    RegionFeatureDataset,
    RegionFeatureConfig,
    RegionFeatureCollator,
    create_region_feature_dataloaders,
)
from examples.comp_hrdoc.models.construct_only import (
    ConstructWithOrderFeatures,
    build_construct_with_order,
    save_construct_model,
    compute_construct_metrics,
    generate_sibling_labels,
)
from examples.comp_hrdoc.utils.experiment_manager import (
    ExperimentManager,
    get_artifact_path,
    ensure_experiment,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Construct module (4.4)")

    # Environment
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test"])

    # Order model (pre-trained)
    parser.add_argument("--order-model-path", type=str, default=None,
                        help="Path to pre-trained Order model. If not specified, uses latest from artifact.")
    parser.add_argument("--freeze-order", action="store_true", default=True,
                        help="Freeze Order model parameters (default: True)")
    parser.add_argument("--no-freeze-order", dest="freeze_order", action="store_false",
                        help="Allow Order model to be fine-tuned")

    # Construct model parameters
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of Transformer layers in Construct module")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Loss weights
    parser.add_argument("--parent-weight", type=float, default=1.0)
    parser.add_argument("--sibling-weight", type=float, default=0.5)
    parser.add_argument("--root-weight", type=float, default=0.3)

    # Data
    parser.add_argument("--features-dir", type=str, default=None)
    parser.add_argument("--max-regions", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)

    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    # Experiment management
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment identifier (None for current/latest)")
    parser.add_argument("--new-exp", action="store_true",
                        help="Create new experiment")
    parser.add_argument("--exp-name", type=str, default="",
                        help="Experiment name (for new experiments)")

    # Quick test
    parser.add_argument("--quick", action="store_true")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_order_model_path(env: str) -> str:
    """Find the best Order model from artifact directory."""
    artifact_root = get_artifact_path(env)

    # Look for order stage in experiments
    exp_dirs = sorted(Path(artifact_root).glob("exp_*"), reverse=True)

    for exp_dir in exp_dirs:
        order_path = exp_dir / "order_comp_hrdoc" / "best_model"
        if order_path.exists() and (order_path / "order_model.pt").exists():
            logger.info(f"Found Order model at: {order_path}")
            return str(order_path)

    raise FileNotFoundError(
        f"No Order model found in {artifact_root}. "
        "Please run train_order.py first or specify --order-model-path."
    )


def train_epoch(
    model: ConstructWithOrderFeatures,
    dataloader,
    optimizer,
    scheduler,
    device,
    args,
    scaler=None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    # Keep Order model in eval mode if frozen
    if model.freeze_order:
        model.order_model.eval()

    total_loss = 0.0
    total_parent_loss = 0.0
    total_sibling_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        # Move to device
        region_features = batch['region_features'].to(device)
        categories = batch['categories'].to(device)
        bboxes = batch['bboxes'].to(device)
        region_mask = batch['region_mask'].to(device)
        reading_orders = batch['reading_orders'].to(device)
        parent_ids = batch['parent_ids'].to(device)

        # Generate sibling labels from parent_ids (requires reading_orders)
        sibling_labels = generate_sibling_labels(parent_ids, reading_orders, region_mask)

        # Forward
        if args.fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    region_features=region_features,
                    categories=categories,
                    bboxes=bboxes,
                    region_mask=region_mask,
                    reading_orders=reading_orders,
                    parent_labels=parent_ids,
                    sibling_labels=sibling_labels,
                )
                loss = outputs['loss'] / args.gradient_accumulation_steps
        else:
            outputs = model(
                region_features=region_features,
                categories=categories,
                bboxes=bboxes,
                region_mask=region_mask,
                reading_orders=reading_orders,
                parent_labels=parent_ids,
                sibling_labels=sibling_labels,
            )
            loss = outputs['loss'] / args.gradient_accumulation_steps

        # Backward
        if args.fp16 and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16 and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs['loss'].item()
        total_parent_loss += outputs['parent_loss'].item()
        total_sibling_loss += outputs['sibling_loss'].item()
        num_batches += 1

        progress_bar.set_postfix({
            'loss': f"{outputs['loss'].item():.4f}",
            'parent': f"{outputs['parent_loss'].item():.4f}",
            'sibling': f"{outputs['sibling_loss'].item():.4f}",
        })

    return {
        'loss': total_loss / num_batches,
        'parent_loss': total_parent_loss / num_batches,
        'sibling_loss': total_sibling_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: ConstructWithOrderFeatures,
    dataloader,
    device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_parent_loss = 0.0
    total_sibling_loss = 0.0
    num_batches = 0

    all_metrics = {
        'parent_accuracy': 0.0,
        'sibling_accuracy': 0.0,
    }
    num_metric_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        region_features = batch['region_features'].to(device)
        categories = batch['categories'].to(device)
        bboxes = batch['bboxes'].to(device)
        region_mask = batch['region_mask'].to(device)
        reading_orders = batch['reading_orders'].to(device)
        parent_ids = batch['parent_ids'].to(device)

        # Generate sibling labels (requires reading_orders)
        sibling_labels = generate_sibling_labels(parent_ids, reading_orders, region_mask)

        outputs = model(
            region_features=region_features,
            categories=categories,
            bboxes=bboxes,
            region_mask=region_mask,
            reading_orders=reading_orders,
            parent_labels=parent_ids,
            sibling_labels=sibling_labels,
        )

        total_loss += outputs['loss'].item()
        total_parent_loss += outputs['parent_loss'].item()
        total_sibling_loss += outputs['sibling_loss'].item()
        num_batches += 1

        # Compute metrics
        metrics = compute_construct_metrics(
            parent_logits=outputs['parent_logits'],
            parent_labels=parent_ids,
            region_mask=region_mask,
            sibling_logits=outputs['sibling_logits'],
            sibling_labels=sibling_labels,
        )

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        num_metric_batches += 1

    result = {
        'loss': total_loss / num_batches,
        'parent_loss': total_parent_loss / num_batches,
        'sibling_loss': total_sibling_loss / num_batches,
    }

    for k in all_metrics:
        result[k] = all_metrics[k] / num_metric_batches

    return result


def main():
    args = parse_args()

    set_seed(args.seed)

    logger.info(f"Environment: {args.env}")
    logger.info(f"Arguments: {args}")

    # Quick test mode
    if args.quick:
        args.max_samples = args.max_samples or 50
        args.num_epochs = min(args.num_epochs, 2)
        args.batch_size = min(args.batch_size, 4)
        logger.info(f"Quick test mode: max_samples={args.max_samples}, epochs={args.num_epochs}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Find Order model path
    if args.order_model_path is None:
        args.order_model_path = find_order_model_path(args.env)
    logger.info(f"Using Order model from: {args.order_model_path}")

    # Setup experiment directory
    artifact_root = get_artifact_path(args.env)
    exp_manager, exp_dir = ensure_experiment(
        artifact_root=artifact_root,
        exp=args.exp,
        new_exp=args.new_exp,
        name=args.exp_name or "Construct Module Training",
        description=f"Train Construct module (4.4) with lr={args.learning_rate}, epochs={args.num_epochs}",
        config=vars(args),
    )

    # Get stage output directory
    output_dir = Path(exp_manager.get_stage_dir(args.exp, "construct", "comp_hrdoc"))
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Stage output directory: {output_dir}")

    # Mark stage as started
    exp_manager.mark_stage_started(args.exp, "construct", "comp_hrdoc")

    # Load data
    logger.info("Loading data...")
    data_config = RegionFeatureConfig(
        env=args.env,
        features_dir=args.features_dir,
        max_regions=args.max_regions,
        max_samples=args.max_samples,
    )

    train_loader, val_loader = create_region_feature_dataloaders(
        config=data_config,
        batch_size=args.batch_size,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Build model
    logger.info("Building Construct model with pre-trained Order...")
    model = build_construct_with_order(
        order_model_path=args.order_model_path,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_order=args.freeze_order,
        device=str(device),
    )
    model = model.to(device)

    # Update loss weights
    model.loss_fn.parent_weight = args.parent_weight
    model.loss_fn.sibling_weight = args.sibling_weight

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Order model frozen: {args.freeze_order}")

    # Optimizer (only trainable parameters)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # FP16
    scaler = None
    if args.fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using FP16 mixed precision")

    # Training loop
    logger.info("Starting training...")
    best_parent_acc = 0.0

    for epoch in range(args.num_epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{args.num_epochs} =====")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, args, scaler
        )
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Parent: {train_metrics['parent_loss']:.4f}, "
            f"Sibling: {train_metrics['sibling_loss']:.4f}"
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Parent: {val_metrics['parent_loss']:.4f}, "
            f"Sibling: {val_metrics['sibling_loss']:.4f}"
        )
        logger.info(
            f"Val - Parent Acc: {val_metrics['parent_accuracy']:.4f}, "
            f"Sibling Acc: {val_metrics.get('sibling_accuracy', 0):.4f}"
        )

        # Save best model
        if val_metrics['parent_accuracy'] > best_parent_acc:
            best_parent_acc = val_metrics['parent_accuracy']
            best_path = output_dir / "best_model"
            save_construct_model(model, str(best_path))
            logger.info(f"Saved best model (Parent Acc={best_parent_acc:.4f})")

            # Update experiment state
            exp_manager.update_stage_state(
                args.exp, "construct", "comp_hrdoc",
                best_checkpoint=str(best_path),
                metrics={
                    'best_parent_accuracy': best_parent_acc,
                    'best_root_accuracy': val_metrics['root_accuracy'],
                    'best_root_f1': val_metrics['root_f1'],
                    'epoch': epoch + 1,
                },
            )

    # Save final model
    final_path = output_dir / "final_model"
    save_construct_model(model, str(final_path))
    logger.info(f"Saved final model to {final_path}")

    # Save Order model path reference
    order_ref_path = output_dir / "order_model_ref.txt"
    with open(order_ref_path, 'w') as f:
        f.write(args.order_model_path)

    # Mark stage as completed
    exp_manager.mark_stage_completed(
        args.exp, "construct", "comp_hrdoc",
        best_checkpoint=str(output_dir / "best_model"),
        metrics={
            'best_parent_accuracy': best_parent_acc,
            'final_parent_accuracy': val_metrics['parent_accuracy'],
            'final_root_accuracy': val_metrics['root_accuracy'],
            'final_root_f1': val_metrics['root_f1'],
            'order_model_path': args.order_model_path,
        },
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best Parent Accuracy: {best_parent_acc:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Order model used: {args.order_model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

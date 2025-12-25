#!/usr/bin/env python
"""Order 模块训练脚本 (4.3)

使用预提取的 LayoutXLM 区域特征训练 Order 模块。
需要先运行 extract_region_features.py 提取特征。

Usage:
    python train_order.py --env test --quick           # 快速测试
    python train_order.py --env test --num-epochs 20   # 完整训练
    python train_order.py --env test --new-exp         # 创建新实验
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
from examples.comp_hrdoc.models.order_from_features import (
    OrderModuleFromFeatures,
    OrderLossFromFeatures,
    build_order_from_features,
    save_order_model,
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
    parser = argparse.ArgumentParser(description="Train Order module (4.3)")

    # Environment
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test"])

    # Model parameters
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-categories", type=int, default=5)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-spatial", action="store_true", default=True)
    parser.add_argument("--no-spatial", dest="use_spatial", action="store_false")

    # Loss weights
    parser.add_argument("--order-weight", type=float, default=1.0)
    parser.add_argument("--relation-weight", type=float, default=0.5)

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

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/order_module")

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


def compute_metrics(
    order_logits: torch.Tensor,
    relation_logits: torch.Tensor,
    reading_orders: torch.Tensor,
    parent_ids: torch.Tensor,
    relations: torch.Tensor,
    region_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    batch_size, num_regions = reading_orders.shape
    device = order_logits.device

    # ============ Order Accuracy (Pairwise) ============
    order_i = reading_orders.unsqueeze(2)
    order_j = reading_orders.unsqueeze(1)
    targets = (order_i < order_j).float()

    predictions = (order_logits > 0).float()

    valid_mask = region_mask.unsqueeze(2) & region_mask.unsqueeze(1)
    diag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=device).unsqueeze(0)
    valid_mask = valid_mask & diag_mask

    correct = ((predictions == targets) & valid_mask).sum().item()
    total_pairs = valid_mask.sum().item()

    pairwise_acc = correct / total_pairs if total_pairs > 0 else 0.0

    # ============ Kendall Tau ============
    total_tau = 0.0
    num_samples = 0

    for b in range(batch_size):
        sample_mask = region_mask[b]
        sample_logits = order_logits[b]
        sample_gt = reading_orders[b]

        valid_indices = sample_mask.nonzero().squeeze(-1)
        if len(valid_indices) < 2:
            continue

        # Predict order from pairwise scores
        scores = (sample_logits > 0).float().sum(dim=1)
        _, pred_order_indices = scores.sort(descending=True)
        pred_order = torch.zeros_like(scores, dtype=torch.long)
        for rank, idx in enumerate(pred_order_indices):
            pred_order[idx] = rank

        # Compute Kendall Tau
        pred_valid = pred_order[valid_indices].cpu().numpy()
        gt_valid = sample_gt[valid_indices].cpu().numpy()

        n = len(pred_valid)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                pred_diff = pred_valid[i] - pred_valid[j]
                gt_diff = gt_valid[i] - gt_valid[j]
                if pred_diff * gt_diff > 0:
                    concordant += 1
                elif pred_diff * gt_diff < 0:
                    discordant += 1

        if concordant + discordant > 0:
            tau = (concordant - discordant) / (concordant + discordant)
            total_tau += tau
            num_samples += 1

    kendall_tau = total_tau / num_samples if num_samples > 0 else 0.0

    # ============ Relation Accuracy ============
    relation_correct = 0
    relation_total = 0

    for b in range(batch_size):
        for i in range(num_regions):
            if not region_mask[b, i]:
                continue
            parent_idx = parent_ids[b, i].item()
            rel_type = relations[b, i].item()

            if parent_idx >= 0 and parent_idx < num_regions and rel_type >= 0:
                pred_rel = relation_logits[b, parent_idx, i].argmax().item()
                if pred_rel == rel_type:
                    relation_correct += 1
                relation_total += 1

    relation_acc = relation_correct / relation_total if relation_total > 0 else 0.0

    return {
        'pairwise_accuracy': pairwise_acc,
        'kendall_tau': kendall_tau,
        'relation_accuracy': relation_acc,
    }


def train_epoch(
    model: OrderModuleFromFeatures,
    loss_fn: OrderLossFromFeatures,
    dataloader,
    optimizer,
    scheduler,
    device,
    args,
    scaler=None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_order_loss = 0.0
    total_relation_loss = 0.0
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
        relations = batch['relations'].to(device)

        # Forward
        if args.fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    region_features=region_features,
                    categories=categories,
                    bboxes=bboxes,
                    region_mask=region_mask,
                )
                losses = loss_fn(
                    order_logits=outputs['order_logits'],
                    relation_logits=outputs['relation_logits'],
                    reading_orders=reading_orders,
                    parent_ids=parent_ids,
                    relations=relations,
                    region_mask=region_mask,
                )
                loss = losses['loss'] / args.gradient_accumulation_steps
        else:
            outputs = model(
                region_features=region_features,
                categories=categories,
                bboxes=bboxes,
                region_mask=region_mask,
            )
            losses = loss_fn(
                order_logits=outputs['order_logits'],
                relation_logits=outputs['relation_logits'],
                reading_orders=reading_orders,
                parent_ids=parent_ids,
                relations=relations,
                region_mask=region_mask,
            )
            loss = losses['loss'] / args.gradient_accumulation_steps

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

        total_loss += losses['loss'].item()
        total_order_loss += losses['order_loss'].item()
        total_relation_loss += losses['relation_loss'].item()
        num_batches += 1

        progress_bar.set_postfix({
            'loss': f"{losses['loss'].item():.4f}",
            'order': f"{losses['order_loss'].item():.4f}",
            'rel': f"{losses['relation_loss'].item():.4f}",
        })

    return {
        'loss': total_loss / num_batches,
        'order_loss': total_order_loss / num_batches,
        'relation_loss': total_relation_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: OrderModuleFromFeatures,
    loss_fn: OrderLossFromFeatures,
    dataloader,
    device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_order_loss = 0.0
    total_relation_loss = 0.0
    num_batches = 0

    all_metrics = {
        'pairwise_accuracy': 0.0,
        'kendall_tau': 0.0,
        'relation_accuracy': 0.0,
    }
    num_metric_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        region_features = batch['region_features'].to(device)
        categories = batch['categories'].to(device)
        bboxes = batch['bboxes'].to(device)
        region_mask = batch['region_mask'].to(device)
        reading_orders = batch['reading_orders'].to(device)
        parent_ids = batch['parent_ids'].to(device)
        relations = batch['relations'].to(device)

        outputs = model(
            region_features=region_features,
            categories=categories,
            bboxes=bboxes,
            region_mask=region_mask,
        )

        losses = loss_fn(
            order_logits=outputs['order_logits'],
            relation_logits=outputs['relation_logits'],
            reading_orders=reading_orders,
            parent_ids=parent_ids,
            relations=relations,
            region_mask=region_mask,
        )

        total_loss += losses['loss'].item()
        total_order_loss += losses['order_loss'].item()
        total_relation_loss += losses['relation_loss'].item()
        num_batches += 1

        # Compute metrics
        metrics = compute_metrics(
            order_logits=outputs['order_logits'],
            relation_logits=outputs['relation_logits'],
            reading_orders=reading_orders,
            parent_ids=parent_ids,
            relations=relations,
            region_mask=region_mask,
        )

        for k, v in metrics.items():
            all_metrics[k] += v
        num_metric_batches += 1

    result = {
        'loss': total_loss / num_batches,
        'order_loss': total_order_loss / num_batches,
        'relation_loss': total_relation_loss / num_batches,
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

    # Setup experiment directory
    artifact_root = get_artifact_path(args.env)
    exp_manager, exp_dir = ensure_experiment(
        artifact_root=artifact_root,
        exp=args.exp,
        new_exp=args.new_exp,
        name=args.exp_name or "Order Module Training",
        description=f"Train Order module (4.3) with lr={args.learning_rate}, epochs={args.num_epochs}",
        config=vars(args),
    )

    # Get stage output directory
    output_dir = Path(exp_manager.get_stage_dir(args.exp, "order", "comp_hrdoc"))
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Stage output directory: {output_dir}")

    # Mark stage as started
    exp_manager.mark_stage_started(args.exp, "order", "comp_hrdoc")

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
    logger.info("Building model...")
    model = build_order_from_features(
        hidden_size=args.hidden_size,
        num_categories=args.num_categories,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_spatial=args.use_spatial,
    )
    model = model.to(device)

    # Loss function
    loss_fn = OrderLossFromFeatures(
        order_weight=args.order_weight,
        relation_weight=args.relation_weight,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
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
    best_kendall_tau = -1.0

    for epoch in range(args.num_epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{args.num_epochs} =====")

        # Train
        train_metrics = train_epoch(
            model, loss_fn, train_loader, optimizer, scheduler, device, args, scaler
        )
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Order: {train_metrics['order_loss']:.4f}, "
            f"Relation: {train_metrics['relation_loss']:.4f}"
        )

        # Evaluate
        val_metrics = evaluate(model, loss_fn, val_loader, device)
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Order: {val_metrics['order_loss']:.4f}, "
            f"Relation: {val_metrics['relation_loss']:.4f}"
        )
        logger.info(
            f"Val - Pairwise Acc: {val_metrics['pairwise_accuracy']:.4f}, "
            f"Kendall Tau: {val_metrics['kendall_tau']:.4f}, "
            f"Relation Acc: {val_metrics['relation_accuracy']:.4f}"
        )

        # Save best model
        if val_metrics['kendall_tau'] > best_kendall_tau:
            best_kendall_tau = val_metrics['kendall_tau']
            best_path = output_dir / "best_model"
            save_order_model(model, str(best_path))
            logger.info(f"Saved best model (Kendall Tau={best_kendall_tau:.4f})")

            # Update experiment state with best metrics
            exp_manager.update_stage_state(
                args.exp, "order", "comp_hrdoc",
                best_checkpoint=str(best_path),
                metrics={
                    'best_kendall_tau': best_kendall_tau,
                    'best_pairwise_acc': val_metrics['pairwise_accuracy'],
                    'epoch': epoch + 1,
                },
            )

    # Save final model
    final_path = output_dir / "final_model"
    save_order_model(model, str(final_path))
    logger.info(f"Saved final model to {final_path}")

    # Mark stage as completed
    exp_manager.mark_stage_completed(
        args.exp, "order", "comp_hrdoc",
        best_checkpoint=str(output_dir / "best_model"),
        metrics={
            'best_kendall_tau': best_kendall_tau,
            'final_kendall_tau': val_metrics['kendall_tau'],
            'final_pairwise_acc': val_metrics['pairwise_accuracy'],
        },
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best Kendall Tau: {best_kendall_tau:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Training script for DOC (Detect-Order-Construct) model

Usage:
    python examples/comp_hrdoc/scripts/train_doc.py --env test
    python examples/comp_hrdoc/scripts/train_doc.py --env test --quick
    python examples/comp_hrdoc/scripts/train_doc.py --env test --new-exp
    python examples/comp_hrdoc/scripts/train_doc.py --env test --use-construct
"""

# ==================== GPU 设置（必须在 import torch 之前）====================
import os
import sys

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from examples.comp_hrdoc.utils.config import setup_environment
setup_environment()
# ==================== GPU 设置结束 ====================

import argparse
import logging
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from examples.comp_hrdoc.data.comp_hrdoc_loader import (
    CompHRDocConfig,
    CompHRDocDataset,
    CompHRDocCollator,
    CompHRDocDocumentCollator,
)
from examples.comp_hrdoc.models import (
    DOCModel,
    build_doc_model,
    save_doc_model,
    compute_order_accuracy,
    OrderOnlyModel,
    build_order_only_model,
    save_order_only_model,
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
    parser = argparse.ArgumentParser(description="Train DOC model")

    # Environment
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--config", type=str, default=None, help="Config file path")

    # Model
    parser.add_argument("--model-type", type=str, default="doc",
                        choices=["doc", "order-only"],
                        help="Model type: doc (full) or order-only")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--order-num-layers", type=int, default=3)
    parser.add_argument("--construct-num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-spatial", action="store_true", default=True)
    parser.add_argument("--no-spatial", dest="use_spatial", action="store_false")
    parser.add_argument("--use-construct", action="store_true", default=True)
    parser.add_argument("--no-construct", dest="use_construct", action="store_false")
    parser.add_argument("--use-semantic", action="store_true", default=False,
                        help="Enable 4.2 semantic classification for end-to-end training")
    parser.add_argument("--semantic-num-layers", type=int, default=1,
                        help="Number of Transformer layers in semantic classification module")
    parser.add_argument("--cls-weight", type=float, default=1.0,
                        help="Weight for classification loss")
    parser.add_argument("--order-weight", type=float, default=1.0,
                        help="Weight for order loss")
    parser.add_argument("--construct-weight", type=float, default=1.0,
                        help="Weight for construct loss")

    # Data
    parser.add_argument("--max-regions", type=int, default=128)
    parser.add_argument("--val-split-ratio", type=float, default=0.1)
    parser.add_argument("--document-level", action="store_true", default=False,
                        help="Enable document-level training (supports cross-page parent)")

    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=500)

    # Experiment management
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment identifier (None for current/latest)")
    parser.add_argument("--new-exp", action="store_true",
                        help="Create new experiment")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name (for new experiments)")

    # Quick test
    parser.add_argument("--quick", action="store_true", help="Quick test with small data")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    args,
    scaler=None,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_order_loss = 0.0
    total_construct_loss = 0.0
    num_batches = 0

    # Check if model uses semantic classification
    use_semantic = getattr(model, 'use_semantic', False)

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        # Move to device
        bboxes = batch["bboxes"].to(device)
        categories = batch["categories"].to(device)
        region_mask = batch["region_mask"].to(device)
        reading_orders = batch["reading_orders"].to(device)
        successor_labels = batch["successor_labels"].to(device)
        parent_ids = batch.get("parent_ids")
        if parent_ids is not None:
            parent_ids = parent_ids.to(device)
        sibling_labels = batch.get("sibling_labels")
        if sibling_labels is not None:
            sibling_labels = sibling_labels.to(device)

        # Normalize bboxes to [0, 999]
        bboxes = bboxes.clamp(0, 999)

        # Forward pass
        if args.fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    bbox=bboxes,
                    categories=categories,
                    region_mask=region_mask,
                    reading_orders=reading_orders,
                    successor_labels=successor_labels,
                    parent_labels=parent_ids,
                    sibling_labels=sibling_labels,
                    category_labels=categories if use_semantic else None,
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps
        else:
            outputs = model(
                bbox=bboxes,
                categories=categories,
                region_mask=region_mask,
                reading_orders=reading_orders,
                successor_labels=successor_labels,
                parent_labels=parent_ids,
                sibling_labels=sibling_labels,
                category_labels=categories if use_semantic else None,
            )
            loss = outputs["loss"] / args.gradient_accumulation_steps

        # Backward pass
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

            scheduler.step()
            optimizer.zero_grad()

        # Record losses
        total_loss += outputs["loss"].item()

        cls_loss = outputs.get("cls_loss", torch.tensor(0.0))
        if isinstance(cls_loss, torch.Tensor):
            cls_loss = cls_loss.item()
        total_cls_loss += cls_loss

        order_loss = outputs.get("order_loss", outputs.get("loss", torch.tensor(0.0)))
        if isinstance(order_loss, torch.Tensor):
            order_loss = order_loss.item()
        total_order_loss += order_loss

        construct_loss = outputs.get("construct_loss", torch.tensor(0.0))
        if isinstance(construct_loss, torch.Tensor):
            construct_loss = construct_loss.item()
        total_construct_loss += construct_loss

        num_batches += 1

        # Update progress bar
        postfix = {
            "loss": f"{outputs['loss'].item():.4f}",
            "order": f"{order_loss:.4f}",
        }
        if use_semantic:
            postfix["cls"] = f"{cls_loss:.4f}"
        progress_bar.set_postfix(postfix)

    return {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "order_loss": total_order_loss / num_batches,
        "construct_loss": total_construct_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model using DOCTask metrics

    计算完整的 Detect-Order-Construct 指标:
    - 4.2 Detect: Accuracy, Macro F1
    - 4.3 Order: Accuracy, F1
    - 4.4 Construct: Parent/Sibling/Root Accuracy, F1
    """
    from examples.comp_hrdoc.tasks import DOCTask
    import math

    model.eval()

    # Initialize metrics
    doc_task = DOCTask({'num_classes': 5})
    doc_task.reset_metrics()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_order_loss = 0.0
    total_construct_loss = 0.0
    num_batches = 0

    # Check if model uses semantic classification and construct
    use_semantic = getattr(model, 'use_semantic', False)
    use_construct = getattr(model, 'use_construct', False)

    for batch in tqdm(dataloader, desc="Evaluating"):
        bboxes = batch["bboxes"].to(device)
        categories = batch["categories"].to(device)
        region_mask = batch["region_mask"].to(device)
        reading_orders = batch["reading_orders"].to(device)
        successor_labels = batch["successor_labels"].to(device)
        parent_ids = batch.get("parent_ids")
        if parent_ids is not None:
            parent_ids = parent_ids.to(device)
        sibling_labels = batch.get("sibling_labels")
        if sibling_labels is not None:
            sibling_labels = sibling_labels.to(device)

        bboxes = bboxes.clamp(0, 999)

        outputs = model(
            bbox=bboxes,
            categories=categories,
            region_mask=region_mask,
            reading_orders=reading_orders,
            successor_labels=successor_labels,
            parent_labels=parent_ids,
            sibling_labels=sibling_labels,
            category_labels=categories if use_semantic else None,
        )

        # Accumulate losses
        loss_val = outputs["loss"].item()

        cls_loss = outputs.get("cls_loss", torch.tensor(0.0))
        if isinstance(cls_loss, torch.Tensor):
            cls_loss = cls_loss.item()
        total_cls_loss += cls_loss

        order_loss = outputs.get("order_loss", outputs.get("loss", torch.tensor(0.0)))
        if isinstance(order_loss, torch.Tensor):
            order_loss = order_loss.item()
        construct_loss = outputs.get("construct_loss", torch.tensor(0.0))
        if isinstance(construct_loss, torch.Tensor):
            construct_loss = construct_loss.item()

        # Skip NaN losses
        if math.isnan(loss_val) or math.isnan(construct_loss):
            logger.warning(f"NaN in eval batch {num_batches}: loss={loss_val}, order={order_loss}, construct={construct_loss}")
            if not math.isnan(order_loss):
                total_loss += order_loss
                total_order_loss += order_loss
        else:
            total_loss += loss_val
            total_order_loss += order_loss
            total_construct_loss += construct_loss

        # Prepare targets dict
        targets = {
            'categories': categories,
            'successor_labels': successor_labels,
            'parent_labels': parent_ids,
            'sibling_labels': sibling_labels,
        }

        # Update metrics using DOCTask
        doc_task.update_metrics(outputs, targets, region_mask)

        num_batches += 1

    # Compute final metrics
    metrics = doc_task.compute_metrics()

    # Add loss metrics
    metrics["loss"] = total_loss / max(num_batches, 1)
    metrics["cls_loss"] = total_cls_loss / max(num_batches, 1)
    metrics["order_loss"] = total_order_loss / max(num_batches, 1)
    metrics["construct_loss"] = total_construct_loss / max(num_batches, 1)

    return metrics


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override with config values (args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    logger.info(f"Environment: {args.env}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Arguments: {args}")

    # CUDA setup
    if torch.cuda.is_available():
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup experiment directory
    stage_name = "doc" if args.use_construct else "order"
    artifact_root = get_artifact_path(args.env)
    exp_manager, exp_dir = ensure_experiment(
        artifact_root=artifact_root,
        exp=args.exp,
        new_exp=args.new_exp,
        name=args.exp_name or f"DOC Model Training ({stage_name})",
        description=f"Train DOC model with lr={args.learning_rate}, epochs={args.num_epochs}, construct={args.use_construct}",
        config=vars(args),
    )

    # Get stage output directory
    output_dir = Path(exp_manager.get_stage_dir(args.exp, stage_name, "comp_hrdoc"))
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Stage output directory: {output_dir}")

    # Mark stage as started
    exp_manager.mark_stage_started(args.exp, stage_name, "comp_hrdoc")

    # Quick test mode
    if args.quick:
        logger.info("Quick test mode enabled")
        args.max_train_samples = args.max_train_samples or 50
        args.max_val_samples = args.max_val_samples or 20
        args.num_epochs = min(args.num_epochs, 2)

    # Create datasets
    mode = "document-level" if args.document_level else "page-level"
    logger.info(f"Creating datasets in {mode} mode...")
    data_config = CompHRDocConfig(
        env=args.env,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        val_split_ratio=args.val_split_ratio,
        use_images=False,
        document_level=args.document_level,
    )

    train_dataset = CompHRDocDataset(data_config, split="train")
    val_dataset = CompHRDocDataset(data_config, split="validation")

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Create dataloaders
    # 文档级别使用更大的 max_regions (默认 512)，页面级别使用 args.max_regions
    if args.document_level:
        max_regions = args.max_regions if args.max_regions > 128 else 512
        collator = CompHRDocDocumentCollator(max_regions=max_regions)
        logger.info(f"Using document-level collator with max_regions={max_regions}")
    else:
        collator = CompHRDocCollator(max_regions=args.max_regions)
        logger.info(f"Using page-level collator with max_regions={args.max_regions}")

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

    # Build model
    if args.model_type == "order-only":
        logger.info("Building Order-only model...")
        model = build_order_only_model(
            hidden_size=args.hidden_size,
            num_categories=5,
            num_heads=args.num_heads,
            num_layers=args.order_num_layers,
            dropout=args.dropout,
        )
        save_fn = save_order_only_model
    else:
        model_desc = "DOC model"
        if args.use_semantic:
            model_desc = "End-to-End DOC model (4.2 + 4.3 + 4.4)"
        elif args.use_construct:
            model_desc = "DOC model (4.3 + 4.4)"
        else:
            model_desc = "DOC model (4.3 only)"
        logger.info(f"Building {model_desc}...")

        model = build_doc_model(
            hidden_size=args.hidden_size,
            num_categories=5,
            num_heads=args.num_heads,
            order_num_layers=args.order_num_layers,
            construct_num_layers=args.construct_num_layers,
            dropout=args.dropout,
            use_spatial=args.use_spatial,
            use_construct=args.use_construct,
            use_semantic=args.use_semantic,
            semantic_num_layers=args.semantic_num_layers,
            cls_weight=args.cls_weight,
            order_weight=args.order_weight,
            construct_weight=args.construct_weight,
        )
        save_fn = save_doc_model

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=num_training_steps,
        pct_start=args.warmup_ratio,
    )

    # FP16 scaler
    scaler = None
    if args.fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    logger.info("Starting training...")
    best_order_acc = 0.0

    for epoch in range(args.num_epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{args.num_epochs} =====")

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            scaler=scaler,
        )
        train_log = (
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Order: {train_metrics['order_loss']:.4f}, "
            f"Construct: {train_metrics['construct_loss']:.4f}"
        )
        if args.use_semantic:
            train_log += f", Cls: {train_metrics['cls_loss']:.4f}"
        logger.info(train_log)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        val_log = (
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Order: {val_metrics['order_loss']:.4f}, "
            f"Construct: {val_metrics['construct_loss']:.4f}"
        )
        if args.use_semantic:
            val_log += f", Cls: {val_metrics['cls_loss']:.4f}"
        logger.info(val_log)

        # Log metrics for each module
        # 4.2 Detect
        if args.use_semantic and 'cls_accuracy' in val_metrics:
            logger.info(
                f"[4.2 Detect] Accuracy: {val_metrics['cls_accuracy']:.4f}, "
                f"Macro F1: {val_metrics.get('cls_macro_f1', 0):.4f}"
            )

        # 4.3 Order
        logger.info(
            f"[4.3 Order] Accuracy: {val_metrics['order_accuracy']:.4f}, "
            f"F1: {val_metrics['order_f1']:.4f}"
        )

        # 4.4 Construct
        if args.use_construct and 'parent_accuracy' in val_metrics:
            logger.info(
                f"[4.4 Construct] Parent Acc: {val_metrics['parent_accuracy']:.4f}, "
                f"Parent F1: {val_metrics.get('parent_f1', 0):.4f}, "
                f"Sibling Acc: {val_metrics.get('sibling_accuracy', 0):.4f}, "
                f"Root Acc: {val_metrics.get('root_accuracy', 0):.4f}"
            )

        # Save best model
        if val_metrics['order_accuracy'] > best_order_acc:
            best_order_acc = val_metrics['order_accuracy']
            best_path = output_dir / "best_model"
            save_fn(model, str(best_path))
            logger.info(f"Saved best model to {best_path}")

    # Save final model
    final_path = output_dir / "final_model"
    save_fn(model, str(final_path))
    logger.info(f"Saved final model to {final_path}")

    # Mark stage as completed
    exp_manager.mark_stage_completed(
        args.exp, stage_name, "comp_hrdoc",
        best_checkpoint=str(output_dir / "best_model"),
        metrics={"best_order_accuracy": best_order_acc},
    )

    logger.info("Training complete!")
    logger.info(f"Best order accuracy: {best_order_acc:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

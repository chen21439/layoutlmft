#!/usr/bin/env python
"""Training script for DOC (Detect-Order-Construct) model

Usage:
    python examples/comp_hrdoc/scripts/train_doc.py --env test
    python examples/comp_hrdoc/scripts/train_doc.py --env test --quick
    python examples/comp_hrdoc/scripts/train_doc.py --env test --order-only
"""

# ==================== GPU 设置（必须在 import torch 之前）====================
import os
import sys


def _setup_gpu_early():
    """在 import torch 之前设置 GPU"""
    env = "dev"
    for i, arg in enumerate(sys.argv):
        if arg == "--env" and i + 1 < len(sys.argv):
            env = sys.argv[i + 1]
            break

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "order.yaml"
    )

    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        gpu_config = config.get('gpu', {})
        cuda_visible_devices = gpu_config.get(env)

        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
            print(f"[GPU Setup] env={env}, CUDA_VISIBLE_DEVICES={cuda_visible_devices}")


_setup_gpu_early()
# ==================== GPU 设置结束 ====================

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.comp_hrdoc.data.comp_hrdoc_loader import (
    CompHRDocConfig,
    CompHRDocDataset,
    CompHRDocCollator,
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

    # Data
    parser.add_argument("--max-regions", type=int, default=128)
    parser.add_argument("--val-split-ratio", type=float, default=0.1)

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
    parser.add_argument("--output-dir", type=str, default="outputs/doc_model")

    # Quick test
    parser.add_argument("--quick", action="store_true", help="Quick test with small data")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file"""
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
    total_order_loss = 0.0
    total_construct_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        # Move to device
        bboxes = batch["bboxes"].to(device)
        categories = batch["categories"].to(device)
        region_mask = batch["region_mask"].to(device)
        reading_orders = batch["reading_orders"].to(device)
        parent_ids = batch.get("parent_ids")
        if parent_ids is not None:
            parent_ids = parent_ids.to(device)

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
                    parent_labels=parent_ids,
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps
        else:
            outputs = model(
                bbox=bboxes,
                categories=categories,
                region_mask=region_mask,
                reading_orders=reading_orders,
                parent_labels=parent_ids,
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
        progress_bar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "order": f"{order_loss:.4f}",
        })

    return {
        "loss": total_loss / num_batches,
        "order_loss": total_order_loss / num_batches,
        "construct_loss": total_construct_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()

    total_loss = 0.0
    total_order_loss = 0.0
    total_construct_loss = 0.0
    total_correct = 0
    total_count = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        bboxes = batch["bboxes"].to(device)
        categories = batch["categories"].to(device)
        region_mask = batch["region_mask"].to(device)
        reading_orders = batch["reading_orders"].to(device)
        parent_ids = batch.get("parent_ids")
        if parent_ids is not None:
            parent_ids = parent_ids.to(device)

        bboxes = bboxes.clamp(0, 999)

        outputs = model(
            bbox=bboxes,
            categories=categories,
            region_mask=region_mask,
            reading_orders=reading_orders,
            parent_labels=parent_ids,
        )

        total_loss += outputs["loss"].item()
        order_loss = outputs.get("order_loss", outputs.get("loss", torch.tensor(0.0)))
        if isinstance(order_loss, torch.Tensor):
            order_loss = order_loss.item()
        total_order_loss += order_loss

        construct_loss = outputs.get("construct_loss", torch.tensor(0.0))
        if isinstance(construct_loss, torch.Tensor):
            construct_loss = construct_loss.item()
        total_construct_loss += construct_loss

        # Compute order predictions
        order_logits = outputs["order_logits"]
        from examples.comp_hrdoc.models.order import predict_reading_order
        pred_orders = predict_reading_order(order_logits, region_mask)

        # Accumulate accuracy stats per batch
        correct = (pred_orders == reading_orders) & region_mask
        total_correct += correct.sum().item()
        total_count += region_mask.sum().item()

        num_batches += 1

    # Compute accuracy
    order_acc = total_correct / max(total_count, 1)

    return {
        "loss": total_loss / num_batches,
        "order_loss": total_order_loss / num_batches,
        "construct_loss": total_construct_loss / num_batches,
        "order_accuracy": order_acc,
    }


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

    # Quick test mode
    if args.quick:
        logger.info("Quick test mode enabled")
        args.max_train_samples = args.max_train_samples or 50
        args.max_val_samples = args.max_val_samples or 20
        args.num_epochs = min(args.num_epochs, 2)

    # Create datasets
    logger.info("Creating datasets...")
    data_config = CompHRDocConfig(
        env=args.env,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        val_split_ratio=args.val_split_ratio,
        use_images=False,
    )

    train_dataset = CompHRDocDataset(data_config, split="train")
    val_dataset = CompHRDocDataset(data_config, split="validation")

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Create dataloaders
    collator = CompHRDocCollator(max_regions=args.max_regions)
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
        logger.info("Building DOC model...")
        model = build_doc_model(
            hidden_size=args.hidden_size,
            num_categories=5,
            num_heads=args.num_heads,
            order_num_layers=args.order_num_layers,
            construct_num_layers=args.construct_num_layers,
            dropout=args.dropout,
            use_spatial=args.use_spatial,
            use_construct=args.use_construct,
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Order: {train_metrics['order_loss']:.4f}, "
            f"Construct: {train_metrics['construct_loss']:.4f}"
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Order: {val_metrics['order_loss']:.4f}, "
            f"Construct: {val_metrics['construct_loss']:.4f}"
        )
        logger.info(f"Val - Order Accuracy: {val_metrics['order_accuracy']:.4f}")

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

    logger.info("Training complete!")
    logger.info(f"Best order accuracy: {best_order_acc:.4f}")


if __name__ == "__main__":
    main()

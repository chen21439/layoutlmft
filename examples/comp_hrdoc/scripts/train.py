#!/usr/bin/env python
"""Order 模块训练脚本

基于 LayoutXLM 训练阅读顺序预测模型。
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from layoutlmft.models.layoutxlm import LayoutXLMTokenizerFast

# 添加 comp_hrdoc 路径
COMP_HRDOC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP_HRDOC_ROOT)

from data.dataset import OrderDataset, DataConfig
from data.collator import OrderDataCollator
from models.build import build_order_model, save_model

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Order 模块训练")

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/layoutxlm-base",
        help="LayoutXLM 预训练模型路径",
    )
    parser.add_argument("--num_labels", type=int, default=16, help="分类标签数")

    # Order 模块参数
    parser.add_argument("--num_layers", type=int, default=3, help="Order Transformer 层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--use_biaffine", action="store_true", default=True, help="使用双仿射变换")

    # 损失权重
    parser.add_argument("--lambda_cls", type=float, default=1.0, help="分类损失权重")
    parser.add_argument("--lambda_order", type=float, default=1.0, help="Order 损失权重")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--max_train_samples", type=int, default=None, help="最大训练样本数")
    parser.add_argument("--max_val_samples", type=int, default=None, help="最大验证样本数")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="outputs/order_model", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 比例")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--fp16", action="store_true", help="使用混合精度训练")
    parser.add_argument("--freeze_backbone", action="store_true", help="冻结 backbone")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--log_steps", type=int, default=100, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存检查点步数")

    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    args,
    scaler=None,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_order_loss = 0.0
    num_steps = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        # 移动数据到设备
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        line_ids = batch["line_ids"].to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)

        reading_order = batch.get("reading_order")
        if reading_order is not None:
            reading_order = reading_order.to(device)

        line_mask = batch.get("line_mask")
        if line_mask is not None:
            line_mask = line_mask.to(device)

        image = batch.get("image")
        if image is not None:
            image = image.to(device)

        num_docs = batch["num_docs"]
        chunks_per_doc = batch["chunks_per_doc"]

        # 前向传播
        if args.fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    bbox=bbox,
                    attention_mask=attention_mask,
                    line_ids=line_ids,
                    image=image,
                    labels=labels,
                    reading_order=reading_order,
                    line_mask=line_mask,
                    num_docs=num_docs,
                    chunks_per_doc=chunks_per_doc,
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps
        else:
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                line_ids=line_ids,
                image=image,
                labels=labels,
                reading_order=reading_order,
                line_mask=line_mask,
                num_docs=num_docs,
                chunks_per_doc=chunks_per_doc,
            )
            loss = outputs["loss"] / args.gradient_accumulation_steps

        # 反向传播
        if args.fp16 and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16 and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            num_steps += 1

        # 记录损失
        total_loss += outputs["loss"].item()
        total_cls_loss += outputs["cls_loss"].item()
        total_order_loss += outputs["order_loss"].item()

        # 更新进度条
        progress_bar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "cls": f"{outputs['cls_loss'].item():.4f}",
            "order": f"{outputs['order_loss'].item():.4f}",
        })

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "order_loss": total_order_loss / num_batches,
    }


def evaluate(model, dataloader, device) -> Dict[str, float]:
    """评估模型"""
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_order_loss = 0.0
    total_order_correct = 0
    total_order_pairs = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            line_ids = batch["line_ids"].to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            reading_order = batch.get("reading_order")
            if reading_order is not None:
                reading_order = reading_order.to(device)

            line_mask = batch.get("line_mask")
            if line_mask is not None:
                line_mask = line_mask.to(device)

            image = batch.get("image")
            if image is not None:
                image = image.to(device)

            num_docs = batch["num_docs"]
            chunks_per_doc = batch["chunks_per_doc"]

            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                line_ids=line_ids,
                image=image,
                labels=labels,
                reading_order=reading_order,
                line_mask=line_mask,
                num_docs=num_docs,
                chunks_per_doc=chunks_per_doc,
            )

            total_loss += outputs["loss"].item()
            total_cls_loss += outputs["cls_loss"].item()
            total_order_loss += outputs["order_loss"].item()

            # 计算阅读顺序准确率
            if reading_order is not None and "order_logits" in outputs:
                order_logits = outputs["order_logits"]
                # 预测：logits > 0 表示 i 在 j 之前
                predictions = (order_logits > 0).float()

                # Ground truth
                order_i = reading_order.unsqueeze(2)
                order_j = reading_order.unsqueeze(1)
                targets = (order_i < order_j).float()

                # 只计算有效位置
                if line_mask is not None:
                    valid_mask = line_mask.unsqueeze(2) & line_mask.unsqueeze(1)
                    num_lines = order_logits.shape[1]
                    diag_mask = ~torch.eye(num_lines, dtype=torch.bool, device=device).unsqueeze(0)
                    valid_mask = valid_mask & diag_mask

                    correct = ((predictions == targets) & valid_mask).sum().item()
                    total_order_correct += correct
                    total_order_pairs += valid_mask.sum().item()

    num_batches = len(dataloader)
    metrics = {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "order_loss": total_order_loss / num_batches,
    }

    if total_order_pairs > 0:
        metrics["order_accuracy"] = total_order_correct / total_order_pairs

    return metrics


def main():
    args = parse_args()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 快速测试模式
    if args.quick:
        args.max_train_samples = 10
        args.max_val_samples = 5
        args.num_epochs = 2
        args.log_steps = 1
        logger.info("Quick test mode enabled")

    # 设置随机种子
    set_seed(args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(args.model_path)

    # 创建数据集
    logger.info("Creating datasets...")
    data_config = DataConfig(
        data_dir=args.data_dir,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    train_dataset = OrderDataset(tokenizer, data_config, split="train")
    val_dataset = OrderDataset(tokenizer, data_config, split="validation")

    logger.info(f"Train dataset: {len(train_dataset)} documents")
    logger.info(f"Validation dataset: {len(val_dataset)} documents")

    # 创建 data collator
    collator = OrderDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    # 创建 dataloader
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

    # 构建模型
    logger.info("Building model...")
    model = build_order_model(
        model_path=args.model_path,
        num_labels=args.num_labels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lambda_cls=args.lambda_cls,
        lambda_order=args.lambda_order,
        freeze_backbone=args.freeze_backbone,
        use_biaffine=args.use_biaffine,
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 混合精度
    scaler = None
    if args.fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using FP16 mixed precision")

    # 训练循环
    logger.info("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{args.num_epochs} =====")

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, args, scaler
        )
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"CLS: {train_metrics['cls_loss']:.4f}, "
            f"Order: {train_metrics['order_loss']:.4f}"
        )

        # 评估
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"CLS: {val_metrics['cls_loss']:.4f}, "
            f"Order: {val_metrics['order_loss']:.4f}"
        )
        if "order_accuracy" in val_metrics:
            logger.info(f"Val - Order Accuracy: {val_metrics['order_accuracy']:.4f}")

        # 保存最佳模型
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = os.path.join(args.output_dir, "best_model")
            save_model(model, save_path)
            logger.info(f"Saved best model to {save_path}")

    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final_model")
    save_model(model, final_path)
    logger.info(f"Saved final model to {final_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

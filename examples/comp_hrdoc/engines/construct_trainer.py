#!/usr/bin/env python
# coding=utf-8
"""
Construct 训练器 - 负责训练/评估循环

遵循项目结构规范：
- engines/ 负责运行骨架（训练/推理/评估的循环与工程能力）
- 不包含任务损失细节、模型结构细节
"""

import os
import logging
from typing import Dict, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader,
    feature_extractor,
    optimizer: Optimizer,
    scheduler: Optional = None,
    device: torch.device = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    compute_cls_loss: bool = False,
    lambda_cls: float = 1.0,
) -> Dict[str, float]:
    """
    训练一个 epoch（Stage1 + Construct 联合训练）

    Args:
        model: ConstructFromFeatures 模型
        dataloader: 训练数据加载器
        feature_extractor: StageFeatureExtractor 实例
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        device: 计算设备
        gradient_accumulation_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪阈值
        scaler: AMP scaler（如果使用混合精度）
        use_amp: 是否使用自动混合精度
        compute_cls_loss: 是否计算 Stage1 分类损失
        lambda_cls: 分类损失权重

    Returns:
        Dict 包含平均 loss 和其他指标
    """
    feature_extractor.model.train()
    model.train()

    total_loss = 0.0
    total_construct_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        # ==================== 特征提取 ====================
        # 梯度自动流通（因为 model.train()）
        line_features, line_mask = feature_extractor.extract_features(
            input_ids=batch["input_ids"],
            bbox=batch["bbox"],
            attention_mask=batch["attention_mask"],
            line_ids=batch.get("line_ids"),
            image=batch.get("image"),
            num_docs=batch.get("num_docs"),
            chunks_per_doc=batch.get("chunks_per_doc"),
        )

        # ==================== Stage1 分类损失（可选）====================
        cls_loss = torch.tensor(0.0, device=device)
        if compute_cls_loss and hasattr(feature_extractor.model, 'cls_head'):
            line_labels = batch.get("line_labels")
            if line_labels is not None:
                line_labels = line_labels.to(device)
                cls_logits = feature_extractor.model.cls_head(line_features)
                # 计算分类损失（只计算有效行）
                valid_mask = line_mask.bool()
                if valid_mask.any():
                    cls_logits_flat = cls_logits[valid_mask]
                    line_labels_flat = line_labels[valid_mask]
                    cls_loss = nn.CrossEntropyLoss()(cls_logits_flat, line_labels_flat)

        # ==================== Construct 前向 ====================
        # 准备 Construct 输入（截断到 max_lines）
        max_lines = line_mask.size(1)  # 实际的 max_lines (受 max_regions 限制)

        # 截断 categories 到 max_lines
        categories = batch.get("line_labels", torch.zeros_like(line_mask, dtype=torch.long))
        if categories.size(1) > max_lines:
            categories = categories[:, :max_lines]
        categories = categories.to(device)

        # 截断 reading_orders 到 max_lines
        reading_orders = batch.get("reading_orders", torch.zeros_like(line_mask, dtype=torch.long))
        if reading_orders.size(1) > max_lines:
            reading_orders = reading_orders[:, :max_lines]
        reading_orders = reading_orders.to(device)

        # 截断 parent_labels 到 max_lines
        parent_labels = batch.get("parent_labels")
        if parent_labels is not None:
            if parent_labels.size(1) > max_lines:
                parent_labels = parent_labels[:, :max_lines]
            parent_labels = parent_labels.to(device)

        # 截断 sibling_labels 到 max_lines
        sibling_labels = batch.get("sibling_labels")
        if sibling_labels is not None:
            if sibling_labels.size(1) > max_lines:
                sibling_labels = sibling_labels[:, :max_lines]
            sibling_labels = sibling_labels.to(device)

        # Forward
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    region_features=line_features,
                    categories=categories,
                    region_mask=line_mask,
                    reading_orders=reading_orders,
                    parent_labels=parent_labels,
                    sibling_labels=sibling_labels,
                )
                construct_loss = outputs["loss"]
                loss = (construct_loss + lambda_cls * cls_loss) / gradient_accumulation_steps
        else:
            outputs = model(
                region_features=line_features,
                categories=categories,
                region_mask=line_mask,
                reading_orders=reading_orders,
                parent_labels=parent_labels,
                sibling_labels=sibling_labels,
            )
            construct_loss = outputs["loss"]
            loss = (construct_loss + lambda_cls * cls_loss) / gradient_accumulation_steps

        # ==================== 反向传播 ====================
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ==================== 梯度累积 & 优化器步进 ====================
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(feature_extractor.model.parameters()) + list(model.parameters()),
                    max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(feature_extractor.model.parameters()) + list(model.parameters()),
                    max_grad_norm
                )
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        # ==================== 统计 ====================
        total_loss += loss.item() * gradient_accumulation_steps
        total_construct_loss += construct_loss.item()
        if compute_cls_loss:
            total_cls_loss += cls_loss.item()
        num_batches += 1

        # 更新进度条
        progress_bar.set_postfix({
            "loss": total_loss / num_batches,
            "construct": total_construct_loss / num_batches,
            "cls": total_cls_loss / num_batches if compute_cls_loss else 0.0,
        })

    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "construct_loss": total_construct_loss / num_batches if num_batches > 0 else 0.0,
        "cls_loss": total_cls_loss / num_batches if num_batches > 0 and compute_cls_loss else 0.0,
    }


def evaluate(
    model: nn.Module,
    dataloader,
    feature_extractor,
    device: torch.device,
) -> Dict[str, float]:
    """
    评估模型

    Args:
        model: ConstructFromFeatures 模型
        dataloader: 验证数据加载器
        feature_extractor: StageFeatureExtractor 实例
        device: 计算设备

    Returns:
        Dict 包含评估指标
    """
    feature_extractor.model.eval()
    model.eval()

    total_loss = 0.0
    total_parent_correct = 0
    total_sibling_correct = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 特征提取（无梯度）
            line_features, line_mask = feature_extractor.extract_features(
                input_ids=batch["input_ids"],
                bbox=batch["bbox"],
                attention_mask=batch["attention_mask"],
                line_ids=batch.get("line_ids"),
                image=batch.get("image"),
                num_docs=batch.get("num_docs"),
                chunks_per_doc=batch.get("chunks_per_doc"),
            )

            # 准备输入（截断到 max_lines）
            max_lines = line_mask.size(1)  # 实际的 max_lines (受 max_regions 限制)

            # 截断 categories 到 max_lines
            categories = batch.get("line_labels", torch.zeros_like(line_mask, dtype=torch.long))
            if categories.size(1) > max_lines:
                categories = categories[:, :max_lines]
            categories = categories.to(device)

            # 截断 reading_orders 到 max_lines
            reading_orders = batch.get("reading_orders", torch.zeros_like(line_mask, dtype=torch.long))
            if reading_orders.size(1) > max_lines:
                reading_orders = reading_orders[:, :max_lines]
            reading_orders = reading_orders.to(device)

            # 截断 parent_labels 到 max_lines
            parent_labels = batch.get("parent_labels")
            if parent_labels is not None:
                if parent_labels.size(1) > max_lines:
                    parent_labels = parent_labels[:, :max_lines]
                parent_labels = parent_labels.to(device)

            # 截断 sibling_labels 到 max_lines
            sibling_labels = batch.get("sibling_labels")
            if sibling_labels is not None:
                if sibling_labels.size(1) > max_lines:
                    sibling_labels = sibling_labels[:, :max_lines]
                sibling_labels = sibling_labels.to(device)

            # Forward
            outputs = model(
                region_features=line_features,
                categories=categories,
                region_mask=line_mask,
                reading_orders=reading_orders,
                parent_labels=parent_labels,
                sibling_labels=sibling_labels,
            )

            total_loss += outputs["loss"].item()
            num_batches += 1

            # 计算准确率
            if parent_labels is not None and "parent_logits" in outputs:
                parent_preds = outputs["parent_logits"].argmax(dim=-1)
                parent_correct = (parent_preds == parent_labels) & line_mask.bool()
                total_parent_correct += parent_correct.sum().item()

            if sibling_labels is not None and "sibling_logits" in outputs:
                sibling_preds = outputs["sibling_logits"].argmax(dim=-1)
                sibling_correct = (sibling_preds == sibling_labels) & line_mask.bool()
                total_sibling_correct += sibling_correct.sum().item()

            total_samples += line_mask.sum().item()

    metrics = {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "parent_accuracy": total_parent_correct / total_samples if total_samples > 0 else 0.0,
        "sibling_accuracy": total_sibling_correct / total_samples if total_samples > 0 else 0.0,
    }

    return metrics


def save_model(
    construct_model: nn.Module,
    save_path: str,
    feature_extractor,
    tokenizer,
) -> None:
    """
    保存 Stage1 + Construct 联合模型

    Args:
        construct_model: ConstructFromFeatures 模型
        save_path: 保存路径
        feature_extractor: StageFeatureExtractor 实例
        tokenizer: 分词器
    """
    os.makedirs(save_path, exist_ok=True)

    # 1. 保存 Construct 权重
    construct_path = os.path.join(save_path, "pytorch_model.bin")
    torch.save(construct_model.state_dict(), construct_path)
    logger.info(f"Saved construct to: {construct_path}")

    # 2. 保存 backbone（HuggingFace 格式）
    backbone_path = os.path.join(save_path, "stage1")
    os.makedirs(backbone_path, exist_ok=True)
    feature_extractor.model.backbone.save_pretrained(backbone_path)
    logger.info(f"Saved backbone to: {backbone_path}")

    # 3. 保存 cls_head
    if hasattr(feature_extractor.model, 'cls_head') and feature_extractor.model.cls_head is not None:
        cls_head_path = os.path.join(save_path, "cls_head.pt")
        torch.save(feature_extractor.model.cls_head.state_dict(), cls_head_path)
        logger.info(f"Saved cls_head to: {cls_head_path}")

    # 4. 保存 line_enhancer
    if hasattr(feature_extractor.model, 'line_enhancer') and feature_extractor.model.line_enhancer is not None:
        line_enhancer_path = os.path.join(save_path, "line_enhancer.pt")
        torch.save(feature_extractor.model.line_enhancer.state_dict(), line_enhancer_path)
        logger.info(f"Saved line_enhancer to: {line_enhancer_path}")

    # 5. 保存 tokenizer
    tokenizer.save_pretrained(save_path)
    logger.info(f"Saved tokenizer to: {save_path}")

    logger.info(f"Model saved successfully to: {save_path}")

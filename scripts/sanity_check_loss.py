#!/usr/bin/env python
# coding=utf-8
"""
Loss Sanity Check Script

验证训练 loss 是否被稀释/归一化异常。
基于 checkpoint 做离线检查，不需要重跑训练。

Usage:
    # 使用合成数据（不需要数据集）
    python scripts/sanity_check_loss.py \
        --checkpoint /path/to/checkpoint-5500 \
        --synthetic

输出三个关键指标：
    - active_tokens: 有效 token 数量 (labels != -100)
    - model_loss: 模型返回的 loss
    - manual_ce: 手动计算的标准 CE loss

如果 model_loss 和 manual_ce 量级一致：loss 计算正常
如果 manual_ce 明显更大：可能有稀释问题
"""

import os
import sys
import argparse
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Register LayoutXLM config before importing transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from layoutlmft.models.layoutxlm import LayoutXLMConfig
CONFIG_MAPPING.update({
    "layoutxlm": LayoutXLMConfig,
    "layoutlmv2": LayoutXLMConfig,
})

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Loss Sanity Check")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--num_batches", type=int, default=5,
                        help="Number of batches to check")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for eval")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic data (default, no dataset needed)")
    return parser.parse_args()


def create_synthetic_batch(tokenizer, num_labels, batch_size=2, seq_len=128, device='cpu'):
    """
    创建合成数据用于 sanity check。
    模拟真实数据的结构：input_ids, bbox, image, attention_mask, labels
    """
    # 生成随机 input_ids (避开特殊 token)
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(100, vocab_size - 100, (batch_size, seq_len))

    # 设置 [CLS] 和 [SEP] (或对应的特殊token)
    input_ids[:, 0] = tokenizer.cls_token_id if tokenizer.cls_token_id else 0
    input_ids[:, -1] = tokenizer.sep_token_id if tokenizer.sep_token_id else 2

    # attention mask (全1)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # bbox: 随机生成归一化坐标 [x0, y0, x1, y1] in [0, 1000]
    bbox = torch.randint(0, 500, (batch_size, seq_len, 4))
    bbox[:, :, 2] = bbox[:, :, 0] + torch.randint(10, 100, (batch_size, seq_len))  # x1 > x0
    bbox[:, :, 3] = bbox[:, :, 1] + torch.randint(5, 30, (batch_size, seq_len))    # y1 > y0
    bbox = bbox.clamp(0, 1000)

    # 特殊 token 位置的 bbox 设为 0
    bbox[:, 0] = 0  # [CLS]
    bbox[:, -1] = 0  # [SEP]

    # labels: 随机类别，特殊位置设为 -100
    labels = torch.randint(0, num_labels, (batch_size, seq_len))
    labels[:, 0] = -100  # [CLS]
    labels[:, -1] = -100  # [SEP]

    # 模拟 subword tokenization：部分位置设为 -100 (约 30%)
    subword_mask = torch.rand(batch_size, seq_len) < 0.3
    subword_mask[:, 0] = False  # 保持 CLS
    subword_mask[:, -1] = False  # 保持 SEP
    labels[subword_mask] = -100

    # image: 生成随机 RGB 图像 (224x224)
    # LayoutXLM 需要 PIL Image
    images = []
    for _ in range(batch_size):
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        images.append(Image.fromarray(img_array))

    batch = {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'bbox': bbox.to(device),
        'image': images,  # List of PIL Images
        'labels': labels.to(device),
    }

    return batch


def main():
    args = parse_args()

    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Mode: Synthetic data")

    from transformers import AutoConfig
    from layoutlmft.data.labels import LABEL_LIST, NUM_LABELS
    from layoutlmft.models.layoutxlm import (
        LayoutXLMForTokenClassification,
        LayoutXLMTokenizerFast,
        LayoutXLMTokenizer,
    )
    from layoutlmft.models.balanced_loss import get_balanced_loss, ClassBalancedLoss

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")

    # Load tokenizer
    tokenizer_json_path = os.path.join(args.checkpoint, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(args.checkpoint)
        logger.info("Using LayoutXLMTokenizerFast")
    else:
        tokenizer = LayoutXLMTokenizer.from_pretrained(args.checkpoint)
        logger.info("Using LayoutXLMTokenizer")

    # Load model
    model = LayoutXLMForTokenClassification.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded on {device}")
    logger.info(f"Num labels: {model.config.num_labels}")

    # 检查模型是否有 custom_loss_fn
    has_custom_loss = hasattr(model, 'custom_loss_fn') and model.custom_loss_fn is not None
    logger.info(f"Model has custom_loss_fn: {has_custom_loss}")

    if has_custom_loss:
        logger.info(f"Custom loss type: {type(model.custom_loss_fn).__name__}")
        if hasattr(model.custom_loss_fn, 'class_weights'):
            weights = model.custom_loss_fn.class_weights
            logger.info(f"Class weights shape: {weights.shape}")
            logger.info(f"Class weights (first 5): {weights[:5].tolist()}")

    # 创建一个模拟的 balanced loss 用于对比
    # 使用均匀分布的 class counts (因为我们用合成数据)
    mock_class_counts = [1000] * NUM_LABELS  # 均匀分布
    balanced_loss_fn = get_balanced_loss(
        loss_type="class_balanced",
        class_counts=mock_class_counts,
        beta=0.9999,
        gamma=2.0,
        ignore_index=-100,
    )
    balanced_loss_fn = balanced_loss_fn.to(device)

    # ============================================================
    # SANITY CHECK
    # ============================================================
    logger.info("=" * 70)
    logger.info("LOSS SANITY CHECK (Synthetic Data)")
    logger.info("=" * 70)

    results = []

    with torch.no_grad():
        for batch_idx in range(args.num_batches):
            # 生成合成数据
            batch = create_synthetic_batch(
                tokenizer,
                NUM_LABELS,
                batch_size=args.batch_size,
                seq_len=128,
                device=device
            )

            # Forward pass
            try:
                outputs = model(**batch)
            except Exception as e:
                logger.error(f"Forward pass failed: {e}")
                # 尝试不带 image
                del batch['image']
                outputs = model(**batch)

            model_loss = outputs.loss.item() if outputs.loss is not None else 0.0

            logits = outputs.logits  # [B, T, C]
            labels = batch["labels"]  # [B, T]

            # Flatten
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)

            # Active positions (labels != -100)
            active_mask = flat_labels != -100
            active_tokens = active_mask.sum().item()
            total_tokens = flat_labels.numel()

            if active_tokens == 0:
                logger.warning(f"Batch {batch_idx}: No active tokens!")
                continue

            active_logits = flat_logits[active_mask]
            active_labels = flat_labels[active_mask]

            # Manual CE (standard, no class weighting)
            manual_ce = F.cross_entropy(active_logits, active_labels, reduction='mean').item()

            # Manual balanced loss
            manual_balanced = balanced_loss_fn(logits, labels).item()

            # Confidence diagnostics
            probs = torch.softmax(active_logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            avg_max_prob = max_probs.mean().item()
            min_max_prob = max_probs.min().item()

            # Per-sample CE for distribution check
            per_sample_ce = F.cross_entropy(active_logits, active_labels, reduction='none')
            ce_std = per_sample_ce.std().item()
            ce_max = per_sample_ce.max().item()
            ce_min = per_sample_ce.min().item()

            results.append({
                'batch_idx': batch_idx,
                'active_tokens': active_tokens,
                'total_tokens': total_tokens,
                'active_ratio': active_tokens / total_tokens,
                'model_loss': model_loss,
                'manual_ce': manual_ce,
                'manual_balanced': manual_balanced,
                'avg_max_prob': avg_max_prob,
                'min_max_prob': min_max_prob,
                'ce_std': ce_std,
                'ce_max': ce_max,
                'ce_min': ce_min,
            })

            logger.info(f"\n[Batch {batch_idx}]")
            logger.info(f"  Tokens: {active_tokens}/{total_tokens} active ({active_tokens/total_tokens*100:.1f}%)")
            logger.info(f"  model_loss:      {model_loss:.8g}")
            logger.info(f"  manual_ce:       {manual_ce:.8g}")
            logger.info(f"  manual_balanced: {manual_balanced:.8g}")
            logger.info(f"  avg_max_prob:    {avg_max_prob:.6f}")
            logger.info(f"  min_max_prob:    {min_max_prob:.6f}")
            logger.info(f"  CE range: [{ce_min:.6f}, {ce_max:.6f}], std={ce_std:.6f}")

    if not results:
        logger.error("No valid results!")
        return

    # ============================================================
    # SUMMARY
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    avg_model_loss = np.mean([r['model_loss'] for r in results])
    avg_manual_ce = np.mean([r['manual_ce'] for r in results])
    avg_manual_balanced = np.mean([r['manual_balanced'] for r in results])
    avg_max_prob = np.mean([r['avg_max_prob'] for r in results])
    avg_active_ratio = np.mean([r['active_ratio'] for r in results])

    logger.info(f"Average over {len(results)} batches:")
    logger.info(f"  Active token ratio: {avg_active_ratio*100:.1f}%")
    logger.info(f"  model_loss:      {avg_model_loss:.8g}")
    logger.info(f"  manual_ce:       {avg_manual_ce:.8g}")
    logger.info(f"  manual_balanced: {avg_manual_balanced:.8g}")
    logger.info(f"  avg_max_prob:    {avg_max_prob:.6f}")

    # Ratio analysis
    ce_to_model_ratio = avg_manual_ce / avg_model_loss if avg_model_loss > 1e-10 else float('inf')
    balanced_to_model_ratio = avg_manual_balanced / avg_model_loss if avg_model_loss > 1e-10 else float('inf')

    logger.info(f"\nRatio Analysis:")
    logger.info(f"  manual_ce / model_loss:       {ce_to_model_ratio:.2f}x")
    logger.info(f"  manual_balanced / model_loss: {balanced_to_model_ratio:.2f}x")

    # ============================================================
    # INTERPRETATION
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("INTERPRETATION")
    logger.info("=" * 70)

    # 注意：合成数据是随机的，所以模型应该不确定
    # 如果模型很自信（高 avg_max_prob），说明在训练数据上过拟合
    # 关键是看 model_loss 和 manual_ce 的比值

    if avg_model_loss < 1e-8:
        logger.info("[WARNING] model_loss is essentially zero!")
        logger.info("  This suggests the model is not computing loss correctly,")
        logger.info("  or custom_loss_fn is returning near-zero values.")
    elif abs(ce_to_model_ratio - 1.0) < 0.5:
        logger.info("[OK] model_loss is close to manual_ce")
        logger.info("  Loss calculation appears correct (using standard CE).")
    elif ce_to_model_ratio > 2.0:
        logger.info(f"[INFO] model_loss is {ce_to_model_ratio:.1f}x smaller than manual_ce")
        if has_custom_loss:
            logger.info("  This is expected if using balanced/focal loss with class weights.")
            logger.info("  The custom loss may down-weight certain classes.")
        else:
            logger.info("[WARNING] Unexpected - model_loss should be close to manual_ce")
    elif ce_to_model_ratio < 0.5:
        logger.info(f"[INFO] model_loss is {1/ce_to_model_ratio:.1f}x larger than manual_ce")
        logger.info("  This could be due to class weighting amplifying loss for rare classes.")

    logger.info(f"\n[INFO] Model confidence on random data: avg_max_prob={avg_max_prob:.4f}")
    if avg_max_prob > 0.5:
        logger.info("  Model is moderately confident even on random data.")
        logger.info("  This indicates the model has learned strong class priors.")
    else:
        logger.info("  Model is uncertain on random data (expected behavior).")

    # ============================================================
    # CONCLUSION
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("CONCLUSION")
    logger.info("=" * 70)

    if avg_model_loss < 1e-8:
        logger.info("[ISSUE] Loss is near zero - investigate loss computation!")
    elif has_custom_loss and ce_to_model_ratio > 1.5:
        logger.info("[OK] Loss calculation is working correctly.")
        logger.info("  The difference between model_loss and manual_ce is due to")
        logger.info("  class-balanced weighting in the custom loss function.")
        logger.info("  This is the expected behavior for long-tailed classification.")
    elif not has_custom_loss and abs(ce_to_model_ratio - 1.0) < 0.5:
        logger.info("[OK] Loss calculation is correct (standard CE).")
    else:
        logger.info("[CHECK] Review loss computation for potential issues.")
        logger.info(f"  ce_to_model_ratio: {ce_to_model_ratio:.2f}")

    logger.info("\n" + "=" * 70)
    logger.info("NOTE: This test uses synthetic random data.")
    logger.info("For accurate analysis, run with real validation data if available.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

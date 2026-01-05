#!/usr/bin/env python
# coding=utf-8
"""
诊断 hrdh document-level 数据预测问题

检查 stage1 模型在 hrdh dev 数据上的预测是否全是 class 6
"""

import os
import sys
import logging

PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)
STAGE_ROOT = os.path.join(PROJECT_ROOT, "examples", "stage")
sys.path.insert(0, STAGE_ROOT)

import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForTokenClassification

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hrdh")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--document_level", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Document level: {args.document_level}")

    # 加载模型
    logger.info("Loading model...")
    model = LayoutLMv2ForTokenClassification.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # 加载数据
    logger.info("Loading dataset...")
    if args.document_level:
        # Document-level: 使用缓存
        cache_dir = os.path.expanduser("~/.cache/hrdoc_doc_level")
        cache_file = os.path.join(cache_dir, f"{args.dataset}_dev.pkl")
        logger.info(f"Looking for cache: {cache_file}")

        if os.path.exists(cache_file):
            import pickle
            with open(cache_file, "rb") as f:
                dataset = pickle.load(f)
            logger.info(f"Loaded from cache: {len(dataset)} documents")
        else:
            logger.error(f"Cache file not found: {cache_file}")
            return
    else:
        # Page-level
        dataset = load_dataset(
            "/root/code/layoutlmft/examples/stage/data",
            args.dataset,
            split="dev",
            trust_remote_code=True,
        )
        logger.info(f"Loaded page-level dataset: {len(dataset)} pages")

    # 检查数据内容
    if args.document_level:
        # Document-level 数据结构
        doc_names = list(dataset.keys())[:args.max_samples]
        logger.info(f"\nChecking {len(doc_names)} documents...")

        all_preds = []
        all_labels = []

        for doc_name in doc_names:
            doc_data = dataset[doc_name]
            pages = doc_data.get("pages", [])

            logger.info(f"\nDocument: {doc_name}, {len(pages)} pages")

            for page_idx, page in enumerate(pages[:2]):  # 只看前2页
                input_ids = torch.tensor(page["input_ids"]).unsqueeze(0).to(device)
                bbox = torch.tensor(page["bbox"]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(page["attention_mask"]).unsqueeze(0).to(device)

                # 图像处理
                if "image" in page:
                    image = page["image"]
                    if isinstance(image, torch.Tensor):
                        image = image.unsqueeze(0).to(device)
                    else:
                        # 跳过没有正确图像的页面
                        logger.info(f"  Page {page_idx}: skipped (no proper image)")
                        continue
                else:
                    logger.info(f"  Page {page_idx}: skipped (no image)")
                    continue

                labels = page.get("labels", None)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        bbox=bbox,
                        attention_mask=attention_mask,
                        image=image,
                    )

                logits = outputs.logits
                preds = logits.argmax(dim=-1)[0].cpu().tolist()

                # 只看非 padding 部分
                mask = attention_mask[0].cpu().tolist()
                valid_preds = [p for p, m in zip(preds, mask) if m == 1]

                if labels is not None:
                    valid_labels = [l for l, m in zip(labels, mask) if m == 1 and l != -100]
                    all_labels.extend(valid_labels)

                all_preds.extend(valid_preds)

                logger.info(f"  Page {page_idx}: preds[:20] = {valid_preds[:20]}")

        # 统计预测分布
        pred_counter = Counter(all_preds)
        logger.info(f"\n=== Prediction Distribution ===")
        for cls, count in sorted(pred_counter.items()):
            pct = count / len(all_preds) * 100
            logger.info(f"  Class {cls}: {count} ({pct:.1f}%)")

        if all_labels:
            label_counter = Counter(all_labels)
            logger.info(f"\n=== Label Distribution ===")
            for cls, count in sorted(label_counter.items()):
                pct = count / len(all_labels) * 100
                logger.info(f"  Class {cls}: {count} ({pct:.1f}%)")

    else:
        # Page-level 数据
        from data import create_data_collator
        collator = create_data_collator(tokenizer, document_level=False)

        all_preds = []
        all_labels = []

        for idx in range(min(args.max_samples, len(dataset))):
            sample = dataset[idx]
            batch = collator([sample])

            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    bbox=batch["bbox"],
                    attention_mask=batch["attention_mask"],
                    image=batch["image"],
                )

            logits = outputs.logits
            preds = logits.argmax(dim=-1)[0].cpu().tolist()

            mask = batch["attention_mask"][0].cpu().tolist()
            valid_preds = [p for p, m in zip(preds, mask) if m == 1]

            if "labels" in batch:
                labels = batch["labels"][0].cpu().tolist()
                valid_labels = [l for l, m in zip(labels, mask) if m == 1 and l != -100]
                all_labels.extend(valid_labels)

            all_preds.extend(valid_preds)

            logger.info(f"Sample {idx}: preds[:20] = {valid_preds[:20]}")

        # 统计预测分布
        pred_counter = Counter(all_preds)
        logger.info(f"\n=== Prediction Distribution ===")
        for cls, count in sorted(pred_counter.items()):
            pct = count / len(all_preds) * 100
            logger.info(f"  Class {cls}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

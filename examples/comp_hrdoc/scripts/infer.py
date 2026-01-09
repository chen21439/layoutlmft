#!/usr/bin/env python
# coding=utf-8
"""
Inference Script for comp_hrdoc - Construct Task Only

只运行 construct 任务，输出 TOC 树结构。

Usage:
    python examples/comp_hrdoc/scripts/infer.py \
        --env test \
        --checkpoint /path/to/checkpoint \
        --data_dir /path/to/data \
        --output_dir /path/to/output

Data directory structure:
    data_dir/
    ├── *.json           # 文档 JSON 文件
    └── images/          # 图片目录
        └── {doc_name}/
            ├── 0.png
            └── 1.png
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import torch

from examples.comp_hrdoc.models import load_doc_model
from examples.comp_hrdoc.data.hrdoc_loader import HRDocDataset, HRDocCollator
from examples.comp_hrdoc.utils.tree_utils import build_tree_from_parents, format_tree_from_parents

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Construct Inference Script")

    parser.add_argument("--env", type=str, default="test",
                        help="Environment: dev or test")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint directory")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory (must have JSON files and images/ subdir)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for predictions")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples (-1 for all)")
    parser.add_argument("--max_lines", type=int, default=256,
                        help="Max lines per document")

    return parser.parse_args()


def decode_parent_from_logits(parent_logits: torch.Tensor, mask: torch.Tensor = None):
    """从 parent_logits 解码得到 toc_parent

    Args:
        parent_logits: [num_regions, num_regions] 父节点预测 logits
        mask: [num_regions] 有效区域掩码

    Returns:
        parents: List[int] 每个节点的父节点索引（自指向表示 root）
    """
    num_regions = parent_logits.size(0)

    # 对每个节点，选择概率最大的父节点
    # parent_logits[i, j] 表示节点 i 的父节点是 j 的概率
    pred_parents = parent_logits.argmax(dim=-1)  # [num_regions]

    parents = []
    for i in range(num_regions):
        if mask is not None and not mask[i]:
            continue
        parent_idx = pred_parents[i].item()
        parents.append(parent_idx)

    return parents


def run_inference(
    model,
    dataloader,
    output_dir: str,
    device: str,
):
    """运行推理并保存结果"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    total_docs = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        bboxes = batch['bboxes'].to(device)
        region_ids = batch['region_ids'].to(device)
        line_mask = batch['line_mask'].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model.predict(
                bbox=bboxes,
                region_ids=region_ids,
                region_mask=line_mask,
            )

        batch_size = bboxes.size(0)

        for b in range(batch_size):
            # 获取该样本的有效区域数
            valid_mask = line_mask[b]
            num_valid = valid_mask.sum().item()

            # 获取文本列表和文档名
            texts = batch['texts'][b][:num_valid]
            doc_name = batch['doc_names'][b]

            # 解码 parent
            parent_logits = outputs['parent_logits'][b]
            parents = decode_parent_from_logits(parent_logits, valid_mask)

            # 构建 predictions 列表
            predictions = []
            for i, (text, parent) in enumerate(zip(texts, parents)):
                predictions.append({
                    "line_id": i,
                    "text": text,
                    "toc_parent": parent,
                })

            # 构建嵌套树结构
            toc_tree = build_tree_from_parents(predictions)

            # 构建结果
            result = {
                "document_name": doc_name,
                "num_sections": len(predictions),
                "predictions": predictions,
                "toc_tree": toc_tree,
            }

            # 保存到文件
            output_path = os.path.join(output_dir, f"{doc_name}_construct.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # 打印 TOC 树
            tree_lines = format_tree_from_parents(parents, texts)
            logger.info(f"\n{'='*60}\n TOC Tree ({doc_name})\n{'='*60}\n" + "\n".join(tree_lines))

            total_docs += 1

    logger.info(f"Inference completed. Total documents: {total_docs}")
    logger.info(f"Results saved to: {output_dir}")


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model = load_doc_model(args.checkpoint, device=device)
    model.eval()
    logger.info("Model loaded successfully")

    # 2. Load data
    logger.info(f"Loading data from: {args.data_dir}")

    # 检查数据目录结构
    data_path = Path(args.data_dir)

    # 支持两种目录结构：
    # 1. data_dir 直接包含 JSON 文件
    # 2. data_dir/train/ 包含 JSON 文件
    dataset = HRDocDataset(
        data_dir=str(data_path),
        max_lines=args.max_lines,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        split='train',
        val_split_ratio=0.0,  # 推理时加载所有数据
    )

    if len(dataset) == 0:
        logger.error("No data loaded!")
        sys.exit(1)

    collator = HRDocCollator(max_lines=args.max_lines)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    logger.info(f"Dataset loaded: {len(dataset)} documents")

    # 3. Run inference
    logger.info("Running inference...")
    run_inference(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()

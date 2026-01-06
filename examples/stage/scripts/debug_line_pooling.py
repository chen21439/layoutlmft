#!/usr/bin/env python
# coding=utf-8
"""
诊断 LinePooling 的 line_mask False 问题

检查为什么 line_mask 中会有 False 值
"""

import os
import sys

PROJECT_ROOT = "/data/LLM_group/layoutlmft"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples", "stage"))

import torch
import pickle
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def test_line_pooling_basic():
    """测试 LinePooling 基本行为"""
    from models.modules.line_pooling import LinePooling

    pooling = LinePooling()

    # 测试 1: 简单连续 line_id
    logger.info("=== 测试 1: 连续 line_id ===")
    hidden = torch.randn(1, 20, 768)
    line_ids = torch.tensor([[
        -1, 0, 0, 1, 2, 3, 4, -1, 5, 6,
        7, 8, 9, -1, -1, -1, -1, -1, -1, -1
    ]])
    features, mask = pooling(hidden, line_ids)
    logger.info(f"line_ids unique: {line_ids[line_ids >= 0].unique(sorted=True).tolist()}")
    logger.info(f"features.shape[0]: {features.shape[0]}")
    logger.info(f"mask: {mask.tolist()}")
    logger.info(f"mask.sum(): {mask.sum().item()}")
    assert mask.sum().item() == features.shape[0], "mask 应该全是 True!"

    # 测试 2: 不连续 line_id
    logger.info("\n=== 测试 2: 不连续 line_id ===")
    line_ids2 = torch.tensor([[
        -1, 0, 0, 2, 2, 5, 5, -1, 10, 10,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    ]])
    features2, mask2 = pooling(hidden, line_ids2)
    logger.info(f"line_ids unique: {line_ids2[line_ids2 >= 0].unique(sorted=True).tolist()}")
    logger.info(f"features2.shape[0]: {features2.shape[0]}")
    logger.info(f"mask2: {mask2.tolist()}")
    logger.info(f"mask2.sum(): {mask2.sum().item()}")
    assert mask2.sum().item() == features2.shape[0], "mask 应该全是 True!"

    # 测试 3: 多 chunk 场景
    logger.info("\n=== 测试 3: 多 chunk ===")
    hidden3 = torch.randn(3, 10, 768)  # 3 chunks
    line_ids3 = torch.tensor([
        [-1, 0, 0, 1, 1, 2, -1, -1, -1, -1],  # chunk 0
        [-1, 2, 3, 3, 4, 4, -1, -1, -1, -1],  # chunk 1
        [-1, 5, 5, 6, 6, 7, -1, -1, -1, -1],  # chunk 2
    ])
    features3, mask3 = pooling(hidden3, line_ids3)
    flat_ids = line_ids3.reshape(-1)
    logger.info(f"line_ids unique: {flat_ids[flat_ids >= 0].unique(sorted=True).tolist()}")
    logger.info(f"features3.shape[0]: {features3.shape[0]}")
    logger.info(f"mask3: {mask3.tolist()}")
    logger.info(f"mask3.sum(): {mask3.sum().item()}")
    assert mask3.sum().item() == features3.shape[0], "mask 应该全是 True!"

    logger.info("\n✓ 基本测试全部通过!")


def test_real_document():
    """测试真实文档 1704.02278"""
    logger.info("\n=== 测试真实文档 1704.02278 ===")

    # 加载缓存
    cache_file = os.path.expanduser("~/.cache/hrdoc_doc_level/hrdh_train.pkl")
    if not os.path.exists(cache_file):
        logger.error(f"缓存文件不存在: {cache_file}")
        return

    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    doc_name = "1704.02278"
    if doc_name not in cache:
        logger.error(f"文档 {doc_name} 不在缓存中")
        logger.info(f"可用文档: {list(cache.keys())[:10]}...")
        return

    doc_data = cache[doc_name]
    pages = doc_data.get("pages", [])
    logger.info(f"文档 {doc_name} 有 {len(pages)} 页")

    # 收集所有 line_ids
    all_line_ids = []
    for page_idx, page in enumerate(pages):
        if "line_ids" in page:
            line_ids = page["line_ids"]
            all_line_ids.append(torch.tensor(line_ids))
            valid_ids = [lid for lid in line_ids if lid >= 0]
            logger.info(f"  Page {page_idx}: line_ids 长度={len(line_ids)}, 有效数={len(valid_ids)}, "
                       f"范围=[{min(valid_ids) if valid_ids else 'N/A'}, {max(valid_ids) if valid_ids else 'N/A'}]")

    if not all_line_ids:
        logger.error("没有找到 line_ids!")
        return

    # 合并所有 chunks
    stacked_line_ids = torch.stack(all_line_ids)  # [num_chunks, seq_len]
    logger.info(f"\n合并后 line_ids shape: {stacked_line_ids.shape}")
    logger.info(f"line_ids max: {stacked_line_ids.max().item()}")

    # 分析 line_id 分布
    flat_ids = stacked_line_ids.reshape(-1)
    valid_ids = flat_ids[flat_ids >= 0]
    unique_ids = valid_ids.unique(sorted=True)

    logger.info(f"\n有效 token 数: {len(valid_ids)}")
    logger.info(f"唯一 line_id 数: {len(unique_ids)}")
    logger.info(f"line_id 范围: [{unique_ids.min().item()}, {unique_ids.max().item()}]")

    # 检查是否有间隙
    expected_range = set(range(unique_ids.max().item() + 1))
    actual_ids = set(unique_ids.tolist())
    missing = expected_range - actual_ids
    if missing:
        logger.warning(f"缺失的 line_id: {sorted(missing)[:20]}...")
        logger.warning(f"共缺失 {len(missing)} 个")
    else:
        logger.info("line_id 连续，无缺失")

    # 测试 LinePooling
    from models.modules.line_pooling import LinePooling
    pooling = LinePooling()

    # 创建假的 hidden states
    hidden = torch.randn(stacked_line_ids.shape[0], stacked_line_ids.shape[1], 768)

    logger.info("\n运行 LinePooling...")
    features, mask = pooling(hidden, stacked_line_ids)

    logger.info(f"features.shape[0]: {features.shape[0]}")
    logger.info(f"mask.sum(): {mask.sum().item()}")
    logger.info(f"mask 中 False 数量: {(~mask).sum().item()}")

    if mask.sum().item() < features.shape[0]:
        logger.error("发现问题: mask 中有 False!")
        zero_indices = (~mask).nonzero(as_tuple=True)[0]
        logger.error(f"False 的索引: {zero_indices.tolist()}")
    else:
        logger.info("✓ mask 全是 True，没有问题!")


def debug_scatter_add():
    """深入调试 scatter_add 行为"""
    logger.info("\n=== 调试 scatter_add ===")

    # 模拟 LinePooling 的 scatter_add 操作
    valid_line_ids = torch.tensor([0, 0, 1, 2, 2, 5, 5, 10])
    unique_line_ids = valid_line_ids.unique(sorted=True)
    num_lines = len(unique_line_ids)

    logger.info(f"valid_line_ids: {valid_line_ids.tolist()}")
    logger.info(f"unique_line_ids: {unique_line_ids.tolist()}")
    logger.info(f"num_lines: {num_lines}")

    # searchsorted 映射
    line_indices = torch.searchsorted(unique_line_ids, valid_line_ids)
    logger.info(f"line_indices: {line_indices.tolist()}")

    # scatter_add
    line_counts = torch.zeros(num_lines)
    line_counts.scatter_add_(0, line_indices, torch.ones_like(line_indices, dtype=torch.float))
    logger.info(f"line_counts: {line_counts.tolist()}")

    # 检查
    line_mask = line_counts > 0
    logger.info(f"line_mask: {line_mask.tolist()}")

    assert line_mask.all(), "应该全是 True!"
    logger.info("✓ scatter_add 行为正常!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                       choices=["basic", "real", "scatter", "all"])
    args = parser.parse_args()

    if args.test in ["basic", "all"]:
        test_line_pooling_basic()

    if args.test in ["scatter", "all"]:
        debug_scatter_add()

    if args.test in ["real", "all"]:
        test_real_document()

    logger.info("\n=== 测试完成 ===")

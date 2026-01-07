#!/usr/bin/env python
"""Order 模块训练脚本

基于 LayoutXLM 训练阅读顺序预测模型。
支持 Comp_HRDoc 数据集（论文作者提供）。
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

# ==================== GPU 设置（必须在 import torch 之前）====================
def _setup_gpu_early():
    """在 import torch 之前设置 GPU"""
    # 解析 --env 参数
    env = "dev"
    for i, arg in enumerate(sys.argv):
        if arg == "--env" and i + 1 < len(sys.argv):
            env = sys.argv[i + 1]
            break
        elif arg.startswith("--env="):
            env = arg.split("=")[1]
            break

    # 根据环境加载对应配置文件 (dev.yaml / test.yaml)
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", f"{env}.yaml"
    )
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            gpu_config = config.get('gpu', {})
            cuda_visible_devices = gpu_config.get('cuda_visible_devices')
            if cuda_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
                print(f"[GPU Setup] env={env}, CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        except Exception as e:
            print(f"[GPU Setup] Warning: Failed to load config: {e}")

_setup_gpu_early()
# ==================== GPU 设置结束 ====================

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

# 添加 comp_hrdoc 路径
COMP_HRDOC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP_HRDOC_ROOT)

from data.hrdoc_loader import HRDocDataset, HRDocLayoutXLMCollator
from models.order_only import build_order_only_model, save_order_only_model, OrderOnlyModel

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Order 模块训练")

    # 环境配置
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test"],
                        help="运行环境: dev(本地开发) 或 test(服务器)")
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh"],
                        help="数据集: hrds(简单) 或 hrdh(困难)")
    parser.add_argument("--covmatch", type=str, default=None,
                        help="Covmatch split directory name (e.g., 'doc_covmatch_dev10_seed42')")

    # 模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/layoutxlm-base",
        help="LayoutXLM 预训练模型路径",
    )
    parser.add_argument("--num_labels", type=int, default=4, help="分类标签数 (Comp_HRDoc: 4)")

    # Order 模块参数
    parser.add_argument("--num_layers", type=int, default=3, help="Order Transformer 层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--use_biaffine", action="store_true", default=True, help="使用双仿射变换")

    # 损失权重
    parser.add_argument("--lambda_cls", type=float, default=0.0, help="分类损失权重 (Order-only 设为 0)")
    parser.add_argument("--lambda_order", type=float, default=1.0, help="Order 损失权重")

    # 数据参数
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--max_regions", type=int, default=128, help="每页最大区域数")
    parser.add_argument("--max_train_samples", type=int, default=None, help="最大训练样本数")
    parser.add_argument("--max_val_samples", type=int, default=None, help="最大验证样本数")
    parser.add_argument("--val_split_ratio", type=float, default=0.1, help="验证集划分比例")
    parser.add_argument("--use_images", action="store_true", help="加载图像数据")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="outputs/order_comp_hrdoc", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 比例")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--fp16", action="store_true", help="使用混合精度训练")
    parser.add_argument("--freeze_backbone", action="store_true", help="冻结 backbone")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--log_steps", type=int, default=50, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存检查点步数")
    parser.add_argument("--eval_steps", type=int, default=200, help="评估间隔步数")

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


def normalize_bboxes(bboxes: torch.Tensor, max_val: int = 1000) -> torch.Tensor:
    """归一化 bbox 到 [0, max_val) 范围

    Args:
        bboxes: [batch, num_regions, 4] raw bbox coordinates
        max_val: 最大值

    Returns:
        归一化后的 bboxes
    """
    # 假设原始 bbox 已经是像素坐标，需要归一化到 [0, 1000)
    # 这里简单处理：如果值已经在合理范围内，直接 clamp
    return bboxes.clamp(0, max_val - 1)


def train_epoch(
    model: OrderOnlyModel,
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
    total_order_loss = 0.0
    num_steps = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        # 移动数据到设备
        bboxes = batch["bboxes"].to(device)
        categories = batch["categories"].to(device)
        region_mask = batch["region_mask"].to(device)
        reading_orders = batch["reading_orders"].to(device)

        # 归一化 bbox
        bboxes = normalize_bboxes(bboxes)

        # 前向传播
        if args.fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    bboxes=bboxes,
                    categories=categories,
                    region_mask=region_mask,
                    reading_orders=reading_orders,
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps
        else:
            outputs = model(
                bboxes=bboxes,
                categories=categories,
                region_mask=region_mask,
                reading_orders=reading_orders,
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
        total_order_loss += outputs["order_loss"].item()

        # 更新进度条
        progress_bar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "order": f"{outputs['order_loss'].item():.4f}",
        })

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "order_loss": total_order_loss / num_batches,
        "cls_loss": 0.0,  # 无分类损失
    }


def compute_kendall_tau(pred_order: torch.Tensor, gt_order: torch.Tensor, mask: torch.Tensor) -> float:
    """计算 Kendall Tau 相关性

    Args:
        pred_order: [N] 预测的顺序索引
        gt_order: [N] 真实的顺序索引
        mask: [N] 有效位置掩码

    Returns:
        Kendall Tau 相关系数 [-1, 1]
    """
    # 只考虑有效位置
    valid_indices = mask.nonzero().squeeze(-1)
    if len(valid_indices) < 2:
        return 1.0  # 无法计算，返回完美相关

    pred_valid = pred_order[valid_indices].cpu().numpy()
    gt_valid = gt_order[valid_indices].cpu().numpy()

    # 计算一致对和不一致对
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
            # ties 不计入

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 1.0

    return (concordant - discordant) / total_pairs


def evaluate(model: OrderOnlyModel, dataloader, device) -> Dict[str, float]:
    """评估模型"""
    model.eval()

    total_loss = 0.0
    total_order_loss = 0.0
    total_order_correct = 0
    total_order_pairs = 0

    # 额外指标
    total_kendall_tau = 0.0
    total_samples = 0
    perfect_order_count = 0  # 完全正确的样本数

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            bboxes = batch["bboxes"].to(device)
            categories = batch["categories"].to(device)
            region_mask = batch["region_mask"].to(device)
            reading_orders = batch["reading_orders"].to(device)

            # 归一化 bbox
            bboxes = normalize_bboxes(bboxes)

            outputs = model(
                bboxes=bboxes,
                categories=categories,
                region_mask=region_mask,
                reading_orders=reading_orders,
            )

            total_loss += outputs["loss"].item()
            total_order_loss += outputs["order_loss"].item()

            # 计算阅读顺序准确率 (pairwise)
            order_logits = outputs["order_logits"]
            predictions = (order_logits > 0).float()

            # Ground truth
            order_i = reading_orders.unsqueeze(2)
            order_j = reading_orders.unsqueeze(1)
            targets = (order_i < order_j).float()

            # 只计算有效位置
            valid_mask = region_mask.unsqueeze(2) & region_mask.unsqueeze(1)
            num_regions = order_logits.shape[1]
            diag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=device).unsqueeze(0)
            valid_mask = valid_mask & diag_mask

            correct = ((predictions == targets) & valid_mask).sum().item()
            total_order_correct += correct
            total_order_pairs += valid_mask.sum().item()

            # 计算额外指标（逐样本）
            batch_size = bboxes.shape[0]
            for b in range(batch_size):
                sample_mask = region_mask[b]
                sample_logits = order_logits[b]
                sample_gt = reading_orders[b]

                # 预测顺序：基于 pairwise 得分
                # 每个区域的得分 = 它比多少其他区域靠前
                scores = (sample_logits > 0).float().sum(dim=1)  # [N]
                _, pred_order_indices = scores.sort(descending=True)

                # 创建预测的顺序（0=第一个，1=第二个，...）
                pred_order = torch.zeros_like(scores, dtype=torch.long)
                for rank, idx in enumerate(pred_order_indices):
                    pred_order[idx] = rank

                # Kendall Tau
                tau = compute_kendall_tau(pred_order, sample_gt, sample_mask)
                total_kendall_tau += tau
                total_samples += 1

                # 检查是否完全正确
                valid_pred = pred_order[sample_mask]
                valid_gt = sample_gt[sample_mask]
                # 将 GT 转换为排名格式
                _, gt_sorted_indices = valid_gt.sort()
                gt_ranks = torch.zeros_like(valid_gt)
                for rank, idx in enumerate(gt_sorted_indices):
                    gt_ranks[idx] = rank

                if torch.equal(valid_pred, gt_ranks):
                    perfect_order_count += 1

    num_batches = len(dataloader)
    metrics = {
        "loss": total_loss / num_batches,
        "order_loss": total_order_loss / num_batches,
        "cls_loss": 0.0,
    }

    if total_order_pairs > 0:
        metrics["pairwise_accuracy"] = total_order_correct / total_order_pairs
        # 保留旧名称以兼容
        metrics["order_accuracy"] = metrics["pairwise_accuracy"]

    if total_samples > 0:
        metrics["kendall_tau"] = total_kendall_tau / total_samples
        metrics["perfect_order_rate"] = perfect_order_count / total_samples

    return metrics


def main():
    args = parse_args()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(f"Environment: {args.env}")
    logger.info(f"Arguments: {args}")

    # 显示 GPU 信息
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all available")
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # 快速测试模式
    if args.quick:
        args.max_train_samples = 50
        args.max_val_samples = 20
        args.num_epochs = 2
        args.log_steps = 5
        logger.info("Quick test mode enabled")

    # 设置随机种子
    set_seed(args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建数据集配置
    logger.info("Creating datasets...")
    from configs.config_loader import load_config as load_global_config
    from transformers import AutoTokenizer

    # 获取数据目录 (使用全局 config)
    global_config = load_global_config(args.env).get_effective_config()
    data_dir = global_config.dataset.get_data_dir(args.dataset)
    logger.info(f"Data directory: {data_dir}")

    # 获取 covmatch (命令行参数优先于配置文件)
    covmatch = args.covmatch or global_config.dataset.covmatch
    logger.info(f"Covmatch: {covmatch}")

    # 创建数据集
    train_dataset = HRDocDataset(
        data_dir=data_dir,
        dataset_name=args.dataset,
        max_lines=args.max_regions,
        max_samples=args.max_train_samples,
        split='train',
        val_split_ratio=args.val_split_ratio,
        covmatch=covmatch,
    )
    val_dataset = HRDocDataset(
        data_dir=data_dir,
        dataset_name=args.dataset,
        max_lines=args.max_regions,
        max_samples=args.max_val_samples,
        split='validation',
        val_split_ratio=args.val_split_ratio,
        covmatch=covmatch,
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # 加载 tokenizer
    model_name = global_config.model.local_path or global_config.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Loaded tokenizer from: {model_name}")

    # 创建 data collator
    collator = HRDocLayoutXLMCollator(
        tokenizer=tokenizer,
        max_length=512,
        max_lines=args.max_regions,
    )

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

    # 构建模型（使用简化版 Order-only 模型）
    logger.info("Building Order-only model...")
    model = build_order_only_model(
        hidden_size=768,
        num_categories=5,  # 0=padding, 1=fig, 2=tab, 3=para, 4=other
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.1,
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
        if "pairwise_accuracy" in val_metrics:
            logger.info(
                f"Val - Pairwise Acc: {val_metrics['pairwise_accuracy']:.4f}, "
                f"Kendall Tau: {val_metrics.get('kendall_tau', 0):.4f}, "
                f"Perfect Rate: {val_metrics.get('perfect_order_rate', 0):.4f}"
            )

        # 保存最佳模型
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = os.path.join(args.output_dir, "best_model")
            save_order_only_model(model, save_path)
            logger.info(f"Saved best model to {save_path}")

    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final_model")
    save_order_only_model(model, final_path)
    logger.info(f"Saved final model to {final_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

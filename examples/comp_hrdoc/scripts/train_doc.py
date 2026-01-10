#!/usr/bin/env python
"""Training script for DOC (Detect-Order-Construct) model

Usage:
    # 基础训练（使用 CompHRDoc 数据）
    python examples/comp_hrdoc/scripts/train_doc.py --env test
    python examples/comp_hrdoc/scripts/train_doc.py --env test --quick
    python examples/comp_hrdoc/scripts/train_doc.py --env test --new-exp

    # 使用 stage 模型特征训练 Construct（深度集成）
    python examples/comp_hrdoc/scripts/train_doc.py --env test --use-stage-features --stage-checkpoint /path/to/joint/checkpoint
    python examples/comp_hrdoc/scripts/train_doc.py --env test --use-stage-features --dataset hrds --quick
"""

# ==================== GPU 设置（必须在 import torch 之前）====================
import os
import sys

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 提前解析 --gpu 参数
if "--gpu" in sys.argv:
    gpu_idx = sys.argv.index("--gpu")
    if gpu_idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[gpu_idx + 1]

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

# 数据加载：复用 stage 的 DataLoader（支持多 chunk 文档级别处理）
from examples.stage.data.hrdoc_data_loader import HRDocDataLoader, HRDocDataLoaderConfig
from examples.stage.joint_data_collator import HRDocDocumentLevelCollator
from examples.comp_hrdoc.data.hrdoc_loader import HRDocDataset

# 标签转换：使用 tree_utils 进行格式A/B双向转换
# 格式A: ref_parent + relation (原始标注，顶层节点 parent=-1)
# 格式B: hierarchical_parent + sibling (训练标签，自指向方案)
from examples.comp_hrdoc.utils.tree_utils import (
    resolve_hierarchical_parents_and_siblings,  # 正向: A → B
    visualize_toc,
)
# 推理解码：复用 engines/predictor
from examples.comp_hrdoc.engines.predictor import (
    decode_construct_outputs,
    convert_to_format_a,
    build_predictions,
)
# 评估指标：复用 metrics/teds
from examples.comp_hrdoc.metrics.teds import TEDSMetric
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
    parser.add_argument("--max-regions", type=int, default=1024,
                        help="Max lines per document (O(n²) memory, default 1024)")
    parser.add_argument("--document-level", action="store_true", default=True,
                        help="Enable document-level training (supports cross-page parent)")

    # Training
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (default 1 for document-level)")
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

    # Stage feature integration
    parser.add_argument("--use-stage-features", action="store_true", default=False,
                        help="Use pre-trained stage model to extract line features for Construct training")
    parser.add_argument("--stage-checkpoint", type=str, default=None,
                        help="Path to stage joint model checkpoint (required if --use-stage-features)")
    parser.add_argument("--dataset", type=str, default="hrds", choices=["hrds", "hrdh", "tender"],
                        help="Dataset to use when --use-stage-features is enabled")
    parser.add_argument("--covmatch", type=str, default=None,
                        help="Covmatch split directory name (e.g., 'doc_covmatch_dev10_seed42')")
    parser.add_argument("--toc-only", action="store_true", default=True,
                        help="Only train on section headings (align with paper 4.4 TOC generation)")
    parser.add_argument("--section-label-id", type=int, default=4,
                        help="Label ID for section headings (default=4 per labels.py: section)")

    # Output
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Output directory (saves directly here, skips exp_manager)")

    # Model checkpoint (HuggingFace style)
    parser.add_argument("--model-name-or-path", type=str, default=None,
                        help="Path to pretrained Construct model checkpoint to load weights from")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume full training state (weights + optimizer + scheduler)")

    # GPU
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID(s) to use, e.g., '0' or '0,1' (sets CUDA_VISIBLE_DEVICES)")

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


# ==================== Stage Feature Training ====================

def convert_stage_labels_to_construct(
    batch: Dict,
    max_lines: int,
    device: torch.device,
) -> tuple:
    """将 stage collator 的输出转换为 Construct 模块需要的标签

    Args:
        batch: stage DocumentLevelCollator 的输出，包含:
            - line_parent_ids: [num_docs, max_lines] 原始 parent_id
            - line_relations: [num_docs, max_lines] 原始 relation
        max_lines: 最大行数（来自 feature_extractor 输出）
        device: 计算设备

    Returns:
        parent_ids: [num_docs, max_lines] 层级父节点（自指向方案）
        sibling_labels: [num_docs, max_lines] 左兄弟索引（-1 表示无左兄弟）
        class_labels: [num_docs, max_lines] 类别标签（如果有）
    """
    num_docs = batch["num_docs"]

    # 初始化输出 tensors
    parent_ids = torch.full((num_docs, max_lines), -1, dtype=torch.long, device=device)
    # sibling_labels: 索引形式 [B, N]，存储每个节点的左兄弟索引，-1 表示无左兄弟
    sibling_labels = torch.full((num_docs, max_lines), -1, dtype=torch.long, device=device)

    # 转换每个文档的标签
    raw_parent_ids = batch["line_parent_ids"]  # [num_docs, N]
    raw_relations = batch["line_relations"]    # [num_docs, N]

    for b in range(num_docs):
        # 获取原始标签（转为 list）
        raw_parents = raw_parent_ids[b].tolist()
        raw_rels = raw_relations[b].tolist()

        # 过滤 padding (-100)
        valid_len = 0
        for p in raw_parents:
            if p == -100:
                break
            valid_len += 1

        if valid_len == 0:
            continue

        raw_parents = raw_parents[:valid_len]
        raw_rels = raw_rels[:valid_len]

        # 使用 tree_utils 转换为层级 parent 和 sibling
        hier_parents, sibling_groups = resolve_hierarchical_parents_and_siblings(
            raw_parents, raw_rels
        )

        # 填充 parent_ids（自指向方案：root 节点 parent == self）
        for i, hp in enumerate(hier_parents):
            if i < max_lines:
                if hp == -1:
                    parent_ids[b, i] = i  # root 自指向
                else:
                    parent_ids[b, i] = hp

        # 填充 sibling_labels（索引形式：每个节点的左兄弟索引）
        # sibling_groups 是兄弟组列表，按节点索引排序后，后续节点的左兄弟是前一个
        for group in sibling_groups:
            sorted_group = sorted(group)  # 按索引排序
            for idx, node_idx in enumerate(sorted_group):
                if node_idx < max_lines:
                    if idx == 0:
                        sibling_labels[b, node_idx] = -1  # 第一个节点无左兄弟
                    else:
                        left_sibling = sorted_group[idx - 1]
                        sibling_labels[b, node_idx] = left_sibling

    # 提取 class_labels（如果有）
    class_labels = None
    if "line_labels" in batch:
        raw_labels = batch["line_labels"]  # [num_docs, N]
        class_labels = torch.full((num_docs, max_lines), -100, dtype=torch.long, device=device)
        for b in range(num_docs):
            n = min(raw_labels.shape[1], max_lines)
            class_labels[b, :n] = raw_labels[b, :n].to(device)

    return parent_ids, sibling_labels, class_labels


def train_epoch_with_stage_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    feature_extractor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    args,
    scaler=None,
) -> Dict[str, float]:
    """Train ConstructFromFeatures model using stage features.

    Args:
        model: ConstructFromFeatures model
        dataloader: HRDocDataLoader
        feature_extractor: StageFeatureExtractor instance
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device
        args: Arguments
        scaler: GradScaler for FP16
    """
    model.train()

    total_loss = 0.0
    total_parent_loss = 0.0
    total_sibling_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training (stage features)")

    for step, batch in enumerate(progress_bar):
        # Extract line features using stage model
        with torch.no_grad():
            line_features, line_mask = feature_extractor.extract_features(
                input_ids=batch["input_ids"],
                bbox=batch["bbox"],
                attention_mask=batch["attention_mask"],
                line_ids=batch.get("line_ids"),
                image=batch.get("image"),
                num_docs=batch.get("num_docs"),
                chunks_per_doc=batch.get("chunks_per_doc"),
            )

        # 从 stage collator 输出转换为 Construct 标签
        # 使用 tree_utils 处理 contain/connect/equality 关系
        _, max_lines_from_features = line_mask.shape
        line_parent_ids, sibling_labels_matrix, line_labels = convert_stage_labels_to_construct(
            batch, max_lines_from_features, device
        )

        # TOC-only mode: compress to section-only subgraph (align with paper 4.4)
        if args.toc_only and line_labels is not None:
            from examples.comp_hrdoc.utils.toc_compress import (
                compress_to_sections_batch,
                generate_sibling_labels_from_parents,
            )
            compressed = compress_to_sections_batch(
                line_features=line_features,
                line_mask=line_mask,
                parent_ids=line_parent_ids,
                line_labels=line_labels,
                reading_orders=None,  # will be generated based on section order
                section_label_id=args.section_label_id,
            )
            # Replace with compressed data
            line_features = compressed["features"]
            line_mask = compressed["mask"]
            line_parent_ids = compressed["parent_ids"]
            line_labels = compressed["categories"]
            reading_orders = compressed["reading_orders"]

            # Generate sibling labels from compressed parent_ids
            sibling_labels = generate_sibling_labels_from_parents(line_parent_ids, line_mask, reading_orders)

            # Skip batch if no sections
            if line_mask.sum() == 0:
                continue
        else:
            # Original logic: all lines
            # Get reading order (use line index as reading order for now)
            batch_size, max_lines = line_mask.shape
            reading_orders = torch.arange(max_lines, device=device).unsqueeze(0).expand(batch_size, -1)

            # 使用已转换的 sibling_labels_matrix
            sibling_labels = sibling_labels_matrix

        # Forward pass
        if args.fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    region_features=line_features,
                    categories=line_labels if line_labels is not None else torch.zeros_like(line_mask, dtype=torch.long),
                    region_mask=line_mask,
                    reading_orders=reading_orders,
                    parent_labels=line_parent_ids,
                    sibling_labels=sibling_labels,
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps
        else:
            outputs = model(
                region_features=line_features,
                categories=line_labels if line_labels is not None else torch.zeros_like(line_mask, dtype=torch.long),
                region_mask=line_mask,
                reading_orders=reading_orders,
                parent_labels=line_parent_ids,
                sibling_labels=sibling_labels,
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
        total_parent_loss += outputs.get("parent_loss", torch.tensor(0.0)).item()
        total_sibling_loss += outputs.get("sibling_loss", torch.tensor(0.0)).item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "parent": f"{outputs.get('parent_loss', torch.tensor(0.0)).item():.4f}",
            "sibling": f"{outputs.get('sibling_loss', torch.tensor(0.0)).item():.4f}",
        })

    return {
        "loss": total_loss / max(num_batches, 1),
        "parent_loss": total_parent_loss / max(num_batches, 1),
        "sibling_loss": total_sibling_loss / max(num_batches, 1),
    }


def compute_prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """计算 Precision, Recall, F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate_with_stage_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    feature_extractor,
    device: torch.device,
    args=None,
    visualize_samples: int = 0,  # 可视化的样本数量
) -> Dict[str, float]:
    """Evaluate ConstructFromFeatures model using stage features."""
    from examples.comp_hrdoc.models.construct_only import compute_construct_metrics
    from examples.comp_hrdoc.metrics.teds import HAS_APTED

    model.eval()

    total_loss = 0.0
    total_parent_loss = 0.0
    total_sibling_loss = 0.0
    num_batches = 0

    # Aggregate metrics
    all_parent_correct = 0
    all_parent_total = 0
    all_root_correct = 0
    all_root_total = 0

    # Sibling metrics (TP/FP/FN for P/R/F1)
    sibling_tp = 0
    sibling_fp = 0
    sibling_fn = 0

    # TEDS - 复用 metrics/teds.TEDSMetric
    teds_metric = TEDSMetric() if HAS_APTED else None

    # 用于可视化的样本
    vis_samples = []

    # Check toc_only mode
    toc_only = args.toc_only if args else False
    section_label_id = args.section_label_id if args else 1

    for batch in tqdm(dataloader, desc="Evaluating (stage features)"):
        # Extract line features
        line_features, line_mask = feature_extractor.extract_features(
            input_ids=batch["input_ids"],
            bbox=batch["bbox"],
            attention_mask=batch["attention_mask"],
            line_ids=batch.get("line_ids"),
            image=batch.get("image"),
            num_docs=batch.get("num_docs"),
            chunks_per_doc=batch.get("chunks_per_doc"),
        )

        # 从 stage collator 输出转换为 Construct 标签
        _, max_lines_from_features = line_mask.shape
        line_parent_ids, sibling_labels_matrix, line_labels = convert_stage_labels_to_construct(
            batch, max_lines_from_features, device
        )

        # TOC-only mode: compress to section-only subgraph
        if toc_only and line_labels is not None:
            from examples.comp_hrdoc.utils.toc_compress import (
                compress_to_sections_batch,
                generate_sibling_labels_from_parents,
            )
            compressed = compress_to_sections_batch(
                line_features=line_features,
                line_mask=line_mask,
                parent_ids=line_parent_ids,
                line_labels=line_labels,
                reading_orders=None,
                section_label_id=section_label_id,
            )
            line_features = compressed["features"]
            line_mask = compressed["mask"]
            line_parent_ids = compressed["parent_ids"]
            line_labels = compressed["categories"]
            reading_orders = compressed["reading_orders"]
            original_indices = compressed["original_indices"]  # 原始 line_id
            sibling_labels = generate_sibling_labels_from_parents(line_parent_ids, line_mask, reading_orders)

            # Skip batch if no sections
            if line_mask.sum() == 0:
                continue
        else:
            batch_size, max_lines = line_mask.shape
            reading_orders = torch.arange(max_lines, device=device).unsqueeze(0).expand(batch_size, -1)
            original_indices = None  # 非 toc_only 模式，索引不变

            # 使用已转换的 sibling_labels_matrix
            sibling_labels = sibling_labels_matrix

        outputs = model(
            region_features=line_features,
            categories=line_labels if line_labels is not None else torch.zeros_like(line_mask, dtype=torch.long),
            region_mask=line_mask,
            reading_orders=reading_orders,
            parent_labels=line_parent_ids,
            sibling_labels=sibling_labels,
        )

        # Accumulate losses
        total_loss += outputs["loss"].item()
        total_parent_loss += outputs.get("parent_loss", torch.tensor(0.0)).item()
        total_sibling_loss += outputs.get("sibling_loss", torch.tensor(0.0)).item()

        # Compute metrics
        if line_parent_ids is not None:
            # Aggregate parent metrics
            has_parent = (line_parent_ids >= 0) & line_mask
            pred_parents = outputs["parent_logits"].argmax(dim=-1)
            correct = (pred_parents == line_parent_ids) & has_parent
            all_parent_correct += correct.sum().item()
            all_parent_total += has_parent.sum().item()

            # Sibling metrics (accuracy for N选1)
            if sibling_labels is not None and "sibling_logits" in outputs:
                pred_siblings = outputs["sibling_logits"].argmax(dim=-1)  # [B, N]
                has_sibling = (sibling_labels >= 0) & line_mask
                correct_sib = (pred_siblings == sibling_labels) & has_sibling
                sibling_tp += correct_sib.sum().item()
                sibling_fp += (has_sibling.sum().item() - correct_sib.sum().item())  # For accuracy calc
                sibling_fn += 0  # Not used for accuracy

            # TEDS 计算（每个样本）- 使用 TEDSMetric
            if teds_metric is not None:
                batch_size = line_parent_ids.shape[0]
                for b in range(batch_size):
                    mask_b = line_mask[b].cpu().tolist()
                    pred_parents_b = pred_parents[b].cpu().tolist()
                    gt_parents_b = line_parent_ids[b].cpu().tolist()

                    # 只取有效节点
                    valid_indices = [i for i, m in enumerate(mask_b) if m]
                    if len(valid_indices) == 0:
                        continue

                    # 构建树（使用实际文本或占位符）
                    text_map = batch.get("line_text_maps", [{}])[b] if "line_text_maps" in batch else {}
                    # toc_only 模式下用 original_indices 获取原始 line_id
                    if original_indices is not None:
                        orig_ids = [original_indices[b, i].item() for i in valid_indices]
                        texts = [text_map.get(oid, f"node_{oid}") for oid in orig_ids]
                    else:
                        texts = [text_map.get(i, f"node_{i}") for i in valid_indices]

                    # 重新映射 parent_ids 到压缩后的索引
                    idx_map = {old: new for new, old in enumerate(valid_indices)}
                    idx_map[-1] = -1  # root 保持 -1

                    pred_parents_mapped = []
                    gt_parents_mapped = []
                    for i in valid_indices:
                        p_pred = pred_parents_b[i]
                        p_gt = gt_parents_b[i]
                        pred_parents_mapped.append(idx_map.get(p_pred, -1))
                        gt_parents_mapped.append(idx_map.get(p_gt, -1))

                    # 获取 sibling 预测和标签（用于反向转换）
                    pred_siblings_b = outputs["sibling_logits"].argmax(dim=-1)[b].cpu().tolist() if "sibling_logits" in outputs else None
                    gt_siblings_b = sibling_labels[b].cpu().tolist() if sibling_labels is not None else None

                    # 提取有效节点的 sibling 并映射索引
                    # 注意：无效的 sibling 应该映射为当前节点的压缩索引（自指向）
                    pred_siblings_mapped = None
                    gt_siblings_mapped = None
                    if pred_siblings_b is not None:
                        pred_siblings_mapped = []
                        for new_idx, old_idx in enumerate(valid_indices):
                            old_sibling = pred_siblings_b[old_idx]
                            if old_sibling in idx_map:
                                pred_siblings_mapped.append(idx_map[old_sibling])
                            else:
                                logger.debug(f"[TEDS] pred_sibling={old_sibling} not in valid_indices, node {old_idx} -> self-pointing")
                                pred_siblings_mapped.append(new_idx)  # 自指向
                    if gt_siblings_b is not None:
                        gt_siblings_mapped = []
                        for new_idx, old_idx in enumerate(valid_indices):
                            old_sibling = gt_siblings_b[old_idx]
                            if old_sibling in idx_map:
                                gt_siblings_mapped.append(idx_map[old_sibling])
                            else:
                                logger.warning(f"[TEDS] gt_sibling={old_sibling} not in valid_indices, node {old_idx} -> self-pointing (unexpected!)")
                                gt_siblings_mapped.append(new_idx)  # 自指向

                    # 使用 predictor 的反向转换：格式B → 格式A
                    pred_ref_parents, pred_relations = convert_to_format_a(
                        pred_parents_mapped, pred_siblings_mapped
                    )
                    gt_ref_parents, gt_relations = convert_to_format_a(
                        gt_parents_mapped, gt_siblings_mapped
                    )

                    # 使用 TEDSMetric.update() 计算
                    teds_metric.update(
                        pred_texts=texts,
                        pred_parent_ids=pred_ref_parents,
                        pred_relations=pred_relations,
                        gt_texts=texts,
                        gt_parent_ids=gt_ref_parents,
                        gt_relations=gt_relations,
                        sample_id=f"batch{num_batches}_sample{b}",
                    )

            # 收集可视化样本
            if len(vis_samples) < visualize_samples:
                batch_size = line_parent_ids.shape[0]
                # 获取 sibling 预测和标签
                pred_siblings_batch = outputs["sibling_logits"].argmax(dim=-1) if "sibling_logits" in outputs else None
                for b in range(batch_size):
                    if len(vis_samples) >= visualize_samples:
                        break
                    mask_b = line_mask[b].cpu().tolist()
                    valid_indices = [i for i, m in enumerate(mask_b) if m]
                    if len(valid_indices) < 2:
                        continue
                    # 获取文本（如果有）
                    text_map = batch.get("line_text_maps", [{}])[b] if "line_text_maps" in batch else {}
                    # toc_only 模式下用 original_indices 获取原始 line_id
                    if original_indices is not None:
                        orig_ids = [original_indices[b, i].item() for i in valid_indices]
                        sample_texts = [text_map.get(oid, f"node_{oid}") for oid in orig_ids]
                    else:
                        sample_texts = [text_map.get(i, f"node_{i}") for i in valid_indices]
                    sample = {
                        "sample_id": f"batch{num_batches}_sample{b}",
                        "texts": sample_texts,
                        "pred_parents": [pred_parents[b, i].item() for i in valid_indices],
                        "gt_parents": [line_parent_ids[b, i].item() for i in valid_indices],
                        "mask": [True] * len(valid_indices),
                    }
                    # 添加 sibling 信息
                    if pred_siblings_batch is not None and sibling_labels is not None:
                        sample["pred_siblings"] = [pred_siblings_batch[b, i].item() for i in valid_indices]
                        sample["gt_siblings"] = [sibling_labels[b, i].item() for i in valid_indices]
                    vis_samples.append(sample)

        num_batches += 1

    # 计算最终指标
    results = {
        "loss": total_loss / max(num_batches, 1),
        "parent_loss": total_parent_loss / max(num_batches, 1),
        "sibling_loss": total_sibling_loss / max(num_batches, 1),
        "parent_accuracy": all_parent_correct / max(all_parent_total, 1),
    }

    # Sibling accuracy (for N选1 CE formulation)
    total_sibling = sibling_tp + sibling_fp  # sibling_tp = correct, sibling_fp = incorrect
    results["sibling_accuracy"] = sibling_tp / max(total_sibling, 1)

    # TEDS - 使用 TEDSMetric.compute()
    if teds_metric is not None:
        teds_result = teds_metric.compute()
        if teds_result.num_samples > 0:
            results["teds_macro"] = teds_result.macro_teds
            results["teds_samples"] = teds_result.num_samples

    # TOC 可视化
    if vis_samples:
        results["_vis_samples"] = vis_samples

    return results


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

    if args.artifact_dir:
        # 直接使用指定的输出目录（和 stage 目录保持一致）
        output_dir = Path(args.artifact_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exp_manager = None
        logger.info(f"Output directory: {output_dir}")
    else:
        # 使用 exp_manager 管理目录结构
        artifact_root = get_artifact_path(args.env)
        exp_manager, exp_dir = ensure_experiment(
            artifact_root=artifact_root,
            exp=args.exp,
            new_exp=args.new_exp,
            name=args.exp_name or f"DOC Model Training ({stage_name})",
            description=f"Train DOC model with lr={args.learning_rate}, epochs={args.num_epochs}, construct={args.use_construct}",
            config=vars(args),
        )
        output_dir = Path(exp_manager.get_stage_dir(args.exp, stage_name, "comp_hrdoc"))
        logger.info(f"Experiment directory: {exp_dir}")
        logger.info(f"Stage output directory: {output_dir}")
        exp_manager.mark_stage_started(args.exp, stage_name, "comp_hrdoc")

    # Quick test mode
    if args.quick:
        logger.info("Quick test mode enabled")
        args.max_train_samples = args.max_train_samples or 50
        args.max_val_samples = args.max_val_samples or 20
        args.num_epochs = min(args.num_epochs, 2)

    # ==================== Stage Feature Mode ====================
    stage_feature_extractor = None
    if args.use_stage_features:
        logger.info("=" * 60)
        logger.info("Stage Feature Mode: Using pre-trained stage model")
        logger.info("=" * 60)

        # Load StageFeatureExtractor
        from examples.comp_hrdoc.utils.stage_feature_extractor import StageFeatureExtractor

        if not args.stage_checkpoint:
            raise ValueError("--stage-checkpoint is required when using --use-stage-features")

        stage_feature_extractor = StageFeatureExtractor(
            checkpoint_path=args.stage_checkpoint,
            device=str(device),
            max_lines=args.max_regions,
        )
        logger.info(f"Loaded StageFeatureExtractor from: {args.stage_checkpoint}")

        # Log toc_only mode
        if args.toc_only:
            logger.info("=" * 60)
            logger.info("TOC-Only Mode: Training on section headings only (paper 4.4)")
            logger.info(f"  Section label ID: {args.section_label_id}")
            logger.info("=" * 60)

        # 获取数据目录 (使用全局 config)
        from configs.config_loader import load_config as load_global_config
        global_config = load_global_config(args.env).get_effective_config()
        data_dir = global_config.dataset.get_data_dir(args.dataset)
        logger.info(f"Data directory: {data_dir}")

        # 获取 covmatch (命令行参数优先于配置文件)
        covmatch = args.covmatch or global_config.dataset.covmatch
        logger.info(f"Covmatch: {covmatch}")

        # 设置 covmatch 环境变量（复用 stage 的方式）
        if covmatch:
            covmatch_dir = global_config.dataset.get_covmatch_dir(args.dataset)
            if args.covmatch:
                # 命令行指定的 covmatch，覆盖配置
                global_config.dataset.covmatch = args.covmatch
                covmatch_dir = global_config.dataset.get_covmatch_dir(args.dataset)
            if os.path.exists(covmatch_dir):
                os.environ["HRDOC_SPLIT_DIR"] = covmatch_dir
                logger.info(f"Covmatch directory: {covmatch_dir}")
            else:
                logger.error(f"Covmatch directory not found: {covmatch_dir}")
                # 列出可用的 covmatch 目录
                parent_dir = os.path.dirname(covmatch_dir)
                if os.path.exists(parent_dir):
                    available = [d for d in os.listdir(parent_dir) if d.startswith("doc_")]
                    if available:
                        logger.error(f"Available: {', '.join(sorted(available)[:5])}")
                raise FileNotFoundError(f"Covmatch directory not found: {covmatch_dir}")

        # 使用 stage_feature_extractor 的 tokenizer
        tokenizer = stage_feature_extractor.tokenizer

        # tender 数据集默认不使用缓存（数据量小，避免缓存问题）
        force_rebuild = args.dataset == "tender"
        if force_rebuild:
            logger.info("tender dataset: force_rebuild enabled by default")

        # 复用 stage 的 DataLoader（支持多 chunk 文档级别处理）
        data_loader_config = HRDocDataLoaderConfig(
            data_dir=data_dir,
            dataset_name=args.dataset,
            document_level=True,  # 文档级别模式，支持多 chunk
            max_length=512,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            force_rebuild=force_rebuild,
        )
        data_loader = HRDocDataLoader(tokenizer, data_loader_config)
        datasets = data_loader.prepare_datasets()

        train_dataset = datasets.get("train", [])
        val_dataset = datasets.get("validation", [])

        logger.info(f"Train dataset: {len(train_dataset)} documents")
        logger.info(f"Val dataset: {len(val_dataset)} documents")

        # 复用 stage 的 HRDocDocumentLevelCollator
        collator = HRDocDocumentLevelCollator(tokenizer)
        logger.info("Using stage HRDocDocumentLevelCollator (multi-chunk support)")

        # 创建 DataLoader
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

        logger.info(f"Using comp_hrdoc HRDocDataset with dataset={args.dataset}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ==================== Standard Mode ====================
    else:
        # 使用 hrdoc_loader (统一的 HRDS/HRDH 加载器)
        from configs.config_loader import load_config as load_global_config
        from transformers import AutoTokenizer

        # 获取数据目录 (使用全局 config)
        global_config = load_global_config(args.env).get_effective_config()
        data_dir = global_config.dataset.get_data_dir(args.dataset)
        logger.info(f"Data directory: {data_dir}")

        # 获取 covmatch (命令行参数优先于配置文件)
        covmatch = args.covmatch or global_config.dataset.covmatch
        if not covmatch:
            raise ValueError("--covmatch is required. Please specify a covmatch split name.")
        logger.info(f"Covmatch: {covmatch}")

        mode = "document-level" if args.document_level else "page-level"
        logger.info(f"Creating datasets in {mode} mode...")

        # 创建数据集
        train_dataset = HRDocDataset(
            data_dir=data_dir,
            dataset_name=args.dataset,
            max_lines=args.max_regions,
            max_samples=args.max_train_samples,
            split='train',
            covmatch=covmatch,
        )
        val_dataset = HRDocDataset(
            data_dir=data_dir,
            dataset_name=args.dataset,
            max_lines=args.max_regions,
            max_samples=args.max_val_samples,
            split='validation',
            covmatch=covmatch,
        )

        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(val_dataset)} samples")

        # 加载 tokenizer
        model_name = global_config.model.local_path or global_config.model.name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded tokenizer from: {model_name}")

        # 创建 collator
        collator = HRDocLayoutXLMCollator(
            tokenizer=tokenizer,
            max_length=512,
            max_lines=args.max_regions,
        )
        logger.info(f"Using HRDocLayoutXLMCollator with max_lines={args.max_regions}")

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
    if args.use_stage_features:
        # Use ConstructFromFeatures when using stage features
        from examples.comp_hrdoc.models.construct_only import (
            ConstructFromFeatures,
            build_construct_from_features,
            save_construct_model,
        )
        logger.info("Building ConstructFromFeatures model (using stage features)...")
        model = build_construct_from_features(
            hidden_size=args.hidden_size,
            num_categories=14,  # HRDoc has 14 classes
            num_heads=args.num_heads,
            num_layers=args.construct_num_layers,
            dropout=args.dropout,
        )
        save_fn = lambda m, p: save_construct_model(m, p, save_order=False)
    elif args.model_type == "order-only":
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

    # Load pretrained weights (model_name_or_path)
    if args.model_name_or_path:
        ckpt_path = Path(args.model_name_or_path)
        model_bin = ckpt_path / "pytorch_model.bin"
        if model_bin.exists():
            logger.info(f"Loading model weights from {model_bin}")
            state_dict = torch.load(model_bin, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            # 兼容旧格式
            construct_pt = ckpt_path / "construct_model.pt"
            if construct_pt.exists():
                logger.info(f"Loading construct_module weights from {construct_pt}")
                state_dict = torch.load(construct_pt, map_location="cpu")
                model.construct_module.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"No model found at {args.model_name_or_path}")

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

    # Resume from checkpoint (full training state)
    start_epoch = 0
    best_metric = 0.0
    if args.resume_from_checkpoint:
        ckpt_path = Path(args.resume_from_checkpoint)
        training_state = ckpt_path / "training_state.pt"
        if training_state.exists():
            logger.info(f"Resuming training from {training_state}")
            state = torch.load(training_state, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state and scheduler is not None:
                scheduler.load_state_dict(state["scheduler"])
            if "scaler" in state and scaler is not None:
                scaler.load_state_dict(state["scaler"])
            start_epoch = state.get("epoch", 0) + 1
            best_metric = state.get("best_metric", 0.0)
            logger.info(f"Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")
        else:
            # Fallback: just load model weights
            model_bin = ckpt_path / "pytorch_model.bin"
            if model_bin.exists():
                logger.info(f"Loading model weights from {model_bin} (no training state)")
                model.load_state_dict(torch.load(model_bin, map_location=device))

    # Training loop
    logger.info("Starting training...")
    best_metric_name = "parent_accuracy" if args.use_stage_features else "order_accuracy"

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{args.num_epochs} =====")

        # Train
        if args.use_stage_features:
            train_metrics = train_epoch_with_stage_features(
                model=model,
                dataloader=train_loader,
                feature_extractor=stage_feature_extractor,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                args=args,
                scaler=scaler,
            )
            train_log = (
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Parent: {train_metrics['parent_loss']:.4f}, "
                f"Sibling: {train_metrics['sibling_loss']:.4f}"
            )
        else:
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
        if args.use_stage_features:
            val_metrics = evaluate_with_stage_features(
                model, val_loader, stage_feature_extractor, device, args,
                visualize_samples=3,  # 每个 epoch 可视化 3 个样本
            )
        else:
            val_metrics = evaluate(model, val_loader, device)
        # Log validation metrics
        if args.use_stage_features:
            val_log = (
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Parent: {val_metrics['parent_loss']:.4f}, "
                f"Sibling: {val_metrics['sibling_loss']:.4f}"
            )
            logger.info(val_log)

            # 4.4 Construct metrics (stage feature mode)
            logger.info(
                f"[4.4 Construct] Parent Acc: {val_metrics['parent_accuracy']:.4f}, "
                f"Sibling Acc: {val_metrics.get('sibling_accuracy', 0):.4f}"
            )
            # TEDS
            if 'teds_macro' in val_metrics:
                logger.info(
                    f"[4.4 Construct] TEDS: {val_metrics['teds_macro']:.4f} "
                    f"(n={val_metrics.get('teds_samples', 0)})"
                )

            # TOC 可视化
            if "_vis_samples" in val_metrics:
                logger.info("\n" + "="*60)
                logger.info("TOC Visualization (3 samples)")
                logger.info("="*60)
                for sample in val_metrics["_vis_samples"]:
                    vis_str = visualize_toc(
                        texts=sample["texts"],
                        pred_parents=sample["pred_parents"],
                        gt_parents=sample["gt_parents"],
                        mask=sample["mask"],
                        sample_id=sample["sample_id"],
                        pred_siblings=sample.get("pred_siblings"),
                        gt_siblings=sample.get("gt_siblings"),
                    )
                    logger.info(vis_str)
        else:
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
        current_metric = val_metrics.get(best_metric_name, 0)
        if current_metric > best_metric:
            best_metric = current_metric
            best_path = output_dir / "best_model"
            save_fn(model, str(best_path))
            # 保存完整训练状态（用于 resume_from_checkpoint）
            training_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "best_metric": best_metric,
            }
            torch.save(training_state, str(best_path / "training_state.pt"))
            logger.info(f"Saved best model to {best_path} ({best_metric_name}: {best_metric:.4f})")

    # Save final model
    final_path = output_dir / "final_model"
    save_fn(model, str(final_path))
    # 保存最终训练状态
    training_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": args.num_epochs - 1,
        "best_metric": best_metric,
    }
    torch.save(training_state, str(final_path / "training_state.pt"))
    logger.info(f"Saved final model to {final_path}")

    # Mark stage as completed
    if exp_manager is not None:
        exp_manager.mark_stage_completed(
            args.exp, stage_name, "comp_hrdoc",
            best_checkpoint=str(output_dir / "best_model"),
            metrics={f"best_{best_metric_name}": best_metric},
        )

    logger.info("Training complete!")
    logger.info(f"Best {best_metric_name}: {best_metric:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

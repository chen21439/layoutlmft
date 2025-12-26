#!/usr/bin/env python
"""Training script for Intra-region Head (Section 4.2.3) with LayoutXLM

Trains the line-level successor prediction model using LayoutXLM for feature encoding.
After training, use the model to group lines into regions via Union-Find.

Usage:
    python examples/comp_hrdoc/scripts/train_intra.py --env test
    python examples/comp_hrdoc/scripts/train_intra.py --env test --quick
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

    # 根据环境加载对应配置文件 (dev.yaml / test.yaml)
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", f"{env}.yaml"
    )

    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        gpu_config = config.get('gpu', {})
        cuda_visible_devices = gpu_config.get('cuda_visible_devices')

        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
            print(f"[GPU Setup] env={env}, CUDA_VISIBLE_DEVICES={cuda_visible_devices}")


_setup_gpu_early()
# ==================== GPU 设置结束 ====================

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from examples.comp_hrdoc.data.line_level_loader import (
    LineLevelDataset,
    create_layoutxlm_line_dataloaders,
)
from examples.comp_hrdoc.data.line_collator_v2 import (
    LineLevelCollatorV2,
    create_dataloaders_v2,
)
from examples.comp_hrdoc.data.hrds_loader import (
    HRDSDataset,
    HRDSLayoutXLMCollator,
    create_hrds_layoutxlm_dataloaders,
)
from examples.comp_hrdoc.models import (
    IntraRegionModule,
    predict_successors,
    group_lines_to_regions,
)
from examples.comp_hrdoc.utils.experiment_manager import (
    ExperimentManager,
    get_artifact_path,
    ensure_experiment,
)

# LayoutXLM imports
from layoutlmft.models.layoutxlm import (
    LayoutXLMModel,
    LayoutXLMConfig,
    LayoutXLMTokenizerFast,
)
from layoutlmft.models.relation_classifier import LineFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Intra-region Head with LayoutXLM")

    # Environment
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--config", type=str, default=None, help="Config file path")

    # LayoutXLM Model
    parser.add_argument("--layoutxlm-path", type=str, default=None,
                        help="Path to LayoutXLM base model (if None, use default)")
    parser.add_argument("--freeze-layoutxlm", action="store_true", default=True,
                        help="Freeze LayoutXLM parameters")
    parser.add_argument("--no-freeze-layoutxlm", dest="freeze_layoutxlm", action="store_false")

    # Intra-region Head
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to data (JSON file for HRDH, directory for HRDS)")
    parser.add_argument("--data-format", type=str, default="hrdh", choices=["hrdh", "hrds"],
                        help="Data format: hrdh=unified JSON file, hrds=individual JSON files per doc")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length for tokenization")
    parser.add_argument("--max-lines", type=int, default=128)
    parser.add_argument("--val-split-ratio", type=float, default=0.1)
    parser.add_argument("--collator-version", type=str, default="v2", choices=["v1", "v2"],
                        help="Collator version: v1=manual concat, v2=is_split_into_words (recommended)")

    # Training
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (overrides exp management)")

    # Experiment management
    parser.add_argument("--exp", type=str, default=None,
                        help="Experiment name (None=current, 'latest', or exp_YYYYMMDD_HHMMSS)")
    parser.add_argument("--new-exp", action="store_true",
                        help="Force create new experiment")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Human-readable experiment name")
    parser.add_argument("--dataset", type=str, default="comp_hrdoc",
                        help="Dataset name for stage directory")

    # Quick test
    parser.add_argument("--quick", action="store_true", help="Quick test with small data")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_data_path(env: str, data_format: str = "hrdh") -> str:
    """Get data path based on environment and format

    Args:
        env: Environment ('dev' or 'test')
        data_format: Data format ('hrdh' for unified JSON, 'hrds' for individual files)
    """
    if data_format == "hrds":
        paths = {
            "dev": "/mnt/e/models/data/Section/HRDS/train",
            "test": "/data/LLM_group/layoutlmft/data/HRDS/train",
        }
    else:  # hrdh
        paths = {
            "dev": "/mnt/e/models/data/Section/Comp_HRDoc/HRDH_MSRA_POD_TRAIN/unified_layout_analysis_train.json",
            "test": "/data/LLM_group/layoutlmft/data/Comp_HRDoc/HRDH_MSRA_POD_TRAIN/unified_layout_analysis_train.json",
        }
    return paths.get(env, paths["dev"])


def get_layoutxlm_path(env: str) -> str:
    """Get LayoutXLM model path based on environment"""
    paths = {
        "dev": "/mnt/e/models/HuggingFace/hub/models--microsoft--layoutxlm-base/snapshots/8e04ebc4d3ba0013cf943b697c0aedf19b06472a",
        "test": "/data/LLM_group/HuggingFace/Hub/models--microsoft--layoutxlm-base/snapshots/8e04ebc4d3ba0013cf943b697c0aedf19b06472a",
    }
    return paths.get(env, paths["dev"])


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IntraRegionWithLayoutXLM(nn.Module):
    """Combined model: LayoutXLM + IntraRegionModule

    1. LayoutXLM encodes tokens -> hidden states
    2. LineFeatureExtractor aggregates to line-level features
    3. IntraRegionModule predicts successor relationships
    """

    def __init__(
        self,
        layoutxlm_model: LayoutXLMModel,
        intra_region: IntraRegionModule,
        freeze_layoutxlm: bool = True,
    ):
        super().__init__()
        self.layoutxlm = layoutxlm_model
        self.intra_region = intra_region
        self.freeze_layoutxlm = freeze_layoutxlm

        if freeze_layoutxlm:
            for param in self.layoutxlm.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ids: torch.Tensor,
        line_mask: torch.Tensor,
        successor_labels: torch.Tensor = None,
        role_labels: torch.Tensor = None,  # 4.2.4: Logical role labels
        image: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            bbox: [batch, seq_len, 4]
            attention_mask: [batch, seq_len]
            line_ids: [batch, seq_len] - which line each token belongs to
            line_mask: [batch, num_lines] - valid lines
            successor_labels: [batch, num_lines] - ground truth successors
            role_labels: [batch, num_lines] - ground truth role labels (4.2.4)
            image: Optional image tensor

        Returns:
            Dict with successor_logits, role_logits, and loss
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Create dummy image if not provided (LayoutXLM requires image input)
        if image is None:
            # LayoutXLM expects [batch, 3, 224, 224] images
            image = torch.zeros(batch_size, 3, 224, 224, device=device)

        # Step 1: LayoutXLM encoding
        if self.freeze_layoutxlm:
            with torch.no_grad():
                outputs = self.layoutxlm(
                    input_ids=input_ids,
                    bbox=bbox,
                    attention_mask=attention_mask,
                    image=image,
                    output_hidden_states=True,
                )
        else:
            outputs = self.layoutxlm(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                image=image,
                output_hidden_states=True,
            )

        # Get last hidden state: [batch, seq_len + visual_tokens, hidden_size]
        hidden_states = outputs.last_hidden_state

        # LayoutXLM adds visual tokens at the end, truncate to text tokens only
        seq_len = input_ids.size(1)
        hidden_states = hidden_states[:, :seq_len, :]

        # Step 2: Extract line-level features
        line_features, extracted_mask = LineFeatureExtractor.extract_line_features(
            hidden_states=hidden_states,
            line_ids=line_ids,
            pooling="mean",
        )

        # Adjust successor_labels and role_labels to match the extracted features shape
        # extracted_mask has shape [batch, actual_num_lines] based on data
        # while line_mask/successor_labels are padded to max_lines from dataloader
        actual_num_lines = line_features.size(1)
        adjusted_successor_labels = None
        adjusted_role_labels = None
        if successor_labels is not None:
            adjusted_successor_labels = successor_labels[:, :actual_num_lines]
        if role_labels is not None:
            adjusted_role_labels = role_labels[:, :actual_num_lines]

        # Step 3: Intra-region head (4.2.3 + 4.2.4)
        outputs = self.intra_region(
            line_features=line_features,
            line_mask=extracted_mask,  # Use extracted mask, not batch mask
            successor_labels=adjusted_successor_labels,
            role_labels=adjusted_role_labels,
        )

        # Also return the actual mask and line count for evaluation
        outputs['actual_line_mask'] = extracted_mask
        outputs['actual_num_lines'] = actual_num_lines

        return outputs


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    args,
    scaler=None,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        line_ids = batch["line_ids"].to(device)
        line_mask = batch["line_mask"].to(device)
        successor_labels = batch["successor_labels"].to(device)
        # 4.2.4: Logical role classification labels
        role_labels = batch.get("class_labels")
        if role_labels is not None:
            role_labels = role_labels.to(device)

        # Forward pass
        if args.fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    bbox=bbox,
                    attention_mask=attention_mask,
                    line_ids=line_ids,
                    line_mask=line_mask,
                    successor_labels=successor_labels,
                    role_labels=role_labels,
                )
                loss = outputs["loss"] / args.gradient_accumulation_steps
        else:
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                line_ids=line_ids,
                line_mask=line_mask,
                successor_labels=successor_labels,
                role_labels=role_labels,
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

        # Compute accuracy
        with torch.no_grad():
            logits = outputs["successor_logits"]
            preds = logits.argmax(dim=-1)  # [batch, actual_num_lines]

            # Get actual dimensions and truncate labels
            actual_num_lines = outputs["actual_num_lines"]
            actual_mask = outputs["actual_line_mask"]
            successor_labels_trunc = successor_labels[:, :actual_num_lines]

            # Only count lines with valid successors
            has_successor = (successor_labels_trunc >= 0) & actual_mask
            correct = (preds == successor_labels_trunc) & has_successor
            total_correct += correct.sum().item()
            total_count += has_successor.sum().item()

        total_loss += outputs["loss"].item()
        num_batches += 1

        # Update progress bar
        acc = total_correct / max(total_count, 1)
        progress_bar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "acc": f"{acc:.4f}",
        })

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    num_batches = 0

    # Region grouping metrics
    total_region_precision = 0.0
    total_region_recall = 0.0
    num_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        line_ids = batch["line_ids"].to(device)
        line_mask = batch["line_mask"].to(device)
        successor_labels = batch["successor_labels"].to(device)
        region_ids = batch["region_ids"].to(device)
        # 4.2.4: Logical role classification labels
        role_labels = batch.get("class_labels")
        if role_labels is not None:
            role_labels = role_labels.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            line_ids=line_ids,
            line_mask=line_mask,
            successor_labels=successor_labels,
            role_labels=role_labels,
        )

        total_loss += outputs["loss"].item()

        # Get actual dimensions used by the model
        actual_num_lines = outputs["actual_num_lines"]
        actual_mask = outputs["actual_line_mask"]

        # Truncate labels and region_ids to match actual dimensions
        successor_labels_trunc = successor_labels[:, :actual_num_lines]
        region_ids_trunc = region_ids[:, :actual_num_lines]

        # Compute successor accuracy
        logits = outputs["successor_logits"]
        preds = logits.argmax(dim=-1)

        has_successor = (successor_labels_trunc >= 0) & actual_mask
        correct = (preds == successor_labels_trunc) & has_successor
        total_correct += correct.sum().item()
        total_count += has_successor.sum().item()

        # Compute region grouping metrics (for each sample in batch)
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            sample_mask = actual_mask[i]
            sample_logits = logits[i]
            sample_region_ids = region_ids_trunc[i]

            # Predict successors
            pred_successors = predict_successors(sample_logits, sample_mask)

            # Group into regions
            pred_regions = group_lines_to_regions(pred_successors, sample_mask)

            # Get ground truth regions
            gt_regions = {}
            n_valid = sample_mask.sum().item()
            for line_idx in range(n_valid):
                rid = sample_region_ids[line_idx].item()
                if rid >= 0:
                    if rid not in gt_regions:
                        gt_regions[rid] = []
                    gt_regions[rid].append(line_idx)
            gt_region_list = list(gt_regions.values())

            # Compute precision/recall (line pairs in same region)
            pred_pairs = set()
            for region in pred_regions:
                for j in range(len(region)):
                    for k in range(j + 1, len(region)):
                        pred_pairs.add((min(region[j], region[k]), max(region[j], region[k])))

            gt_pairs = set()
            for region in gt_region_list:
                for j in range(len(region)):
                    for k in range(j + 1, len(region)):
                        gt_pairs.add((min(region[j], region[k]), max(region[j], region[k])))

            if len(pred_pairs) > 0:
                precision = len(pred_pairs & gt_pairs) / len(pred_pairs)
            else:
                precision = 1.0 if len(gt_pairs) == 0 else 0.0

            if len(gt_pairs) > 0:
                recall = len(pred_pairs & gt_pairs) / len(gt_pairs)
            else:
                recall = 1.0

            total_region_precision += precision
            total_region_recall += recall
            num_samples += 1

        num_batches += 1

    avg_precision = total_region_precision / max(num_samples, 1)
    avg_recall = total_region_recall / max(num_samples, 1)
    f1 = 2 * avg_precision * avg_recall / max(avg_precision + avg_recall, 1e-8)

    return {
        "loss": total_loss / max(num_batches, 1),
        "successor_accuracy": total_correct / max(total_count, 1),
        "region_precision": avg_precision,
        "region_recall": avg_recall,
        "region_f1": f1,
    }


def save_model(model, output_path: str):
    """Save model checkpoint (only IntraRegionModule, not LayoutXLM)"""
    os.makedirs(output_path, exist_ok=True)
    # Only save the intra_region part (LayoutXLM is frozen)
    torch.save({
        "intra_region": model.intra_region.state_dict(),
    }, os.path.join(output_path, "checkpoint.pt"))
    logger.info(f"Saved IntraRegionModule to {output_path}")


def load_model(model, checkpoint_path: str):
    """Load model checkpoint"""
    checkpoint = torch.load(os.path.join(checkpoint_path, "checkpoint.pt"))
    model.intra_region.load_state_dict(checkpoint["intra_region"])
    logger.info(f"Loaded IntraRegionModule from {checkpoint_path}")


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    logger.info(f"Environment: {args.env}")
    logger.info(f"Arguments: {args}")

    # CUDA setup
    if torch.cuda.is_available():
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup experiment management
    exp_manager = None
    if args.output_dir:
        # Legacy mode: use explicit output directory
        output_dir = Path(args.output_dir)
        logger.info(f"Using explicit output directory: {output_dir}")
    else:
        # Experiment management mode
        artifact_path = get_artifact_path(args.env)
        exp_manager, exp_dir = ensure_experiment(
            artifact_root=artifact_path,
            exp=args.exp,
            new_exp=args.new_exp,
            name=args.exp_name or f"Intra-region {args.dataset}",
            description=f"Intra-region training for {args.dataset}",
        )
        output_dir = Path(exp_manager.get_stage_dir(args.exp, "intra", args.dataset))
        logger.info(f"Experiment directory: {exp_dir}")
        logger.info(f"Stage output directory: {output_dir}")

    # Paths
    data_path = args.data_path or get_data_path(args.env, args.data_format)
    layoutxlm_path = args.layoutxlm_path or get_layoutxlm_path(args.env)
    logger.info(f"Data format: {args.data_format}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"LayoutXLM path: {layoutxlm_path}")

    # Quick test mode
    if args.quick:
        logger.info("Quick test mode enabled")
        args.max_train_samples = args.max_train_samples or 20
        args.max_val_samples = args.max_val_samples or 10
        args.num_epochs = min(args.num_epochs, 2)

    # Load LayoutXLM tokenizer
    logger.info("Loading LayoutXLM tokenizer...")
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(layoutxlm_path)

    # Create dataloaders
    logger.info(f"Creating dataloaders (format: {args.data_format}, collator: {args.collator_version})...")

    if args.data_format == "hrds":
        # HRDS format: individual JSON files per document
        train_loader, val_loader = create_hrds_layoutxlm_dataloaders(
            data_dir=data_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_lines=args.max_lines,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            val_split_ratio=args.val_split_ratio,
        )
    elif args.collator_version == "v2":
        # V2: is_split_into_words=True approach (recommended)
        train_dataset = LineLevelDataset(
            data_path=data_path,
            max_lines=args.max_lines,
            max_samples=args.max_train_samples,
            split='train',
            val_split_ratio=args.val_split_ratio,
        )
        val_dataset = LineLevelDataset(
            data_path=data_path,
            max_lines=args.max_lines,
            max_samples=args.max_val_samples,
            split='validation',
            val_split_ratio=args.val_split_ratio,
        )
        train_loader, val_loader = create_dataloaders_v2(
            dataset_train=train_dataset,
            dataset_val=val_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_lines=args.max_lines,
        )
    else:
        # V1: manual concat approach
        train_loader, val_loader = create_layoutxlm_line_dataloaders(
            data_path=data_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_lines=args.max_lines,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            val_split_ratio=args.val_split_ratio,
        )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Load LayoutXLM model
    logger.info("Loading LayoutXLM base model...")
    layoutxlm_config = LayoutXLMConfig.from_pretrained(layoutxlm_path)
    layoutxlm_model = LayoutXLMModel.from_pretrained(layoutxlm_path, config=layoutxlm_config)

    # Build IntraRegionModule
    logger.info("Building IntraRegionModule...")
    intra_region = IntraRegionModule(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # Combined model
    model = IntraRegionWithLayoutXLM(
        layoutxlm_model=layoutxlm_model,
        intra_region=intra_region,
        freeze_layoutxlm=args.freeze_layoutxlm,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer (only for trainable parameters)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Calculate training steps
    num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    logger.info(f"Calculated training steps: {num_training_steps} "
                f"(batches={len(train_loader)}, epochs={args.num_epochs}, "
                f"grad_accum={args.gradient_accumulation_steps})")

    # Create learning rate scheduler with fallback
    # OneCycleLR can fail with certain PyTorch versions due to edge cases
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

    scheduler = None
    if num_training_steps >= 10:
        try:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.learning_rate,
                total_steps=num_training_steps,
                pct_start=args.warmup_ratio,
            )
            logger.info(f"Using OneCycleLR scheduler with {num_training_steps} steps")
        except Exception as e:
            logger.warning(f"OneCycleLR failed ({e}), falling back to CosineAnnealingLR")
            scheduler = None

    if scheduler is None:
        if num_training_steps >= 2:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
            logger.info(f"Using CosineAnnealingLR scheduler with T_max={num_training_steps}")
        else:
            scheduler = LambdaLR(optimizer, lambda _: 1.0)
            logger.warning(f"Training steps ({num_training_steps}) too small, using constant LR")

    # FP16 scaler
    scaler = None
    if args.fp16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    logger.info("Starting training...")
    best_f1 = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mark stage as started
    if exp_manager:
        exp_manager.mark_stage_started(args.exp, "intra", args.dataset)

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
            f"Accuracy: {train_metrics['accuracy']:.4f}"
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Successor Acc: {val_metrics['successor_accuracy']:.4f}"
        )
        logger.info(
            f"Val - Region P: {val_metrics['region_precision']:.4f}, "
            f"R: {val_metrics['region_recall']:.4f}, "
            f"F1: {val_metrics['region_f1']:.4f}"
        )

        # Save best model
        if val_metrics['region_f1'] > best_f1:
            best_f1 = val_metrics['region_f1']
            best_path = output_dir / "best_model"
            save_model(model, str(best_path))
            logger.info(f"Saved best model (F1={best_f1:.4f})")

    # Save final model
    final_path = output_dir / "final_model"
    save_model(model, str(final_path))

    # Mark stage as completed
    if exp_manager:
        exp_manager.mark_stage_completed(
            args.exp, "intra", args.dataset,
            best_checkpoint="best_model",
            metrics={
                "best_region_f1": best_f1,
                "final_successor_acc": val_metrics['successor_accuracy'],
                "final_region_precision": val_metrics['region_precision'],
                "final_region_recall": val_metrics['region_recall'],
            },
        )

    logger.info("Training complete!")
    logger.info(f"Best Region F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

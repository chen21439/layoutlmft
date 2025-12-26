#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 训练脚本

使用论文定义的 14 个语义类别（Line 级别标注，不使用 BIO）。
标签定义统一使用 layoutlmft.data.labels 模块。
"""

import logging
import os
import sys
import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import layoutlmft.data.datasets.hrdoc
import transformers
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.trainers import FunsdTrainer as Trainer
from layoutlmft.data.labels import (
    LABEL_LIST,
    NUM_LABELS,
    get_id2label,
    get_label2id,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertTokenizerFast,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers import TrainerCallback
from layoutlmft.models.layoutxlm import LayoutXLMForTokenClassification, LayoutXLMConfig
from layoutlmft.models.layoutxlm import LayoutXLMTokenizerFast
import layoutlmft
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# 添加项目路径（用于导入 data 模块和 metrics 模块）
STAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_ROOT = os.path.dirname(STAGE_ROOT)  # examples/ 目录，用于导入统一的 metrics 模块
sys.path.insert(0, STAGE_ROOT)
sys.path.insert(0, EXAMPLES_ROOT)
from data import HRDocDataLoader, HRDocDataLoaderConfig, load_hrdoc_raw_datasets

# Register both layoutxlm and layoutlmv2 (LayoutXLM uses layoutlmv2 as model_type in config.json)
CONFIG_MAPPING.update({
    "layoutxlm": LayoutXLMConfig,
    "layoutlmv2": LayoutXLMConfig,  # LayoutXLM's config.json has model_type="layoutlmv2"
})

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


class TokenizerSaveCallback(TrainerCallback):
    """Callback to explicitly save tokenizer to each checkpoint directory.

    Ensures both legacy format (sentencepiece) and tokenizer.json are saved.
    HuggingFace Trainer's default save_pretrained may not save tokenizer.json.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        """Called after checkpoint is saved."""
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_dir):
            # Check if tokenizer.json already exists
            if not os.path.exists(os.path.join(checkpoint_dir, "tokenizer.json")):
                # Save both formats: legacy (sentencepiece) + tokenizer.json
                self.tokenizer.save_pretrained(checkpoint_dir, legacy_format=True)
                self.tokenizer.save_pretrained(checkpoint_dir, legacy_format=False)
                logger.info(f"Saved tokenizer (both formats) to {checkpoint_dir}")


class LineLevelEvalCallback(TrainerCallback):
    """
    在每次评估后运行 LINE 级别诊断

    诊断内容：
    1. label=-100 但参与投票的 token 占比
    2. TOKEN vs LINE 指标对比
    """

    def __init__(self, eval_dataset, data_collator, label_list):
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.label_list = label_list

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """在 Trainer.evaluate() 后运行 LINE 级别诊断"""
        if model is None:
            return

        try:
            from torch.utils.data import DataLoader
            from metrics.line_eval import compute_line_level_metrics_batch

            logger.info("")
            logger.info("=" * 65)
            logger.info(f"[LINE-LEVEL DIAG] Step {state.global_step}")
            logger.info("=" * 65)

            # 创建小批量评估（避免太慢）
            max_samples = min(500, len(self.eval_dataset))
            subset = torch.utils.data.Subset(self.eval_dataset, range(max_samples))
            eval_dataloader = DataLoader(
                subset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
            )

            device = next(model.parameters()).device
            model.eval()

            # 累积统计
            all_token_preds = []
            all_token_labels = []
            all_line_ids = []
            diag_total_tokens = 0
            diag_voted_tokens = 0
            diag_label_minus100_voted = 0

            with torch.no_grad():
                for batch in eval_dataloader:
                    # 将所有 tensor 移到 device
                    input_ids = batch["input_ids"].to(device)
                    bbox = batch["bbox"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_on_device = batch["labels"].to(device)

                    # 处理 image（ImageList 类型）
                    # detectron2 用 .tensor（单数），shim 用 .tensors（复数）
                    image = batch.get("image")
                    if image is not None:
                        if hasattr(image, 'tensor'):
                            # detectron2 的 ImageList（使用 .tensor 单数）
                            from detectron2.structures import ImageList as D2ImageList
                            image = D2ImageList(image.tensor.to(device), image.image_sizes)
                        elif hasattr(image, 'tensors'):
                            # shim ImageList（使用 .tensors 复数）
                            class SimpleImageList:
                                def __init__(self, tensors, image_sizes):
                                    self.tensors = tensors
                                    self.image_sizes = image_sizes
                            image = SimpleImageList(image.tensors.to(device), image.image_sizes)
                        elif isinstance(image, torch.Tensor):
                            image = image.to(device)

                    outputs = model(
                        input_ids=input_ids,
                        bbox=bbox,
                        attention_mask=attention_mask,
                        image=image,
                    )
                    preds = outputs.logits.argmax(dim=-1)

                    for b in range(preds.shape[0]):
                        token_preds = preds[b].cpu().tolist()
                        token_labels = batch["labels"][b].cpu().tolist()

                        if "line_ids" in batch:
                            line_ids = batch["line_ids"][b].cpu().tolist()
                            all_line_ids.append(line_ids)

                            # 诊断统计（只统计文本 token，不含 CLS/SEP/PAD）
                            for pred, label, line_id in zip(token_preds, token_labels, line_ids):
                                if line_id >= 0:  # 只统计真正的文本 token
                                    diag_total_tokens += 1
                                    diag_voted_tokens += 1
                                    if label == -100:
                                        diag_label_minus100_voted += 1

                        all_token_preds.append(token_preds)
                        all_token_labels.append(token_labels)

            # 打印诊断
            if diag_total_tokens > 0:
                logger.info(f"[DIAG] 投票统计 (前 {max_samples} 样本, 不含 special tokens):")
                logger.info(f"  文本 token: {diag_total_tokens}")
                if diag_voted_tokens > 0:
                    pct = diag_label_minus100_voted / diag_voted_tokens * 100
                    logger.info(f"  未监督 (label=-100): {diag_label_minus100_voted} ({pct:.1f}%)")
                    if pct > 30:
                        logger.warning(f"  ⚠️  {pct:.1f}% 的文本 token 未被监督，可能影响 LINE 级别指标！")

            # 计算 LINE 级别指标
            if all_line_ids:
                line_metrics = compute_line_level_metrics_batch(
                    batch_token_predictions=all_token_preds,
                    batch_token_labels=all_token_labels,
                    batch_line_ids=all_line_ids,
                    num_classes=len(self.label_list),
                    class_names=self.label_list,
                )
                logger.info(f"[DIAG] LINE 级别指标:")
                logger.info(f"  Accuracy: {line_metrics.accuracy:.2%}")
                logger.info(f"  Macro-F1: {line_metrics.macro_f1:.2%}")
            else:
                logger.warning("[DIAG] line_ids 不可用，无法计算 LINE 级别指标")

            logger.info("=" * 65)

        except Exception as e:
            logger.warning(f"[LINE-LEVEL DIAG] 诊断失败: {e}")


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 保留 line_ids 列用于行级评估（Trainer 默认会移除未使用的列）
    # 必须在两个解析分支之后设置
    training_args.remove_unused_columns = False

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 使用统一数据加载器加载数据集
    # dataset_name 用于区分不同数据集的缓存（hrds, hrdh, tender 等）
    dataset_name = data_args.dataset_name or "hrdoc"
    loader_config = HRDocDataLoaderConfig(
        data_dir=os.environ.get("HRDOC_DATA_DIR"),
        dataset_name=dataset_name,
        max_length=512,
        preprocessing_num_workers=data_args.preprocessing_num_workers or 4,
        max_train_samples=data_args.max_train_samples,
        max_val_samples=data_args.max_val_samples,
        max_test_samples=data_args.max_test_samples,
        label_all_tokens=data_args.label_all_tokens,
        pad_to_max_length=data_args.pad_to_max_length,
        force_rebuild=data_args.force_rebuild,
    )

    # 先加载原始数据集（用于 column_names、features 和 balanced loss 计算）
    # 使用统一的数据加载函数
    datasets = load_hrdoc_raw_datasets(
        data_dir=os.environ.get("HRDOC_DATA_DIR"),
        dataset_name=dataset_name,
        force_rebuild=data_args.force_rebuild,
    )

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    elif "validation" in datasets:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    elif "test" in datasets:
        column_names = datasets["test"].column_names
        features = datasets["test"].features
    else:
        # Fallback to first available split
        first_split = list(datasets.keys())[0]
        column_names = datasets[first_split].column_names
        features = datasets[first_split].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    remove_columns = column_names

    # 使用统一的标签定义（从 labels.py 导入）
    # 论文定义的 14 个语义类别（Line 级别标注，不使用 BIO）
    label_list = LABEL_LIST
    num_labels = NUM_LABELS
    label_to_id = get_label2id()
    id2label = get_id2label()
    label2id = label_to_id  # 别名

    logger.info(f"Using {num_labels} labels from labels.py: {label_list}")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # 上面已经有：
    # config = AutoConfig.from_pretrained(
    #     args.model_name_or_path,
    #     cache_dir=args.cache_dir,
    # )

    # Determine model type from config (most reliable, works for checkpoints)
    model_type = getattr(config, "model_type", None)

    # Load tokenizer: use fast tokenizer directly, fail if not available
    # Fast tokenizer can be built from sentencepiece even without tokenizer.json
    if model_type == "layoutxlm":
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        assert tokenizer.is_fast, "LayoutXLM requires fast tokenizer"
        logger.info("Using LayoutXLMTokenizerFast")
    elif model_type == "layoutlmv2":
        # LayoutLMv2 uses BERT tokenizer (vocab.txt)
        tokenizer = BertTokenizerFast.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        logger.info("Using LayoutLMv2 tokenizer (BERT)")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
        )
        logger.info("Using AutoTokenizer")

    model = LayoutXLMForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Setup balanced loss for long-tailed classification (only for training)
    if training_args.do_train and data_args.loss_type != "ce":
        from layoutlmft.models.balanced_loss import get_balanced_loss

        # Compute class counts from training data
        logger.info("Computing class counts for balanced loss...")
        class_counts = [0] * num_labels
        for sample in datasets["train"]:
            for label_id in sample[label_column_name]:
                if 0 <= label_id < num_labels:
                    class_counts[label_id] += 1

        logger.info(f"Class counts (first 10): {class_counts[:10]}")
        logger.info(f"Total samples: {sum(class_counts)}")

        # Create balanced loss
        balanced_loss = get_balanced_loss(
            loss_type=data_args.loss_type,
            class_counts=class_counts,
            beta=data_args.loss_beta,
            gamma=data_args.loss_gamma,
            tau=data_args.loss_tau,
            ignore_index=-100,
        )
        model.set_loss_function(balanced_loss)
        logger.info(f"Using {data_args.loss_type} loss function")
    elif training_args.do_train:
        logger.info("Using standard CrossEntropyLoss")

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # ==================== 使用统一数据加载器进行 Tokenization ====================
    # 统一数据加载器实现按行边界切分的 tokenization：
    # - 确保一整行不会被截断到两个 chunk 中
    # - 如果当前 chunk 放不下完整的一行，该行会被放到下一个 chunk
    padding = "max_length" if data_args.pad_to_max_length else False

    # 创建数据加载器（包含 line_ids 用于行级评估）
    data_loader = HRDocDataLoader(
        tokenizer=tokenizer,
        config=loader_config,
        include_line_info=True,  # 包含 line_ids 用于行级评估
    )

    # 使用已加载的原始数据集
    data_loader._raw_datasets = datasets

    # 准备 tokenized 数据集
    tokenized_datasets = data_loader.prepare_datasets()

    train_dataset = None
    eval_dataset = None
    test_dataset = None

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        logger.info(f"Train dataset: {len(train_dataset)} samples")

    if training_args.do_eval:
        if "validation" in tokenized_datasets:
            eval_dataset = tokenized_datasets["validation"]
            logger.info(f"Using validation split for evaluation: {len(eval_dataset)} samples")
        else:
            raise ValueError("--do_eval requires a validation or test dataset")

    if training_args.do_predict:
        if "test" in tokenized_datasets:
            test_dataset = tokenized_datasets["test"]
            logger.info(f"Test dataset: {len(test_dataset)} samples")
        else:
            raise ValueError("--do_predict requires a test dataset")

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    # Metrics - use local path if SEQEVAL_PATH is set (for offline mode)
    seqeval_path = os.environ.get("SEQEVAL_PATH", "seqeval")
    metric = load_metric(seqeval_path)

    # Key class pairs to monitor for confusion (使用论文14类标签)
    MONITOR_PAIRS = [
        ("mail", "affili"),
        ("figure", "table"),
        ("caption", "caption"),  # 图表标题混淆
        ("section", "paraline"),
        ("fstline", "paraline"),
    ]

    # =========================================================================
    # compute_metrics: TOKEN 级别评估（用于训练过程中的快速反馈）
    # =========================================================================
    # 注意：此函数计算的是 TOKEN 级别的准确率和 F1，不是 LINE 级别！
    #
    # 原因：HuggingFace Trainer 的 compute_metrics 只接收 (predictions, labels)，
    #       无法直接访问 line_ids，因此无法做行级聚合。
    #
    # 官方行级评估：
    #   - 使用 metrics.line_eval 模块（Single Source of Truth）
    #   - 或使用 util/hrdoc_eval.py 的端到端评估
    #   - 行级评估使用多数投票聚合 token → line
    #
    # Token vs Line 级别指标差异：
    #   - Token 级别会被 paraline/fstline 主导（样本量大）
    #   - Line 级别更能反映少数类的真实表现
    #   - 通常 Token Acc > Line Acc（因为 token 级别有部分正确也算）
    # =========================================================================
    def compute_metrics(p):
        """Token 级别评估（训练快速反馈用，非官方指标）"""
        from collections import Counter, defaultdict

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        # 直接使用标签，不需要 BIO 前缀处理
        true_predictions = [
            [label_list[pr] for (pr, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (pr, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # 计算准确率（Line级别分类）
        total_correct = 0
        total_count = 0
        for preds, gts in zip(true_predictions, true_labels):
            for pred, gt in zip(preds, gts):
                total_count += 1
                if pred == gt:
                    total_correct += 1

        overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

        # === Per-class diagnosis ===
        # Count per-class: TP, FP, FN, predicted count, gt count
        class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "pred_count": 0, "gt_count": 0})
        confusion = defaultdict(Counter)  # {gt_class: {pred_class: count}}

        for preds, gts in zip(true_predictions, true_labels):
            for pred, gt in zip(preds, gts):
                # 直接使用标签（无 BIO 前缀）
                class_stats[gt]["gt_count"] += 1
                class_stats[pred]["pred_count"] += 1
                confusion[gt][pred] += 1

                if pred == gt:
                    class_stats[gt]["tp"] += 1
                else:
                    class_stats[gt]["fn"] += 1
                    class_stats[pred]["fp"] += 1

        # Build final results
        # 计算 macro F1（只计算有样本的类别，避免无样本类别拉低分数）
        f1_scores = []
        f1_scores_all = []  # 包含所有类别（用于兼容性）
        for cls in label_list:
            stats = class_stats[cls]
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            gt_count = stats["gt_count"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores_all.append(f1)
            # 只有当该类别在 GT 或 Pred 中出现过时才计入 macro F1
            if gt_count > 0 or stats["pred_count"] > 0:
                f1_scores.append(f1)

        # 使用有样本类别的 macro F1（更合理的评估）
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        # 也计算包含所有类别的 macro F1（用于对比）
        macro_f1_all = np.mean(f1_scores_all)

        final_results = {
            "accuracy": overall_accuracy,
            "macro_f1": macro_f1,
            "macro_f1_all": macro_f1_all,  # 包含所有14类（用于对比）
        }

        # Add per-class metrics for all 14 classes
        for cls in label_list:
            stats = class_stats.get(cls, {"tp": 0, "fp": 0, "fn": 0, "pred_count": 0, "gt_count": 0})
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            final_results[f"{cls}_precision"] = precision
            final_results[f"{cls}_recall"] = recall
            final_results[f"{cls}_f1"] = f1
            final_results[f"{cls}_gt_count"] = stats["gt_count"]
            final_results[f"{cls}_pred_count"] = stats["pred_count"]

        # Add confusion metrics for monitored pairs
        for cls_a, cls_b in MONITOR_PAIRS:
            if cls_a == cls_b:
                continue
            # How often cls_a is predicted as cls_b
            a_to_b = confusion[cls_a].get(cls_b, 0)
            a_total = class_stats[cls_a]["gt_count"]
            a_to_b_rate = a_to_b / a_total if a_total > 0 else 0.0
            final_results[f"confusion_{cls_a}_to_{cls_b}"] = a_to_b_rate

            # How often cls_b is predicted as cls_a (reverse)
            b_to_a = confusion[cls_b].get(cls_a, 0)
            b_total = class_stats[cls_b]["gt_count"]
            b_to_a_rate = b_to_a / b_total if b_total > 0 else 0.0
            final_results[f"confusion_{cls_b}_to_{cls_a}"] = b_to_a_rate

        # Log per-class summary (for visibility in training logs)
        logger.info("=" * 60)
        logger.info("Per-Class Metrics [TOKEN-LEVEL] (14 classes):")
        logger.info("⚠️  注意：这是 TOKEN 级别指标，LINE 级别请看端到端评估")
        logger.info(f"{'Class':<12} {'Prec':>7} {'Recall':>7} {'F1':>7} {'GT':>6} {'Pred':>6}")
        logger.info("-" * 55)
        for cls in label_list:
            stats = class_stats[cls]
            if stats["gt_count"] > 0 or stats["pred_count"] > 0:
                tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                logger.info(f"{cls:<12} {prec:>7.1%} {rec:>7.1%} {f1:>7.1%} {stats['gt_count']:>6} {stats['pred_count']:>6}")

        logger.info("-" * 55)
        logger.info(f"[TOKEN] Overall Accuracy: {overall_accuracy:.1%}")
        logger.info(f"[TOKEN] Macro F1: {macro_f1:.1%} (only classes with samples)")
        logger.info(f"[TOKEN] Macro F1 (all 14): {macro_f1_all:.1%} (includes zero-sample classes)")
        logger.info("-" * 55)
        logger.info("Confusion pairs (GT -> Pred rate):")
        for cls_a, cls_b in MONITOR_PAIRS:
            if cls_a == cls_b:
                continue
            a_to_b = confusion[cls_a].get(cls_b, 0)
            a_total = class_stats[cls_a]["gt_count"]
            rate = a_to_b / a_total if a_total > 0 else 0.0
            if a_total > 0:
                logger.info(f"  {cls_a} -> {cls_b}: {rate:.1%} ({a_to_b}/{a_total})")
        logger.info("=" * 60)

        return final_results

    # Setup callbacks
    callbacks = []

    # Add tokenizer save callback to ensure tokenizer.json is saved to checkpoints
    callbacks.append(TokenizerSaveCallback(tokenizer))
    logger.info("TokenizerSaveCallback enabled")

    # Early stopping (optional)
    if training_args.load_best_model_at_end:
        try:
            from transformers import EarlyStoppingCallback
            # Early stopping with patience of 5 evaluations
            early_stopping_patience = int(os.environ.get("EARLY_STOPPING_PATIENCE", "5"))
            if early_stopping_patience > 0:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
                logger.info(f"EarlyStoppingCallback enabled with patience={early_stopping_patience}")
        except ImportError:
            logger.warning("EarlyStoppingCallback not available in this transformers version")

    # LINE 级别诊断回调（每次评估后运行）
    if eval_dataset is not None:
        callbacks.append(LineLevelEvalCallback(eval_dataset, data_collator, label_list))
        logger.info("LineLevelEvalCallback enabled for TOKEN vs LINE diagnostics")

    # Setup class-balanced batch sampler (Step 3) if enabled
    train_sampler = None
    if training_args.do_train and data_args.use_class_balanced_sampler:
        from layoutlmft.data.class_balanced_sampler import (
            ClassBalancedBatchSampler,
            get_hrdoc_rare_classes,
        )

        # Get rare class IDs
        rare_classes = get_hrdoc_rare_classes(label_list)
        logger.info(f"Class-balanced sampling enabled with {len(rare_classes)} rare classes")

        # Create sampler
        train_sampler = ClassBalancedBatchSampler(
            dataset=train_dataset,
            label_column="labels",
            batch_size=training_args.per_device_train_batch_size,
            rare_classes=rare_classes,
            rare_ratio=data_args.rare_class_ratio,
            drop_last=training_args.dataloader_drop_last,
            seed=training_args.seed,
        )

    # Initialize our Trainer with label_list for per-class monitoring
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        label_list=label_list,  # Pass label_list for per-class loss monitoring
        train_sampler=train_sampler,  # Custom sampler for class-balanced batching
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        # Ensure both tokenizer formats are saved to output directory
        if not os.path.exists(os.path.join(training_args.output_dir, "tokenizer.json")):
            tokenizer.save_pretrained(training_args.output_dir, legacy_format=True)
            tokenizer.save_pretrained(training_args.output_dir, legacy_format=False)
            logger.info(f"Saved tokenizer (both formats) to {training_args.output_dir}")

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # =========================================================================
        # 行级评估（与 token 级评估对比，验证一致性）
        # =========================================================================
        logger.info("*** Line-Level Evaluation ***")
        try:
            from metrics.line_eval import compute_line_level_metrics_batch
            from torch.utils.data import DataLoader

            # 创建评估 DataLoader
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=data_collator,
            )

            model.eval()
            all_token_preds = []
            all_token_labels = []
            all_line_ids = []

            with torch.no_grad():
                for batch in eval_dataloader:
                    device = training_args.device
                    # 移动普通 tensor 到 device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    # 处理 image（ImageList 类型需要特殊处理）
                    image = batch.get("image")
                    if image is not None:
                        if hasattr(image, 'tensor'):
                            # detectron2 的 ImageList（使用 .tensor 单数）
                            from detectron2.structures import ImageList as D2ImageList
                            image = D2ImageList(image.tensor.to(device), image.image_sizes)
                        elif hasattr(image, 'tensors'):
                            # shim ImageList（使用 .tensors 复数）
                            class SimpleImageList:
                                def __init__(self, tensors, image_sizes):
                                    self.tensors = tensors
                                    self.image_sizes = image_sizes
                            image = SimpleImageList(image.tensors.to(device), image.image_sizes)
                        elif isinstance(image, torch.Tensor):
                            image = image.to(device)

                    outputs = model(
                        input_ids=batch["input_ids"],
                        bbox=batch["bbox"],
                        attention_mask=batch["attention_mask"],
                        image=image,
                    )
                    logits = outputs.logits  # [batch, seq_len, num_classes]

                    # 获取预测
                    preds = logits.argmax(dim=-1)  # [batch, seq_len]

                    # 收集数据
                    for b in range(preds.shape[0]):
                        all_token_preds.append(preds[b].cpu().tolist())
                        all_token_labels.append(batch["labels"][b].cpu().tolist())
                        if "line_ids" in batch:
                            all_line_ids.append(batch["line_ids"][b].cpu().tolist())

            # 计算行级指标
            if all_line_ids:
                line_metrics = compute_line_level_metrics_batch(
                    batch_token_predictions=all_token_preds,
                    batch_token_labels=all_token_labels,
                    batch_line_ids=all_line_ids,
                    num_classes=len(label_list),
                    class_names=label_list,
                )
                line_metrics.log_summary(class_names=label_list, title="Line-Level Metrics (对比验证)")

                logger.info("=" * 65)
                logger.info("Token vs Line 指标对比:")
                logger.info(f"  [TOKEN] Accuracy: {metrics.get('eval_accuracy', 0):.2%}")
                logger.info(f"  [LINE]  Accuracy: {line_metrics.accuracy:.2%}")
                logger.info(f"  [TOKEN] Macro-F1: {metrics.get('eval_macro_f1', 0):.2%}")
                logger.info(f"  [LINE]  Macro-F1: {line_metrics.macro_f1:.2%}")
                logger.info("=" * 65)
            else:
                logger.warning("line_ids not available, skipping line-level evaluation")

        except Exception as e:
            logger.warning(f"Line-level evaluation failed: {e}")

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

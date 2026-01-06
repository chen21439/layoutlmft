#!/usr/bin/env python
# coding=utf-8
"""
HRDoc Line-Level Training Script

Stage 1 独立训练脚本（Line-level 分类），与联合训练的 Stage 1 逻辑完全对齐。

特点：
- 使用 LayoutXLMForLineLevelClassification（mean pooling）
- 使用 LineLevelDataCollator 提供 line_ids 和 line_labels
- 损失和评估都在 line-level 进行

与 JointModel 的对齐：
- 使用相同的 LinePooling 模块
- 使用相同的 LineClassificationHead
- 使用相同的损失计算方式

注意：推荐使用 train_joint.py --mode stage1 进行训练。
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import layoutlmft.data.datasets.hrdoc
import transformers
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

# 添加项目路径
STAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE_ROOT)

# 导入 line-level 模型和数据加载器
from data import HRDocDataLoader, HRDocDataLoaderConfig, load_hrdoc_raw_datasets
from data.line_level_collator import LineLevelDataCollator
from models.stage1_line_level_model import LayoutXLMForLineLevelClassification

# Register both layoutxlm and layoutlmv2
CONFIG_MAPPING.update({
    "layoutxlm": LayoutXLMConfig,
    "layoutlmv2": LayoutXLMConfig,
})

check_min_version("4.5.0")

logger = logging.getLogger(__name__)


class TokenizerSaveCallback(TrainerCallback):
    """Callback to explicitly save tokenizer to each checkpoint directory."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        """Called after checkpoint is saved."""
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_dir):
            if not os.path.exists(os.path.join(checkpoint_dir, "tokenizer.json")):
                self.tokenizer.save_pretrained(checkpoint_dir, legacy_format=True)
                self.tokenizer.save_pretrained(checkpoint_dir, legacy_format=False)
                logger.info(f"Saved tokenizer (both formats) to {checkpoint_dir}")


class LineLevelEvalCallback(TrainerCallback):
    """
    Line-level 评估回调

    在每次评估后打印 line-level 指标
    """

    def __init__(self, eval_dataset, data_collator, label_list):
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.label_list = label_list

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """在 Trainer.evaluate() 后运行"""
        if model is None:
            return

        try:
            from torch.utils.data import DataLoader
            from metrics.line_eval import compute_line_level_metrics_batch

            logger.info("")
            logger.info("=" * 65)
            logger.info(f"[LINE-LEVEL EVAL] Step {state.global_step}")
            logger.info("=" * 65)

            # 创建小批量评估
            max_samples = min(500, len(self.eval_dataset))
            subset = torch.utils.data.Subset(self.eval_dataset, range(max_samples))
            eval_dataloader = DataLoader(
                subset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
            )

            device = next(model.parameters()).device
            model.eval()

            # 收集预测和标签
            all_line_preds = []
            all_line_labels = []

            with torch.no_grad():
                for batch in eval_dataloader:
                    # 移动到 device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    # 前向传播
                    outputs = model(**batch)
                    logits = outputs.logits  # [batch, max_lines, num_classes]

                    # 获取预测
                    preds = logits.argmax(dim=-1)  # [batch, max_lines]

                    # 收集数据
                    if "line_labels" in batch:
                        for b in range(preds.shape[0]):
                            sample_preds = preds[b].cpu().tolist()
                            sample_labels = batch["line_labels"][b].cpu().tolist()

                            # 只保留有效的（label != -100）
                            valid_preds = []
                            valid_labels = []
                            for pred, label in zip(sample_preds, sample_labels):
                                if label != -100:
                                    valid_preds.append(pred)
                                    valid_labels.append(label)

                            if valid_preds:
                                all_line_preds.extend(valid_preds)
                                all_line_labels.extend(valid_labels)

            # 计算指标
            if all_line_preds and all_line_labels:
                # 简单的准确率和 macro F1
                correct = sum(p == l for p, l in zip(all_line_preds, all_line_labels))
                accuracy = correct / len(all_line_labels)

                # Per-class F1
                from collections import defaultdict
                class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

                for pred, label in zip(all_line_preds, all_line_labels):
                    if pred == label:
                        class_stats[label]["tp"] += 1
                    else:
                        class_stats[label]["fn"] += 1
                        class_stats[pred]["fp"] += 1

                f1_scores = []
                for cls_id in range(len(self.label_list)):
                    stats = class_stats[cls_id]
                    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    f1_scores.append(f1)

                macro_f1 = np.mean(f1_scores)

                logger.info(f"[LINE-LEVEL] Accuracy: {accuracy:.2%}")
                logger.info(f"[LINE-LEVEL] Macro-F1: {macro_f1:.2%}")
                logger.info(f"[LINE-LEVEL] Samples:  {len(all_line_labels)}")
            else:
                logger.warning("[LINE-LEVEL] No valid samples to evaluate")

            logger.info("=" * 65)

        except Exception as e:
            logger.warning(f"[LINE-LEVEL EVAL] Failed: {e}")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 保留 line_ids 列用于行级评估
    training_args.remove_unused_columns = False

    # Detecting last checkpoint
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

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    # 使用统一数据加载器加载数据集
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

    # 加载原始数据集
    datasets = load_hrdoc_raw_datasets(
        data_dir=os.environ.get("HRDOC_DATA_DIR"),
        dataset_name=dataset_name,
        force_rebuild=data_args.force_rebuild,
    )

    # 获取 label 信息
    label_list = LABEL_LIST
    num_labels = NUM_LABELS
    label_to_id = get_label2id()
    id2label = get_id2label()

    logger.info(f"Using {num_labels} labels: {label_list}")
    logger.info("Mode: LINE-LEVEL classification (mean pooling)")

    # Load config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label_to_id,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load tokenizer
    model_type = getattr(config, "model_type", None)

    if model_type == "layoutxlm":
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        logger.info("Using LayoutXLMTokenizerFast")
    elif model_type == "layoutlmv2":
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

    # Load backbone model first
    backbone_model = LayoutXLMForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Wrap with line-level model
    model = LayoutXLMForLineLevelClassification(
        backbone_model=backbone_model,
        num_classes=num_labels,
        hidden_size=config.hidden_size,
        cls_dropout=0.1,
    )

    logger.info("Created LayoutXLMForLineLevelClassification wrapper")

    # Tokenizer check
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer."
        )

    # 使用统一数据加载器
    data_loader = HRDocDataLoader(
        tokenizer=tokenizer,
        config=loader_config,
        include_line_info=True,  # 包含 line_ids
    )

    data_loader._raw_datasets = datasets
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
            logger.info(f"Eval dataset: {len(eval_dataset)} samples")
        else:
            raise ValueError("--do_eval requires a validation dataset")

    if training_args.do_predict:
        if "test" in tokenized_datasets:
            test_dataset = tokenized_datasets["test"]
            logger.info(f"Test dataset: {len(test_dataset)} samples")
        else:
            raise ValueError("--do_predict requires a test dataset")

    # Data collator - use LineLevelDataCollator
    data_collator = LineLevelDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=True,
        max_length=512,
    )

    logger.info("Using LineLevelDataCollator")

    # Metrics
    def compute_metrics(p):
        """Line-level 评估指标"""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)  # [batch, max_lines]

        # 展平并过滤 -100
        all_preds = []
        all_labels = []
        for pred_seq, label_seq in zip(predictions, labels):
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    all_preds.append(pred)
                    all_labels.append(label)

        # 计算准确率
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_labels) if all_labels else 0.0

        # 计算 macro F1
        from collections import defaultdict
        class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for pred, label in zip(all_preds, all_labels):
            if pred == label:
                class_stats[label]["tp"] += 1
            else:
                class_stats[label]["fn"] += 1
                class_stats[pred]["fp"] += 1

        f1_scores = []
        for cls_id in range(num_labels):
            stats = class_stats[cls_id]
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = np.mean(f1_scores)

        logger.info("=" * 60)
        logger.info("[LINE-LEVEL] Metrics:")
        logger.info(f"  Accuracy: {accuracy:.2%}")
        logger.info(f"  Macro-F1: {macro_f1:.2%}")
        logger.info("=" * 60)

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

    # Setup callbacks
    callbacks = []
    callbacks.append(TokenizerSaveCallback(tokenizer))

    if eval_dataset is not None:
        callbacks.append(LineLevelEvalCallback(eval_dataset, data_collator, label_list))

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()

        # Save tokenizer
        if not os.path.exists(os.path.join(training_args.output_dir, "tokenizer.json")):
            tokenizer.save_pretrained(training_args.output_dir, legacy_format=True)
            tokenizer.save_pretrained(training_args.output_dir, legacy_format=False)

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

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(test_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()

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
from layoutlmft.models.layoutxlm import LayoutXLMTokenizer, LayoutXLMTokenizerFast
import layoutlmft
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Register both layoutxlm and layoutlmv2 (LayoutXLM uses layoutlmv2 as model_type in config.json)
CONFIG_MAPPING.update({
    "layoutxlm": LayoutXLMConfig,
    "layoutlmv2": LayoutXLMConfig,  # LayoutXLM's config.json has model_type="layoutlmv2"
})

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


class TokenizerCopyCallback(TrainerCallback):
    """Callback to copy tokenizer.json to each checkpoint directory.

    LayoutXLMTokenizerFast requires tokenizer.json, but HuggingFace Trainer
    doesn't save it to checkpoint directories by default.
    """

    def __init__(self, src_tokenizer_json: str):
        self.src_tokenizer_json = src_tokenizer_json

    def on_save(self, args, state, control, **kwargs):
        """Called after checkpoint is saved."""
        if not os.path.exists(self.src_tokenizer_json):
            return

        # Find the latest checkpoint directory
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_dir):
            dst_tokenizer_json = os.path.join(checkpoint_dir, "tokenizer.json")
            if not os.path.exists(dst_tokenizer_json):
                shutil.copy(self.src_tokenizer_json, dst_tokenizer_json)
                logger.info(f"Copied tokenizer.json to {checkpoint_dir}")


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

    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))

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

    # Determine if this is LayoutXLM or LayoutLMv2 by checking for sentencepiece model
    # LayoutXLM uses XLMRoberta tokenizer (sentencepiece), LayoutLMv2 uses BERT tokenizer (vocab.txt)
    model_path = model_args.model_name_or_path
    is_layoutxlm = os.path.exists(os.path.join(model_path, "sentencepiece.bpe.model")) or \
                   "layoutxlm" in model_path.lower()

    if is_layoutxlm:
        # LayoutXLM uses XLMRoberta tokenizer (sentencepiece)
        # Try fast tokenizer first, fall back to slow tokenizer if tokenizer.json not found
        tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            tokenizer = LayoutXLMTokenizerFast.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
            )
            logger.info("Using LayoutXLM fast tokenizer (XLMRoberta/sentencepiece)")
        else:
            # Checkpoint doesn't have tokenizer.json, use slow tokenizer
            tokenizer = LayoutXLMTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
            )
            logger.info("Using LayoutXLM slow tokenizer (tokenizer.json not found in checkpoint)")
    elif getattr(config, "model_type", None) == "layoutlmv2":
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

    # Setup balanced loss for long-tailed classification
    if data_args.loss_type != "ce":
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
    else:
        logger.info("Using standard CrossEntropyLoss")

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=512,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            image = examples["image"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    # label can be either a string (from raw data) or an int (from ClassLabel feature)
                    lbl = label[word_idx]
                    label_id = lbl if isinstance(lbl, int) else label_to_id[lbl]
                    label_ids.append(label_id)
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        lbl = label[word_idx]
                        label_id = lbl if isinstance(lbl, int) else label_to_id[lbl]
                        label_ids.append(label_id)
                    else:
                        label_ids.append(-100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            images.append(image)
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["image"] = images
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        # Try validation split first, fall back to test split
        if "validation" in datasets:
            eval_dataset = datasets["validation"]
            logger.info("Using validation split for evaluation")
        elif "test" in datasets:
            eval_dataset = datasets["test"]
            logger.warning("No validation split found, using test split for evaluation during training")
        else:
            raise ValueError("--do_eval requires a validation or test dataset")

        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

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

    def compute_metrics(p):
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
        # 计算 macro F1
        f1_scores = []
        for cls in label_list:
            stats = class_stats[cls]
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = np.mean(f1_scores)

        final_results = {
            "accuracy": overall_accuracy,
            "macro_f1": macro_f1,
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
        logger.info("Per-Class Metrics (14 classes):")
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
        logger.info(f"Overall Accuracy: {overall_accuracy:.1%}")
        logger.info(f"Macro F1: {macro_f1:.1%}")
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

    # Add tokenizer copy callback for LayoutXLM (copies tokenizer.json to each checkpoint)
    src_tokenizer_json = os.path.join(model_args.model_name_or_path, "tokenizer.json")
    if os.path.exists(src_tokenizer_json):
        callbacks.append(TokenizerCopyCallback(src_tokenizer_json))
        logger.info(f"TokenizerCopyCallback enabled, will copy tokenizer.json to checkpoints")

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

        # 确保 tokenizer.json 被复制到输出目录（LayoutXLM TokenizerFast 需要）
        src_tokenizer_json = os.path.join(model_args.model_name_or_path, "tokenizer.json")
        dst_tokenizer_json = os.path.join(training_args.output_dir, "tokenizer.json")
        if os.path.exists(src_tokenizer_json) and not os.path.exists(dst_tokenizer_json):
            shutil.copy(src_tokenizer_json, dst_tokenizer_json)
            logger.info(f"Copied tokenizer.json to {dst_tokenizer_json}")

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

#!/usr/bin/env python
# coding=utf-8
"""快速评估 HRDoc 模型"""

import logging
import os
import sys
import numpy as np
from datasets import load_dataset, load_metric

import layoutlmft.data.datasets.hrdoc_test
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.trainers import FunsdTrainer as Trainer
from transformers import (
    BertTokenizerFast,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForTokenClassification, LayoutLMv2Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

CONFIG_MAPPING.update({"layoutlmv2": LayoutLMv2Config})

logger = logging.getLogger(__name__)

def main():
    # 简化的参数
    # 获取项目根目录（脚本在 examples/ 下，根目录是上一级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    model_path = os.path.join(project_root, "output/hrdoc_test")
    output_dir = os.path.join(project_root, "output/hrdoc_eval")

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_predict=True,
        per_device_eval_batch_size=4,
        overwrite_output_dir=True,
        logging_steps=50,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    set_seed(training_args.seed)

    # 加载测试数据集
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc_test.__file__))

    if "test" not in datasets:
        raise ValueError("--do_predict requires a test dataset")

    test_dataset = datasets["test"]
    column_names = test_dataset.column_names
    features = test_dataset.features
    text_column_name = "tokens"
    label_column_name = "ner_tags"

    # 获取标签列表
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
    num_labels = len(label_list)

    # 加载模型和tokenizer
    config = LayoutLMv2Config.from_pretrained(model_path, num_labels=num_labels)

    tokenizer = BertTokenizerFast.from_pretrained(model_path)

    model = LayoutLMv2ForTokenClassification.from_pretrained(
        model_path,
        config=config,
    )

    # Tokenize and align labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_overflowing_tokens=True,
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
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
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

    # 处理测试数据
    test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=column_names,
        num_proc=1,
    )

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )

    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Predict
    logger.info("*** Predict ***")
    predictions, labels, metrics = trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=2)

    # Log metrics
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    print("\n=== 测试结果 ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()

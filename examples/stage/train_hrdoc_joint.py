#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 联合训练脚本 - 端到端多任务学习
实现论文中的联合训练：L_total = L_cls + α₁·L_par + α₂·L_rel

三个子任务：
1. SubTask1: 语义单元分类（token-level）
2. SubTask2: 父节点查找（line-level）
3. SubTask3: 关系分类（edge-level）
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset, load_metric

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import layoutlmft.data.datasets.hrdoc
from layoutlmft.models.layoutlmv2 import LayoutLMv2Config
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments

from hrdoc_joint_model import HRDocJointModel
from joint_data_collator import HRDocJointDataCollator, SEMANTIC_CLASS2ID
from train_parent_finder import ChildParentDistributionMatrix

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

CONFIG_MAPPING.update({"layoutlmv2": LayoutLMv2Config})

# 检查transformers版本
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


@dataclass
class JointModelArguments(ModelArguments):
    """联合模型的参数"""

    # 语义类别数（line-level）
    num_semantic_classes: int = field(
        default=21,
        metadata={"help": "Number of semantic classes (line-level, without BIO prefix)"}
    )

    # 关系类别数
    num_relations: int = field(
        default=5,
        metadata={"help": "Number of relation types"}
    )

    # GRU隐藏层大小
    gru_hidden_size: int = field(
        default=512,
        metadata={"help": "GRU hidden size for parent finder"}
    )

    # 是否使用soft-mask
    use_soft_mask: bool = field(
        default=True,
        metadata={"help": "Whether to use soft-mask for parent finder"}
    )

    # Loss权重
    alpha1: float = field(
        default=1.0,
        metadata={"help": "Weight for parent finding loss"}
    )

    alpha2: float = field(
        default=1.0,
        metadata={"help": "Weight for relation classification loss"}
    )


def main():
    # ==================== 1. 参数解析 ====================
    parser = HfArgumentParser((JointModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ==================== 2. Logging设置 ====================
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log信息
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f", distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")

    # ==================== 3. Checkpoint检测 ====================
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

    # ==================== 4. 设置随机种子 ====================
    set_seed(training_args.seed)

    # ==================== 5. 加载数据集 ====================
    logger.info("Loading dataset...")
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    # 获取标签列表
    if isinstance(features["ner_tags"].feature, list):
        label_list = features["ner_tags"].feature[0].names
    else:
        label_list = features["ner_tags"].feature.names
    label_to_id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    logger.info(f"Number of labels (token-level): {num_labels}")
    logger.info(f"Number of semantic classes (line-level): {model_args.num_semantic_classes}")
    logger.info(f"Number of relations: {model_args.num_relations}")

    # ==================== 6. 加载Tokenizer ====================
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
    )

    # ==================== 7. 加载模型 ====================
    logger.info("Loading model...")

    # 配置
    config = LayoutLMv2Config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    # 添加联合训练的配置
    config.num_semantic_classes = model_args.num_semantic_classes
    config.num_relations = model_args.num_relations
    config.gru_hidden_size = model_args.gru_hidden_size
    config.use_soft_mask = model_args.use_soft_mask
    config.alpha1 = model_args.alpha1
    config.alpha2 = model_args.alpha2

    # 加载模型
    if last_checkpoint:
        model_path = last_checkpoint
        logger.info(f"Loading from checkpoint: {model_path}")
    else:
        model_path = model_args.model_name_or_path

    model = HRDocJointModel.from_pretrained(
        model_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # ==================== 8. 构建Child-Parent Distribution Matrix ====================
    if model_args.use_soft_mask and training_args.do_train:
        logger.info("Building Child-Parent Distribution Matrix from training data...")
        cp_matrix = ChildParentDistributionMatrix(
            num_classes=model_args.num_semantic_classes,
            pseudo_count=5
        )

        # 统计训练集中的parent-child关系
        for example in datasets["train"]:
            line_semantic_labels = example.get("line_semantic_labels", [])
            line_parent_ids = example.get("line_parent_ids", [])

            for i, (child_label, parent_id) in enumerate(zip(line_semantic_labels, line_parent_ids)):
                if parent_id == -1:
                    parent_label = -1  # ROOT
                elif parent_id < len(line_semantic_labels):
                    parent_label = line_semantic_labels[parent_id]
                else:
                    continue

                cp_matrix.update(child_label, parent_label)

        cp_matrix.build()
        model.set_child_parent_matrix(cp_matrix.get_tensor(device=training_args.device))
        logger.info("Child-Parent Distribution Matrix built successfully")

    # ==================== 9. Data Collator ====================
    data_collator = HRDocJointDataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=512,
        label_pad_token_id=-100,
    )

    # ==================== 10. Metrics ====================
    def compute_metrics(p):
        """计算三个任务的评估指标"""
        predictions, labels = p
        # TODO: 实现三个任务的评估指标
        # 1. Token-level F1 for semantic classification
        # 2. Accuracy for parent finding
        # 3. F1 for relation classification
        return {}

    # ==================== 11. Trainer ====================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"] if training_args.do_train else None,
        eval_dataset=datasets["test"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ==================== 12. 训练 ====================
    if training_args.do_train:
        logger.info("*** Train ***")

        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        trainer.save_model()

        metrics["train_samples"] = len(datasets["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ==================== 13. 评估 ====================
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ==================== 14. 预测 ====================
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(datasets["test"], metric_key_prefix="predict")

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()

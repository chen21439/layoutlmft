#!/usr/bin/env python
# coding=utf-8
"""
从已训练的 LayoutLMv2 模型提取行级特征并缓存到磁盘
这样可以快速迭代关系分类器，不需要每次都运行大模型
"""

import logging
import os
import sys
import torch
import pickle
from tqdm import tqdm
from datasets import load_dataset

import layoutlmft.data.datasets.hrdoc
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.models.relation_classifier import LineFeatureExtractor
from transformers import (
    BertTokenizerFast,
    set_seed,
)
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForTokenClassification, LayoutLMv2Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

CONFIG_MAPPING.update({"layoutlmv2": LayoutLMv2Config})

logger = logging.getLogger(__name__)


def main():
    # 配置
    # 获取项目根目录（脚本在 examples/ 下，根目录是上一级）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 模型路径（优先从环境变量读取，默认使用 E 盘训练的模型）
    model_path = os.getenv("LAYOUTLMFT_MODEL_PATH", "/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000")
    # 特征输出到 E 盘节省系统盘空间
    output_dir = os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features")
    # 数据集路径（优先从环境变量读取）
    data_dir = os.getenv("HRDOC_DATA_DIR", "/mnt/e/models/data/Section/HRDS")

    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"数据集路径: {data_dir}")
    logger.info(f"模型路径: {model_path}")

    # 设置数据集路径环境变量（传递给 hrdoc.py）
    os.environ["HRDOC_DATA_DIR"] = data_dir

    # 加载数据集
    logger.info("加载数据集...")
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))

    # 检查是否有 train 和 test 集
    has_train = "train" in datasets
    has_validation = "test" in datasets
    logger.info(f"数据集包含: {list(datasets.keys())}")

    # 是否限制数据量（默认-1表示不限制，提取所有数据）
    num_samples = int(os.getenv("LAYOUTLMFT_NUM_SAMPLES", "-1"))
    if num_samples > 0:
        logger.info(f"限制样本数: {num_samples}")
        if has_train and len(datasets["train"]) > num_samples:
            datasets["train"] = datasets["train"].select(range(num_samples))
        if has_validation and len(datasets["test"]) > num_samples:
            datasets["test"] = datasets["test"].select(range(num_samples))
    else:
        logger.info("提取所有数据（不限制样本数）")

    train_samples = len(datasets.get('train', [])) if has_train else 0
    val_samples = len(datasets.get('test', [])) if has_validation else 0
    logger.info(f"实际样本数 - train: {train_samples}, validation: {val_samples}")

    # 加载模型和tokenizer
    logger.info("加载模型...")
    config = LayoutLMv2Config.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_path, config=config)
    model = model.to(device)
    model.eval()

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )

    # 分批次提取特征的函数
    def extract_features_in_batches(raw_dataset, split_name, batch_size=50, samples_per_chunk=1000):
        """分批次提取特征，并分块保存避免一次性加载所有数据

        Args:
            raw_dataset: 原始数据集
            split_name: 数据集名称（train/validation）
            batch_size: 每批次处理的样本数
            samples_per_chunk: 每个chunk文件保存的样本数

        Returns:
            chunk_files: 保存的chunk文件路径列表
        """
        import numpy as np

        logger.info(f"开始分批次提取 {split_name} 集的行级特征...")
        logger.info(f"总样本数: {len(raw_dataset)}, 批次大小: {batch_size}")
        logger.info(f"每 {samples_per_chunk} 个样本保存一个chunk文件")

        feature_extractor = LineFeatureExtractor()
        chunk_features = []  # 当前chunk的特征
        chunk_files = []  # 已保存的chunk文件列表
        chunk_idx = 0  # chunk编号

        num_batches = (len(raw_dataset) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(raw_dataset))

                logger.info(f"处理批次 {batch_idx+1}/{num_batches} (样本 {start_idx}-{end_idx-1})")

                # 处理当前批次的每个样本
                for idx in tqdm(range(start_idx, end_idx), desc=f"批次{batch_idx+1}"):
                    # 获取原始样本
                    raw_sample = raw_dataset[idx]

                    # Tokenize单个样本
                    tokenized = tokenizer(
                        raw_sample["tokens"],
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        is_split_into_words=True,
                    )

                    # 对齐标签和bbox
                    word_ids = tokenized.word_ids()

                    labels = []
                    bboxes = []
                    token_line_ids = []

                    previous_word_idx = None
                    for word_idx in word_ids:
                        if word_idx is None:
                            labels.append(-100)
                            bboxes.append([0, 0, 0, 0])
                            token_line_ids.append(-1)
                        elif word_idx != previous_word_idx:
                            labels.append(raw_sample["ner_tags"][word_idx])
                            bboxes.append(raw_sample["bboxes"][word_idx])
                            token_line_ids.append(raw_sample["line_ids"][word_idx])
                        else:
                            labels.append(-100)
                            bboxes.append(raw_sample["bboxes"][word_idx])
                            token_line_ids.append(raw_sample["line_ids"][word_idx])
                        previous_word_idx = word_idx

                    # 计算行级 bbox 和 labels
                    valid_line_ids = [lid for lid in token_line_ids if lid >= 0]
                    if len(valid_line_ids) > 0:
                        max_line_id = max(valid_line_ids)
                        num_lines = max_line_id + 1

                        line_bboxes = np.zeros((num_lines, 4), dtype=np.float32)
                        line_bboxes[:, 0] = 1e9
                        line_bboxes[:, 1] = 1e9
                        line_bboxes[:, 2] = -1e9
                        line_bboxes[:, 3] = -1e9

                        # 收集每个line的所有标签（用于多数投票）
                        from collections import defaultdict, Counter
                        line_label_votes = defaultdict(list)

                        for bbox, label, lid in zip(bboxes, labels, token_line_ids):
                            if lid < 0:
                                continue
                            # 更新bbox
                            x1, y1, x2, y2 = bbox
                            line_bboxes[lid, 0] = min(line_bboxes[lid, 0], x1)
                            line_bboxes[lid, 1] = min(line_bboxes[lid, 1], y1)
                            line_bboxes[lid, 2] = max(line_bboxes[lid, 2], x2)
                            line_bboxes[lid, 3] = max(line_bboxes[lid, 3], y2)

                            # 收集标签（忽略特殊标签-100）
                            if label != -100:
                                line_label_votes[lid].append(label)

                        # 计算每个line的最终标签（多数投票）
                        line_labels = []
                        for lid in range(num_lines):
                            if lid in line_label_votes and len(line_label_votes[lid]) > 0:
                                # 使用多数投票
                                most_common_label = Counter(line_label_votes[lid]).most_common(1)[0][0]
                                line_labels.append(most_common_label)
                            else:
                                # 如果没有有效标签，使用-1
                                line_labels.append(-1)
                    else:
                        line_bboxes = np.zeros((0, 4), dtype=np.float32)
                        line_labels = []

                    # 准备模型输入
                    sample_dict = {
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"],
                        "bbox": bboxes,
                        "labels": labels,
                        "image": raw_sample["image"],
                    }

                    batch = data_collator([sample_dict])

                    # 转移到GPU
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                        elif k == "image" and hasattr(v, "to"):
                            batch[k] = v.to(device)

                    # 前向传播
                    outputs = model(
                        input_ids=batch["input_ids"],
                        bbox=batch["bbox"],
                        attention_mask=batch["attention_mask"],
                        image=batch["image"],
                        output_hidden_states=True,
                    )

                    # 提取hidden states
                    hidden_states = outputs.hidden_states[-1]
                    text_seq_len = batch["input_ids"].shape[1]
                    hidden_states = hidden_states[:, :text_seq_len, :]

                    # 提取行级特征
                    line_ids_tensor = torch.tensor(token_line_ids, device=device).unsqueeze(0)
                    line_features, line_mask = feature_extractor.extract_line_features(
                        hidden_states, line_ids_tensor, pooling="mean"
                    )

                    # 保存到当前chunk
                    page_data = {
                        "line_features": line_features.cpu(),
                        "line_mask": line_mask.cpu(),
                        "line_parent_ids": raw_sample["line_parent_ids"],
                        "line_relations": raw_sample["line_relations"],
                        "line_bboxes": line_bboxes,
                        "line_labels": line_labels,  # 新增：行级语义标签
                        "page_idx": idx,
                    }

                    chunk_features.append(page_data)

                    # 检查是否需要保存当前chunk
                    if len(chunk_features) >= samples_per_chunk:
                        chunk_file = os.path.join(output_dir, f"{split_name}_line_features_chunk_{chunk_idx:04d}.pkl")
                        logger.info(f"保存chunk {chunk_idx} 到 {chunk_file} ({len(chunk_features)} 页)")
                        with open(chunk_file, "wb") as f:
                            pickle.dump(chunk_features, f)
                        chunk_files.append(chunk_file)
                        chunk_features = []  # 清空当前chunk
                        chunk_idx += 1

                logger.info(f"批次 {batch_idx+1} 完成，已提取 {chunk_idx * samples_per_chunk + len(chunk_features)} 页特征")

        # 保存剩余的特征（最后一个不完整的chunk）
        if len(chunk_features) > 0:
            chunk_file = os.path.join(output_dir, f"{split_name}_line_features_chunk_{chunk_idx:04d}.pkl")
            logger.info(f"保存最后的chunk {chunk_idx} 到 {chunk_file} ({len(chunk_features)} 页)")
            with open(chunk_file, "wb") as f:
                pickle.dump(chunk_features, f)
            chunk_files.append(chunk_file)

        return chunk_files

    # 提取特征（分批次 + 分块保存）
    batch_size = int(os.getenv("LAYOUTLMFT_BATCH_SIZE", "50"))
    samples_per_chunk = int(os.getenv("LAYOUTLMFT_SAMPLES_PER_CHUNK", "1000"))

    train_chunk_files = []
    # 如果有训练集，提取训练集特征
    if has_train:
        train_chunk_files = extract_features_in_batches(
            datasets["train"], "train",
            batch_size=batch_size,
            samples_per_chunk=samples_per_chunk
        )
        logger.info(f"✓ 训练集完成！共保存 {len(train_chunk_files)} 个chunk文件")

    # 如果有validation集（test集），也提取validation特征（分批次 + 分块保存）
    valid_chunk_files = []
    if has_validation:
        valid_chunk_files = extract_features_in_batches(
            datasets["test"], "validation",
            batch_size=batch_size,
            samples_per_chunk=samples_per_chunk
        )
        logger.info(f"✓ Validation集完成！共保存 {len(valid_chunk_files)} 个chunk文件")

    logger.info(f"\n" + "="*50)
    logger.info(f"✓ 全部完成！")
    if has_train:
        logger.info(f"  训练集: {len(train_chunk_files)} 个chunk文件")
    if has_validation:
        logger.info(f"  验证集: {len(valid_chunk_files)} 个chunk文件")
    logger.info(f"  每个chunk最多 {samples_per_chunk} 页")
    logger.info(f"  保存目录: {output_dir}")
    if train_chunk_files:
        logger.info(f"\n  训练集chunk文件:")
        for i, f in enumerate(train_chunk_files):
            logger.info(f"    {i}: {os.path.basename(f)}")
    if valid_chunk_files:
        logger.info(f"\n  验证集chunk文件:")
        for i, f in enumerate(valid_chunk_files):
            logger.info(f"    {i}: {os.path.basename(f)}")
    logger.info(f"="*50)


if __name__ == "__main__":
    main()

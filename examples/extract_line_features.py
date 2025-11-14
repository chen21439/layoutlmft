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

    # 模型路径（优先从环境变量读取）
    model_path = os.getenv("LAYOUTLMFT_MODEL_PATH", os.path.join(project_root, "output/hrdoc_test"))
    # 特征输出到 E 盘节省系统盘空间
    output_dir = os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features")

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

    # 加载数据集
    logger.info("加载训练数据集...")
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))
    train_dataset = datasets["train"]

    # 加载模型和tokenizer
    logger.info("加载模型...")
    config = LayoutLMv2Config.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_path, config=config)
    model = model.to(device)
    model.eval()

    # Tokenize数据
    logger.info("Tokenizing数据...")
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        line_ids_list = []
        line_parent_ids_list = []
        line_relations_list = []

        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples["ner_tags"][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            image = examples["image"][org_batch_index]
            line_ids = examples["line_ids"][org_batch_index]

            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            token_line_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                    token_line_ids.append(-1)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                    bbox_inputs.append(bbox[word_idx])
                    token_line_ids.append(line_ids[word_idx])
                else:
                    label_ids.append(-100)
                    bbox_inputs.append(bbox[word_idx])
                    token_line_ids.append(line_ids[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            images.append(image)
            line_ids_list.append(token_line_ids)
            line_parent_ids_list.append(examples["line_parent_ids"][org_batch_index])
            line_relations_list.append(examples["line_relations"][org_batch_index])

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["image"] = images
        tokenized_inputs["line_ids"] = line_ids_list
        tokenized_inputs["line_parent_ids"] = line_parent_ids_list
        tokenized_inputs["line_relations"] = line_relations_list

        return tokenized_inputs

    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=1,
    )

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )

    # 提取特征
    logger.info("开始提取行级特征...")
    feature_extractor = LineFeatureExtractor()

    all_page_features = []

    with torch.no_grad():
        for idx in tqdm(range(len(train_dataset)), desc="提取特征"):
            # 获取单个样本
            sample = train_dataset[idx]

            # 保存元数据（不传给模型）
            line_parent_ids = sample.pop("line_parent_ids")
            line_relations = sample.pop("line_relations")
            line_ids = sample["line_ids"]  # 保留line_ids用于特征提取

            # 计算行级 bbox（在 collator 之前，因为需要原始 bbox 数据）
            import numpy as np
            token_bboxes = sample["bbox"]  # List[List[int, int, int, int]]
            token_line_ids = line_ids      # List[int]

            # 找到最大行 id
            valid_line_ids = [lid for lid in token_line_ids if lid >= 0]
            if len(valid_line_ids) > 0:
                max_line_id = max(valid_line_ids)
                num_lines = max_line_id + 1

                # 初始化行级 bbox
                line_bboxes = np.zeros((num_lines, 4), dtype=np.float32)
                line_bboxes[:, 0] = 1e9   # x1 初始化为极大值
                line_bboxes[:, 1] = 1e9   # y1
                line_bboxes[:, 2] = -1e9  # x2 初始化为极小值
                line_bboxes[:, 3] = -1e9  # y2

                # 对每个 token，更新其所属行的 bbox
                for bbox, lid in zip(token_bboxes, token_line_ids):
                    if lid < 0:
                        continue
                    x1, y1, x2, y2 = bbox
                    line_bboxes[lid, 0] = min(line_bboxes[lid, 0], x1)
                    line_bboxes[lid, 1] = min(line_bboxes[lid, 1], y1)
                    line_bboxes[lid, 2] = max(line_bboxes[lid, 2], x2)
                    line_bboxes[lid, 3] = max(line_bboxes[lid, 3], y2)
            else:
                # 如果没有有效的行，创建空数组
                line_bboxes = np.zeros((0, 4), dtype=np.float32)

            # 准备输入（不包含元数据）
            batch = data_collator([sample])

            # 将所有张量转移到device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif k == "image" and hasattr(v, "to"):
                    batch[k] = v.to(device)

            # 前向传播获取hidden states
            outputs = model(
                input_ids=batch["input_ids"],
                bbox=batch["bbox"],
                attention_mask=batch["attention_mask"],
                image=batch["image"],
                output_hidden_states=True,
            )

            # 获取最后一层hidden states
            hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]

            # 注意：LayoutLMv2的输出序列长度可能比输入长（因为视觉embedding）
            # 我们只取前面对应文本token的部分
            text_seq_len = batch["input_ids"].shape[1]
            hidden_states = hidden_states[:, :text_seq_len, :]  # [1, 512, hidden_size]

            # 提取行级特征
            line_ids_tensor = torch.tensor(line_ids, device=device).unsqueeze(0)
            line_features, line_mask = feature_extractor.extract_line_features(
                hidden_states, line_ids_tensor, pooling="mean"
            )

            # 保存特征和元数据
            page_data = {
                "line_features": line_features.cpu(),  # [1, max_lines, hidden_size]
                "line_mask": line_mask.cpu(),          # [1, max_lines]
                "line_parent_ids": line_parent_ids,    # List[int]
                "line_relations": line_relations,      # List[str]
                "line_bboxes": line_bboxes,            # numpy array [num_lines, 4]
                "page_idx": idx,
            }

            all_page_features.append(page_data)

    # 保存到磁盘
    output_file = os.path.join(output_dir, "train_line_features.pkl")
    logger.info(f"保存特征到 {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(all_page_features, f)

    logger.info(f"✓ 完成！共提取 {len(all_page_features)} 页的特征")
    logger.info(f"  特征维度: {all_page_features[0]['line_features'].shape}")
    logger.info(f"  保存路径: {output_file}")


if __name__ == "__main__":
    main()

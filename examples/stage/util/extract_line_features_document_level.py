#!/usr/bin/env python
# coding=utf-8
"""
从已训练的 LayoutXLM/LayoutLMv2 模型提取行级特征并缓存到磁盘
【文档级别版本】按文档聚合所有页面的 line_features,支持跨页父子关系
"""

import logging
import os
import sys
import torch
import pickle
import argparse
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

import layoutlmft.data.datasets.hrdoc
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.models.relation_classifier import LineFeatureExtractor
from transformers import (
    BertTokenizerFast,
    set_seed,
)
from layoutlmft.models.layoutxlm import LayoutXLMForTokenClassification, LayoutXLMConfig, LayoutXLMTokenizerFast
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# Register both layoutxlm and layoutlmv2 (LayoutXLM's config.json has model_type="layoutlmv2")
CONFIG_MAPPING.update({
    "layoutxlm": LayoutXLMConfig,
    "layoutlmv2": LayoutXLMConfig,
})

logger = logging.getLogger(__name__)


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="文档级别特征提取")
    parser.add_argument("--data_dir", type=str, default=None, help="HRDS数据集路径")
    parser.add_argument("--model_path", type=str, default=None, help="LayoutLMv2模型路径")
    parser.add_argument("--output_dir", type=str, default=None, help="特征输出目录")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer路径（用于从原始模型加载tokenizer）")
    parser.add_argument("--num_samples", type=int, default=None, help="限制处理的样本数（用于快速测试）")
    parser.add_argument("--docs_per_chunk", type=int, default=None, help="每个chunk包含的文档数")
    args = parser.parse_args()

    # 配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 参数优先级: 命令行参数 > 环境变量 > 默认值
    # 模型路径
    model_path = args.model_path or os.getenv("LAYOUTLMFT_MODEL_PATH", "/mnt/e/models/train_data/layoutlmft/hrdoc_train/checkpoint-5000")
    # 特征输出目录
    output_dir = args.output_dir or os.getenv("LAYOUTLMFT_FEATURES_DIR", "/mnt/e/models/train_data/layoutlmft/line_features_doc")
    # 数据集路径
    data_dir = args.data_dir or os.getenv("HRDOC_DATA_DIR", "/mnt/e/models/data/Section/HRDS")

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
    logger.info(f"输出目录: {output_dir}")

    # 设置数据集路径环境变量(传递给 hrdoc.py)
    os.environ["HRDOC_DATA_DIR"] = data_dir

    # 加载数据集
    logger.info("加载数据集...")
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))

    # 检查是否有 train 和 test 集
    has_train = "train" in datasets
    has_validation = "test" in datasets
    logger.info(f"数据集包含: {list(datasets.keys())}")

    # 是否限制数据量(默认-1表示不限制,提取所有数据)
    # 参数优先级: 命令行参数 > 环境变量 > 默认值(-1)
    num_samples = args.num_samples if args.num_samples is not None else int(os.getenv("LAYOUTLMFT_NUM_SAMPLES", "-1"))
    if num_samples > 0:
        logger.info(f"限制样本数: {num_samples}")
        if has_train and len(datasets["train"]) > num_samples:
            datasets["train"] = datasets["train"].select(range(num_samples))
        if has_validation and len(datasets["test"]) > num_samples:
            datasets["test"] = datasets["test"].select(range(num_samples))
    else:
        logger.info("提取所有数据(不限制样本数)")

    train_pages = len(datasets.get('train', [])) if has_train else 0
    val_pages = len(datasets.get('test', [])) if has_validation else 0
    logger.info(f"实际页面数 - train: {train_pages}, validation: {val_pages}")

    # 加载模型和tokenizer
    logger.info("加载模型...")
    config = LayoutXLMConfig.from_pretrained(model_path)

    # Tokenizer 路径：优先使用指定路径，否则使用模型路径
    tokenizer_path = args.tokenizer_path or model_path

    # 根据模型类型选择 tokenizer
    # LayoutXLM 使用 sentencepiece, LayoutLMv2 使用 vocab.txt
    is_layoutxlm = os.path.exists(os.path.join(tokenizer_path, "sentencepiece.bpe.model")) or \
                   "layoutxlm" in tokenizer_path.lower()

    if is_layoutxlm:
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(tokenizer_path)
        logger.info(f"使用 LayoutXLM tokenizer (XLMRoberta/sentencepiece) from {tokenizer_path}")
    else:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        logger.info(f"使用 LayoutLMv2 tokenizer (BERT) from {tokenizer_path}")

    model = LayoutXLMForTokenClassification.from_pretrained(model_path, config=config)
    model = model.to(device)
    model.eval()

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8,
        padding="max_length",
        max_length=512,
    )

    # 文档级别特征提取函数
    def extract_document_features(raw_dataset, split_name, batch_size=50, docs_per_chunk=100):
        """提取特征并按文档聚合(论文方法)

        Args:
            raw_dataset: 原始数据集(每个sample是一页)
            split_name: 数据集名称(train/validation)
            batch_size: 每批次处理的页面数
            docs_per_chunk: 每个chunk文件保存的文档数

        Returns:
            chunk_files: 保存的chunk文件路径列表
        """
        import numpy as np

        logger.info(f"\n{'='*60}")
        logger.info(f"开始提取 {split_name} 集的文档级别特征")
        logger.info(f"总页面数: {len(raw_dataset)}")
        logger.info(f"批次大小: {batch_size} 页")
        logger.info(f"每 {docs_per_chunk} 个文档保存一个chunk")
        logger.info(f"{'='*60}\n")

        feature_extractor = LineFeatureExtractor()

        # 第一步: 按document_name收集所有页面
        logger.info("步骤 1/3: 按文档分组所有页面...")
        document_pages = defaultdict(list)
        for idx in tqdm(range(len(raw_dataset)), desc="收集页面"):
            raw_sample = raw_dataset[idx]
            document_name = raw_sample.get("document_name", f"unknown_{idx}")
            page_number = raw_sample.get("page_number", idx)
            document_pages[document_name].append({
                "sample": raw_sample,
                "page_number": page_number,
                "sample_idx": idx
            })

        # 按页码排序
        for doc_name in document_pages:
            document_pages[doc_name].sort(key=lambda x: x["page_number"])

        num_documents = len(document_pages)
        logger.info(f"✓ 共 {num_documents} 个文档")

        # 第二步: 逐文档提取特征
        logger.info("\n步骤 2/3: 逐文档提取特征...")

        document_features = []  # 所有文档的特征
        chunk_files = []
        chunk_idx = 0

        with torch.no_grad():
            for doc_idx, (doc_name, pages) in enumerate(tqdm(document_pages.items(), desc="处理文档")):
                num_pages = len(pages)

                # 收集当前文档所有页的 line_features
                all_line_features = []
                all_line_masks = []
                all_line_parent_ids_raw = []  # 原始的文档全局parent_id (修复前的bug数据)
                all_line_relations = []
                all_line_bboxes = []
                all_line_labels = []
                all_original_line_ids = []  # 记录每行的原始文档全局line_id，用于parent_id重映射

                # 逐页提取特征
                for page_info in pages:
                    raw_sample = page_info["sample"]
                    page_num = page_info["page_number"]

                    # Tokenize
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
                    # raw_sample["line_parent_ids"] 和 ["line_relations"] 本身就是line级别的,直接使用!
                    valid_line_ids = [lid for lid in token_line_ids if lid >= 0]
                    if len(valid_line_ids) > 0:
                        max_line_id = max(valid_line_ids)
                        min_line_id = min(valid_line_ids)
                        num_lines = max_line_id - min_line_id + 1

                        line_bboxes = np.zeros((num_lines, 4), dtype=np.float32)
                        line_bboxes[:, 0] = 1e9
                        line_bboxes[:, 1] = 1e9
                        line_bboxes[:, 2] = -1e9
                        line_bboxes[:, 3] = -1e9

                        from collections import Counter
                        line_label_votes = defaultdict(list)

                        # 遍历所有token,聚合bbox和label为line级别
                        for bbox, label, lid in zip(bboxes, labels, token_line_ids):
                            if lid < 0:
                                continue
                            local_lid = lid - min_line_id

                            # 更新bbox
                            x1, y1, x2, y2 = bbox
                            line_bboxes[local_lid, 0] = min(line_bboxes[local_lid, 0], x1)
                            line_bboxes[local_lid, 1] = min(line_bboxes[local_lid, 1], y1)
                            line_bboxes[local_lid, 2] = max(line_bboxes[local_lid, 2], x2)
                            line_bboxes[local_lid, 3] = max(line_bboxes[local_lid, 3], y2)

                            # 收集label投票
                            if label != -100:
                                line_label_votes[local_lid].append(label)

                        # 按顺序构建line级别的label列表
                        line_labels = []
                        for local_lid in range(num_lines):
                            # label (多数投票)
                            if local_lid in line_label_votes and len(line_label_votes[local_lid]) > 0:
                                most_common_label = Counter(line_label_votes[local_lid]).most_common(1)[0][0]
                                line_labels.append(most_common_label)
                            else:
                                line_labels.append(-1)

                        # parent_ids 和 relations 直接从raw_sample获取 (本身就是line级别)
                        # raw_sample["line_parent_ids"][i] 对应当前页第 i 个line (line_id = min_line_id + i)
                        page_line_parent_ids = raw_sample["line_parent_ids"]
                        page_line_relations = raw_sample["line_relations"]

                        # 记录当前页每行的原始文档全局line_id (用于后续parent_id重映射)
                        # line_id范围是 [min_line_id, max_line_id]
                        page_original_line_ids = list(range(min_line_id, min_line_id + num_lines))
                    else:
                        line_bboxes = np.zeros((0, 4), dtype=np.float32)
                        line_labels = []
                        page_line_parent_ids = []
                        page_line_relations = []
                        page_original_line_ids = []

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

                    # 收集当前页的特征(只保留有效行,去掉padding)
                    page_line_features = line_features.squeeze(0).cpu()  # [max_lines, H]
                    page_line_mask = line_mask.squeeze(0).cpu()  # [max_lines]

                    # 只保留有效行 - 所有字段都要保持一致!
                    num_valid_lines = page_line_mask.sum().item()
                    all_line_features.append(page_line_features[:num_valid_lines])  # [num_valid_lines, H]
                    all_line_masks.append(page_line_mask[:num_valid_lines])  # [num_valid_lines]
                    all_line_parent_ids.extend(page_line_parent_ids[:num_valid_lines])  # 只取有效行的parent_ids
                    all_line_relations.extend(page_line_relations[:num_valid_lines])    # 只取有效行的relations
                    all_line_bboxes.append(line_bboxes[:num_valid_lines])  # 只取有效行的bboxes
                    all_line_labels.extend(line_labels[:num_valid_lines])  # 只取有效行的labels

                # 拼接所有页的特征 - 关键步骤!
                doc_line_features = torch.cat(all_line_features, dim=0)  # [total_lines, H]
                doc_line_mask = torch.cat(all_line_masks, dim=0)  # [total_lines]
                doc_line_bboxes = np.concatenate(all_line_bboxes, axis=0)  # [total_lines, 4]

                # 保存文档级别的特征
                document_data = {
                    "line_features": doc_line_features.unsqueeze(0),  # [1, total_lines, H]
                    "line_mask": doc_line_mask.unsqueeze(0),  # [1, total_lines]
                    "line_parent_ids": all_line_parent_ids,  # 全局 parent_id
                    "line_relations": all_line_relations,
                    "line_bboxes": doc_line_bboxes,
                    "line_labels": all_line_labels,
                    "document_name": doc_name,
                    "num_pages": num_pages,
                    "num_lines": len(all_line_parent_ids)
                }

                document_features.append(document_data)

                # 检查是否需要保存当前chunk
                if len(document_features) >= docs_per_chunk:
                    chunk_file = os.path.join(output_dir, f"{split_name}_line_features_chunk_{chunk_idx:04d}.pkl")
                    logger.info(f"保存chunk {chunk_idx}: {len(document_features)} 个文档 -> {chunk_file}")
                    with open(chunk_file, "wb") as f:
                        pickle.dump(document_features, f)
                    chunk_files.append(chunk_file)
                    document_features = []
                    chunk_idx += 1

        # 保存剩余的文档
        if len(document_features) > 0:
            chunk_file = os.path.join(output_dir, f"{split_name}_line_features_chunk_{chunk_idx:04d}.pkl")
            logger.info(f"保存最后的chunk {chunk_idx}: {len(document_features)} 个文档 -> {chunk_file}")
            with open(chunk_file, "wb") as f:
                pickle.dump(document_features, f)
            chunk_files.append(chunk_file)

        logger.info(f"\n✓ {split_name} 集完成!")
        logger.info(f"  总文档数: {num_documents}")
        logger.info(f"  保存的chunk文件数: {len(chunk_files)}")

        return chunk_files

    # 提取特征
    batch_size = int(os.getenv("LAYOUTLMFT_BATCH_SIZE", "50"))
    # 参数优先级: 命令行参数 > 环境变量 > 默认值(100)
    docs_per_chunk = args.docs_per_chunk if args.docs_per_chunk is not None else int(os.getenv("LAYOUTLMFT_DOCS_PER_CHUNK", "100"))

    train_chunk_files = []
    if has_train:
        train_chunk_files = extract_document_features(
            datasets["train"], "train",
            batch_size=batch_size,
            docs_per_chunk=docs_per_chunk
        )

    valid_chunk_files = []
    if has_validation:
        valid_chunk_files = extract_document_features(
            datasets["test"], "validation",
            batch_size=batch_size,
            docs_per_chunk=docs_per_chunk
        )

    logger.info(f"\n" + "="*60)
    logger.info(f"✓ 全部完成!")
    logger.info(f"  输出目录: {output_dir}")
    if train_chunk_files:
        logger.info(f"  训练集: {len(train_chunk_files)} 个chunk文件")
    if valid_chunk_files:
        logger.info(f"  验证集: {len(valid_chunk_files)} 个chunk文件")
    logger.info(f"="*60)


if __name__ == "__main__":
    main()

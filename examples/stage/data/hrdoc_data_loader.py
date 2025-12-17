#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 统一数据加载模块

提供 Stage 1 和联合训练共用的数据加载逻辑。

核心功能：
1. 按行边界切分 tokenization（确保一行不会被截断到两个 chunk）
2. 统一的 label 映射
3. 支持页面级别和文档级别特征
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


# ==================== 标签定义 ====================
# HRDoc 数据集的类别标签（与论文一致）
LABEL_LIST = [
    "title",      # 0: 标题
    "author",     # 1: 作者
    "abstract",   # 2: 摘要
    "keywords",   # 3: 关键词
    "section",    # 4: 章节标题
    "para",       # 5: 段落
    "list",       # 6: 列表
    "bib",        # 7: 参考文献
    "equation",   # 8: 公式
    "figure",     # 9: 图片
    "table",      # 10: 表格
    "caption",    # 11: 图表标题
    "header",     # 12: 页眉
    "footer",     # 13: 页脚
    "footnote",   # 14: 脚注
    "opara",      # 15: 其他段落（嵌套段落）
]

NUM_LABELS = len(LABEL_LIST)


def get_label2id() -> Dict[str, int]:
    """获取 label -> id 映射"""
    return {label: idx for idx, label in enumerate(LABEL_LIST)}


def get_id2label() -> Dict[int, str]:
    """获取 id -> label 映射"""
    return {idx: label for idx, label in enumerate(LABEL_LIST)}


# ==================== 按行边界切分的 Tokenization ====================

def tokenize_with_line_boundary(
    tokenizer,
    tokens: List[str],
    bboxes: List[List[int]],
    labels: List[Any],
    max_length: int = 512,
    label2id: Optional[Dict[str, int]] = None,
    line_ids: Optional[List[int]] = None,
    line_parent_ids: Optional[List[int]] = None,
    line_relations: Optional[List[int]] = None,
    image: Optional[Any] = None,
    document_name: Optional[str] = None,
    page_number: Optional[int] = None,
    label_all_tokens: bool = True,
) -> List[Dict[str, Any]]:
    """
    按行边界切分的 tokenization

    关键特性：确保一行不会被截断到两个 chunk 中。
    如果当前 chunk 放不下完整的一行，就把该行放到下一个 chunk。

    Args:
        tokenizer: HuggingFace tokenizer
        tokens: 行文本列表，每个元素是一整行
        bboxes: 每行的 bbox
        labels: 每行的标签
        max_length: 最大序列长度（默认 512）
        label2id: 标签到 ID 的映射
        line_ids: 每行的 line_id（可选）
        line_parent_ids: 每行的 parent_id（可选）
        line_relations: 每行的 relation（可选）
        image: 页面图像（可选）
        document_name: 文档名称（可选）
        page_number: 页码（可选）

    Returns:
        List[Dict]: chunk 列表，每个 chunk 包含完整的行
    """
    if label2id is None:
        label2id = get_label2id()

    # 预留 [CLS] 和 [SEP] 的位置
    effective_max_length = max_length - 2

    # Step 1: 对每行单独 tokenize，获取每行的 token 数量
    line_token_counts = []
    line_tokenized = []

    for line_text in tokens:
        # 对单行进行 tokenize（不加特殊 token）
        encoded = tokenizer.encode(line_text, add_special_tokens=False)
        line_token_counts.append(len(encoded))
        line_tokenized.append(encoded)

    # Step 2: 按行边界累积，生成 chunks
    chunks = []
    current_chunk_lines = []  # 当前 chunk 包含的行索引
    current_token_count = 0

    for line_idx, token_count in enumerate(line_token_counts):
        # 检查单行是否超过限制
        if token_count > effective_max_length:
            # 单行太长，需要截断（这是不得已的情况）
            logger.warning(
                f"Line {line_idx} has {token_count} tokens, exceeding max_length {effective_max_length}. "
                f"Will be truncated."
            )
            # 如果当前 chunk 有内容，先保存
            if current_chunk_lines:
                chunks.append(current_chunk_lines)
                current_chunk_lines = []
                current_token_count = 0
            # 单行作为一个 chunk（会被截断）
            chunks.append([line_idx])
            continue

        # 检查加入当前行后是否会超过限制
        if current_token_count + token_count > effective_max_length:
            # 当前 chunk 放不下这一行，保存当前 chunk，开始新 chunk
            if current_chunk_lines:
                chunks.append(current_chunk_lines)
            current_chunk_lines = [line_idx]
            current_token_count = token_count
        else:
            # 可以放入当前 chunk
            current_chunk_lines.append(line_idx)
            current_token_count += token_count

    # 保存最后一个 chunk
    if current_chunk_lines:
        chunks.append(current_chunk_lines)

    # Step 3: 为每个 chunk 构建完整的 tokenized 输出
    results = []

    for chunk_idx, chunk_line_indices in enumerate(chunks):
        # 收集当前 chunk 的行
        chunk_tokens = [tokens[i] for i in chunk_line_indices]
        chunk_bboxes = [bboxes[i] for i in chunk_line_indices]
        chunk_labels = [labels[i] for i in chunk_line_indices]

        # Tokenize（使用 is_split_into_words=True）
        tokenized = tokenizer(
            chunk_tokens,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            is_split_into_words=True,
            return_tensors=None,
        )

        # 对齐 labels, bboxes, line_ids
        word_ids = tokenized.word_ids()

        aligned_labels = []
        aligned_bboxes = []
        aligned_line_ids = []  # 使用 chunk 内的本地索引（word_idx）

        prev_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # 特殊 token ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
                aligned_bboxes.append([0, 0, 0, 0])
                aligned_line_ids.append(-1)
            elif word_idx != prev_word_idx:
                # 新行的第一个 token
                # Label
                lbl = chunk_labels[word_idx]
                label_id = lbl if isinstance(lbl, int) else label2id.get(lbl, 0)
                aligned_labels.append(label_id)

                # Bbox
                aligned_bboxes.append(chunk_bboxes[word_idx])

                # Line ID：使用 word_idx 作为 chunk 内的本地 line_id
                # 这样 Stage 2 的 line feature 聚合会正确工作
                aligned_line_ids.append(word_idx)
            else:
                # 同一行的后续 token
                # label_all_tokens=True: 所有 token 都用真实标签（推荐，用于行级分类）
                # label_all_tokens=False: 只有行首 token 有标签，其他 -100（NER 风格）
                if label_all_tokens:
                    lbl = chunk_labels[word_idx]
                    label_id = lbl if isinstance(lbl, int) else label2id.get(lbl, 0)
                    aligned_labels.append(label_id)
                else:
                    aligned_labels.append(-100)
                aligned_bboxes.append(chunk_bboxes[word_idx])
                aligned_line_ids.append(word_idx)

            prev_word_idx = word_idx

        # 构建当前 chunk 的 line-level 信息
        # 关键：需要将 line_parent_ids 重映射到 chunk 内的本地索引
        # 如果 parent 不在当前 chunk 中，设为 -1（ROOT）
        chunk_line_parent_ids = []
        chunk_line_relations = []

        # 构建原始行索引 -> chunk 内本地索引的映射
        original_to_local = {orig_idx: local_idx for local_idx, orig_idx in enumerate(chunk_line_indices)}

        if line_parent_ids is not None:
            for original_idx in chunk_line_indices:
                if original_idx < len(line_parent_ids):
                    original_parent = line_parent_ids[original_idx]
                    if original_parent == -1:
                        # ROOT 节点保持 -1
                        chunk_line_parent_ids.append(-1)
                    elif original_parent in original_to_local:
                        # parent 在当前 chunk 中，映射到本地索引
                        chunk_line_parent_ids.append(original_to_local[original_parent])
                    else:
                        # parent 不在当前 chunk 中（跨 chunk 引用），设为 -1
                        chunk_line_parent_ids.append(-1)
                else:
                    chunk_line_parent_ids.append(-1)

        if line_relations is not None:
            for original_idx in chunk_line_indices:
                if original_idx < len(line_relations):
                    chunk_line_relations.append(line_relations[original_idx])
                else:
                    chunk_line_relations.append(-100)

        # 计算 line-level bboxes
        line_bboxes = compute_line_bboxes(aligned_bboxes, aligned_line_ids)

        # 构建结果
        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": aligned_labels,
            "bbox": aligned_bboxes,
            "line_ids": aligned_line_ids,
            "line_bboxes": line_bboxes,
            "chunk_line_indices": chunk_line_indices,  # 记录原始行索引，用于调试
        }

        if image is not None:
            result["image"] = image

        if chunk_line_parent_ids:
            result["line_parent_ids"] = chunk_line_parent_ids

        if chunk_line_relations:
            result["line_relations"] = chunk_line_relations

        if document_name is not None:
            result["document_name"] = document_name

        if page_number is not None:
            result["page_number"] = page_number

        results.append(result)

    return results


def compute_line_bboxes(
    token_bboxes: List[List[int]],
    token_line_ids: List[int],
) -> List[List[float]]:
    """
    从 token-level bboxes 计算 line-level bboxes

    Args:
        token_bboxes: token 级别的 bbox 列表
        token_line_ids: token 级别的 line_id 列表

    Returns:
        line_bboxes: line 级别的 bbox 列表
    """
    from collections import defaultdict

    line_bbox_accum = defaultdict(lambda: [float('inf'), float('inf'), float('-inf'), float('-inf')])

    for bbox, line_id in zip(token_bboxes, token_line_ids):
        if line_id < 0:
            continue

        x1, y1, x2, y2 = bbox
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue

        line_bbox_accum[line_id][0] = min(line_bbox_accum[line_id][0], x1)
        line_bbox_accum[line_id][1] = min(line_bbox_accum[line_id][1], y1)
        line_bbox_accum[line_id][2] = max(line_bbox_accum[line_id][2], x2)
        line_bbox_accum[line_id][3] = max(line_bbox_accum[line_id][3], y2)

    if not line_bbox_accum:
        return []

    max_line_id = max(line_bbox_accum.keys())
    line_bboxes = []

    for line_id in range(max_line_id + 1):
        if line_id in line_bbox_accum:
            bbox = line_bbox_accum[line_id]
            # 处理未更新的默认值
            if bbox[0] == float('inf'):
                line_bboxes.append([0.0, 0.0, 0.0, 0.0])
            else:
                line_bboxes.append(list(bbox))
        else:
            line_bboxes.append([0.0, 0.0, 0.0, 0.0])

    return line_bboxes


# ==================== HRDoc 数据加载器 ====================

@dataclass
class HRDocDataLoaderConfig:
    """数据加载器配置"""
    data_dir: str = None
    max_length: int = 512
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    label_all_tokens: bool = True  # True: 行内所有 token 都用真实标签参与训练
    pad_to_max_length: bool = True


class HRDocDataLoader:
    """
    HRDoc 统一数据加载器

    支持 Stage 1 训练和联合训练，使用按行边界切分的 tokenization。
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[HRDocDataLoaderConfig] = None,
        include_line_info: bool = True,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            config: 数据加载器配置
            include_line_info: 是否包含 line_ids, line_parent_ids, line_relations
        """
        self.tokenizer = tokenizer
        self.config = config or HRDocDataLoaderConfig()
        self.include_line_info = include_line_info
        self.label2id = get_label2id()
        self.id2label = get_id2label()

        self._raw_datasets = None
        self._tokenized_datasets = None

    def load_raw_datasets(self) -> Dict:
        """加载原始数据集"""
        import layoutlmft.data.datasets.hrdoc

        if self.config.data_dir:
            os.environ["HRDOC_DATA_DIR"] = self.config.data_dir

        logger.info(f"Loading HRDoc dataset from {self.config.data_dir or 'default path'}")
        self._raw_datasets = load_dataset(
            os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__)
        )

        return self._raw_datasets

    def tokenize_and_align(self, examples: Dict) -> Dict:
        """
        批量 tokenization，使用按行边界切分

        Args:
            examples: 批量样本

        Returns:
            tokenized 结果
        """
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_bboxes = []
        all_images = []
        all_line_ids = []
        all_line_parent_ids = []
        all_line_relations = []
        all_line_bboxes = []
        all_document_names = []
        all_page_numbers = []

        batch_size = len(examples["tokens"])

        for idx in range(batch_size):
            tokens = examples["tokens"][idx]
            bboxes = examples["bboxes"][idx]
            labels = examples["ner_tags"][idx]
            image = examples.get("image", [None] * batch_size)[idx]

            # 获取 line-level 信息
            line_ids = examples.get("line_ids", [None] * batch_size)[idx]
            line_parent_ids = examples.get("line_parent_ids", [None] * batch_size)[idx]
            line_relations = examples.get("line_relations", [None] * batch_size)[idx]
            document_name = examples.get("document_name", [None] * batch_size)[idx]
            page_number = examples.get("page_number", [None] * batch_size)[idx]

            # 按行边界切分 tokenization
            chunks = tokenize_with_line_boundary(
                tokenizer=self.tokenizer,
                tokens=tokens,
                bboxes=bboxes,
                labels=labels,
                max_length=self.config.max_length,
                label2id=self.label2id,
                line_ids=line_ids if self.include_line_info else None,
                line_parent_ids=line_parent_ids if self.include_line_info else None,
                line_relations=line_relations if self.include_line_info else None,
                image=image,
                document_name=document_name,
                page_number=page_number,
                label_all_tokens=self.config.label_all_tokens,
            )

            # 收集所有 chunks
            for chunk in chunks:
                all_input_ids.append(chunk["input_ids"])
                all_attention_mask.append(chunk["attention_mask"])
                all_labels.append(chunk["labels"])
                all_bboxes.append(chunk["bbox"])
                all_line_bboxes.append(chunk["line_bboxes"])

                if image is not None:
                    all_images.append(chunk.get("image"))

                if self.include_line_info:
                    all_line_ids.append(chunk.get("line_ids", []))
                    all_line_parent_ids.append(chunk.get("line_parent_ids", []))
                    all_line_relations.append(chunk.get("line_relations", []))

                if document_name is not None:
                    all_document_names.append(chunk.get("document_name"))

                if page_number is not None:
                    all_page_numbers.append(chunk.get("page_number"))

        result = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "bbox": all_bboxes,
            "line_bboxes": all_line_bboxes,
        }

        if all_images:
            result["image"] = all_images

        if self.include_line_info:
            result["line_ids"] = all_line_ids
            result["line_parent_ids"] = all_line_parent_ids
            result["line_relations"] = all_line_relations

        if all_document_names:
            result["document_name"] = all_document_names

        if all_page_numbers:
            result["page_number"] = all_page_numbers

        return result

    def prepare_datasets(self) -> Dict:
        """
        准备 tokenized 数据集

        Returns:
            Dict of tokenized datasets
        """
        if self._raw_datasets is None:
            self.load_raw_datasets()

        remove_columns = self._raw_datasets["train"].column_names

        tokenized_datasets = {}

        # 训练集
        if "train" in self._raw_datasets:
            train_dataset = self._raw_datasets["train"]
            if self.config.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.config.max_train_samples))

            logger.info("Tokenizing train dataset...")
            tokenized_datasets["train"] = train_dataset.map(
                self.tokenize_and_align,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.config.preprocessing_num_workers,
                load_from_cache_file=not self.config.overwrite_cache,
            )
            logger.info(f"Train dataset: {len(tokenized_datasets['train'])} samples")

        # 验证集（优先使用 validation，否则使用 test）
        if "validation" in self._raw_datasets:
            val_dataset = self._raw_datasets["validation"]
            if self.config.max_val_samples is not None:
                val_dataset = val_dataset.select(range(self.config.max_val_samples))

            logger.info("Tokenizing validation dataset...")
            tokenized_datasets["validation"] = val_dataset.map(
                self.tokenize_and_align,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.config.preprocessing_num_workers,
                load_from_cache_file=not self.config.overwrite_cache,
            )
            logger.info(f"Validation dataset: {len(tokenized_datasets['validation'])} samples")
        elif "test" in self._raw_datasets:
            val_dataset = self._raw_datasets["test"]
            if self.config.max_val_samples is not None:
                val_dataset = val_dataset.select(range(self.config.max_val_samples))

            logger.info("Tokenizing validation dataset (from test split)...")
            tokenized_datasets["validation"] = val_dataset.map(
                self.tokenize_and_align,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.config.preprocessing_num_workers,
                load_from_cache_file=not self.config.overwrite_cache,
            )
            logger.info(f"Validation dataset (from test): {len(tokenized_datasets['validation'])} samples")

        # 测试集
        if "test" in self._raw_datasets:
            test_dataset = self._raw_datasets["test"]
            if self.config.max_test_samples is not None:
                test_dataset = test_dataset.select(range(self.config.max_test_samples))

            logger.info("Tokenizing test dataset...")
            tokenized_datasets["test"] = test_dataset.map(
                self.tokenize_and_align,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.config.preprocessing_num_workers,
                load_from_cache_file=not self.config.overwrite_cache,
            )
            logger.info(f"Test dataset: {len(tokenized_datasets['test'])} samples")

        self._tokenized_datasets = tokenized_datasets
        return tokenized_datasets

    def get_train_dataset(self):
        """获取训练数据集"""
        if self._tokenized_datasets is None:
            self.prepare_datasets()
        return self._tokenized_datasets.get("train")

    def get_validation_dataset(self):
        """获取验证数据集"""
        if self._tokenized_datasets is None:
            self.prepare_datasets()
        return self._tokenized_datasets.get("validation")

    def get_test_dataset(self):
        """获取测试数据集"""
        if self._tokenized_datasets is None:
            self.prepare_datasets()
        return self._tokenized_datasets.get("test")

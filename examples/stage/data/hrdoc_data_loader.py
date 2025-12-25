#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 统一数据加载模块

提供 Stage 1 和联合训练共用的数据加载逻辑。

核心功能：
1. 支持页面级别和文档级别两种模式
2. 按行边界切分 tokenization（确保一行不会被截断到两个 chunk）
3. 页面级别：每个 chunk 独立，parent_id 映射到本地索引（快速训练）
4. 文档级别：保持全局 line_id 和 parent_id（支持跨页关系，用于推理）
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


# ==================== 独立数据集加载函数 ====================

def load_hrdoc_raw_datasets(data_dir: str = None, force_rebuild: bool = True, dataset_name: str = "hrdoc"):
    """
    独立的数据集加载函数（不需要 tokenizer）

    Args:
        data_dir: 数据目录
        force_rebuild: 是否强制重建缓存
        dataset_name: 数据集名称，用于区分不同数据集的缓存（如 hrds, hrdh, tender）
    """
    import layoutlmft.data.datasets.hrdoc

    if data_dir:
        os.environ["HRDOC_DATA_DIR"] = data_dir

    actual_dir = data_dir or os.environ.get("HRDOC_DATA_DIR", "default")
    print(f"[DataLoader] Loading dataset '{dataset_name}' from: {actual_dir}", flush=True)
    logger.info(f"Loading dataset '{dataset_name}' from {actual_dir}")

    # 使用 dataset_name 作为配置名称，确保不同数据集使用不同缓存
    # 注意：HuggingFace datasets 缓存 key 基于脚本 hash + config name，不包含环境变量
    if force_rebuild:
        print(f"[DataLoader] Force rebuild enabled, ignoring cache", flush=True)
        datasets = load_dataset(
            os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__),
            name=dataset_name,  # 使用数据集名称区分缓存（hrds, hrdh, tender 等）
            download_mode="force_redownload",
        )
    else:
        datasets = load_dataset(
            os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__),
            name=dataset_name,  # 使用数据集名称区分缓存（hrds, hrdh, tender 等）
        )

    train_count = len(datasets.get("train", []))
    val_count = len(datasets.get("validation", []))
    test_count = len(datasets.get("test", []))
    print(f"[DataLoader] Dataset '{dataset_name}' loaded: train={train_count}, validation={val_count}, test={test_count}", flush=True)

    return datasets


# ==================== 标签定义 ====================
LABEL_LIST = [
    "title", "author", "abstract", "keywords", "section", "para", "list", "bib",
    "equation", "figure", "table", "caption", "header", "footer", "footnote", "opara",
]
NUM_LABELS = len(LABEL_LIST)


def get_label2id() -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(LABEL_LIST)}


def get_id2label() -> Dict[int, str]:
    return {idx: label for idx, label in enumerate(LABEL_LIST)}


# ==================== 页面级别 Tokenization（本地索引，快速训练）====================

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
    页面级别 tokenization（parent_id 映射到 chunk 内本地索引）

    用于快速训练，每个 chunk 是独立样本，跨 chunk 的 parent 设为 -1 (ROOT)
    """
    if label2id is None:
        label2id = get_label2id()

    effective_max_length = max_length - 2

    # Step 1: 统计每行 token 数量
    line_token_counts = []
    for line_text in tokens:
        encoded = tokenizer.encode(line_text, add_special_tokens=False)
        line_token_counts.append(len(encoded))

    # Step 2: 按行边界累积生成 chunks
    chunks = []
    current_chunk_lines = []
    current_token_count = 0

    for line_idx, token_count in enumerate(line_token_counts):
        if token_count > effective_max_length:
            logger.warning(f"Line {line_idx} has {token_count} tokens, exceeding max_length. Will be truncated.")
            if current_chunk_lines:
                chunks.append(current_chunk_lines)
                current_chunk_lines = []
                current_token_count = 0
            chunks.append([line_idx])
            continue

        if current_token_count + token_count > effective_max_length:
            if current_chunk_lines:
                chunks.append(current_chunk_lines)
            current_chunk_lines = [line_idx]
            current_token_count = token_count
        else:
            current_chunk_lines.append(line_idx)
            current_token_count += token_count

    if current_chunk_lines:
        chunks.append(current_chunk_lines)

    # Step 3: 为每个 chunk 构建 tokenized 输出
    results = []

    for chunk_idx, chunk_line_indices in enumerate(chunks):
        chunk_tokens = [tokens[i] for i in chunk_line_indices]
        chunk_bboxes = [bboxes[i] for i in chunk_line_indices]
        chunk_labels = [labels[i] for i in chunk_line_indices]

        tokenized = tokenizer(
            chunk_tokens,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            is_split_into_words=True,
            return_tensors=None,
        )

        word_ids = tokenized.word_ids()

        aligned_labels = []
        aligned_bboxes = []
        aligned_line_ids = []  # chunk 内本地索引

        prev_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                aligned_labels.append(-100)
                aligned_bboxes.append([0, 0, 0, 0])
                aligned_line_ids.append(-1)
            elif word_idx != prev_word_idx:
                lbl = chunk_labels[word_idx]
                label_id = lbl if isinstance(lbl, int) else label2id.get(lbl, 0)
                aligned_labels.append(label_id)
                aligned_bboxes.append(chunk_bboxes[word_idx])
                aligned_line_ids.append(word_idx)  # 本地索引
            else:
                if label_all_tokens:
                    lbl = chunk_labels[word_idx]
                    label_id = lbl if isinstance(lbl, int) else label2id.get(lbl, 0)
                    aligned_labels.append(label_id)
                else:
                    aligned_labels.append(-100)
                aligned_bboxes.append(chunk_bboxes[word_idx])
                aligned_line_ids.append(word_idx)

            prev_word_idx = word_idx

        # 关键：将 line_parent_ids 重映射到 chunk 内的本地索引
        chunk_line_parent_ids = []
        chunk_line_relations = []

        # 构建原始行索引 -> chunk 内本地索引的映射
        original_to_local = {orig_idx: local_idx for local_idx, orig_idx in enumerate(chunk_line_indices)}

        if line_parent_ids is not None:
            for original_idx in chunk_line_indices:
                if original_idx < len(line_parent_ids):
                    original_parent = line_parent_ids[original_idx]
                    if original_parent == -1:
                        chunk_line_parent_ids.append(-1)
                    elif original_parent in original_to_local:
                        chunk_line_parent_ids.append(original_to_local[original_parent])
                    else:
                        # parent 不在当前 chunk 中，设为 -1 (ROOT)
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

        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": aligned_labels,
            "bbox": aligned_bboxes,
            "line_ids": aligned_line_ids,
            "line_bboxes": line_bboxes,
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


# ==================== 文档级别 Tokenization（全局索引，用于推理）====================

def tokenize_page_with_line_boundary(
    tokenizer,
    tokens: List[str],
    bboxes: List[List[int]],
    labels: List[Any],
    line_ids: List[int],
    max_length: int = 512,
    label2id: Optional[Dict[str, int]] = None,
    image: Optional[Any] = None,
    page_number: Optional[int] = None,
    label_all_tokens: bool = True,
) -> List[Dict[str, Any]]:
    """
    文档级别 tokenization（保持全局 line_id，用于跨页关系）
    """
    if label2id is None:
        label2id = get_label2id()

    effective_max_length = max_length - 2

    line_token_counts = []
    for line_text in tokens:
        encoded = tokenizer.encode(line_text, add_special_tokens=False)
        line_token_counts.append(len(encoded))

    chunks = []
    current_chunk_lines = []
    current_token_count = 0

    for line_idx, token_count in enumerate(line_token_counts):
        if token_count > effective_max_length:
            logger.warning(f"Line {line_idx} has {token_count} tokens, exceeding max_length. Will be truncated.")
            if current_chunk_lines:
                chunks.append(current_chunk_lines)
                current_chunk_lines = []
                current_token_count = 0
            chunks.append([line_idx])
            continue

        if current_token_count + token_count > effective_max_length:
            if current_chunk_lines:
                chunks.append(current_chunk_lines)
            current_chunk_lines = [line_idx]
            current_token_count = token_count
        else:
            current_chunk_lines.append(line_idx)
            current_token_count += token_count

    if current_chunk_lines:
        chunks.append(current_chunk_lines)

    results = []

    for chunk_idx, chunk_line_indices in enumerate(chunks):
        chunk_tokens = [tokens[i] for i in chunk_line_indices]
        chunk_bboxes = [bboxes[i] for i in chunk_line_indices]
        chunk_labels = [labels[i] for i in chunk_line_indices]
        chunk_global_line_ids = [line_ids[i] for i in chunk_line_indices]

        tokenized = tokenizer(
            chunk_tokens,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            is_split_into_words=True,
            return_tensors=None,
        )

        word_ids = tokenized.word_ids()

        aligned_labels = []
        aligned_bboxes = []
        aligned_line_ids = []

        prev_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                aligned_labels.append(-100)
                aligned_bboxes.append([0, 0, 0, 0])
                aligned_line_ids.append(-1)
            elif word_idx != prev_word_idx:
                lbl = chunk_labels[word_idx]
                label_id = lbl if isinstance(lbl, int) else label2id.get(lbl, 0)
                aligned_labels.append(label_id)
                aligned_bboxes.append(chunk_bboxes[word_idx])
                aligned_line_ids.append(chunk_global_line_ids[word_idx])  # 全局索引
            else:
                if label_all_tokens:
                    lbl = chunk_labels[word_idx]
                    label_id = lbl if isinstance(lbl, int) else label2id.get(lbl, 0)
                    aligned_labels.append(label_id)
                else:
                    aligned_labels.append(-100)
                aligned_bboxes.append(chunk_bboxes[word_idx])
                aligned_line_ids.append(chunk_global_line_ids[word_idx])

            prev_word_idx = word_idx

        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": aligned_labels,
            "bbox": aligned_bboxes,
            "line_ids": aligned_line_ids,
            "global_line_ids_in_chunk": chunk_global_line_ids,
        }

        if image is not None:
            result["image"] = image

        if page_number is not None:
            result["page_number"] = page_number

        results.append(result)

    return results


def compute_line_bboxes(
    token_bboxes: List[List[int]],
    token_line_ids: List[int],
) -> List[List[float]]:
    """从 token-level bboxes 计算 line-level bboxes"""
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
    dataset_name: str = "hrdoc"  # 数据集名称，用于区分缓存（如 hrds, hrdh, tender）
    max_length: int = 512
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    label_all_tokens: bool = True
    pad_to_max_length: bool = True
    force_rebuild: bool = True
    document_level: bool = False  # False=页面级别（快速训练），True=文档级别（推理）


class HRDocDataLoader:
    """
    HRDoc 统一数据加载器

    支持两种模式：
    - document_level=False（默认）：页面级别，每个 chunk 独立，快速训练
    - document_level=True：文档级别，保持跨页关系，用于推理
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[HRDocDataLoaderConfig] = None,
        include_line_info: bool = True,
    ):
        self.tokenizer = tokenizer
        self.config = config or HRDocDataLoaderConfig()
        self.include_line_info = include_line_info
        self.label2id = get_label2id()
        self.id2label = get_id2label()

        self._raw_datasets = None
        self._tokenized_datasets = None

    def load_raw_datasets(self) -> Dict:
        """加载原始数据集"""
        self._raw_datasets = load_hrdoc_raw_datasets(
            data_dir=self.config.data_dir,
            force_rebuild=self.config.force_rebuild,
            dataset_name=self.config.dataset_name,
        )
        return self._raw_datasets

    def prepare_datasets(self) -> Dict:
        """准备 tokenized 数据集"""
        if self._raw_datasets is None:
            self.load_raw_datasets()

        if self.config.document_level:
            return self._prepare_document_level_datasets()
        else:
            return self._prepare_page_level_datasets()

    def _prepare_page_level_datasets(self) -> Dict:
        """
        页面级别数据准备（快速训练模式）

        每个 chunk 是独立样本，parent_id 映射到本地索引
        """
        print("[DataLoader] Using PAGE-LEVEL mode (fast training)", flush=True)

        tokenized_datasets = {}
        remove_columns = self._raw_datasets["train"].column_names

        def tokenize_and_align(examples: Dict) -> Dict:
            all_input_ids = []
            all_attention_mask = []
            all_labels = []
            all_bboxes = []
            all_images = []
            all_line_ids = []
            all_line_parent_ids = []
            all_line_relations = []
            all_line_bboxes = []

            batch_size = len(examples["tokens"])

            for idx in range(batch_size):
                tokens = examples["tokens"][idx]
                bboxes = examples["bboxes"][idx]
                labels = examples["ner_tags"][idx]
                image = examples.get("image", [None] * batch_size)[idx]
                line_ids = examples.get("line_ids", [None] * batch_size)[idx]
                line_parent_ids = examples.get("line_parent_ids", [None] * batch_size)[idx]
                line_relations = examples.get("line_relations", [None] * batch_size)[idx]
                document_name = examples.get("document_name", [None] * batch_size)[idx]
                page_number = examples.get("page_number", [None] * batch_size)[idx]

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

                for chunk in chunks:
                    all_input_ids.append(chunk["input_ids"])
                    all_attention_mask.append(chunk["attention_mask"])
                    all_labels.append(chunk["labels"])
                    all_bboxes.append(chunk["bbox"])
                    all_line_bboxes.append(chunk.get("line_bboxes", []))

                    if image is not None:
                        all_images.append(chunk.get("image"))

                    if self.include_line_info:
                        all_line_ids.append(chunk.get("line_ids", []))
                        all_line_parent_ids.append(chunk.get("line_parent_ids", []))
                        all_line_relations.append(chunk.get("line_relations", []))

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

            return result

        # 训练集
        if "train" in self._raw_datasets:
            train_dataset = self._raw_datasets["train"]
            if self.config.max_train_samples is not None:
                train_dataset = train_dataset.select(range(min(self.config.max_train_samples, len(train_dataset))))

            print(f"[DataLoader] Tokenizing train dataset ({len(train_dataset)} pages)...", flush=True)
            tokenized_datasets["train"] = train_dataset.map(
                tokenize_and_align,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.config.preprocessing_num_workers,
                load_from_cache_file=not self.config.overwrite_cache,
            )
            print(f"[DataLoader] Train: {len(tokenized_datasets['train'])} chunks", flush=True)

        # 验证集
        val_split = "validation" if "validation" in self._raw_datasets else "test"
        if val_split in self._raw_datasets:
            val_dataset = self._raw_datasets[val_split]
            if self.config.max_val_samples is not None:
                val_dataset = val_dataset.select(range(min(self.config.max_val_samples, len(val_dataset))))

            print(f"[DataLoader] Tokenizing validation dataset ({len(val_dataset)} pages)...", flush=True)
            tokenized_datasets["validation"] = val_dataset.map(
                tokenize_and_align,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.config.preprocessing_num_workers,
                load_from_cache_file=not self.config.overwrite_cache,
            )
            print(f"[DataLoader] Validation: {len(tokenized_datasets['validation'])} chunks", flush=True)

        # 测试集
        if "test" in self._raw_datasets:
            test_dataset = self._raw_datasets["test"]
            if self.config.max_test_samples is not None:
                test_dataset = test_dataset.select(range(min(self.config.max_test_samples, len(test_dataset))))

            print(f"[DataLoader] Tokenizing test dataset ({len(test_dataset)} pages)...", flush=True)
            tokenized_datasets["test"] = test_dataset.map(
                tokenize_and_align,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.config.preprocessing_num_workers,
                load_from_cache_file=not self.config.overwrite_cache,
            )
            print(f"[DataLoader] Test: {len(tokenized_datasets['test'])} chunks", flush=True)

        self._tokenized_datasets = tokenized_datasets
        return tokenized_datasets

    def _prepare_document_level_datasets(self) -> Dict:
        """
        文档级别数据准备（用于推理）

        每个样本是一个完整文档，保持全局 line_id
        """
        print("[DataLoader] Using DOCUMENT-LEVEL mode (for inference)", flush=True)

        tokenized_datasets = {}

        def process_split(split_name, max_samples=None):
            if split_name not in self._raw_datasets:
                return None

            dataset = self._raw_datasets[split_name]

            print(f"[DataLoader] Grouping {split_name} pages by document...", flush=True)
            doc_pages = {}

            for page_idx in range(len(dataset)):
                page = dataset[page_idx]
                doc_name = page["document_name"]
                if doc_name not in doc_pages:
                    doc_pages[doc_name] = []
                doc_pages[doc_name].append(page)

            print(f"[DataLoader] Found {len(doc_pages)} documents from {len(dataset)} pages", flush=True)

            doc_names = list(doc_pages.keys())
            if max_samples is not None:
                doc_names = doc_names[:max_samples]

            print(f"[DataLoader] Tokenizing {split_name} ({len(doc_names)} documents)...", flush=True)

            processed_docs = []
            for doc_name in doc_names:
                pages = doc_pages[doc_name]
                # 确保按整数排序（防御性处理，以防 page_number 是字符串）
                pages = sorted(pages, key=lambda p: int(p["page_number"]) if isinstance(p["page_number"], str) else p["page_number"])
                result = self._process_document_pages(doc_name, pages)
                if result is not None:
                    processed_docs.append(result)

            print(f"[DataLoader] {split_name}: {len(processed_docs)} documents", flush=True)
            return processed_docs

        tokenized_datasets["train"] = process_split("train", self.config.max_train_samples)

        if "validation" in self._raw_datasets:
            tokenized_datasets["validation"] = process_split("validation", self.config.max_val_samples)
        elif "test" in self._raw_datasets:
            tokenized_datasets["validation"] = process_split("test", self.config.max_val_samples)

        if "test" in self._raw_datasets:
            tokenized_datasets["test"] = process_split("test", self.config.max_test_samples)

        self._tokenized_datasets = tokenized_datasets
        return tokenized_datasets

    def _process_document_pages(self, document_name: str, pages: List[Dict]) -> Optional[Dict]:
        """处理一个文档的所有页面，聚合为文档级别样本"""
        all_chunks = []
        all_parent_ids = []
        all_relations = []

        for page in pages:
            page_number = page["page_number"]
            tokens = page["tokens"]
            bboxes = page["bboxes"]
            labels = page["ner_tags"]
            image = page["image"]
            line_ids = page["line_ids"]
            page_parent_ids = page["line_parent_ids"]
            page_relations = page["line_relations"]

            chunks = tokenize_page_with_line_boundary(
                tokenizer=self.tokenizer,
                tokens=tokens,
                bboxes=bboxes,
                labels=labels,
                line_ids=line_ids,
                max_length=self.config.max_length,
                label2id=self.label2id,
                image=image,
                page_number=page_number,
                label_all_tokens=self.config.label_all_tokens,
            )

            all_chunks.extend(chunks)
            all_parent_ids.extend(page_parent_ids)
            all_relations.extend(page_relations)

        if len(all_chunks) == 0:
            return None

        return {
            "document_name": document_name,
            "num_pages": len(pages),
            "chunks": all_chunks,
            "line_parent_ids": all_parent_ids,
            "line_relations": all_relations,
        }

    def get_train_dataset(self):
        if self._tokenized_datasets is None:
            self.prepare_datasets()
        return self._tokenized_datasets.get("train")

    def get_validation_dataset(self):
        if self._tokenized_datasets is None:
            self.prepare_datasets()
        return self._tokenized_datasets.get("validation")

    def get_test_dataset(self):
        if self._tokenized_datasets is None:
            self.prepare_datasets()
        return self._tokenized_datasets.get("test")

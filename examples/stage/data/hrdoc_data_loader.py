#!/usr/bin/env python
# coding=utf-8
"""
HRDoc 统一数据加载模块

提供 Stage 1 和联合训练共用的数据加载逻辑。

核心功能：
1. 文档级别数据加载（一个样本 = 整个文档）
2. 按行边界切分 tokenization（确保一行不会被截断到两个 chunk）
3. 保持全局 line_id 和 parent_id（不重映射，支持跨页关系）
4. Stage 1 逐页处理，Stage 2/3/4 处理整个文档
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


# ==================== 独立数据集加载函数 ====================

def load_hrdoc_raw_datasets(data_dir: str = None, force_rebuild: bool = True):
    """
    独立的数据集加载函数（不需要 tokenizer）

    可以在创建 HRDocDataLoader 之前调用，获取原始数据集。

    Args:
        data_dir: 数据目录路径，如果为 None 则使用环境变量 HRDOC_DATA_DIR
        force_rebuild: 是否强制重新构建数据集（默认 True，不使用缓存）

    Returns:
        datasets: HuggingFace datasets 对象
    """
    import layoutlmft.data.datasets.hrdoc

    if data_dir:
        os.environ["HRDOC_DATA_DIR"] = data_dir

    actual_dir = data_dir or os.environ.get("HRDOC_DATA_DIR", "default")
    print(f"[DataLoader] Loading HRDoc dataset from: {actual_dir}", flush=True)
    logger.info(f"Loading HRDoc dataset from {actual_dir}")

    # 根据 force_rebuild 决定是否使用缓存（使用字符串兼容旧版本 datasets）
    if force_rebuild:
        print(f"[DataLoader] Force rebuild enabled, ignoring cache", flush=True)
        datasets = load_dataset(
            os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__),
            download_mode="force_redownload",
        )
    else:
        datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.hrdoc.__file__))

    # 打印加载结果
    train_count = len(datasets.get("train", []))
    val_count = len(datasets.get("validation", []))
    test_count = len(datasets.get("test", []))
    print(f"[DataLoader] Dataset loaded: train={train_count}, validation={val_count}, test={test_count}", flush=True)

    return datasets


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
    按行边界切分的单页 tokenization（保持全局 line_id）

    关键特性：
    1. 确保一行不会被截断到两个 chunk 中
    2. 保持全局 line_id（不重映射）

    Args:
        tokenizer: HuggingFace tokenizer
        tokens: 行文本列表
        bboxes: 每行的 bbox
        labels: 每行的标签
        line_ids: 每行的全局 line_id（不重映射）
        max_length: 最大序列长度
        label2id: 标签到 ID 的映射
        image: 页面图像
        page_number: 页码

    Returns:
        List[Dict]: chunk 列表，每个 chunk 包含完整的行，使用全局 line_id
    """
    if label2id is None:
        label2id = get_label2id()

    # 预留 [CLS] 和 [SEP] 的位置
    effective_max_length = max_length - 2

    # Step 1: 对每行单独 tokenize，获取每行的 token 数量
    line_token_counts = []
    for line_text in tokens:
        encoded = tokenizer.encode(line_text, add_special_tokens=False)
        line_token_counts.append(len(encoded))

    # Step 2: 按行边界累积，生成 chunks
    chunks = []
    current_chunk_lines = []
    current_token_count = 0

    for line_idx, token_count in enumerate(line_token_counts):
        if token_count > effective_max_length:
            logger.warning(
                f"Line {line_idx} has {token_count} tokens, exceeding max_length. Will be truncated."
            )
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
        aligned_line_ids = []  # 使用全局 line_id

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
                # 使用全局 line_id（关键改动：不再使用 chunk 内本地索引）
                aligned_line_ids.append(chunk_global_line_ids[word_idx])
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
            "line_ids": aligned_line_ids,  # 全局 line_id
            "global_line_ids_in_chunk": chunk_global_line_ids,  # 该 chunk 包含的全局 line_id 列表
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
    force_rebuild: bool = True  # 默认强制重建数据集（不使用缓存）


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
        """加载原始数据集（使用统一的加载函数）"""
        self._raw_datasets = load_hrdoc_raw_datasets(
            data_dir=self.config.data_dir,
            force_rebuild=self.config.force_rebuild,
        )
        return self._raw_datasets

    def prepare_datasets(self) -> Dict:
        """
        准备 tokenized 数据集（文档级别）

        数据流：
        1. hrdoc.py 输出页级别样本（每页一个样本）
        2. 本方法按 document_name 聚合页面
        3. 输出文档级别样本（每个文档一个样本）

        Returns:
            Dict of tokenized datasets，每个样本是一个文档
        """
        if self._raw_datasets is None:
            self.load_raw_datasets()

        tokenized_datasets = {}

        def process_split(split_name, max_samples=None):
            """处理单个数据集 split"""
            if split_name not in self._raw_datasets:
                return None

            dataset = self._raw_datasets[split_name]

            # Step 1: 按 document_name 分组页面
            print(f"[DataLoader] Grouping {split_name} pages by document...", flush=True)
            doc_pages = {}  # {document_name: [page1, page2, ...]}

            for page_idx in range(len(dataset)):
                page = dataset[page_idx]
                doc_name = page["document_name"]

                if doc_name not in doc_pages:
                    doc_pages[doc_name] = []
                doc_pages[doc_name].append(page)

            print(f"[DataLoader] Found {len(doc_pages)} documents from {len(dataset)} pages", flush=True)

            # Step 2: 限制文档数量（如果指定）
            doc_names = list(doc_pages.keys())
            if max_samples is not None:
                doc_names = doc_names[:max_samples]

            # Step 3: 处理每个文档
            print(f"[DataLoader] Tokenizing {split_name} dataset ({len(doc_names)} documents)...", flush=True)

            processed_docs = []
            for doc_name in doc_names:
                pages = doc_pages[doc_name]
                # 按 page_number 排序
                pages = sorted(pages, key=lambda p: p["page_number"])

                result = self._process_document_pages(doc_name, pages)
                if result is not None:
                    processed_docs.append(result)

            print(f"[DataLoader] {split_name} tokenization done: {len(processed_docs)} documents", flush=True)
            return processed_docs

        # 训练集
        tokenized_datasets["train"] = process_split("train", self.config.max_train_samples)

        # 验证集
        if "validation" in self._raw_datasets:
            tokenized_datasets["validation"] = process_split("validation", self.config.max_val_samples)
        elif "test" in self._raw_datasets:
            tokenized_datasets["validation"] = process_split("test", self.config.max_val_samples)

        # 测试集
        if "test" in self._raw_datasets:
            tokenized_datasets["test"] = process_split("test", self.config.max_test_samples)

        self._tokenized_datasets = tokenized_datasets
        return tokenized_datasets

    def _process_document_pages(self, document_name: str, pages: List[Dict]) -> Optional[Dict]:
        """
        处理一个文档的所有页面，聚合为文档级别样本

        Args:
            document_name: 文档名称
            pages: 该文档的所有页面（已按 page_number 排序）

        Returns:
            文档级别的处理结果
        """
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

            # 提取该页的唯一 line_ids（保持顺序）
            unique_line_ids = []
            seen = set()
            for lid in line_ids:
                if lid not in seen:
                    unique_line_ids.append(lid)
                    seen.add(lid)

            # 按行边界切分 tokenization
            chunks = tokenize_page_with_line_boundary(
                tokenizer=self.tokenizer,
                tokens=tokens,
                bboxes=bboxes,
                labels=labels,
                line_ids=unique_line_ids,
                max_length=self.config.max_length,
                label2id=self.label2id,
                image=image,
                page_number=page_number,
                label_all_tokens=self.config.label_all_tokens,
            )

            all_chunks.extend(chunks)

            # 聚合 parent_ids 和 relations（全局索引）
            all_parent_ids.extend(page_parent_ids)
            all_relations.extend(page_relations)

        if len(all_chunks) == 0:
            return None

        return {
            "document_name": document_name,
            "num_pages": len(pages),
            "chunks": all_chunks,  # 所有页的 chunks（按顺序）
            "line_parent_ids": all_parent_ids,  # 文档级别，全局索引
            "line_relations": all_relations,
        }

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

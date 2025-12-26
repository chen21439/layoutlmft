"""数据集定义

复用 HRDoc 数据加载逻辑，添加 Order 任务所需的数据处理。
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

# 复用现有的数据加载逻辑
from examples.stage.data.hrdoc_data_loader import (
    load_hrdoc_raw_datasets,
    tokenize_page_with_line_boundary,
    LABEL_LIST,
    NUM_LABELS,
    get_label2id,
    get_id2label,
)

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = None
    max_length: int = 512
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    label_all_tokens: bool = True
    force_rebuild: bool = True


class OrderDataset(Dataset):
    """Order 任务数据集

    每个样本是一个文档，包含：
    - 多个 chunks（tokenized）
    - line 级别的阅读顺序信息（从 line_id 推断）
    """

    def __init__(
        self,
        tokenizer,
        config: DataConfig,
        split: str = "train",
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            config: 数据配置
            split: 数据集划分 ("train", "validation", "test")
        """
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.label2id = get_label2id()

        # 加载并处理数据
        self.documents = self._load_and_process()

    def _load_and_process(self) -> List[Dict]:
        """加载并处理数据"""
        # 加载原始数据集
        raw_datasets = load_hrdoc_raw_datasets(
            data_dir=self.config.data_dir,
            force_rebuild=self.config.force_rebuild,
        )

        if self.split not in raw_datasets:
            logger.warning(f"Split '{self.split}' not found, using 'train'")
            self.split = "train"

        dataset = raw_datasets[self.split]

        # 限制样本数
        max_samples = None
        if self.split == "train":
            max_samples = self.config.max_train_samples
        elif self.split in ["validation", "test"]:
            max_samples = self.config.max_val_samples

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        logger.info(f"Processing {self.split} dataset ({len(dataset)} documents)...")

        # 处理每个文档
        processed_docs = []
        for doc_idx in range(len(dataset)):
            doc = dataset[doc_idx]
            result = self._process_document(doc)
            if result is not None:
                processed_docs.append(result)

        logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs

    def _process_document(self, doc: Dict) -> Optional[Dict]:
        """处理单个文档"""
        document_name = doc["document_name"]
        pages = doc["pages"]

        all_chunks = []
        all_line_ids = set()  # 收集所有 line_id

        for page in pages:
            page_number = page["page_number"]
            tokens = page["tokens"]
            bboxes = page["bboxes"]
            labels = page["ner_tags"]
            image = page["image"]
            line_ids = page["line_ids"]

            # 收集 line_id
            for lid in line_ids:
                if lid >= 0:
                    all_line_ids.add(lid)

            # 提取唯一 line_ids（保持顺序）
            unique_line_ids = []
            seen = set()
            for lid in line_ids:
                if lid not in seen:
                    unique_line_ids.append(lid)
                    seen.add(lid)

            # Tokenize
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

        if len(all_chunks) == 0:
            return None

        # line_id 本身就是阅读顺序（在 HRDoc 中）
        # 排序后的 line_ids 就是阅读顺序
        sorted_line_ids = sorted(all_line_ids)
        num_lines = len(sorted_line_ids)

        # 创建 line_id -> reading_order 映射
        line_id_to_order = {lid: idx for idx, lid in enumerate(sorted_line_ids)}

        return {
            "document_name": document_name,
            "num_pages": len(pages),
            "chunks": all_chunks,
            "num_lines": num_lines,
            "sorted_line_ids": sorted_line_ids,
            "line_id_to_order": line_id_to_order,
            # 原始数据中的 parent 信息（用于后续任务）
            "line_parent_ids": doc.get("line_parent_ids", []),
            "line_relations": doc.get("line_relations", []),
        }

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict:
        return self.documents[idx]


def create_datasets(
    tokenizer,
    config: DataConfig,
) -> Dict[str, OrderDataset]:
    """创建训练、验证、测试数据集

    Args:
        tokenizer: HuggingFace tokenizer
        config: 数据配置

    Returns:
        Dict of datasets
    """
    datasets = {}

    datasets["train"] = OrderDataset(tokenizer, config, split="train")

    # 尝试创建验证集
    try:
        datasets["validation"] = OrderDataset(tokenizer, config, split="validation")
    except Exception:
        try:
            datasets["validation"] = OrderDataset(tokenizer, config, split="test")
        except Exception:
            logger.warning("No validation dataset available")

    return datasets

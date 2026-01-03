#!/usr/bin/env python
# coding=utf-8
"""
Inference Data Loader - Direct JSON Loading for Inference

Directly loads JSON files from data_dir/test/*.json and images from data_dir/images/
without depending on HuggingFace Datasets builder.

Features:
- Reuses tokenize_page_with_line_boundary from hrdoc_data_loader.py
- Output format compatible with HRDocDocumentLevelCollator
- Supports document-level inference mode
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any

from layoutlmft.data.utils import load_image, normalize_bbox, group_items_by_page
from layoutlmft.data.labels import trans_class, LABEL2ID

from .hrdoc_data_loader import (
    tokenize_page_with_line_boundary,
    get_label2id,
    get_id2label,
)

logger = logging.getLogger(__name__)


def load_json_files(json_dir: str) -> List[Dict]:
    """Load all JSON files from a directory."""
    json_files = []
    if not os.path.exists(json_dir):
        logger.warning(f"Directory not found: {json_dir}")
        return json_files

    for filename in sorted(os.listdir(json_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(json_dir, filename)
            json_files.append({
                "filepath": filepath,
                "filename": filename,
                "doc_name": filename.replace('.json', ''),
            })

    return json_files


def load_single_document(
    json_info: Dict,
    img_dir: str,
) -> Optional[Dict]:
    """
    Load a single document from JSON file.

    Args:
        json_info: Dict with filepath, filename, doc_name
        img_dir: Directory containing images

    Returns:
        Dict with document data or None if loading fails
    """
    filepath = json_info["filepath"]
    doc_name = json_info["doc_name"]

    try:
        with open(filepath, 'r', encoding='utf8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None

    # 按页分组（复用 utils.group_items_by_page）
    pages_data = group_items_by_page(data)
    if not pages_data:
        logger.warning(f"Unknown data format in {filepath}")
        return None

    # 调试：打印第一个 item 的字段名
    first_page_items = list(pages_data.values())[0]
    if first_page_items:
        print(f"[DEBUG] JSON item keys: {list(first_page_items[0].keys())}")
        print(f"[DEBUG] First item: {first_page_items[0]}")

    # Process each page
    pages = []
    for page_num in sorted(pages_data.keys()):
        form_data = pages_data[page_num]

        # Find image
        img_path = os.path.join(img_dir, doc_name, f"{page_num}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, doc_name, f"{page_num}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, doc_name, f"{doc_name}_{page_num}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f"Image not found for {doc_name} page {page_num}")
            continue

        try:
            image, size = load_image(img_path)
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            continue

        # Process lines
        tokens, bboxes, ner_tags, line_ids = [], [], [], []
        line_parent_ids, line_relations = [], []

        for item in form_data:
            label = trans_class(
                item.get("class", item.get("label", "paraline")),
                all_lines=form_data,
                unit=item
            )
            if isinstance(label, str):
                label = LABEL2ID.get(label, 0)

            words = item.get("words", [{"text": item.get("text", ""), "box": item.get("box", [0, 0, 0, 0])}])
            words = [w for w in words if w.get("text", "").strip()]
            if not words:
                words = [{"text": "[EMPTY]", "box": item.get("box", [0, 0, 0, 0])}]

            # 确保 line_id 是整数
            raw_line_id = item.get("line_id", item.get("id", len(line_parent_ids)))
            try:
                item_line_id = int(raw_line_id)
            except (ValueError, TypeError):
                item_line_id = len(line_parent_ids)
            parent_id = item.get("parent_id", -1)
            if parent_id == "" or parent_id is None:
                parent_id = -1
            else:
                try:
                    parent_id = int(parent_id)
                except:
                    parent_id = -1

            relation = item.get("relation", "none") or "none"
            line_parent_ids.append(parent_id)
            line_relations.append(relation)

            for w in words:
                tokens.append(w.get("text", ""))
                ner_tags.append(label)
                bboxes.append(normalize_bbox(w.get("box", [0, 0, 0, 0]), size))
                line_ids.append(item_line_id)

        if tokens:
            pages.append({
                "document_name": doc_name,
                "page_number": page_num,
                "tokens": tokens,
                "bboxes": bboxes,
                "ner_tags": ner_tags,
                "image": image,
                "line_ids": line_ids,
                "line_parent_ids": line_parent_ids,
                "line_relations": line_relations,
            })

    if not pages:
        return None

    return {
        "document_name": doc_name,
        "pages": pages,
    }


class InferenceDataLoader:
    """
    Inference-only data loader that directly reads JSON files.

    Directory structure expected:
    data_dir/
    |-- test/           # JSON files
    |   |-- doc1.json
    |   +-- doc2.json
    +-- images/         # Image files
        |-- doc1/
        |   |-- 0.png
        |   +-- 1.png
        +-- doc2/
            +-- 0.png
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        label_all_tokens: bool = True,
    ):
        """
        Args:
            data_dir: Root data directory (contains test/ and images/ subdirs)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to load (None for all)
            label_all_tokens: Whether to label all tokens in a word
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.label_all_tokens = label_all_tokens

        self.label2id = get_label2id()
        self.id2label = get_id2label()

        # 自动检测目录结构（复用 hrdoc.py 的逻辑）
        # JSON 目录：优先 data_dir/test/，否则 data_dir/
        test_dir = os.path.join(data_dir, "test")
        if os.path.isdir(test_dir):
            self.json_dir = test_dir
        else:
            self.json_dir = data_dir

        # 图片目录：优先 data_dir/images/，否则 data_dir/../images/（HRDS 格式）
        img_dir = os.path.join(data_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(os.path.dirname(data_dir), "images")
        self.img_dir = img_dir

        self._dataset = None

    def load(self) -> List[Dict]:
        """
        Load and process all documents.

        Returns:
            List of document-level samples, each containing:
            - document_name: str
            - num_pages: int
            - chunks: List of tokenized chunks
            - line_parent_ids: List[int]
            - line_relations: List[str]
        """
        if self._dataset is not None:
            return self._dataset

        json_files = load_json_files(self.json_dir)
        logger.info(f"[InferenceDataLoader] Found {len(json_files)} JSON files in {self.json_dir}")

        if self.max_samples is not None and self.max_samples > 0:
            json_files = json_files[:self.max_samples]
            logger.info(f"[InferenceDataLoader] Limited to {len(json_files)} samples")

        processed_docs = []
        for json_info in json_files:
            doc_data = load_single_document(json_info, self.img_dir)
            if doc_data is None:
                continue

            result = self._process_document(doc_data)
            if result is not None:
                processed_docs.append(result)

        logger.info(f"[InferenceDataLoader] Loaded {len(processed_docs)} documents")
        self._dataset = processed_docs
        return processed_docs

    def _process_document(self, doc_data: Dict) -> Optional[Dict]:
        """
        Process a single document, tokenizing all pages.

        Args:
            doc_data: Dict with document_name and pages

        Returns:
            Dict compatible with HRDocDocumentLevelCollator
        """
        document_name = doc_data["document_name"]
        pages = doc_data["pages"]

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

            # Reuse tokenize_page_with_line_boundary
            chunks = tokenize_page_with_line_boundary(
                tokenizer=self.tokenizer,
                tokens=tokens,
                bboxes=bboxes,
                labels=labels,
                line_ids=line_ids,
                max_length=self.max_length,
                label2id=self.label2id,
                image=image,
                page_number=page_number,
                document_name=document_name,
                label_all_tokens=self.label_all_tokens,
            )

            all_chunks.extend(chunks)
            all_parent_ids.extend(page_parent_ids)
            all_relations.extend(page_relations)

        if len(all_chunks) == 0:
            return None

        # 调试日志
        total_lines = sum(len(p["line_parent_ids"]) for p in pages)
        print(f"[DEBUG] Document '{document_name}':")
        print(f"  pages: {len(pages)}, total_lines: {total_lines}, chunks: {len(all_chunks)}")
        print(f"  all_parent_ids length: {len(all_parent_ids)}")
        # 打印每页的 line_ids 范围
        for p in pages[:3]:  # 只打印前3页
            unique_line_ids = sorted(set(p["line_ids"]))
            print(f"  page {p['page_number']}: line_ids range [{min(unique_line_ids)}, {max(unique_line_ids)}], count={len(unique_line_ids)}")

        return {
            "document_name": document_name,
            "num_pages": len(pages),
            "chunks": all_chunks,
            "line_parent_ids": all_parent_ids,
            "line_relations": all_relations,
            "json_path": os.path.join(self.json_dir, f"{document_name}.json"),
        }

    def get_dataset(self) -> List[Dict]:
        """Get the dataset, loading if necessary."""
        if self._dataset is None:
            self.load()
        return self._dataset

"""HRDoc Data Loader

统一加载 HRDS (HRDoc-Simple) 和 HRDH (HRDoc-Hard) 数据集。
两个数据集格式相同，仅难度不同。

数据格式：
- 每个文档一个 JSON 文件
- 每行包含: text, box, class, page, is_meta, line_id, parent_id, relation

Relation types:
- "contain": 父子包含关系 (section -> paragraph)
- "connect": 阅读顺序延续 (line -> next line)
- "meta": 元信息行 (title, author, etc.)
- "equality": 兄弟关系 (table cells, equations)

使用示例:
    # HRDS
    dataset = HRDocDataset(data_dir="/path/to/HRDS/train", dataset_name="hrds")

    # HRDH
    dataset = HRDocDataset(data_dir="/path/to/HRDH/train", dataset_name="hrdh")
"""

import json
import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from layoutlmft.data.utils import load_image

from ..utils.tree_utils import (
    resolve_hierarchical_parents_and_siblings,
    build_sibling_matrix,
    RELATION_STR_TO_INT,
)

logger = logging.getLogger(__name__)


class HRDocDataset(Dataset):
    """统一的 HRDoc 数据集加载器

    支持 HRDS (HRDoc-Simple) 和 HRDH (HRDoc-Hard)，格式相同。
    每个样本是一个文档（一个 JSON 文件）。

    目录结构支持：
    1. covmatch 模式 (推荐)：
       data_dir/
       ├── train/           # JSON 文件目录
       └── covmatch/
           └── doc_covmatch_xxx/
               ├── train_doc_ids.json
               └── dev_doc_ids.json

    2. 简单模式 (直接分割)：
       data_dir/
       └── train/           # JSON 文件目录
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "hrds",
        max_lines: int = 256,
        max_samples: int = None,
        split: str = 'train',
        val_split_ratio: float = 0.1,
        covmatch: str = None,
    ):
        """
        Args:
            data_dir: 数据集根目录 (e.g., .../HRDS)
            dataset_name: 数据集名称 ("hrds" 或 "hrdh")，仅用于日志
            max_lines: 每个样本最大行数
            max_samples: 限制样本数量（用于调试）
            split: 'train' 或 'validation'
            val_split_ratio: 验证集比例（仅在无 covmatch 时使用）
            covmatch: covmatch 目录名 (e.g., "doc_covmatch_dev10_seed42")
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.max_lines = max_lines
        self.samples = []

        # 确定图像目录
        data_path = Path(data_dir)
        self.images_dir = data_path / "images"
        if not self.images_dir.exists():
            logger.warning(f"Images directory not found: {self.images_dir}")
            self.images_dir = None

        self._load_data(data_dir, max_samples, split, val_split_ratio, covmatch)

    def _load_data(
        self,
        data_dir: str,
        max_samples: int,
        split: str,
        val_split_ratio: float,
        covmatch: str,
    ):
        """Load JSON files from directory with covmatch support"""
        data_path = Path(data_dir)

        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # 确定 JSON 文件目录 (优先 train 子目录)
        train_dir = data_path / "train"
        if train_dir.exists():
            json_dir = train_dir
        else:
            json_dir = data_path

        # Get all JSON files
        json_files = sorted(json_dir.glob("*.json"))

        if len(json_files) == 0:
            raise ValueError(f"No JSON files found in {json_dir}")

        logger.info(f"[{self.dataset_name}] Found {len(json_files)} JSON files in {json_dir}")

        # 加载 covmatch split（如果指定）
        train_doc_ids = None
        dev_doc_ids = None

        if covmatch:
            covmatch_dir = data_path / "covmatch" / covmatch
            train_ids_file = covmatch_dir / "train_doc_ids.json"
            dev_ids_file = covmatch_dir / "dev_doc_ids.json"

            if train_ids_file.exists() and dev_ids_file.exists():
                with open(train_ids_file, 'r') as f:
                    train_doc_ids = set(json.load(f))
                with open(dev_ids_file, 'r') as f:
                    dev_doc_ids = set(json.load(f))
                logger.info(f"[{self.dataset_name}] Using covmatch split: {len(train_doc_ids)} train, {len(dev_doc_ids)} dev")
            else:
                logger.warning(f"[{self.dataset_name}] Covmatch files not found in {covmatch_dir}, falling back to ratio split")

        # Process each file and filter by split
        all_samples = []
        for json_file in json_files:
            doc_id = json_file.stem  # 文件名（不含扩展名）作为 doc_id

            # 根据 covmatch 或 ratio 决定是否包含此文件
            if train_doc_ids is not None and dev_doc_ids is not None:
                # covmatch 模式
                if split == 'train' and doc_id not in train_doc_ids:
                    continue
                if split == 'validation' and doc_id not in dev_doc_ids:
                    continue
            # ratio 模式在后面处理

            sample = self._process_file(json_file)
            if sample is not None:
                all_samples.append(sample)

        # 如果没有使用 covmatch，按比例分割
        if train_doc_ids is None or dev_doc_ids is None:
            n_total = len(all_samples)
            n_val = int(n_total * val_split_ratio)

            if split == 'train':
                all_samples = all_samples[n_val:]
            else:
                all_samples = all_samples[:n_val]

        self.samples = all_samples

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info(f"[{self.dataset_name}] Loaded {len(self.samples)} samples for {split}")

    def _process_file(self, json_path: Path) -> Optional[Dict]:
        """Process one JSON file into a sample

        Extracts:
        - Line bboxes and text
        - Region assignments based on parent_id hierarchy
        - Successor labels based on 'connect' relations
        - Semantic class labels
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                lines_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
            return None

        if not lines_data:
            return None

        # Build line_id to index mapping
        line_id_to_idx = {line['line_id']: i for i, line in enumerate(lines_data)}

        # Group by page
        pages = {}
        for i, line in enumerate(lines_data):
            page = line.get('page', 0)
            if page not in pages:
                pages[page] = []
            pages[page].append((i, line))

        # Process all pages
        all_lines = []
        all_region_ids = []
        all_successor_labels = []
        all_class_labels = []

        # Determine regions by finding root nodes (parent_id = -1) and their descendants
        region_assignments = self._assign_regions(lines_data, line_id_to_idx)

        # Build successor map based on 'connect' relation
        successor_map = self._build_successor_map(lines_data, line_id_to_idx)

        for page_idx in sorted(pages.keys()):
            page_lines = pages[page_idx]

            for orig_idx, line in page_lines:
                bbox = line.get('box', [0, 0, 0, 0])
                text = line.get('text', '')
                semantic_class = line.get('class', 'unknown')

                all_lines.append({
                    'bbox': bbox,
                    'text': text,
                    'class': semantic_class,
                    'page': page_idx,
                    'line_id': line.get('line_id', orig_idx),
                    'parent_id': line.get('parent_id', -1),
                    'relation': line.get('relation', ''),
                    'is_meta': line.get('is_meta', False),
                })

                # Region ID
                region_id = region_assignments.get(orig_idx, orig_idx)
                all_region_ids.append(region_id)

                # Successor label (index in all_lines, not line_id)
                succ_idx = successor_map.get(orig_idx, -1)
                all_successor_labels.append(succ_idx)

                # Class label
                all_class_labels.append(semantic_class)

        if len(all_lines) == 0:
            return None

        # Truncate to max_lines
        if len(all_lines) > self.max_lines:
            all_lines = all_lines[:self.max_lines]
            all_region_ids = all_region_ids[:self.max_lines]
            all_successor_labels = all_successor_labels[:self.max_lines]
            all_class_labels = all_class_labels[:self.max_lines]

            # Fix successor labels pointing beyond max_lines
            for i, succ in enumerate(all_successor_labels):
                if succ >= self.max_lines:
                    all_successor_labels[i] = -1

        return {
            'file_name': json_path.name,
            'doc_name': json_path.stem,  # 文档名（不含扩展名），用于构建图像路径
            'lines': all_lines,
            'region_ids': all_region_ids,
            'successor_labels': all_successor_labels,
            'class_labels': all_class_labels,
            'num_lines': len(all_lines),
        }

    def _assign_regions(
        self,
        lines_data: List[Dict],
        line_id_to_idx: Dict[int, int],
    ) -> Dict[int, int]:
        """Assign region IDs based on intra-region reading order (connect relations).

        Per paper Section 4.2.3:
        - A "text region" is defined by intra-region reading order relationships
        - Lines connected by successor chains belong to the same region
        - Uses Union-Find to group lines connected by "connect" relations

        This aligns region grouping with successor prediction:
        - successor_labels: derived from "connect" relations
        - region_ids: also derived from "connect" relations (same definition)
        """
        n = len(lines_data)
        parent = list(range(n))  # Union-Find parent array

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union lines based on "connect" relations ONLY (intra-region reading order)
        # This matches the paper's definition: region = lines connected by successor chains
        for i, line in enumerate(lines_data):
            parent_id = line.get('parent_id', -1)
            relation = line.get('relation', '')
            # Only "connect" relation represents intra-region reading order
            if parent_id >= 0 and relation == 'connect' and parent_id in line_id_to_idx:
                parent_idx = line_id_to_idx[parent_id]
                union(i, parent_idx)

        # Assign region IDs (root of each component)
        region_map = {}
        for i in range(n):
            region_map[i] = find(i)

        return region_map

    def _build_successor_map(
        self,
        lines_data: List[Dict],
        line_id_to_idx: Dict[int, int],
    ) -> Dict[int, int]:
        """Build successor map based on 'connect' relation

        If line B has parent_id = A and relation = "connect",
        then A's successor is B.

        Returns: Dict mapping line index to successor index
        """
        successor_map = {}

        for i, line in enumerate(lines_data):
            parent_id = line.get('parent_id', -1)
            relation = line.get('relation', '')

            if parent_id >= 0 and relation == 'connect':
                if parent_id in line_id_to_idx:
                    parent_idx = line_id_to_idx[parent_id]
                    # Parent's successor is current line
                    successor_map[parent_idx] = i

        return successor_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        lines = sample['lines']

        # 提取 parent_ids 和 relations
        parent_ids = []
        relations = []
        pages = set()
        for line in lines:
            parent_ids.append(line.get('parent_id', -1))
            rel_str = line.get('relation', '')
            relations.append(RELATION_STR_TO_INT.get(rel_str, 0))  # 默认 contain
            pages.add(line.get('page', 0))

        return {
            'file_name': sample['file_name'],
            'doc_name': sample['doc_name'],
            'pages': sorted(pages),  # 该文档包含的页码列表
            'bboxes': [line['bbox'] for line in lines],
            'texts': [line['text'] for line in lines],
            'class_labels': sample['class_labels'],
            'region_ids': sample['region_ids'],
            'successor_labels': sample['successor_labels'],
            'parent_ids': parent_ids,
            'relations': relations,
            'num_lines': sample['num_lines'],
            'images_dir': str(self.images_dir) if self.images_dir else None,
        }


class HRDocCollator:
    """Collator for HRDoc batches"""

    # Semantic class to index mapping
    CLASS_TO_IDX = {
        'title': 0, 'author': 1, 'affili': 2, 'mail': 3,
        'sec1': 4, 'sec2': 5, 'sec3': 6,
        'fstline': 7, 'para': 8, 'opara': 9,
        'fig': 10, 'figcap': 11, 'tab': 12, 'tabcap': 13,
        'equ': 14, 'alg': 15, 'fnote': 16, 'foot': 17,
        'unknown': 18,
    }
    NUM_CLASSES = 19

    def __init__(self, max_lines: int = 256):
        self.max_lines = max_lines

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_lines = min(max(s['num_lines'] for s in batch), self.max_lines)

        # Initialize tensors
        bboxes = torch.zeros(batch_size, max_lines, 4, dtype=torch.float)
        region_ids = torch.full((batch_size, max_lines), -1, dtype=torch.long)
        successor_labels = torch.full((batch_size, max_lines), -1, dtype=torch.long)
        class_labels = torch.full((batch_size, max_lines), -1, dtype=torch.long)
        line_mask = torch.zeros(batch_size, max_lines, dtype=torch.bool)

        # Construct 训练需要的 tensors
        parent_ids = torch.full((batch_size, max_lines), -1, dtype=torch.long)
        sibling_labels = torch.zeros(batch_size, max_lines, max_lines, dtype=torch.long)

        for i, sample in enumerate(batch):
            n_lines = min(sample['num_lines'], max_lines)

            # Bboxes
            for j in range(n_lines):
                bboxes[i, j] = torch.tensor(sample['bboxes'][j], dtype=torch.float)

            # Region IDs
            region_ids[i, :n_lines] = torch.tensor(sample['region_ids'][:n_lines])

            # Successor labels
            for j in range(n_lines):
                succ = sample['successor_labels'][j]
                if 0 <= succ < n_lines:
                    successor_labels[i, j] = succ
                else:
                    successor_labels[i, j] = -1

            # Class labels
            for j in range(n_lines):
                cls = sample['class_labels'][j]
                class_labels[i, j] = self.CLASS_TO_IDX.get(cls, self.CLASS_TO_IDX['unknown'])

            # Mask
            line_mask[i, :n_lines] = True

            # 使用 tree_utils 计算层级 parent_ids 和 sibling_labels
            sample_parent_ids = sample.get('parent_ids', [])[:n_lines]
            sample_relations = sample.get('relations', [])[:n_lines]

            if sample_parent_ids and sample_relations:
                # 修正超出范围的 parent_id
                fixed_parent_ids = [p if p < n_lines else -1 for p in sample_parent_ids]

                hierarchical_parents, sibling_groups = resolve_hierarchical_parents_and_siblings(
                    fixed_parent_ids, sample_relations
                )

                # 填充 parent_ids
                for j, hp in enumerate(hierarchical_parents):
                    if j < max_lines:
                        parent_ids[i, j] = hp

                # 填充 sibling_labels
                for group in sibling_groups:
                    for j_idx in range(len(group)):
                        for k_idx in range(j_idx + 1, len(group)):
                            j, k = group[j_idx], group[k_idx]
                            if j < max_lines and k < max_lines:
                                sibling_labels[i, j, k] = 1
                                sibling_labels[i, k, j] = 1

        return {
            'bboxes': bboxes,
            'region_ids': region_ids,
            'successor_labels': successor_labels,
            'class_labels': class_labels,
            'line_mask': line_mask,
            'texts': [s['texts'] for s in batch],
            'parent_ids': parent_ids,
            'sibling_labels': sibling_labels,
        }


class HRDocLayoutXLMCollator:
    """Collator that tokenizes text for LayoutXLM

    复用 stage 的 tokenize_with_line_boundary 函数，保证 line_ids 正确对齐。
    """

    CLASS_TO_IDX = HRDocCollator.CLASS_TO_IDX
    NUM_CLASSES = HRDocCollator.NUM_CLASSES

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        max_lines: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_lines = max_lines

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # 复用 stage 的文档级别 tokenization 函数（全局索引）
        from examples.stage.data.hrdoc_data_loader import tokenize_page_with_line_boundary

        batch_size = len(batch)

        all_input_ids = []
        all_bbox = []
        all_attention_mask = []
        all_line_ids = []
        all_line_parent_ids = []  # 全局索引
        all_line_relations = []

        for sample in batch:
            texts = sample['texts']
            bboxes = sample['bboxes']
            n_lines = min(sample['num_lines'], self.max_lines)

            # Normalize bboxes to [0, 1000]
            normalized_bboxes = []
            for bbox in bboxes[:n_lines]:
                normalized_bboxes.append([
                    max(0, min(1000, int(bbox[0]))),
                    max(0, min(1000, int(bbox[1]))),
                    max(0, min(1000, int(bbox[2]))),
                    max(0, min(1000, int(bbox[3]))),
                ])

            # class_labels 转换为整数
            class_labels_int = []
            for cls in sample['class_labels'][:n_lines]:
                class_labels_int.append(self.CLASS_TO_IDX.get(cls, self.CLASS_TO_IDX['unknown']))

            # 调用 stage 的文档级别 tokenization 函数
            chunks = tokenize_page_with_line_boundary(
                tokenizer=self.tokenizer,
                tokens=texts[:n_lines],
                bboxes=normalized_bboxes,
                labels=class_labels_int,
                line_ids=list(range(n_lines)),  # 全局 line_id
                max_length=self.max_length,
                label2id=None,
                line_parent_ids=sample.get('parent_ids', [])[:n_lines],
                line_relations=sample.get('relations', [])[:n_lines],
                label_all_tokens=True,
            )

            # 使用第一个 chunk
            if chunks:
                chunk = chunks[0]
                all_input_ids.append(chunk["input_ids"])
                all_bbox.append(chunk["bbox"])
                all_attention_mask.append(chunk["attention_mask"])
                all_line_ids.append(chunk.get("line_ids", []))
                all_line_parent_ids.append(chunk.get("line_parent_ids", []))
                all_line_relations.append(chunk.get("line_relations", []))

        # 确定 batch 内最大行数（基于实际 tokenize 的行数）
        max_lines_in_batch = 0
        for line_ids in all_line_ids:
            unique_lines = set(lid for lid in line_ids if lid >= 0)
            if unique_lines:
                max_lines_in_batch = max(max_lines_in_batch, max(unique_lines) + 1)

        max_lines_in_batch = min(max_lines_in_batch, self.max_lines)
        if max_lines_in_batch == 0:
            max_lines_in_batch = 1  # 至少有一行

        # 创建 line-level tensors
        region_ids = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        successor_labels = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        class_labels = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        line_mask = torch.zeros(batch_size, max_lines_in_batch, dtype=torch.bool)
        line_bboxes = torch.zeros(batch_size, max_lines_in_batch, 4, dtype=torch.float)
        parent_ids = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        sibling_labels = torch.zeros(batch_size, max_lines_in_batch, max_lines_in_batch, dtype=torch.long)

        for i, sample in enumerate(batch):
            # 找出实际 tokenize 的行
            line_ids_list = all_line_ids[i]
            tokenized_lines = set(lid for lid in line_ids_list if lid >= 0)

            if not tokenized_lines:
                continue

            actual_n_lines = max(tokenized_lines) + 1

            # 填充 line-level tensors
            for j in tokenized_lines:
                if j >= max_lines_in_batch:
                    continue

                region_ids[i, j] = sample['region_ids'][j] if j < len(sample['region_ids']) else -1

                succ = sample['successor_labels'][j] if j < len(sample['successor_labels']) else -1
                if 0 <= succ < actual_n_lines and succ in tokenized_lines:
                    successor_labels[i, j] = succ
                else:
                    successor_labels[i, j] = -1

                cls = sample['class_labels'][j] if j < len(sample['class_labels']) else 'unknown'
                class_labels[i, j] = self.CLASS_TO_IDX.get(cls, self.CLASS_TO_IDX['unknown'])

                line_mask[i, j] = True

                # line_bboxes
                if j < len(sample['bboxes']):
                    bbox = sample['bboxes'][j]
                    line_bboxes[i, j] = torch.tensor([
                        max(0, min(1000, int(bbox[0]))),
                        max(0, min(1000, int(bbox[1]))),
                        max(0, min(1000, int(bbox[2]))),
                        max(0, min(1000, int(bbox[3]))),
                    ], dtype=torch.float)

            # chunk 返回的 parent_ids 已经是全局索引
            chunk_parent_ids = all_line_parent_ids[i]
            chunk_relations = all_line_relations[i]

            # 获取 chunk 中的 line_id 列表（用于确定有效行）
            line_ids_in_chunk = list(tokenized_lines)

            if chunk_parent_ids and chunk_relations:
                # 使用 tree_utils 处理 equality 关系（全局索引）
                hierarchical_parents, sibling_groups = resolve_hierarchical_parents_and_siblings(
                    chunk_parent_ids, chunk_relations
                )

                # 填充 parent_ids（全局索引，直接使用）
                for idx, hp in enumerate(hierarchical_parents):
                    line_id = line_ids_in_chunk[idx] if idx < len(line_ids_in_chunk) else idx
                    if line_id < max_lines_in_batch:
                        parent_ids[i, line_id] = hp

                # 填充 sibling_labels（全局索引）
                for group in sibling_groups:
                    for j_idx in range(len(group)):
                        for k_idx in range(j_idx + 1, len(group)):
                            j, k = group[j_idx], group[k_idx]
                            if j < max_lines_in_batch and k < max_lines_in_batch:
                                sibling_labels[i, j, k] = 1
                                sibling_labels[i, k, j] = 1

        # 加载图像
        all_images = []
        for sample in batch:
            images_dir = sample.get('images_dir')
            doc_name = sample.get('doc_name', '')
            pages = sample.get('pages', [0])
            page = pages[0] if pages else 0  # 使用第一页的图像

            image_loaded = False
            if images_dir:
                images_path = Path(images_dir)
                # 尝试多种图像路径格式
                possible_paths = [
                    # HRDS 格式: images/{doc_name}/{doc_name}_{page}.jpg
                    images_path / doc_name / f"{doc_name}_{page}.jpg",
                    images_path / doc_name / f"{doc_name}_{page}.png",
                    # HRDH 格式: images/{doc_name}/{page}.png
                    images_path / doc_name / f"{page}.png",
                    images_path / doc_name / f"{page}.jpg",
                    # 单页格式: images/{doc_name}.png
                    images_path / f"{doc_name}.png",
                    images_path / f"{doc_name}.jpg",
                ]

                for img_path in possible_paths:
                    if img_path.exists():
                        try:
                            img_array, _ = load_image(str(img_path))
                            all_images.append(img_array)
                            image_loaded = True
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load image {img_path}: {e}")

            # 如果没有找到图像，使用占位符
            if not image_loaded:
                # 占位符: 3x224x224 填充 ImageNet 均值
                placeholder = np.zeros((3, 224, 224), dtype=np.uint8)
                placeholder[0, :, :] = int(0.485 * 255)
                placeholder[1, :, :] = int(0.456 * 255)
                placeholder[2, :, :] = int(0.406 * 255)
                all_images.append(placeholder)

        # Stack images: [batch_size, 3, 224, 224]
        image = torch.tensor(np.stack(all_images), dtype=torch.float) / 255.0

        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'bbox': torch.tensor(all_bbox, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
            'image': image,
            'line_ids': torch.tensor(all_line_ids, dtype=torch.long),
            'region_ids': region_ids,
            'successor_labels': successor_labels,
            'class_labels': class_labels,
            'line_mask': line_mask,
            'line_bboxes': line_bboxes,
            'parent_ids': parent_ids,
            'sibling_labels': sibling_labels,
        }


def create_hrdoc_dataloaders(
    data_dir: str,
    dataset_name: str = "hrds",
    batch_size: int = 4,
    max_lines: int = 256,
    max_train_samples: int = None,
    max_val_samples: int = None,
    val_split_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for HRDoc (HRDS/HRDH)

    Args:
        data_dir: 数据目录，包含 JSON 文件
        dataset_name: 数据集名称 ("hrds" 或 "hrdh")
        batch_size: Batch size
        max_lines: Maximum lines per sample
        max_train_samples: Limit training samples
        max_val_samples: Limit validation samples
        val_split_ratio: Validation split ratio
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = HRDocDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_lines=max_lines,
        max_samples=max_train_samples,
        split='train',
        val_split_ratio=val_split_ratio,
    )

    val_dataset = HRDocDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_lines=max_lines,
        max_samples=max_val_samples,
        split='validation',
        val_split_ratio=val_split_ratio,
    )

    collator = HRDocCollator(max_lines=max_lines)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def create_hrdoc_layoutxlm_dataloaders(
    data_dir: str,
    tokenizer,
    dataset_name: str = "hrds",
    batch_size: int = 4,
    max_length: int = 512,
    max_lines: int = 256,
    max_train_samples: int = None,
    max_val_samples: int = None,
    val_split_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with LayoutXLM tokenization

    Args:
        data_dir: 数据目录，包含 JSON 文件
        tokenizer: LayoutXLM tokenizer
        dataset_name: 数据集名称 ("hrds" 或 "hrdh")
        batch_size: Batch size
        max_length: Max sequence length
        max_lines: Maximum lines per sample
        max_train_samples: Limit training samples
        max_val_samples: Limit validation samples
        val_split_ratio: Validation split ratio
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = HRDocDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_lines=max_lines,
        max_samples=max_train_samples,
        split='train',
        val_split_ratio=val_split_ratio,
    )

    val_dataset = HRDocDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_lines=max_lines,
        max_samples=max_val_samples,
        split='validation',
        val_split_ratio=val_split_ratio,
    )

    collator = HRDocLayoutXLMCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        max_lines=max_lines,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ==================== 向后兼容别名 ====================
# 保持旧的 HRDS* 名称可用

HRDSDataset = HRDocDataset
HRDSCollator = HRDocCollator
HRDSLayoutXLMCollator = HRDocLayoutXLMCollator
create_hrds_dataloaders = create_hrdoc_dataloaders
create_hrds_layoutxlm_dataloaders = create_hrdoc_layoutxlm_dataloaders


# Quick test
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/data/LLM_group/layoutlmft/data/HRDS/train'
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'hrds'

    train_loader, val_loader = create_hrdoc_dataloaders(
        data_dir=data_dir,
        dataset_name=dataset_name,
        batch_size=2,
        max_lines=64,
        max_train_samples=10,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    for batch in train_loader:
        print(f"\nBatch shapes:")
        print(f"  bboxes: {batch['bboxes'].shape}")
        print(f"  region_ids: {batch['region_ids'].shape}")
        print(f"  successor_labels: {batch['successor_labels'].shape}")
        print(f"  class_labels: {batch['class_labels'].shape}")
        print(f"  line_mask: {batch['line_mask'].shape}")
        print(f"  parent_ids: {batch['parent_ids'].shape}")
        print(f"  sibling_labels: {batch['sibling_labels'].shape}")
        print(f"  Valid lines: {batch['line_mask'].sum()}")

        # Check parent and sibling labels
        for i in range(batch['bboxes'].size(0)):
            valid = batch['line_mask'][i].sum().item()
            parents = batch['parent_ids'][i, :int(valid)]
            has_parent = (parents >= 0).sum().item()
            sibling_count = batch['sibling_labels'][i].sum().item() // 2  # 对称矩阵
            print(f"  Sample {i}: {valid} lines, {has_parent} have parent, {sibling_count} sibling pairs")
        break

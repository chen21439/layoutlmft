"""HRDS Data Loader

Loads HRDS dataset (HRDoc-Simple) where each document is a separate JSON file.
Fields are similar to HRDH but with individual files per document.

Data structure:
- Each JSON file is a list of line objects
- Each line has: text, box, class, page, is_meta, line_id, parent_id, relation

Relation types:
- "contain": Child is contained in parent (section -> paragraph)
- "connect": Continues from parent (line -> next line in reading order)
- "meta": Metadata lines (title, author, etc.)
- "equality": Equivalent elements (table cells, equations)
"""

import json
import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class HRDSDataset(Dataset):
    """Dataset for HRDS (HRDoc-Simple) format

    Each sample is one document (one JSON file).
    Extracts line-level features for intra-region training.
    """

    def __init__(
        self,
        data_dir: str,
        max_lines: int = 256,
        max_samples: int = None,
        split: str = 'train',
        val_split_ratio: float = 0.1,
    ):
        """
        Args:
            data_dir: Directory containing JSON files (e.g., .../HRDS/train/)
            max_lines: Maximum lines per sample
            max_samples: Limit number of samples (for debugging)
            split: 'train' or 'validation'
            val_split_ratio: Ratio for validation split (only used if no separate val dir)
        """
        self.data_dir = data_dir
        self.max_lines = max_lines
        self.samples = []

        self._load_data(data_dir, max_samples, split, val_split_ratio)

    def _load_data(
        self,
        data_dir: str,
        max_samples: int,
        split: str,
        val_split_ratio: float,
    ):
        """Load all JSON files from directory"""
        data_path = Path(data_dir)

        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Get all JSON files
        json_files = sorted(data_path.glob("*.json"))

        if len(json_files) == 0:
            raise ValueError(f"No JSON files found in {data_dir}")

        # Process each file
        all_samples = []
        for json_file in json_files:
            sample = self._process_file(json_file)
            if sample is not None:
                all_samples.append(sample)

        # Split train/val
        n_total = len(all_samples)
        n_val = int(n_total * val_split_ratio)

        if split == 'train':
            self.samples = all_samples[n_val:]
        else:
            self.samples = all_samples[:n_val]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info(f"Loaded {len(self.samples)} samples for {split} from {data_dir}")

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

        return {
            'file_name': sample['file_name'],
            'bboxes': [line['bbox'] for line in sample['lines']],
            'texts': [line['text'] for line in sample['lines']],
            'class_labels': sample['class_labels'],
            'region_ids': sample['region_ids'],
            'successor_labels': sample['successor_labels'],
            'num_lines': sample['num_lines'],
        }


class HRDSCollator:
    """Collator for HRDS batches"""

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

        return {
            'bboxes': bboxes,
            'region_ids': region_ids,
            'successor_labels': successor_labels,
            'class_labels': class_labels,
            'line_mask': line_mask,
            'texts': [s['texts'] for s in batch],
        }


class HRDSLayoutXLMCollator:
    """Collator that tokenizes text for LayoutXLM"""

    CLASS_TO_IDX = HRDSCollator.CLASS_TO_IDX
    NUM_CLASSES = HRDSCollator.NUM_CLASSES

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
        batch_size = len(batch)

        all_input_ids = []
        all_bbox = []
        all_attention_mask = []
        all_line_ids = []

        # Line-level tensors
        max_lines_in_batch = min(max(s['num_lines'] for s in batch), self.max_lines)
        region_ids = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        successor_labels = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        class_labels = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        line_mask = torch.zeros(batch_size, max_lines_in_batch, dtype=torch.bool)
        line_bboxes = torch.zeros(batch_size, max_lines_in_batch, 4, dtype=torch.float)

        for i, sample in enumerate(batch):
            texts = sample['texts']
            bboxes = sample['bboxes']
            n_lines = min(sample['num_lines'], self.max_lines)

            # Tokenize all lines
            input_ids = [self.tokenizer.cls_token_id]
            bbox = [[0, 0, 0, 0]]
            line_ids = [-1]

            tokenized_lines = set()
            reached_max_length = False

            for line_idx in range(n_lines):
                if reached_max_length:
                    break

                text = texts[line_idx] if line_idx < len(texts) else ""
                line_bbox = bboxes[line_idx] if line_idx < len(bboxes) else [0, 0, 0, 0]

                # Normalize bbox to [0, 1000]
                line_bbox = [
                    max(0, min(1000, int(line_bbox[0]))),
                    max(0, min(1000, int(line_bbox[1]))),
                    max(0, min(1000, int(line_bbox[2]))),
                    max(0, min(1000, int(line_bbox[3]))),
                ]

                # Tokenize
                if text.strip():
                    tokens = self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=50,
                    )
                else:
                    tokens = [self.tokenizer.unk_token_id]

                added_any = False
                for token_id in tokens:
                    if len(input_ids) >= self.max_length - 1:
                        reached_max_length = True
                        break
                    input_ids.append(token_id)
                    bbox.append(line_bbox)
                    line_ids.append(line_idx)
                    added_any = True

                if added_any:
                    tokenized_lines.add(line_idx)
                    line_bboxes[i, line_idx] = torch.tensor(line_bbox, dtype=torch.float)

            actual_n_lines = max(tokenized_lines) + 1 if tokenized_lines else 0

            # Add SEP
            input_ids.append(self.tokenizer.sep_token_id)
            bbox.append([0, 0, 0, 0])
            line_ids.append(-1)

            # Pad
            seq_len = len(input_ids)
            padding_len = self.max_length - seq_len

            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            bbox = bbox + [[0, 0, 0, 0]] * padding_len
            attention_mask = [1] * seq_len + [0] * padding_len
            line_ids = line_ids + [-1] * padding_len

            all_input_ids.append(input_ids)
            all_bbox.append(bbox)
            all_attention_mask.append(attention_mask)
            all_line_ids.append(line_ids)

            # Fill line-level tensors
            for j in range(actual_n_lines):
                if j in tokenized_lines:
                    region_ids[i, j] = sample['region_ids'][j] if j < len(sample['region_ids']) else -1

                    succ = sample['successor_labels'][j] if j < len(sample['successor_labels']) else -1
                    if 0 <= succ < actual_n_lines and succ in tokenized_lines:
                        successor_labels[i, j] = succ
                    else:
                        successor_labels[i, j] = -1

                    cls = sample['class_labels'][j] if j < len(sample['class_labels']) else 'unknown'
                    class_labels[i, j] = self.CLASS_TO_IDX.get(cls, self.CLASS_TO_IDX['unknown'])

                    line_mask[i, j] = True

        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'bbox': torch.tensor(all_bbox, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
            'line_ids': torch.tensor(all_line_ids, dtype=torch.long),
            'region_ids': region_ids,
            'successor_labels': successor_labels,
            'class_labels': class_labels,
            'line_mask': line_mask,
            'line_bboxes': line_bboxes,
        }


def create_hrds_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    max_lines: int = 256,
    max_train_samples: int = None,
    max_val_samples: int = None,
    val_split_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for HRDS

    Args:
        data_dir: Directory containing JSON files (e.g., .../HRDS/train/)
        batch_size: Batch size
        max_lines: Maximum lines per sample
        max_train_samples: Limit training samples
        max_val_samples: Limit validation samples
        val_split_ratio: Validation split ratio
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = HRDSDataset(
        data_dir=data_dir,
        max_lines=max_lines,
        max_samples=max_train_samples,
        split='train',
        val_split_ratio=val_split_ratio,
    )

    val_dataset = HRDSDataset(
        data_dir=data_dir,
        max_lines=max_lines,
        max_samples=max_val_samples,
        split='validation',
        val_split_ratio=val_split_ratio,
    )

    collator = HRDSCollator(max_lines=max_lines)

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


def create_hrds_layoutxlm_dataloaders(
    data_dir: str,
    tokenizer,
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
        data_dir: Directory containing JSON files
        tokenizer: LayoutXLM tokenizer
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
    train_dataset = HRDSDataset(
        data_dir=data_dir,
        max_lines=max_lines,
        max_samples=max_train_samples,
        split='train',
        val_split_ratio=val_split_ratio,
    )

    val_dataset = HRDSDataset(
        data_dir=data_dir,
        max_lines=max_lines,
        max_samples=max_val_samples,
        split='validation',
        val_split_ratio=val_split_ratio,
    )

    collator = HRDSLayoutXLMCollator(
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


# Quick test
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/data/LLM_group/layoutlmft/data/HRDS/train'

    train_loader, val_loader = create_hrds_dataloaders(
        data_dir=data_dir,
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
        print(f"  Valid lines: {batch['line_mask'].sum()}")

        # Check successor labels
        for i in range(batch['bboxes'].size(0)):
            valid = batch['line_mask'][i].sum().item()
            succs = batch['successor_labels'][i, :int(valid)]
            has_succ = (succs >= 0).sum().item()
            print(f"  Sample {i}: {valid} lines, {has_succ} have successors")
        break

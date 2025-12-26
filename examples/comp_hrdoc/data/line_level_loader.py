"""Line-level Data Loader for Intra-region Training

Extracts line-level features and successor labels from Comp_HRDoc dataset.
Used for training the Intra-region Head (Section 4.2.3).

Supports two modes:
1. Simple mode: Returns bboxes and texts for SimpleBboxEncoder
2. LayoutXLM mode: Tokenizes text and returns input_ids, bbox, line_ids for LayoutXLM
"""

import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class LineLevelDataset(Dataset):
    """Dataset for line-level intra-region training

    Each sample contains all lines from one page, with:
    - Line bboxes and text
    - Region assignments (which region each line belongs to)
    - Successor labels (next line in same region)
    """

    def __init__(
        self,
        data_path: str,
        max_lines: int = 256,
        max_samples: int = None,
        split: str = 'train',
        val_split_ratio: float = 0.1,
    ):
        """
        Args:
            data_path: Path to unified_layout_analysis_train.json
            max_lines: Maximum lines per sample
            max_samples: Limit number of samples (for debugging)
            split: 'train' or 'validation'
            val_split_ratio: Ratio for validation split
        """
        self.data_path = data_path
        self.max_lines = max_lines
        self.samples = []

        self._load_data(data_path, max_samples, split, val_split_ratio)

    def _load_data(
        self,
        data_path: str,
        max_samples: int,
        split: str,
        val_split_ratio: float,
    ):
        """Load and process data"""
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Group annotations by image
        image_anns = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_anns:
                image_anns[img_id] = []
            image_anns[img_id].append(ann)

        # Process each image
        all_samples = []
        for img_id, anns in image_anns.items():
            sample = self._process_image(img_id, anns, data['images'])
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

        logger.info(f"Loaded {len(self.samples)} samples for {split}")

    def _process_image(
        self,
        img_id: int,
        anns: List[Dict],
        images: List[Dict],
    ) -> Optional[Dict]:
        """Process one image into line-level sample"""

        # Get image info
        img_info = None
        for img in images:
            if img['id'] == img_id:
                img_info = img
                break

        if img_info is None:
            return None

        # Collect all lines from all annotations
        lines = []
        region_ids = []
        successor_labels = []

        for ann_idx, ann in enumerate(anns):
            textline_polys = ann.get('textline_polys', [])
            textline_contents = ann.get('textline_contents', [])

            n_lines = len(textline_polys)
            if n_lines == 0:
                continue

            region_start_idx = len(lines)

            for line_idx in range(n_lines):
                # Line bbox from polygon
                poly = textline_polys[line_idx]
                if len(poly) >= 4:
                    x_coords = poly[0::2]
                    y_coords = poly[1::2]
                    x1 = min(x_coords)
                    y1 = min(y_coords)
                    x2 = max(x_coords)
                    y2 = max(y_coords)
                    bbox = [x1, y1, x2, y2]
                else:
                    bbox = [0, 0, 0, 0]

                # Line text
                text = textline_contents[line_idx] if line_idx < len(textline_contents) else ""

                lines.append({
                    'bbox': bbox,
                    'text': text,
                    'region_id': ann_idx,
                    'line_idx_in_region': line_idx,
                })
                region_ids.append(ann_idx)

                # Successor: next line in same region, or -1 if last line
                if line_idx < n_lines - 1:
                    successor_labels.append(region_start_idx + line_idx + 1)
                else:
                    successor_labels.append(-1)  # Last line in region

        if len(lines) == 0:
            return None

        # Truncate to max_lines
        if len(lines) > self.max_lines:
            lines = lines[:self.max_lines]
            region_ids = region_ids[:self.max_lines]
            successor_labels = successor_labels[:self.max_lines]
            # Fix successor labels that point beyond max_lines
            for i, succ in enumerate(successor_labels):
                if succ >= self.max_lines:
                    successor_labels[i] = -1

        return {
            'image_id': img_id,
            'file_name': img_info.get('file_name', ''),
            'width': img_info.get('width', 1000),
            'height': img_info.get('height', 1000),
            'lines': lines,
            'region_ids': region_ids,
            'successor_labels': successor_labels,
            'num_lines': len(lines),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract line bboxes
        bboxes = [line['bbox'] for line in sample['lines']]

        return {
            'image_id': sample['image_id'],
            'bboxes': bboxes,
            'texts': [line['text'] for line in sample['lines']],
            'region_ids': sample['region_ids'],
            'successor_labels': sample['successor_labels'],
            'num_lines': sample['num_lines'],
        }


class LineLevelCollator:
    """Collator for line-level batches"""

    def __init__(self, max_lines: int = 256):
        self.max_lines = max_lines

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_lines = min(max(s['num_lines'] for s in batch), self.max_lines)

        # Initialize tensors
        bboxes = torch.zeros(batch_size, max_lines, 4, dtype=torch.float)
        region_ids = torch.full((batch_size, max_lines), -1, dtype=torch.long)
        successor_labels = torch.full((batch_size, max_lines), -1, dtype=torch.long)
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
                if succ >= 0 and succ < n_lines:
                    successor_labels[i, j] = succ
                else:
                    successor_labels[i, j] = -1

            # Mask
            line_mask[i, :n_lines] = True

        return {
            'bboxes': bboxes,
            'region_ids': region_ids,
            'successor_labels': successor_labels,
            'line_mask': line_mask,
            'texts': [s['texts'] for s in batch],  # Keep as list for tokenization
        }


class LineLevelLayoutXLMCollator:
    """Collator that tokenizes text for LayoutXLM

    For each page:
    1. Tokenize each line's text
    2. Assign line's bbox to all tokens in that line
    3. Track line_ids (which line each token belongs to)
    4. Pad to max_length
    """

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
        all_line_ids = []  # Which line each token belongs to

        # Line-level tensors
        max_lines_in_batch = min(max(s['num_lines'] for s in batch), self.max_lines)
        region_ids = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        successor_labels = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        line_mask = torch.zeros(batch_size, max_lines_in_batch, dtype=torch.bool)
        line_bboxes = torch.zeros(batch_size, max_lines_in_batch, 4, dtype=torch.float)

        for i, sample in enumerate(batch):
            texts = sample['texts']
            bboxes = sample['bboxes']
            n_lines = min(sample['num_lines'], self.max_lines)

            # Tokenize all lines for this sample
            input_ids = [self.tokenizer.cls_token_id]
            bbox = [[0, 0, 0, 0]]  # CLS token bbox
            line_ids = [-1]  # CLS doesn't belong to any line

            # Track which lines actually got tokenized (have at least one token)
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

                # Tokenize line text
                if text.strip():
                    tokens = self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=50,  # Max tokens per line
                    )
                else:
                    # Empty line: use a single UNK token
                    tokens = [self.tokenizer.unk_token_id]

                # Add tokens
                added_any = False
                for token_id in tokens:
                    if len(input_ids) >= self.max_length - 1:  # Reserve space for SEP
                        reached_max_length = True
                        break
                    input_ids.append(token_id)
                    bbox.append(line_bbox)
                    line_ids.append(line_idx)
                    added_any = True

                if added_any:
                    tokenized_lines.add(line_idx)
                    # Store line bbox
                    line_bboxes[i, line_idx] = torch.tensor(line_bbox, dtype=torch.float)

            # Actual number of tokenized lines
            actual_n_lines = max(tokenized_lines) + 1 if tokenized_lines else 0

            # Add SEP token
            input_ids.append(self.tokenizer.sep_token_id)
            bbox.append([0, 0, 0, 0])
            line_ids.append(-1)

            # Pad to max_length
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

            # Fill line-level tensors - only for tokenized lines
            for j in range(actual_n_lines):
                if j in tokenized_lines:
                    region_ids[i, j] = sample['region_ids'][j] if j < len(sample['region_ids']) else -1
                    succ = sample['successor_labels'][j] if j < len(sample['successor_labels']) else -1
                    # Only set successor if it points to a tokenized line
                    if succ >= 0 and succ < actual_n_lines and succ in tokenized_lines:
                        successor_labels[i, j] = succ
                    else:
                        successor_labels[i, j] = -1  # Last line or points to non-tokenized line
                    line_mask[i, j] = True

        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'bbox': torch.tensor(all_bbox, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
            'line_ids': torch.tensor(all_line_ids, dtype=torch.long),
            'region_ids': region_ids,
            'successor_labels': successor_labels,
            'line_mask': line_mask,
            'line_bboxes': line_bboxes,
        }


def create_line_level_dataloaders(
    data_path: str,
    batch_size: int = 4,
    max_lines: int = 256,
    max_train_samples: int = None,
    max_val_samples: int = None,
    val_split_ratio: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders

    Args:
        data_path: Path to unified_layout_analysis_train.json
        batch_size: Batch size
        max_lines: Maximum lines per sample
        max_train_samples: Limit training samples
        max_val_samples: Limit validation samples
        val_split_ratio: Validation split ratio
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = LineLevelDataset(
        data_path=data_path,
        max_lines=max_lines,
        max_samples=max_train_samples,
        split='train',
        val_split_ratio=val_split_ratio,
    )

    val_dataset = LineLevelDataset(
        data_path=data_path,
        max_lines=max_lines,
        max_samples=max_val_samples,
        split='validation',
        val_split_ratio=val_split_ratio,
    )

    collator = LineLevelCollator(max_lines=max_lines)

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


def create_layoutxlm_line_dataloaders(
    data_path: str,
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
        data_path: Path to unified_layout_analysis_train.json
        tokenizer: LayoutXLM tokenizer
        batch_size: Batch size
        max_length: Max sequence length for tokenization
        max_lines: Maximum lines per sample
        max_train_samples: Limit training samples
        max_val_samples: Limit validation samples
        val_split_ratio: Validation split ratio
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = LineLevelDataset(
        data_path=data_path,
        max_lines=max_lines,
        max_samples=max_train_samples,
        split='train',
        val_split_ratio=val_split_ratio,
    )

    val_dataset = LineLevelDataset(
        data_path=data_path,
        max_lines=max_lines,
        max_samples=max_val_samples,
        split='validation',
        val_split_ratio=val_split_ratio,
    )

    collator = LineLevelLayoutXLMCollator(
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

    data_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/mnt/e/models/data/Section/Comp_HRDoc/HRDH_MSRA_POD_TRAIN/unified_layout_analysis_train.json'

    train_loader, val_loader = create_line_level_dataloaders(
        data_path=data_path,
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
        print(f"  line_mask: {batch['line_mask'].shape}")
        print(f"  Valid lines: {batch['line_mask'].sum()}")

        # Check successor labels
        for i in range(batch['bboxes'].size(0)):
            valid = batch['line_mask'][i].sum().item()
            succs = batch['successor_labels'][i, :valid]
            has_succ = (succs >= 0).sum().item()
            print(f"  Sample {i}: {valid} lines, {has_succ} have successors")
        break

"""Line-level Data Collator V2 - Using is_split_into_words=True

This version uses HuggingFace's word-to-token alignment mechanism,
consistent with the stage directory's approach.

Key differences from line_level_loader.py:
1. Uses tokenizer(texts, is_split_into_words=True) instead of manual encode
2. Uses word_ids() to track which line each token belongs to
3. Splits into multiple chunks when exceeding max_length (line boundary aware)
"""

import torch
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class LineLevelCollatorV2:
    """Collator using is_split_into_words=True approach

    For each page:
    1. Count tokens per line to plan chunks
    2. Tokenize with is_split_into_words=True
    3. Use word_ids() to align bbox and track line_ids
    4. Handle line boundary aware truncation
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        max_lines: int = 256,
        label_all_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_lines = max_lines
        self.label_all_tokens = label_all_tokens

    def _count_line_tokens(self, texts: List[str]) -> List[int]:
        """Count tokens per line without special tokens"""
        counts = []
        for text in texts:
            if text.strip():
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                counts.append(len(tokens))
            else:
                counts.append(1)  # Empty line gets 1 UNK token
        return counts

    def _split_into_chunks(
        self,
        texts: List[str],
        token_counts: List[int],
    ) -> List[List[int]]:
        """Split lines into chunks that fit within max_length

        Each chunk is a list of line indices.
        Respects line boundaries (won't split a line across chunks).
        """
        effective_max = self.max_length - 2  # Reserve for [CLS] and [SEP]

        chunks = []
        current_chunk = []
        current_count = 0

        for line_idx, token_count in enumerate(token_counts):
            # If single line exceeds limit, it gets its own chunk (will be truncated)
            if token_count > effective_max:
                if current_chunk:
                    chunks.append(current_chunk)
                chunks.append([line_idx])
                current_chunk = []
                current_count = 0
                continue

            # Check if adding this line exceeds limit
            if current_count + token_count > effective_max:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [line_idx]
                current_count = token_count
            else:
                current_chunk.append(line_idx)
                current_count += token_count

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _tokenize_chunk(
        self,
        texts: List[str],
        bboxes: List[List[int]],
        chunk_line_indices: List[int],
    ) -> Dict:
        """Tokenize a chunk of lines using is_split_into_words=True"""

        # Get texts and bboxes for this chunk
        chunk_texts = [texts[i] for i in chunk_line_indices]
        chunk_bboxes = [bboxes[i] for i in chunk_line_indices]

        # Handle empty texts
        chunk_texts = [t if t.strip() else "[UNK]" for t in chunk_texts]

        # Tokenize with is_split_into_words=True
        tokenized = self.tokenizer(
            chunk_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors=None,
        )

        # Get word_ids for alignment
        word_ids = tokenized.word_ids()

        # Align bboxes and build line_ids
        aligned_bboxes = []
        line_ids = []  # Local index within chunk (0, 1, 2, ...)

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                aligned_bboxes.append([0, 0, 0, 0])
                line_ids.append(-1)
            else:
                # Get bbox from chunk, normalize to [0, 1000]
                bbox = chunk_bboxes[word_idx]
                bbox = [
                    max(0, min(1000, int(bbox[0]))),
                    max(0, min(1000, int(bbox[1]))),
                    max(0, min(1000, int(bbox[2]))),
                    max(0, min(1000, int(bbox[3]))),
                ]
                aligned_bboxes.append(bbox)
                line_ids.append(word_idx)  # Local line index in chunk

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "bbox": aligned_bboxes,
            "line_ids": line_ids,
            "chunk_line_indices": chunk_line_indices,  # Original line indices
            "num_lines_in_chunk": len(chunk_line_indices),
        }

    def _process_sample(
        self,
        sample: Dict,
    ) -> List[Dict]:
        """Process one sample into one or more chunks"""

        texts = sample['texts']
        bboxes = sample['bboxes']
        region_ids = sample['region_ids']
        successor_labels = sample['successor_labels']
        n_lines = min(sample['num_lines'], self.max_lines)

        # Truncate to max_lines
        texts = texts[:n_lines]
        bboxes = bboxes[:n_lines]
        region_ids = region_ids[:n_lines]
        successor_labels = successor_labels[:n_lines]

        # Count tokens per line
        token_counts = self._count_line_tokens(texts)

        # Split into chunks
        chunks = self._split_into_chunks(texts, token_counts)

        results = []
        for chunk_line_indices in chunks:
            # Tokenize chunk
            chunk_data = self._tokenize_chunk(texts, bboxes, chunk_line_indices)

            # Build mapping: original line idx -> local chunk idx
            orig_to_local = {orig: local for local, orig in enumerate(chunk_line_indices)}

            # Remap region_ids and successor_labels to local indices
            local_region_ids = []
            local_successor_labels = []
            local_line_bboxes = []

            for local_idx, orig_idx in enumerate(chunk_line_indices):
                # Region ID (keep original)
                local_region_ids.append(region_ids[orig_idx])

                # Successor: remap to local index or -1 if outside chunk
                orig_succ = successor_labels[orig_idx]
                if orig_succ == -1:
                    local_successor_labels.append(-1)
                elif orig_succ in orig_to_local:
                    local_successor_labels.append(orig_to_local[orig_succ])
                else:
                    # Successor is in another chunk, treat as last line
                    local_successor_labels.append(-1)

                # Line bbox
                bbox = bboxes[orig_idx]
                bbox = [
                    max(0, min(1000, int(bbox[0]))),
                    max(0, min(1000, int(bbox[1]))),
                    max(0, min(1000, int(bbox[2]))),
                    max(0, min(1000, int(bbox[3]))),
                ]
                local_line_bboxes.append(bbox)

            chunk_data["region_ids"] = local_region_ids
            chunk_data["successor_labels"] = local_successor_labels
            chunk_data["line_bboxes"] = local_line_bboxes

            results.append(chunk_data)

        return results

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples

        Each sample may produce multiple chunks.
        All chunks are flattened into the batch.
        """

        all_chunks = []
        for sample in batch:
            chunks = self._process_sample(sample)
            all_chunks.extend(chunks)

        if len(all_chunks) == 0:
            # Return empty batch
            return {
                'input_ids': torch.zeros(0, self.max_length, dtype=torch.long),
                'bbox': torch.zeros(0, self.max_length, 4, dtype=torch.long),
                'attention_mask': torch.zeros(0, self.max_length, dtype=torch.long),
                'line_ids': torch.zeros(0, self.max_length, dtype=torch.long),
                'region_ids': torch.zeros(0, 0, dtype=torch.long),
                'successor_labels': torch.zeros(0, 0, dtype=torch.long),
                'line_mask': torch.zeros(0, 0, dtype=torch.bool),
                'line_bboxes': torch.zeros(0, 0, 4, dtype=torch.float),
            }

        batch_size = len(all_chunks)
        max_lines_in_batch = max(c['num_lines_in_chunk'] for c in all_chunks)

        # Token-level tensors
        input_ids = torch.stack([
            torch.tensor(c['input_ids'], dtype=torch.long) for c in all_chunks
        ])
        bbox = torch.stack([
            torch.tensor(c['bbox'], dtype=torch.long) for c in all_chunks
        ])
        attention_mask = torch.stack([
            torch.tensor(c['attention_mask'], dtype=torch.long) for c in all_chunks
        ])
        line_ids = torch.stack([
            torch.tensor(c['line_ids'], dtype=torch.long) for c in all_chunks
        ])

        # Line-level tensors (need padding)
        region_ids = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        successor_labels = torch.full((batch_size, max_lines_in_batch), -1, dtype=torch.long)
        line_mask = torch.zeros(batch_size, max_lines_in_batch, dtype=torch.bool)
        line_bboxes = torch.zeros(batch_size, max_lines_in_batch, 4, dtype=torch.float)

        for i, chunk in enumerate(all_chunks):
            n_lines = chunk['num_lines_in_chunk']
            region_ids[i, :n_lines] = torch.tensor(chunk['region_ids'], dtype=torch.long)
            successor_labels[i, :n_lines] = torch.tensor(chunk['successor_labels'], dtype=torch.long)
            line_mask[i, :n_lines] = True
            line_bboxes[i, :n_lines] = torch.tensor(chunk['line_bboxes'], dtype=torch.float)

        return {
            'input_ids': input_ids,
            'bbox': bbox,
            'attention_mask': attention_mask,
            'line_ids': line_ids,
            'region_ids': region_ids,
            'successor_labels': successor_labels,
            'line_mask': line_mask,
            'line_bboxes': line_bboxes,
        }


def create_dataloaders_v2(
    dataset_train,
    dataset_val,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    max_lines: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders using V2 collator

    Args:
        dataset_train: Training dataset (LineLevelDataset)
        dataset_val: Validation dataset (LineLevelDataset)
        tokenizer: LayoutXLM tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        max_lines: Max lines per sample
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    collator = LineLevelCollatorV2(
        tokenizer=tokenizer,
        max_length=max_length,
        max_lines=max_lines,
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# Quick test
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/root/code/layoutlmft')

    from transformers import AutoTokenizer
    from examples.comp_hrdoc.data.line_level_loader import LineLevelDataset

    logging.basicConfig(level=logging.INFO)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/data/LLM_group/layoutlmft/layoutxlm-base',
        use_fast=True,
    )

    # Load dataset
    data_path = '/data/LLM_group/layoutlmft/Comp_HRDoc/HRDH_MSRA_POD_TRAIN/unified_layout_analysis_train.json'

    train_dataset = LineLevelDataset(
        data_path=data_path,
        max_lines=256,
        max_samples=10,
        split='train',
    )

    # Create collator
    collator = LineLevelCollatorV2(
        tokenizer=tokenizer,
        max_length=512,
        max_lines=256,
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"Train batches: {len(train_loader)}")

    for batch in train_loader:
        print(f"\nBatch shapes:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  bbox: {batch['bbox'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  line_ids: {batch['line_ids'].shape}")
        print(f"  region_ids: {batch['region_ids'].shape}")
        print(f"  successor_labels: {batch['successor_labels'].shape}")
        print(f"  line_mask: {batch['line_mask'].shape}")
        print(f"  line_bboxes: {batch['line_bboxes'].shape}")

        # Check line distribution
        for i in range(batch['input_ids'].size(0)):
            valid_lines = batch['line_mask'][i].sum().item()
            valid_tokens = batch['attention_mask'][i].sum().item()
            print(f"  Sample {i}: {valid_lines} lines, {valid_tokens} tokens")

        break

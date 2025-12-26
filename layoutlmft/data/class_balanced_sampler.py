# coding=utf-8
"""
Class-Balanced Batch Sampler for Long-Tailed Token Classification

Ensures each batch contains samples covering long-tail classes,
especially the easily confused class pairs.

Key idea: Instead of purely random sampling, we ensure that each batch
has a minimum representation of rare classes (MAIL, AFFILI, FIG, TAB, etc.)
"""

import torch
from torch.utils.data import Sampler
from typing import List, Dict, Iterator, Optional, Set
from collections import defaultdict
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that ensures coverage of long-tail classes.

    Strategy:
    1. Group samples by which rare classes they contain
    2. Each batch: pick some samples that contain rare classes + fill with random samples
    3. This ensures gradients flow through rare class boundaries every batch

    Args:
        dataset: The dataset to sample from
        label_column: Column name containing labels
        batch_size: Number of samples per batch
        rare_classes: Set of class IDs considered "rare" / long-tail
        rare_ratio: Fraction of batch that should contain rare classes (default 0.3)
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(
        self,
        dataset,
        label_column: str,
        batch_size: int,
        rare_classes: Optional[Set[int]] = None,
        rare_ratio: float = 0.3,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.label_column = label_column
        self.batch_size = batch_size
        self.rare_ratio = rare_ratio
        self.drop_last = drop_last
        self.seed = seed

        # Build index: which samples contain which rare classes
        self.rare_classes = rare_classes or set()
        self.samples_with_rare: Dict[int, List[int]] = defaultdict(list)  # class_id -> [sample_indices]
        self.samples_without_rare: List[int] = []

        self._build_index()

        # Calculate number of rare samples per batch
        self.num_rare_per_batch = max(1, int(batch_size * rare_ratio))
        self.num_random_per_batch = batch_size - self.num_rare_per_batch

        logger.info(f"ClassBalancedBatchSampler initialized:")
        logger.info(f"  - Total samples: {len(dataset)}")
        logger.info(f"  - Rare classes: {len(self.rare_classes)}")
        logger.info(f"  - Samples with rare classes: {sum(len(v) for v in self.samples_with_rare.values())}")
        logger.info(f"  - Rare samples per batch: {self.num_rare_per_batch}")
        logger.info(f"  - Random samples per batch: {self.num_random_per_batch}")

    def _build_index(self):
        """Build index of which samples contain rare classes."""
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            labels = sample[self.label_column]

            # Check if sample contains any rare class
            has_rare = False
            for label in labels:
                if label in self.rare_classes:
                    self.samples_with_rare[label].append(idx)
                    has_rare = True

            if not has_rare:
                self.samples_without_rare.append(idx)

        # Log class distribution
        for cls_id in sorted(self.rare_classes):
            count = len(self.samples_with_rare.get(cls_id, []))
            logger.info(f"  - Class {cls_id}: {count} samples")

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches with guaranteed rare class coverage."""
        rng = random.Random(self.seed)

        # Shuffle indices
        all_rare_samples = []
        for cls_samples in self.samples_with_rare.values():
            all_rare_samples.extend(cls_samples)
        all_rare_samples = list(set(all_rare_samples))  # Deduplicate
        rng.shuffle(all_rare_samples)

        other_samples = self.samples_without_rare.copy()
        rng.shuffle(other_samples)

        # 如果没有稀有样本或没有其他样本，降级为普通随机采样
        if not all_rare_samples and not other_samples:
            logger.warning("No samples found, falling back to index range")
            all_samples = list(range(len(self.dataset)))
            rng.shuffle(all_samples)
            for i in range(0, len(all_samples), self.batch_size):
                batch = all_samples[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch
            return

        # 如果没有稀有样本，所有样本都作为 other_samples
        if not all_rare_samples:
            logger.warning("No rare samples found, using all samples as random pool")
            all_rare_samples = other_samples.copy()

        # 如果没有其他样本，从稀有样本中填充
        if not other_samples:
            logger.warning("No other samples found, using rare samples as random pool")
            other_samples = all_rare_samples.copy()

        # Generate batches
        rare_ptr = 0
        other_ptr = 0

        num_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            num_batches += 1

        for _ in range(num_batches):
            batch = []

            # Add rare samples (with wraparound)
            for _ in range(self.num_rare_per_batch):
                if rare_ptr >= len(all_rare_samples):
                    rng.shuffle(all_rare_samples)
                    rare_ptr = 0
                batch.append(all_rare_samples[rare_ptr])
                rare_ptr += 1

            # Fill with other samples (with wraparound)
            for _ in range(self.num_random_per_batch):
                if other_ptr >= len(other_samples):
                    rng.shuffle(other_samples)
                    other_ptr = 0
                batch.append(other_samples[other_ptr])
                other_ptr += 1

            # Shuffle within batch
            rng.shuffle(batch)

            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class ConfusedPairSampler(Sampler[List[int]]):
    """
    Sampler that ensures batches contain samples from easily confused class pairs.

    This is more targeted than ClassBalancedBatchSampler - it specifically ensures
    that samples from confused pairs (e.g., MAIL/AFFILI, FIG/TAB) appear together
    in batches, forcing the model to learn discriminative features.

    Args:
        dataset: The dataset to sample from
        label_column: Column name containing labels
        batch_size: Number of samples per batch
        confused_pairs: List of (class_id_1, class_id_2) pairs that are easily confused
        pair_ratio: Fraction of batch from confused pairs (default 0.4)
    """

    def __init__(
        self,
        dataset,
        label_column: str,
        batch_size: int,
        confused_pairs: List[tuple],
        pair_ratio: float = 0.4,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.label_column = label_column
        self.batch_size = batch_size
        self.confused_pairs = confused_pairs
        self.pair_ratio = pair_ratio
        self.drop_last = drop_last
        self.seed = seed

        # Build index for each class in confused pairs
        self.class_to_samples: Dict[int, List[int]] = defaultdict(list)
        self.other_samples: List[int] = []

        # Get all classes in confused pairs
        self.confused_classes = set()
        for c1, c2 in confused_pairs:
            self.confused_classes.add(c1)
            self.confused_classes.add(c2)

        self._build_index()

        # Samples per pair per batch
        self.num_pair_samples = max(2, int(batch_size * pair_ratio))
        self.num_other_samples = batch_size - self.num_pair_samples

        logger.info(f"ConfusedPairSampler initialized:")
        logger.info(f"  - Confused pairs: {confused_pairs}")
        logger.info(f"  - Pair samples per batch: {self.num_pair_samples}")

    def _build_index(self):
        """Build index of samples per confused class."""
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            labels = sample[self.label_column]

            found_confused = False
            for label in labels:
                if label in self.confused_classes:
                    self.class_to_samples[label].append(idx)
                    found_confused = True

            if not found_confused:
                self.other_samples.append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches with confused pairs."""
        rng = random.Random(self.seed)

        # Prepare sample pools
        class_pools = {cls: list(samples) for cls, samples in self.class_to_samples.items()}
        for pool in class_pools.values():
            rng.shuffle(pool)
        class_ptrs = {cls: 0 for cls in class_pools}

        other_pool = self.other_samples.copy()
        rng.shuffle(other_pool)
        other_ptr = 0

        num_batches = len(self.dataset) // self.batch_size
        if not self.drop_last:
            num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size

        pair_idx = 0
        for _ in range(num_batches):
            batch = []

            # Select a confused pair to focus on this batch
            c1, c2 = self.confused_pairs[pair_idx % len(self.confused_pairs)]
            pair_idx += 1

            # Add samples from both classes in the pair
            samples_per_class = self.num_pair_samples // 2

            for cls in [c1, c2]:
                pool = class_pools.get(cls, [])
                if not pool:
                    continue

                for _ in range(samples_per_class):
                    if class_ptrs[cls] >= len(pool):
                        rng.shuffle(pool)
                        class_ptrs[cls] = 0
                    batch.append(pool[class_ptrs[cls]])
                    class_ptrs[cls] += 1

            # Fill with other samples
            while len(batch) < self.batch_size:
                if other_ptr >= len(other_pool):
                    rng.shuffle(other_pool)
                    other_ptr = 0
                batch.append(other_pool[other_ptr])
                other_ptr += 1

            rng.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def get_hrdoc_rare_classes(label_list: List[str]) -> Set[int]:
    """
    Get rare class IDs for HRDoc dataset.

    使用论文 14 类标签（小写，无 BIO 前缀）:
    - paraline: ~77% (dominant)
    - section/title/fstline: medium frequency
    - mail/affili/figure/table/caption/equation/footer/header/footnote/author: rare
    """
    # 稀有类（论文14类标签，小写）
    rare_class_names = {
        "mail", "affili", "figure", "table", "caption",
        "equation", "footer", "header", "footnote", "author"
    }

    rare_ids = set()
    for idx, label in enumerate(label_list):
        if label.lower() in rare_class_names:
            rare_ids.add(idx)

    return rare_ids


def get_hrdoc_confused_pairs(label_list: List[str]) -> List[tuple]:
    """
    Get easily confused class pairs for HRDoc.

    使用论文 14 类标签（小写，无 BIO 前缀）:
    - mail ↔ affili: Both are author-related metadata
    - figure ↔ table: Both are non-text elements
    - section ↔ fstline: Both can start a section
    """
    # Build label name to ID mapping
    name_to_id = {label.lower(): idx for idx, label in enumerate(label_list)}

    # 容易混淆的类对（论文14类标签）
    confused_pair_names = [
        ("mail", "affili"),
        ("figure", "table"),
        ("section", "fstline"),
        ("section", "paraline"),
    ]

    pairs = []
    for name1, name2 in confused_pair_names:
        id1 = name_to_id.get(name1)
        id2 = name_to_id.get(name2)
        if id1 is not None and id2 is not None:
            pairs.append((id1, id2))

    return pairs

# coding=utf-8
"""
Supervised Contrastive Loss for Learning Hard Boundaries

Implements contrastive learning objectives to increase class margin,
especially for easily confused class pairs like:
- MAIL ↔ AFFILI
- FIG ↔ TAB
- FIGCAP ↔ TABCAP

Reference:
- Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
- Hard Negative Mining for Metric Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for token classification.

    For each anchor token embedding:
    - Positive: other tokens with the same class
    - Negative: tokens with different classes (especially confused classes)

    Loss = -log(sum(exp(sim(anchor, positive)/tau)) / sum(exp(sim(anchor, all)/tau)))

    Args:
        temperature: Softmax temperature (default 0.07)
        base_temperature: Base temperature for scaling
        ignore_index: Label to ignore (default -100)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_index = ignore_index

        logger.info(f"SupervisedContrastiveLoss initialized with temp={temperature}")

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: Token embeddings, shape (batch, seq_len, hidden_dim)
            labels: Token labels, shape (batch, seq_len)
            mask: Optional attention mask

        Returns:
            Scalar loss
        """
        batch_size, seq_len, hidden_dim = embeddings.shape

        # Flatten
        embeddings_flat = embeddings.view(-1, hidden_dim)  # (B*S, D)
        labels_flat = labels.view(-1)  # (B*S,)

        # Create valid mask (not padding, not ignore_index)
        valid_mask = labels_flat != self.ignore_index
        if mask is not None:
            valid_mask = valid_mask & mask.view(-1).bool()

        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Get valid embeddings and labels
        valid_embeddings = embeddings_flat[valid_mask]  # (N, D)
        valid_labels = labels_flat[valid_mask]  # (N,)

        # L2 normalize embeddings
        valid_embeddings = F.normalize(valid_embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(valid_embeddings, valid_embeddings.T)  # (N, N)
        sim_matrix = sim_matrix / self.temperature

        # Create label mask: 1 if same class, 0 otherwise
        label_mask = (valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)).float()

        # Remove self-similarity (diagonal)
        n = valid_embeddings.shape[0]
        self_mask = torch.eye(n, device=embeddings.device)
        label_mask = label_mask - self_mask

        # For numerical stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits) * (1 - self_mask)  # Exclude self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        num_positives = label_mask.sum(dim=1)
        mean_log_prob_pos = (label_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

        # Loss is negative mean log-likelihood
        loss = -mean_log_prob_pos

        # Only compute loss for samples with at least one positive
        valid_loss_mask = num_positives > 0
        if valid_loss_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        loss = loss[valid_loss_mask].mean()
        loss = loss * (self.temperature / self.base_temperature)

        return loss


class HardNegativeContrastiveLoss(nn.Module):
    """
    Contrastive loss with hard negative mining for confused class pairs.

    Instead of using all negatives, focus on hard negatives from
    confused class pairs (e.g., MAIL-AFFILI, FIG-TAB).

    Args:
        confused_pairs: List of (class_id_1, class_id_2) confused pairs
        temperature: Softmax temperature
        margin: Margin for triplet-style loss
    """

    def __init__(
        self,
        confused_pairs: List[Tuple[int, int]],
        temperature: float = 0.1,
        margin: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.ignore_index = ignore_index

        # Build confused class mapping
        self.confused_with = {}
        for c1, c2 in confused_pairs:
            self.confused_with.setdefault(c1, set()).add(c2)
            self.confused_with.setdefault(c2, set()).add(c1)

        logger.info(f"HardNegativeContrastiveLoss initialized")
        logger.info(f"  - Confused pairs: {confused_pairs}")
        logger.info(f"  - Temperature: {temperature}, Margin: {margin}")

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute hard negative contrastive loss.

        For each anchor:
        - Positive: same class
        - Hard negative: confused class (e.g., MAIL anchor -> AFFILI negatives)

        Loss = max(0, margin + sim(anchor, hard_neg) - sim(anchor, positive))
        """
        batch_size, seq_len, hidden_dim = embeddings.shape

        # Flatten
        embeddings_flat = embeddings.view(-1, hidden_dim)
        labels_flat = labels.view(-1)

        # Valid mask
        valid_mask = labels_flat != self.ignore_index
        if mask is not None:
            valid_mask = valid_mask & mask.view(-1).bool()

        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        valid_embeddings = embeddings_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]

        # Normalize
        valid_embeddings = F.normalize(valid_embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(valid_embeddings, valid_embeddings.T)

        n = valid_embeddings.shape[0]
        device = embeddings.device

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_triplets = 0

        # For each anchor in confused classes
        for anchor_idx in range(n):
            anchor_label = valid_labels[anchor_idx].item()

            # Skip if not in confused classes
            if anchor_label not in self.confused_with:
                continue

            confused_classes = self.confused_with[anchor_label]

            # Find positive (same class, different position)
            pos_mask = (valid_labels == anchor_label)
            pos_mask[anchor_idx] = False  # Exclude self

            if not pos_mask.any():
                continue

            # Find hard negatives (confused classes)
            neg_mask = torch.zeros(n, dtype=torch.bool, device=device)
            for conf_cls in confused_classes:
                neg_mask |= (valid_labels == conf_cls)

            if not neg_mask.any():
                continue

            # Get similarities
            pos_sims = sim_matrix[anchor_idx][pos_mask]
            neg_sims = sim_matrix[anchor_idx][neg_mask]

            # Hardest positive (furthest)
            hardest_pos_sim = pos_sims.min()

            # Hardest negative (closest)
            hardest_neg_sim = neg_sims.max()

            # Triplet loss with margin
            triplet_loss = F.relu(self.margin + hardest_neg_sim - hardest_pos_sim)
            total_loss = total_loss + triplet_loss
            num_triplets += 1

        if num_triplets > 0:
            return total_loss / num_triplets
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class CombinedClassificationContrastiveLoss(nn.Module):
    """
    Combines standard classification loss with contrastive loss.

    Total Loss = classification_loss + lambda * contrastive_loss

    Args:
        classification_loss: The main classification loss (CE, Focal, etc.)
        contrastive_loss: Contrastive loss module
        contrastive_weight: Weight for contrastive loss (lambda)
    """

    def __init__(
        self,
        classification_loss: nn.Module,
        contrastive_loss: nn.Module,
        contrastive_weight: float = 0.1,
    ):
        super().__init__()
        self.classification_loss = classification_loss
        self.contrastive_loss = contrastive_loss
        self.contrastive_weight = contrastive_weight

        logger.info(f"CombinedLoss: contrastive_weight={contrastive_weight}")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            logits: Classification logits (B, S, C)
            labels: Ground truth labels (B, S)
            embeddings: Token embeddings for contrastive loss (B, S, D)
            attention_mask: Attention mask (B, S)

        Returns:
            Combined scalar loss
        """
        # Classification loss
        cls_loss = self.classification_loss(logits, labels)

        # Contrastive loss (if embeddings provided)
        if embeddings is not None and self.contrastive_weight > 0:
            contrast_loss = self.contrastive_loss(embeddings, labels, attention_mask)
            total_loss = cls_loss + self.contrastive_weight * contrast_loss
        else:
            total_loss = cls_loss

        return total_loss


def get_hrdoc_confused_pairs_ids(label_list: List[str]) -> List[Tuple[int, int]]:
    """
    Get confused class pair IDs for HRDoc dataset.

    使用论文 14 类标签（小写，无 BIO 前缀）
    Returns list of (class_id_1, class_id_2) tuples.
    """
    # Build name to ID mapping
    name_to_id = {label.lower(): idx for idx, label in enumerate(label_list)}

    # 容易混淆的类对（论文14类标签，小写）
    confused_pair_names = [
        ("mail", "affili"),      # Author metadata confusion
        ("figure", "table"),     # Non-text element confusion
        ("section", "fstline"),  # Section start confusion
        ("paraline", "fstline"), # Paragraph line confusion
    ]

    pairs = []
    for name1, name2 in confused_pair_names:
        id1 = name_to_id.get(name1)
        id2 = name_to_id.get(name2)
        if id1 is not None and id2 is not None:
            pairs.append((id1, id2))
            logger.info(f"Confused pair: {name1}({id1}) <-> {name2}({id2})")

    return pairs

"""Order Module for DOC Model

Based on "Detect-Order-Construct" paper.
Implements inter-region reading order prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .embeddings import (
    MultiModalEmbedding,
    SpatialCompatibilityFeatures,
    PositionalEmbedding2D,
    RegionTypeEmbedding,
)


class OrderTransformerEncoder(nn.Module):
    """3-layer Transformer Encoder for Order Module

    Enhances page object representations via self-attention.
    Architecture: 3 layers, 12 heads, 768 hidden dim (per paper)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 3,
        dropout: float = 0.1,
        ffn_dim: int = None,
    ):
        super().__init__()

        if ffn_dim is None:
            ffn_dim = hidden_size * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [batch, num_regions, hidden_size]
            mask: [batch, num_regions] True for valid positions

        Returns:
            [batch, num_regions, hidden_size] enhanced features
        """
        if mask is not None:
            # TransformerEncoder expects True = ignore
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        return self.encoder(features, src_key_padding_mask=src_key_padding_mask)


class ReadingOrderHead(nn.Module):
    """Inter-region Reading Order Prediction Head

    Predicts succeeding page objects using dependency parsing style.
    Combines:
    - Dot product of query/key projections
    - MLP of spatial compatibility features
    """

    def __init__(
        self,
        hidden_size: int = 768,
        spatial_dim: int = 128,
        dropout: float = 0.1,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_spatial = use_spatial

        # Query/Key projections for attention-style scoring
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)

        # Biaffine transformation for pairwise scoring
        self.biaffine_weight = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.biaffine_weight)
        self.biaffine_bias = nn.Parameter(torch.zeros(1))

        # Spatial compatibility MLP
        if use_spatial:
            self.spatial_features = SpatialCompatibilityFeatures(spatial_dim)
            self.spatial_score = nn.Linear(spatial_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_size ** -0.5

    def forward(
        self,
        features: torch.Tensor,      # [batch, num_regions, hidden_size]
        bbox: torch.Tensor = None,   # [batch, num_regions, 4]
        mask: torch.Tensor = None,   # [batch, num_regions]
    ) -> torch.Tensor:
        """
        Args:
            features: [batch, num_regions, hidden_size] enhanced region features
            bbox: [batch, num_regions, 4] bounding boxes for spatial features
            mask: [batch, num_regions] valid region mask

        Returns:
            [batch, num_regions, num_regions] order logits
            order_logits[i,j] > 0 means region i comes before region j
        """
        batch_size, num_regions, _ = features.shape

        # Query/Key projections
        query = self.query_proj(features)  # [B, N, H]
        key = self.key_proj(features)      # [B, N, H]

        # Biaffine scoring: query @ W @ key^T
        # [B, N, H] @ [H, H] @ [B, H, N] -> [B, N, N]
        scores = torch.einsum('bih,hd,bjd->bij', query, self.biaffine_weight, key)
        scores = scores * self.scale + self.biaffine_bias

        # Add spatial compatibility scores
        if self.use_spatial and bbox is not None:
            spatial_feat = self.spatial_features(bbox, bbox)  # [B, N, N, spatial_dim]
            spatial_score = self.spatial_score(spatial_feat).squeeze(-1)  # [B, N, N]
            scores = scores + spatial_score

        # Apply mask
        if mask is not None:
            # Mask invalid positions
            row_mask = ~mask.unsqueeze(2)  # [B, N, 1]
            col_mask = ~mask.unsqueeze(1)  # [B, 1, N]
            combined_mask = row_mask | col_mask
            scores = scores.masked_fill(combined_mask, float('-inf'))

            # Mask diagonal (self-loops)
            diag_mask = torch.eye(num_regions, dtype=torch.bool, device=features.device)
            scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

        return scores


class RelationTypeHead(nn.Module):
    """Relation Type Classification Head

    Predicts relationship type between region pairs:
    - 0: No relation
    - 1: Text region reading order (sequential)
    - 2: Graphical region relation
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_relations: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Bilinear classifier for relation type
        self.head_proj = nn.Linear(hidden_size, hidden_size)
        self.tail_proj = nn.Linear(hidden_size, hidden_size)

        self.bilinear = nn.Bilinear(hidden_size, hidden_size, num_relations)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,  # [batch, num_regions, hidden_size]
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [batch, num_regions, hidden_size]
            mask: [batch, num_regions]

        Returns:
            [batch, num_regions, num_regions, num_relations] relation logits
        """
        batch_size, num_regions, hidden_size = features.shape

        head = self.head_proj(features)  # [B, N, H]
        tail = self.tail_proj(features)  # [B, N, H]

        # Expand for pairwise computation
        head = head.unsqueeze(2).expand(-1, -1, num_regions, -1)  # [B, N, N, H]
        tail = tail.unsqueeze(1).expand(-1, num_regions, -1, -1)  # [B, N, N, H]

        head = head.reshape(batch_size * num_regions * num_regions, hidden_size)
        tail = tail.reshape(batch_size * num_regions * num_regions, hidden_size)

        # Bilinear classification
        relation_logits = self.bilinear(head, tail)  # [B*N*N, num_relations]
        relation_logits = relation_logits.reshape(batch_size, num_regions, num_regions, -1)

        return relation_logits


class OrderModule(nn.Module):
    """Complete Order Module

    Processes detected page objects to determine reading sequences.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 5,
        num_heads: int = 12,
        num_layers: int = 3,
        num_relations: int = 3,
        dropout: float = 0.1,
        use_spatial: bool = True,
        use_visual: bool = False,
        use_text: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Multi-modal embedding
        self.embedding = MultiModalEmbedding(
            hidden_size=hidden_size,
            num_categories=num_categories,
            use_visual=use_visual,
            use_text=use_text,
            dropout=dropout,
        )

        # Transformer encoder
        self.transformer = OrderTransformerEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Reading order prediction head
        self.order_head = ReadingOrderHead(
            hidden_size=hidden_size,
            use_spatial=use_spatial,
            dropout=dropout,
        )

        # Relation type classification head
        self.relation_head = RelationTypeHead(
            hidden_size=hidden_size,
            num_relations=num_relations,
            dropout=dropout,
        )

    def forward(
        self,
        bbox: torch.Tensor,                    # [batch, num_regions, 4]
        categories: torch.Tensor,              # [batch, num_regions]
        region_mask: torch.Tensor,             # [batch, num_regions]
        visual_features: torch.Tensor = None,  # [batch, num_regions, visual_dim]
        text_features: torch.Tensor = None,    # [batch, num_regions, text_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bbox: [batch, num_regions, 4] bounding boxes
            categories: [batch, num_regions] region categories
            region_mask: [batch, num_regions] valid region mask
            visual_features: optional visual features
            text_features: optional text features

        Returns:
            Dict with:
                - embeddings: [batch, num_regions, hidden_size]
                - enhanced_features: [batch, num_regions, hidden_size]
                - order_logits: [batch, num_regions, num_regions]
                - relation_logits: [batch, num_regions, num_regions, num_relations]
        """
        # Multi-modal embedding
        embeddings = self.embedding(
            bbox=bbox,
            categories=categories,
            visual_features=visual_features,
            text_features=text_features,
        )

        # Transformer enhancement
        enhanced = self.transformer(embeddings, mask=region_mask)

        # Reading order prediction
        order_logits = self.order_head(enhanced, bbox=bbox, mask=region_mask)

        # Relation type prediction
        relation_logits = self.relation_head(enhanced, mask=region_mask)

        return {
            'embeddings': embeddings,
            'enhanced_features': enhanced,
            'order_logits': order_logits,
            'relation_logits': relation_logits,
        }


class OrderLoss(nn.Module):
    """Order Module Loss

    Combines:
    - Reading order loss (softmax cross-entropy, dependency parsing style)
    - Relation type loss (cross-entropy)
    """

    def __init__(
        self,
        order_weight: float = 1.0,
        relation_weight: float = 0.5,
        use_softmax: bool = True,  # Paper uses softmax CE instead of BCE
    ):
        super().__init__()
        self.order_weight = order_weight
        self.relation_weight = relation_weight
        self.use_softmax = use_softmax

        if not use_softmax:
            self.order_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.relation_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def forward(
        self,
        order_logits: torch.Tensor,      # [batch, N, N]
        relation_logits: torch.Tensor,   # [batch, N, N, num_relations]
        reading_orders: torch.Tensor,    # [batch, N] ground truth order indices
        relation_labels: torch.Tensor = None,  # [batch, N, N] relation types
        mask: torch.Tensor = None,       # [batch, N]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            order_logits: [batch, N, N] pairwise order scores
            relation_logits: [batch, N, N, num_relations] relation predictions
            reading_orders: [batch, N] ground truth reading order (0, 1, 2, ...)
            relation_labels: [batch, N, N] ground truth relation types
            mask: [batch, N] valid region mask

        Returns:
            Dict with order_loss, relation_loss, total_loss
        """
        batch_size, num_regions = reading_orders.shape
        device = order_logits.device

        # Create ground truth order matrix
        # target[i,j] = 1 if order[i] < order[j] (i comes before j)
        order_i = reading_orders.unsqueeze(2)  # [B, N, 1]
        order_j = reading_orders.unsqueeze(1)  # [B, 1, N]
        order_target = (order_i < order_j).float()  # [B, N, N]

        # Valid pair mask
        if mask is not None:
            valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
        else:
            valid_mask = torch.ones(batch_size, num_regions, num_regions,
                                   dtype=torch.bool, device=device)

        # Exclude diagonal
        diag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=device).unsqueeze(0)
        valid_mask = valid_mask & diag_mask

        # Order loss
        if self.use_softmax:
            # Softmax cross-entropy (dependency parsing style)
            # For each region i, predict which region j it points to (successor)
            # Mask invalid positions with -inf
            order_logits_masked = order_logits.clone()
            order_logits_masked[~valid_mask] = float('-inf')

            # Find successor for each region (next in reading order)
            # successor[i] = argmin_{j: order[j] > order[i]} order[j]
            order_expanded = reading_orders.unsqueeze(2).expand(-1, -1, num_regions)
            order_diff = order_j.expand(-1, num_regions, -1) - order_i.expand(-1, -1, num_regions)

            # Only consider j where order[j] > order[i]
            successor_mask = (order_diff > 0) & valid_mask
            order_diff_masked = order_diff.float()
            order_diff_masked[~successor_mask] = float('inf')

            # Successor is the one with smallest positive order difference
            _, successors = order_diff_masked.min(dim=2)  # [B, N]

            # For regions with no successor (last in order), use a dummy target
            has_successor = successor_mask.any(dim=2)  # [B, N]

            # Cross-entropy loss
            order_logits_flat = order_logits_masked.reshape(-1, num_regions)  # [B*N, N]
            successors_flat = successors.reshape(-1)  # [B*N]
            has_successor_flat = has_successor.reshape(-1)  # [B*N]

            if mask is not None:
                mask_flat = mask.reshape(-1)
                has_successor_flat = has_successor_flat & mask_flat

            # Replace all-inf rows with zeros to avoid NaN in softmax
            all_inf_rows = torch.isinf(order_logits_flat).all(dim=1)
            order_logits_flat = order_logits_flat.clone()
            order_logits_flat[all_inf_rows] = 0.0
            # Also set dummy target for these rows
            successors_flat = successors_flat.clone()
            successors_flat[all_inf_rows] = 0

            order_loss_flat = F.cross_entropy(
                order_logits_flat,
                successors_flat,
                reduction='none'
            )

            # Only count loss for valid regions with successors
            order_loss_flat = order_loss_flat * has_successor_flat.float()
            num_valid = has_successor_flat.sum().clamp(min=1)
            order_loss = order_loss_flat.sum() / num_valid

        else:
            # BCE loss (original implementation)
            order_loss_matrix = self.order_criterion(order_logits, order_target)
            order_loss_matrix = order_loss_matrix * valid_mask.float()
            num_valid = valid_mask.sum().clamp(min=1)
            order_loss = order_loss_matrix.sum() / num_valid

        # Relation loss
        relation_loss = torch.tensor(0.0, device=device)
        if relation_labels is not None and self.relation_weight > 0:
            # [B, N, N, C] -> [B*N*N, C]
            relation_logits_flat = relation_logits.reshape(-1, relation_logits.size(-1))
            relation_labels_flat = relation_labels.reshape(-1)

            relation_loss_flat = self.relation_criterion(
                relation_logits_flat,
                relation_labels_flat
            )

            # Mask
            if mask is not None:
                valid_mask_flat = valid_mask.reshape(-1)
                relation_loss_flat = relation_loss_flat * valid_mask_flat.float()
                num_valid_rel = valid_mask_flat.sum().clamp(min=1)
            else:
                num_valid_rel = relation_loss_flat.numel()

            relation_loss = relation_loss_flat.sum() / num_valid_rel

        # Total loss
        total_loss = self.order_weight * order_loss + self.relation_weight * relation_loss

        return {
            'order_loss': order_loss,
            'relation_loss': relation_loss,
            'loss': total_loss,
        }


def predict_reading_order(order_logits: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Predict reading order from pairwise logits

    Uses greedy decoding: repeatedly select the region with most "comes before" relations.

    Args:
        order_logits: [batch, N, N] pairwise scores
        mask: [batch, N] valid region mask

    Returns:
        [batch, N] predicted reading order indices (0 = first, 1 = second, ...)
    """
    batch_size, num_regions, _ = order_logits.shape
    device = order_logits.device

    # Count how many regions each region comes before
    # Higher score = earlier in reading order
    if mask is not None:
        order_logits = order_logits.clone()
        row_mask = ~mask.unsqueeze(2)
        col_mask = ~mask.unsqueeze(1)
        order_logits[row_mask | col_mask] = float('-inf')

    # Count wins (i comes before j if logits[i,j] > 0)
    wins = (order_logits > 0).float().sum(dim=2)  # [B, N]

    if mask is not None:
        wins[~mask] = -1  # Invalid regions get lowest priority

    # Sort by wins (descending) to get order
    _, order_indices = wins.sort(dim=1, descending=True)

    # Convert to reading order (inverse permutation)
    reading_order = torch.zeros_like(order_indices)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_regions)
    reading_order[batch_indices, order_indices] = torch.arange(num_regions, device=device).unsqueeze(0).expand(batch_size, -1)

    if mask is not None:
        reading_order[~mask] = -1

    return reading_order

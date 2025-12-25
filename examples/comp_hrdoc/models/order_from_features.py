"""Order Module using Pre-extracted Region Features

Simplified Order module that takes pre-extracted LayoutXLM region features
instead of line features. Used for independent 4.3 training.

Based on order.py but skips 4.3.1 (TextRegionAttentionFusion) since
region features are already extracted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .order import (
    OrderTransformerEncoder,
    InterRegionOrderHead,
    RelationTypeHead,
)
from .embeddings import RegionTypeEmbedding


class OrderModuleFromFeatures(nn.Module):
    """Order Module that uses pre-extracted region features.

    This is a simplified version of OrderModule for independent 4.3 training.
    It skips the TextRegionAttentionFusion step since we already have
    region-level features from LayoutXLM.

    Pipeline:
        pre-extracted region_features (768-dim)
            ↓
        Add category/type embedding (Eq. 14)
            ↓
        OrderTransformerEncoder (3 layers, Eq. 15)
            ↓
        InterRegionOrderHead (Eq. 15) + RelationTypeHead (Eq. 16)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 5,  # 0=pad, 1=fig, 2=tab, 3=para, 4=other
        num_heads: int = 12,
        num_layers: int = 3,
        ffn_dim: int = 2048,
        proj_size: int = 2048,
        mlp_hidden: int = 1024,
        num_relations: int = 3,
        dropout: float = 0.1,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Category/type embedding (Eq. 13)
        self.type_embedding = RegionTypeEmbedding(
            num_categories=num_categories,
            hidden_size=hidden_size,
        )

        # Combine pre-extracted features with type embedding (Eq. 14)
        # U_hat = FC(concat(U, R))
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        self.combine_norm = nn.LayerNorm(hidden_size)

        # 4.3.2: Transformer encoder (3 layers)
        self.transformer = OrderTransformerEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # 4.3.3: Inter-region order prediction head
        self.order_head = InterRegionOrderHead(
            hidden_size=hidden_size,
            proj_size=proj_size,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            use_spatial=use_spatial,
        )

        # 4.3.4: Relation type classification head
        self.relation_head = RelationTypeHead(
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_relations=num_relations,
            dropout=dropout,
        )

    def forward(
        self,
        region_features: torch.Tensor,  # [batch, num_regions, hidden_size]
        categories: torch.Tensor,  # [batch, num_regions] category/type ids
        bboxes: torch.Tensor,  # [batch, num_regions, 4]
        region_mask: torch.Tensor,  # [batch, num_regions]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            region_features: [batch, num_regions, hidden_size] pre-extracted features
            categories: [batch, num_regions] category ids (GT during training)
            bboxes: [batch, num_regions, 4] normalized bounding boxes
            region_mask: [batch, num_regions] valid region mask

        Returns:
            Dict with:
                - enhanced_features: [batch, num_regions, hidden_size]
                - order_logits: [batch, num_regions, num_regions]
                - relation_logits: [batch, num_regions, num_regions, num_relations]
        """
        # Add type embedding (Eq. 13-14)
        type_emb = self.type_embedding(categories)  # [batch, num_regions, hidden_size]

        # Combine features: FC(concat(U, R))
        combined = torch.cat([region_features, type_emb], dim=-1)
        combined = self.combine(combined)
        combined = self.combine_norm(combined)

        # 4.3.2: Transformer enhancement
        enhanced = self.transformer(combined, mask=region_mask)

        # 4.3.3: Order prediction
        order_logits = self.order_head(enhanced, bbox=bboxes, mask=region_mask)

        # 4.3.4: Relation type prediction
        relation_logits = self.relation_head(enhanced, mask=region_mask)

        return {
            'enhanced_features': enhanced,
            'order_logits': order_logits,
            'relation_logits': relation_logits,
        }


class OrderLossFromFeatures(nn.Module):
    """Loss function for Order module with pre-extracted features.

    Combines:
    - Reading order loss (dependency parsing style)
    - Relation type loss (cross-entropy)
    """

    def __init__(
        self,
        order_weight: float = 1.0,
        relation_weight: float = 0.5,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.order_weight = order_weight
        self.relation_weight = relation_weight
        self.ignore_index = ignore_index

    def forward(
        self,
        order_logits: torch.Tensor,  # [batch, N, N]
        relation_logits: torch.Tensor,  # [batch, N, N, num_relations]
        reading_orders: torch.Tensor,  # [batch, N] reading order indices
        parent_ids: torch.Tensor,  # [batch, N] parent indices (-1 for root)
        relations: torch.Tensor,  # [batch, N] relation types
        region_mask: torch.Tensor,  # [batch, N]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            order_logits: [batch, N, N] pairwise order scores
            relation_logits: [batch, N, N, num_relations]
            reading_orders: [batch, N] ground truth reading order (0, 1, 2, ...)
            parent_ids: [batch, N] parent region index (-1 for root)
            relations: [batch, N] relation type for each region
            region_mask: [batch, N] valid region mask

        Returns:
            Dict with loss values
        """
        batch_size, num_regions = reading_orders.shape
        device = order_logits.device

        # ============ Reading Order Loss ============
        # Use pairwise BCE: predict i < j if order[i] < order[j]
        order_i = reading_orders.unsqueeze(2)  # [B, N, 1]
        order_j = reading_orders.unsqueeze(1)  # [B, 1, N]
        targets = (order_i < order_j).float()  # [B, N, N]

        # Valid pair mask
        valid_mask = region_mask.unsqueeze(2) & region_mask.unsqueeze(1)  # [B, N, N]
        diag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=device).unsqueeze(0)
        valid_mask = valid_mask & diag_mask

        # BCE loss
        order_loss = F.binary_cross_entropy_with_logits(
            order_logits, targets, reduction='none'
        )
        order_loss = (order_loss * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)

        # ============ Relation Type Loss ============
        # For each region, predict its relation type to its parent
        # relation_logits: [B, N, N, C]
        # We want: relation_logits[b, parent_id[b, i], i, :] -> relations[b, i]

        relation_loss = torch.tensor(0.0, device=device)
        num_valid_relations = 0

        for b in range(batch_size):
            for i in range(num_regions):
                if not region_mask[b, i]:
                    continue
                parent_idx = parent_ids[b, i].item()
                rel_type = relations[b, i].item()

                if parent_idx >= 0 and parent_idx < num_regions and rel_type >= 0:
                    # Get logits for (parent -> child) pair
                    pair_logits = relation_logits[b, parent_idx, i]  # [num_relations]
                    target = torch.tensor(rel_type, device=device, dtype=torch.long)
                    relation_loss = relation_loss + F.cross_entropy(
                        pair_logits.unsqueeze(0), target.unsqueeze(0)
                    )
                    num_valid_relations += 1

        if num_valid_relations > 0:
            relation_loss = relation_loss / num_valid_relations

        # ============ Total Loss ============
        total_loss = self.order_weight * order_loss + self.relation_weight * relation_loss

        return {
            'loss': total_loss,
            'order_loss': order_loss,
            'relation_loss': relation_loss,
        }


def build_order_from_features(
    hidden_size: int = 768,
    num_categories: int = 5,
    num_heads: int = 12,
    num_layers: int = 3,
    dropout: float = 0.1,
    use_spatial: bool = True,
) -> OrderModuleFromFeatures:
    """Build Order module for training with pre-extracted features."""
    return OrderModuleFromFeatures(
        hidden_size=hidden_size,
        num_categories=num_categories,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_spatial=use_spatial,
    )


def save_order_model(model: OrderModuleFromFeatures, save_path: str):
    """Save model checkpoint."""
    import os
    import json

    os.makedirs(save_path, exist_ok=True)

    # Save weights
    model_path = os.path.join(save_path, "order_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save config
    config = {
        'hidden_size': model.hidden_size,
        'num_categories': model.type_embedding.embedding.num_embeddings,
    }
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {save_path}")


def load_order_model(model_path: str, device: str = "cuda") -> OrderModuleFromFeatures:
    """Load model checkpoint."""
    import os
    import json

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = build_order_from_features(**config)

    weights_path = os.path.join(model_path, "order_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    return model.to(device)

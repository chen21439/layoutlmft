"""Construct Module using Pre-trained Order Features (4.4)

使用已训练好的 4.3 Order 模型提取的特征来训练 4.4 Construct 模块。
支持两种模式：
1. 使用预训练 Order 模型的 enhanced_features
2. 使用预提取的 region_features（简化版）

Based on "Detect-Order-Construct" paper Section 4.4.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .construct import (
    ConstructModule,
    ConstructLoss,
    build_tree_from_predictions,
)
from .order_from_features import (
    OrderModuleFromFeatures,
    load_order_model,
)
from .embeddings import RegionTypeEmbedding


class ConstructWithOrderFeatures(nn.Module):
    """Construct Module using pre-trained Order model features.

    Pipeline:
        Pre-trained Order Model (frozen)
            ↓
        enhanced_features [B, N, 768]
            ↓
        GT reading_order [B, N]
            ↓
        ConstructModule (trainable)
            ↓
        parent/sibling predictions
    """

    def __init__(
        self,
        order_model: OrderModuleFromFeatures,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 3,
        dropout: float = 0.1,
        freeze_order: bool = True,
    ):
        """
        Args:
            order_model: Pre-trained Order module
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of Transformer layers in Construct
            dropout: Dropout rate
            freeze_order: Whether to freeze Order model parameters
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.freeze_order = freeze_order

        # Pre-trained Order model
        self.order_model = order_model

        # Freeze Order model if specified
        if freeze_order:
            for param in self.order_model.parameters():
                param.requires_grad = False
            self.order_model.eval()

        # Construct module (trainable)
        self.construct_module = ConstructModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Loss function
        self.loss_fn = ConstructLoss()

    def forward(
        self,
        region_features: torch.Tensor,  # [batch, num_regions, hidden_size]
        categories: torch.Tensor,        # [batch, num_regions]
        bboxes: torch.Tensor,            # [batch, num_regions, 4]
        region_mask: torch.Tensor,       # [batch, num_regions]
        reading_orders: torch.Tensor,    # [batch, num_regions] GT reading order
        parent_labels: torch.Tensor = None,   # [batch, num_regions] GT parent indices
        sibling_labels: torch.Tensor = None,  # [batch, num_regions, num_regions]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            region_features: Pre-extracted LayoutXLM features
            categories: Region category IDs
            bboxes: Bounding boxes
            region_mask: Valid region mask
            reading_orders: Ground truth reading order (0, 1, 2, ...)
            parent_labels: Ground truth parent indices (-1 for root)
            sibling_labels: Ground truth sibling matrix

        Returns:
            Dict with construct predictions and loss
        """
        device = region_features.device

        # Get enhanced features from Order model
        if self.freeze_order:
            with torch.no_grad():
                order_outputs = self.order_model(
                    region_features=region_features,
                    categories=categories,
                    bboxes=bboxes,
                    region_mask=region_mask,
                )
        else:
            order_outputs = self.order_model(
                region_features=region_features,
                categories=categories,
                bboxes=bboxes,
                region_mask=region_mask,
            )

        enhanced_features = order_outputs['enhanced_features']

        # Construct module with GT reading order
        construct_outputs = self.construct_module(
            features=enhanced_features,
            reading_order=reading_orders,
            mask=region_mask,
        )

        outputs = {
            'enhanced_features': enhanced_features,
            'construct_features': construct_outputs['construct_features'],
            'parent_logits': construct_outputs['parent_logits'],
            'sibling_logits': construct_outputs['sibling_logits'],
            'order_logits': order_outputs['order_logits'],  # From frozen Order
        }

        # Compute loss if labels provided
        if parent_labels is not None:
            loss_dict = self.loss_fn(
                parent_logits=construct_outputs['parent_logits'],
                sibling_logits=construct_outputs['sibling_logits'],
                parent_labels=parent_labels,
                sibling_labels=sibling_labels,
                mask=region_mask,
            )
            outputs['loss'] = loss_dict['loss']
            outputs['parent_loss'] = loss_dict['parent_loss']
            outputs['sibling_loss'] = loss_dict['sibling_loss']

        return outputs

    def predict(
        self,
        region_features: torch.Tensor,
        categories: torch.Tensor,
        bboxes: torch.Tensor,
        region_mask: torch.Tensor,
        reading_orders: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference mode.

        If reading_orders not provided, use Order model to predict.
        """
        with torch.no_grad():
            order_outputs = self.order_model(
                region_features=region_features,
                categories=categories,
                bboxes=bboxes,
                region_mask=region_mask,
            )

            # Use GT or predicted reading order
            if reading_orders is None:
                # Predict from order logits
                from .order import predict_reading_order
                reading_orders = predict_reading_order(
                    order_outputs['order_logits'],
                    mask=region_mask,
                )

            construct_outputs = self.construct_module(
                features=order_outputs['enhanced_features'],
                reading_order=reading_orders,
                mask=region_mask,
            )

            # Build trees
            batch_size = region_features.size(0)
            trees = []
            for b in range(batch_size):
                tree = build_tree_from_predictions(
                    parent_logits=construct_outputs['parent_logits'][b],
                    mask=region_mask[b] if region_mask is not None else None,
                )
                trees.append(tree)

            return {
                'reading_order': reading_orders,
                'trees': trees,
                'parent_logits': construct_outputs['parent_logits'],
                'sibling_logits': construct_outputs['sibling_logits'],
            }


class ConstructFromFeatures(nn.Module):
    """Simplified Construct Module using pre-extracted features directly.

    不依赖 Order 模型，直接使用预提取的 region_features。
    适用于简化训练场景。

    Pipeline:
        region_features [B, N, 768]
            ↓
        Add type embedding
            ↓
        ConstructModule (with RoPE using reading_order)
            ↓
        parent/sibling predictions
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 5,
        num_heads: int = 12,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Category embedding
        self.type_embedding = RegionTypeEmbedding(
            num_categories=num_categories,
            hidden_size=hidden_size,
        )

        # Combine features with type embedding
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        self.combine_norm = nn.LayerNorm(hidden_size)

        # Construct module
        self.construct_module = ConstructModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Loss function
        self.loss_fn = ConstructLoss()

    def forward(
        self,
        region_features: torch.Tensor,
        categories: torch.Tensor,
        region_mask: torch.Tensor,
        reading_orders: torch.Tensor,
        parent_labels: torch.Tensor = None,
        sibling_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Add type embedding
        type_emb = self.type_embedding(categories)
        combined = torch.cat([region_features, type_emb], dim=-1)
        features = self.combine(combined)
        features = self.combine_norm(features)

        # Construct with RoPE
        construct_outputs = self.construct_module(
            features=features,
            reading_order=reading_orders,
            mask=region_mask,
        )

        outputs = {
            'construct_features': construct_outputs['construct_features'],
            'parent_logits': construct_outputs['parent_logits'],
            'sibling_logits': construct_outputs['sibling_logits'],
        }

        # Compute loss
        if parent_labels is not None:
            loss_dict = self.loss_fn(
                parent_logits=construct_outputs['parent_logits'],
                sibling_logits=construct_outputs['sibling_logits'],
                parent_labels=parent_labels,
                sibling_labels=sibling_labels,
                mask=region_mask,
            )
            outputs['loss'] = loss_dict['loss']
            outputs['parent_loss'] = loss_dict['parent_loss']
            outputs['sibling_loss'] = loss_dict['sibling_loss']

        return outputs


# ==================== Factory Functions ====================

def build_construct_with_order(
    order_model_path: str,
    hidden_size: int = 768,
    num_heads: int = 12,
    num_layers: int = 3,
    dropout: float = 0.1,
    freeze_order: bool = True,
    device: str = "cuda",
) -> ConstructWithOrderFeatures:
    """Build Construct model with pre-trained Order model.

    Args:
        order_model_path: Path to pre-trained Order model checkpoint
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of Construct Transformer layers
        dropout: Dropout rate
        freeze_order: Whether to freeze Order model
        device: Device to load model on

    Returns:
        ConstructWithOrderFeatures model
    """
    # Load pre-trained Order model
    order_model = load_order_model(order_model_path, device=device)

    model = ConstructWithOrderFeatures(
        order_model=order_model,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        freeze_order=freeze_order,
    )

    return model


def build_construct_from_features(
    hidden_size: int = 768,
    num_categories: int = 5,
    num_heads: int = 12,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> ConstructFromFeatures:
    """Build simplified Construct model."""
    return ConstructFromFeatures(
        hidden_size=hidden_size,
        num_categories=num_categories,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


# ==================== Save/Load ====================

def save_construct_model(
    model: ConstructWithOrderFeatures,
    save_path: str,
    save_order: bool = False,
):
    """Save Construct model checkpoint.

    Args:
        model: Construct model to save (ConstructFromFeatures or ConstructWithOrderFeatures)
        save_path: Directory to save to
        save_order: Whether to also save Order model weights (only for ConstructWithOrderFeatures)
    """
    os.makedirs(save_path, exist_ok=True)

    # 保存完整模型权重（包括 type_embedding, combine 等）
    model_path = os.path.join(save_path, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)

    # 兼容旧格式：也保存 construct_module 权重
    construct_path = os.path.join(save_path, "construct_model.pt")
    torch.save(model.construct_module.state_dict(), construct_path)

    # Optionally save full model (including Order) for ConstructWithOrderFeatures
    if save_order and hasattr(model, 'order_model'):
        full_path = os.path.join(save_path, "full_model.pt")
        torch.save(model.state_dict(), full_path)

    # Save config
    config = {
        'hidden_size': model.hidden_size,
        'num_layers': model.construct_module.transformer.layers.__len__(),
        'num_categories': getattr(model, 'type_embedding', None) and model.type_embedding.num_categories or 5,
        'freeze_order': getattr(model, 'freeze_order', True),
        'model_type': model.__class__.__name__,
    }
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Construct model saved to {save_path}")


def load_construct_model(
    construct_path: str,
    order_model_path: str,
    device: str = "cuda",
) -> ConstructWithOrderFeatures:
    """Load Construct model from checkpoint.

    Args:
        construct_path: Path to Construct checkpoint
        order_model_path: Path to Order model checkpoint
        device: Device to load on

    Returns:
        ConstructWithOrderFeatures model
    """
    # Load config
    config_path = os.path.join(construct_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Build model
    model = build_construct_with_order(
        order_model_path=order_model_path,
        hidden_size=config['hidden_size'],
        num_layers=config.get('num_layers', 3),
        freeze_order=config.get('freeze_order', True),
        device=device,
    )

    # Load Construct weights
    construct_weights_path = os.path.join(construct_path, "construct_model.pt")
    model.construct_module.load_state_dict(
        torch.load(construct_weights_path, map_location=device)
    )

    return model.to(device)


# ==================== Metrics ====================

def compute_construct_metrics(
    parent_logits: torch.Tensor,  # [batch, N, N]
    parent_labels: torch.Tensor,  # [batch, N]
    region_mask: torch.Tensor,    # [batch, N]
    sibling_logits: torch.Tensor = None,  # [batch, N, N]
    sibling_labels: torch.Tensor = None,  # [batch, N]
) -> Dict[str, float]:
    """Compute Construct module evaluation metrics.

    论文自指向方案：所有有效节点都计算指标，包括 root（自指向）。

    Args:
        parent_logits: Parent prediction logits
        parent_labels: Ground truth parent indices (self-index for root)
        region_mask: Valid region mask
        sibling_logits: Sibling prediction logits (optional)
        sibling_labels: Ground truth left sibling indices (optional)

    Returns:
        Dict with parent_accuracy and sibling_accuracy
    """
    batch_size, num_regions = parent_labels.shape
    device = parent_logits.device

    # Predict parent for each node
    pred_parents = parent_logits.argmax(dim=-1)  # [batch, N]

    # Parent accuracy: 论文自指向方案，所有有效节点都计算
    valid = (parent_labels >= 0) & (parent_labels < num_regions) & region_mask
    if valid.any():
        correct = (pred_parents == parent_labels) & valid
        parent_acc = correct.sum().float() / valid.sum().float()
    else:
        parent_acc = torch.tensor(0.0, device=device)

    result = {
        'parent_accuracy': parent_acc.item() if isinstance(parent_acc, torch.Tensor) else parent_acc,
    }

    # Sibling accuracy (if provided)
    if sibling_logits is not None and sibling_labels is not None:
        pred_siblings = sibling_logits.argmax(dim=-1)  # [batch, N]
        has_sibling = (sibling_labels >= 0) & region_mask
        if has_sibling.any():
            correct = (pred_siblings == sibling_labels) & has_sibling
            sibling_acc = correct.sum().float() / has_sibling.sum().float()
        else:
            sibling_acc = torch.tensor(0.0, device=device)
        result['sibling_accuracy'] = sibling_acc.item() if isinstance(sibling_acc, torch.Tensor) else sibling_acc

    return result


def generate_sibling_labels(
    parent_ids: torch.Tensor,  # [batch, N]
    reading_orders: torch.Tensor,  # [batch, N] reading order positions
    region_mask: torch.Tensor,  # [batch, N]
) -> torch.Tensor:
    """Generate left sibling labels from parent IDs and reading order.

    For each node, find its left sibling (same parent, immediately before in reading order).
    论文自指向方案：root 节点的 parent_id == self_index。

    Args:
        parent_ids: [batch, N] parent index for each node (self-index for root)
        reading_orders: [batch, N] reading order positions (0, 1, 2, ...)
        region_mask: [batch, N] valid region mask

    Returns:
        sibling_labels: [batch, N] index of left sibling (-1 if no left sibling)
    """
    batch_size, num_regions = parent_ids.shape
    device = parent_ids.device

    sibling_labels = torch.full(
        (batch_size, num_regions),
        fill_value=-1,
        dtype=torch.long, device=device
    )

    for b in range(batch_size):
        # Group nodes by parent
        parent_to_children = {}
        for i in range(num_regions):
            if not region_mask[b, i]:
                continue
            parent_i = parent_ids[b, i].item()
            # 论文自指向方案：root 自指向，跳过
            if parent_i == i:  # Root (self-pointing) has no siblings
                continue
            if parent_i not in parent_to_children:
                parent_to_children[parent_i] = []
            parent_to_children[parent_i].append(i)

        # For each group of siblings, sort by reading order and assign left sibling
        for parent, children in parent_to_children.items():
            if len(children) < 2:
                continue
            # Sort by reading order
            children_sorted = sorted(children, key=lambda x: reading_orders[b, x].item())
            # Assign left sibling for each node (except the first one)
            for idx in range(1, len(children_sorted)):
                curr_node = children_sorted[idx]
                left_sibling = children_sorted[idx - 1]
                sibling_labels[b, curr_node] = left_sibling

    return sibling_labels


def generate_sibling_matrix(
    parent_ids: torch.Tensor,  # [batch, N]
    region_mask: torch.Tensor,  # [batch, N]
) -> torch.Tensor:
    """Generate sibling matrix from parent IDs (legacy format).

    Two nodes are siblings if they have the same parent.
    This is kept for backward compatibility.
    论文自指向方案：root 节点的 parent_id == self_index。

    Args:
        parent_ids: [batch, N] parent index for each node (self-index for root)
        region_mask: [batch, N] valid region mask

    Returns:
        sibling_matrix: [batch, N, N] 1 if siblings, 0 otherwise
    """
    batch_size, num_regions = parent_ids.shape
    device = parent_ids.device

    sibling_matrix = torch.zeros(
        batch_size, num_regions, num_regions,
        dtype=torch.long, device=device
    )

    for b in range(batch_size):
        for i in range(num_regions):
            if not region_mask[b, i]:
                continue
            parent_i = parent_ids[b, i].item()
            # 论文自指向方案：root 自指向，跳过
            if parent_i == i:  # Root (self-pointing) has no siblings
                continue

            for j in range(num_regions):
                if i == j or not region_mask[b, j]:
                    continue
                parent_j = parent_ids[b, j].item()
                # 两个节点有相同的 parent（且都不是 root）则为 sibling
                if parent_i == parent_j and parent_j != j:
                    sibling_matrix[b, i, j] = 1

    return sibling_matrix

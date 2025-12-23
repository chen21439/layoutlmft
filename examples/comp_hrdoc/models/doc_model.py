"""Unified DOC (Detect-Order-Construct) Model

Based on "Detect-Order-Construct: A Unified Framework for
Hierarchical Document Structure Analysis" paper.

Combines three stages:
1. Detect: Region detection and classification (handled by data)
2. Order: Reading order prediction
3. Construct: Hierarchical structure construction
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any

from .order import OrderModule, OrderLoss, predict_reading_order
from .construct import ConstructModule, ConstructLoss, build_tree_from_predictions


class DOCModel(nn.Module):
    """Complete Detect-Order-Construct Model

    This model takes detected regions (from Detect stage) and:
    1. Predicts reading order (Order stage)
    2. Predicts hierarchical structure (Construct stage)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 5,
        num_heads: int = 12,
        order_num_layers: int = 3,
        construct_num_layers: int = 3,
        num_relations: int = 3,
        dropout: float = 0.1,
        use_spatial: bool = True,
        use_visual: bool = False,
        use_text: bool = False,
        use_construct: bool = True,
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            num_categories: Number of region categories
            num_heads: Number of attention heads
            order_num_layers: Number of Transformer layers in Order module
            construct_num_layers: Number of Transformer layers in Construct module
            num_relations: Number of relation types
            dropout: Dropout rate
            use_spatial: Whether to use spatial features in Order module
            use_visual: Whether to use visual features
            use_text: Whether to use text features
            use_construct: Whether to use Construct module
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.use_construct = use_construct

        # Order Module
        self.order_module = OrderModule(
            hidden_size=hidden_size,
            num_categories=num_categories,
            num_heads=num_heads,
            num_layers=order_num_layers,
            num_relations=num_relations,
            dropout=dropout,
            use_spatial=use_spatial,
            use_visual=use_visual,
            use_text=use_text,
        )

        # Construct Module (optional)
        if use_construct:
            self.construct_module = ConstructModule(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=construct_num_layers,
                dropout=dropout,
            )

        # Loss functions
        self.order_loss_fn = OrderLoss()
        if use_construct:
            self.construct_loss_fn = ConstructLoss()

    def forward(
        self,
        bbox: torch.Tensor,                    # [batch, num_regions, 4]
        categories: torch.Tensor,              # [batch, num_regions]
        region_mask: torch.Tensor,             # [batch, num_regions]
        visual_features: torch.Tensor = None,  # [batch, num_regions, visual_dim]
        text_features: torch.Tensor = None,    # [batch, num_regions, text_dim]
        reading_orders: torch.Tensor = None,   # [batch, num_regions] GT reading order
        relation_labels: torch.Tensor = None,  # [batch, num_regions, num_regions]
        parent_labels: torch.Tensor = None,    # [batch, num_regions] GT parent indices
        sibling_labels: torch.Tensor = None,   # [batch, num_regions, num_regions]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through DOC model

        Args:
            bbox: Bounding boxes [batch, num_regions, 4]
            categories: Region categories [batch, num_regions]
            region_mask: Valid region mask [batch, num_regions]
            visual_features: Optional visual features
            text_features: Optional text features
            reading_orders: Ground truth reading orders (for training)
            relation_labels: Ground truth relation labels (for training)
            parent_labels: Ground truth parent indices (for training)
            sibling_labels: Ground truth sibling matrix (for training)

        Returns:
            Dict containing:
                - order_logits: [batch, num_regions, num_regions]
                - relation_logits: [batch, num_regions, num_regions, num_relations]
                - enhanced_features: [batch, num_regions, hidden_size]
                - parent_logits: [batch, num_regions, num_regions] (if use_construct)
                - sibling_logits: [batch, num_regions, num_regions, 2] (if use_construct)
                - root_logits: [batch, num_regions] (if use_construct)
                - loss: Total loss (if training labels provided)
                - order_loss: Order module loss
                - construct_loss: Construct module loss (if use_construct)
        """
        device = bbox.device
        outputs = {}

        # ==================== Order Stage ====================
        order_outputs = self.order_module(
            bbox=bbox,
            categories=categories,
            region_mask=region_mask,
            visual_features=visual_features,
            text_features=text_features,
        )

        outputs['embeddings'] = order_outputs['embeddings']
        outputs['enhanced_features'] = order_outputs['enhanced_features']
        outputs['order_logits'] = order_outputs['order_logits']
        outputs['relation_logits'] = order_outputs['relation_logits']

        # Order loss
        order_loss = torch.tensor(0.0, device=device)
        if reading_orders is not None:
            order_loss_dict = self.order_loss_fn(
                order_logits=order_outputs['order_logits'],
                relation_logits=order_outputs['relation_logits'],
                reading_orders=reading_orders,
                relation_labels=relation_labels,
                mask=region_mask,
            )
            order_loss = order_loss_dict['loss']
            outputs['order_loss'] = order_loss_dict['order_loss']
            outputs['relation_loss'] = order_loss_dict['relation_loss']

        # ==================== Construct Stage ====================
        construct_loss = torch.tensor(0.0, device=device)

        if self.use_construct:
            # Use predicted or ground truth reading order
            if reading_orders is not None:
                positions = reading_orders
            else:
                # Predict reading order from logits
                positions = predict_reading_order(
                    order_outputs['order_logits'],
                    mask=region_mask
                )

            construct_outputs = self.construct_module(
                features=order_outputs['enhanced_features'],
                reading_order=positions,
                mask=region_mask,
            )

            outputs['construct_features'] = construct_outputs['construct_features']
            outputs['parent_logits'] = construct_outputs['parent_logits']
            outputs['sibling_logits'] = construct_outputs['sibling_logits']
            outputs['root_logits'] = construct_outputs['root_logits']

            # Construct loss
            if parent_labels is not None:
                construct_loss_dict = self.construct_loss_fn(
                    parent_logits=construct_outputs['parent_logits'],
                    sibling_logits=construct_outputs['sibling_logits'],
                    root_logits=construct_outputs['root_logits'],
                    parent_labels=parent_labels,
                    sibling_labels=sibling_labels,
                    mask=region_mask,
                )
                construct_loss = construct_loss_dict['loss']
                outputs['construct_loss'] = construct_loss
                outputs['parent_loss'] = construct_loss_dict['parent_loss']
                outputs['sibling_loss'] = construct_loss_dict['sibling_loss']
                outputs['root_loss'] = construct_loss_dict['root_loss']

        # Total loss
        total_loss = order_loss + construct_loss
        outputs['loss'] = total_loss

        # For compatibility with order_only training
        outputs['cls_loss'] = torch.tensor(0.0, device=device)

        return outputs

    def predict(
        self,
        bbox: torch.Tensor,
        categories: torch.Tensor,
        region_mask: torch.Tensor,
        visual_features: torch.Tensor = None,
        text_features: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """Inference mode - predict reading order and hierarchy

        Returns:
            Dict with:
                - reading_order: [batch, num_regions] predicted reading order
                - tree_structure: List of tree nodes (if use_construct)
        """
        with torch.no_grad():
            outputs = self.forward(
                bbox=bbox,
                categories=categories,
                region_mask=region_mask,
                visual_features=visual_features,
                text_features=text_features,
            )

            # Predict reading order
            reading_order = predict_reading_order(
                outputs['order_logits'],
                mask=region_mask,
            )

            results = {
                'reading_order': reading_order,
                'order_logits': outputs['order_logits'],
            }

            # Predict hierarchy
            if self.use_construct:
                batch_size = bbox.size(0)
                trees = []
                for b in range(batch_size):
                    tree = build_tree_from_predictions(
                        parent_logits=outputs['parent_logits'][b],
                        root_logits=outputs['root_logits'][b],
                        mask=region_mask[b] if region_mask is not None else None,
                    )
                    trees.append(tree)
                results['tree_structure'] = trees
                results['parent_logits'] = outputs['parent_logits']
                results['root_logits'] = outputs['root_logits']

            return results


def build_doc_model(
    hidden_size: int = 768,
    num_categories: int = 5,
    num_heads: int = 12,
    order_num_layers: int = 3,
    construct_num_layers: int = 3,
    num_relations: int = 3,
    dropout: float = 0.1,
    use_spatial: bool = True,
    use_visual: bool = False,
    use_text: bool = False,
    use_construct: bool = True,
) -> DOCModel:
    """Build DOC model with specified configuration"""
    return DOCModel(
        hidden_size=hidden_size,
        num_categories=num_categories,
        num_heads=num_heads,
        order_num_layers=order_num_layers,
        construct_num_layers=construct_num_layers,
        num_relations=num_relations,
        dropout=dropout,
        use_spatial=use_spatial,
        use_visual=use_visual,
        use_text=use_text,
        use_construct=use_construct,
    )


def save_doc_model(model: DOCModel, save_path: str, config: dict = None):
    """Save DOC model and configuration"""
    os.makedirs(save_path, exist_ok=True)

    # Save model weights
    model_path = os.path.join(save_path, "doc_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save configuration
    if config is None:
        config = {
            'hidden_size': model.hidden_size,
            'use_construct': model.use_construct,
        }

    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {save_path}")


def load_doc_model(
    model_path: str,
    device: str = "cuda",
    **override_config
) -> DOCModel:
    """Load DOC model from checkpoint"""

    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Override with provided config
    config.update(override_config)

    # Build model
    model = build_doc_model(**config)

    # Load weights
    weights_path = os.path.join(model_path, "doc_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    return model.to(device)


# ==================== Metrics ====================

def compute_order_accuracy(
    pred_order: torch.Tensor,  # [batch, N]
    true_order: torch.Tensor,  # [batch, N]
    mask: torch.Tensor = None, # [batch, N]
) -> float:
    """Compute pairwise order accuracy

    For each valid pair (i, j), check if relative order is correct.
    """
    batch_size, num_regions = pred_order.shape

    if mask is None:
        mask = torch.ones_like(pred_order, dtype=torch.bool)

    total_correct = 0
    total_pairs = 0

    for b in range(batch_size):
        valid = mask[b]
        p_order = pred_order[b][valid]
        t_order = true_order[b][valid]

        n = valid.sum().item()
        if n < 2:
            continue

        # Compute pairwise correctness
        for i in range(n):
            for j in range(i + 1, n):
                # True: i before j if t_order[i] < t_order[j]
                true_i_before_j = t_order[i] < t_order[j]
                pred_i_before_j = p_order[i] < p_order[j]

                if true_i_before_j == pred_i_before_j:
                    total_correct += 1
                total_pairs += 1

    if total_pairs == 0:
        return 0.0

    return total_correct / total_pairs


def compute_tree_accuracy(
    pred_parents: torch.Tensor,  # [batch, N, N] parent logits
    true_parents: torch.Tensor,  # [batch, N] parent indices
    mask: torch.Tensor = None,
) -> Dict[str, float]:
    """Compute tree structure accuracy

    Returns:
        Dict with parent_accuracy, root_accuracy
    """
    batch_size, num_nodes = true_parents.shape

    if mask is None:
        mask = torch.ones_like(true_parents, dtype=torch.bool)

    # Predict parent for each node
    pred_parent_indices = pred_parents.argmax(dim=-1)  # [batch, N]

    # Parent accuracy (excluding roots)
    has_parent = (true_parents >= 0) & mask
    if has_parent.any():
        correct = (pred_parent_indices == true_parents) & has_parent
        parent_acc = correct.sum().float() / has_parent.sum().float()
    else:
        parent_acc = 0.0

    # Root accuracy
    is_root = (true_parents == -1) & mask
    pred_is_root = (pred_parent_indices == torch.arange(
        num_nodes, device=pred_parents.device
    ).unsqueeze(0))  # diagonal = self-reference = root

    # For root detection, we need to check root_logits separately
    # This is a simplified version
    root_acc = 0.0

    return {
        'parent_accuracy': parent_acc.item() if isinstance(parent_acc, torch.Tensor) else parent_acc,
        'root_accuracy': root_acc,
    }

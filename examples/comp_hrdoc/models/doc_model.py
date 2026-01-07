"""Unified DOC (Detect-Order-Construct) Model

Based on "Detect-Order-Construct: A Unified Framework for
Hierarchical Document Structure Analysis" paper.

Combines three stages:
1. Detect: Region detection (handled by data)
2. Semantic Classification: Region type classification (4.2)
3. Order: Reading order prediction (4.3)
4. Construct: Hierarchical structure construction (4.4)

End-to-end training: 4.2 + 4.3 + 4.4
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any

import torch.nn.functional as F

from .order import (
    OrderTransformerEncoder,
    InterRegionOrderHead,
    RelationTypeHead,
    OrderLoss,
    predict_reading_order,
)
from .embeddings import RegionTypeEmbedding, PositionalEmbedding2D
from .construct import ConstructModule, ConstructLoss, build_tree_from_predictions


# ==================== 4.2 Semantic Classification Module ====================

class SemanticClassificationModule(nn.Module):
    """Semantic Classification Module (Paper Section 4.2)

    Predicts region categories (figure, table, paragraph, etc.) from
    spatial features. This enables end-to-end training without requiring
    pre-labeled categories.

    Architecture:
        bbox → PositionalEmbedding2D → [TransformerEncoder] → ClassificationHead → category_logits
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 5,  # 0=pad, 1=fig, 2=tab, 3=para, 4=other
        num_heads: int = 12,
        num_layers: int = 1,  # Lightweight: 1 layer is sufficient
        dropout: float = 0.1,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_categories = num_categories
        self.use_transformer = use_transformer

        # Position embedding from bbox
        self.pos_embedding = PositionalEmbedding2D(
            hidden_size=hidden_size,
            use_learned=True,
        )

        # Optional Transformer for context modeling
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_categories),
        )

    def forward(
        self,
        bbox: torch.Tensor,           # [batch, num_regions, 4]
        region_mask: torch.Tensor,    # [batch, num_regions]
        visual_features: torch.Tensor = None,  # [batch, num_regions, hidden_size]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bbox: Normalized bounding boxes [batch, N, 4]
            region_mask: Valid region mask [batch, N]
            visual_features: Optional visual features from backbone

        Returns:
            Dict with:
                - category_logits: [batch, N, num_categories]
                - category_features: [batch, N, hidden_size] encoded features
        """
        batch_size, num_regions = bbox.shape[:2]

        # Get positional features
        pos_features = self.pos_embedding(bbox)  # [B, N, H]

        # Combine with visual features if available
        if visual_features is not None:
            features = pos_features + visual_features
        else:
            features = pos_features

        # Apply Transformer for context
        if self.use_transformer:
            # Create attention mask (True = masked/ignored)
            attn_mask = ~region_mask  # [B, N]
            features = self.transformer(
                features,
                src_key_padding_mask=attn_mask,
            )

        # Classification
        category_logits = self.classifier(features)  # [B, N, num_categories]

        return {
            'category_logits': category_logits,
            'category_features': features,
        }


class SemanticClassificationLoss(nn.Module):
    """Loss function for semantic classification"""

    def __init__(self, ignore_index: int = 0):  # 0 is padding
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(
        self,
        logits: torch.Tensor,   # [batch, N, num_categories]
        labels: torch.Tensor,   # [batch, N]
        mask: torch.Tensor,     # [batch, N]
    ) -> torch.Tensor:
        """Compute classification loss with masking"""
        batch_size, num_regions, num_classes = logits.shape

        # Flatten for cross entropy
        logits_flat = logits.view(-1, num_classes)
        labels_flat = labels.view(-1)

        # Compute loss
        loss = self.criterion(logits_flat, labels_flat)
        loss = loss.view(batch_size, num_regions)

        # Apply mask
        loss = (loss * mask.float()).sum() / mask.sum().clamp(min=1)

        return loss


class PairwiseOrderLoss(nn.Module):
    """Order Loss using pairwise BCE with reading order indices.

    Unlike OrderLoss (which uses successor indices), this loss function
    works with reading order positions (0, 1, 2, ...) and computes
    pairwise binary cross-entropy.

    Predicts: i comes before j if reading_order[i] < reading_order[j]
    """

    def __init__(
        self,
        order_weight: float = 1.0,
        relation_weight: float = 0.5,
    ):
        super().__init__()
        self.order_weight = order_weight
        self.relation_weight = relation_weight
        self.relation_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def forward(
        self,
        order_logits: torch.Tensor,  # [batch, N, N]
        relation_logits: torch.Tensor,  # [batch, N, N, num_relations]
        order_labels: torch.Tensor,  # [batch, N] reading order indices (0, 1, 2, ...)
        relation_labels: torch.Tensor = None,  # [batch, N, N]
        mask: torch.Tensor = None,  # [batch, N]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            order_logits: [batch, N, N] pairwise order scores
            relation_logits: [batch, N, N, num_relations]
            order_labels: [batch, N] reading order positions (0, 1, 2, ...)
            relation_labels: [batch, N, N] ground truth relation types
            mask: [batch, N] valid region mask

        Returns:
            Dict with order_loss, relation_loss, loss
        """
        batch_size, num_regions = order_labels.shape
        device = order_logits.device

        if mask is None:
            mask = torch.ones(batch_size, num_regions, dtype=torch.bool, device=device)

        # ============ Pairwise Order Loss (BCE) ============
        # Target: 1 if order[i] < order[j], else 0
        order_i = order_labels.unsqueeze(2)  # [B, N, 1]
        order_j = order_labels.unsqueeze(1)  # [B, 1, N]
        targets = (order_i < order_j).float()  # [B, N, N]

        # Valid pair mask
        valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
        diag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=device).unsqueeze(0)
        valid_mask = valid_mask & diag_mask

        # BCE loss
        order_loss = F.binary_cross_entropy_with_logits(
            order_logits, targets, reduction='none'
        )
        order_loss = (order_loss * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)

        # ============ Relation Loss ============
        relation_loss = torch.tensor(0.0, device=device)
        if relation_labels is not None and self.relation_weight > 0:
            relation_logits_flat = relation_logits.view(-1, relation_logits.size(-1))
            relation_labels_flat = relation_labels.view(-1)

            relation_loss_flat = self.relation_criterion(relation_logits_flat, relation_labels_flat)
            relation_loss_flat = relation_loss_flat * valid_mask.view(-1).float()
            relation_loss = relation_loss_flat.sum() / valid_mask.sum().clamp(min=1)

        # ============ Total Loss ============
        total_loss = self.order_weight * order_loss + self.relation_weight * relation_loss

        return {
            'order_loss': order_loss,
            'relation_loss': relation_loss,
            'loss': total_loss,
        }


class FeatureBasedOrderModule(nn.Module):
    """Order Module for feature-based input (used by DOCModel).

    This module is designed for training with pre-extracted region features
    (like OrderModuleFromFeatures), but can optionally incorporate visual
    and text features.

    Unlike the full OrderModule (designed for Detect->Order pipeline),
    this module takes:
    - bbox: Region bounding boxes
    - categories: Region category IDs
    - region_mask: Valid region mask
    - visual_features: Optional visual features
    - text_features: Optional text features
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 5,
        num_heads: int = 12,
        num_layers: int = 3,
        ffn_dim: int = 2048,
        proj_size: int = 2048,
        mlp_hidden: int = 1024,
        num_relations: int = 3,
        dropout: float = 0.1,
        use_spatial: bool = True,
        use_visual: bool = False,
        use_text: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_visual = use_visual
        self.use_text = use_text

        # Category/type embedding
        self.type_embedding = RegionTypeEmbedding(
            num_categories=num_categories,
            hidden_size=hidden_size,
        )

        # Positional embedding (2D from bbox)
        self.pos_embedding = PositionalEmbedding2D(
            hidden_size=hidden_size,
            use_learned=True,
        )

        # Optional feature projections
        if use_visual:
            self.visual_proj = nn.Linear(hidden_size, hidden_size)
        if use_text:
            self.text_proj = nn.Linear(hidden_size, hidden_size)

        # Combine all features into final representation
        # Input size depends on which features are used
        combine_input_size = hidden_size * 2  # type_emb + pos_emb
        if use_visual:
            combine_input_size += hidden_size
        if use_text:
            combine_input_size += hidden_size

        self.combine = nn.Linear(combine_input_size, hidden_size)
        self.combine_norm = nn.LayerNorm(hidden_size)

        # Transformer encoder (3 layers)
        self.transformer = OrderTransformerEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # Order prediction head
        self.order_head = InterRegionOrderHead(
            hidden_size=hidden_size,
            proj_size=proj_size,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            use_spatial=use_spatial,
        )

        # Relation type classification head
        self.relation_head = RelationTypeHead(
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_relations=num_relations,
            dropout=dropout,
        )

    def forward(
        self,
        bbox: torch.Tensor,  # [batch, num_regions, 4]
        categories: torch.Tensor,  # [batch, num_regions]
        region_mask: torch.Tensor,  # [batch, num_regions]
        visual_features: torch.Tensor = None,  # [batch, num_regions, hidden_size]
        text_features: torch.Tensor = None,  # [batch, num_regions, hidden_size]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bbox: [batch, num_regions, 4] normalized bounding boxes
            categories: [batch, num_regions] category/type IDs
            region_mask: [batch, num_regions] valid region mask
            visual_features: Optional [batch, num_regions, hidden_size] visual features
            text_features: Optional [batch, num_regions, hidden_size] text features

        Returns:
            Dict with:
                - embeddings: [batch, num_regions, hidden_size] input embeddings
                - enhanced_features: [batch, num_regions, hidden_size]
                - order_logits: [batch, num_regions, num_regions]
                - relation_logits: [batch, num_regions, num_regions, num_relations]
        """
        # Build embeddings from type and position
        type_emb = self.type_embedding(categories)  # [batch, num_regions, hidden_size]
        pos_emb = self.pos_embedding(bbox)  # [batch, num_regions, hidden_size]

        # Collect all feature components
        feature_parts = [type_emb, pos_emb]

        if self.use_visual and visual_features is not None:
            visual_proj = self.visual_proj(visual_features)
            feature_parts.append(visual_proj)

        if self.use_text and text_features is not None:
            text_proj = self.text_proj(text_features)
            feature_parts.append(text_proj)

        # Combine features
        combined = torch.cat(feature_parts, dim=-1)
        embeddings = self.combine(combined)
        embeddings = self.combine_norm(embeddings)

        # Transformer enhancement
        enhanced = self.transformer(embeddings, mask=region_mask)

        # Order prediction
        order_logits = self.order_head(enhanced, bbox=bbox, mask=region_mask)

        # Relation type prediction
        relation_logits = self.relation_head(enhanced, mask=region_mask)

        return {
            'embeddings': embeddings,
            'enhanced_features': enhanced,
            'order_logits': order_logits,
            'relation_logits': relation_logits,
        }


class DOCModel(nn.Module):
    """Complete Detect-Order-Construct Model (End-to-End)

    This model implements the full DOC pipeline:
    1. Semantic Classification (4.2): Predict region categories
    2. Order (4.3): Predict reading order
    3. Construct (4.4): Predict hierarchical structure

    End-to-end training enables joint optimization of all stages.
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
        use_semantic: bool = False,  # NEW: Enable 4.2 semantic classification
        semantic_num_layers: int = 1,
        cls_weight: float = 1.0,  # Weight for classification loss
        order_weight: float = 1.0,  # Weight for order loss
        construct_weight: float = 1.0,  # Weight for construct loss
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
            use_semantic: Whether to use Semantic Classification module (4.2)
            semantic_num_layers: Number of Transformer layers in Semantic module
            cls_weight: Weight for classification loss
            order_weight: Weight for order loss
            construct_weight: Weight for construct loss
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_categories = num_categories
        self.use_construct = use_construct
        self.use_semantic = use_semantic
        self.cls_weight = cls_weight
        self.order_weight = order_weight
        self.construct_weight = construct_weight

        # ==================== 4.2 Semantic Classification (optional) ====================
        if use_semantic:
            self.semantic_module = SemanticClassificationModule(
                hidden_size=hidden_size,
                num_categories=num_categories,
                num_heads=num_heads,
                num_layers=semantic_num_layers,
                dropout=dropout,
                use_transformer=True,
            )
            self.semantic_loss_fn = SemanticClassificationLoss(ignore_index=0)

        # ==================== 4.3 Order Module ====================
        self.order_module = FeatureBasedOrderModule(
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

        # ==================== 4.4 Construct Module (optional) ====================
        if use_construct:
            self.construct_module = ConstructModule(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=construct_num_layers,
                dropout=dropout,
            )

        # Loss functions (论文4.2.3: Softmax CE for order prediction)
        self.order_loss_fn = OrderLoss()
        if use_construct:
            self.construct_loss_fn = ConstructLoss()

    def forward(
        self,
        bbox: torch.Tensor,                    # [batch, num_regions, 4]
        categories: torch.Tensor = None,       # [batch, num_regions] GT categories (optional if use_semantic)
        region_mask: torch.Tensor = None,      # [batch, num_regions]
        visual_features: torch.Tensor = None,  # [batch, num_regions, visual_dim]
        text_features: torch.Tensor = None,    # [batch, num_regions, text_dim]
        reading_orders: torch.Tensor = None,   # [batch, num_regions] GT reading order positions
        successor_labels: torch.Tensor = None, # [batch, num_regions] GT successor indices (论文4.2.3格式)
        relation_labels: torch.Tensor = None,  # [batch, num_regions, num_regions]
        parent_labels: torch.Tensor = None,    # [batch, num_regions] GT parent indices
        sibling_labels: torch.Tensor = None,   # [batch, num_regions, num_regions]
        category_labels: torch.Tensor = None,  # [batch, num_regions] GT categories for 4.2 loss
        use_predicted_categories: bool = False,  # Use predicted categories instead of GT
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through DOC model (End-to-End)

        Args:
            bbox: Bounding boxes [batch, num_regions, 4]
            categories: Region categories [batch, num_regions] (used if not use_semantic)
            region_mask: Valid region mask [batch, num_regions]
            visual_features: Optional visual features
            text_features: Optional text features
            reading_orders: Ground truth reading orders (for training)
            relation_labels: Ground truth relation labels (for training)
            parent_labels: Ground truth parent indices (for training)
            sibling_labels: Ground truth sibling matrix (for training)
            category_labels: Ground truth categories for semantic classification loss
            use_predicted_categories: If True, use predicted categories for Order module

        Returns:
            Dict containing:
                - category_logits: [batch, N, num_categories] (if use_semantic)
                - order_logits: [batch, num_regions, num_regions]
                - relation_logits: [batch, num_regions, num_regions, num_relations]
                - enhanced_features: [batch, num_regions, hidden_size]
                - parent_logits: [batch, num_regions, num_regions] (if use_construct)
                - sibling_logits: [batch, num_regions, num_regions] (if use_construct)
                - loss: Total loss (if training labels provided)
                - cls_loss: Semantic classification loss (if use_semantic)
                - order_loss: Order module loss
                - construct_loss: Construct module loss (if use_construct)
        """
        device = bbox.device
        batch_size = bbox.shape[0]
        outputs = {}

        # Default region mask
        if region_mask is None:
            region_mask = torch.ones(batch_size, bbox.shape[1], dtype=torch.bool, device=device)

        # ==================== 4.2 Semantic Classification Stage ====================
        cls_loss = torch.tensor(0.0, device=device)

        if self.use_semantic:
            semantic_outputs = self.semantic_module(
                bbox=bbox,
                region_mask=region_mask,
                visual_features=visual_features,
            )

            outputs['category_logits'] = semantic_outputs['category_logits']
            outputs['category_features'] = semantic_outputs['category_features']

            # Determine categories to use for Order module
            if use_predicted_categories or categories is None:
                # Use predicted categories (inference mode or end-to-end)
                pred_categories = semantic_outputs['category_logits'].argmax(dim=-1)
                categories_for_order = pred_categories
            else:
                # Use GT categories (teacher forcing during training)
                categories_for_order = categories

            # Compute semantic classification loss
            if category_labels is not None:
                cls_loss = self.semantic_loss_fn(
                    logits=semantic_outputs['category_logits'],
                    labels=category_labels,
                    mask=region_mask,
                )
            elif categories is not None:
                # Use categories as labels if category_labels not provided
                cls_loss = self.semantic_loss_fn(
                    logits=semantic_outputs['category_logits'],
                    labels=categories,
                    mask=region_mask,
                )

            outputs['cls_loss'] = cls_loss
        else:
            # Not using semantic classification, use provided categories
            categories_for_order = categories
            outputs['cls_loss'] = cls_loss

        # ==================== 4.3 Order Stage ====================
        order_outputs = self.order_module(
            bbox=bbox,
            categories=categories_for_order,
            region_mask=region_mask,
            visual_features=visual_features,
            text_features=text_features,
        )

        outputs['embeddings'] = order_outputs['embeddings']
        outputs['enhanced_features'] = order_outputs['enhanced_features']
        outputs['order_logits'] = order_outputs['order_logits']
        outputs['relation_logits'] = order_outputs['relation_logits']

        # Order loss (论文4.2.3: 使用successor_labels + Softmax CE)
        order_loss = torch.tensor(0.0, device=device)
        if successor_labels is not None:
            order_loss_dict = self.order_loss_fn(
                order_logits=order_outputs['order_logits'],
                relation_logits=order_outputs['relation_logits'],
                order_labels=successor_labels,  # 使用 successor indices
                relation_labels=relation_labels,
                mask=region_mask,
            )
            order_loss = order_loss_dict['loss']
            outputs['order_loss'] = order_loss_dict['order_loss']
            outputs['relation_loss'] = order_loss_dict['relation_loss']

        # ==================== 4.4 Construct Stage ====================
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

            # Construct loss
            if parent_labels is not None:
                construct_loss_dict = self.construct_loss_fn(
                    parent_logits=construct_outputs['parent_logits'],
                    sibling_logits=construct_outputs['sibling_logits'],
                    parent_labels=parent_labels,
                    sibling_labels=sibling_labels,
                    mask=region_mask,
                )
                construct_loss = construct_loss_dict['loss']
                outputs['construct_loss'] = construct_loss
                outputs['parent_loss'] = construct_loss_dict['parent_loss']
                outputs['sibling_loss'] = construct_loss_dict['sibling_loss']

        # ==================== Total Loss (Weighted Combination) ====================
        total_loss = (
            self.cls_weight * cls_loss +
            self.order_weight * order_loss +
            self.construct_weight * construct_loss
        )
        outputs['loss'] = total_loss

        return outputs

    def predict(
        self,
        bbox: torch.Tensor,
        categories: torch.Tensor = None,
        region_mask: torch.Tensor = None,
        visual_features: torch.Tensor = None,
        text_features: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """Inference mode - predict categories, reading order and hierarchy

        If use_semantic=True, categories are predicted automatically.
        Otherwise, categories must be provided.

        Returns:
            Dict with:
                - predicted_categories: [batch, N] (if use_semantic)
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
                use_predicted_categories=self.use_semantic,  # Use predicted if semantic enabled
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

            # Include predicted categories if semantic classification enabled
            if self.use_semantic:
                pred_categories = outputs['category_logits'].argmax(dim=-1)
                results['predicted_categories'] = pred_categories
                results['category_logits'] = outputs['category_logits']

            # Predict hierarchy
            if self.use_construct:
                batch_size = bbox.size(0)
                trees = []
                for b in range(batch_size):
                    tree = build_tree_from_predictions(
                        parent_logits=outputs['parent_logits'][b],
                        mask=region_mask[b] if region_mask is not None else None,
                    )
                    trees.append(tree)
                results['tree_structure'] = trees
                results['parent_logits'] = outputs['parent_logits']
                results['sibling_logits'] = outputs['sibling_logits']

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
    use_semantic: bool = False,
    semantic_num_layers: int = 1,
    cls_weight: float = 1.0,
    order_weight: float = 1.0,
    construct_weight: float = 1.0,
) -> DOCModel:
    """Build DOC model with specified configuration

    Args:
        use_semantic: Enable 4.2 semantic classification for end-to-end training
        semantic_num_layers: Number of Transformer layers in semantic module
        cls_weight: Weight for classification loss
        order_weight: Weight for order loss
        construct_weight: Weight for construct loss
    """
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
        use_semantic=use_semantic,
        semantic_num_layers=semantic_num_layers,
        cls_weight=cls_weight,
        order_weight=order_weight,
        construct_weight=construct_weight,
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
            'num_categories': model.num_categories,
            'use_construct': model.use_construct,
            'use_semantic': model.use_semantic,
            'cls_weight': model.cls_weight,
            'order_weight': model.order_weight,
            'construct_weight': model.construct_weight,
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
    pred_siblings: torch.Tensor = None,  # [batch, N, N] sibling logits (optional)
    true_siblings: torch.Tensor = None,  # [batch, N] sibling indices (optional)
) -> Dict[str, float]:
    """Compute tree structure accuracy

    Returns:
        Dict with parent_accuracy (and sibling_accuracy if provided)
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

    result = {
        'parent_accuracy': parent_acc.item() if isinstance(parent_acc, torch.Tensor) else parent_acc,
    }

    # Sibling accuracy (if provided)
    if pred_siblings is not None and true_siblings is not None:
        pred_sibling_indices = pred_siblings.argmax(dim=-1)  # [batch, N]
        has_sibling = (true_siblings >= 0) & mask
        if has_sibling.any():
            correct = (pred_sibling_indices == true_siblings) & has_sibling
            sibling_acc = correct.sum().float() / has_sibling.sum().float()
        else:
            sibling_acc = 0.0
        result['sibling_accuracy'] = sibling_acc.item() if isinstance(sibling_acc, torch.Tensor) else sibling_acc

    return result

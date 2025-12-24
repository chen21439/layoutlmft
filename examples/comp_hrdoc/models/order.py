"""Order Module for DOC Model (Section 4.3)

Based on "Detect-Order-Construct" paper.
Implements inter-region reading order prediction.

Key components:
- 4.3.1: Multi-modal Feature Extraction (Attention Fusion for text regions)
- 4.3.2: Multi-modal Feature Enhancement (3-layer Transformer)
- 4.3.3: Inter-region Reading Order Prediction Head
- 4.3.4: Relation Type Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .embeddings import (
    PositionalEmbedding2D,
    RegionTypeEmbedding,
)
from .intra_region import SpatialCompatibilityFeatures


# =============================================================================
# 4.3.1: Text Region Attention Fusion (Eq. 10-12)
# =============================================================================

class TextRegionAttentionFusion(nn.Module):
    """Attention-based fusion of text line features into region features.

    Based on paper Eq. (10), (11), (12):
        α_j = FC1(tanh(FC2(F_t_j)))           # Eq. 10
        w_j = softmax(α_j)                     # Eq. 11
        U_region = Σ w_j * F_t_j              # Eq. 12

    where:
        - FC1: 1 node (outputs attention score)
        - FC2: 1024 nodes (projects features)
        - F_t_j: text line features from Detect module
    """

    def __init__(
        self,
        hidden_size: int = 768,
        attention_hidden: int = 1024,  # FC2 has 1024 nodes per paper
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # FC2: hidden_size -> 1024 (per paper)
        self.fc2 = nn.Linear(hidden_size, attention_hidden)
        # FC1: 1024 -> 1 (per paper)
        self.fc1 = nn.Linear(attention_hidden, 1)

    def forward(
        self,
        line_features: torch.Tensor,  # [batch, num_lines, hidden_size]
        regions: List[List[List[int]]],  # [batch][num_regions][line_indices]
        line_mask: torch.Tensor = None,  # [batch, num_lines]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse text line features into region features using attention.

        Args:
            line_features: [batch, num_lines, hidden_size] from Detect module
            regions: List of regions per batch, each region is a list of line indices
            line_mask: [batch, num_lines] valid line mask

        Returns:
            region_features: [batch, max_regions, hidden_size]
            region_mask: [batch, max_regions] valid region mask
        """
        batch_size = line_features.size(0)
        device = line_features.device

        # Find max number of regions across batch
        max_regions = max(len(r) for r in regions) if regions else 1

        # Initialize output tensors
        region_features = torch.zeros(
            batch_size, max_regions, self.hidden_size, device=device
        )
        region_mask = torch.zeros(
            batch_size, max_regions, dtype=torch.bool, device=device
        )

        # Process each batch
        for b in range(batch_size):
            batch_regions = regions[b]

            for r_idx, line_indices in enumerate(batch_regions):
                if len(line_indices) == 0:
                    continue

                # Get line features for this region
                line_idx_tensor = torch.tensor(line_indices, device=device, dtype=torch.long)
                region_line_features = line_features[b, line_idx_tensor]  # [num_lines_in_region, H]

                # Check line mask if provided
                if line_mask is not None:
                    valid_lines = line_mask[b, line_idx_tensor]
                    if not valid_lines.any():
                        continue
                    # Only use valid lines
                    region_line_features = region_line_features[valid_lines]

                if region_line_features.size(0) == 0:
                    continue

                # Compute attention scores (Eq. 10)
                # α = FC1(tanh(FC2(F_t)))
                projected = self.fc2(region_line_features)  # [num_lines, 1024]
                alpha = self.fc1(torch.tanh(projected))  # [num_lines, 1]

                # Compute attention weights (Eq. 11)
                weights = F.softmax(alpha, dim=0)  # [num_lines, 1]

                # Weighted sum (Eq. 12)
                fused = (region_line_features * weights).sum(dim=0)  # [H]

                region_features[b, r_idx] = fused
                region_mask[b, r_idx] = True

        return region_features, region_mask


class RegionFeatureBuilder(nn.Module):
    """Build final region representations by combining multi-modal features.

    Based on paper Eq. (13) and (14):
        R = LN(ReLU(FC(Embedding(r))))        # Eq. 13 (region type embedding)
        U_hat = FC(concat(U, R))              # Eq. 14 (final representation)

    For text regions: U comes from attention fusion (Eq. 12)
    For graphical objects: U comes from visual + positional features
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 10,  # Number of logical role categories
        attention_hidden: int = 1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Attention fusion for text regions (Eq. 10-12)
        self.attention_fusion = TextRegionAttentionFusion(
            hidden_size=hidden_size,
            attention_hidden=attention_hidden,
        )

        # Region type embedding (Eq. 13)
        self.type_embedding = RegionTypeEmbedding(
            num_categories=num_categories,
            hidden_size=hidden_size,
        )

        # 2D positional embedding for graphical objects
        self.pos_embedding = PositionalEmbedding2D(
            hidden_size=hidden_size,
            use_learned=True,
        )

        # Final combination (Eq. 14): FC(concat(U, R))
        self.combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        line_features: torch.Tensor,  # [batch, num_lines, hidden_size] from Detect
        regions: List[List[List[int]]],  # [batch][num_regions][line_indices]
        region_roles: List[List[int]],  # [batch][num_regions] logical roles
        region_bboxes: torch.Tensor,  # [batch, max_regions, 4]
        line_mask: torch.Tensor = None,
        graphical_features: torch.Tensor = None,  # [batch, num_graphical, hidden_size]
        graphical_bboxes: torch.Tensor = None,  # [batch, num_graphical, 4]
        graphical_roles: torch.Tensor = None,  # [batch, num_graphical]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build region features from text line features and graphical objects.

        Args:
            line_features: [batch, num_lines, hidden_size] from Detect module
            regions: Text region groupings from Detect module
            region_roles: Logical roles for each text region
            region_bboxes: [batch, max_regions, 4] bounding boxes
            line_mask: [batch, num_lines] valid line mask
            graphical_features: Optional graphical object features
            graphical_bboxes: Optional graphical object bounding boxes
            graphical_roles: Optional graphical object roles

        Returns:
            all_features: [batch, num_objects, hidden_size] all page object features
            all_bboxes: [batch, num_objects, 4] all bounding boxes
            all_mask: [batch, num_objects] valid mask
        """
        batch_size = line_features.size(0)
        device = line_features.device

        # 1. Fuse text line features into region features (Eq. 10-12)
        text_region_features, text_region_mask = self.attention_fusion(
            line_features, regions, line_mask
        )
        num_text_regions = text_region_features.size(1)

        # 2. Get region type embeddings (Eq. 13)
        # Convert region_roles list to tensor
        max_regions = num_text_regions
        role_tensor = torch.zeros(batch_size, max_regions, dtype=torch.long, device=device)
        for b in range(batch_size):
            for r_idx, role in enumerate(region_roles[b]):
                if r_idx < max_regions:
                    role_tensor[b, r_idx] = role

        type_emb = self.type_embedding(role_tensor)  # [B, num_regions, H]

        # 3. Combine features (Eq. 14): U_hat = FC(concat(U, R))
        combined = torch.cat([text_region_features, type_emb], dim=-1)  # [B, N, 2H]
        text_region_features = self.combine(combined)  # [B, N, H]

        # 4. Handle graphical objects if provided
        if graphical_features is not None and graphical_bboxes is not None:
            num_graphical = graphical_features.size(1)

            # Get positional embedding for graphical objects
            pos_emb = self.pos_embedding(graphical_bboxes)

            # Get type embedding for graphical objects
            if graphical_roles is not None:
                graph_type_emb = self.type_embedding(graphical_roles)
            else:
                graph_type_emb = torch.zeros_like(graphical_features)

            # Combine graphical features
            graph_combined = torch.cat([graphical_features + pos_emb, graph_type_emb], dim=-1)
            graphical_features = self.combine(graph_combined)

            # Concatenate text regions and graphical objects
            all_features = torch.cat([text_region_features, graphical_features], dim=1)
            all_bboxes = torch.cat([region_bboxes[:, :num_text_regions], graphical_bboxes], dim=1)

            graphical_mask = torch.ones(batch_size, num_graphical, dtype=torch.bool, device=device)
            all_mask = torch.cat([text_region_mask, graphical_mask], dim=1)
        else:
            all_features = text_region_features
            all_bboxes = region_bboxes[:, :num_text_regions]
            all_mask = text_region_mask

        return all_features, all_bboxes, all_mask


# =============================================================================
# 4.3.2: Transformer Encoder (3-layer)
# =============================================================================

class OrderTransformerEncoder(nn.Module):
    """3-layer Transformer Encoder for Order Module.

    Based on paper Section 4.3.2:
    "we utilize a three-layer Transformer encoder"

    Enhances page object representations via self-attention.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 3,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
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
            src_key_padding_mask = ~mask  # True = ignore
        else:
            src_key_padding_mask = None

        return self.encoder(features, src_key_padding_mask=src_key_padding_mask)


# =============================================================================
# 4.3.3: Inter-region Reading Order Prediction Head
# =============================================================================

class InterRegionOrderHead(nn.Module):
    """Inter-region Reading Order Prediction Head.

    Based on paper Section 4.3.3, similar to 4.2.3 but for regions:
    s(i,j) = FC_q(F_i) · FC_k(F_j) + MLP(g_ij)

    FC_q, FC_k have 2048 nodes each.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        proj_size: int = 2048,
        mlp_hidden: int = 1024,
        dropout: float = 0.1,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_spatial = use_spatial

        # FC_q and FC_k: 2048 nodes each
        self.head_proj = nn.Linear(hidden_size, proj_size)
        self.dep_proj = nn.Linear(hidden_size, proj_size)

        # Biaffine weight matrix
        self.biaffine = nn.Parameter(torch.zeros(proj_size, proj_size))
        nn.init.xavier_uniform_(self.biaffine)

        # Spatial compatibility features
        if use_spatial:
            self.spatial_features = SpatialCompatibilityFeatures(mlp_hidden_size=mlp_hidden)

        self.scale = proj_size ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,  # [batch, num_regions, hidden_size]
        bbox: torch.Tensor = None,  # [batch, num_regions, 4]
        mask: torch.Tensor = None,  # [batch, num_regions]
    ) -> torch.Tensor:
        """
        Returns:
            [batch, num_regions, num_regions] order logits
        """
        batch_size, num_regions, _ = features.shape
        device = features.device

        # FC_q and FC_k projections
        head_repr = self.dropout(self.head_proj(features))  # [B, N, 2048]
        dep_repr = self.dropout(self.dep_proj(features))    # [B, N, 2048]

        # Biaffine scoring
        scores = torch.einsum('bih,hd,bjd->bij', head_repr, self.biaffine, dep_repr)
        scores = scores * self.scale

        # Add spatial scores
        if self.use_spatial and bbox is not None:
            spatial_scores = self.spatial_features(bbox)
            scores = scores + spatial_scores

        # Apply mask
        if mask is not None:
            row_mask = ~mask.unsqueeze(2)
            col_mask = ~mask.unsqueeze(1)
            combined_mask = row_mask | col_mask
            scores = scores.masked_fill(combined_mask, -1e9)

        # Diagonal mask
        diag = torch.eye(num_regions, dtype=torch.bool, device=device)
        scores = scores.masked_fill(diag.unsqueeze(0), -1e9)

        return scores


# =============================================================================
# 4.3.4: Relation Type Classification Head
# =============================================================================

class RelationTypeHead(nn.Module):
    """Relation Type Classification Head.

    Based on paper Section 4.3.4 and Eq. (16):
    p = BiLinear(FC_q(F_i), FC_k(F_j))

    Predicts relationship type between region pairs:
    - 0: No relation
    - 1: Text region reading order (sequential)
    - 2: Graphical region relation (caption/footnote to figure/table)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        proj_size: int = 2048,
        num_relations: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_size = proj_size

        # FC_q and FC_k: 2048 nodes each
        self.head_proj = nn.Linear(hidden_size, proj_size)
        self.tail_proj = nn.Linear(hidden_size, proj_size)

        self.bilinear = nn.Bilinear(proj_size, proj_size, num_relations)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,  # [batch, num_regions, hidden_size]
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Returns:
            [batch, num_regions, num_regions, num_relations] relation logits
        """
        batch_size, num_regions, _ = features.shape

        head = self.dropout(self.head_proj(features))  # [B, N, 2048]
        tail = self.dropout(self.tail_proj(features))  # [B, N, 2048]

        # Expand for pairwise computation
        head = head.unsqueeze(2).expand(-1, -1, num_regions, -1)
        tail = tail.unsqueeze(1).expand(-1, num_regions, -1, -1)

        head = head.reshape(batch_size * num_regions * num_regions, self.proj_size)
        tail = tail.reshape(batch_size * num_regions * num_regions, self.proj_size)

        # Bilinear classification
        relation_logits = self.bilinear(head, tail)
        relation_logits = relation_logits.reshape(batch_size, num_regions, num_regions, -1)

        return relation_logits


# =============================================================================
# Complete Order Module
# =============================================================================

class OrderModule(nn.Module):
    """Complete Order Module (Section 4.3).

    Processes detected page objects (text regions + graphical objects)
    to determine reading sequences.

    Input from Detect Module:
    - line_features: [B, num_lines, H] enhanced text line features
    - regions: List of text regions (line index groupings)
    - region_roles: Logical roles for each region

    Output:
    - order_logits: [B, N, N] pairwise reading order scores
    - relation_logits: [B, N, N, C] relation type predictions
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 10,
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

        # 4.3.1: Region feature builder (attention fusion + type embedding)
        self.feature_builder = RegionFeatureBuilder(
            hidden_size=hidden_size,
            num_categories=num_categories,
            attention_hidden=mlp_hidden,
        )

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
        line_features: torch.Tensor,  # [batch, num_lines, hidden_size]
        regions: List[List[List[int]]],  # [batch][num_regions][line_indices]
        region_roles: List[List[int]],  # [batch][num_regions]
        region_bboxes: torch.Tensor,  # [batch, max_regions, 4]
        line_mask: torch.Tensor = None,
        graphical_features: torch.Tensor = None,
        graphical_bboxes: torch.Tensor = None,
        graphical_roles: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            line_features: [batch, num_lines, hidden_size] from Detect module
            regions: Text region groupings from Detect module
            region_roles: Logical roles for each text region
            region_bboxes: [batch, max_regions, 4] region bounding boxes
            line_mask: [batch, num_lines] valid line mask
            graphical_*: Optional graphical object inputs

        Returns:
            Dict with:
                - region_features: [batch, num_objects, hidden_size]
                - enhanced_features: [batch, num_objects, hidden_size]
                - order_logits: [batch, num_objects, num_objects]
                - relation_logits: [batch, num_objects, num_objects, num_relations]
                - object_mask: [batch, num_objects]
        """
        # 4.3.1: Build region features
        region_features, all_bboxes, object_mask = self.feature_builder(
            line_features=line_features,
            regions=regions,
            region_roles=region_roles,
            region_bboxes=region_bboxes,
            line_mask=line_mask,
            graphical_features=graphical_features,
            graphical_bboxes=graphical_bboxes,
            graphical_roles=graphical_roles,
        )

        # 4.3.2: Transformer enhancement
        enhanced = self.transformer(region_features, mask=object_mask)

        # 4.3.3: Order prediction
        order_logits = self.order_head(enhanced, bbox=all_bboxes, mask=object_mask)

        # 4.3.4: Relation type prediction
        relation_logits = self.relation_head(enhanced, mask=object_mask)

        return {
            'region_features': region_features,
            'enhanced_features': enhanced,
            'order_logits': order_logits,
            'relation_logits': relation_logits,
            'object_mask': object_mask,
            'object_bboxes': all_bboxes,
        }


# =============================================================================
# Order Loss
# =============================================================================

class OrderLoss(nn.Module):
    """Order Module Loss.

    Combines:
    - Reading order loss (softmax cross-entropy, dependency parsing style)
    - Relation type loss (cross-entropy)
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
        order_labels: torch.Tensor,  # [batch, N] successor indices (-1 for last)
        relation_labels: torch.Tensor = None,  # [batch, N, N]
        mask: torch.Tensor = None,  # [batch, N]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            order_logits: [batch, N, N] pairwise order scores
            relation_logits: [batch, N, N, num_relations] relation predictions
            order_labels: [batch, N] ground truth successor index (-1 for no successor)
            relation_labels: [batch, N, N] ground truth relation types
            mask: [batch, N] valid object mask

        Returns:
            Dict with order_loss, relation_loss, loss
        """
        batch_size, num_objects = order_labels.shape
        device = order_logits.device

        if mask is None:
            mask = torch.ones(batch_size, num_objects, dtype=torch.bool, device=device)

        # Order loss (softmax cross-entropy)
        # Handle objects pointing to themselves (last in sequence)
        order_labels_fixed = order_labels.clone()
        last_mask = order_labels == -1
        indices = torch.arange(num_objects, device=device).unsqueeze(0).expand(batch_size, -1)
        order_labels_fixed = torch.where(last_mask, indices, order_labels_fixed)

        # Prepare logits for loss
        logits_for_loss = order_logits.clone()

        # Allow self-pointing for last objects
        for b in range(batch_size):
            for i in range(num_objects):
                if last_mask[b, i] and mask[b, i]:
                    logits_for_loss[b, i, i] = order_logits[b, i].max() + 1

        # Cross-entropy loss
        logits_flat = logits_for_loss.view(-1, num_objects)
        labels_flat = order_labels_fixed.clamp(0, num_objects - 1).view(-1)
        mask_flat = mask.view(-1)

        order_loss_flat = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        order_loss_flat = order_loss_flat * mask_flat.float()
        order_loss = order_loss_flat.sum() / mask_flat.sum().clamp(min=1)

        # Relation loss
        relation_loss = torch.tensor(0.0, device=device)
        if relation_labels is not None and self.relation_weight > 0:
            valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            diag_mask = ~torch.eye(num_objects, dtype=torch.bool, device=device).unsqueeze(0)
            valid_mask = valid_mask & diag_mask

            relation_logits_flat = relation_logits.view(-1, relation_logits.size(-1))
            relation_labels_flat = relation_labels.view(-1)

            relation_loss_flat = self.relation_criterion(relation_logits_flat, relation_labels_flat)
            relation_loss_flat = relation_loss_flat * valid_mask.view(-1).float()
            relation_loss = relation_loss_flat.sum() / valid_mask.sum().clamp(min=1)

        total_loss = self.order_weight * order_loss + self.relation_weight * relation_loss

        return {
            'order_loss': order_loss,
            'relation_loss': relation_loss,
            'loss': total_loss,
        }


# =============================================================================
# Prediction utilities
# =============================================================================

# =============================================================================
# DOCPipeline: Complete 4.2 + 4.3 Integration
# =============================================================================

class DOCPipeline(nn.Module):
    """Complete DOC Pipeline integrating Detect (4.2) and Order (4.3) modules.

    This is the main entry point for the "Detect-Order-Construct" paper's
    reading order prediction system.

    Flow:
        LayoutXLM Output → DetectModule (4.2) → OrderModule (4.3) → Reading Order

    4.2 DetectModule:
        - Intra-region reading order prediction (grouping lines into regions)
        - Logical role classification for each line/region

    4.3 OrderModule:
        - Inter-region reading order prediction
        - Relation type classification between regions
    """

    def __init__(
        self,
        # Input dimension from LayoutXLM
        input_size: int = 768,
        # Shared dimensions
        hidden_size: int = 768,
        proj_size: int = 2048,
        mlp_hidden: int = 1024,
        # DetectModule (4.2) params
        detect_num_heads: int = 12,
        detect_num_layers: int = 1,
        detect_ffn_dim: int = 2048,
        num_roles: int = 10,
        # OrderModule (4.3) params
        order_num_heads: int = 12,
        order_num_layers: int = 3,
        order_ffn_dim: int = 2048,
        num_relations: int = 3,
        # General params
        dropout: float = 0.1,
        use_spatial: bool = True,
    ):
        """
        Args:
            input_size: Feature dimension from LayoutXLM (768)
            hidden_size: Transformer hidden dimension (768)
            proj_size: FC projection size for biaffine (2048)
            mlp_hidden: Spatial MLP hidden dimension (1024)
            detect_num_heads: Heads for Detect Transformer (12)
            detect_num_layers: Layers for Detect Transformer (1)
            detect_ffn_dim: FFN dim for Detect Transformer (2048)
            num_roles: Number of logical role categories
            order_num_heads: Heads for Order Transformer (12)
            order_num_layers: Layers for Order Transformer (3)
            order_ffn_dim: FFN dim for Order Transformer (2048)
            num_relations: Number of relation type categories
            dropout: Dropout rate
            use_spatial: Whether to use spatial features
        """
        super().__init__()
        self.hidden_size = hidden_size

        # 4.2: Detect Module
        from .intra_region import DetectModule
        self.detect = DetectModule(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_heads=detect_num_heads,
            num_layers=detect_num_layers,
            ffn_dim=detect_ffn_dim,
            mlp_hidden=mlp_hidden,
            num_roles=num_roles,
            dropout=dropout,
            use_spatial=use_spatial,
        )

        # 4.3: Order Module
        self.order = OrderModule(
            hidden_size=hidden_size,
            num_categories=num_roles,
            num_heads=order_num_heads,
            num_layers=order_num_layers,
            ffn_dim=order_ffn_dim,
            proj_size=proj_size,
            mlp_hidden=mlp_hidden,
            num_relations=num_relations,
            dropout=dropout,
            use_spatial=use_spatial,
        )

        # Order loss
        self.order_loss_fn = OrderLoss()

    def forward(
        self,
        line_features: torch.Tensor,  # [batch, num_lines, input_size]
        line_bboxes: torch.Tensor,  # [batch, num_lines, 4]
        line_mask: torch.Tensor = None,  # [batch, num_lines]
        # 4.2 labels
        successor_labels: torch.Tensor = None,  # [batch, num_lines]
        role_labels: torch.Tensor = None,  # [batch, num_lines]
        # 4.3 labels
        region_order_labels: torch.Tensor = None,  # [batch, max_regions]
        relation_labels: torch.Tensor = None,  # [batch, max_regions, max_regions]
        # Loss weights
        lambda_detect: float = 1.0,
        lambda_order: float = 1.0,
        lambda_relation: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through complete DOC pipeline.

        Args:
            line_features: [batch, num_lines, input_size] from LayoutXLM
            line_bboxes: [batch, num_lines, 4] line bounding boxes
            line_mask: [batch, num_lines] valid line mask
            successor_labels: [batch, num_lines] GT successor indices for 4.2
            role_labels: [batch, num_lines] GT role labels for 4.2
            region_order_labels: [batch, max_regions] GT region order for 4.3
            relation_labels: [batch, N, N] GT relation types for 4.3
            lambda_detect: Weight for detect module loss
            lambda_order: Weight for order prediction loss
            lambda_relation: Weight for relation classification loss

        Returns:
            Dict containing outputs from both modules and combined loss
        """
        batch_size = line_features.size(0)
        device = line_features.device
        outputs = {}

        # =====================================================================
        # 4.2: Detect Module - Intra-region order + Logical role
        # =====================================================================
        detect_outputs = self.detect(
            line_features=line_features,
            line_bboxes=line_bboxes,
            line_mask=line_mask,
            successor_labels=successor_labels,
            role_labels=role_labels,
        )

        outputs['detect_loss'] = detect_outputs['loss']
        outputs['intra_loss'] = detect_outputs['intra_loss']
        outputs['role_loss'] = detect_outputs['role_loss']
        outputs['successor_logits'] = detect_outputs['successor_logits']
        outputs['role_logits'] = detect_outputs['role_logits']
        outputs['enhanced_features'] = detect_outputs['enhanced_features']

        # Get region predictions from Detect module
        detect_predictions = self.detect.predict(
            line_features=line_features,
            line_bboxes=line_bboxes,
            line_mask=line_mask,
        )

        regions = detect_predictions['regions']  # List[List[List[int]]]
        region_roles = detect_predictions['region_roles']  # List[List[int]]
        outputs['regions'] = regions
        outputs['region_roles'] = region_roles

        # =====================================================================
        # Compute region bounding boxes from line bboxes
        # =====================================================================
        max_regions = max(len(r) for r in regions) if regions else 1
        region_bboxes = torch.zeros(batch_size, max_regions, 4, device=device)

        for b in range(batch_size):
            for r_idx, line_indices in enumerate(regions[b]):
                if len(line_indices) == 0:
                    continue
                # Union of line bboxes
                idx_tensor = torch.tensor(line_indices, device=device)
                region_line_bboxes = line_bboxes[b, idx_tensor]
                region_bboxes[b, r_idx, 0] = region_line_bboxes[:, 0].min()  # x1
                region_bboxes[b, r_idx, 1] = region_line_bboxes[:, 1].min()  # y1
                region_bboxes[b, r_idx, 2] = region_line_bboxes[:, 2].max()  # x2
                region_bboxes[b, r_idx, 3] = region_line_bboxes[:, 3].max()  # y2

        outputs['region_bboxes'] = region_bboxes

        # =====================================================================
        # 4.3: Order Module - Inter-region order + Relation type
        # =====================================================================
        order_outputs = self.order(
            line_features=detect_outputs['enhanced_features'],
            regions=regions,
            region_roles=region_roles,
            region_bboxes=region_bboxes,
            line_mask=line_mask,
        )

        outputs['order_logits'] = order_outputs['order_logits']
        outputs['relation_logits'] = order_outputs['relation_logits']
        outputs['region_features'] = order_outputs['region_features']
        outputs['object_mask'] = order_outputs['object_mask']

        # =====================================================================
        # Compute Order loss if labels provided
        # =====================================================================
        order_loss = torch.tensor(0.0, device=device)
        relation_loss = torch.tensor(0.0, device=device)

        if region_order_labels is not None:
            order_loss_dict = self.order_loss_fn(
                order_logits=order_outputs['order_logits'],
                relation_logits=order_outputs['relation_logits'],
                order_labels=region_order_labels,
                relation_labels=relation_labels,
                mask=order_outputs['object_mask'],
            )
            order_loss = order_loss_dict['order_loss']
            relation_loss = order_loss_dict['relation_loss']

        outputs['order_loss'] = order_loss
        outputs['relation_loss'] = relation_loss

        # =====================================================================
        # Combined loss
        # =====================================================================
        total_loss = (
            lambda_detect * detect_outputs['loss'] +
            lambda_order * order_loss +
            lambda_relation * relation_loss
        )
        outputs['loss'] = total_loss

        return outputs

    def predict(
        self,
        line_features: torch.Tensor,  # [batch, num_lines, input_size]
        line_bboxes: torch.Tensor,  # [batch, num_lines, 4]
        line_mask: torch.Tensor = None,
    ) -> Dict[str, any]:
        """Inference mode - full pipeline prediction.

        Returns:
            Dict with:
                - regions: List of text regions (line groupings)
                - region_roles: Logical role for each region
                - line_roles: Role for each line
                - reading_order: Global reading order indices
                - relation_types: Predicted relation types between regions
        """
        with torch.no_grad():
            outputs = self.forward(
                line_features=line_features,
                line_bboxes=line_bboxes,
                line_mask=line_mask,
            )

        batch_size = line_features.size(0)

        # Get reading order from order logits
        reading_orders = predict_reading_order(
            outputs['order_logits'],
            outputs['object_mask'],
        )

        # Get relation predictions
        relation_preds = outputs['relation_logits'].argmax(dim=-1)

        return {
            # From 4.2 Detect
            'regions': outputs['regions'],
            'region_roles': outputs['region_roles'],
            'line_roles': outputs['role_logits'].argmax(dim=-1),
            'successor_logits': outputs['successor_logits'],
            # From 4.3 Order
            'reading_order': reading_orders,
            'relation_types': relation_preds,
            'order_logits': outputs['order_logits'],
            # Features
            'region_features': outputs['region_features'],
            'region_bboxes': outputs['region_bboxes'],
        }


def predict_reading_order(
    order_logits: torch.Tensor,  # [batch, N, N]
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Predict reading order from pairwise logits.

    Uses greedy decoding based on successor predictions.

    Returns:
        [batch, N] predicted reading order indices (0 = first, 1 = second, ...)
    """
    batch_size, num_regions, _ = order_logits.shape
    device = order_logits.device

    if mask is None:
        mask = torch.ones(batch_size, num_regions, dtype=torch.bool, device=device)

    # Get predicted successor for each object
    logits_masked = order_logits.clone()
    row_mask = ~mask.unsqueeze(2)
    col_mask = ~mask.unsqueeze(1)
    logits_masked[row_mask | col_mask] = float('-inf')

    successors = logits_masked.argmax(dim=-1)  # [B, N]

    # Build reading order by following successor chain
    reading_order = torch.full((batch_size, num_regions), -1, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Find start (object with no predecessor)
        has_pred = torch.zeros(num_regions, dtype=torch.bool, device=device)
        for i in range(num_regions):
            if mask[b, i]:
                succ = successors[b, i].item()
                if succ != i and 0 <= succ < num_regions:
                    has_pred[succ] = True

        # Find starting objects
        starts = []
        for i in range(num_regions):
            if mask[b, i] and not has_pred[i]:
                starts.append(i)

        # Follow chains
        order_idx = 0
        visited = set()
        for start in starts:
            curr = start
            while curr not in visited and mask[b, curr]:
                visited.add(curr)
                reading_order[b, curr] = order_idx
                order_idx += 1

                succ = successors[b, curr].item()
                if succ == curr or not (0 <= succ < num_regions) or not mask[b, succ]:
                    break
                curr = succ

    return reading_order

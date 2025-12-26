"""Intra-region Reading Order and Logical Role Classification

Based on Section 4.2.3 and 4.2.4 of "Detect-Order-Construct" paper.

4.2.3: Intra-region Reading Order Relation Prediction Head
- Predicts successor relationships between text lines within the same region
- Uses dependency parsing framework with softmax cross-entropy loss
- Includes spatial compatibility features (18-dim) for pairwise scoring

4.2.4: Logical Role Classification Head
- Predicts logical role label for each text line
- Uses plurality voting to determine region's logical role
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# =============================================================================
# 1. Feature Projection Layer (768 -> 1024)
# =============================================================================

class FeatureProjection(nn.Module):
    """Feature projection layer for optional dimension adjustment.

    Note: Per paper Section 4.2.2, the Transformer hidden_size is 768.
    The 1024 dimension mentioned in 4.2.1 refers to FC/MLP intermediate layers.

    This projection is optional - if input_size == output_size, it acts as
    a simple LayerNorm + Linear transformation for feature refinement.
    """

    def __init__(
        self,
        input_size: int = 768,
        output_size: int = 768,  # Keep 768 to match paper's Transformer design
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        if input_size != output_size:
            self.proj = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(),
                nn.LayerNorm(output_size),
            )
        else:
            # Simple refinement when dimensions match
            self.proj = nn.Sequential(
                nn.LayerNorm(input_size),
                nn.Linear(input_size, output_size),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_lines, input_size]
        Returns:
            [batch, num_lines, output_size]
        """
        return self.proj(x)


# =============================================================================
# 2. Spatial Compatibility Features (Paper Eq. 8, 9)
# =============================================================================

class SpatialCompatibilityFeatures(nn.Module):
    """Spatial Compatibility Features for pairwise line scoring.

    Based on paper Eq. (8) and (9):
    g_ij = concat([delta(b_i, b_j), delta(b_i, b_u), delta(b_j, b_u)])

    where:
    - b_i, b_j: bounding boxes of line i and j
    - b_u: union bounding box of b_i and b_j
    - delta: box delta function (6-dim each)

    Each delta contains:
    - (cx_j - cx_i) / w_ref
    - (cy_j - cy_i) / h_ref
    - (w_j - w_i) / w_ref
    - (h_j - h_i) / h_ref
    - log(w_j / w_i)
    - log(h_j / h_i)

    Total: 18 dimensions

    Paper: MLP consists of 2 fully-connected layers with 1024 nodes and 1 node.
    """

    def __init__(self, mlp_hidden_size: int = 1024):
        super().__init__()
        # 18-dim spatial features -> 1024 -> 1 (per paper)
        self.mlp = nn.Sequential(
            nn.Linear(18, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1),
        )

    def _compute_box_delta(
        self,
        box_a: torch.Tensor,  # [B, N, M, 4] or broadcastable
        box_b: torch.Tensor,  # [B, N, M, 4] or broadcastable
        ref_box: torch.Tensor,  # [B, N, M, 4] reference for normalization
    ) -> torch.Tensor:
        """Compute 6-dim box delta between box_a and box_b.

        Args:
            box_a: source box [x1, y1, x2, y2]
            box_b: target box [x1, y1, x2, y2]
            ref_box: reference box for normalization

        Returns:
            [B, N, M, 6] delta features
        """
        # Extract coordinates
        x1_a, y1_a, x2_a, y2_a = box_a[..., 0], box_a[..., 1], box_a[..., 2], box_a[..., 3]
        x1_b, y1_b, x2_b, y2_b = box_b[..., 0], box_b[..., 1], box_b[..., 2], box_b[..., 3]
        x1_r, y1_r, x2_r, y2_r = ref_box[..., 0], ref_box[..., 1], ref_box[..., 2], ref_box[..., 3]

        # Compute centers and sizes
        cx_a, cy_a = (x1_a + x2_a) / 2, (y1_a + y2_a) / 2
        cx_b, cy_b = (x1_b + x2_b) / 2, (y1_b + y2_b) / 2

        w_a, h_a = (x2_a - x1_a).clamp(min=1), (y2_a - y1_a).clamp(min=1)
        w_b, h_b = (x2_b - x1_b).clamp(min=1), (y2_b - y1_b).clamp(min=1)
        w_r, h_r = (x2_r - x1_r).clamp(min=1), (y2_r - y1_r).clamp(min=1)

        # Compute delta (6 dimensions)
        delta = torch.stack([
            (cx_b - cx_a) / w_r,           # normalized x offset
            (cy_b - cy_a) / h_r,           # normalized y offset
            (w_b - w_a) / w_r,             # normalized width diff
            (h_b - h_a) / h_r,             # normalized height diff
            torch.log(w_b / w_a + 1e-6),   # log width ratio
            torch.log(h_b / h_a + 1e-6),   # log height ratio
        ], dim=-1)

        return delta

    def forward(
        self,
        line_bboxes: torch.Tensor,  # [batch, num_lines, 4]
    ) -> torch.Tensor:
        """Compute spatial compatibility scores for all line pairs.

        Args:
            line_bboxes: [batch, num_lines, 4] bounding boxes [x1, y1, x2, y2]

        Returns:
            [batch, num_lines, num_lines] spatial scores
        """
        batch_size, num_lines, _ = line_bboxes.shape
        device = line_bboxes.device

        # Expand for pairwise computation
        # box_i: [B, N, 1, 4] -> [B, N, N, 4]
        # box_j: [B, 1, N, 4] -> [B, N, N, 4]
        box_i = line_bboxes.unsqueeze(2).expand(-1, -1, num_lines, -1)
        box_j = line_bboxes.unsqueeze(1).expand(-1, num_lines, -1, -1)

        # Compute union bounding box
        x1_u = torch.min(box_i[..., 0], box_j[..., 0])
        y1_u = torch.min(box_i[..., 1], box_j[..., 1])
        x2_u = torch.max(box_i[..., 2], box_j[..., 2])
        y2_u = torch.max(box_i[..., 3], box_j[..., 3])
        box_u = torch.stack([x1_u, y1_u, x2_u, y2_u], dim=-1)  # [B, N, N, 4]

        # Compute three delta vectors (Eq. 8)
        delta_ij = self._compute_box_delta(box_i, box_j, box_u)  # [B, N, N, 6]
        delta_iu = self._compute_box_delta(box_i, box_u, box_u)  # [B, N, N, 6]
        delta_ju = self._compute_box_delta(box_j, box_u, box_u)  # [B, N, N, 6]

        # Concatenate to get 18-dim spatial features (Eq. 8)
        g_ij = torch.cat([delta_ij, delta_iu, delta_ju], dim=-1)  # [B, N, N, 18]

        # Pass through MLP to get scalar scores
        spatial_scores = self.mlp(g_ij).squeeze(-1)  # [B, N, N]

        return spatial_scores


# =============================================================================
# 3. Intra-region Reading Order Head (Paper Section 4.2.3)
# =============================================================================

class IntraRegionHead(nn.Module):
    """Intra-region Reading Order Relation Prediction Head

    Based on paper Eq. (6) and (7):
    s(i,j) = FC_q(F_i) · FC_k(F_j) + MLP(g_ij)

    where:
    - FC_q, FC_k: fully-connected layers with 2048 nodes
    - g_ij: 18-dim spatial compatibility features
    - MLP: 2-layer MLP with 1024 and 1 nodes

    Paper Section 4.2.2 specifies:
    - Transformer: 1-layer, 12 heads, 768 hidden dim, 2048 FFN dim

    Uses softmax cross-entropy loss (dependency parsing style).
    """

    def __init__(
        self,
        hidden_size: int = 768,   # Paper 4.2.2: 768
        proj_size: int = 2048,    # FC_q, FC_k dimension (per paper)
        num_heads: int = 12,      # Paper 4.2.2: 12
        num_layers: int = 1,      # Paper 4.2.2: 1
        ffn_dim: int = 2048,      # Paper 4.2.2: 2048
        mlp_hidden: int = 1024,   # Paper: MLP with 1024 nodes
        dropout: float = 0.1,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_spatial = use_spatial

        # 4.2.2: Lightweight Transformer encoder for feature enhancement
        # "1-layer Transformer encoder, head number=12, hidden dim=768, FFN dim=2048"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ===== Succeeding Head (forward direction) =====
        # Biaffine scoring (paper Eq. 7): predicts which line is the successor
        # FC_q and FC_k: 2048 nodes each
        self.succ_head_proj = nn.Linear(hidden_size, proj_size)  # FC_q for successor
        self.succ_dep_proj = nn.Linear(hidden_size, proj_size)   # FC_k for successor
        self.succ_biaffine = nn.Parameter(torch.zeros(proj_size, proj_size))
        nn.init.xavier_uniform_(self.succ_biaffine)

        # ===== Preceding Head (backward direction) =====
        # Additional head per paper: "we employ an additional relation prediction head
        # to further identify the preceding text-line for each text-line"
        self.pred_head_proj = nn.Linear(hidden_size, proj_size)  # FC_q for predecessor
        self.pred_dep_proj = nn.Linear(hidden_size, proj_size)   # FC_k for predecessor
        self.pred_biaffine = nn.Parameter(torch.zeros(proj_size, proj_size))
        nn.init.xavier_uniform_(self.pred_biaffine)

        # Spatial compatibility features (paper Eq. 8, 9)
        # MLP with 1024 hidden nodes - shared between both heads
        if use_spatial:
            self.spatial_features = SpatialCompatibilityFeatures(mlp_hidden_size=mlp_hidden)

        self.scale = proj_size ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        line_features: torch.Tensor,  # [batch, num_lines, hidden_size]
        line_bboxes: torch.Tensor = None,  # [batch, num_lines, 4]
        line_mask: torch.Tensor = None,  # [batch, num_lines] True for valid lines
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            line_features: [batch, num_lines, hidden_size] line-level features
            line_bboxes: [batch, num_lines, 4] bounding boxes for spatial features
            line_mask: [batch, num_lines] True for valid lines

        Returns:
            Dict with:
                - successor_logits: [batch, num_lines, num_lines]
                  successor_logits[b, i, j] = score that line j is successor of line i
                - predecessor_logits: [batch, num_lines, num_lines]
                  predecessor_logits[b, i, j] = score that line j is predecessor of line i
                - enhanced_features: [batch, num_lines, hidden_size]
        """
        batch_size, num_lines, hidden_size = line_features.shape
        device = line_features.device

        # Create attention mask for transformer
        if line_mask is not None:
            attn_mask = ~line_mask  # True means ignore
        else:
            attn_mask = None

        # 4.2.2: Apply transformer to enhance features
        enhanced = self.transformer(
            line_features,
            src_key_padding_mask=attn_mask,
        )

        # Compute spatial scores (shared by both heads)
        spatial_scores = None
        if self.use_spatial and line_bboxes is not None:
            spatial_scores = self.spatial_features(line_bboxes)  # [B, N, N]

        # ===== Succeeding Head: predict successor for each line =====
        succ_head = self.dropout(self.succ_head_proj(enhanced))  # [B, N, 2048]
        succ_dep = self.dropout(self.succ_dep_proj(enhanced))    # [B, N, 2048]
        succ_scores = torch.einsum('bih,hd,bjd->bij', succ_head, self.succ_biaffine, succ_dep)
        succ_scores = succ_scores * self.scale
        if spatial_scores is not None:
            succ_scores = succ_scores + spatial_scores

        # ===== Preceding Head: predict predecessor for each line =====
        pred_head = self.dropout(self.pred_head_proj(enhanced))  # [B, N, 2048]
        pred_dep = self.dropout(self.pred_dep_proj(enhanced))    # [B, N, 2048]
        pred_scores = torch.einsum('bih,hd,bjd->bij', pred_head, self.pred_biaffine, pred_dep)
        pred_scores = pred_scores * self.scale
        if spatial_scores is not None:
            pred_scores = pred_scores + spatial_scores

        # Mask invalid positions (apply to both heads)
        if line_mask is not None:
            row_mask = ~line_mask.unsqueeze(2)  # [B, N, 1]
            col_mask = ~line_mask.unsqueeze(1)  # [B, 1, N]
            combined_mask = row_mask | col_mask
            succ_scores = succ_scores.masked_fill(combined_mask, -1e4)  # fp16 safe
            pred_scores = pred_scores.masked_fill(combined_mask, -1e4)  # fp16 safe

        # Diagonal mask: line cannot be its own successor/predecessor
        # (unless it's the first/last line in region - handled in loss)
        diag = torch.eye(num_lines, dtype=torch.bool, device=device)
        succ_scores = succ_scores.masked_fill(diag.unsqueeze(0), -1e4)  # fp16 safe
        pred_scores = pred_scores.masked_fill(diag.unsqueeze(0), -1e4)  # fp16 safe

        return {
            'successor_logits': succ_scores,
            'predecessor_logits': pred_scores,
            'enhanced_features': enhanced,
        }


# =============================================================================
# 4. Logical Role Classification Head (Paper Section 4.2.4)
# =============================================================================

class LogicalRoleHead(nn.Module):
    """Logical Role Classification Head

    Based on paper Section 4.2.4:
    - Predicts logical role label for each text line
    - Region's role is determined by plurality voting of its lines

    Logical roles include: paragraph, list/list-item, title, section heading,
    header, footer, footnote, caption, etc.
    """

    def __init__(
        self,
        hidden_size: int = 768,  # Same as Transformer hidden_size
        num_roles: int = 10,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: Input feature dimension
            num_roles: Number of logical role categories
            dropout: Dropout rate
        """
        super().__init__()
        self.num_roles = num_roles

        # Simple classifier per paper
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_roles),
        )

    def forward(
        self,
        line_features: torch.Tensor,  # [batch, num_lines, hidden_size]
    ) -> torch.Tensor:
        """
        Args:
            line_features: [batch, num_lines, hidden_size]

        Returns:
            [batch, num_lines, num_roles] logits for each line
        """
        return self.classifier(line_features)

    @staticmethod
    def aggregate_region_roles(
        line_logits: torch.Tensor,  # [num_lines, num_roles]
        regions: List[List[int]],
    ) -> List[int]:
        """Aggregate line-level predictions to region-level using plurality voting.

        Args:
            line_logits: [num_lines, num_roles] line-level role logits
            regions: List of regions, each is a list of line indices

        Returns:
            List of predicted roles for each region
        """
        line_preds = line_logits.argmax(dim=-1)  # [num_lines]
        region_roles = []

        for region in regions:
            if len(region) == 0:
                region_roles.append(0)
                continue

            # Get predictions for lines in this region
            region_preds = line_preds[region]

            # Plurality voting
            vote_counts = torch.bincount(region_preds, minlength=line_logits.size(-1))
            region_role = vote_counts.argmax().item()
            region_roles.append(region_role)

        return region_roles


class LogicalRoleLoss(nn.Module):
    """Loss function for Logical Role Classification"""

    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            reduction='none',
            ignore_index=ignore_index,
        )

    def forward(
        self,
        role_logits: torch.Tensor,   # [batch, num_lines, num_roles]
        role_labels: torch.Tensor,   # [batch, num_lines]
        line_mask: torch.Tensor = None,  # [batch, num_lines]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            role_logits: [batch, num_lines, num_roles]
            role_labels: [batch, num_lines] ground truth role indices
            line_mask: [batch, num_lines] valid line mask

        Returns:
            Dict with loss value
        """
        batch_size, num_lines, num_roles = role_logits.shape
        device = role_logits.device

        # Flatten for cross entropy
        logits_flat = role_logits.view(-1, num_roles)  # [B*N, C]
        labels_flat = role_labels.view(-1)  # [B*N]

        loss_flat = self.criterion(logits_flat, labels_flat)

        # Apply mask
        if line_mask is not None:
            mask_flat = line_mask.view(-1).float()
            loss_flat = loss_flat * mask_flat
            loss = loss_flat.sum() / mask_flat.sum().clamp(min=1)
        else:
            loss = loss_flat.mean()

        return {'loss': loss}


# =============================================================================
# 5. Intra-region Loss (Paper Section 4.2.3)
# =============================================================================

class IntraRegionLoss(nn.Module):
    """Loss function for Intra-region Reading Order Head (Bidirectional)

    Uses softmax cross-entropy (dependency parsing style).
    Supports both successor and predecessor prediction heads per paper:
    "we employ an additional relation prediction head to further identify
    the preceding text-line for each text-line"
    """

    def __init__(self, pred_weight: float = 1.0):
        """
        Args:
            pred_weight: Weight for predecessor loss (default 1.0 = equal weight)
        """
        super().__init__()
        self.pred_weight = pred_weight

    def _compute_direction_loss(
        self,
        logits: torch.Tensor,  # [batch, num_lines, num_lines]
        labels: torch.Tensor,  # [batch, num_lines]
        line_mask: torch.Tensor,  # [batch, num_lines]
        is_self_pointing: torch.Tensor,  # [batch, num_lines] lines that point to self
    ) -> torch.Tensor:
        """Compute loss for one direction (successor or predecessor)."""
        batch_size, num_lines = labels.shape
        device = logits.device

        # Handle self-pointing lines (first/last in region)
        labels_fixed = labels.clone()
        line_indices = torch.arange(num_lines, device=device).unsqueeze(0).expand(batch_size, -1)
        labels_fixed = torch.where(is_self_pointing, line_indices, labels_fixed)

        # Temporarily unmask diagonal for self-pointing lines
        logits_for_loss = logits.clone()
        for b in range(batch_size):
            for i in range(num_lines):
                if is_self_pointing[b, i] and line_mask[b, i]:
                    logits_for_loss[b, i, i] = logits[b, i].max() + 1

        # Compute cross entropy
        logits_flat = logits_for_loss.view(-1, num_lines)
        labels_flat = labels_fixed.clamp(min=0, max=num_lines-1).view(-1)
        valid_flat = line_mask.view(-1)

        loss_flat = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        loss_flat = loss_flat * valid_flat.float()
        loss = loss_flat.sum() / valid_flat.sum().clamp(min=1)

        return loss

    def forward(
        self,
        successor_logits: torch.Tensor,  # [batch, num_lines, num_lines]
        successor_labels: torch.Tensor,  # [batch, num_lines] index of successor
        predecessor_logits: torch.Tensor = None,  # [batch, num_lines, num_lines]
        predecessor_labels: torch.Tensor = None,  # [batch, num_lines] index of predecessor
        line_mask: torch.Tensor = None,  # [batch, num_lines]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            successor_logits: [batch, num_lines, num_lines] successor prediction scores
            successor_labels: [batch, num_lines] ground truth successor index
                              -1 means pointing to self (last line in region)
            predecessor_logits: [batch, num_lines, num_lines] predecessor prediction scores
            predecessor_labels: [batch, num_lines] ground truth predecessor index
                                -1 means pointing to self (first line in region)
            line_mask: [batch, num_lines] valid line mask

        Returns:
            Dict with loss values
        """
        batch_size, num_lines = successor_labels.shape
        device = successor_logits.device

        if line_mask is None:
            line_mask = torch.ones(batch_size, num_lines, dtype=torch.bool, device=device)

        # Successor loss (last lines point to self)
        last_line_mask = successor_labels == -1
        succ_loss = self._compute_direction_loss(
            successor_logits, successor_labels, line_mask, last_line_mask
        )

        # Predecessor loss (first lines point to self)
        if predecessor_logits is not None and predecessor_labels is not None:
            first_line_mask = predecessor_labels == -1
            pred_loss = self._compute_direction_loss(
                predecessor_logits, predecessor_labels, line_mask, first_line_mask
            )
            total_loss = succ_loss + self.pred_weight * pred_loss
        else:
            pred_loss = torch.tensor(0.0, device=device)
            total_loss = succ_loss

        return {
            'loss': total_loss,
            'succ_loss': succ_loss,
            'pred_loss': pred_loss,
            'num_valid': line_mask.sum(),
        }


# =============================================================================
# 6. Union-Find for Region Grouping
# =============================================================================

class UnionFind:
    """Union-Find data structure for grouping lines into regions"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def predict_successors(
    successor_logits: torch.Tensor,  # [num_lines, num_lines]
    line_mask: torch.Tensor = None,  # [num_lines]
    threshold: float = 0.5,  # Confidence threshold for having a successor
) -> torch.Tensor:
    """Predict successor for each line

    Args:
        successor_logits: [num_lines, num_lines] scores (diagonal is masked to -1e9)
        line_mask: [num_lines] valid lines
        threshold: Confidence threshold. If max softmax prob < threshold, predict -1 (no successor)

    Returns:
        [num_lines] predicted successor index, -1 for no successor or invalid lines
    """
    num_lines = successor_logits.size(0)
    device = successor_logits.device

    if line_mask is None:
        line_mask = torch.ones(num_lines, dtype=torch.bool, device=device)

    # The diagonal is masked to -1e9 in forward(), so we need to handle "no successor" case
    # Strategy: Check confidence via softmax. If max prob < threshold, predict -1

    # Compute softmax probabilities (diagonal will have ~0 prob due to -1e9 masking)
    probs = F.softmax(successor_logits, dim=-1)  # [num_lines, num_lines]

    # Get max probability and argmax for each line
    max_probs, argmax_indices = probs.max(dim=-1)  # [num_lines], [num_lines]

    # If max prob is below threshold, it means no confident successor -> predict -1
    # This handles the "last line in region" case where no other line is a good successor
    has_successor = max_probs >= threshold

    # Build successor predictions
    successors = torch.where(
        has_successor & line_mask,
        argmax_indices,
        torch.tensor(-1, device=device, dtype=argmax_indices.dtype)
    )

    return successors


def predict_successors_bidirectional(
    successor_logits: torch.Tensor,  # [num_lines, num_lines]
    predecessor_logits: torch.Tensor,  # [num_lines, num_lines]
    line_mask: torch.Tensor = None,  # [num_lines]
    threshold: float = 0.5,
    combine_method: str = "agreement",  # "agreement", "average", "successor_only"
) -> torch.Tensor:
    """Predict successors using both forward and backward heads

    Per paper: "The prediction results from both relation prediction heads
    are then combined to obtain the final results."

    Args:
        successor_logits: [num_lines, num_lines] forward head scores
        predecessor_logits: [num_lines, num_lines] backward head scores
        line_mask: [num_lines] valid lines
        threshold: Confidence threshold
        combine_method:
            - "agreement": Only accept if both heads agree (A→B and B←A)
            - "average": Average the forward and transposed backward scores
            - "successor_only": Use only successor head (fallback)

    Returns:
        [num_lines] predicted successor index, -1 for no successor
    """
    num_lines = successor_logits.size(0)
    device = successor_logits.device

    if line_mask is None:
        line_mask = torch.ones(num_lines, dtype=torch.bool, device=device)

    if combine_method == "successor_only" or predecessor_logits is None:
        return predict_successors(successor_logits, line_mask, threshold)

    # Forward: successor_logits[i, j] = score that j is successor of i
    # Backward: predecessor_logits[j, i] = score that i is predecessor of j
    # If A→B, then predecessor_logits[B, A] should be high
    # So we transpose predecessor_logits to get: [i, j] = score that i→j from backward view

    pred_transposed = predecessor_logits.T  # [num_lines, num_lines]

    if combine_method == "average":
        # Average forward and backward scores
        combined_logits = (successor_logits + pred_transposed) / 2
        return predict_successors(combined_logits, line_mask, threshold)

    elif combine_method == "agreement":
        # Get predictions from both heads
        succ_probs = F.softmax(successor_logits, dim=-1)
        pred_probs = F.softmax(pred_transposed, dim=-1)

        # For each line i, find argmax from successor head
        succ_max_probs, succ_preds = succ_probs.max(dim=-1)

        # For each line i, find argmax from predecessor head (transposed)
        pred_max_probs, pred_preds = pred_probs.max(dim=-1)

        # Check agreement: if succ_head says i→j and pred_head also agrees
        # We use succ_preds as base and verify with pred_head
        successors = torch.full((num_lines,), -1, device=device, dtype=torch.long)

        for i in range(num_lines):
            if not line_mask[i]:
                continue

            j = succ_preds[i].item()

            # Check confidence
            if succ_max_probs[i] < threshold:
                continue

            # Check if j is valid
            if j < 0 or j >= num_lines or not line_mask[j]:
                continue

            # Check agreement: pred_head should say j's predecessor is i
            # i.e., argmax of predecessor_logits[j, :] should be i
            j_pred = predecessor_logits[j].argmax().item()
            j_pred_prob = F.softmax(predecessor_logits[j], dim=-1).max()

            if j_pred == i and j_pred_prob >= threshold:
                # Both heads agree: i → j
                successors[i] = j
            elif succ_max_probs[i] >= threshold * 1.5:
                # High confidence from successor head, accept anyway
                successors[i] = j

        return successors

    else:
        return predict_successors(successor_logits, line_mask, threshold)


def group_lines_to_regions(
    successors: torch.Tensor,  # [num_lines] successor index
    line_mask: torch.Tensor = None,  # [num_lines]
) -> List[List[int]]:
    """Group lines into regions using Union-Find

    Args:
        successors: [num_lines] predicted successor for each line
        line_mask: [num_lines] valid lines

    Returns:
        List of regions, each region is a list of line indices
    """
    num_lines = successors.size(0)
    device = successors.device

    if line_mask is None:
        line_mask = torch.ones(num_lines, dtype=torch.bool, device=device)

    # Convert to CPU for Union-Find
    successors_np = successors.cpu().numpy()
    mask_np = line_mask.cpu().numpy()

    # Initialize Union-Find
    uf = UnionFind(num_lines)

    # Union connected lines
    for i in range(num_lines):
        succ = successors_np[i]
        if mask_np[i] and 0 <= succ < num_lines and succ != i:
            if mask_np[succ]:
                uf.union(i, succ)

    # Group by root
    groups = defaultdict(list)
    for i in range(num_lines):
        if mask_np[i]:
            root = uf.find(i)
            groups[root].append(i)

    # Sort each group by line index (preserve reading order)
    regions = []
    for root in sorted(groups.keys()):
        region = sorted(groups[root])
        regions.append(region)

    return regions


# =============================================================================
# 7. Complete Detect Module (4.2.3 + 4.2.4)
# =============================================================================

class DetectModule(nn.Module):
    """Complete Detect Module combining 4.2.3 and 4.2.4

    This module implements:
    - Feature projection (optional, for dimension adjustment)
    - Intra-region reading order prediction (4.2.3)
    - Logical role classification (4.2.4)

    Paper Section 4.2.2 specifies:
    - Transformer: 1-layer, 12 heads, 768 hidden dim, 2048 FFN dim
    - FC_q/FC_k: 2048 nodes
    - Spatial MLP: 1024 hidden nodes

    Input: Line-level features from LayoutXLM (768-dim)
    Output:
        - Successor predictions for grouping lines into regions
        - Logical role predictions for each line/region
    """

    def __init__(
        self,
        input_size: int = 768,     # LayoutXLM output dimension
        hidden_size: int = 768,    # Paper 4.2.2: 768
        proj_size: int = 2048,     # Paper: FC_q/FC_k have 2048 nodes
        num_heads: int = 12,       # Paper 4.2.2: 12
        num_layers: int = 1,       # Paper 4.2.2: 1
        ffn_dim: int = 2048,       # Paper 4.2.2: 2048
        mlp_hidden: int = 1024,    # Paper: spatial MLP with 1024 nodes
        num_roles: int = 10,
        dropout: float = 0.1,
        use_spatial: bool = True,
    ):
        """
        Args:
            input_size: Input feature dimension from LayoutXLM (768)
            hidden_size: Transformer hidden dimension (768 per paper 4.2.2)
            proj_size: FC_q/FC_k projection size (2048 per paper)
            num_heads: Number of attention heads (12 per paper)
            num_layers: Number of transformer layers (1 per paper)
            ffn_dim: Transformer feedforward dimension (2048 per paper)
            mlp_hidden: Spatial MLP hidden dimension (1024 per paper)
            num_roles: Number of logical role categories
            dropout: Dropout rate
            use_spatial: Whether to use spatial compatibility features
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.use_spatial = use_spatial

        # Feature projection (optional, when input_size != hidden_size)
        self.feature_proj = FeatureProjection(input_size, hidden_size)

        # 4.2.3: Intra-region reading order head
        self.intra_head = IntraRegionHead(
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            use_spatial=use_spatial,
        )

        # 4.2.4: Logical role classification head
        self.role_head = LogicalRoleHead(
            hidden_size=hidden_size,
            num_roles=num_roles,
            dropout=dropout,
        )

        # Loss functions
        self.intra_loss_fn = IntraRegionLoss()
        self.role_loss_fn = LogicalRoleLoss()

    @staticmethod
    def _compute_predecessor_labels(
        successor_labels: torch.Tensor,  # [batch, num_lines]
        line_mask: torch.Tensor,  # [batch, num_lines]
    ) -> torch.Tensor:
        """Compute predecessor labels from successor labels.

        If successor_labels[i] = j, then predecessor_labels[j] = i.
        First line in each region has predecessor_labels = -1 (points to self).

        Args:
            successor_labels: [batch, num_lines] index of successor, -1 for last line
            line_mask: [batch, num_lines] valid line mask

        Returns:
            [batch, num_lines] index of predecessor, -1 for first line
        """
        batch_size, num_lines = successor_labels.shape
        device = successor_labels.device

        # Initialize all to -1 (no predecessor / first line in region)
        predecessor_labels = torch.full_like(successor_labels, -1)

        for b in range(batch_size):
            for i in range(num_lines):
                if not line_mask[b, i]:
                    continue
                j = successor_labels[b, i].item()
                # If line i has successor j, then line j has predecessor i
                if 0 <= j < num_lines and j != i and line_mask[b, j]:
                    predecessor_labels[b, j] = i

        return predecessor_labels

    def forward(
        self,
        line_features: torch.Tensor,     # [batch, num_lines, input_size]
        line_bboxes: torch.Tensor = None,  # [batch, num_lines, 4]
        line_mask: torch.Tensor = None,  # [batch, num_lines]
        successor_labels: torch.Tensor = None,  # [batch, num_lines]
        role_labels: torch.Tensor = None,  # [batch, num_lines]
        lambda_intra: float = 1.0,
        lambda_role: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Detect module.

        Args:
            line_features: [batch, num_lines, input_size] from LayoutXLM
            line_bboxes: [batch, num_lines, 4] bounding boxes
            line_mask: [batch, num_lines] valid line mask
            successor_labels: [batch, num_lines] ground truth successor indices
            role_labels: [batch, num_lines] ground truth role labels
            lambda_intra: Weight for intra-region loss
            lambda_role: Weight for role classification loss

        Returns:
            Dict containing:
                - projected_features: [batch, num_lines, hidden_size]
                - enhanced_features: [batch, num_lines, hidden_size]
                - successor_logits: [batch, num_lines, num_lines]
                - role_logits: [batch, num_lines, num_roles]
                - loss: Total loss (if labels provided)
                - intra_loss: Intra-region loss
                - role_loss: Role classification loss
        """
        device = line_features.device
        outputs = {}

        # Feature projection
        projected = self.feature_proj(line_features)  # [B, N, 1024]
        outputs['projected_features'] = projected

        # 4.2.3: Intra-region reading order prediction (bidirectional)
        intra_outputs = self.intra_head(
            line_features=projected,
            line_bboxes=line_bboxes,
            line_mask=line_mask,
        )
        outputs['enhanced_features'] = intra_outputs['enhanced_features']
        outputs['successor_logits'] = intra_outputs['successor_logits']
        outputs['predecessor_logits'] = intra_outputs['predecessor_logits']

        # 4.2.4: Logical role classification
        role_logits = self.role_head(intra_outputs['enhanced_features'])
        outputs['role_logits'] = role_logits

        # Compute losses
        intra_loss = torch.tensor(0.0, device=device)
        role_loss = torch.tensor(0.0, device=device)

        if successor_labels is not None:
            # Compute predecessor_labels from successor_labels
            # If line j is successor of line i, then line i is predecessor of line j
            predecessor_labels = self._compute_predecessor_labels(successor_labels, line_mask)

            intra_loss_dict = self.intra_loss_fn(
                successor_logits=intra_outputs['successor_logits'],
                successor_labels=successor_labels,
                predecessor_logits=intra_outputs['predecessor_logits'],
                predecessor_labels=predecessor_labels,
                line_mask=line_mask,
            )
            intra_loss = intra_loss_dict['loss']
            outputs['succ_loss'] = intra_loss_dict.get('succ_loss', intra_loss)
            outputs['pred_loss'] = intra_loss_dict.get('pred_loss', torch.tensor(0.0))

        if role_labels is not None:
            role_loss_dict = self.role_loss_fn(
                role_logits,
                role_labels,
                line_mask,
            )
            role_loss = role_loss_dict['loss']

        outputs['intra_loss'] = intra_loss
        outputs['role_loss'] = role_loss
        outputs['loss'] = lambda_intra * intra_loss + lambda_role * role_loss

        return outputs

    def predict(
        self,
        line_features: torch.Tensor,  # [batch, num_lines, input_size]
        line_bboxes: torch.Tensor = None,
        line_mask: torch.Tensor = None,
        successor_threshold: float = 0.5,  # Confidence threshold for successor prediction
        use_bidirectional: bool = True,  # Use both heads for prediction
        combine_method: str = "agreement",  # "agreement", "average", "successor_only"
    ) -> Dict[str, any]:
        """Inference mode - predict regions and roles.

        Args:
            line_features: [batch, num_lines, input_size] line-level features
            line_bboxes: [batch, num_lines, 4] bounding boxes
            line_mask: [batch, num_lines] valid line mask
            successor_threshold: Confidence threshold for predicting successors.
                If max softmax prob < threshold, predicts -1 (no successor).
            use_bidirectional: Use both successor and predecessor heads.
            combine_method: How to combine bidirectional predictions.

        Returns:
            Dict with:
                - successors: [batch, num_lines] predicted successor indices (-1 = no successor)
                - regions: List of List of line indices per region
                - line_roles: [batch, num_lines] predicted roles for each line
                - region_roles: List of predicted roles for each region
        """
        with torch.no_grad():
            outputs = self.forward(
                line_features=line_features,
                line_bboxes=line_bboxes,
                line_mask=line_mask,
            )

        batch_size = line_features.size(0)
        results = {
            'successor_logits': outputs['successor_logits'],
            'predecessor_logits': outputs['predecessor_logits'],
            'role_logits': outputs['role_logits'],
        }

        # Process each sample in batch
        all_successors = []
        all_regions = []
        all_line_roles = []
        all_region_roles = []

        for b in range(batch_size):
            succ_logits = outputs['successor_logits'][b]
            pred_logits = outputs['predecessor_logits'][b]
            mask = line_mask[b] if line_mask is not None else None

            # Predict successors using bidirectional heads
            if use_bidirectional:
                successors = predict_successors_bidirectional(
                    succ_logits, pred_logits, mask,
                    threshold=successor_threshold,
                    combine_method=combine_method,
                )
            else:
                successors = predict_successors(succ_logits, mask, threshold=successor_threshold)
            all_successors.append(successors)

            # Group into regions
            regions = group_lines_to_regions(successors, mask)
            all_regions.append(regions)

            # Predict line roles
            line_roles = outputs['role_logits'][b].argmax(dim=-1)
            all_line_roles.append(line_roles)

            # Aggregate to region roles using plurality voting
            region_roles = LogicalRoleHead.aggregate_region_roles(
                outputs['role_logits'][b], regions
            )
            all_region_roles.append(region_roles)

        results['successors'] = torch.stack(all_successors)
        results['regions'] = all_regions
        results['line_roles'] = torch.stack(all_line_roles)
        results['region_roles'] = all_region_roles

        return results


# =============================================================================
# 8. Backward Compatibility: Keep old class names
# =============================================================================

class IntraRegionModule(DetectModule):
    """Alias for DetectModule for backward compatibility"""
    pass


class MultiModalLineEncoder(nn.Module):
    """Multi-modal feature encoder for text lines (kept for compatibility)

    Note: When using LayoutXLM, this is not needed as LayoutXLM already
    provides multi-modal features. This class is kept for cases where
    separate visual/text/position features need to be combined.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        visual_size: int = 256,
        pos_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.visual_proj = nn.Linear(visual_size, hidden_size)
        self.pos_embedding = nn.Sequential(
            nn.Linear(4, pos_size),
            nn.GELU(),
            nn.Linear(pos_size, hidden_size),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor = None,
        line_bboxes: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, num_lines, hidden_size = text_features.shape
        device = text_features.device

        features = [text_features]

        if visual_features is not None:
            features.append(self.visual_proj(visual_features))
        else:
            features.append(torch.zeros_like(text_features))

        if line_bboxes is not None:
            bbox_norm = line_bboxes.float() / 1000.0
            features.append(self.pos_embedding(bbox_norm))
        else:
            features.append(torch.zeros_like(text_features))

        concat = torch.cat(features, dim=-1)
        fused = self.fusion(concat)
        return self.norm(fused)

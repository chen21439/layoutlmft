"""Construct Module for DOC Model

Based on "Detect-Order-Construct" paper.
Generates hierarchical document structure (TOC) from section headings.
Uses Rotary Positional Embedding (RoPE) to incorporate reading order.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .embeddings import RotaryPositionalEmbedding


class RoPETransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer with Rotary Positional Embedding"""

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        # Self-attention with RoPE
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )

        # Layer norms (pre-norm)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            mask: [batch, seq_len] True for valid positions
            positions: [batch, seq_len] position indices for RoPE

        Returns:
            [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # Pre-norm
        h = self.norm1(x)

        # QKV projections
        q = self.q_proj(h).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        if positions is not None:
            # Use provided positions (reading order)
            q = self._apply_rope_with_positions(q, positions)
            k = self._apply_rope_with_positions(k, positions)
        else:
            # Use sequential positions
            q = self.rope(q)
            k = self.rope(k)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask
        if mask is not None:
            # [batch, 1, 1, seq_len]
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        # Handle all-inf rows to avoid NaN in softmax
        all_inf = torch.isinf(attn_weights).all(dim=-1, keepdim=True)
        attn_weights = attn_weights.masked_fill(all_inf, 0.0)

        attn_probs = F.softmax(attn_weights, dim=-1)
        # Zero out attention for all-masked rows
        attn_probs = attn_probs.masked_fill(all_inf, 0.0)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Residual
        x = x + attn_output

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x

    def _apply_rope_with_positions(
        self,
        x: torch.Tensor,  # [batch, heads, seq_len, head_dim]
        positions: torch.Tensor,  # [batch, seq_len]
    ) -> torch.Tensor:
        """Apply RoPE using custom position indices"""
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Get max position for cache
        max_pos = positions.max().item() + 1
        if max_pos > self.rope.max_seq_len:
            self.rope._update_cos_sin_cache(int(max_pos))

        # Gather cos/sin for positions
        cos = self.rope.cos_cached.squeeze(0).squeeze(0)  # [max_len, head_dim]
        sin = self.rope.sin_cached.squeeze(0).squeeze(0)  # [max_len, head_dim]

        # Clamp positions to valid range
        positions = positions.clamp(0, cos.size(0) - 1)

        # Gather: [batch, seq_len] -> [batch, seq_len, head_dim]
        pos_cos = cos[positions]  # [batch, seq_len, head_dim]
        pos_sin = sin[positions]  # [batch, seq_len, head_dim]

        # Expand for heads: [batch, 1, seq_len, head_dim]
        pos_cos = pos_cos.unsqueeze(1)
        pos_sin = pos_sin.unsqueeze(1)

        # Apply rotation
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * pos_cos + rotated * pos_sin


class RoPETransformerEncoder(nn.Module):
    """Transformer Encoder with RoPE for Construct Module"""

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 3,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, positions=positions)
        return self.final_norm(x)


class TreeRelationHead(nn.Module):
    """Tree-aware Relation Prediction Head

    Predicts parent-child and sibling relationships for TOC generation.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Parent prediction: for each node, predict which node is its parent
        self.parent_query = nn.Linear(hidden_size, hidden_size)
        self.parent_key = nn.Linear(hidden_size, hidden_size)
        self.parent_biaffine = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.parent_biaffine)

        # Sibling prediction: predict if two nodes are siblings
        # Paper 公式18-19: f_ij = FC_q(F_Si) ◦ FC_k(F_Sj) - 使用点积而非 Bilinear
        self.sibling_query = nn.Linear(hidden_size, hidden_size)
        self.sibling_key = nn.Linear(hidden_size, hidden_size)

        # Root prediction: predict if a node is root
        self.root_classifier = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_size ** -0.5

    def forward(
        self,
        features: torch.Tensor,  # [batch, num_nodes, hidden_size]
        mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch, num_nodes, hidden_size] node features
            mask: [batch, num_nodes] valid node mask

        Returns:
            Dict with:
                - parent_logits: [batch, num_nodes, num_nodes]
                  parent_logits[i, j] = score that j is parent of i
                - sibling_logits: [batch, num_nodes, num_nodes]
                  sibling_logits[i, j] = score that j is next sibling of i (点积)
                - root_logits: [batch, num_nodes]
        """
        batch_size, num_nodes, hidden_size = features.shape

        # Parent prediction
        parent_q = self.parent_query(features)  # [B, N, H] - child queries
        parent_k = self.parent_key(features)    # [B, N, H] - parent keys

        # parent_logits[i, j] = likelihood that j is parent of i
        parent_logits = torch.einsum(
            'bih,hd,bjd->bij',
            parent_q, self.parent_biaffine, parent_k
        ) * self.scale

        # Sibling prediction - 论文公式18-19: f_ij = FC_q(F_Si) ◦ FC_k(F_Sj)
        # 使用点积代替 Bilinear，避免 backward 时巨大的显存占用
        sib_q = self.sibling_query(features)  # [B, N, H]
        sib_k = self.sibling_key(features)    # [B, N, H]

        # 点积计算: [B, N, H] @ [B, H, N] -> [B, N, N]
        sibling_logits = torch.bmm(sib_q, sib_k.transpose(1, 2)) * self.scale

        # Root prediction
        root_logits = self.root_classifier(features).squeeze(-1)  # [B, N]

        # Apply mask with large negative value instead of -inf for numerical stability
        if mask is not None:
            row_mask = ~mask.unsqueeze(2)
            col_mask = ~mask.unsqueeze(1)
            combined_mask = row_mask | col_mask

            # Use -1e4 instead of -inf to avoid NaN in softmax (fp16 safe)
            parent_logits = parent_logits.masked_fill(combined_mask, -1e4)
            # Diagonal: node cannot be its own parent
            diag = torch.eye(num_nodes, dtype=torch.bool, device=features.device)
            parent_logits = parent_logits.masked_fill(diag.unsqueeze(0), -1e4)

        return {
            'parent_logits': parent_logits,
            'sibling_logits': sibling_logits,
            'root_logits': root_logits,
        }


class ConstructModule(nn.Module):
    """Complete Construct Module

    Generates hierarchical structure from enhanced region features.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 3,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Transformer with RoPE
        self.transformer = RoPETransformerEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # Tree relation prediction
        self.tree_head = TreeRelationHead(
            hidden_size=hidden_size,
            dropout=dropout,
        )

    def forward(
        self,
        features: torch.Tensor,       # [batch, num_regions, hidden_size]
        reading_order: torch.Tensor,  # [batch, num_regions] reading order positions
        mask: torch.Tensor = None,    # [batch, num_regions]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch, num_regions, hidden_size] region features (from Order module)
            reading_order: [batch, num_regions] reading order positions (0, 1, 2, ...)
            mask: [batch, num_regions] valid region mask

        Returns:
            Dict with tree relation predictions
        """
        # Apply Transformer with RoPE using reading order as positions
        enhanced = self.transformer(
            features,
            mask=mask,
            positions=reading_order,
        )

        # Predict tree relations
        tree_outputs = self.tree_head(enhanced, mask=mask)

        return {
            'construct_features': enhanced,
            **tree_outputs,
        }


class ConstructLoss(nn.Module):
    """Construct Module Loss

    Combines:
    - Parent prediction loss (softmax CE)
    - Sibling prediction loss (BCE)
    - Root prediction loss (BCE)
    """

    def __init__(
        self,
        parent_weight: float = 1.0,
        sibling_weight: float = 0.5,
        root_weight: float = 0.3,
    ):
        super().__init__()
        self.parent_weight = parent_weight
        self.sibling_weight = sibling_weight
        self.root_weight = root_weight

    def forward(
        self,
        parent_logits: torch.Tensor,   # [batch, N, N]
        sibling_logits: torch.Tensor,  # [batch, N, N]
        root_logits: torch.Tensor,     # [batch, N]
        parent_labels: torch.Tensor,   # [batch, N] index of parent (-1 for root)
        sibling_labels: torch.Tensor = None,  # [batch, N, N] 1 if siblings
        mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            parent_logits: [batch, N, N] parent prediction scores
            sibling_logits: [batch, N, N] sibling prediction scores (点积)
            root_logits: [batch, N] root prediction scores
            parent_labels: [batch, N] ground truth parent indices (-1 = root)
            sibling_labels: [batch, N, N] ground truth sibling matrix
            mask: [batch, N] valid node mask

        Returns:
            Dict with parent_loss, sibling_loss, root_loss, total_loss
        """
        batch_size, num_nodes = parent_labels.shape
        device = parent_logits.device

        # Valid nodes (not padding)
        if mask is None:
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)

        # Root labels: 1 if parent_label == -1
        root_labels = (parent_labels == -1).float()

        # Non-root mask (nodes that have a parent)
        has_parent = (parent_labels >= 0) & mask

        # Parent loss (softmax cross-entropy for non-root nodes)
        parent_loss = torch.tensor(0.0, device=device)
        # Valid parent: has_parent AND parent_labels < num_nodes
        valid_parent = has_parent & (parent_labels < num_nodes)
        if valid_parent.any():
            # For nodes with valid parents, compute CE loss
            parent_logits_flat = parent_logits.view(-1, num_nodes)
            parent_labels_clamped = parent_labels.clamp(min=0, max=num_nodes-1)
            parent_labels_flat = parent_labels_clamped.view(-1)
            valid_parent_flat = valid_parent.view(-1)

            loss_flat = F.cross_entropy(
                parent_logits_flat,
                parent_labels_flat,
                reduction='none'
            )

            loss_flat = loss_flat * valid_parent_flat.float()
            parent_loss = loss_flat.sum() / valid_parent_flat.sum().clamp(min=1)

        # Root loss (BCE)
        root_loss = F.binary_cross_entropy_with_logits(
            root_logits,
            root_labels,
            reduction='none'
        )
        root_loss = (root_loss * mask.float()).sum() / mask.sum().clamp(min=1)

        # Sibling loss (if labels provided) - 使用 BCE，对应论文的点积实现
        sibling_loss = torch.tensor(0.0, device=device)
        if sibling_labels is not None and self.sibling_weight > 0:
            valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)
            # 使用 BCEWithLogitsLoss，因为现在 sibling_logits 是 [B, N, N] 的点积分数
            loss = F.binary_cross_entropy_with_logits(
                sibling_logits,
                sibling_labels.float(),
                reduction='none'
            )
            loss = loss * valid_pairs.float()
            sibling_loss = loss.sum() / valid_pairs.sum().clamp(min=1)

        # Total loss
        total_loss = (
            self.parent_weight * parent_loss +
            self.sibling_weight * sibling_loss +
            self.root_weight * root_loss
        )

        return {
            'parent_loss': parent_loss,
            'sibling_loss': sibling_loss,
            'root_loss': root_loss,
            'loss': total_loss,
        }


def build_tree_from_predictions(
    parent_logits: torch.Tensor,  # [num_nodes, num_nodes]
    root_logits: torch.Tensor,    # [num_nodes]
    mask: torch.Tensor = None,    # [num_nodes]
) -> List[Dict]:
    """Build tree structure from predictions

    Args:
        parent_logits: [num_nodes, num_nodes]
        root_logits: [num_nodes]
        mask: [num_nodes] valid nodes

    Returns:
        List of dicts with 'id', 'parent_id', 'children' keys
    """
    num_nodes = parent_logits.size(0)

    if mask is None:
        mask = torch.ones(num_nodes, dtype=torch.bool, device=parent_logits.device)

    # Find root (highest root score among valid nodes)
    root_scores = root_logits.clone()
    root_scores[~mask] = float('-inf')
    root_idx = root_scores.argmax().item()

    # For each non-root node, find parent
    nodes = []
    for i in range(num_nodes):
        if not mask[i]:
            continue

        node = {'id': i, 'children': []}

        if i == root_idx:
            node['parent_id'] = -1
        else:
            # Find best parent (exclude self)
            parent_scores = parent_logits[i].clone()
            parent_scores[i] = float('-inf')
            parent_scores[~mask] = float('-inf')
            node['parent_id'] = parent_scores.argmax().item()

        nodes.append(node)

    # Build children lists
    id_to_node = {n['id']: n for n in nodes}
    for node in nodes:
        if node['parent_id'] >= 0 and node['parent_id'] in id_to_node:
            id_to_node[node['parent_id']]['children'].append(node['id'])

    return nodes

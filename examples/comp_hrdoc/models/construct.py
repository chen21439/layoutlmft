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
    """Tree-aware Relation Prediction Head (Paper Section 4.4.1)

    Predicts parent-child and sibling relationships for TOC generation.
    Uses dot product scoring (Eq. 18-19): f_ij = FC_q(F_Si) · FC_k(F_Sj)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Parent prediction (Eq. 18-19): f_ij = FC_q(F_Si) · FC_k(F_Sj)
        # For each node i, predict which node j is its parent
        self.parent_query = nn.Linear(hidden_size, hidden_size)
        self.parent_key = nn.Linear(hidden_size, hidden_size)

        # Sibling prediction (same formulation as parent)
        # For each node i, predict which node j is its left sibling
        self.sibling_query = nn.Linear(hidden_size, hidden_size)
        self.sibling_key = nn.Linear(hidden_size, hidden_size)

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
                  sibling_logits[i, j] = score that j is left sibling of i
        """
        batch_size, num_nodes, hidden_size = features.shape

        # Parent prediction: dot product (Eq. 18-19)
        parent_q = self.parent_query(features)  # [B, N, H] - child queries
        parent_k = self.parent_key(features)    # [B, N, H] - parent keys
        # parent_logits[i, j] = likelihood that j is parent of i
        parent_logits = torch.bmm(parent_q, parent_k.transpose(1, 2)) * self.scale

        # Sibling prediction: dot product (same as parent)
        sib_q = self.sibling_query(features)  # [B, N, H]
        sib_k = self.sibling_key(features)    # [B, N, H]
        # sibling_logits[i, j] = likelihood that j is left sibling of i
        sibling_logits = torch.bmm(sib_q, sib_k.transpose(1, 2)) * self.scale

        # Apply mask with large negative value instead of -inf for numerical stability
        if mask is not None:
            row_mask = ~mask.unsqueeze(2)
            col_mask = ~mask.unsqueeze(1)
            combined_mask = row_mask | col_mask

            # Use -1e4 instead of -inf to avoid NaN in softmax (fp16 safe)
            parent_logits = parent_logits.masked_fill(combined_mask, -1e4)
            sibling_logits = sibling_logits.masked_fill(combined_mask, -1e4)

            # Diagonal: node cannot be its own sibling, but CAN be its own parent (root self-pointing)
            # 论文 Section 4.4.1: "its parent-child relationship is defined as pointing to itself"
            diag = torch.eye(num_nodes, dtype=torch.bool, device=features.device)
            # Note: parent_logits keeps diagonal for root self-pointing
            sibling_logits = sibling_logits.masked_fill(diag.unsqueeze(0), -1e4)

        return {
            'parent_logits': parent_logits,
            'sibling_logits': sibling_logits,
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
    """Construct Module Loss (Paper Section 4.4.1)

    Combines (using softmax CE as dependency parsing, per paper):
    - Parent prediction loss (softmax CE, N选1)
    - Sibling prediction loss (softmax CE, N选1)

    Note: 采用论文自指向方案，Root 节点的 parent label 指向自己（而不是 -1）。
    这样 logits 维度是 [B, N, N]，所有 label 都在 [0, N-1] 范围内。
    """

    def __init__(
        self,
        parent_weight: float = 1.0,
        sibling_weight: float = 0.5,
    ):
        super().__init__()
        self.parent_weight = parent_weight
        self.sibling_weight = sibling_weight

    def forward(
        self,
        parent_logits: torch.Tensor,   # [batch, N, N]
        sibling_logits: torch.Tensor,  # [batch, N, N]
        parent_labels: torch.Tensor,   # [batch, N] index of parent (self-index for root)
        sibling_labels: torch.Tensor = None,  # [batch, N] index of left sibling (-1 for none)
        mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            parent_logits: [batch, N, N] parent prediction scores
            sibling_logits: [batch, N, N] sibling prediction scores
            parent_labels: [batch, N] ground truth parent indices
                           Root 节点的 parent_label == self_index (自指向)
                           其他节点指向其父节点
            sibling_labels: [batch, N] ground truth left sibling indices (-1 = no left sibling)
            mask: [batch, N] valid node mask

        Returns:
            Dict with parent_loss, sibling_loss, total_loss
        """
        batch_size, num_nodes = parent_labels.shape
        device = parent_logits.device

        # Valid nodes (not padding)
        if mask is None:
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)

        # Parent loss: 所有有效节点都计算 loss（包括 root 自指向）
        # 论文方案：parent_label 在 [0, N-1] 范围内，root 指向自己
        parent_loss = torch.tensor(0.0, device=device)
        valid_parent = mask & (parent_labels >= 0) & (parent_labels < num_nodes)
        if valid_parent.any():
            parent_logits_flat = parent_logits.view(-1, num_nodes)
            # Clamp labels to avoid CUDA assert (padding positions have -1)
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

        # Sibling loss (softmax CE, N选1, same as parent)
        sibling_loss = torch.tensor(0.0, device=device)
        if sibling_labels is not None and self.sibling_weight > 0:
            # Nodes that have a left sibling
            has_sibling = (sibling_labels >= 0) & mask
            valid_sibling = has_sibling & (sibling_labels < num_nodes)

            if valid_sibling.any():
                sibling_logits_flat = sibling_logits.view(-1, num_nodes)
                sibling_labels_clamped = sibling_labels.clamp(min=0, max=num_nodes-1)
                sibling_labels_flat = sibling_labels_clamped.view(-1)
                valid_sibling_flat = valid_sibling.view(-1)

                loss_flat = F.cross_entropy(
                    sibling_logits_flat,
                    sibling_labels_flat,
                    reduction='none'
                )
                loss_flat = loss_flat * valid_sibling_flat.float()
                sibling_loss = loss_flat.sum() / valid_sibling_flat.sum().clamp(min=1)

        # Total loss
        total_loss = (
            self.parent_weight * parent_loss +
            self.sibling_weight * sibling_loss
        )

        return {
            'parent_loss': parent_loss,
            'sibling_loss': sibling_loss,
            'loss': total_loss,
        }


def build_tree_from_predictions(
    parent_logits: torch.Tensor,  # [num_nodes, num_nodes]
    mask: torch.Tensor = None,    # [num_nodes]
) -> List[Dict]:
    """Build tree structure from predictions

    采用论文自指向方案：
    - Root 节点预测自己为 parent（parent_pred == self_index）
    - 其他节点预测其父节点

    Args:
        parent_logits: [num_nodes, num_nodes]
        mask: [num_nodes] valid nodes

    Returns:
        List of dicts with 'id', 'parent_id', 'children' keys
        其中 parent_id == -1 表示 root（与外部接口保持一致）
    """
    num_nodes = parent_logits.size(0)

    if mask is None:
        mask = torch.ones(num_nodes, dtype=torch.bool, device=parent_logits.device)

    # For each node, find its predicted parent
    parent_preds = parent_logits.argmax(dim=-1)  # [num_nodes]

    # Build nodes
    nodes = []
    for i in range(num_nodes):
        if not mask[i]:
            continue

        node = {'id': i, 'children': []}
        pred_parent = parent_preds[i].item()

        # 论文自指向方案：如果预测指向自己，则为 root
        if pred_parent == i:
            node['parent_id'] = -1  # 转换回 -1 以兼容外部接口
        elif mask[pred_parent]:
            node['parent_id'] = pred_parent
        else:
            # Fallback: find best valid parent (excluding self for non-roots)
            parent_scores = parent_logits[i].clone()
            parent_scores[~mask] = float('-inf')
            # 排除自己只在预测明显错误时使用
            if parent_scores.max() > float('-inf'):
                node['parent_id'] = parent_scores.argmax().item()
                if node['parent_id'] == i:
                    node['parent_id'] = -1  # 自指向 -> root
            else:
                node['parent_id'] = -1  # 无有效候选，设为 root

        nodes.append(node)

    # Build children lists
    id_to_node = {n['id']: n for n in nodes}
    for node in nodes:
        if node['parent_id'] >= 0 and node['parent_id'] in id_to_node:
            id_to_node[node['parent_id']]['children'].append(node['id'])

    return nodes

"""Multi-modal Embeddings for DOC Model

Based on "Detect-Order-Construct" paper.
Implements visual, text, and 2D positional embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEmbedding2D(nn.Module):
    """2D Positional Embedding

    Encodes bounding box and size information via MLP.
    Input: normalized bbox [x1, y1, x2, y2] in [0, 1]
    Output: positional embedding of hidden_size
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        max_position: int = 1000,
        use_learned: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_learned = use_learned

        if use_learned:
            # Learned position embeddings (like LayoutLM)
            self.x_embedding = nn.Embedding(max_position, hidden_size // 4)
            self.y_embedding = nn.Embedding(max_position, hidden_size // 4)
            self.w_embedding = nn.Embedding(max_position, hidden_size // 4)
            self.h_embedding = nn.Embedding(max_position, hidden_size // 4)
        else:
            # MLP-based embedding
            # Input: [x1, y1, x2, y2, w, h, cx, cy] = 8 dims
            self.mlp = nn.Sequential(
                nn.Linear(8, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, hidden_size),
            )

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bbox: [batch, num_regions, 4] normalized coordinates [x1, y1, x2, y2]
                  Values should be in [0, 1] or [0, 1000]

        Returns:
            [batch, num_regions, hidden_size] positional embeddings
        """
        if self.use_learned:
            # Scale to [0, 999] if normalized
            if bbox.max() <= 1.0:
                bbox = bbox * 999

            x1 = bbox[..., 0].clamp(0, 999).long()
            y1 = bbox[..., 1].clamp(0, 999).long()
            x2 = bbox[..., 2].clamp(0, 999).long()
            y2 = bbox[..., 3].clamp(0, 999).long()

            w = (x2 - x1).clamp(0, 999)
            h = (y2 - y1).clamp(0, 999)

            x_emb = self.x_embedding(x1)
            y_emb = self.y_embedding(y1)
            w_emb = self.w_embedding(w)
            h_emb = self.h_embedding(h)

            return torch.cat([x_emb, y_emb, w_emb, h_emb], dim=-1)
        else:
            # Compute additional features
            x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            features = torch.stack([x1, y1, x2, y2, w, h, cx, cy], dim=-1)
            return self.mlp(features)


class RegionTypeEmbedding(nn.Module):
    """Region Type Embedding

    Learned embedding from logical role category.
    Based on paper Eq. (13): R = LN(ReLU(FC(Embedding(r))))
    """

    def __init__(
        self,
        num_categories: int = 5,  # 0=padding, 1=fig, 2=tab, 3=para, 4=other
        hidden_size: int = 1024,
    ):
        super().__init__()
        # Embedding layer with 1024 hidden dimension (per paper)
        self.embedding = nn.Embedding(num_categories, hidden_size, padding_idx=0)
        # FC + ReLU + LayerNorm (per paper Eq. 13)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, categories: torch.Tensor) -> torch.Tensor:
        """
        Args:
            categories: [batch, num_regions] category IDs (0-4)

        Returns:
            [batch, num_regions, hidden_size] category embeddings
        """
        categories = categories.clamp(0, self.embedding.num_embeddings - 1)
        # Eq. (13): R = LN(ReLU(FC(Embedding(r))))
        emb = self.embedding(categories)
        emb = self.fc(emb)
        emb = F.relu(emb)
        emb = self.layer_norm(emb)
        return emb


class SpatialCompatibilityFeatures(nn.Module):
    """Spatial Compatibility Features

    Computes 18-dimensional spatial features for region pairs,
    used for reading order prediction.
    """

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        # 18-dim spatial features -> hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(18, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        bbox_i: torch.Tensor,  # [batch, num_regions, 4]
        bbox_j: torch.Tensor,  # [batch, num_regions, 4]
    ) -> torch.Tensor:
        """Compute spatial features for all pairs (i, j)

        Args:
            bbox_i: [batch, num_regions, 4] source bboxes
            bbox_j: [batch, num_regions, 4] target bboxes

        Returns:
            [batch, num_regions, num_regions, hidden_size] spatial features
        """
        batch_size, num_regions, _ = bbox_i.shape

        # Expand for pairwise computation
        # bbox_i: [batch, num_regions, 1, 4]
        # bbox_j: [batch, 1, num_regions, 4]
        bi = bbox_i.unsqueeze(2)
        bj = bbox_j.unsqueeze(1)

        # Extract coordinates
        x1_i, y1_i, x2_i, y2_i = bi[..., 0], bi[..., 1], bi[..., 2], bi[..., 3]
        x1_j, y1_j, x2_j, y2_j = bj[..., 0], bj[..., 1], bj[..., 2], bj[..., 3]

        # Compute derived features
        w_i, h_i = x2_i - x1_i, y2_i - y1_i
        w_j, h_j = x2_j - x1_j, y2_j - y1_j
        cx_i, cy_i = (x1_i + x2_i) / 2, (y1_i + y2_i) / 2
        cx_j, cy_j = (x1_j + x2_j) / 2, (y1_j + y2_j) / 2

        # 18 spatial features
        features = [
            # Position differences
            cx_j - cx_i,  # horizontal offset
            cy_j - cy_i,  # vertical offset
            x1_j - x1_i,  # left edge diff
            y1_j - y1_i,  # top edge diff
            x2_j - x2_i,  # right edge diff
            y2_j - y2_i,  # bottom edge diff

            # Size ratios (log-scale)
            torch.log((w_j + 1) / (w_i + 1)),
            torch.log((h_j + 1) / (h_i + 1)),

            # Overlap features
            torch.clamp(torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j), min=0),  # x overlap
            torch.clamp(torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j), min=0),  # y overlap

            # Distance features
            torch.sqrt((cx_j - cx_i) ** 2 + (cy_j - cy_i) ** 2 + 1e-6),  # euclidean
            torch.abs(cx_j - cx_i),  # manhattan x
            torch.abs(cy_j - cy_i),  # manhattan y

            # Relative position indicators
            (cx_j > cx_i).float(),  # j is to the right
            (cy_j > cy_i).float(),  # j is below
            (x1_j > x2_i).float(),  # j is completely to the right
            (y1_j > y2_i).float(),  # j is completely below

            # Area ratio
            torch.log((w_j * h_j + 1) / (w_i * h_i + 1)),
        ]

        # Stack: [batch, num_regions, num_regions, 18]
        spatial_features = torch.stack(features, dim=-1)

        return self.mlp(spatial_features)


class MultiModalEmbedding(nn.Module):
    """Multi-Modal Embedding Module

    Combines visual, text, and 2D positional embeddings
    into unified region representations.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_categories: int = 5,
        visual_dim: int = 1024,
        text_dim: int = 768,
        use_visual: bool = True,
        use_text: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_visual = use_visual
        self.use_text = use_text

        # 2D Positional Embedding (always used)
        self.pos_embedding = PositionalEmbedding2D(hidden_size)

        # Region Type Embedding
        self.type_embedding = RegionTypeEmbedding(num_categories, hidden_size)

        # Visual embedding projection
        if use_visual:
            self.visual_proj = nn.Linear(visual_dim, hidden_size)

        # Text embedding projection
        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden_size)

        # Combine embeddings
        num_inputs = 2  # pos + type
        if use_visual:
            num_inputs += 1
        if use_text:
            num_inputs += 1

        self.combine = nn.Sequential(
            nn.Linear(hidden_size * num_inputs, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        bbox: torch.Tensor,           # [batch, num_regions, 4]
        categories: torch.Tensor,     # [batch, num_regions]
        visual_features: Optional[torch.Tensor] = None,  # [batch, num_regions, visual_dim]
        text_features: Optional[torch.Tensor] = None,    # [batch, num_regions, text_dim]
    ) -> torch.Tensor:
        """
        Args:
            bbox: [batch, num_regions, 4] bounding boxes
            categories: [batch, num_regions] region categories
            visual_features: [batch, num_regions, visual_dim] visual features
            text_features: [batch, num_regions, text_dim] text features

        Returns:
            [batch, num_regions, hidden_size] multi-modal embeddings
        """
        embeddings = []

        # 2D Positional Embedding
        pos_emb = self.pos_embedding(bbox)
        embeddings.append(pos_emb)

        # Region Type Embedding
        type_emb = self.type_embedding(categories)
        embeddings.append(type_emb)

        # Visual Embedding
        if self.use_visual and visual_features is not None:
            vis_emb = self.visual_proj(visual_features)
            embeddings.append(vis_emb)
        elif self.use_visual:
            # Placeholder zeros if visual features not provided
            embeddings.append(torch.zeros_like(pos_emb))

        # Text Embedding
        if self.use_text and text_features is not None:
            txt_emb = self.text_proj(text_features)
            embeddings.append(txt_emb)
        elif self.use_text:
            # Placeholder zeros if text features not provided
            embeddings.append(torch.zeros_like(pos_emb))

        # Combine
        combined = torch.cat(embeddings, dim=-1)
        output = self.combine(combined)
        output = self.layer_norm(output)

        return output


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)

    Used in Construct module to incorporate reading order information.
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for all positions
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """Apply rotary positional embedding

        Args:
            x: [batch, heads, seq_len, dim] or [batch, seq_len, dim]

        Returns:
            Tensor with RoPE applied
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)

        return self._apply_rotary_pos_emb(x, self.cos_cached[:, :, :seq_len],
                                           self.sin_cached[:, :, :seq_len])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        # Handle different input shapes
        if x.dim() == 3:
            cos = cos.squeeze(1)
            sin = sin.squeeze(1)

        return (x * cos) + (self._rotate_half(x) * sin)

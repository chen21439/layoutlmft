"""Order-only 模型

简化版 Order 模型，直接使用区域级特征（bbox + category），
不需要 LayoutXLM backbone。适用于 Comp_HRDoc 数据集训练。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class RegionEmbedding(nn.Module):
    """区域特征嵌入

    将 bbox 和 category 编码为特征向量。
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 5,  # 0=padding, 1=fig, 2=tab, 3=para, 4=other
        max_position: int = 1000,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Category embedding
        self.category_embedding = nn.Embedding(num_categories, hidden_size, padding_idx=0)

        # Bbox embedding (4 coordinates -> hidden_size)
        self.bbox_embedding = nn.Linear(4, hidden_size)

        # Position embedding (for x, y positions)
        self.x_position_embedding = nn.Embedding(max_position, hidden_size // 4)
        self.y_position_embedding = nn.Embedding(max_position, hidden_size // 4)
        self.width_embedding = nn.Embedding(max_position, hidden_size // 4)
        self.height_embedding = nn.Embedding(max_position, hidden_size // 4)

        # Combine embeddings
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        bboxes: torch.Tensor,  # [batch, num_regions, 4] - pixel coordinates (x1, y1, x2, y2)
        categories: torch.Tensor,  # [batch, num_regions]
    ) -> torch.Tensor:
        """
        Args:
            bboxes: [batch, num_regions, 4] bbox coordinates (x1, y1, x2, y2) in pixels
            categories: [batch, num_regions] category ids (1-4, or -1 for invalid)

        Returns:
            [batch, num_regions, hidden_size] region embeddings
        """
        batch_size, num_regions = categories.shape

        # Category embedding - clamp to valid range [0, 4]
        # -1 (invalid) -> 0 (padding)
        categories_clamped = categories.clamp(0, 4)
        cat_emb = self.category_embedding(categories_clamped)  # [batch, num_regions, hidden_size]

        # Bbox position embedding
        # First extract raw coordinates
        x1 = bboxes[..., 0]
        y1 = bboxes[..., 1]
        x2 = bboxes[..., 2]
        y2 = bboxes[..., 3]

        # Calculate width and height before any clamping
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)

        # Now clamp all values to [0, 999] for embedding lookup
        x1_idx = x1.clamp(0, 999).long()
        y1_idx = y1.clamp(0, 999).long()
        w_idx = w.clamp(0, 999).long()
        h_idx = h.clamp(0, 999).long()

        x_emb = self.x_position_embedding(x1_idx)
        y_emb = self.y_position_embedding(y1_idx)
        w_emb = self.width_embedding(w_idx)
        h_emb = self.height_embedding(h_idx)

        # Concatenate position embeddings
        pos_emb = torch.cat([x_emb, y_emb, w_emb, h_emb], dim=-1)  # [batch, num_regions, hidden_size]

        # Combine all embeddings
        combined = torch.cat([cat_emb, pos_emb], dim=-1)  # [batch, num_regions, hidden_size * 2]
        output = self.combine(combined)  # [batch, num_regions, hidden_size]
        output = self.layer_norm(output)

        return output


class OrderTransformer(nn.Module):
    """Order Transformer 编码器

    3层 Transformer，用于捕获区域间的上下文关系。
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        features: torch.Tensor,  # [batch, num_regions, hidden_size]
        mask: torch.Tensor = None,  # [batch, num_regions] - True for valid
    ) -> torch.Tensor:
        """
        Args:
            features: [batch, num_regions, hidden_size]
            mask: [batch, num_regions] boolean mask (True = valid)

        Returns:
            [batch, num_regions, hidden_size] enhanced features
        """
        # Convert mask to attention mask format
        # nn.TransformerEncoder expects key_padding_mask where True = ignore
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True -> ignore
        else:
            key_padding_mask = None

        output = self.transformer(features, src_key_padding_mask=key_padding_mask)
        return output


class BiaffineScorer(nn.Module):
    """双仿射评分模块

    计算区域对之间的顺序关系得分。
    """

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()

        # Project to head/tail representations
        self.head_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tail_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Biaffine weight
        self.biaffine_weight = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.biaffine_weight)

        # Bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, num_regions, hidden_size]

        Returns:
            [batch, num_regions, num_regions] pairwise scores
            score[i,j] > 0 means region i comes before region j
        """
        head = self.head_proj(features)  # [batch, num_regions, hidden_size]
        tail = self.tail_proj(features)  # [batch, num_regions, hidden_size]

        # Biaffine: head @ W @ tail.T
        # [batch, num_regions, hidden_size] @ [hidden_size, hidden_size] @ [batch, hidden_size, num_regions]
        scores = torch.einsum('bih,hd,bjd->bij', head, self.biaffine_weight, tail)
        scores = scores + self.bias

        return scores


class OrderLoss(nn.Module):
    """Order 损失函数

    使用 BCE loss 训练成对顺序预测。
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        logits: torch.Tensor,  # [batch, num_regions, num_regions]
        reading_orders: torch.Tensor,  # [batch, num_regions] - reading order indices
        mask: torch.Tensor,  # [batch, num_regions] - valid region mask
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_regions, num_regions] pairwise scores
            reading_orders: [batch, num_regions] ground truth reading order
            mask: [batch, num_regions] valid region mask

        Returns:
            Scalar loss value
        """
        batch_size, num_regions = reading_orders.shape
        device = logits.device

        # Create ground truth: target[i,j] = 1 if order[i] < order[j]
        order_i = reading_orders.unsqueeze(2)  # [batch, num_regions, 1]
        order_j = reading_orders.unsqueeze(1)  # [batch, 1, num_regions]
        targets = (order_i < order_j).float()  # [batch, num_regions, num_regions]

        # Create valid mask for pairs
        # Both regions must be valid, and they must be different
        valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [batch, num_regions, num_regions]
        diag_mask = ~torch.eye(num_regions, dtype=torch.bool, device=device).unsqueeze(0)
        valid_mask = valid_mask & diag_mask

        # Compute BCE loss
        loss = self.bce(logits, targets)

        # Apply mask and average
        loss = loss * valid_mask.float()

        num_valid_pairs = valid_mask.sum()
        if num_valid_pairs > 0:
            loss = loss.sum() / num_valid_pairs
        else:
            loss = loss.sum() * 0.0  # No valid pairs

        return loss


class OrderOnlyModel(nn.Module):
    """Order-only 完整模型

    直接使用区域级特征进行阅读顺序预测。
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_categories: int = 5,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.region_embedding = RegionEmbedding(
            hidden_size=hidden_size,
            num_categories=num_categories,
        )

        self.transformer = OrderTransformer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.scorer = BiaffineScorer(hidden_size=hidden_size, dropout=dropout)
        self.loss_fn = OrderLoss()

    def forward(
        self,
        bboxes: torch.Tensor,  # [batch, num_regions, 4]
        categories: torch.Tensor,  # [batch, num_regions]
        region_mask: torch.Tensor,  # [batch, num_regions]
        reading_orders: torch.Tensor = None,  # [batch, num_regions] - for training
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bboxes: [batch, num_regions, 4] normalized bbox coordinates
            categories: [batch, num_regions] category ids (1-4)
            region_mask: [batch, num_regions] valid region mask
            reading_orders: [batch, num_regions] ground truth reading order (for training)

        Returns:
            Dict with:
                - order_logits: [batch, num_regions, num_regions] pairwise scores
                - loss: scalar (if reading_orders provided)
        """
        # Embed regions
        region_features = self.region_embedding(bboxes, categories)

        # Apply transformer
        enhanced_features = self.transformer(region_features, region_mask)

        # Compute pairwise scores
        order_logits = self.scorer(enhanced_features)

        outputs = {
            'order_logits': order_logits,
            'enhanced_features': enhanced_features,
        }

        # Compute loss if training
        if reading_orders is not None:
            loss = self.loss_fn(order_logits, reading_orders, region_mask)
            outputs['loss'] = loss
            outputs['order_loss'] = loss  # Alias for compatibility
            outputs['cls_loss'] = torch.tensor(0.0, device=bboxes.device)

        return outputs

    def predict_order(
        self,
        bboxes: torch.Tensor,
        categories: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> torch.Tensor:
        """预测阅读顺序

        Returns:
            [batch, num_regions] predicted reading order indices
        """
        with torch.no_grad():
            outputs = self.forward(bboxes, categories, region_mask)
            order_logits = outputs['order_logits']

            # For each region, count how many regions it comes before
            # The region with the most "comes before" relations is first
            scores = (order_logits > 0).float().sum(dim=2)  # [batch, num_regions]

            # Sort by scores (descending) to get order
            _, predicted_order = scores.sort(dim=1, descending=True)

            return predicted_order


def build_order_only_model(
    hidden_size: int = 768,
    num_categories: int = 5,
    num_heads: int = 8,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> OrderOnlyModel:
    """构建 Order-only 模型"""
    return OrderOnlyModel(
        hidden_size=hidden_size,
        num_categories=num_categories,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def save_order_only_model(model: OrderOnlyModel, save_path: str):
    """保存模型"""
    import os
    os.makedirs(save_path, exist_ok=True)

    model_path = os.path.join(save_path, "order_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save config
    import json
    config = {
        'hidden_size': model.region_embedding.hidden_size,
        'num_categories': model.region_embedding.category_embedding.num_embeddings,
    }
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {save_path}")


def load_order_only_model(model_path: str, device: str = "cuda") -> OrderOnlyModel:
    """加载模型"""
    import os
    import json

    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Build model
    model = build_order_only_model(**config)

    # Load weights
    weights_path = os.path.join(model_path, "order_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    return model.to(device)

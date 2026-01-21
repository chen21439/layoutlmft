"""模型构建工厂

提供统一的模型构建和加载接口。
"""

import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .backbone import LayoutXLMBackbone
from .heads import OrderHead, OrderLoss
from .modules.pooling import aggregate_document_line_features


class OrderModel(nn.Module):
    """Order 模块完整模型

    组合 LayoutXLM backbone + 行特征聚合 + Order 预测头
    用于训练阅读顺序预测任务。
    """

    def __init__(
        self,
        backbone: LayoutXLMBackbone,
        order_head: OrderHead,
        lambda_cls: float = 1.0,
        lambda_order: float = 1.0,
    ):
        """
        Args:
            backbone: LayoutXLM 基座模型
            order_head: Order 预测头
            lambda_cls: 分类损失权重
            lambda_order: Order 损失权重
        """
        super().__init__()
        self.backbone = backbone
        self.order_head = order_head
        self.order_loss_fn = OrderLoss()

        self.lambda_cls = lambda_cls
        self.lambda_order = lambda_order

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ids: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        reading_order: Optional[torch.Tensor] = None,
        line_mask: Optional[torch.Tensor] = None,
        num_docs: Optional[int] = None,
        chunks_per_doc: Optional[list] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            input_ids: [total_chunks, seq_len]
            bbox: [total_chunks, seq_len, 4]
            attention_mask: [total_chunks, seq_len]
            line_ids: [total_chunks, seq_len] 每个 token 的 line_id
            image: [total_chunks, 3, H, W] 可选
            labels: [total_chunks, seq_len] 分类标签，可选
            reading_order: [num_docs, max_lines] 阅读顺序，可选
            line_mask: [num_docs, max_lines] 有效行掩码，可选
            num_docs: batch 中的文档数
            chunks_per_doc: 每个文档的 chunk 数

        Returns:
            Dict containing:
                - loss: 总损失
                - cls_loss: 分类损失
                - order_loss: Order 损失
                - order_logits: 阅读顺序预测
                - logits: 分类预测
        """
        device = input_ids.device
        total_chunks = input_ids.shape[0]

        # ==================== Stage 1: Backbone 特征提取 ====================
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            labels=labels,
            output_hidden_states=True,
        )

        cls_loss = backbone_outputs.get("loss", torch.tensor(0.0, device=device))
        logits = backbone_outputs["logits"]
        hidden_states = backbone_outputs["hidden_states"]

        outputs = {
            "logits": logits,
            "cls_loss": cls_loss,
        }

        # 如果没有文档级信息，直接返回
        if num_docs is None or chunks_per_doc is None:
            outputs["loss"] = cls_loss * self.lambda_cls
            return outputs

        # ==================== Stage 2: 行特征聚合 ====================
        text_seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :text_seq_len, :]

        # 按文档聚合行特征
        doc_line_features_list = []
        doc_line_masks_list = []

        chunk_idx = 0
        for doc_idx in range(num_docs):
            num_chunks_in_doc = chunks_per_doc[doc_idx]

            # 该文档的 hidden states 和 line_ids
            doc_hidden = text_hidden[chunk_idx:chunk_idx + num_chunks_in_doc]
            doc_line_ids = line_ids[chunk_idx:chunk_idx + num_chunks_in_doc]

            # 聚合行特征
            doc_features, doc_mask = aggregate_document_line_features(
                doc_hidden, doc_line_ids
            )
            doc_line_features_list.append(doc_features)
            doc_line_masks_list.append(doc_mask)

            chunk_idx += num_chunks_in_doc

        # 填充到相同长度
        max_lines = max(f.shape[0] for f in doc_line_features_list)
        hidden_dim = doc_line_features_list[0].shape[1]

        line_features = torch.zeros(num_docs, max_lines, hidden_dim, device=device)
        aggregated_line_mask = torch.zeros(num_docs, max_lines, dtype=torch.bool, device=device)

        for doc_idx, (features, mask) in enumerate(zip(doc_line_features_list, doc_line_masks_list)):
            num_lines = features.shape[0]
            line_features[doc_idx, :num_lines] = features
            aggregated_line_mask[doc_idx, :num_lines] = mask

        # ==================== Stage 3: Order 预测 ====================
        order_outputs = self.order_head(line_features, aggregated_line_mask)
        order_logits = order_outputs["order_logits"]
        outputs["order_logits"] = order_logits
        outputs["enhanced_features"] = order_outputs["enhanced_features"]

        # 计算 Order 损失
        order_loss = torch.tensor(0.0, device=device)
        if reading_order is not None and self.lambda_order > 0:
            # 确保 reading_order 和 aggregated_line_mask 对齐
            actual_max_lines = order_logits.shape[1]

            # 截断或填充 reading_order
            if reading_order.shape[1] > actual_max_lines:
                reading_order = reading_order[:, :actual_max_lines]
            elif reading_order.shape[1] < actual_max_lines:
                padding = torch.full(
                    (num_docs, actual_max_lines - reading_order.shape[1]),
                    -1, dtype=reading_order.dtype, device=device
                )
                reading_order = torch.cat([reading_order, padding], dim=1)

            order_loss = self.order_loss_fn(
                order_logits,
                reading_order,
                aggregated_line_mask,
            )

        outputs["order_loss"] = order_loss

        # 总损失
        total_loss = cls_loss * self.lambda_cls + order_loss * self.lambda_order
        outputs["loss"] = total_loss

        return outputs


def build_order_model(
    model_path: str,
    num_labels: int = 16,
    hidden_size: int = 768,
    num_heads: int = 8,
    num_layers: int = 3,
    dropout: float = 0.1,
    lambda_cls: float = 1.0,
    lambda_order: float = 1.0,
    freeze_backbone: bool = False,
    use_biaffine: bool = True,
) -> OrderModel:
    """构建 Order 模型

    Args:
        model_path: LayoutXLM 预训练模型路径
        num_labels: 分类标签数
        hidden_size: 隐藏层维度
        num_heads: Transformer 注意力头数
        num_layers: Order Transformer 层数
        dropout: Dropout 比例
        lambda_cls: 分类损失权重
        lambda_order: Order 损失权重
        freeze_backbone: 是否冻结 backbone
        use_biaffine: 是否使用双仿射变换

    Returns:
        OrderModel 实例
    """
    # 构建 backbone
    backbone = LayoutXLMBackbone(
        model_path=model_path,
        num_labels=num_labels,
        freeze_backbone=freeze_backbone,
    )

    # 构建 Order Head
    order_head = OrderHead(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_biaffine=use_biaffine,
    )

    return OrderModel(
        backbone=backbone,
        order_head=order_head,
        lambda_cls=lambda_cls,
        lambda_order=lambda_order,
    )


def save_model(model: OrderModel, save_path: str):
    """保存模型

    Args:
        model: 要保存的模型
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)

    # 保存 backbone (LayoutXLM)
    backbone_path = os.path.join(save_path, "backbone")
    model.backbone.save_pretrained(backbone_path)

    # 保存 Order head
    order_head_path = os.path.join(save_path, "order_head.pt")
    torch.save(model.order_head.state_dict(), order_head_path)

    print(f"Model saved to {save_path}")


def load_model(model_path: str, device: str = "cuda") -> OrderModel:
    """加载模型

    Args:
        model_path: 模型路径
        device: 设备

    Returns:
        加载的 OrderModel
    """
    backbone_path = os.path.join(model_path, "backbone")
    order_head_path = os.path.join(model_path, "order_head.pt")

    # 构建模型
    model = build_order_model(model_path=backbone_path)

    # 加载 Order head 权重
    if os.path.exists(order_head_path):
        model.order_head.load_state_dict(torch.load(order_head_path, map_location=device))

    return model.to(device)

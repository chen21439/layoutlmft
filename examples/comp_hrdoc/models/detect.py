#!/usr/bin/env python
# coding=utf-8
"""
Stage1 特征提取模块

包含 backbone + line_pooling + line_enhancer 的特征提取流程。
用于 comp_hrdoc 的训练和推理，不依赖 stage 目录。

架构:
    LayoutXLM Backbone
           ↓
    LinePooling (token → line)
           ↓
    LineFeatureEnhancer (可选)
           ↓
    Line Features [L, 768]

复用模块（全部在 comp_hrdoc 内部）:
- models/modules/line_pooling.py: LinePooling
- models/modules/line_transformer.py: LineFeatureEnhancer
- models/heads/classification_head.py: LineClassificationHead
"""

import os
import logging
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn

# 本地导入
from .modules.line_pooling import LinePooling
from .modules.line_transformer import LineFeatureEnhancer
from .heads.classification_head import LineClassificationHead

# ImageList 支持（detectron2 或 shim）
try:
    from detectron2.structures import ImageList
except ImportError:
    ImageList = None

logger = logging.getLogger(__name__)


class Stage1Backbone(nn.Module):
    """Stage1 特征提取器

    封装 backbone + line_pooling + line_enhancer，用于提取 line-level 特征。

    与 stage/JointModel 的区别：
    - 不包含 stage3/stage4（parent/relation 预测）
    - 专注于特征提取，用于 Construct 模块

    Example:
        >>> backbone = LayoutXLMForTokenClassification.from_pretrained(...)
        >>> model = Stage1Backbone(backbone, use_line_enhancer=True)
        >>> hidden = model.encode_with_micro_batch(input_ids, bbox, attention_mask, image)
        >>> line_features, line_mask = model.line_pooling(hidden, line_ids)
        >>> if model.line_enhancer:
        ...     line_features = model.line_enhancer(line_features, line_mask)
    """

    def __init__(
        self,
        backbone_model,  # LayoutXLMForTokenClassification
        num_classes: int = 14,
        hidden_size: int = 768,
        use_line_enhancer: bool = True,
        use_cls_head: bool = False,
        micro_batch_size: int = 8,
        cls_dropout: float = 0.1,
    ):
        """
        Args:
            backbone_model: LayoutXLM 模型（LayoutXLMForTokenClassification）
            num_classes: 分类类别数（默认 14）
            hidden_size: Hidden dimension
            use_line_enhancer: 是否启用行间特征增强
            use_cls_head: 是否包含分类头
            micro_batch_size: micro-batch 大小
            cls_dropout: 分类头 dropout
        """
        super().__init__()

        # ========== Backbone ==========
        self.backbone = backbone_model
        self.hidden_size = hidden_size
        self.micro_batch_size = micro_batch_size

        # ========== LinePooling ==========
        self.line_pooling = LinePooling(pooling_method="mean")

        # ========== LineFeatureEnhancer (可选) ==========
        self.use_line_enhancer = use_line_enhancer
        if use_line_enhancer:
            self.line_enhancer = LineFeatureEnhancer(
                hidden_size=hidden_size,
                num_heads=12,  # 论文配置
                ffn_dim=2048,  # 论文配置
                dropout=0.1,
                num_layers=1,  # 论文配置
                enabled=True,
            )
            logger.info(f"[Stage1Backbone] LineFeatureEnhancer enabled (1 layer, 12 heads, FFN=2048)")
        else:
            self.line_enhancer = None

        # ========== Classification Head (可选) ==========
        self.use_cls_head = use_cls_head
        if use_cls_head:
            self.cls_head = LineClassificationHead(
                hidden_size=hidden_size,
                num_classes=num_classes,
                dropout=cls_dropout,
            )
        else:
            self.cls_head = None

    def encode_with_micro_batch(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor = None,
        micro_batch_size: int = None,
        no_grad: bool = False,
    ) -> torch.Tensor:
        """
        使用 micro-batching 获取 backbone hidden states

        Args:
            input_ids: [num_chunks, seq_len]
            bbox: [num_chunks, seq_len, 4]
            attention_mask: [num_chunks, seq_len]
            image: [num_chunks, C, H, W] or List[Tensor] or None
            micro_batch_size: micro-batch 大小
            no_grad: 是否禁用梯度

        Returns:
            hidden_states: [num_chunks, seq_len+visual_len, hidden_dim]
        """
        device = input_ids.device
        total_chunks = input_ids.shape[0]
        micro_bs = micro_batch_size if micro_batch_size is not None else self.micro_batch_size

        # 检测 image 类型
        image_is_list = isinstance(image, list) if image is not None else False
        image_is_imagelist = (ImageList is not None and isinstance(image, ImageList)) if image is not None else False

        def _run_backbone(ids, bb, mask, img):
            return self.backbone(
                input_ids=ids,
                bbox=bb,
                attention_mask=mask,
                image=img,
                output_hidden_states=True,
            )

        def _slice_imagelist(img_list, start, end):
            """切片 ImageList"""
            if hasattr(img_list, 'tensor'):
                sliced_tensor = img_list.tensor[start:end].to(device)
            else:
                sliced_tensor = img_list.tensors[start:end].to(device)
            sliced_sizes = img_list.image_sizes[start:end]
            return ImageList(sliced_tensor, sliced_sizes)

        # 验证：image 数量必须与 chunks 数量匹配
        if image_is_list and image and len(image) != total_chunks:
            raise ValueError(
                f"Image/Chunks mismatch: len(image)={len(image)}, total_chunks={total_chunks}."
            )

        # 分批处理
        all_hidden = []

        for start_idx in range(0, total_chunks, micro_bs):
            end_idx = min(start_idx + micro_bs, total_chunks)

            mb_input_ids = input_ids[start_idx:end_idx]
            mb_bbox = bbox[start_idx:end_idx]
            mb_attention_mask = attention_mask[start_idx:end_idx]

            if image_is_list and image:
                mb_images = image[start_idx:end_idx]
                mb_image = torch.stack([
                    torch.tensor(img) if not isinstance(img, torch.Tensor) else img
                    for img in mb_images
                ]).to(device)
            elif image_is_imagelist:
                mb_image = _slice_imagelist(image, start_idx, end_idx)
            elif image is not None:
                mb_image = image[start_idx:end_idx]
            else:
                mb_image = None

            if no_grad:
                with torch.no_grad():
                    mb_outputs = _run_backbone(mb_input_ids, mb_bbox, mb_attention_mask, mb_image)
            else:
                mb_outputs = _run_backbone(mb_input_ids, mb_bbox, mb_attention_mask, mb_image)

            all_hidden.append(mb_outputs.hidden_states[-1])
            del mb_image

        return torch.cat(all_hidden, dim=0)

    def extract_line_features(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ids: torch.Tensor,
        image: torch.Tensor = None,
        no_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整的特征提取流程：backbone → line_pooling → line_enhancer

        Args:
            input_ids: [num_chunks, seq_len]
            bbox: [num_chunks, seq_len, 4]
            attention_mask: [num_chunks, seq_len]
            line_ids: [num_chunks, seq_len]
            image: [num_chunks, C, H, W] or None
            no_grad: 是否禁用梯度

        Returns:
            line_features: [num_lines, hidden_dim]
            line_mask: [num_lines]
        """
        # Step 1: Backbone
        hidden_states = self.encode_with_micro_batch(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            no_grad=no_grad,
        )

        # 截取文本部分（排除视觉 tokens）
        seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :seq_len, :]

        # Step 2: Line Pooling
        line_features, line_mask = self.line_pooling(text_hidden, line_ids)

        # Step 3: Line Enhancer (可选)
        if self.line_enhancer is not None:
            # 添加 batch 维度
            line_features = line_features.unsqueeze(0)  # [1, L, H]
            line_mask = line_mask.unsqueeze(0)  # [1, L]
            line_features = self.line_enhancer(line_features, line_mask)
            # 移除 batch 维度
            line_features = line_features.squeeze(0)  # [L, H]
            line_mask = line_mask.squeeze(0)  # [L]

        return line_features, line_mask

    def freeze_backbone(self):
        """冻结 backbone 参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("[Stage1Backbone] Backbone frozen")

    def unfreeze_backbone(self):
        """解冻 backbone 参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("[Stage1Backbone] Backbone unfrozen")

    def freeze_visual(self):
        """冻结视觉编码器"""
        if hasattr(self.backbone, 'layoutlmv2'):
            visual = self.backbone.layoutlmv2.visual
            for param in visual.parameters():
                param.requires_grad = False
            visual.eval()
            logger.info("[Stage1Backbone] Visual encoder frozen")


def build_stage1_backbone(
    checkpoint_path: str,
    device: str = None,
    use_line_enhancer: bool = True,
    use_cls_head: bool = False,
) -> Tuple['Stage1Backbone', Any]:
    """
    从 checkpoint 构建 Stage1Backbone

    支持两种格式：
    1. 联合训练 checkpoint（包含 stage1/ 子目录）
    2. 标准 LayoutXLM checkpoint

    Args:
        checkpoint_path: checkpoint 路径
        device: 计算设备
        use_line_enhancer: 是否启用 line_enhancer
        use_cls_head: 是否包含分类头

    Returns:
        model: Stage1Backbone 实例
        tokenizer: tokenizer 实例
    """
    from layoutlmft.models.layoutxlm import (
        LayoutXLMForTokenClassification,
        LayoutXLMConfig,
        LayoutXLMTokenizerFast,
    )
    from layoutlmft.data.labels import NUM_LABELS, get_id2label, get_label2id

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 检测 checkpoint 格式
    stage1_subdir = os.path.join(checkpoint_path, "stage1")
    if os.path.isdir(stage1_subdir):
        # 联合训练格式
        backbone_path = stage1_subdir
    else:
        # 标准格式
        backbone_path = checkpoint_path

    logger.info(f"Loading backbone from: {backbone_path}")

    # 加载 backbone
    config = LayoutXLMConfig.from_pretrained(backbone_path)
    config.num_labels = NUM_LABELS
    config.id2label = get_id2label()
    config.label2id = get_label2id()

    backbone = LayoutXLMForTokenClassification.from_pretrained(
        backbone_path,
        config=config,
    )

    # 创建 Stage1Backbone
    model = Stage1Backbone(
        backbone_model=backbone,
        use_line_enhancer=use_line_enhancer,
        use_cls_head=use_cls_head,
    )

    # 加载 cls_head 权重（如果存在）
    if use_cls_head:
        cls_head_path = os.path.join(checkpoint_path, "cls_head.pt")
        if os.path.exists(cls_head_path):
            logger.info(f"Loading cls_head from: {cls_head_path}")
            cls_head_state = torch.load(cls_head_path, map_location="cpu")
            model.cls_head.load_state_dict(cls_head_state)

    # 加载 line_enhancer 权重（如果存在）
    if use_line_enhancer:
        line_enhancer_path = os.path.join(checkpoint_path, "line_enhancer.pt")
        if os.path.exists(line_enhancer_path):
            logger.info(f"Loading line_enhancer from: {line_enhancer_path}")
            line_enhancer_state = torch.load(line_enhancer_path, map_location="cpu")
            model.line_enhancer.load_state_dict(line_enhancer_state)
            logger.info("  line_enhancer loaded successfully")
        else:
            logger.warning(f"line_enhancer weights not found: {line_enhancer_path}")
            logger.warning("  line_enhancer will use random initialization")

    model = model.to(device)

    # 加载 tokenizer
    try:
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(checkpoint_path)
        logger.info(f"Loaded tokenizer from: {checkpoint_path}")
    except Exception:
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(backbone_path)
        logger.info(f"Loaded tokenizer from: {backbone_path}")

    logger.info("[Stage1Backbone] Model loaded successfully")
    return model, tokenizer


def save_stage1_backbone(
    model: 'Stage1Backbone',
    save_path: str,
    tokenizer=None,
):
    """
    保存 Stage1Backbone

    保存结构：
    - stage1/: backbone 权重 (HuggingFace 格式)
    - cls_head.pt: 分类头权重（如果存在）
    - line_enhancer.pt: line_enhancer 权重（如果存在）
    - tokenizer files

    Args:
        model: Stage1Backbone 实例
        save_path: 保存路径
        tokenizer: tokenizer 实例（可选）
    """
    os.makedirs(save_path, exist_ok=True)

    # 保存 backbone
    backbone_path = os.path.join(save_path, "stage1")
    os.makedirs(backbone_path, exist_ok=True)
    model.backbone.save_pretrained(backbone_path)
    logger.info(f"Saved backbone to: {backbone_path}")

    # 保存 cls_head
    if model.cls_head is not None:
        cls_head_path = os.path.join(save_path, "cls_head.pt")
        torch.save(model.cls_head.state_dict(), cls_head_path)
        logger.info(f"Saved cls_head to: {cls_head_path}")

    # 保存 line_enhancer
    if model.line_enhancer is not None:
        line_enhancer_path = os.path.join(save_path, "line_enhancer.pt")
        torch.save(model.line_enhancer.state_dict(), line_enhancer_path)
        logger.info(f"Saved line_enhancer to: {line_enhancer_path}")

    # 保存 tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        logger.info(f"Saved tokenizer to: {save_path}")

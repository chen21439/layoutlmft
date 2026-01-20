#!/usr/bin/env python
# coding=utf-8
"""
Stage1 + Construct 联合训练模型

复用现有组件实现 backbone → line_pooling → construct 的端到端训练。

核心设计：
1. 复用 stage/models/build.py 的 load_joint_model() 加载 backbone + line_pooling
2. 复用 comp_hrdoc/models/construct_only.py 的 ConstructFromFeatures
3. 梯度从 construct loss 回传到 backbone

调用链：
    JointModel.backbone (trainable)
        ↓
    JointModel.line_pooling
        ↓
    ConstructFromFeatures (trainable)
        ↓
    loss.backward() → 更新 backbone + construct
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 添加 stage 目录到 path
_STAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "stage"))
if _STAGE_ROOT not in sys.path:
    sys.path.insert(0, _STAGE_ROOT)


class Stage1ConstructModel(nn.Module):
    """
    Stage1 + Construct 联合训练模型

    组件：
    - joint_model: JointModel (包含 backbone + line_pooling)
    - construct: ConstructFromFeatures

    训练时 backbone 和 construct 都参与梯度更新。
    """

    def __init__(
        self,
        joint_model: nn.Module,
        construct: nn.Module,
        freeze_visual: bool = True,
        micro_batch_size: int = 8,
    ):
        """
        Args:
            joint_model: 已加载的 JointModel（包含 backbone + line_pooling）
            construct: ConstructFromFeatures 模块
            freeze_visual: 是否冻结视觉编码器
            micro_batch_size: backbone 前向计算的 micro-batch 大小
        """
        super().__init__()
        self.joint_model = joint_model
        self.construct = construct
        self.micro_batch_size = micro_batch_size

        # 冻结视觉编码器（可选）
        if freeze_visual:
            self._freeze_visual_encoder()

    def _freeze_visual_encoder(self):
        """冻结 LayoutXLM 的视觉编码器"""
        backbone = self.joint_model.backbone
        if hasattr(backbone, 'layoutlmv2'):
            visual = backbone.layoutlmv2.visual
            for param in visual.parameters():
                param.requires_grad = False
            logger.info("Froze visual encoder")

    @property
    def backbone(self):
        return self.joint_model.backbone

    @property
    def line_pooling(self):
        return self.joint_model.line_pooling

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ids: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        num_docs: int = 1,
        chunks_per_doc: Optional[list] = None,
        # Construct 标签
        categories: Optional[torch.Tensor] = None,
        reading_orders: Optional[torch.Tensor] = None,
        parent_labels: Optional[torch.Tensor] = None,
        sibling_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: [total_chunks, seq_len]
            bbox: [total_chunks, seq_len, 4]
            attention_mask: [total_chunks, seq_len]
            line_ids: [total_chunks, seq_len]
            image: [total_chunks, C, H, W] or List
            num_docs: 文档数量
            chunks_per_doc: 每个文档的 chunk 数量
            categories: [num_docs, max_lines] 行分类标签
            reading_orders: [num_docs, max_lines] 阅读顺序
            parent_labels: [num_docs, max_lines] 父节点标签
            sibling_labels: [num_docs, max_lines] 左兄弟标签

        Returns:
            包含 loss, parent_logits, sibling_logits 等的字典
        """
        device = input_ids.device

        # Step 1: Backbone 前向（使用 micro-batch，梯度开启）
        hidden_states = self.joint_model.encode_with_micro_batch(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            micro_batch_size=self.micro_batch_size,
            no_grad=False,  # 关键：允许梯度回传
        )

        # 截取文本部分（排除视觉 tokens）
        text_seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :text_seq_len, :]

        # Step 2: Line Pooling 聚合
        line_features, line_mask = self._aggregate_to_lines(
            text_hidden, line_ids, num_docs, chunks_per_doc
        )

        # Step 3: Construct 前向
        max_lines = line_features.shape[1]

        # 准备 reading_orders（如果没有提供）
        if reading_orders is None:
            reading_orders = torch.arange(max_lines, device=device).unsqueeze(0).expand(num_docs, -1)

        # 准备 categories（如果没有提供）
        if categories is None:
            categories = torch.zeros(num_docs, max_lines, dtype=torch.long, device=device)

        construct_outputs = self.construct(
            region_features=line_features,
            categories=categories,
            region_mask=line_mask,
            reading_orders=reading_orders,
            parent_labels=parent_labels,
            sibling_labels=sibling_labels,
        )

        return construct_outputs

    def _aggregate_to_lines(
        self,
        text_hidden: torch.Tensor,
        line_ids: torch.Tensor,
        num_docs: int,
        chunks_per_doc: Optional[list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 token-level hidden states 聚合到 line-level features

        复用 JointModel.line_pooling
        """
        device = text_hidden.device
        total_chunks = text_hidden.shape[0]

        # 默认：每个 chunk 是一个文档
        if chunks_per_doc is None:
            chunks_per_doc = [1] * total_chunks
            num_docs = total_chunks

        doc_features_list = []
        doc_masks_list = []

        chunk_idx = 0
        for doc_idx in range(num_docs):
            n_chunks = chunks_per_doc[doc_idx]
            doc_hidden = text_hidden[chunk_idx:chunk_idx + n_chunks]
            doc_line_ids = line_ids[chunk_idx:chunk_idx + n_chunks]

            # 使用 JointModel 的 line_pooling
            features, mask = self.line_pooling(doc_hidden, doc_line_ids)
            doc_features_list.append(features)
            doc_masks_list.append(mask)

            chunk_idx += n_chunks

        # 填充到相同长度
        max_lines = max(f.shape[0] for f in doc_features_list)
        hidden_size = doc_features_list[0].shape[1]

        line_features = torch.zeros(num_docs, max_lines, hidden_size, device=device)
        line_mask = torch.zeros(num_docs, max_lines, dtype=torch.bool, device=device)

        for doc_idx, (features, mask) in enumerate(zip(doc_features_list, doc_masks_list)):
            n = features.shape[0]
            line_features[doc_idx, :n] = features
            line_mask[doc_idx, :n] = mask

        return line_features, line_mask

    def get_trainable_param_groups(
        self,
        backbone_lr: float = 2e-5,
        construct_lr: float = 5e-5,
    ) -> list:
        """
        获取分层学习率的参数组

        Args:
            backbone_lr: backbone 学习率（通常较小）
            construct_lr: construct 学习率（通常较大）

        Returns:
            optimizer 的 param_groups
        """
        # Backbone 参数（排除冻结的视觉编码器）
        backbone_params = [
            p for p in self.joint_model.backbone.parameters()
            if p.requires_grad
        ]

        # Line pooling 参数
        line_pooling_params = list(self.joint_model.line_pooling.parameters())

        # Construct 参数
        construct_params = list(self.construct.parameters())

        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": line_pooling_params, "lr": backbone_lr, "name": "line_pooling"},
            {"params": construct_params, "lr": construct_lr, "name": "construct"},
        ]


def build_stage1_construct_model(
    joint_checkpoint: str,
    construct_checkpoint: Optional[str] = None,
    num_categories: int = 14,
    num_heads: int = 12,
    num_layers: int = 3,
    dropout: float = 0.1,
    freeze_visual: bool = True,
    micro_batch_size: int = 8,
    device: torch.device = None,
) -> Tuple[Stage1ConstructModel, Any]:
    """
    构建 Stage1 + Construct 联合训练模型

    Args:
        joint_checkpoint: stage JointModel checkpoint 路径
        construct_checkpoint: ConstructFromFeatures checkpoint 路径（可选）
        num_categories: 分类类别数
        num_heads: Construct Transformer heads
        num_layers: Construct Transformer layers
        dropout: dropout rate
        freeze_visual: 是否冻结视觉编码器
        micro_batch_size: backbone micro-batch 大小
        device: 目标设备

    Returns:
        (model, tokenizer) 元组
    """
    from models.build import load_joint_model
    from examples.comp_hrdoc.models.construct_only import (
        ConstructFromFeatures,
        build_construct_from_features,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("Building Stage1 + Construct Model")
    logger.info("=" * 60)

    # 1. 加载 JointModel（包含 backbone + line_pooling）
    logger.info(f"Loading JointModel from: {joint_checkpoint}")
    joint_model, tokenizer = load_joint_model(
        model_path=joint_checkpoint,
        device=device,
    )
    # 设置为训练模式
    joint_model.train()
    logger.info("  JointModel loaded (train mode)")

    # 2. 构建或加载 ConstructFromFeatures
    construct = build_construct_from_features(
        hidden_size=768,
        num_categories=num_categories,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    if construct_checkpoint and os.path.exists(construct_checkpoint):
        logger.info(f"Loading Construct weights from: {construct_checkpoint}")
        # 支持三种格式：
        # 1. 目录下的 construct_model.pt
        # 2. 目录下的 pytorch_model.bin (ConstructFromFeatures 完整权重)
        # 3. 直接指定文件路径
        ckpt_file = None
        if os.path.isdir(construct_checkpoint):
            # 优先查找 construct_model.pt
            construct_pt = os.path.join(construct_checkpoint, "construct_model.pt")
            pytorch_bin = os.path.join(construct_checkpoint, "pytorch_model.bin")
            if os.path.exists(construct_pt):
                ckpt_file = construct_pt
            elif os.path.exists(pytorch_bin):
                ckpt_file = pytorch_bin
        else:
            ckpt_file = construct_checkpoint

        if ckpt_file and os.path.exists(ckpt_file):
            state_dict = torch.load(ckpt_file, map_location="cpu")
            # 过滤掉 RoPE 缓存 buffer
            state_dict = {k: v for k, v in state_dict.items()
                          if 'cos_cached' not in k and 'sin_cached' not in k}
            missing, unexpected = construct.load_state_dict(state_dict, strict=False)
            logger.info(f"  Construct weights loaded from: {ckpt_file}")
            if missing:
                logger.warning(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
            if unexpected:
                logger.warning(f"  Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Unexpected keys: {unexpected}")
        else:
            logger.warning(f"  No weights file found in {construct_checkpoint}, initializing from scratch")
    else:
        logger.info("  Construct initialized from scratch")

    construct = construct.to(device)

    # 3. 组装联合模型
    model = Stage1ConstructModel(
        joint_model=joint_model,
        construct=construct,
        freeze_visual=freeze_visual,
        micro_batch_size=micro_batch_size,
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")
    logger.info("=" * 60)

    return model, tokenizer


def save_stage1_construct_model(
    model: Stage1ConstructModel,
    save_path: str,
    tokenizer: Any = None,
):
    """
    保存 Stage1 + Construct 模型

    保存格式：
    - stage1/: backbone 权重（HuggingFace 格式）
    - construct_model.pt: Construct 权重
    - tokenizer: tokenizer 文件
    """
    os.makedirs(save_path, exist_ok=True)

    # 保存 backbone
    backbone_path = os.path.join(save_path, "stage1")
    os.makedirs(backbone_path, exist_ok=True)
    model.backbone.save_pretrained(backbone_path)
    logger.info(f"Saved backbone to: {backbone_path}")

    # 保存 construct
    construct_path = os.path.join(save_path, "construct_model.pt")
    torch.save(model.construct.state_dict(), construct_path)
    logger.info(f"Saved construct to: {construct_path}")

    # 保存 tokenizer
    if tokenizer:
        tokenizer.save_pretrained(save_path)
        logger.info(f"Saved tokenizer to: {save_path}")


def load_stage1_construct_model(
    model_path: str,
    num_categories: int = 14,
    num_heads: int = 12,
    num_layers: int = 3,
    dropout: float = 0.1,
    device: torch.device = None,
) -> Tuple[Stage1ConstructModel, Any]:
    """
    加载 Stage1 + Construct 联合训练保存的模型（推理用）

    支持的模型路径格式：
    - model_path/stage1/: backbone 权重
    - model_path/pytorch_model.bin 或 construct_model.pt: Construct 权重
    - model_path/tokenizer files

    Args:
        model_path: 模型保存路径
        num_categories: 分类类别数
        num_heads: Construct Transformer heads
        num_layers: Construct Transformer layers
        dropout: dropout rate
        device: 目标设备

    Returns:
        (model, tokenizer) 元组
    """
    from layoutlmft.models.layoutxlm import (
        LayoutXLMForTokenClassification,
        LayoutXLMConfig,
        LayoutXLMTokenizerFast,
    )
    from layoutlmft.data.labels import NUM_LABELS, get_id2label, get_label2id
    from examples.comp_hrdoc.models.construct_only import build_construct_from_features

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info(f"Loading Stage1 + Construct Model from: {model_path}")
    logger.info("=" * 60)

    # ==================== 检查文件存在 ====================
    stage1_path = os.path.join(model_path, "stage1")
    if not os.path.isdir(stage1_path):
        raise ValueError(f"stage1/ subdirectory not found: {stage1_path}")

    # Construct 权重：优先 pytorch_model.bin，其次 construct_model.pt
    construct_weight_path = None
    for name in ["pytorch_model.bin", "construct_model.pt"]:
        candidate = os.path.join(model_path, name)
        if os.path.exists(candidate):
            construct_weight_path = candidate
            break
    if construct_weight_path is None:
        raise ValueError(f"Construct weights not found in: {model_path}")

    # ==================== 加载 Backbone ====================
    logger.info(f"Loading backbone from: {stage1_path}")
    stage1_config = LayoutXLMConfig.from_pretrained(stage1_path)
    stage1_config.num_labels = NUM_LABELS
    stage1_config.id2label = get_id2label()
    stage1_config.label2id = get_label2id()

    backbone = LayoutXLMForTokenClassification.from_pretrained(
        stage1_path,
        config=stage1_config,
    )
    logger.info("  Backbone loaded")

    # ==================== 构建 JointModel ====================
    # 复用完整的 JointModel 以获得正确的 encode_with_micro_batch 实现
    # stage3/stage4 使用 dummy 模块（不会被调用）
    import sys
    STAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "stage"))
    if STAGE_ROOT not in sys.path:
        sys.path.insert(0, STAGE_ROOT)
    from models.joint_model import JointModel
    from train_parent_finder import SimpleParentFinder
    from layoutlmft.models.relation_classifier import MultiClassRelationClassifier

    # 创建 dummy stage3/stage4
    dummy_stage3 = SimpleParentFinder(hidden_size=768, dropout=0.0)
    dummy_stage4 = MultiClassRelationClassifier(hidden_size=768, num_relations=3, dropout=0.0)

    # 创建完整的 JointModel
    joint_model = JointModel(
        stage1_model=backbone,
        stage3_model=dummy_stage3,
        stage4_model=dummy_stage4,
        use_gru=False,
    )

    # ==================== 加载 Construct ====================
    logger.info(f"Loading construct from: {construct_weight_path}")
    construct = build_construct_from_features(
        hidden_size=768,
        num_categories=num_categories,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    state_dict = torch.load(construct_weight_path, map_location="cpu")
    # 过滤掉 RoPE 缓存 buffer
    state_dict = {k: v for k, v in state_dict.items()
                  if 'cos_cached' not in k and 'sin_cached' not in k}
    missing, unexpected = construct.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
    if unexpected:
        logger.warning(f"  Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Unexpected keys: {unexpected}")
    logger.info("  Construct loaded")

    # ==================== 组装模型 ====================
    model = Stage1ConstructModel(
        joint_model=joint_model,
        construct=construct,
        freeze_visual=True,  # 推理时视觉编码器冻结
    )
    model = model.to(device)
    model.eval()

    # ==================== 加载 Tokenizer ====================
    try:
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(model_path)
        logger.info(f"  Tokenizer loaded from: {model_path}")
    except Exception:
        tokenizer = LayoutXLMTokenizerFast.from_pretrained(stage1_path)
        logger.info(f"  Tokenizer loaded from: {stage1_path}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total params: {total_params:,}")
    logger.info("=" * 60)

    return model, tokenizer

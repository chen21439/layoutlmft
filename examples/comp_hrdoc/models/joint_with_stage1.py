"""Joint Model with Stage1 (LayoutXLM) + Construct

将 Stage1 (LayoutXLM 分类) 和 Construct (TOC 构建) 联合训练。

架构:
    LayoutXLM Backbone
           ↓
    LinePooling (token → line)
           ↓
    Line Features [L, 768]
           ↓
    ┌──────┴──────┐
    ↓             ↓
  ClsHead      Construct
  (分类)    (RoPE + TreeHead)
    ↓             ↓
  L_cls       L_construct
           ↓
Total Loss = λ1*L_cls + λ2*L_construct

复用模块（全部在 comp_hrdoc 内部）:
- models/modules/line_pooling.py: LinePooling
- models/heads/classification_head.py: LineClassificationHead
- models/construct.py: ConstructModule
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

# 复用 comp_hrdoc 内部的模块
from .modules.line_pooling import LinePooling
from .modules.attention_pooling import AttentionPooling, extract_section_tokens
from .heads.classification_head import LineClassificationHead
from .construct import ConstructModule, ConstructLoss


class JointModelWithStage1(nn.Module):
    """Stage1 + Construct 联合模型

    将 LayoutXLM backbone 的输出通过 LinePooling 聚合到 line-level，
    然后分别送入分类头和 Construct 模块。

    支持:
    - 端到端联合训练（backbone + cls_head + construct）
    - 冻结 backbone，只训练 cls_head + construct
    - 只训练 construct（使用 GT 类别）
    """

    def __init__(
        self,
        backbone_model,  # LayoutXLMForTokenClassification
        num_classes: int = 14,
        hidden_size: int = 768,
        num_heads: int = 12,
        construct_num_layers: int = 3,
        dropout: float = 0.1,
        lambda_cls: float = 1.0,
        lambda_construct: float = 1.0,
        section_label_id: int = 4,  # section 类的 label id
        freeze_backbone: bool = False,
        cls_dropout: float = 0.1,
        use_token_level_construct: bool = False,  # 是否使用 token-level 特征
        max_tokens_per_section: int = 64,  # 每个 section 保留的最大 token 数
    ):
        """
        Args:
            backbone_model: LayoutXLM 模型（LayoutXLMForTokenClassification）
            num_classes: 分类类别数（默认 14）
            hidden_size: Hidden dimension
            num_heads: Attention heads
            construct_num_layers: Construct Transformer 层数
            dropout: Dropout rate
            lambda_cls: 分类 loss 权重
            lambda_construct: Construct loss 权重
            section_label_id: Section 类的 label ID（用于过滤送入 Construct 的行）
            freeze_backbone: 是否冻结 backbone
            cls_dropout: 分类头 dropout
            use_token_level_construct: 是否使用 token-level 特征构建 TOC
                - False: 使用 line-level 特征（默认，与原有逻辑一致）
                - True: 使用 section 行对应的 token-level 特征 + AttentionPooling
            max_tokens_per_section: 每个 section 保留的最大 token 数（仅 token-level 模式有效）
        """
        super().__init__()

        # ========== Backbone ==========
        self.backbone = backbone_model
        self.hidden_size = hidden_size

        # ========== Stage1: LinePooling + ClsHead ==========
        self.line_pooling = LinePooling(pooling_method="mean")
        self.cls_head = LineClassificationHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=cls_dropout,
        )

        # ========== Token-level Construct 配置 ==========
        self.use_token_level_construct = use_token_level_construct
        self.max_tokens_per_section = max_tokens_per_section

        # 如果使用 token-level，添加 AttentionPooling 模块
        if use_token_level_construct:
            self.section_token_pooling = AttentionPooling(
                hidden_size=hidden_size,
                dropout=dropout,
            )

        # ========== Construct Module ==========
        self.construct = ConstructModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=construct_num_layers,
            dropout=dropout,
        )

        # ========== Loss ==========
        self.construct_loss_fn = ConstructLoss()

        # ========== Config ==========
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_construct = lambda_construct
        self.section_label_id = section_label_id
        self._backbone_frozen = False

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """冻结 backbone 参数"""
        self.backbone.requires_grad_(False)
        self._backbone_frozen = True
        frozen_count = sum(p.numel() for p in self.backbone.parameters())
        print(f"[JointModelWithStage1] Frozen backbone: {frozen_count:,} parameters")

    def unfreeze_backbone(self):
        """解冻 backbone 参数"""
        self.backbone.requires_grad_(True)
        self._backbone_frozen = False
        print("[JointModelWithStage1] Unfrozen backbone")

    def freeze_visual_encoder(self):
        """冻结视觉编码器（ResNet + visual_proj），只训练文本 Transformer

        复用 stage/models/joint_model.py 的实现逻辑
        """
        frozen_count = 0
        # LayoutXLM 的结构: backbone.layoutlmv2.visual / backbone.layoutlmv2.visual_proj
        if hasattr(self.backbone, 'layoutlmv2'):
            layoutlmv2 = self.backbone.layoutlmv2
            if hasattr(layoutlmv2, 'visual'):
                layoutlmv2.visual.requires_grad_(False)
                frozen_count += sum(p.numel() for p in layoutlmv2.visual.parameters())
            if hasattr(layoutlmv2, 'visual_proj'):
                layoutlmv2.visual_proj.requires_grad_(False)
                frozen_count += sum(p.numel() for p in layoutlmv2.visual_proj.parameters())
        self._visual_frozen = True
        print(f"[JointModelWithStage1] Frozen visual encoder: {frozen_count:,} parameters")

    def load_construct_weights(self, construct_checkpoint: str):
        """加载 Construct 模块权重（兼容新旧格式）

        TODO: 模型格式统一后删除旧格式兼容逻辑

        Args:
            construct_checkpoint: Construct 模型 checkpoint 路径
        """
        import os
        model_bin = os.path.join(construct_checkpoint, "pytorch_model.bin")
        if os.path.exists(model_bin):
            state_dict = torch.load(model_bin, map_location="cpu")
        else:
            construct_pt = os.path.join(construct_checkpoint, "construct_model.pt")
            if os.path.exists(construct_pt):
                state_dict = torch.load(construct_pt, map_location="cpu")
            else:
                raise FileNotFoundError(f"No construct weights found at {construct_checkpoint}")

        # 过滤 RoPE 缓存
        state_dict = {k: v for k, v in state_dict.items()
                      if 'cos_cached' not in k and 'sin_cached' not in k}

        # 检测格式并加载
        # TODO: 模型格式统一后删除旧格式分支
        has_construct_prefix = any(k.startswith('construct.') for k in state_dict.keys())
        has_construct_module_prefix = any(k.startswith('construct_module.') for k in state_dict.keys())

        if has_construct_prefix:
            # 新格式 (JointModelWithStage1)：直接加载到整个模型
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"[JointModelWithStage1] Loaded new format checkpoint")
        elif has_construct_module_prefix:
            # 旧格式 (ConstructFromFeatures)：去掉 construct_module. 前缀
            mapped = {k.replace('construct_module.', ''): v
                      for k, v in state_dict.items()
                      if k.startswith('construct_module.')}
            missing, unexpected = self.construct.load_state_dict(mapped, strict=False)
            print(f"[JointModelWithStage1] Loaded old format checkpoint (ConstructFromFeatures, mapped construct_module.*)")
        else:
            # 直接格式：key 已经是 transformer.xxx
            missing, unexpected = self.construct.load_state_dict(state_dict, strict=False)
            print(f"[JointModelWithStage1] Loaded old format checkpoint (direct)")

        if missing:
            print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"[JointModelWithStage1] Loaded construct weights from {construct_checkpoint}")

    def get_param_groups(
        self,
        lr_backbone: float,
        lr_construct: float,
        weight_decay: float = 0.01,
    ) -> List[Dict]:
        """获取分组参数（用于不同学习率）

        Args:
            lr_backbone: Backbone + cls_head 学习率
            lr_construct: Construct 模块学习率
            weight_decay: 权重衰减

        Returns:
            参数组列表，可直接传给 optimizer
        """
        effective_lr_backbone = 0.0 if self._backbone_frozen else lr_backbone

        param_groups = [
            # Backbone
            {
                "params": list(self.backbone.parameters()),
                "lr": effective_lr_backbone,
                "weight_decay": weight_decay if not self._backbone_frozen else 0.0,
                "name": "backbone",
            },
            # Classification head
            {
                "params": list(self.cls_head.parameters()),
                "lr": lr_backbone,  # cls_head 跟随 backbone 学习率
                "weight_decay": weight_decay,
                "name": "cls_head",
            },
            # Construct module
            {
                "params": list(self.construct.parameters()),
                "lr": lr_construct,
                "weight_decay": weight_decay,
                "name": "construct",
            },
        ]

        return param_groups

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ids: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        line_labels: Optional[torch.Tensor] = None,  # [num_docs, max_lines]
        parent_labels: Optional[torch.Tensor] = None,  # [num_docs, max_sections]
        sibling_labels: Optional[torch.Tensor] = None,  # [num_docs, max_sections]
        reading_orders: Optional[torch.Tensor] = None,  # [num_docs, max_sections]
        num_docs: Optional[int] = None,
        chunks_per_doc: Optional[List[int]] = None,
        return_line_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """前向传播（复用 stage/models/joint_model.py 的文档级别处理逻辑）

        Args:
            input_ids: [total_chunks, seq_len]
            bbox: [total_chunks, seq_len, 4]
            attention_mask: [total_chunks, seq_len]
            line_ids: [total_chunks, seq_len] 每个 token 所属的 line_id
            image: [total_chunks, 3, H, W] or List[Tensor] 可选
            line_labels: [num_docs, max_lines] 行级别标签
            parent_labels: [num_docs, max_sections] Section 的 parent 标签
            sibling_labels: [num_docs, max_sections] Section 的 sibling 标签
            reading_orders: [num_docs, max_sections] Section 的阅读顺序
            num_docs: 文档数量
            chunks_per_doc: 每个文档的 chunk 数量列表
            return_line_features: 是否返回 line features

        Returns:
            Dict:
                - loss: 总损失
                - cls_loss: 分类损失
                - construct_loss: Construct 损失
                - cls_logits: [num_docs, max_lines, num_classes]
                - parent_logits: [num_docs, max_sections, max_sections]
                - sibling_logits: [num_docs, max_sections, max_sections]
                - line_features: [num_docs, max_lines, hidden] (如果 return_line_features=True)
        """
        device = input_ids.device
        total_chunks = input_ids.shape[0]
        outputs = {}

        # ========== Step 0: 处理 image list（复用 stage/models/joint_model.py 逻辑） ==========
        if isinstance(image, list) and image:
            image = torch.stack([
                torch.tensor(img) if not isinstance(img, torch.Tensor) else img
                for img in image
            ]).to(device)

        # ========== Step 1: Backbone ==========
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            output_hidden_states=True,
        )
        hidden_states = backbone_outputs.hidden_states[-1]  # [total_chunks, seq_len+visual_len, H]

        # 截取文本部分的 hidden states（排除视觉 tokens）
        text_seq_len = input_ids.shape[1]
        hidden_states = hidden_states[:, :text_seq_len, :]  # [total_chunks, seq_len, H]

        # ========== Step 2: LinePooling（复用 stage/models/joint_model.py 文档级别逻辑） ==========
        # 文档级别模式：每个样本是一个文档，包含多个 chunks
        if num_docs is None or chunks_per_doc is None:
            raise ValueError("num_docs and chunks_per_doc are required for document-level training")

        doc_line_features_list = []
        doc_line_masks_list = []

        chunk_idx = 0
        for doc_idx in range(num_docs):
            num_chunks_in_doc = chunks_per_doc[doc_idx]

            # 收集该文档所有 chunks 的 hidden states 和 line_ids
            doc_hidden = hidden_states[chunk_idx:chunk_idx + num_chunks_in_doc]
            doc_line_ids = line_ids[chunk_idx:chunk_idx + num_chunks_in_doc]

            # 使用 LinePooling 聚合
            doc_features, doc_mask = self.line_pooling(doc_hidden, doc_line_ids)
            doc_line_features_list.append(doc_features)
            doc_line_masks_list.append(doc_mask)

            chunk_idx += num_chunks_in_doc

        # 填充到相同长度
        max_lines = max(f.shape[0] for f in doc_line_features_list)
        batch_line_features = torch.zeros(num_docs, max_lines, self.hidden_size, device=device)
        batch_line_masks = torch.zeros(num_docs, max_lines, dtype=torch.bool, device=device)

        for doc_idx, (features, mask) in enumerate(zip(doc_line_features_list, doc_line_masks_list)):
            num_lines_in_doc = features.shape[0]
            batch_line_features[doc_idx, :num_lines_in_doc] = features
            batch_line_masks[doc_idx, :num_lines_in_doc] = mask

        # ========== Step 3: Classification ==========
        all_cls_logits = []
        all_cls_losses = []

        for doc_idx in range(num_docs):
            num_lines = int(batch_line_masks[doc_idx].sum().item())
            doc_features = batch_line_features[doc_idx, :num_lines]

            cls_logits = self.cls_head(doc_features)
            all_cls_logits.append(cls_logits)

            # 计算分类 loss
            if line_labels is not None:
                sample_labels = line_labels[doc_idx, :num_lines]
                valid_mask = sample_labels != -100
                if valid_mask.any():
                    loss = F.cross_entropy(
                        cls_logits[valid_mask],
                        sample_labels[valid_mask],
                        reduction="mean",
                    )
                    all_cls_losses.append(loss)

        # Padding cls_logits to [num_docs, max_lines, num_classes]
        batch_cls_logits = torch.zeros(num_docs, max_lines, self.num_classes, device=device)
        for doc_idx, cls_logits in enumerate(all_cls_logits):
            num_lines = cls_logits.shape[0]
            batch_cls_logits[doc_idx, :num_lines] = cls_logits

        outputs["cls_logits"] = batch_cls_logits

        # 分类 loss
        cls_loss = torch.tensor(0.0, device=device)
        if all_cls_losses:
            cls_loss = torch.stack(all_cls_losses).mean()
        outputs["cls_loss"] = cls_loss

        if return_line_features:
            outputs["line_features"] = batch_line_features
            outputs["line_masks"] = batch_line_masks

        # ========== Step 4: Filter Sections for Construct ==========
        # 根据预测或 GT 类别过滤 section 行
        if line_labels is not None:
            # 训练时使用 GT 类别（需要 padding 到 max_lines）
            padded_line_labels = torch.full((num_docs, max_lines), -100, dtype=torch.long, device=device)
            for doc_idx in range(num_docs):
                n = min(line_labels.shape[1], max_lines)
                padded_line_labels[doc_idx, :n] = line_labels[doc_idx, :n]
            section_mask = (padded_line_labels == self.section_label_id)
        else:
            # 推理时使用预测类别
            pred_classes = batch_cls_logits.argmax(dim=-1)
            section_mask = (pred_classes == self.section_label_id)

        # 提取 section features
        all_section_features = []
        all_section_masks = []
        max_sections = 0

        # 记录每个文档的 section line indices（用于 token-level 模式）
        doc_section_line_indices = []

        for doc_idx in range(num_docs):
            mask = section_mask[doc_idx] & batch_line_masks[doc_idx]
            indices = mask.nonzero(as_tuple=True)[0]
            num_sections = len(indices)
            max_sections = max(max_sections, num_sections)
            doc_section_line_indices.append(indices)

            if num_sections > 0:
                section_feats = batch_line_features[doc_idx, indices]
            else:
                section_feats = torch.zeros(1, self.hidden_size, device=device)
                num_sections = 1

            all_section_features.append((section_feats, num_sections))

        # Padding sections
        if max_sections == 0:
            max_sections = 1

        padded_section_features = []
        padded_section_masks = []

        for section_feats, num_sections in all_section_features:
            if section_feats.shape[0] < max_sections:
                pad_size = max_sections - section_feats.shape[0]
                section_feats = torch.cat([
                    section_feats,
                    torch.zeros(pad_size, self.hidden_size, device=device)
                ], dim=0)

            mask = torch.zeros(max_sections, dtype=torch.bool, device=device)
            mask[:num_sections] = True

            padded_section_features.append(section_feats)
            padded_section_masks.append(mask)

        batch_section_features = torch.stack(padded_section_features, dim=0)  # [num_docs, max_S, H]
        batch_section_masks = torch.stack(padded_section_masks, dim=0)  # [num_docs, max_S]

        # ========== Step 4.5: Token-level Section Features (可选) ==========
        if self.use_token_level_construct:
            # 从 token-level hidden_states 提取 section 行对应的 tokens
            # 然后使用 AttentionPooling 聚合

            all_token_pooled_features = []

            chunk_idx = 0
            for doc_idx in range(num_docs):
                num_chunks_in_doc = chunks_per_doc[doc_idx]

                # 该文档的 hidden_states 和 line_ids
                doc_hidden = hidden_states[chunk_idx:chunk_idx + num_chunks_in_doc]
                doc_line_ids = line_ids[chunk_idx:chunk_idx + num_chunks_in_doc]

                # 该文档的 section line indices
                section_indices = doc_section_line_indices[doc_idx]

                if len(section_indices) > 0:
                    # 提取 section tokens
                    section_tokens, section_token_mask = extract_section_tokens(
                        hidden_states=doc_hidden,
                        line_ids=doc_line_ids,
                        section_line_indices=section_indices,
                        max_tokens_per_section=self.max_tokens_per_section,
                    )
                    # AttentionPooling 聚合
                    pooled_features = self.section_token_pooling(section_tokens, section_token_mask)
                else:
                    # 无 section，使用零向量
                    pooled_features = torch.zeros(1, self.hidden_size, device=device)

                all_token_pooled_features.append(pooled_features)
                chunk_idx += num_chunks_in_doc

            # Padding token-pooled features
            padded_token_features = []
            for doc_idx, pooled_feats in enumerate(all_token_pooled_features):
                num_sections = pooled_feats.shape[0]
                if num_sections < max_sections:
                    pad_size = max_sections - num_sections
                    pooled_feats = torch.cat([
                        pooled_feats,
                        torch.zeros(pad_size, self.hidden_size, device=device)
                    ], dim=0)
                padded_token_features.append(pooled_feats)

            # 使用 token-level pooled features 替换 line-level features
            batch_section_features = torch.stack(padded_token_features, dim=0)  # [num_docs, max_S, H]

        # ========== Step 5: Construct Module ==========
        # Reading order（如果没有提供，使用顺序索引）
        if reading_orders is None:
            reading_orders = torch.arange(max_sections, device=device).unsqueeze(0).expand(num_docs, -1)

        construct_outputs = self.construct(
            features=batch_section_features,
            reading_order=reading_orders,
            mask=batch_section_masks,
        )

        outputs["parent_logits"] = construct_outputs["parent_logits"]
        outputs["sibling_logits"] = construct_outputs["sibling_logits"]
        outputs["construct_features"] = construct_outputs["construct_features"]

        # ========== Step 6: Construct Loss ==========
        construct_loss = torch.tensor(0.0, device=device)
        if parent_labels is not None and sibling_labels is not None:
            # 调整标签维度
            if parent_labels.shape[1] != max_sections:
                # 需要 padding 或 truncate
                if parent_labels.shape[1] < max_sections:
                    pad_size = max_sections - parent_labels.shape[1]
                    parent_labels = F.pad(parent_labels, (0, pad_size), value=-100)
                    sibling_labels = F.pad(sibling_labels, (0, pad_size), value=-100)
                else:
                    parent_labels = parent_labels[:, :max_sections]
                    sibling_labels = sibling_labels[:, :max_sections]

            construct_loss = self.construct_loss_fn(
                parent_logits=construct_outputs["parent_logits"],
                sibling_logits=construct_outputs["sibling_logits"],
                parent_labels=parent_labels,
                sibling_labels=sibling_labels,
                mask=batch_section_masks,
            )

        outputs["construct_loss"] = construct_loss

        # ========== Total Loss ==========
        total_loss = self.lambda_cls * cls_loss + self.lambda_construct * construct_loss
        outputs["loss"] = total_loss

        return outputs


def build_joint_model_with_stage1(
    layoutxlm_path: str,
    num_classes: int = 14,
    hidden_size: int = 768,
    construct_num_layers: int = 3,
    dropout: float = 0.1,
    lambda_cls: float = 1.0,
    lambda_construct: float = 1.0,
    section_label_id: int = 4,
    freeze_backbone: bool = False,
    use_token_level_construct: bool = False,
    max_tokens_per_section: int = 64,
    device: str = "cuda",
) -> Tuple[JointModelWithStage1, any]:
    """构建 Stage1 + Construct 联合模型

    Args:
        layoutxlm_path: LayoutXLM 模型路径（HuggingFace 或本地）
        use_token_level_construct: 是否使用 token-level 特征构建 TOC
        max_tokens_per_section: 每个 section 保留的最大 token 数
        其他参数同 JointModelWithStage1

    Returns:
        (model, tokenizer)
    """
    # 复用 stage/models/build.py 的路径解析逻辑
    from examples.stage.models.build import resolve_stage1_paths

    stage1_path, tokenizer_path = resolve_stage1_paths(layoutxlm_path)

    from layoutlmft.models.layoutxlm import (
        LayoutXLMForTokenClassification,
        LayoutXLMConfig,
        LayoutXLMTokenizerFast,
    )

    # 加载 LayoutXLM
    config = LayoutXLMConfig.from_pretrained(stage1_path)
    config.num_labels = num_classes

    backbone = LayoutXLMForTokenClassification.from_pretrained(stage1_path, config=config)
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(tokenizer_path)

    # 创建联合模型
    model = JointModelWithStage1(
        backbone_model=backbone,
        num_classes=num_classes,
        hidden_size=hidden_size,
        construct_num_layers=construct_num_layers,
        dropout=dropout,
        lambda_cls=lambda_cls,
        lambda_construct=lambda_construct,
        section_label_id=section_label_id,
        freeze_backbone=freeze_backbone,
        use_token_level_construct=use_token_level_construct,
        max_tokens_per_section=max_tokens_per_section,
    )

    model = model.to(device)

    return model, tokenizer

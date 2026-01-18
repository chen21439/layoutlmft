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

复用模块:
- examples/stage/models/modules/line_pooling.py: LinePooling
- examples/stage/models/heads/classification_head.py: LineClassificationHead
- examples/comp_hrdoc/models/construct.py: ConstructModule
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

# 添加 stage 目录到 path
_STAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "stage"))
if _STAGE_ROOT not in sys.path:
    sys.path.insert(0, _STAGE_ROOT)

# 复用 stage 的共享模块
from models.modules import LinePooling
from models.heads import LineClassificationHead

# 复用 comp_hrdoc 的 Construct 模块
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

        if has_construct_prefix:
            # 新格式：直接加载到整个模型
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"[JointModelWithStage1] Loaded new format checkpoint")
        else:
            # 旧格式：加载到 construct 子模块
            missing, unexpected = self.construct.load_state_dict(state_dict, strict=False)
            print(f"[JointModelWithStage1] Loaded old format checkpoint (ConstructFromFeatures)")

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
        line_labels: Optional[torch.Tensor] = None,  # [batch, max_lines]
        parent_labels: Optional[torch.Tensor] = None,  # [batch, max_sections]
        sibling_labels: Optional[torch.Tensor] = None,  # [batch, max_sections]
        reading_orders: Optional[torch.Tensor] = None,  # [batch, max_sections]
        return_line_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            input_ids: [batch, seq_len]
            bbox: [batch, seq_len, 4]
            attention_mask: [batch, seq_len]
            line_ids: [batch, seq_len] 每个 token 所属的 line_id
            image: [batch, 3, H, W] 可选
            line_labels: [batch, max_lines] 行级别标签
            parent_labels: [batch, max_sections] Section 的 parent 标签
            sibling_labels: [batch, max_sections] Section 的 sibling 标签
            reading_orders: [batch, max_sections] Section 的阅读顺序
            return_line_features: 是否返回 line features

        Returns:
            Dict:
                - loss: 总损失
                - cls_loss: 分类损失
                - construct_loss: Construct 损失
                - cls_logits: [batch, max_lines, num_classes]
                - parent_logits: [batch, max_sections, max_sections]
                - sibling_logits: [batch, max_sections, max_sections]
                - line_features: [batch, max_lines, hidden] (如果 return_line_features=True)
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        outputs = {}

        # ========== Step 1: Backbone ==========
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            output_hidden_states=True,
        )
        hidden_states = backbone_outputs.hidden_states[-1]  # [B, seq_len, H]

        # ========== Step 2: LinePooling ==========
        # 逐样本处理（因为每个样本的行数可能不同）
        all_line_features = []
        all_line_masks = []
        all_cls_logits = []
        all_cls_losses = []
        max_lines = 0

        for b in range(batch_size):
            sample_hidden = hidden_states[b:b+1]
            sample_line_ids = line_ids[b:b+1]

            line_features, line_mask = self.line_pooling(sample_hidden, sample_line_ids)
            num_lines = line_features.shape[0]
            max_lines = max(max_lines, num_lines)

            all_line_features.append(line_features)
            all_line_masks.append(line_mask)

            # ========== Step 3: Classification ==========
            cls_logits = self.cls_head(line_features)
            all_cls_logits.append(cls_logits)

            # 计算分类 loss
            if line_labels is not None:
                sample_labels = line_labels[b, :num_lines]
                valid_mask = sample_labels != -100
                if valid_mask.any():
                    loss = F.cross_entropy(
                        cls_logits[valid_mask],
                        sample_labels[valid_mask],
                        reduction="mean",
                    )
                    all_cls_losses.append(loss)

        # Padding
        padded_line_features = []
        padded_cls_logits = []
        padded_line_masks = []

        for b in range(batch_size):
            lf = all_line_features[b]
            cl = all_cls_logits[b]
            lm = all_line_masks[b]
            num_lines = lf.shape[0]

            if num_lines < max_lines:
                pad_size = max_lines - num_lines
                lf = torch.cat([lf, torch.zeros(pad_size, self.hidden_size, device=device)], dim=0)
                cl = torch.cat([cl, torch.zeros(pad_size, self.num_classes, device=device)], dim=0)
                lm = torch.cat([lm, torch.zeros(pad_size, dtype=torch.bool, device=device)], dim=0)

            padded_line_features.append(lf)
            padded_cls_logits.append(cl)
            padded_line_masks.append(lm)

        batch_line_features = torch.stack(padded_line_features, dim=0)  # [B, max_lines, H]
        batch_cls_logits = torch.stack(padded_cls_logits, dim=0)  # [B, max_lines, C]
        batch_line_masks = torch.stack(padded_line_masks, dim=0)  # [B, max_lines]

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
            # 训练时使用 GT 类别
            section_mask = (line_labels == self.section_label_id)
        else:
            # 推理时使用预测类别
            pred_classes = batch_cls_logits.argmax(dim=-1)
            section_mask = (pred_classes == self.section_label_id)

        # 提取 section features
        # 这里简化处理：假设每个样本的 section 数量相同（用 padding）
        # TODO: 更灵活的处理方式
        all_section_features = []
        all_section_masks = []
        max_sections = 0

        for b in range(batch_size):
            mask = section_mask[b] & batch_line_masks[b]
            indices = mask.nonzero(as_tuple=True)[0]
            num_sections = len(indices)
            max_sections = max(max_sections, num_sections)

            if num_sections > 0:
                section_feats = batch_line_features[b, indices]
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

        batch_section_features = torch.stack(padded_section_features, dim=0)  # [B, max_S, H]
        batch_section_masks = torch.stack(padded_section_masks, dim=0)  # [B, max_S]

        # ========== Step 5: Construct Module ==========
        # Reading order（如果没有提供，使用顺序索引）
        if reading_orders is None:
            reading_orders = torch.arange(max_sections, device=device).unsqueeze(0).expand(batch_size, -1)

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
    device: str = "cuda",
) -> Tuple[JointModelWithStage1, any]:
    """构建 Stage1 + Construct 联合模型

    Args:
        layoutxlm_path: LayoutXLM 模型路径（HuggingFace 或本地）
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
    )

    model = model.to(device)

    return model, tokenizer

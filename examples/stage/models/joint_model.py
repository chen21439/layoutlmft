#!/usr/bin/env python
# coding=utf-8
"""
JointModel - HRDoc 联合训练模型

=== 整体流程 ===

    ┌─────────────────────────────────────────────────────────────────┐
    │  LayoutLM Backbone                                               │
    │  input_ids [B, seq] → hidden_states [B, seq, 768]               │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────┐
                  │   LinePooling (聚合模块)     │
                  │   tokens → lines            │
                  │   [B, seq, 768] → [L, 768]  │
                  └─────────────────────────────┘
                                  │
                                  ▼
                        Line Features [L, 768]
                                  │
          ┌───────────────────────┼───────────────────┐
          ▼                       ▼                   ▼
   ┌────────────┐         ┌────────────┐       ┌────────────┐
   │ Stage 1    │         │ Stage 3    │       │ Stage 4    │
   │ 分类 Head  │         │ Parent     │       │ Relation   │
   │ cls_loss   │         │ parent_loss│       │ rel_loss   │
   └────────────┘         └────────────┘       └────────────┘

=== 损失计算 ===

总 Loss = λ_cls * L_cls + λ_par * L_par + λ_rel * L_rel

其中：
- L_cls: Line-level 分类损失（CrossEntropy）
- L_par: 父节点预测损失
- L_rel: 关系分类损失

此文件只包含模型定义，不包含训练循环、数据加载等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers.modeling_outputs import TokenClassifierOutput

# ImageList 支持（detectron2 或 shim）
try:
    from detectron2.structures import ImageList
except ImportError:
    ImageList = None

# 导入共享模块
import sys
import os

# 使用绝对路径导入 stage/models 模块
_stage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage"))
if _stage_dir not in sys.path:
    sys.path.insert(0, _stage_dir)

try:
    from models.modules import LinePooling
    from models.heads import LineClassificationHead
except ImportError:
    # 备用导入方式：如果上面失败，尝试从当前目录相对导入
    import importlib.util

    # 加载 line_pooling.py
    lp_path = os.path.join(_stage_dir, "models", "modules", "line_pooling.py")
    spec = importlib.util.spec_from_file_location("line_pooling", lp_path)
    line_pooling_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(line_pooling_module)
    LinePooling = line_pooling_module.LinePooling

    # 加载 classification_head.py
    ch_path = os.path.join(_stage_dir, "models", "heads", "classification_head.py")
    spec = importlib.util.spec_from_file_location("classification_head", ch_path)
    classification_head_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classification_head_module)
    LineClassificationHead = classification_head_module.LineClassificationHead


class JointModel(nn.Module):
    """
    联合模型：包含 Stage 1/2/3/4 的所有模块

    架构:
        Backbone (LayoutLM) → LinePooling → [cls_head, stage3, stage4]

    论文公式: L_total = L_cls + α₁·L_par + α₂·L_rel
    """

    def __init__(
        self,
        stage1_model,  # LayoutXLMForTokenClassification（使用其 backbone）
        stage3_model: nn.Module,  # ParentFinderGRU 或 SimpleParentFinder
        stage4_model: nn.Module,  # MultiClassRelationClassifier
        feature_extractor=None,  # 不再使用，保留参数以兼容旧代码
        num_classes: int = 14,  # 分类类别数
        hidden_size: int = 768,  # LayoutLM hidden size
        lambda_cls: float = 1.0,  # Stage 1 分类损失权重
        lambda_parent: float = 1.0,
        lambda_rel: float = 1.0,
        section_parent_weight: float = 1.0,  # section 类型的 parent loss 权重
        use_line_level_cls: bool = True,  # True=使用line-level mean pool (推荐), False=使用原有token-level+投票
        use_focal_loss: bool = True,
        use_gru: bool = True,
        stage1_micro_batch_size: int = 8,
        stage1_no_grad: bool = False,
        freeze_visual: bool = False,
        cls_dropout: float = 0.1,  # 分类头 dropout
        use_gt_class: bool = False,  # 使用 GT class 而不是 Stage1 预测（用于 Stage2 训练）
    ):
        super().__init__()

        # ========== Backbone ==========
        # 使用 LayoutXLM 的 backbone（不使用其内置分类头）
        # stage1_model 是 LayoutXLMForTokenClassification，其 backbone 在 .layoutlmv2
        self.backbone = stage1_model

        # ========== 共享模块 ==========
        # LinePooling: Token-level → Line-level 特征聚合
        self.line_pooling = LinePooling(pooling_method="mean")

        # ========== Stage 1: Line-level 分类头 ==========
        self.cls_head = LineClassificationHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=cls_dropout,
        )

        # ========== Stage 3 & 4 ==========
        self.stage3 = stage3_model
        self.stage4 = stage4_model

        # ========== 损失权重 ==========
        self.lambda_cls = lambda_cls
        self.lambda_parent = lambda_parent
        self.lambda_rel = lambda_rel
        self.section_parent_weight = section_parent_weight
        self.use_gru = use_gru
        self.use_line_level_cls = use_line_level_cls  # 新增：控制分类方式
        self.stage1_micro_batch_size = stage1_micro_batch_size
        self.stage1_no_grad = stage1_no_grad
        self.num_classes = num_classes
        self.use_gt_class = use_gt_class  # 使用 GT class 而不是 Stage1 预测

        # 保存旧接口的引用（兼容性）
        self.stage1 = stage1_model
        self.feature_extractor = feature_extractor

        # 冻结状态记录（不在 __init__ 中立即冻结，由外部调用 freeze 方法）
        self._stage1_frozen = False
        self._visual_frozen = False

        # 兼容旧代码：如果 __init__ 时指定了冻结，立即执行
        if stage1_no_grad:
            self.freeze_stage1()
        elif freeze_visual:
            self.freeze_visual_encoder()

        # 关系分类损失
        if use_focal_loss:
            from layoutlmft.models.relation_classifier import FocalLoss
            self.relation_criterion = FocalLoss(gamma=2.0)
        else:
            self.relation_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def freeze_stage1(self):
        """
        冻结 Stage1 backbone（使用 PyTorch 官方 API）

        PyTorch API: nn.Module.requires_grad_(False) 递归冻结所有参数
        """
        self.stage1.requires_grad_(False)
        self._stage1_frozen = True
        self.stage1_no_grad = True
        frozen_count = sum(p.numel() for p in self.stage1.parameters())
        print(f"[JointModel] Frozen stage1: {frozen_count:,} parameters")

    def freeze_visual_encoder(self):
        """
        冻结视觉编码器（ResNet + visual_proj），只训练 Transformer

        PyTorch API: nn.Module.requires_grad_(False) 递归冻结所有参数
        """
        frozen_count = 0
        if hasattr(self.stage1, 'layoutlmv2'):
            layoutlmv2 = self.stage1.layoutlmv2
            # 使用 PyTorch 官方 API 冻结
            if hasattr(layoutlmv2, 'visual'):
                layoutlmv2.visual.requires_grad_(False)
                frozen_count += sum(p.numel() for p in layoutlmv2.visual.parameters())
            if hasattr(layoutlmv2, 'visual_proj'):
                layoutlmv2.visual_proj.requires_grad_(False)
                frozen_count += sum(p.numel() for p in layoutlmv2.visual_proj.parameters())
        self._visual_frozen = True
        print(f"[JointModel] Frozen visual encoder: {frozen_count:,} parameters")

    # 兼容旧代码
    _freeze_visual_encoder = freeze_visual_encoder

    def get_param_groups(self, lr_stage1: float, lr_stage34: float, weight_decay: float = 0.01):
        """
        返回用于 optimizer 的参数组（始终包含所有模块，保持结构一致）

        路径 A 设计：冻结通过 requires_grad=False 实现，参数仍在 optimizer 中
        - 冻结的参数 grad=None，optimizer 不会更新
        - 这样 resume_from_checkpoint 时参数组结构一致，不会 mismatch

        Args:
            lr_stage1: Stage1 学习率（冻结时设为 0）
            lr_stage34: Stage3/4 学习率
            weight_decay: 权重衰减

        Returns:
            list: optimizer 参数组（固定 4 组：stage1, cls_head, stage3, stage4）
        """
        # 冻结时 lr=0（双保险：requires_grad=False + lr=0）
        effective_lr_stage1 = 0.0 if self._stage1_frozen else lr_stage1

        param_groups = [
            # Stage1: 始终包含，冻结时 lr=0
            {
                "params": list(self.stage1.parameters()),
                "lr": effective_lr_stage1,
                "weight_decay": weight_decay if not self._stage1_frozen else 0.0,
                "name": "stage1",
            },
            # cls_head: 跟随 stage1
            {
                "params": list(self.cls_head.parameters()),
                "lr": effective_lr_stage1,
                "weight_decay": weight_decay if not self._stage1_frozen else 0.0,
                "name": "cls_head",
            },
            # Stage3: 始终可训练
            {
                "params": list(self.stage3.parameters()),
                "lr": lr_stage34,
                "weight_decay": 0.0,
                "name": "stage3",
            },
            # Stage4: 始终可训练
            {
                "params": list(self.stage4.parameters()),
                "lr": lr_stage34,
                "weight_decay": 0.0,
                "name": "stage4",
            },
        ]

        return param_groups

    # 兼容旧代码
    get_trainable_param_groups = get_param_groups

    def encode_with_micro_batch(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor = None,
        micro_batch_size: int = None,
        no_grad: bool = None,
    ) -> torch.Tensor:
        """
        使用 micro-batching 获取 backbone hidden states（可复用的前向计算组件）

        供 forward() 和 predictor.py 调用，避免代码重复。

        Args:
            input_ids: [num_chunks, seq_len]
            bbox: [num_chunks, seq_len, 4]
            attention_mask: [num_chunks, seq_len]
            image: [num_chunks, C, H, W] or List[Tensor] or None
            micro_batch_size: micro-batch 大小（默认使用 self.stage1_micro_batch_size）
            no_grad: 是否禁用梯度（默认使用 self.stage1_no_grad）

        Returns:
            hidden_states: [num_chunks, seq_len+visual_len, hidden_dim]
        """
        device = input_ids.device
        total_chunks = input_ids.shape[0]
        micro_bs = micro_batch_size if micro_batch_size is not None else self.stage1_micro_batch_size
        use_no_grad = no_grad if no_grad is not None else self.stage1_no_grad

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
            """切片 ImageList，返回新的 ImageList（保持在同一设备）"""
            # detectron2 的 ImageList 使用 .tensor，shim 版本使用 .tensors
            if hasattr(img_list, 'tensor'):
                sliced_tensor = img_list.tensor[start:end].to(device)
            else:
                sliced_tensor = img_list.tensors[start:end].to(device)
            sliced_sizes = img_list.image_sizes[start:end]
            return ImageList(sliced_tensor, sliced_sizes)

        if total_chunks <= micro_bs:
            # 小 batch，直接处理
            if image_is_list and image:
                img_tensor = torch.tensor(image[0]) if not isinstance(image[0], torch.Tensor) else image[0]
                img_tensor = img_tensor.unsqueeze(0).to(device)
            else:
                img_tensor = image

            if use_no_grad:
                with torch.no_grad():
                    outputs = _run_backbone(input_ids, bbox, attention_mask, img_tensor)
            else:
                outputs = _run_backbone(input_ids, bbox, attention_mask, img_tensor)
            return outputs.hidden_states[-1]

        # 大 batch，分批处理
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

            if use_no_grad:
                with torch.no_grad():
                    mb_outputs = _run_backbone(mb_input_ids, mb_bbox, mb_attention_mask, mb_image)
            else:
                mb_outputs = _run_backbone(mb_input_ids, mb_bbox, mb_attention_mask, mb_image)

            all_hidden.append(mb_outputs.hidden_states[-1])
            del mb_image

        return torch.cat(all_hidden, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor = None,
        labels: torch.Tensor = None,  # Token-level labels（用于提取 line_labels）
        line_ids: Optional[torch.Tensor] = None,
        line_parent_ids: Optional[torch.Tensor] = None,
        line_relations: Optional[torch.Tensor] = None,
        line_bboxes: Optional[torch.Tensor] = None,
        line_labels: Optional[torch.Tensor] = None,  # 新增：Line-level labels
        num_docs: Optional[int] = None,
        chunks_per_doc: Optional[list] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        === 流程 ===
        1. Backbone: 获取 token-level hidden states
        2. LinePooling: 聚合到 line-level features
        3. Stage 1: Line-level 分类（cls_head）
        4. Stage 3: 父节点预测
        5. Stage 4: 关系分类

        === 参数 ===
        - input_ids: [total_chunks, seq_len]
        - line_ids: [total_chunks, seq_len]，每个 token 的 line_id（-1 表示忽略）
        - line_labels: [num_docs, max_lines]，每行的分类标签
        - line_parent_ids: [num_docs, max_lines]，每行的父节点 ID
        - line_relations: [num_docs, max_lines]，每行与父节点的关系
        """
        device = input_ids.device
        total_chunks = input_ids.shape[0]

        # ==================== Step 1: Backbone 获取 hidden states ====================
        # 使用 encode_with_micro_batch 复用 micro-batching 逻辑
        hidden_states = self.encode_with_micro_batch(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
        )

        # ==================== Step 2: LinePooling 聚合 ====================
        # 截取文本部分的 hidden states（排除视觉 tokens）
        text_seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :text_seq_len, :]

        # 检测模式：页面级别 vs 文档级别
        is_page_level = (num_docs is None or chunks_per_doc is None)

        if is_page_level:
            # ========== 页面级别模式 ==========
            # 每个样本是一个 chunk，直接聚合
            batch_size = total_chunks
            num_docs = batch_size

            # 逐样本聚合（因为每个样本的 line 数量不同）
            doc_line_features_list = []
            doc_line_masks_list = []
            for b in range(batch_size):
                sample_hidden = text_hidden[b:b+1]  # [1, seq_len, H]
                sample_line_ids = line_ids[b:b+1]  # [1, seq_len]
                features, mask = self.line_pooling(sample_hidden, sample_line_ids)
                doc_line_features_list.append(features)
                doc_line_masks_list.append(mask)

            # 填充到相同长度
            max_lines = max(f.shape[0] for f in doc_line_features_list)
            hidden_dim = doc_line_features_list[0].shape[1]
            line_features = torch.zeros(num_docs, max_lines, hidden_dim, device=device)
            line_mask = torch.zeros(num_docs, max_lines, dtype=torch.bool, device=device)

            for b, (features, mask) in enumerate(zip(doc_line_features_list, doc_line_masks_list)):
                num_lines_in_doc = features.shape[0]
                line_features[b, :num_lines_in_doc] = features
                line_mask[b, :num_lines_in_doc] = mask
        else:
            # ========== 文档级别模式 ==========
            # 每个样本是一个文档，包含多个 chunks
            doc_line_features_list = []
            doc_line_masks_list = []

            chunk_idx = 0
            for doc_idx in range(num_docs):
                num_chunks_in_doc = chunks_per_doc[doc_idx]

                # 收集该文档所有 chunks 的 hidden states 和 line_ids
                doc_hidden = text_hidden[chunk_idx:chunk_idx + num_chunks_in_doc]
                doc_line_ids = line_ids[chunk_idx:chunk_idx + num_chunks_in_doc]

                # 使用 LinePooling 聚合
                doc_features, doc_mask = self.line_pooling(doc_hidden, doc_line_ids)
                doc_line_features_list.append(doc_features)
                doc_line_masks_list.append(doc_mask)

                chunk_idx += num_chunks_in_doc

            # 填充到相同长度
            max_lines = max(f.shape[0] for f in doc_line_features_list)
            hidden_dim = doc_line_features_list[0].shape[1]

            line_features = torch.zeros(num_docs, max_lines, hidden_dim, device=device)
            line_mask = torch.zeros(num_docs, max_lines, dtype=torch.bool, device=device)

            for doc_idx, (features, mask) in enumerate(zip(doc_line_features_list, doc_line_masks_list)):
                num_lines_in_doc = features.shape[0]
                line_features[doc_idx, :num_lines_in_doc] = features
                line_mask[doc_idx, :num_lines_in_doc] = mask

        # ==================== Stage 1: 分类 ====================
        # 初始化 outputs 字典
        outputs = {
            "loss": torch.tensor(0.0, device=device),
            "logits": None,
            "cls_loss": torch.tensor(0.0, device=device),
        }

        cls_loss = torch.tensor(0.0, device=device)
        cls_correct = 0
        cls_total = 0

        if self.lambda_cls > 0:
            if self.use_line_level_cls:
                # ========== 新方式：Line-level mean pooling + 分类头 ==========
                # 从 token labels 提取 line_labels
                if line_labels is None and labels is not None:
                    line_labels = torch.full((num_docs, line_features.shape[1]), -100, dtype=torch.long, device=device)

                    if is_page_level:
                        for b in range(num_docs):
                            sample_line_ids = line_ids[b]
                            sample_labels = labels[b] if labels.dim() > 1 else labels
                            num_lines = int(line_mask[b].sum().item())

                            for line_idx in range(num_lines):
                                token_mask = (sample_line_ids == line_idx)
                                if token_mask.any():
                                    first_token_idx = token_mask.nonzero(as_tuple=True)[0][0]
                                    if sample_labels.dim() > 0 and first_token_idx < len(sample_labels):
                                        label = sample_labels[first_token_idx].item()
                                        if label >= 0:
                                            line_labels[b, line_idx] = label
                    else:
                        chunk_idx = 0
                        for doc_idx in range(num_docs):
                            num_chunks = chunks_per_doc[doc_idx]
                            num_lines = int(line_mask[doc_idx].sum().item())

                            doc_line_ids_flat = line_ids[chunk_idx:chunk_idx + num_chunks].reshape(-1)
                            doc_labels_flat = labels[chunk_idx:chunk_idx + num_chunks].reshape(-1)

                            for line_idx in range(num_lines):
                                token_mask = (doc_line_ids_flat == line_idx)
                                if token_mask.any():
                                    first_token_idx = token_mask.nonzero(as_tuple=True)[0][0]
                                    label = doc_labels_flat[first_token_idx].item()
                                    if label >= 0:
                                        line_labels[doc_idx, line_idx] = label

                            chunk_idx += num_chunks

                # Line-level 分类
                all_cls_logits = []
                for b in range(num_docs):
                    sample_features = line_features[b]
                    num_lines = int(line_mask[b].sum().item())

                    if num_lines > 0:
                        valid_features = sample_features[:num_lines]
                        logits = self.cls_head(valid_features)
                        all_cls_logits.append(logits)

                        if line_labels is not None:
                            sample_labels = line_labels[b, :num_lines]
                            valid_indices = sample_labels != -100
                            if valid_indices.any():
                                valid_logits = logits[valid_indices]
                                valid_targets = sample_labels[valid_indices]
                                loss = F.cross_entropy(valid_logits, valid_targets)
                                cls_loss = cls_loss + loss

                                preds = valid_logits.argmax(dim=-1)
                                cls_correct += (preds == valid_targets).sum().item()
                                cls_total += valid_targets.numel()

                if cls_total > 0:
                    cls_loss = cls_loss / num_docs
                    self._cls_acc = cls_correct / cls_total

                if all_cls_logits:
                    max_len = max(l.shape[0] for l in all_cls_logits)
                    padded_logits = torch.zeros(num_docs, max_len, self.num_classes, device=device)
                    for b, logits in enumerate(all_cls_logits):
                        padded_logits[b, :logits.shape[0]] = logits
                    outputs["logits"] = padded_logits

            else:
                # ========== 默认方式：Token-level 分类（使用 LayoutXLM 内置分类头）==========
                # 直接使用 backbone 的 token-level 输出计算损失
                # 注意：hidden_states 已经在 Step 1 获取，这里使用 backbone 的分类头
                if labels is not None:
                    # 重新前向传播获取 token-level logits（因为之前没有传 labels）
                    # 为了效率，直接使用 backbone 的 classifier
                    if hasattr(self.backbone, 'classifier'):
                        token_logits = self.backbone.classifier(hidden_states[:, :text_seq_len, :])
                    elif hasattr(self.backbone, 'dropout') and hasattr(self.backbone, 'classifier'):
                        x = self.backbone.dropout(hidden_states[:, :text_seq_len, :])
                        token_logits = self.backbone.classifier(x)
                    else:
                        # 回退：重新调用 backbone
                        token_logits = None

                    if token_logits is not None:
                        # 计算 token-level 损失
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                        active_loss = attention_mask.view(-1) == 1
                        active_logits = token_logits.view(-1, self.num_classes)[active_loss]
                        active_labels = labels.view(-1)[active_loss]
                        cls_loss = loss_fct(active_logits, active_labels)

                        # 计算准确率
                        valid_mask = active_labels != -100
                        if valid_mask.any():
                            preds = active_logits[valid_mask].argmax(dim=-1)
                            targets = active_labels[valid_mask]
                            cls_correct = (preds == targets).sum().item()
                            cls_total = targets.numel()
                            self._cls_acc = cls_correct / cls_total

                        outputs["logits"] = token_logits

            outputs["cls_loss"] = cls_loss
            outputs["loss"] = outputs["loss"] + cls_loss * self.lambda_cls

        # ==================== Stage 3: Parent Finding ====================
        parent_loss = torch.tensor(0.0, device=device)
        parent_correct = 0
        parent_total = 0
        gru_hidden = None  # GRU 隐状态，用于 Stage 4

        # section 类型 ID（用于加权）
        SECTION_ID = 4

        if self.lambda_parent > 0:
            # 加权 parent loss 相关变量
            weighted_parent_loss = torch.tensor(0.0, device=device)
            total_weight = 0.0
            section_count = 0  # 统计 section 样本数

            if self.use_gru:
                # 准备传入 Stage 3 的 cls_logits
                stage1_cls_logits = None

                if self.use_gt_class and line_labels is not None:
                    # 使用 GT class 构建 one-hot 向量（用于 Stage 2 训练）
                    # line_labels: [num_docs, max_lines]，值为类别索引或 -100
                    max_lines = line_labels.shape[1]
                    gt_one_hot = torch.zeros(num_docs, max_lines, self.num_classes, device=device)
                    for b in range(num_docs):
                        num_lines = int(line_mask[b].sum().item())
                        for line_idx in range(num_lines):
                            label = line_labels[b, line_idx].item()
                            if 0 <= label < self.num_classes:
                                gt_one_hot[b, line_idx, label] = 1.0
                    stage1_cls_logits = gt_one_hot * 10.0  # 放大以模拟高置信度 logits
                    if not hasattr(self, '_logged_using_gt_class'):
                        print(f"[JointModel] 使用 GT class (one-hot) 作为 Stage3 的 cls_logits")
                        self._logged_using_gt_class = True
                elif outputs.get("logits") is not None:
                    if self.use_line_level_cls:
                        # Line-level 模式：直接使用 outputs["logits"]
                        # outputs["logits"]: [num_docs, max_lines, num_classes]
                        stage1_cls_logits = outputs["logits"]
                    else:
                        # Token-level 模式：使用 cls_head 对 line_features 做预测
                        # line_features 已经是 line-level 的，可以直接用
                        with torch.no_grad():  # 避免额外的梯度计算
                            all_cls_logits = []
                            for b in range(num_docs):
                                sample_features = line_features[b]
                                num_lines = int(line_mask[b].sum().item())
                                if num_lines > 0:
                                    valid_features = sample_features[:num_lines]
                                    logits = self.cls_head(valid_features)
                                    all_cls_logits.append(logits)

                            if all_cls_logits:
                                max_len = max(l.shape[0] for l in all_cls_logits)
                                padded_logits = torch.zeros(num_docs, max_len, self.num_classes, device=device)
                                for b, logits in enumerate(all_cls_logits):
                                    padded_logits[b, :logits.shape[0]] = logits
                                stage1_cls_logits = padded_logits

                # 论文对齐：获取 GRU 隐状态用于 Stage 4
                # 传入外部 cls_logits（如果有的话）
                parent_logits, gru_hidden = self.stage3(
                    line_features, line_mask,
                    return_gru_hidden=True,
                    cls_logits=stage1_cls_logits  # 传入 Stage 1 的分类 logits 或 GT one-hot
                )
                # gru_hidden: [num_docs, L+1, gru_hidden_size]，包括 ROOT

                for b in range(num_docs):
                    sample_parent_ids = line_parent_ids[b]
                    sample_mask = line_mask[b]
                    num_lines = int(sample_mask.sum().item())

                    for child_idx in range(num_lines):
                        gt_parent = sample_parent_ids[child_idx].item()

                        if gt_parent == -100:
                            continue
                        if gt_parent >= child_idx:
                            continue

                        target_idx = gt_parent + 1 if gt_parent >= 0 else 0
                        child_logits = parent_logits[b, child_idx + 1, :child_idx + 2]

                        if torch.isinf(child_logits).all():
                            continue

                        child_logits = torch.where(
                            torch.isinf(child_logits),
                            torch.full_like(child_logits, -1e4),
                            child_logits
                        )

                        target = torch.tensor([target_idx], device=device)
                        loss = F.cross_entropy(child_logits.unsqueeze(0), target)

                        if not torch.isnan(loss):
                            # 获取 child 类型，计算权重
                            child_type = -1
                            if line_labels is not None and b < line_labels.shape[0] and child_idx < line_labels.shape[1]:
                                child_type = line_labels[b, child_idx].item()
                            weight = self.section_parent_weight if child_type == SECTION_ID else 1.0
                            if child_type == SECTION_ID:
                                section_count += 1

                            weighted_parent_loss = weighted_parent_loss + loss * weight
                            total_weight += weight
                            parent_total += 1

                        pred_parent = child_logits.argmax().item()
                        if pred_parent == target_idx:
                            parent_correct += 1
            else:
                for b in range(num_docs):
                    sample_features = line_features[b]
                    sample_mask = line_mask[b]
                    sample_parent_ids = line_parent_ids[b]

                    num_lines = sample_mask.sum().item()
                    if num_lines <= 1:
                        continue

                    for child_idx in range(1, int(num_lines)):
                        gt_parent = sample_parent_ids[child_idx].item()

                        if gt_parent < 0 or gt_parent >= child_idx:
                            continue

                        parent_candidates = sample_features[:child_idx]
                        child_feat = sample_features[child_idx]

                        scores = self.stage3(parent_candidates, child_feat)

                        target = torch.tensor([gt_parent], device=device)
                        loss = F.cross_entropy(scores.unsqueeze(0), target)

                        # 获取 child 类型，计算权重
                        child_type = -1
                        if line_labels is not None and b < line_labels.shape[0] and child_idx < line_labels.shape[1]:
                            child_type = line_labels[b, child_idx].item()
                        weight = self.section_parent_weight if child_type == SECTION_ID else 1.0

                        weighted_parent_loss = weighted_parent_loss + loss * weight
                        total_weight += weight

                        pred_parent = scores.argmax().item()
                        if pred_parent == gt_parent:
                            parent_correct += 1
                        parent_total += 1

            # 计算加权平均的 parent loss
            if total_weight > 0:
                parent_loss = weighted_parent_loss / total_weight
                self._parent_acc = parent_correct / parent_total
                # 打印加权统计（每 100 步打印一次）
                if not hasattr(self, '_parent_debug_step'):
                    self._parent_debug_step = 0
                self._parent_debug_step += 1
                if self._parent_debug_step % 100 == 1:
                    other_count = parent_total - section_count
                    print(f"[Parent Loss] section_weight={self.section_parent_weight}, "
                          f"section_samples={section_count}, other_samples={other_count}, "
                          f"total_weight={total_weight:.1f}, line_labels={'available' if line_labels is not None else 'None'}")
            elif parent_total > 0:
                parent_loss = parent_loss / parent_total
                self._parent_acc = parent_correct / parent_total

            outputs["parent_loss"] = parent_loss
            outputs["loss"] = outputs["loss"] + parent_loss * self.lambda_parent

        # ==================== Stage 4: Relation Classification ====================
        rel_loss = torch.tensor(0.0, device=device)
        rel_correct = 0
        rel_total = 0

        # 调试统计
        debug_label_counts = {0: 0, 1: 0, 2: 0}  # connect, contain, equality
        debug_pred_counts = {0: 0, 1: 0, 2: 0}
        debug_skipped_parent = 0
        debug_skipped_label = 0

        if self.lambda_rel > 0 and line_relations is not None:
            if gru_hidden is None:
                gru_hidden = line_features
                use_gru_offset = False
            else:
                use_gru_offset = True

            for b in range(num_docs):
                sample_mask = line_mask[b]
                sample_parent_ids = line_parent_ids[b]
                sample_relations = line_relations[b]

                num_lines = int(sample_mask.sum().item())

                for child_idx in range(num_lines):
                    parent_idx = sample_parent_ids[child_idx].item()
                    rel_label = sample_relations[child_idx].item()

                    if parent_idx < 0 or parent_idx >= num_lines:
                        debug_skipped_parent += 1
                        continue
                    if rel_label == -100:
                        debug_skipped_label += 1
                        continue

                    # 统计 label 分布
                    if rel_label in debug_label_counts:
                        debug_label_counts[rel_label] += 1

                    if use_gru_offset:
                        parent_gru_idx = parent_idx + 1
                        child_gru_idx = child_idx + 1
                        parent_feat = gru_hidden[b, parent_gru_idx]
                        child_feat = gru_hidden[b, child_gru_idx]
                    else:
                        parent_feat = gru_hidden[b, parent_idx]
                        child_feat = gru_hidden[b, child_idx]

                    rel_logits = self.stage4(
                        parent_feat.unsqueeze(0),
                        child_feat.unsqueeze(0),
                    )

                    target = torch.tensor([rel_label], device=device)
                    loss = F.cross_entropy(rel_logits, target)
                    rel_loss = rel_loss + loss

                    pred_rel = rel_logits.argmax(dim=1).item()
                    if pred_rel in debug_pred_counts:
                        debug_pred_counts[pred_rel] += 1
                    if pred_rel == rel_label:
                        rel_correct += 1
                    rel_total += 1

            if rel_total > 0:
                rel_loss = rel_loss / rel_total
                self._rel_acc = rel_correct / rel_total

            # 打印调试信息（每 100 步打印一次）
            if not hasattr(self, '_debug_step'):
                self._debug_step = 0
            self._debug_step += 1
            if self._debug_step % 100 == 1:
                print(f"[Stage4 Debug] rel_total={rel_total}, skipped_parent={debug_skipped_parent}, skipped_label={debug_skipped_label}")
                print(f"[Stage4 Debug] Label dist: connect={debug_label_counts[0]}, contain={debug_label_counts[1]}, equality={debug_label_counts[2]}")
                print(f"[Stage4 Debug] Pred dist:  connect={debug_pred_counts[0]}, contain={debug_pred_counts[1]}, equality={debug_pred_counts[2]}")
                print(f"[Stage4 Debug] use_gru_offset={use_gru_offset}, gru_hidden shape={gru_hidden.shape if gru_hidden is not None else None}")

            outputs["rel_loss"] = rel_loss
            outputs["loss"] = outputs["loss"] + rel_loss * self.lambda_rel

        # 保存完整的 outputs 供 compute_loss 使用
        self._outputs_dict = outputs
        return TokenClassifierOutput(
            loss=outputs["loss"],
            logits=outputs["logits"],
        )

    def _aggregate_document_line_features(
        self,
        doc_hidden: torch.Tensor,
        doc_line_ids: torch.Tensor,
    ) -> tuple:
        """
        从文档的所有 chunks 中聚合 line features（向量化版本）

        Args:
            doc_hidden: [num_chunks, seq_len, hidden_dim]
            doc_line_ids: [num_chunks, seq_len]，每个 token 的全局 line_id

        Returns:
            features: [num_lines, hidden_dim]
            mask: [num_lines]，有效行的 mask
        """
        device = doc_hidden.device
        hidden_dim = doc_hidden.shape[-1]

        # 展平（使用 reshape 兼容非连续 tensor）
        flat_hidden = doc_hidden.reshape(-1, hidden_dim)  # [N, hidden_dim]
        flat_line_ids = doc_line_ids.reshape(-1)  # [N]

        # 获取有效 token（line_id >= 0）
        valid_mask = flat_line_ids >= 0
        valid_line_ids = flat_line_ids[valid_mask]
        valid_hidden = flat_hidden[valid_mask]

        if len(valid_line_ids) == 0:
            return torch.zeros(1, hidden_dim, device=device), torch.zeros(1, dtype=torch.bool, device=device)

        # 获取唯一的 line_id 并排序
        unique_line_ids = valid_line_ids.unique()
        unique_line_ids = unique_line_ids.sort()[0]
        num_lines = len(unique_line_ids)

        # 创建 line_id 到连续索引的映射（向量化）
        # 使用 searchsorted 进行快速映射
        line_indices = torch.searchsorted(unique_line_ids, valid_line_ids)

        # 使用 scatter_add 聚合 features
        line_features = torch.zeros(num_lines, hidden_dim, device=device)
        line_features.scatter_add_(0, line_indices.unsqueeze(1).expand(-1, hidden_dim), valid_hidden)

        # 统计每个 line 的 token 数量
        line_counts = torch.zeros(num_lines, device=device)
        line_counts.scatter_add_(0, line_indices, torch.ones_like(line_indices, dtype=torch.float))

        # 计算平均值
        valid_counts = line_counts.clamp(min=1)
        line_features = line_features / valid_counts.unsqueeze(1)

        # 创建 mask
        line_mask = line_counts > 0

        return line_features, line_mask

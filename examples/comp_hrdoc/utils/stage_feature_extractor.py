#!/usr/bin/env python
# coding=utf-8
"""
Stage 模型特征提取器

使用 stage 目录训练好的 JointModel 提取 line-level 特征，
供 comp_hrdoc 的 Construct 模块使用。

功能：
1. 加载训练好的 stage JointModel
2. 使用 backbone + line_pooling 提取 line-level 特征
3. 返回 line_features [num_docs, max_lines, hidden_size]

使用方式：
    from examples.comp_hrdoc.utils.stage_feature_extractor import StageFeatureExtractor

    extractor = StageFeatureExtractor(checkpoint_path="/path/to/joint/checkpoint")
    line_features, line_mask = extractor.extract_features(batch)
"""

import os
import sys
import logging
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 添加 stage 目录到 path
_STAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "stage"))
if _STAGE_ROOT not in sys.path:
    sys.path.insert(0, _STAGE_ROOT)


class StageFeatureExtractor:
    """
    使用 stage 训练好的模型提取 line-level 特征

    只使用 JointModel 的 backbone + line_pooling，不需要 stage3/4。
    输出特征可直接用于 comp_hrdoc 的 Construct 模块。

    Example:
        >>> extractor = StageFeatureExtractor("/path/to/checkpoint")
        >>> features, mask = extractor.extract_features(batch)
        >>> # features: [num_docs, max_lines, 768]
        >>> # mask: [num_docs, max_lines]
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        config: Any = None,
        max_lines: int = 128,
    ):
        """
        Args:
            checkpoint_path: stage JointModel checkpoint 路径
            device: 计算设备 ("cuda" / "cpu")
            config: 配置对象（可选，用于 tokenizer fallback）
            max_lines: 固定的最大行数（用于 padding）
        """
        self.checkpoint_path = checkpoint_path
        self.max_lines = max_lines
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 加载模型
        self.model, self.tokenizer = self._load_model(checkpoint_path, config)
        self.model.eval()

        logger.info(f"StageFeatureExtractor initialized")
        logger.info(f"  checkpoint: {checkpoint_path}")
        logger.info(f"  device: {self.device}")
        logger.info(f"  hidden_size: 768")

    def _load_model(self, checkpoint_path: str, config: Any = None):
        """加载 stage JointModel

        支持两种路径格式：
        1. 标准 JointModel checkpoint（包含 stage3.pt, stage4.pt 等）
           - 使用 load_joint_model() 加载完整的 JointModel
        2. 联合训练保存的路径（包含 stage1/ 子目录，但没有 stage3.pt）
           - 只加载 backbone + line_pooling
        """
        # 检查是否是联合训练保存的格式
        stage1_subdir = os.path.join(checkpoint_path, "stage1")
        stage3_file = os.path.join(checkpoint_path, "stage3.pt")
        is_joint_training_format = os.path.isdir(stage1_subdir) and not os.path.exists(stage3_file)

        if is_joint_training_format:
            # 联合训练格式：只加载 backbone + line_pooling
            logger.info(f"Detected joint training checkpoint format")
            return self._load_from_joint_training_checkpoint(checkpoint_path)
        else:
            # 标准 JointModel 格式
            from models.build import load_joint_model
            model, tokenizer = load_joint_model(
                model_path=checkpoint_path,
                device=self.device,
                config=config,
            )
            return model, tokenizer

    def _load_from_joint_training_checkpoint(self, checkpoint_path: str):
        """从联合训练 checkpoint 加载 backbone + line_pooling

        联合训练 checkpoint 结构：
        - stage1/: backbone 权重 (HuggingFace 格式)
        - pytorch_model.bin: Construct 权重 (不需要加载)
        - tokenizer files

        复用 JointModel 以获得完整的 encode_with_micro_batch 实现
        """
        from layoutlmft.models.layoutxlm import (
            LayoutXLMForTokenClassification,
            LayoutXLMConfig,
            LayoutXLMTokenizerFast,
        )
        from layoutlmft.data.labels import NUM_LABELS, get_id2label, get_label2id
        from models.joint_model import JointModel

        # 延迟导入 stage3/stage4 相关模块
        import sys
        STAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stage_dir = os.path.join(STAGE_ROOT, "stage")
        if stage_dir not in sys.path:
            sys.path.insert(0, stage_dir)
        from train_parent_finder import SimpleParentFinder
        from layoutlmft.models.relation_classifier import MultiClassRelationClassifier

        stage1_path = os.path.join(checkpoint_path, "stage1")
        logger.info(f"Loading backbone from: {stage1_path}")

        # 加载 backbone
        stage1_config = LayoutXLMConfig.from_pretrained(stage1_path)
        stage1_config.num_labels = NUM_LABELS
        stage1_config.id2label = get_id2label()
        stage1_config.label2id = get_label2id()

        backbone = LayoutXLMForTokenClassification.from_pretrained(
            stage1_path,
            config=stage1_config,
        )

        # 创建 dummy stage3/stage4（只为了能创建 JointModel，不会被使用）
        dummy_stage3 = SimpleParentFinder(hidden_size=768, dropout=0.0)
        dummy_stage4 = MultiClassRelationClassifier(hidden_size=768, num_relations=3, dropout=0.0)

        # 创建完整的 JointModel，复用其 encode_with_micro_batch
        model = JointModel(
            stage1_model=backbone,
            stage3_model=dummy_stage3,
            stage4_model=dummy_stage4,
            use_gru=False,
        )

        # 加载 cls_head 权重（如果存在）
        cls_head_path = os.path.join(checkpoint_path, "cls_head.pt")
        if os.path.exists(cls_head_path):
            logger.info(f"Loading cls_head from: {cls_head_path}")
            cls_head_state = torch.load(cls_head_path, map_location="cpu")
            model.cls_head.load_state_dict(cls_head_state)
            logger.info("  cls_head loaded successfully")
        else:
            logger.warning(f"cls_head weights not found: {cls_head_path}")
            logger.warning("  cls_head will use random initialization")

        # 加载 line_enhancer 权重（论文 4.2.2 行间特征增强，如果存在）
        line_enhancer_path = os.path.join(checkpoint_path, "line_enhancer.pt")
        if os.path.exists(line_enhancer_path):
            if hasattr(model, 'line_enhancer') and model.line_enhancer is not None:
                logger.info(f"Loading line_enhancer from: {line_enhancer_path}")
                line_enhancer_state = torch.load(line_enhancer_path, map_location="cpu")
                model.line_enhancer.load_state_dict(line_enhancer_state)
                logger.info("  line_enhancer loaded successfully")
            else:
                logger.warning(f"line_enhancer.pt exists but model.line_enhancer is None")
                logger.warning("  Model was created without use_line_enhancer=True")
        else:
            if hasattr(model, 'line_enhancer') and model.line_enhancer is not None:
                logger.warning(f"line_enhancer weights not found: {line_enhancer_path}")
                logger.warning("  line_enhancer will use random initialization")

        model = model.to(self.device)

        # 加载 tokenizer
        try:
            tokenizer = LayoutXLMTokenizerFast.from_pretrained(checkpoint_path)
            logger.info(f"Loaded tokenizer from: {checkpoint_path}")
        except Exception:
            tokenizer = LayoutXLMTokenizerFast.from_pretrained(stage1_path)
            logger.info(f"Loaded tokenizer from: {stage1_path}")

        return model, tokenizer

    def extract_features(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ids: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        num_docs: Optional[int] = None,
        chunks_per_doc: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取 line-level 特征（统一接口）

        梯度行为由模型状态自动控制：
        - 训练时：model.train() → 自动有梯度
        - 推理时：model.eval() + with torch.no_grad() → 无梯度

        Args:
            input_ids: [total_chunks, seq_len] tokenized input
            bbox: [total_chunks, seq_len, 4] bounding boxes
            attention_mask: [total_chunks, seq_len]
            line_ids: [total_chunks, seq_len] 每个 token 的 line_id
            image: [total_chunks, C, H, W] 可选图像输入
            num_docs: 文档数量（文档级别模式）
            chunks_per_doc: 每个文档的 chunk 数量列表

        Returns:
            line_features: [num_docs, max_lines, hidden_size]
            line_mask: [num_docs, max_lines] bool mask
        """
        # 移动到设备
        input_ids = input_ids.to(self.device)
        bbox = bbox.to(self.device)
        attention_mask = attention_mask.to(self.device)
        line_ids = line_ids.to(self.device)
        # image 可能是 list（来自 DocumentLevelCollator），由模型内部处理
        if image is not None and isinstance(image, torch.Tensor):
            image = image.to(self.device)

        # Step 1: 获取 backbone hidden states
        # 梯度由模型状态（train/eval）自动控制
        hidden_states = self.model.encode_with_micro_batch(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
        )

        # 截取文本部分（排除视觉 tokens）
        seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :seq_len, :]  # [total_chunks, seq_len, H]

        # Step 2: Line Pooling 聚合
        total_chunks = input_ids.shape[0]
        is_page_level = (num_docs is None or chunks_per_doc is None)

        if is_page_level:
            # 页面级别：每个 chunk 是一个样本
            line_features, line_mask = self._aggregate_page_level(text_hidden, line_ids)
        else:
            # 文档级别：多个 chunks 聚合为一个文档
            line_features, line_mask = self._aggregate_document_level(
                text_hidden, line_ids, num_docs, chunks_per_doc
            )

        # Step 3: Line Feature Enhancement (行间特征增强)
        # 参考论文 4.2.2 Multi-modal Feature Enhancement Module
        # 如果 JointModel 包含 line_enhancer，应用它以增强行间上下文交互
        if hasattr(self.model, 'line_enhancer') and self.model.line_enhancer is not None:
            line_features = self.model.line_enhancer(line_features, line_mask)

        return line_features, line_mask

    def set_train_mode(self, freeze_visual: bool = True):
        """
        设置 backbone 为训练模式（用于 train_stage1）

        Args:
            freeze_visual: 是否冻结视觉编码器
        """
        self.model.train()
        if freeze_visual:
            backbone = self.model.backbone
            if hasattr(backbone, 'layoutlmv2'):
                visual = backbone.layoutlmv2.visual
                for param in visual.parameters():
                    param.requires_grad = False
                visual.eval()  # 保持 BatchNorm 等层为 eval 模式
                logger.info("Froze visual encoder, set to eval mode")

    def set_eval_mode(self):
        """设置 backbone 为评估模式"""
        self.model.eval()

    def get_trainable_params(self) -> list:
        """
        获取可训练参数（用于 train_stage1 的优化器）

        Returns:
            可训练参数列表
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def _aggregate_page_level(
        self,
        text_hidden: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """页面级别聚合：每个 chunk 独立处理"""
        batch_size = text_hidden.shape[0]
        device = text_hidden.device

        doc_line_features_list = []
        doc_line_masks_list = []

        for b in range(batch_size):
            sample_hidden = text_hidden[b:b+1]  # [1, seq_len, H]
            sample_line_ids = line_ids[b:b+1]   # [1, seq_len]
            features, mask = self.model.line_pooling(sample_hidden, sample_line_ids)
            doc_line_features_list.append(features)
            doc_line_masks_list.append(mask)

        # 填充到相同长度
        return self._pad_features(doc_line_features_list, doc_line_masks_list, device)

    def _aggregate_document_level(
        self,
        text_hidden: torch.Tensor,
        line_ids: torch.Tensor,
        num_docs: int,
        chunks_per_doc: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """文档级别聚合：多个 chunks 聚合为一个文档"""
        device = text_hidden.device

        doc_line_features_list = []
        doc_line_masks_list = []

        chunk_idx = 0
        for doc_idx in range(num_docs):
            num_chunks = chunks_per_doc[doc_idx]

            # 收集该文档所有 chunks
            doc_hidden = text_hidden[chunk_idx:chunk_idx + num_chunks]
            doc_line_ids = line_ids[chunk_idx:chunk_idx + num_chunks]

            # 使用 line_pooling 聚合
            features, mask = self.model.line_pooling(doc_hidden, doc_line_ids)
            doc_line_features_list.append(features)
            doc_line_masks_list.append(mask)

            chunk_idx += num_chunks

        return self._pad_features(doc_line_features_list, doc_line_masks_list, device)

    def _pad_features(
        self,
        features_list: list,
        masks_list: list,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将不同长度的特征填充到固定长度 (self.max_lines)"""
        num_docs = len(features_list)
        max_lines = self.max_lines  # 使用固定长度
        hidden_dim = features_list[0].shape[1]

        line_features = torch.zeros(num_docs, max_lines, hidden_dim, device=device)
        line_mask = torch.zeros(num_docs, max_lines, dtype=torch.bool, device=device)

        for b, (features, mask) in enumerate(zip(features_list, masks_list)):
            num_lines = min(features.shape[0], max_lines)  # 截断超长的
            line_features[b, :num_lines] = features[:num_lines]
            line_mask[b, :num_lines] = mask[:num_lines]

        return line_features, line_mask

    def extract_from_batch(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 batch dict 提取特征（便捷方法）

        Args:
            batch: 包含 input_ids, bbox, attention_mask, line_ids 等的字典

        Returns:
            line_features: [num_docs, max_lines, hidden_size]
            line_mask: [num_docs, max_lines]
        """
        return self.extract_features(
            input_ids=batch["input_ids"],
            bbox=batch["bbox"],
            attention_mask=batch["attention_mask"],
            line_ids=batch.get("line_ids"),
            image=batch.get("image"),
            num_docs=batch.get("num_docs"),
            chunks_per_doc=batch.get("chunks_per_doc"),
        )

    @property
    def hidden_size(self) -> int:
        """返回特征维度"""
        return 768

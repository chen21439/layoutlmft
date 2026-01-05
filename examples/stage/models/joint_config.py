#!/usr/bin/env python
# coding=utf-8
"""
JointConfig - HRDoc 联合模型配置

继承 PretrainedConfig，支持 HF 标准的 save_pretrained()/from_pretrained()
"""

from transformers import PretrainedConfig


class JointConfig(PretrainedConfig):
    """
    JointModel 的配置类，继承 HuggingFace PretrainedConfig

    支持:
        - save_pretrained() / from_pretrained() 标准 HF API
        - 自动保存/加载 config.json
        - 与 HF Hub 兼容
    """
    model_type = "joint_hrdoc"

    def __init__(
        self,
        # Stage1 (LayoutXLM) 配置
        stage1_model_name_or_path: str = "microsoft/layoutxlm-base",
        hidden_size: int = 768,
        num_classes: int = 14,

        # Stage3 (ParentFinder) 配置
        use_gru: bool = True,
        gru_hidden_size: int = 512,
        use_soft_mask: bool = True,

        # Stage4 (RelationClassifier) 配置
        num_relations: int = 3,

        # Loss 权重
        lambda_cls: float = 1.0,
        lambda_parent: float = 1.0,
        lambda_rel: float = 1.0,

        # 训练配置
        use_focal_loss: bool = True,
        use_line_level_cls: bool = True,
        cls_dropout: float = 0.1,
        stage3_dropout: float = 0.1,

        # 冻结配置（用于两阶段训练）
        stage1_no_grad: bool = False,
        freeze_visual: bool = False,
        teacher_forcing: bool = True,

        # 其他
        stage1_micro_batch_size: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Stage1 配置
        self.stage1_model_name_or_path = stage1_model_name_or_path
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Stage3 配置
        self.use_gru = use_gru
        self.gru_hidden_size = gru_hidden_size
        self.use_soft_mask = use_soft_mask

        # Stage4 配置
        self.num_relations = num_relations

        # Loss 权重
        self.lambda_cls = lambda_cls
        self.lambda_parent = lambda_parent
        self.lambda_rel = lambda_rel

        # 训练配置
        self.use_focal_loss = use_focal_loss
        self.use_line_level_cls = use_line_level_cls
        self.cls_dropout = cls_dropout
        self.stage3_dropout = stage3_dropout

        # 冻结配置
        self.stage1_no_grad = stage1_no_grad
        self.freeze_visual = freeze_visual
        self.teacher_forcing = teacher_forcing

        # 其他
        self.stage1_micro_batch_size = stage1_micro_batch_size

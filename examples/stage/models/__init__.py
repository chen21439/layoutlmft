"""
stage/models - 模型组件

子模块:
- modules: 可复用的网络组件（LinePooling 等）
- heads: 任务预测头（LineClassificationHead 等）
- stage1_line_level_model: Stage 1 Line-level 分类模型
- joint_model: 联合训练模型 (Stage 1/3/4)
- build: 模型构建工厂函数
"""

from .modules import LinePooling, aggregate_line_features
from .heads import LineClassificationHead
from .stage1_line_level_model import LayoutXLMForLineLevelClassification
from .joint_model import JointModel
from .build import build_joint_model, load_joint_model, get_latest_joint_checkpoint

__all__ = [
    # 共享模块
    "LinePooling",
    "aggregate_line_features",
    "LineClassificationHead",
    # Stage 1 模型
    "LayoutXLMForLineLevelClassification",
    # 联合模型
    "JointModel",
    "build_joint_model",
    "load_joint_model",
    "get_latest_joint_checkpoint",
]

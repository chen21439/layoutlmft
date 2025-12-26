"""
stage/models - 模型组件

子模块:
- modules: 可复用的网络组件（LinePooling 等）
- heads: 任务预测头（LineClassificationHead 等）
- stage1_line_level_model: Stage 1 Line-level 分类模型（复用共享模块）
"""

from .modules import LinePooling, aggregate_line_features
from .heads import LineClassificationHead
from .stage1_line_level_model import LayoutXLMForLineLevelClassification

__all__ = [
    # 共享模块
    "LinePooling",
    "aggregate_line_features",
    "LineClassificationHead",
    # Stage 1 模型（复用共享模块）
    "LayoutXLMForLineLevelClassification",
]

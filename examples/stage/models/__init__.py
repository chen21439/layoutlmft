"""
stage/models - 模型组件

子模块:
- modules: 可复用的网络组件（聚合、注意力等）
- heads: 任务预测头（分类、关系等）
"""

from .modules import LinePooling, aggregate_line_features
from .heads import LineClassificationHead

__all__ = [
    "LinePooling",
    "aggregate_line_features",
    "LineClassificationHead",
]

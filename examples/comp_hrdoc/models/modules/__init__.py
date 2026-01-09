# modules - 可复用网络组件
# attention.py: 注意力机制
# pooling.py: 特征池化
# position.py: 位置编码
# line_transformer.py: 行间特征增强

from .pooling import LineFeatureAggregator, aggregate_document_line_features
from .line_transformer import LineFeatureEnhancer

__all__ = [
    "LineFeatureAggregator",
    "aggregate_document_line_features",
    "LineFeatureEnhancer",
]

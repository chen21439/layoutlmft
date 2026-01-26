# modules - 可复用网络组件
# attention.py: 注意力机制
# pooling.py: 特征池化（慢版本）
# line_pooling.py: 高效 LinePooling（使用 scatter_add）
# attention_pooling.py: Section Token Attention Pooling
# position.py: 位置编码
# line_transformer.py: 行间特征增强

from .pooling import LineFeatureAggregator, aggregate_document_line_features
from .line_pooling import LinePooling, aggregate_line_features
from .line_transformer import LineFeatureEnhancer
from .attention_pooling import AttentionPooling, MultiHeadAttentionPooling, extract_section_tokens

__all__ = [
    "LineFeatureAggregator",
    "aggregate_document_line_features",
    "LinePooling",
    "aggregate_line_features",
    "LineFeatureEnhancer",
    "AttentionPooling",
    "MultiHeadAttentionPooling",
    "extract_section_tokens",
]

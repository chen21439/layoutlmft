# models - 网络结构定义
# embeddings.py: 多模态嵌入 (visual, text, 2D positional, RoPE)
# order.py: Order 模块 (阅读顺序预测)
# construct.py: Construct 模块 (层级结构构建)
# doc_model.py: 统一的 DOC 模型
# order_only.py: 简化版 Order-only 模型
# backbone.py: LayoutXLM 基座封装
# heads.py: 任务预测头
# build.py: 模型构建工厂

from .embeddings import (
    PositionalEmbedding2D,
    RegionTypeEmbedding,
    MultiModalEmbedding,
    RotaryPositionalEmbedding,
    SpatialCompatibilityFeatures,
)

from .order import (
    OrderModule,
    OrderTransformerEncoder,
    ReadingOrderHead,
    RelationTypeHead,
    OrderLoss,
    predict_reading_order,
)

from .construct import (
    ConstructModule,
    RoPETransformerEncoder,
    TreeRelationHead,
    ConstructLoss,
    build_tree_from_predictions,
)

from .doc_model import (
    DOCModel,
    build_doc_model,
    save_doc_model,
    load_doc_model,
    compute_order_accuracy,
    compute_tree_accuracy,
)

from .order_only import (
    OrderOnlyModel,
    build_order_only_model,
    save_order_only_model,
    load_order_only_model,
)

__all__ = [
    # Embeddings
    'PositionalEmbedding2D',
    'RegionTypeEmbedding',
    'MultiModalEmbedding',
    'RotaryPositionalEmbedding',
    'SpatialCompatibilityFeatures',
    # Order Module
    'OrderModule',
    'OrderTransformerEncoder',
    'ReadingOrderHead',
    'RelationTypeHead',
    'OrderLoss',
    'predict_reading_order',
    # Construct Module
    'ConstructModule',
    'RoPETransformerEncoder',
    'TreeRelationHead',
    'ConstructLoss',
    'build_tree_from_predictions',
    # DOC Model
    'DOCModel',
    'build_doc_model',
    'save_doc_model',
    'load_doc_model',
    'compute_order_accuracy',
    'compute_tree_accuracy',
    # Order-only Model
    'OrderOnlyModel',
    'build_order_only_model',
    'save_order_only_model',
    'load_order_only_model',
]

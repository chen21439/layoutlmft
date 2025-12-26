# models - 网络结构定义
# embeddings.py: 多模态嵌入 (visual, text, 2D positional, RoPE)
# intra_region.py: Intra-region Head (行级别后继预测, Section 4.2.3)
# order.py: Order 模块 (阅读顺序预测)
# construct.py: Construct 模块 (层级结构构建)
# doc_model.py: 统一的 DOC 模型
# order_only.py: 简化版 Order-only 模型
# order_from_features.py: 使用预提取特征的 Order 模块
# construct_only.py: 使用预训练 Order 特征的 Construct 模块 (4.4)
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
    InterRegionOrderHead,
    RelationTypeHead,
    OrderLoss,
    predict_reading_order,
    TextRegionAttentionFusion,
    RegionFeatureBuilder,
    DOCPipeline,
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

from .order_from_features import (
    OrderModuleFromFeatures,
    OrderLossFromFeatures,
    build_order_from_features,
    save_order_model,
    load_order_model,
)

from .construct_only import (
    ConstructWithOrderFeatures,
    ConstructFromFeatures,
    build_construct_with_order,
    build_construct_from_features,
    save_construct_model,
    load_construct_model,
    compute_construct_metrics,
    generate_sibling_labels,
)

from .intra_region import (
    IntraRegionHead,
    IntraRegionLoss,
    IntraRegionModule,
    MultiModalLineEncoder,
    UnionFind,
    predict_successors,
    predict_successors_bidirectional,
    group_lines_to_regions,
    DetectModule,
    FeatureProjection,
    SpatialCompatibilityFeatures as IntraSpatialFeatures,
    LogicalRoleHead,
    LogicalRoleLoss,
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
    'InterRegionOrderHead',
    'RelationTypeHead',
    'OrderLoss',
    'predict_reading_order',
    'TextRegionAttentionFusion',
    'RegionFeatureBuilder',
    'DOCPipeline',
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
    # Order from Features
    'OrderModuleFromFeatures',
    'OrderLossFromFeatures',
    'build_order_from_features',
    'save_order_model',
    'load_order_model',
    # Construct-only Model (4.4)
    'ConstructWithOrderFeatures',
    'ConstructFromFeatures',
    'build_construct_with_order',
    'build_construct_from_features',
    'save_construct_model',
    'load_construct_model',
    'compute_construct_metrics',
    'generate_sibling_labels',
    # Intra-region Module (4.2)
    'IntraRegionHead',
    'IntraRegionLoss',
    'IntraRegionModule',
    'MultiModalLineEncoder',
    'UnionFind',
    'predict_successors',
    'predict_successors_bidirectional',
    'group_lines_to_regions',
    'DetectModule',
    'FeatureProjection',
    'IntraSpatialFeatures',
    'LogicalRoleHead',
    'LogicalRoleLoss',
]

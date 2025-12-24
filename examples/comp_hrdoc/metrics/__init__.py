# metrics - 评估指标
#
# 包含 Detect-Order-Construct 三个模块的评估指标:
# - teds.py: TEDS 树编辑距离 (Order, Construct)
# - classification.py: 分类 F1 (Detect)
# - reading_order.py: 阅读顺序 TEDS (Order)
# - common.py: 通用工具

from .teds import (
    TEDSMetric,
    TEDSResult,
    Node,
    tree_edit_distance,
    generate_doc_tree,
    sequence_edit_distance,
    transfer_tree_to_chain,
    split_chain_by_tag,
    min_edit_distance_between_groups,
)

from .classification import (
    ClassificationMetric,
    ClassificationResult,
    CLASS2ID,
    ID2CLASS,
    normalize_class,
    class_to_id,
)

from .reading_order import (
    ReadingOrderMetric,
    ReadingOrderResult,
    IntraRegionOrderMetric,
)

from .common import (
    format_metrics,
    save_metrics,
    load_metrics,
    MetricAggregator,
    compute_pairwise_accuracy,
    compute_successor_accuracy,
    EvaluationReport,
)

__all__ = [
    # TEDS
    'TEDSMetric',
    'TEDSResult',
    'Node',
    'tree_edit_distance',
    'generate_doc_tree',
    'sequence_edit_distance',
    'transfer_tree_to_chain',
    'split_chain_by_tag',
    'min_edit_distance_between_groups',
    # Classification
    'ClassificationMetric',
    'ClassificationResult',
    'CLASS2ID',
    'ID2CLASS',
    'normalize_class',
    'class_to_id',
    # Reading Order
    'ReadingOrderMetric',
    'ReadingOrderResult',
    'IntraRegionOrderMetric',
    # Common
    'format_metrics',
    'save_metrics',
    'load_metrics',
    'MetricAggregator',
    'compute_pairwise_accuracy',
    'compute_successor_accuracy',
    'EvaluationReport',
]

#!/usr/bin/env python
# coding=utf-8
"""
Tasks module for HRDoc training

包含各个子任务的封装：
- semantic_cls: Semantic Classification (SemanticClassificationTask)
- parent_finding: Parent Finding (M_cp, ParentFindingTask)
"""

from .semantic_cls import SemanticClassificationTask
from .parent_finding import (
    ChildParentDistributionMatrix,
    build_child_parent_matrix_from_dataset,
    ParentFindingTask,
)
from .losses import FocalLoss, ClassBalancedLoss, BalancedFocalLoss, get_balanced_loss

__all__ = [
    # Semantic Classification
    "SemanticClassificationTask",
    # Parent Finding
    "ChildParentDistributionMatrix",
    "build_child_parent_matrix_from_dataset",
    "ParentFindingTask",
    # Losses
    "FocalLoss",
    "ClassBalancedLoss",
    "BalancedFocalLoss",
    "get_balanced_loss",
]

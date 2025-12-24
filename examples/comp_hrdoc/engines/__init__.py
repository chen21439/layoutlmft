# engines - 运行骨架
#
# 训练/推理/评估的循环与工程能力
# - evaluator.py: 评估管线

from .evaluator import (
    DOCEvaluator,
    EvaluatorConfig,
    evaluate_doc,
)

__all__ = [
    'DOCEvaluator',
    'EvaluatorConfig',
    'evaluate_doc',
]

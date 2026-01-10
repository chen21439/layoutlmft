# engines - 运行骨架
#
# 训练/推理/评估的循环与工程能力
# - predictor.py: 推理管线
# - evaluator.py: 评估管线

from .predictor import (
    ConstructPredictor,
    decode_construct_outputs,
    convert_to_format_a,
    build_predictions,
    format_result_as_tree,
)

from .evaluator import (
    DOCEvaluator,
    EvaluatorConfig,
    evaluate_doc,
)

__all__ = [
    # predictor
    'ConstructPredictor',
    'decode_construct_outputs',
    'convert_to_format_a',
    'build_predictions',
    'format_result_as_tree',
    # evaluator
    'DOCEvaluator',
    'EvaluatorConfig',
    'evaluate_doc',
]

# tasks - 任务定义层
# base.py: Task 基类接口 (loss/decode/metrics)
# doc_tasks.py: DOC模型任务 (Detect/Order/Construct)

from .base import BaseTask

from .doc_tasks import (
    DetectTask,
    OrderTask,
    ConstructTask,
    DOCTask,
)

__all__ = [
    'BaseTask',
    'DetectTask',
    'OrderTask',
    'ConstructTask',
    'DOCTask',
]

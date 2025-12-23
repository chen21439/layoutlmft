"""Task 基类接口定义"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from torch import Tensor


class BaseTask(ABC):
    """任务基类，定义统一接口"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """计算任务损失

        Args:
            outputs: 模型输出
            targets: 标签
            mask: 有效位置掩码

        Returns:
            loss: 损失值
        """
        pass

    @abstractmethod
    def decode(
        self,
        outputs: Dict[str, Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """解码模型输出为预测结果

        Args:
            outputs: 模型输出

        Returns:
            predictions: 解码后的预测结果
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Dict[str, float]:
        """计算评估指标

        Args:
            predictions: 预测结果
            targets: 真实标签

        Returns:
            metrics: 指标字典
        """
        pass

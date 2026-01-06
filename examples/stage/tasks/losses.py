# coding=utf-8
"""
Balanced Loss Functions for Long-Tailed Classification

Implements:
1. Class-Balanced Loss (Effective Number) - Cui et al., CVPR 2019
2. Logit Adjustment / Balanced Softmax - Menon et al., ICLR 2021
3. Focal Loss - Lin et al., ICCV 2017

Reference:
- Class-Balanced Loss: https://arxiv.org/abs/1901.05555
- Logit Adjustment: https://arxiv.org/abs/2007.07314
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_class_weights(class_counts: List[int], beta: float = 0.9999, mode: str = "effective_number") -> torch.Tensor:
    """
    Compute class weights for balanced loss.

    Args:
        class_counts: Number of samples per class
        beta: Hyperparameter for effective number (default 0.9999)
        mode: Weight computation mode
            - "effective_number": (1 - beta) / (1 - beta^n) - Cui et al.
            - "inverse_freq": 1 / count (normalized)
            - "inverse_sqrt": 1 / sqrt(count) (normalized)

    Returns:
        Tensor of class weights, shape (num_classes,)
    """
    class_counts = np.array(class_counts, dtype=np.float64)
    num_classes = len(class_counts)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1.0)

    if mode == "effective_number":
        # Effective number of samples: E_n = (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
    elif mode == "inverse_freq":
        weights = 1.0 / class_counts
    elif mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(class_counts)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize so that sum = num_classes (average weight = 1)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss using Effective Number of Samples.

    From: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)

    Loss = sum_i w_i * CE(y_i, p_i)
    where w_i = (1 - beta) / (1 - beta^{n_i})
    """

    def __init__(
        self,
        class_counts: List[int],
        beta: float = 0.9999,
        gamma: float = 0.0,  # Focal loss gamma, 0 = no focal
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

        weights = compute_class_weights(class_counts, beta, mode="effective_number")
        self.register_buffer("class_weights", weights)

        logger.info(f"ClassBalancedLoss initialized with beta={beta}, gamma={gamma}")
        logger.info(f"Class weights (first 10): {weights[:10].tolist()}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, num_classes) or (batch * seq_len, num_classes)
            labels: (batch, seq_len) or (batch * seq_len,)

        Returns:
            Scalar loss
        """
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

        # Get valid mask (not ignore_index)
        valid_mask = labels != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # Compute cross-entropy per sample
        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')

        # Apply focal loss modulation if gamma > 0
        if self.gamma > 0:
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
            ce_loss = focal_weight * ce_loss

        # Apply class weights
        weights = self.class_weights[valid_labels]
        weighted_loss = ce_loss * weights

        return weighted_loss.mean()


class LogitAdjustedLoss(nn.Module):
    """
    Logit Adjustment / Balanced Softmax for Long-Tailed Recognition.

    From: "Long-tail learning via logit adjustment" (ICLR 2021)

    Adjusts logits by class prior: logit_adjusted = logit + tau * log(pi)
    where pi is the class prior (frequency).

    This is equivalent to Balanced Softmax when tau=1.
    """

    def __init__(
        self,
        class_counts: List[int],
        tau: float = 1.0,  # Temperature for adjustment, 1.0 = balanced softmax
        ignore_index: int = -100,
    ):
        super().__init__()
        self.tau = tau
        self.ignore_index = ignore_index

        # Compute class prior (frequency)
        class_counts = np.array(class_counts, dtype=np.float64)
        class_counts = np.maximum(class_counts, 1.0)  # Avoid log(0)
        class_prior = class_counts / class_counts.sum()

        # Log prior adjustment: tau * log(pi)
        log_prior = torch.tensor(np.log(class_prior), dtype=torch.float32)
        self.register_buffer("log_prior_adjustment", self.tau * log_prior)

        logger.info(f"LogitAdjustedLoss initialized with tau={tau}")
        logger.info(f"Log prior adjustment (first 10): {self.log_prior_adjustment[:10].tolist()}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, num_classes) or (batch * seq_len, num_classes)
            labels: (batch, seq_len) or (batch * seq_len,)

        Returns:
            Scalar loss
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch_size, seq_len, num_classes = logits.shape
            logits = logits.view(-1, num_classes)
            labels = labels.view(-1)

        # Adjust logits by log prior
        # During training, we subtract log(pi) to counteract class imbalance
        # This encourages the model to predict rare classes more often
        adjusted_logits = logits - self.log_prior_adjustment.unsqueeze(0)

        return F.cross_entropy(adjusted_logits, labels, ignore_index=self.ignore_index)


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection.

    From: "Focal Loss for Dense Object Detection" (ICCV 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,  # Class weights
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

        logger.info(f"FocalLoss initialized with gamma={gamma}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, num_classes) or (batch * seq_len, num_classes)
            labels: (batch, seq_len) or (batch * seq_len,)

        Returns:
            Scalar loss
        """
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

        # Get valid mask
        valid_mask = labels != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # Compute cross-entropy
        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')

        # Compute pt (probability of correct class)
        pt = torch.exp(-ce_loss)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[valid_labels]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        return loss.mean()


class BalancedFocalLoss(nn.Module):
    """
    Combines Class-Balanced weights with Focal Loss.

    This is the CB-Focal variant from the Class-Balanced Loss paper.
    """

    def __init__(
        self,
        class_counts: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

        weights = compute_class_weights(class_counts, beta, mode="effective_number")
        self.register_buffer("class_weights", weights)

        logger.info(f"BalancedFocalLoss initialized with beta={beta}, gamma={gamma}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, num_classes) or (batch * seq_len, num_classes)
            labels: (batch, seq_len) or (batch * seq_len,)

        Returns:
            Scalar loss
        """
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

        valid_mask = labels != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]

        # Cross-entropy
        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')

        # Focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        # Class-balanced weight
        cb_weight = self.class_weights[valid_labels]

        # Combined loss
        loss = cb_weight * focal_weight * ce_loss

        return loss.mean()


def get_balanced_loss(
    loss_type: str,
    class_counts: List[int],
    beta: float = 0.9999,
    gamma: float = 2.0,
    tau: float = 1.0,
    ignore_index: int = -100,
    **kwargs
) -> nn.Module:
    """
    Factory function to create balanced loss.

    Args:
        loss_type: One of "class_balanced", "logit_adjusted", "focal", "balanced_focal", "ce"
        class_counts: Number of samples per class
        beta: Hyperparameter for effective number (class_balanced, balanced_focal)
        gamma: Focal loss gamma (focal, balanced_focal, class_balanced)
        tau: Logit adjustment temperature (logit_adjusted)
        ignore_index: Label index to ignore

    Returns:
        Loss module
    """
    if loss_type == "class_balanced":
        return ClassBalancedLoss(class_counts, beta=beta, gamma=gamma, ignore_index=ignore_index)
    elif loss_type == "logit_adjusted":
        return LogitAdjustedLoss(class_counts, tau=tau, ignore_index=ignore_index)
    elif loss_type == "focal":
        # Compute class weights for alpha
        weights = compute_class_weights(class_counts, beta, mode="effective_number")
        return FocalLoss(alpha=weights, gamma=gamma, ignore_index=ignore_index)
    elif loss_type == "balanced_focal":
        return BalancedFocalLoss(class_counts, beta=beta, gamma=gamma, ignore_index=ignore_index)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

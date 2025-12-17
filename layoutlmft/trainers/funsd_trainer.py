from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from transformers import Trainer


logger = logging.getLogger(__name__)


class FunsdTrainer(Trainer):
    """Extended Trainer with per-class loss monitoring and custom sampler support."""

    def __init__(
        self,
        *args,
        label_list: Optional[List[str]] = None,
        train_sampler: Optional[Sampler] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.label_list = label_list
        self.per_class_loss_accumulator = defaultdict(lambda: {"sum": 0.0, "count": 0})
        self._eval_step_count = 0
        self._custom_train_sampler = train_sampler

    def get_train_dataloader(self) -> DataLoader:
        """Override to use custom sampler if provided."""
        if self._custom_train_sampler is not None:
            # Use custom sampler (e.g., ClassBalancedBatchSampler)
            logger.info("Using custom train sampler for class-balanced batching")
            return DataLoader(
                self.train_dataset,
                batch_sampler=self._custom_train_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            # Use default behavior
            return super().get_train_dataloader()

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.

        NOTE: line_ids is extracted and stored in self._current_line_ids for Callback use,
        but removed from inputs since the model doesn't accept it.
        """
        # 提取 line_ids 供 Callback 使用（行级评估需要）
        # 模型的 forward() 不接受 line_ids 参数，所以需要移除
        if "line_ids" in inputs:
            self._current_line_ids = inputs.pop("line_ids")
        else:
            self._current_line_ids = None

        for k, v in inputs.items():
            if hasattr(v, "to") and hasattr(v, "device"):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def compute_per_class_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute per-class mean loss (cross-entropy).

        Args:
            logits: Model output logits, shape (batch, seq_len, num_labels)
            labels: Ground truth labels, shape (batch, seq_len), -100 for ignored

        Returns:
            Dict mapping class name to mean loss
        """
        if self.label_list is None:
            return {}

        # Flatten
        logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq, num_labels)
        labels_flat = labels.view(-1)  # (batch*seq,)

        # Compute per-sample loss (no reduction)
        loss_per_sample = F.cross_entropy(logits_flat, labels_flat, reduction='none', ignore_index=-100)

        # Group by class
        per_class_loss = {}
        for label_id in range(len(self.label_list)):
            mask = (labels_flat == label_id)
            if mask.sum() > 0:
                class_loss = loss_per_sample[mask].mean().item()
                label_name = self.label_list[label_id]
                # Extract base class (remove B-/I- prefix)
                base_name = label_name[2:] if label_name.startswith(('B-', 'I-')) else label_name
                if base_name not in per_class_loss:
                    per_class_loss[base_name] = {"sum": 0.0, "count": 0}
                per_class_loss[base_name]["sum"] += class_loss * mask.sum().item()
                per_class_loss[base_name]["count"] += mask.sum().item()

        # Compute mean
        result = {}
        for cls, stats in per_class_loss.items():
            if stats["count"] > 0:
                result[cls] = stats["sum"] / stats["count"]

        return result

    def evaluation_loop(self, *args, **kwargs):
        """Override to reset per-class loss accumulator before evaluation."""
        self.per_class_loss_accumulator = defaultdict(lambda: {"sum": 0.0, "count": 0})
        self._eval_step_count = 0
        return super().evaluation_loop(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to compute per-class loss during evaluation."""
        # Get the standard prediction step result
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # Compute per-class loss if we have logits and labels
        if logits is not None and labels is not None and self.label_list is not None:
            # logits might be tuple, get first element
            if isinstance(logits, tuple):
                logits_tensor = logits[0]
            else:
                logits_tensor = logits

            per_class = self.compute_per_class_loss(logits_tensor, labels)
            for cls, cls_loss in per_class.items():
                self.per_class_loss_accumulator[cls]["sum"] += cls_loss
                self.per_class_loss_accumulator[cls]["count"] += 1

            self._eval_step_count += 1

            # Log every 100 steps during eval
            if self._eval_step_count % 100 == 0:
                key_classes = ["MAIL", "FIG", "TAB", "SEC1", "SEC2", "PARA"]
                loss_strs = []
                for cls in key_classes:
                    if self.per_class_loss_accumulator[cls]["count"] > 0:
                        mean_loss = self.per_class_loss_accumulator[cls]["sum"] / self.per_class_loss_accumulator[cls]["count"]
                        loss_strs.append(f"{cls}:{mean_loss:.3f}")
                if loss_strs:
                    logger.info(f"[Eval step {self._eval_step_count}] Per-class loss: {', '.join(loss_strs)}")

        return loss, logits, labels

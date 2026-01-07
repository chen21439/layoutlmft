#!/usr/bin/env python
# coding=utf-8
"""
JointTrainer - 联合训练 Trainer

继承 HuggingFace Trainer，实现：
1. create_optimizer: 为 Stage 1 和 Stage 3/4 设置不同学习率
2. compute_loss: 直接使用 JointModel 的 forward 返回的 loss
3. log: 记录多个 loss（cls_loss, parent_loss, rel_loss）
4. _save: 分别保存各 Stage 权重
"""

import logging
import os
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from transformers import Trainer

logger = logging.getLogger(__name__)


class JointTrainer(Trainer):
    """
    联合训练 Trainer，继承 HuggingFace Trainer

    自定义内容：
    1. create_optimizer: 为 Stage 1 和 Stage 3/4 设置不同学习率
    2. compute_loss: 直接使用 JointModel 的 forward 返回的 loss
    3. log: 记录多个 loss（cls_loss, parent_loss, rel_loss）
    """

    def __init__(
        self,
        model_args=None,
        learning_rate_stage34: float = 5e-4,
        document_level: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_args = model_args
        self.learning_rate_stage34 = learning_rate_stage34
        self.document_level = document_level

        # 用于记录各 loss
        self._current_loss_dict = {}

    def create_optimizer(self):
        """
        创建优化器，为不同模块设置不同学习率

        路径 A 设计：始终保持 4 个参数组（stage1, cls_head, stage3, stage4）
        - 冻结的模块通过 requires_grad=False + lr=0 实现不更新
        - 参数组结构不变，确保 resume_from_checkpoint 时 optimizer state 能正确加载
        """
        if self.optimizer is not None:
            return self.optimizer

        # 获取参数组（固定 4 组，冻结模块 lr=0）
        optimizer_grouped_parameters = self.model.get_param_groups(
            lr_stage1=self.args.learning_rate,
            lr_stage34=self.learning_rate_stage34,
            weight_decay=self.args.weight_decay,
        )

        # 打印参数组信息
        total_params = sum(sum(p.numel() for p in g["params"]) for g in optimizer_grouped_parameters)
        trainable_params = sum(sum(p.numel() for p in g["params"] if p.requires_grad) for g in optimizer_grouped_parameters)
        logger.info(f"[Optimizer] Creating AdamW with {len(optimizer_grouped_parameters)} param groups")
        logger.info(f"[Optimizer] Total params: {total_params:,}, Trainable: {trainable_params:,}")
        for g in optimizer_grouped_parameters:
            name = g.get("name", "unknown")
            num_params = sum(p.numel() for p in g["params"])
            num_trainable = sum(p.numel() for p in g["params"] if p.requires_grad)
            frozen_tag = " (frozen)" if g["lr"] == 0 else ""
            logger.info(f"  {name}: lr={g['lr']}, params={num_params:,}, trainable={num_trainable:,}{frozen_tag}")

        self.optimizer = AdamW(optimizer_grouped_parameters)
        return self.optimizer

    def training_step(self, model, inputs):
        """覆盖 training_step 以捕获 OOM 时的文档信息"""
        # 只记录轻量信息，避免引用 GPU tensor
        self._current_doc_name = inputs.get("doc_name", None)
        self._current_doc_id = inputs.get("doc_id", None)

        try:
            return super().training_step(model, inputs)
        except torch.cuda.OutOfMemoryError:
            if self.is_world_process_zero():
                self.log({
                    "oom_doc_name": str(self._current_doc_name),
                    "oom_doc_id": str(self._current_doc_id),
                })
            raise
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                if self.is_world_process_zero():
                    self.log({
                        "oom_doc_name": str(self._current_doc_name),
                        "oom_doc_id": str(self._current_doc_id),
                    })
            raise

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算 loss，JointModel.forward 已经返回组合后的 loss"""

        try:
            outputs = model(**inputs)
        except Exception as e:
            # 记录出错时的文档信息（先转到 CPU 避免 CUDA 错误）
            logger.error(f"[ERROR] Forward pass failed at step {self.state.global_step}")
            logger.error(f"[ERROR] Input keys: {inputs.keys()}")
            if "doc_id" in inputs:
                logger.error(f"[ERROR] doc_id: {inputs['doc_id']}")
            # 包在 try 中避免 CUDA 错误时的二次报错
            try:
                if "input_ids" in inputs:
                    ids = inputs["input_ids"].cpu()
                    logger.error(f"[ERROR] input_ids shape: {ids.shape}, min: {ids.min()}, max: {ids.max()}")
                if "bbox" in inputs:
                    bbox = inputs["bbox"].cpu()
                    logger.error(f"[ERROR] bbox shape: {bbox.shape}")
                    logger.error(f"[ERROR] bbox x min/max: {bbox[:,:,0].min()}/{bbox[:,:,0].max()}, y min/max: {bbox[:,:,1].min()}/{bbox[:,:,1].max()}")
                    logger.error(f"[ERROR] bbox x2 min/max: {bbox[:,:,2].min()}/{bbox[:,:,2].max()}, y2 min/max: {bbox[:,:,3].min()}/{bbox[:,:,3].max()}")
                    # 检查是否有超出 0-1000 范围的坐标
                    if bbox.max() > 1000 or bbox.min() < 0:
                        logger.error(f"[ERROR] bbox has out-of-range values! Valid range: 0-1000")
                if "image" in inputs:
                    img = inputs["image"]
                    if hasattr(img, 'shape'):
                        logger.error(f"[ERROR] image shape: {img.shape}, dtype: {img.dtype}")
                    elif hasattr(img, 'tensor'):
                        logger.error(f"[ERROR] image (ImageList) tensor shape: {img.tensor.shape}")
                    elif isinstance(img, list):
                        logger.error(f"[ERROR] image is list, len={len(img)}")
                        if img:
                            first_img = img[0]
                            if hasattr(first_img, 'shape'):
                                logger.error(f"[ERROR] first image shape: {first_img.shape}")
                    else:
                        logger.error(f"[ERROR] image type: {type(img)}")
                if "line_ids" in inputs:
                    line_ids = inputs["line_ids"]
                    if hasattr(line_ids, 'shape'):
                        line_ids_cpu = line_ids.cpu()
                        logger.error(f"[ERROR] line_ids shape: {line_ids_cpu.shape}, max: {line_ids_cpu.max()}")
                    else:
                        logger.error(f"[ERROR] line_ids type: {type(line_ids)}")
            except Exception as log_e:
                logger.error(f"[ERROR] Failed to log input details: {log_e}")
            raise e
        loss = outputs.loss  # TokenClassifierOutput 使用属性访问

        # 从模型实例获取完整的 outputs 字典（包含各阶段 loss）
        outputs_dict = getattr(model, "_outputs_dict", {})

        # 记录各 loss 用于日志
        self._current_loss_dict = {
            "cls_loss": outputs_dict.get("cls_loss", torch.tensor(0.0)).item() if "cls_loss" in outputs_dict else 0.0,
        }
        if "parent_loss" in outputs_dict:
            self._current_loss_dict["parent_loss"] = outputs_dict["parent_loss"].item()
        if "rel_loss" in outputs_dict:
            self._current_loss_dict["rel_loss"] = outputs_dict["rel_loss"].item()
        # accuracy 从模型实例属性读取
        if hasattr(model, "_parent_acc"):
            self._current_loss_dict["parent_acc"] = model._parent_acc
        if hasattr(model, "_rel_acc"):
            self._current_loss_dict["rel_acc"] = model._rel_acc

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """扩展日志，添加各阶段 loss 和 stage34 学习率"""

        # 添加各 loss 到日志
        if self._current_loss_dict:
            logs.update(self._current_loss_dict)

        # 添加 stage34 的学习率（param_groups[2] 是 stage3）
        if self.optimizer is not None and len(self.optimizer.param_groups) >= 3:
            lr_stage34 = self.optimizer.param_groups[2]["lr"]
            logs["lr_stage34"] = lr_stage34
            # 如果 learning_rate 是 0（stage1 冻结），用 stage34 的 lr 替代显示
            if logs.get("learning_rate", 0) == 0:
                logs["learning_rate"] = lr_stage34

        super().log(logs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写评估方法

        文档级别模式下跳过 Trainer 内置的 prediction_loop（形状不一致会失败），
        只触发 callbacks（E2EEvaluationCallback 会运行正确的评估）。
        """
        # 判断是否需要跳过 prediction_loop
        # 1. document_level 模式：输出形状不一致，Trainer 内置评估会失败
        # 2. stage34 模式：lambda_cls=0 导致 logits=None，prediction_loop 会报 IndexError
        #    （TODO: 此分支用于排查 page-level vs document-level 问题，排查完成后可考虑删除）
        is_stage34 = self.model_args and getattr(self.model_args, 'mode', None) == "stage34"
        skip_prediction_loop = self.document_level or is_stage34

        if skip_prediction_loop:
            # 跳过 prediction_loop，直接调用 callbacks
            self.model.eval()

            # 创建共享 metrics dict，让 callback 写入评估结果
            metrics = {}

            # 手动遍历 callbacks 并调用 on_evaluate（传递 model 和 metrics）
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, 'on_evaluate'):
                    callback.on_evaluate(
                        self.args, self.state, self.control,
                        model=self.model, metrics=metrics
                    )

            # 添加 eval_ 前缀（HF 规范）
            prefixed_metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

            self.model.train()
            return prefixed_metrics
        else:
            # 页面级别 + stage1 模式：正常评估
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """保存模型，分别保存各 Stage + Trainer 状态"""

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Stage 1: 保存完整模型（config + 权重），可独立加载
        stage1_dir = os.path.join(output_dir, "stage1")
        self.model.stage1.save_pretrained(stage1_dir)

        # Stage 3/4: 保存为 PyTorch 格式（用于单独加载推理，续训时由 pytorch_model.bin 恢复）
        torch.save(self.model.stage3.state_dict(), os.path.join(output_dir, "stage3.pt"))
        torch.save(self.model.stage4.state_dict(), os.path.join(output_dir, "stage4.pt"))

        # 保存完整模型状态（标准格式，供 Trainer resume 使用）
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        # 保存 tokenizer 到根目录（两种格式：legacy + tokenizer.json）
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir, legacy_format=True)   # sentencepiece.bpe.model
                self.tokenizer.save_pretrained(output_dir, legacy_format=False)  # tokenizer.json
                logger.info(f"Tokenizer saved to {output_dir}")
            except Exception as e:
                logger.warning(f"Failed to save tokenizer: {e}")
                logger.warning("Tokenizer files may be incomplete, but model weights are saved correctly")
        else:
            logger.warning("Tokenizer is None, skipping tokenizer save")

        # 保存 trainer_state.json（用于续训）
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        logger.info(f"Model saved to {output_dir}")

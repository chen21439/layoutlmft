#!/usr/bin/env python
# coding=utf-8
"""
JointModel - HRDoc 联合训练模型

将 Stage 1/2/3/4 组合为一个端到端模型：
1. Stage 1: LayoutXLM 分类 (产生分类 loss + hidden states)
2. Stage 2: 从 hidden states 提取 line-level 特征
3. Stage 3: ParentFinder 训练 (产生 parent loss)
4. Stage 4: RelationClassifier 训练 (产生 relation loss)

总 Loss = λ1 * L_cls + λ2 * L_par + λ3 * L_rel (论文公式)

此文件只包含模型定义，不包含训练循环、数据加载等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers.modeling_outputs import TokenClassifierOutput


class JointModel(nn.Module):
    """
    联合模型：包含 Stage 1/2/3/4 的所有模块

    论文公式: L_total = L_cls + α₁·L_par + α₂·L_rel
    """

    def __init__(
        self,
        stage1_model,  # LayoutXLMForTokenClassification
        stage3_model: nn.Module,  # ParentFinderGRU 或 SimpleParentFinder
        stage4_model: nn.Module,  # MultiClassRelationClassifier
        feature_extractor,  # LineFeatureExtractor
        lambda_cls: float = 1.0,
        lambda_parent: float = 1.0,
        lambda_rel: float = 1.0,
        use_focal_loss: bool = True,
        use_gru: bool = False,
        stage1_micro_batch_size: int = 8,  # Stage1 micro-batch 大小，防止显存爆炸
        stage1_no_grad: bool = False,  # 是否对 Stage1 使用 no_grad（节省显存但不反传）
    ):
        super().__init__()

        self.stage1 = stage1_model
        self.stage3 = stage3_model
        self.stage4 = stage4_model
        self.feature_extractor = feature_extractor

        self.lambda_cls = lambda_cls
        self.lambda_parent = lambda_parent
        self.lambda_rel = lambda_rel
        self.use_gru = use_gru
        self.stage1_micro_batch_size = stage1_micro_batch_size
        self.stage1_no_grad = stage1_no_grad

        # 如果冻结 Stage 1，同时冻结其参数（不计算梯度）
        if stage1_no_grad:
            for param in self.stage1.parameters():
                param.requires_grad = False

        # 关系分类损失
        if use_focal_loss:
            from layoutlmft.models.relation_classifier import FocalLoss
            self.relation_criterion = FocalLoss(gamma=2.0)
        else:
            self.relation_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor = None,
        labels: torch.Tensor = None,
        line_ids: Optional[torch.Tensor] = None,
        line_parent_ids: Optional[torch.Tensor] = None,
        line_relations: Optional[torch.Tensor] = None,
        line_bboxes: Optional[torch.Tensor] = None,
        num_docs: Optional[int] = None,
        chunks_per_doc: Optional[list] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，返回 loss 和各阶段输出

        文档级别处理：
        - input_ids: [total_chunks, seq_len]，所有文档的 chunks 展平
        - line_ids: [total_chunks, seq_len]，每个 token 的全局 line_id
        - line_parent_ids: [num_docs, max_lines]，文档级别的 parent_ids
        - line_relations: [num_docs, max_lines]，文档级别的 relations
        - num_docs: batch 中的文档数量
        - chunks_per_doc: 每个文档的 chunk 数量列表
        """
        device = input_ids.device
        total_chunks = input_ids.shape[0]

        # 检查 image 是 list 还是 tensor
        # Collator 现在保持 image 为 list，避免一次性加载所有图片到 GPU
        image_is_list = isinstance(image, list) if image is not None else False

        # ==================== Stage 1: Classification (逐 chunk 处理) ====================
        # 使用配置的 micro_batch_size，冻结 Stage 1 时可以用更大的 batch
        micro_bs = self.stage1_micro_batch_size

        if total_chunks <= micro_bs:
            # 小 batch，直接处理
            if image_is_list and image:
                # 将单张图片转为 tensor 并移到 GPU
                img_tensor = torch.tensor(image[0]) if not isinstance(image[0], torch.Tensor) else image[0]
                img_tensor = img_tensor.unsqueeze(0).to(device)
            else:
                img_tensor = image
            stage1_outputs = self.stage1(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                image=img_tensor,
                labels=labels,
                output_hidden_states=True,
            )
            cls_loss = stage1_outputs.loss
            logits = stage1_outputs.logits
            hidden_states = stage1_outputs.hidden_states[-1]
        else:
            # 大 batch，分批处理
            all_logits = []
            all_hidden = []
            total_cls_loss = 0.0
            num_micro_batches = 0

            for start_idx in range(0, total_chunks, micro_bs):
                end_idx = min(start_idx + micro_bs, total_chunks)

                # 切分 micro-batch
                mb_input_ids = input_ids[start_idx:end_idx]
                mb_bbox = bbox[start_idx:end_idx]
                mb_attention_mask = attention_mask[start_idx:end_idx]
                mb_labels = labels[start_idx:end_idx] if labels is not None else None

                # 图片按需加载到 GPU（关键：避免 OOM）
                if image_is_list and image:
                    mb_images = image[start_idx:end_idx]
                    mb_image = torch.stack([
                        torch.tensor(img) if not isinstance(img, torch.Tensor) else img
                        for img in mb_images
                    ]).to(device)
                elif image is not None:
                    mb_image = image[start_idx:end_idx]
                else:
                    mb_image = None

                # 根据配置决定是否使用 no_grad
                if self.stage1_no_grad:
                    with torch.no_grad():
                        mb_outputs = self.stage1(
                            input_ids=mb_input_ids,
                            bbox=mb_bbox,
                            attention_mask=mb_attention_mask,
                            image=mb_image,
                            labels=mb_labels,
                            output_hidden_states=True,
                        )
                else:
                    mb_outputs = self.stage1(
                        input_ids=mb_input_ids,
                        bbox=mb_bbox,
                        attention_mask=mb_attention_mask,
                        image=mb_image,
                        labels=mb_labels,
                        output_hidden_states=True,
                    )

                all_logits.append(mb_outputs.logits)
                all_hidden.append(mb_outputs.hidden_states[-1])

                if mb_outputs.loss is not None:
                    total_cls_loss = total_cls_loss + mb_outputs.loss
                    num_micro_batches += 1

                # 释放当前 micro-batch 的图片显存
                del mb_image

            # 合并结果
            logits = torch.cat(all_logits, dim=0)
            hidden_states = torch.cat(all_hidden, dim=0)
            cls_loss = total_cls_loss / max(num_micro_batches, 1)

        # 当冻结 Stage 1 时，cls_loss 只用于监控，不加入总 loss
        if self.stage1_no_grad:
            outputs = {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),  # 初始化为0，后续加 parent_loss/rel_loss
                "cls_loss": cls_loss.detach() if cls_loss is not None else torch.tensor(0.0),  # 仅监控
                "logits": logits,
            }
        else:
            outputs = {
                "loss": cls_loss * self.lambda_cls,
                "cls_loss": cls_loss,
                "logits": logits,
            }

        # 如果没有 line 信息或文档信息，直接返回
        if line_ids is None or line_parent_ids is None:
            return TokenClassifierOutput(
                loss=outputs["loss"],
                logits=logits,
            )

        # ==================== Stage 2: Feature Extraction ====================
        text_seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :text_seq_len, :]

        # 检测模式：页面级别 vs 文档级别
        is_page_level = (num_docs is None or chunks_per_doc is None)

        if is_page_level:
            # ========== 页面级别模式（快速训练）==========
            # 每个样本是一个 chunk，直接使用 feature_extractor
            # 这与历史版本的处理方式一致
            batch_size = total_chunks
            line_features, line_mask = self.feature_extractor.extract_line_features(
                text_hidden, line_ids, pooling="mean"
            )
            num_docs = batch_size  # 用于后续循环
        else:
            # ========== 文档级别模式（用于推理）==========
            # 每个样本是一个文档，包含多个 chunks，需要聚合
            doc_line_features_list = []
            doc_line_masks_list = []

            chunk_idx = 0
            for doc_idx in range(num_docs):
                num_chunks_in_doc = chunks_per_doc[doc_idx]

                # 收集该文档所有 chunks 的 hidden states 和 line_ids
                doc_hidden = text_hidden[chunk_idx:chunk_idx + num_chunks_in_doc]
                doc_line_ids = line_ids[chunk_idx:chunk_idx + num_chunks_in_doc]

                # 聚合该文档的 line features
                doc_features, doc_mask = self._aggregate_document_line_features(
                    doc_hidden, doc_line_ids
                )
                doc_line_features_list.append(doc_features)
                doc_line_masks_list.append(doc_mask)

                chunk_idx += num_chunks_in_doc

            # 填充到相同长度
            max_lines = max(f.shape[0] for f in doc_line_features_list)
            hidden_dim = doc_line_features_list[0].shape[1]

            line_features = torch.zeros(num_docs, max_lines, hidden_dim, device=device)
            line_mask = torch.zeros(num_docs, max_lines, dtype=torch.bool, device=device)

            for doc_idx, (features, mask) in enumerate(zip(doc_line_features_list, doc_line_masks_list)):
                num_lines_in_doc = features.shape[0]
                line_features[doc_idx, :num_lines_in_doc] = features
                line_mask[doc_idx, :num_lines_in_doc] = mask

        # ==================== Stage 3: Parent Finding ====================
        parent_loss = torch.tensor(0.0, device=device)
        parent_correct = 0
        parent_total = 0
        gru_hidden = None  # GRU 隐状态，用于 Stage 4

        if self.lambda_parent > 0:
            if self.use_gru:
                # 论文对齐：获取 GRU 隐状态用于 Stage 4
                parent_logits, gru_hidden = self.stage3(
                    line_features, line_mask, return_gru_hidden=True
                )
                # gru_hidden: [num_docs, L+1, gru_hidden_size]，包括 ROOT

                for b in range(num_docs):
                    sample_parent_ids = line_parent_ids[b]
                    sample_mask = line_mask[b]
                    num_lines = int(sample_mask.sum().item())

                    for child_idx in range(num_lines):
                        gt_parent = sample_parent_ids[child_idx].item()

                        if gt_parent == -100:
                            continue
                        if gt_parent >= child_idx:
                            continue

                        target_idx = gt_parent + 1 if gt_parent >= 0 else 0
                        child_logits = parent_logits[b, child_idx + 1, :child_idx + 2]

                        if torch.isinf(child_logits).all():
                            continue

                        child_logits = torch.where(
                            torch.isinf(child_logits),
                            torch.full_like(child_logits, -1e4),
                            child_logits
                        )

                        target = torch.tensor([target_idx], device=device)
                        loss = F.cross_entropy(child_logits.unsqueeze(0), target)

                        if not torch.isnan(loss):
                            parent_loss = parent_loss + loss
                            parent_total += 1

                        pred_parent = child_logits.argmax().item()
                        if pred_parent == target_idx:
                            parent_correct += 1
            else:
                for b in range(num_docs):
                    sample_features = line_features[b]
                    sample_mask = line_mask[b]
                    sample_parent_ids = line_parent_ids[b]

                    num_lines = sample_mask.sum().item()
                    if num_lines <= 1:
                        continue

                    for child_idx in range(1, int(num_lines)):
                        gt_parent = sample_parent_ids[child_idx].item()

                        if gt_parent < 0 or gt_parent >= child_idx:
                            continue

                        parent_candidates = sample_features[:child_idx]
                        child_feat = sample_features[child_idx]

                        scores = self.stage3(parent_candidates, child_feat)

                        target = torch.tensor([gt_parent], device=device)
                        loss = F.cross_entropy(scores.unsqueeze(0), target)
                        parent_loss = parent_loss + loss

                        pred_parent = scores.argmax().item()
                        if pred_parent == gt_parent:
                            parent_correct += 1
                        parent_total += 1

            if parent_total > 0:
                parent_loss = parent_loss / parent_total
                self._parent_acc = parent_correct / parent_total

            outputs["parent_loss"] = parent_loss
            outputs["loss"] = outputs["loss"] + parent_loss * self.lambda_parent

        # ==================== Stage 4: Relation Classification ====================
        rel_loss = torch.tensor(0.0, device=device)
        rel_correct = 0
        rel_total = 0

        # 调试统计
        debug_label_counts = {0: 0, 1: 0, 2: 0}  # connect, contain, equality
        debug_pred_counts = {0: 0, 1: 0, 2: 0}
        debug_skipped_parent = 0
        debug_skipped_label = 0

        if self.lambda_rel > 0 and line_relations is not None:
            if gru_hidden is None:
                gru_hidden = line_features
                use_gru_offset = False
            else:
                use_gru_offset = True

            for b in range(num_docs):
                sample_mask = line_mask[b]
                sample_parent_ids = line_parent_ids[b]
                sample_relations = line_relations[b]

                num_lines = int(sample_mask.sum().item())

                for child_idx in range(num_lines):
                    parent_idx = sample_parent_ids[child_idx].item()
                    rel_label = sample_relations[child_idx].item()

                    if parent_idx < 0 or parent_idx >= num_lines:
                        debug_skipped_parent += 1
                        continue
                    if rel_label == -100:
                        debug_skipped_label += 1
                        continue

                    # 统计 label 分布
                    if rel_label in debug_label_counts:
                        debug_label_counts[rel_label] += 1

                    if use_gru_offset:
                        parent_gru_idx = parent_idx + 1
                        child_gru_idx = child_idx + 1
                        parent_feat = gru_hidden[b, parent_gru_idx]
                        child_feat = gru_hidden[b, child_gru_idx]
                    else:
                        parent_feat = gru_hidden[b, parent_idx]
                        child_feat = gru_hidden[b, child_idx]

                    rel_logits = self.stage4(
                        parent_feat.unsqueeze(0),
                        child_feat.unsqueeze(0),
                    )

                    target = torch.tensor([rel_label], device=device)
                    loss = F.cross_entropy(rel_logits, target)
                    rel_loss = rel_loss + loss

                    pred_rel = rel_logits.argmax(dim=1).item()
                    if pred_rel in debug_pred_counts:
                        debug_pred_counts[pred_rel] += 1
                    if pred_rel == rel_label:
                        rel_correct += 1
                    rel_total += 1

            if rel_total > 0:
                rel_loss = rel_loss / rel_total
                self._rel_acc = rel_correct / rel_total

            # 打印调试信息（每 100 步打印一次）
            if not hasattr(self, '_debug_step'):
                self._debug_step = 0
            self._debug_step += 1
            if self._debug_step % 100 == 1:
                print(f"[Stage4 Debug] rel_total={rel_total}, skipped_parent={debug_skipped_parent}, skipped_label={debug_skipped_label}")
                print(f"[Stage4 Debug] Label dist: connect={debug_label_counts[0]}, contain={debug_label_counts[1]}, equality={debug_label_counts[2]}")
                print(f"[Stage4 Debug] Pred dist:  connect={debug_pred_counts[0]}, contain={debug_pred_counts[1]}, equality={debug_pred_counts[2]}")
                print(f"[Stage4 Debug] use_gru_offset={use_gru_offset}, gru_hidden shape={gru_hidden.shape if gru_hidden is not None else None}")

            outputs["rel_loss"] = rel_loss
            outputs["loss"] = outputs["loss"] + rel_loss * self.lambda_rel

        # 保存完整的 outputs 供 compute_loss 使用
        self._outputs_dict = outputs
        return TokenClassifierOutput(
            loss=outputs["loss"],
            logits=outputs["logits"],
        )

    def _aggregate_document_line_features(
        self,
        doc_hidden: torch.Tensor,
        doc_line_ids: torch.Tensor,
    ) -> tuple:
        """
        从文档的所有 chunks 中聚合 line features（向量化版本）

        Args:
            doc_hidden: [num_chunks, seq_len, hidden_dim]
            doc_line_ids: [num_chunks, seq_len]，每个 token 的全局 line_id

        Returns:
            features: [num_lines, hidden_dim]
            mask: [num_lines]，有效行的 mask
        """
        device = doc_hidden.device
        hidden_dim = doc_hidden.shape[-1]

        # 展平（使用 reshape 兼容非连续 tensor）
        flat_hidden = doc_hidden.reshape(-1, hidden_dim)  # [N, hidden_dim]
        flat_line_ids = doc_line_ids.reshape(-1)  # [N]

        # 获取有效 token（line_id >= 0）
        valid_mask = flat_line_ids >= 0
        valid_line_ids = flat_line_ids[valid_mask]
        valid_hidden = flat_hidden[valid_mask]

        if len(valid_line_ids) == 0:
            return torch.zeros(1, hidden_dim, device=device), torch.zeros(1, dtype=torch.bool, device=device)

        # 获取唯一的 line_id 并排序
        unique_line_ids = valid_line_ids.unique()
        unique_line_ids = unique_line_ids.sort()[0]
        num_lines = len(unique_line_ids)

        # 创建 line_id 到连续索引的映射（向量化）
        # 使用 searchsorted 进行快速映射
        line_indices = torch.searchsorted(unique_line_ids, valid_line_ids)

        # 使用 scatter_add 聚合 features
        line_features = torch.zeros(num_lines, hidden_dim, device=device)
        line_features.scatter_add_(0, line_indices.unsqueeze(1).expand(-1, hidden_dim), valid_hidden)

        # 统计每个 line 的 token 数量
        line_counts = torch.zeros(num_lines, device=device)
        line_counts.scatter_add_(0, line_indices, torch.ones_like(line_indices, dtype=torch.float))

        # 计算平均值
        valid_counts = line_counts.clamp(min=1)
        line_features = line_features / valid_counts.unsqueeze(1)

        # 创建 mask
        line_mask = line_counts > 0

        return line_features, line_mask

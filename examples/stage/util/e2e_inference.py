#!/usr/bin/env python
# coding=utf-8
"""
端到端推理核心模块 (E2E Inference Core)

将 Stage 1/2/3/4 的推理逻辑抽离为可复用的函数。
训练/评估/推理脚本都调用此模块，避免代码重复。

设计原则：
- 输入输出契约清晰（batch dict -> prediction dict）
- 不依赖具体的运行模式（train/eval/infer）
- 可单独测试
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import Counter

try:
    from .eval_utils import aggregate_token_to_line_predictions
except ImportError:
    from eval_utils import aggregate_token_to_line_predictions


# ==================== 输出数据结构 ====================

@dataclass
class E2EInferenceOutput:
    """端到端推理输出结构"""
    # Stage 1: 分类
    line_classes: Dict[int, int] = field(default_factory=dict)  # {line_id: class_id}

    # Stage 3: 父节点
    line_parents: List[int] = field(default_factory=list)  # [parent_id for each line]

    # Stage 4: 关系
    line_relations: List[int] = field(default_factory=list)  # [relation_id for each line]

    # 元信息
    num_lines: int = 0
    line_ids: List[int] = field(default_factory=list)  # 实际的 line_id 列表


# ==================== 核心推理函数 ====================

def run_e2e_inference_single(
    model,
    batch: Dict[str, torch.Tensor],
    batch_idx: int = 0,
    device: torch.device = None,
) -> E2EInferenceOutput:
    """
    对单个样本运行端到端推理 (Stage 1/2/3/4)

    Args:
        model: JointModel，包含 stage1, stage3, stage4, feature_extractor, use_gru
        batch: 包含 input_ids, bbox, attention_mask, image, line_ids 等
        batch_idx: batch 中的样本索引
        device: 设备

    Returns:
        E2EInferenceOutput: 包含分类、父节点、关系的预测结果
    """
    if device is None:
        device = next(model.stage1.parameters()).device

    # 处理 image：文档级别模式下可能是 list，需要转换为 tensor
    image = batch.get("image")
    if image is not None and isinstance(image, list):
        image = torch.stack([
            torch.tensor(img) if not isinstance(img, torch.Tensor) else img
            for img in image
        ]).to(device)

    # ==================== Stage 1: Classification ====================
    stage1_outputs = model.stage1(
        input_ids=batch["input_ids"],
        bbox=batch["bbox"],
        attention_mask=batch["attention_mask"],
        image=image,
        output_hidden_states=True,
    )

    logits = stage1_outputs.logits
    hidden_states = stage1_outputs.hidden_states[-1]

    # 获取 line_ids
    line_ids_tensor = batch.get("line_ids")
    if line_ids_tensor is None:
        return E2EInferenceOutput()

    line_ids_b = line_ids_tensor[batch_idx]
    sample_logits = logits[batch_idx]

    # Token -> Line 聚合（多数投票）
    line_ids_list = line_ids_b.cpu().tolist()
    token_preds = [sample_logits[i].argmax().item() for i in range(sample_logits.shape[0])]
    line_classes = aggregate_token_to_line_predictions(token_preds, line_ids_list, method="majority")

    if not line_classes:
        return E2EInferenceOutput()

    # ==================== Stage 2: Feature Extraction ====================
    text_seq_len = batch["input_ids"].shape[1]
    text_hidden = hidden_states[batch_idx:batch_idx+1, :text_seq_len, :]

    line_features, line_mask = model.feature_extractor.extract_line_features(
        text_hidden, line_ids_b.unsqueeze(0), pooling="mean"
    )

    line_features = line_features[0]  # [max_lines, H]
    line_mask = line_mask[0]
    actual_num_lines = int(line_mask.sum().item())

    if actual_num_lines == 0:
        return E2EInferenceOutput(line_classes=line_classes)

    # ==================== Stage 3: Parent Finding ====================
    pred_parents = [-1] * actual_num_lines
    gru_hidden = None

    use_gru = getattr(model, 'use_gru', False)

    if use_gru:
        parent_logits, gru_hidden = model.stage3(
            line_features.unsqueeze(0),
            line_mask.unsqueeze(0),
            return_gru_hidden=True
        )
        gru_hidden = gru_hidden[0]  # [L+1, gru_hidden_size]

        for child_idx in range(actual_num_lines):
            child_logits = parent_logits[0, child_idx + 1, :child_idx + 2]
            pred_parent_idx = child_logits.argmax().item()
            pred_parents[child_idx] = pred_parent_idx - 1  # -1 means ROOT
    else:
        for child_idx in range(1, actual_num_lines):
            parent_candidates = line_features[:child_idx]
            child_feat = line_features[child_idx]
            scores = model.stage3(parent_candidates, child_feat)
            pred_parents[child_idx] = scores.argmax().item()

    # ==================== Stage 4: Relation Classification ====================
    pred_relations = [0] * actual_num_lines  # Default: connect (0)

    for child_idx in range(actual_num_lines):
        parent_idx = pred_parents[child_idx]
        if parent_idx < 0 or parent_idx >= actual_num_lines:
            continue

        if gru_hidden is not None:
            parent_gru_idx = parent_idx + 1
            child_gru_idx = child_idx + 1
            parent_feat = gru_hidden[parent_gru_idx]
            child_feat = gru_hidden[child_gru_idx]
        else:
            parent_feat = line_features[parent_idx]
            child_feat = line_features[child_idx]

        rel_logits = model.stage4(
            parent_feat.unsqueeze(0),
            child_feat.unsqueeze(0),
        )
        pred_relations[child_idx] = rel_logits.argmax(dim=1).item()

    # 构建输出
    sorted_line_ids = sorted(line_classes.keys())

    return E2EInferenceOutput(
        line_classes=line_classes,
        line_parents=pred_parents,
        line_relations=pred_relations,
        num_lines=actual_num_lines,
        line_ids=sorted_line_ids,
    )


def run_e2e_inference_batch(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device = None,
) -> List[E2EInferenceOutput]:
    """
    对整个 batch 运行端到端推理

    Args:
        model: JointModel
        batch: batch dict
        device: 设备

    Returns:
        List[E2EInferenceOutput]: 每个样本的预测结果
    """
    batch_size = batch["input_ids"].shape[0]
    results = []

    for b in range(batch_size):
        result = run_e2e_inference_single(model, batch, batch_idx=b, device=device)
        results.append(result)

    return results


def run_e2e_inference_document(
    model,
    batch: Dict[str, torch.Tensor],
    doc_idx: int = 0,
    device: torch.device = None,
) -> E2EInferenceOutput:
    """
    文档级别端到端推理 (Stage 1/2/3/4)

    处理一个文档的所有 chunks，聚合后进行 Stage 2/3/4 推理。

    Args:
        model: JointModel，包含 stage1, stage3, stage4, feature_extractor, use_gru
        batch: 文档级别 batch，包含:
            - input_ids: [total_chunks, seq_len]
            - num_docs: 文档数量
            - chunks_per_doc: 每个文档的 chunk 数量
            - line_ids: [total_chunks, seq_len]，全局 line_id
        doc_idx: 要推理的文档索引（batch 中可能有多个文档）
        device: 设备

    Returns:
        E2EInferenceOutput: 文档级别的预测结果
    """
    if device is None:
        device = next(model.stage1.parameters()).device

    num_docs = batch.get("num_docs", 1)
    chunks_per_doc = batch.get("chunks_per_doc", [batch["input_ids"].shape[0]])

    # 计算该文档的 chunks 范围
    chunk_start = sum(chunks_per_doc[:doc_idx])
    chunk_end = chunk_start + chunks_per_doc[doc_idx]
    num_chunks = chunks_per_doc[doc_idx]

    # ==================== Stage 1: Classification ====================
    # 只处理该文档的 chunks
    doc_input_ids = batch["input_ids"][chunk_start:chunk_end]
    doc_bbox = batch["bbox"][chunk_start:chunk_end]
    doc_attention_mask = batch["attention_mask"][chunk_start:chunk_end]
    doc_line_ids = batch["line_ids"][chunk_start:chunk_end]

    # 处理 image：文档级别模式下可能是 list，需要转换为 tensor
    doc_image = batch.get("image")
    if doc_image is not None:
        if isinstance(doc_image, list):
            # list of images -> tensor
            doc_image = torch.stack([
                torch.tensor(img) if not isinstance(img, torch.Tensor) else img
                for img in doc_image[chunk_start:chunk_end]
            ]).to(device)
        else:
            doc_image = doc_image[chunk_start:chunk_end]

    stage1_outputs = model.stage1(
        input_ids=doc_input_ids,
        bbox=doc_bbox,
        attention_mask=doc_attention_mask,
        image=doc_image,
        output_hidden_states=True,
    )

    logits = stage1_outputs.logits  # [num_chunks, seq_len, num_classes]
    hidden_states = stage1_outputs.hidden_states[-1]  # [num_chunks, seq_len+?, hidden]

    # 从所有 chunks 收集 line_classes（使用全局 line_id）
    line_classes = {}
    line_votes = {}  # {line_id: Counter of class votes}

    for chunk_idx in range(num_chunks):
        chunk_line_ids = doc_line_ids[chunk_idx].cpu().tolist()
        chunk_logits = logits[chunk_idx]
        chunk_preds = [chunk_logits[i].argmax().item() for i in range(chunk_logits.shape[0])]

        for token_idx, (line_id, pred) in enumerate(zip(chunk_line_ids, chunk_preds)):
            if line_id < 0:
                continue
            if line_id not in line_votes:
                line_votes[line_id] = Counter()
            line_votes[line_id][pred] += 1

    # 多数投票确定每行的类别
    for line_id, votes in line_votes.items():
        line_classes[line_id] = votes.most_common(1)[0][0]

    if not line_classes:
        return E2EInferenceOutput()

    # ==================== Stage 2: Feature Extraction ====================
    # 从所有 chunks 聚合 line features
    text_seq_len = doc_input_ids.shape[1]
    text_hidden = hidden_states[:, :text_seq_len, :]  # [num_chunks, seq_len, hidden]
    hidden_dim = text_hidden.shape[-1]

    # 收集所有有效的 line_id（向量化聚合）
    valid_line_ids_list = sorted(line_classes.keys())
    num_lines = len(valid_line_ids_list)
    valid_line_ids_tensor = torch.tensor(valid_line_ids_list, device=device)

    # 展平
    flat_hidden = text_hidden.view(-1, hidden_dim)  # [N, hidden_dim]
    flat_line_ids = doc_line_ids.view(-1)  # [N]

    # 获取有效 token
    valid_mask = flat_line_ids >= 0
    valid_token_line_ids = flat_line_ids[valid_mask]
    valid_hidden = flat_hidden[valid_mask]

    # 使用 searchsorted 映射 line_id 到索引
    line_indices = torch.searchsorted(valid_line_ids_tensor, valid_token_line_ids)

    # 过滤掉不在 valid_line_ids 中的 token
    in_bounds = (line_indices < num_lines) & (valid_line_ids_tensor[line_indices.clamp(max=num_lines-1)] == valid_token_line_ids)
    line_indices = line_indices[in_bounds]
    valid_hidden = valid_hidden[in_bounds]

    # 使用 scatter_add 聚合
    line_features = torch.zeros(num_lines, hidden_dim, device=device)
    line_features.scatter_add_(0, line_indices.unsqueeze(1).expand(-1, hidden_dim), valid_hidden)

    line_counts = torch.zeros(num_lines, device=device)
    line_counts.scatter_add_(0, line_indices, torch.ones_like(line_indices, dtype=torch.float))

    # 计算平均值
    valid_counts = line_counts.clamp(min=1)
    line_features = line_features / valid_counts.unsqueeze(1)

    # 创建 mask
    line_mask = line_counts > 0

    actual_num_lines = int(line_mask.sum().item())
    if actual_num_lines == 0:
        return E2EInferenceOutput(line_classes=line_classes)

    # ==================== Stage 3: Parent Finding ====================
    pred_parents = [-1] * actual_num_lines
    gru_hidden = None

    use_gru = getattr(model, 'use_gru', False)

    if use_gru:
        parent_logits, gru_hidden = model.stage3(
            line_features.unsqueeze(0),
            line_mask.unsqueeze(0),
            return_gru_hidden=True
        )
        gru_hidden = gru_hidden[0]  # [L+1, gru_hidden_size]

        for child_idx in range(actual_num_lines):
            child_logits = parent_logits[0, child_idx + 1, :child_idx + 2]
            pred_parent_idx = child_logits.argmax().item()
            pred_parents[child_idx] = pred_parent_idx - 1  # -1 means ROOT
    else:
        for child_idx in range(1, actual_num_lines):
            parent_candidates = line_features[:child_idx]
            child_feat = line_features[child_idx]
            scores = model.stage3(parent_candidates, child_feat)
            pred_parents[child_idx] = scores.argmax().item()

    # ==================== Stage 4: Relation Classification ====================
    pred_relations = [0] * actual_num_lines  # Default: connect (0)

    for child_idx in range(actual_num_lines):
        parent_idx = pred_parents[child_idx]
        if parent_idx < 0 or parent_idx >= actual_num_lines:
            continue

        if gru_hidden is not None:
            parent_gru_idx = parent_idx + 1
            child_gru_idx = child_idx + 1
            parent_feat = gru_hidden[parent_gru_idx]
            child_feat = gru_hidden[child_gru_idx]
        else:
            parent_feat = line_features[parent_idx]
            child_feat = line_features[child_idx]

        rel_logits = model.stage4(
            parent_feat.unsqueeze(0),
            child_feat.unsqueeze(0),
        )
        pred_relations[child_idx] = rel_logits.argmax(dim=1).item()

    return E2EInferenceOutput(
        line_classes=line_classes,
        line_parents=pred_parents,
        line_relations=pred_relations,
        num_lines=actual_num_lines,
        line_ids=valid_line_ids,
    )

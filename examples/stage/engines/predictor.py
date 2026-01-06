#!/usr/bin/env python
# coding=utf-8
"""
Predictor - 统一推理接口

支持页面级别和文档级别的推理，使用 Batch 抽象层隐藏差异。

=== 推理流程（与训练一致）===

    Stage 1: 所有行预测类别
         ↓
    Stage 3: 所有行预测 parent（与训练一致）
         ↓
    Stage 4: 根据 Stage 1 预测的 class 判断
      - meta 类 → 直接填 relation="meta"，不调用模型
      - 非meta类 → 调用模型预测 relation

设计原则：
- 接收 Sample，返回 PredictionOutput
- 不关心 Sample 来自页面级别还是文档级别
- 内部处理多 chunk 聚合
- 使用 tasks/ 中的 decode 逻辑，确保训练和评估一致
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.batch import Sample, BatchBase, wrap_batch
from tasks import SemanticClassificationTask
from tasks.parent_finding import ParentFindingTask
from layoutlmft.data.labels import is_meta_class


@dataclass
class PredictionOutput:
    """单个样本的预测输出"""
    # Stage 1: 分类
    line_classes: Dict[int, int] = field(default_factory=dict)  # {line_id: class_id}

    # Stage 3: 父节点
    line_parents: List[int] = field(default_factory=list)  # [parent_id for each line]

    # Stage 4: 关系
    line_relations: List[int] = field(default_factory=list)  # [relation_id for each line]

    # 元信息
    num_lines: int = 0
    line_ids: List[int] = field(default_factory=list)

    # 置信度（可选）
    line_class_probs: Optional[Dict[int, List[float]]] = None


class Predictor:
    """
    统一推理器

    使用方式：
        predictor = Predictor(model, device)
        for sample in batch:
            output = predictor.predict(sample)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        micro_batch_size: int = 1,
    ):
        """
        Args:
            model: JointModel 或类似结构（需要有 stage1, stage3, stage4, feature_extractor）
            device: 计算设备
            micro_batch_size: Stage 1 推理时的 micro-batch 大小（默认 1，与训练一致）
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.micro_batch_size = micro_batch_size

        # 使用 tasks/ 中的统一 decode 逻辑（复用，不重复实现）
        self.cls_task = SemanticClassificationTask(model=model, use_line_level=True)
        self.parent_task = ParentFindingTask()

    def predict(self, sample: Sample) -> PredictionOutput:
        """
        对单个样本进行推理

        Args:
            sample: Sample 对象（可能包含多个 chunks）

        Returns:
            PredictionOutput: 预测结果
        """
        sample = sample.to(self.device)

        if sample.is_document_level:
            return self._predict_document(sample)
        else:
            return self._predict_page(sample)

    def _predict_page(self, sample: Sample) -> PredictionOutput:
        """页面级别推理（单 chunk）"""
        # 添加 batch 维度
        input_ids = sample.input_ids.unsqueeze(0)
        bbox = sample.bbox.unsqueeze(0)
        attention_mask = sample.attention_mask.unsqueeze(0)
        image = sample.image.unsqueeze(0) if sample.image is not None else None
        line_ids = sample.line_ids.unsqueeze(0) if sample.line_ids is not None else None

        return self._run_inference(input_ids, bbox, attention_mask, image, line_ids)

    def _predict_document(self, sample: Sample) -> PredictionOutput:
        """文档级别推理（多 chunk 聚合）"""
        # 文档级别：input_ids 已经是 [num_chunks, seq_len]
        input_ids = sample.input_ids
        bbox = sample.bbox
        attention_mask = sample.attention_mask
        image = sample.image
        line_ids = sample.line_ids

        return self._run_inference(input_ids, bbox, attention_mask, image, line_ids)

    def _run_inference(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        image: Optional[torch.Tensor],
        line_ids: Optional[torch.Tensor],
    ) -> PredictionOutput:
        """
        核心推理逻辑

        Args:
            input_ids: [num_chunks, seq_len]
            bbox: [num_chunks, seq_len, 4]
            attention_mask: [num_chunks, seq_len]
            image: [num_chunks, C, H, W] or None
            line_ids: [num_chunks, seq_len] or None

        Returns:
            PredictionOutput
        """
        if line_ids is None:
            return PredictionOutput()

        num_chunks = input_ids.shape[0]

        # ==================== Stage 1: Classification ====================
        # 使用 encode_with_micro_batch 复用 micro-batching 逻辑（与训练一致）
        # 推理时强制 no_grad=True 节省显存
        hidden_states = self.model.encode_with_micro_batch(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
            micro_batch_size=self.micro_batch_size,
            no_grad=True,
        )

        # 截取文本部分的 hidden states（排除视觉 tokens）
        seq_len = input_ids.shape[1]
        text_hidden = hidden_states[:, :seq_len, :]  # [num_chunks, seq_len, H]

        # 使用 SemanticClassificationTask 进行分类（与训练时一致）
        # 内部使用 model.line_pooling + model.cls_head（line-level 模式）
        line_classes = self.cls_task.decode(
            hidden_states=text_hidden,
            line_ids=line_ids,
        )

        if not line_classes:
            return PredictionOutput()

        # ==================== Stage 2: Feature Extraction ====================
        # 使用 model.line_pooling 聚合（与训练一致，不再区分页面/文档级别）
        # line_pooling 内部自动处理多 chunk 聚合
        line_features, line_mask = self.model.line_pooling(text_hidden, line_ids)
        # line_features: [num_lines, H], line_mask: [num_lines]

        actual_num_lines = int(line_mask.sum().item())

        if actual_num_lines == 0:
            return PredictionOutput(line_classes=line_classes)

        # ==================== Stage 3: Parent Finding ====================
        # 所有行都参与（与训练一致）
        pred_parents = [-1] * actual_num_lines
        gru_hidden = None

        use_gru = getattr(self.model, 'use_gru', False)

        # 获取 cls_logits（用于 soft-mask）
        # 如果模型有 cls_head（line-level 分类），使用它来获取 cls_logits
        cls_logits = None
        if hasattr(self.model, 'cls_head') and self.model.cls_head is not None:
            valid_features = line_features[:actual_num_lines]  # [L, H]
            cls_logits = self.model.cls_head(valid_features)  # [L, num_classes]
            cls_logits = cls_logits.unsqueeze(0)  # [1, L, num_classes]

        if use_gru:
            # 调试日志：Stage 3 输入
            print(f"\n[Stage3 Debug] Input shapes:")
            print(f"  line_features: {line_features.shape}")  # 应该是 [426, H]
            print(f"  line_mask: {line_mask.shape}, sum={line_mask.sum().item()}")  # 应该是 [426], sum=426
            print(f"  cls_logits: {cls_logits.shape if cls_logits is not None else None}")  # 应该是 [1, 426, 14]

            parent_logits, gru_hidden = self.model.stage3(
                line_features.unsqueeze(0),
                line_mask.unsqueeze(0),
                return_gru_hidden=True,
                cls_logits=cls_logits  # 传入分类 logits 用于 soft-mask
            )

            # 调试日志：Stage 3 输出
            print(f"[Stage3 Debug] Output shapes:")
            print(f"  parent_logits: {parent_logits.shape}")  # 应该是 [1, 427, 427]
            print(f"  gru_hidden: {gru_hidden.shape}")  # 应该是 [1, 427, 512]

            # 打印 parent_logits 的一些统计
            print(f"[Stage3 Debug] parent_logits stats:")
            print(f"  min={parent_logits.min().item():.4f}, max={parent_logits.max().item():.4f}")
            # 打印第10行的候选父节点分数
            if parent_logits.shape[1] > 10:
                row10_logits = parent_logits[0, 10, :10]  # 第10行（line_id=9）的候选父节点
                print(f"  Row 10 logits (candidates 0-9): {row10_logits.tolist()}")
                print(f"  Row 10 argmax: {row10_logits.argmax().item()}")

            gru_hidden = gru_hidden[0]  # [L+1, gru_hidden_size]

            # 复用 tasks/parent_finding.py 的 decode 逻辑
            parent_preds = self.parent_task.decode(parent_logits, line_mask.unsqueeze(0))
            pred_parents = parent_preds[0].tolist()  # [L] -> list

            # 调试日志：预测结果统计
            from collections import Counter
            parent_counter = Counter(pred_parents)
            print(f"[Stage3 Debug] Prediction stats:")
            print(f"  pred_parents length: {len(pred_parents)}")
            print(f"  pred_parents[:20]: {pred_parents[:20]}")
            print(f"  ROOT (-1) count: {parent_counter[-1]}")
            print(f"  Non-ROOT count: {len(pred_parents) - parent_counter[-1]}")
        else:
            for child_idx in range(1, actual_num_lines):
                parent_candidates = line_features[:child_idx]
                child_feat = line_features[child_idx]
                scores = self.model.stage3(parent_candidates, child_feat)
                pred_parents[child_idx] = scores.argmax().item()

        # ==================== Stage 4: Relation Classification ====================
        # 根据 Stage 1 预测的 class 判断是否为 meta 类
        # - meta 类：跳过，不调用模型（输出时直接填 "meta"）
        # - 非 meta 类：调用模型预测 relation
        sorted_line_ids = sorted(line_classes.keys())
        pred_relations = [0] * actual_num_lines  # 默认 connect

        for child_idx in range(actual_num_lines):
            # 获取该行预测的类别
            line_id = sorted_line_ids[child_idx] if child_idx < len(sorted_line_ids) else child_idx
            pred_class = line_classes.get(line_id, 0)

            # meta 类跳过模型预测（输出时用 is_meta_class 判断填 "meta"）
            if is_meta_class(pred_class):
                continue

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

            rel_logits = self.model.stage4(
                parent_feat.unsqueeze(0),
                child_feat.unsqueeze(0),
            )
            pred_relations[child_idx] = rel_logits.argmax(dim=1).item()

        # 构建输出

        return PredictionOutput(
            line_classes=line_classes,
            line_parents=pred_parents,
            line_relations=pred_relations,
            num_lines=actual_num_lines,
            line_ids=sorted_line_ids,
        )

    def predict_batch(self, batch: BatchBase) -> List[PredictionOutput]:
        """
        对整个 batch 进行推理

        Args:
            batch: BatchBase 对象

        Returns:
            List[PredictionOutput]: 每个样本的预测结果
        """
        results = []
        for sample in batch:
            with torch.no_grad():
                result = self.predict(sample)
            results.append(result)
        return results

    def predict_single_from_batch(
        self,
        batch: Dict[str, Any],
        batch_idx: int = 0,
    ) -> PredictionOutput:
        """
        从页面级别 batch 中提取单个样本进行推理

        Args:
            batch: 页面级别 batch，input_ids 形状为 [batch_size, seq_len]
            batch_idx: 要推理的样本索引

        Returns:
            PredictionOutput: 预测结果
        """
        # 提取单个样本
        input_ids = batch["input_ids"][batch_idx:batch_idx+1]
        bbox = batch["bbox"][batch_idx:batch_idx+1]
        attention_mask = batch["attention_mask"][batch_idx:batch_idx+1]

        line_ids = batch.get("line_ids")
        if line_ids is not None:
            line_ids = line_ids[batch_idx:batch_idx+1]

        # 处理 image
        image = batch.get("image")
        if image is not None:
            if isinstance(image, list):
                image = [image[batch_idx]]
            else:
                image = image[batch_idx:batch_idx+1]

        # 移动到设备
        input_ids = input_ids.to(self.device)
        bbox = bbox.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if line_ids is not None:
            line_ids = line_ids.to(self.device)
        if image is not None and isinstance(image, torch.Tensor):
            image = image.to(self.device)

        return self._run_inference(input_ids, bbox, attention_mask, image, line_ids)

    def predict_from_dict(
        self,
        batch: Dict[str, Any],
        doc_idx: int = 0,
    ) -> PredictionOutput:
        """
        从文档级别 batch dict 进行推理

        Args:
            batch: 包含 input_ids, bbox, attention_mask, image, line_ids 等的字典
                - 文档级别：input_ids 形状为 [total_chunks, seq_len]
                - 包含 num_docs 和 chunks_per_doc 字段
            doc_idx: 文档索引

        Returns:
            PredictionOutput: 预测结果
        """
        # 获取文档范围
        num_docs = batch.get("num_docs", 1)
        chunks_per_doc = batch.get("chunks_per_doc", [batch["input_ids"].shape[0]])

        chunk_start = sum(chunks_per_doc[:doc_idx])
        chunk_end = chunk_start + chunks_per_doc[doc_idx]

        # 提取该文档的数据
        input_ids = batch["input_ids"][chunk_start:chunk_end]
        bbox = batch["bbox"][chunk_start:chunk_end]
        attention_mask = batch["attention_mask"][chunk_start:chunk_end]
        line_ids = batch.get("line_ids")
        if line_ids is not None:
            line_ids = line_ids[chunk_start:chunk_end]

        # 处理 image
        image = batch.get("image")
        if image is not None:
            if isinstance(image, list):
                image = image[chunk_start:chunk_end]
            else:
                image = image[chunk_start:chunk_end]

        # 移动到设备
        input_ids = input_ids.to(self.device)
        bbox = bbox.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if line_ids is not None:
            line_ids = line_ids.to(self.device)
        if image is not None and isinstance(image, torch.Tensor):
            image = image.to(self.device)

        return self._run_inference(input_ids, bbox, attention_mask, image, line_ids)

    # ==================== 推理并保存 ====================

    def predict_and_save(
        self,
        dataloader,
        output_dir: str,
        verbose: bool = True,
    ) -> List[str]:
        """
        对 dataloader 进行推理并保存结果

        每个文档保存为 {document_name}_infer.json，格式与原始 JSON 一致（数组格式）。

        Args:
            dataloader: DataLoader
            output_dir: 输出目录
            verbose: 是否显示进度条

        Returns:
            output_files: 保存的文件路径列表
        """
        import json
        import os
        import time
        from tqdm import tqdm
        from data.batch import wrap_batch

        # 统计信息
        total_docs = 0
        total_pages = 0
        total_lines = 0
        start_time = time.time()

        # 标签映射
        try:
            from layoutlmft.data.labels import ID2LABEL
            print(f"[DEBUG] ID2LABEL loaded successfully: {ID2LABEL}")
        except ImportError as e:
            print(f"[DEBUG] Failed to import ID2LABEL: {e}")
            ID2LABEL = {i: f"cls_{i}" for i in range(14)}

        RELATION_LABELS = {0: "connect", 1: "contain", 2: "equality"}

        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()

        output_files = []
        iterator = tqdm(dataloader, desc="Inference") if verbose else dataloader

        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(iterator):
                batch = wrap_batch(raw_batch)
                batch = batch.to(self.device)

                # 获取文档信息
                document_names = raw_batch.get("document_names", [])
                json_paths = raw_batch.get("json_paths", [])

                for sample_idx, sample in enumerate(batch):
                    pred = self.predict(sample)

                    # 获取文档名和 JSON 路径
                    doc_name = document_names[sample_idx] if sample_idx < len(document_names) else f"doc_{batch_idx}_{sample_idx}"
                    json_path = json_paths[sample_idx] if sample_idx < len(json_paths) else ""

                    # 更新统计
                    total_docs += 1
                    num_chunks = raw_batch.get("chunks_per_doc", [1])[sample_idx] if "chunks_per_doc" in raw_batch else 1
                    total_pages += num_chunks  # chunks 约等于页数
                    total_lines += pred.num_lines

                    print(f"\n[Predictor] Processing: {doc_name}")
                    print(f"  num_lines: {pred.num_lines}, chunks: {num_chunks}")
                    # 调试：打印部分 line_classes
                    sample_classes = {k: v for i, (k, v) in enumerate(pred.line_classes.items()) if i < 10}
                    print(f"  line_classes sample (first 10): {sample_classes}")
                    # 调试：统计各类别预测数量
                    from collections import Counter
                    class_counter = Counter(pred.line_classes.values())
                    class_stats = {ID2LABEL.get(k, f"cls_{k}"): v for k, v in sorted(class_counter.items())}
                    print(f"  class distribution: {class_stats}")

                    # 加载原始 JSON
                    original_data = []
                    if json_path and os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            original_data = json.load(f)
                        print(f"  Loaded original JSON: {len(original_data)} items")
                    else:
                        print(f"  Warning: Original JSON not found: {json_path}")

                    # 构建预测结果数组
                    sorted_line_ids = sorted(pred.line_classes.keys())

                    # 创建 line_id -> prediction 映射（覆盖原字段）
                    pred_map = {}
                    for idx, line_id in enumerate(sorted_line_ids):
                        pred_class = pred.line_classes.get(line_id, 0)
                        pred_parent = pred.line_parents[idx] if idx < len(pred.line_parents) else -1
                        pred_relation = pred.line_relations[idx] if idx < len(pred.line_relations) else 0

                        # meta 类直接填 "meta"，非 meta 类用模型预测值
                        if is_meta_class(pred_class):
                            relation_str = "meta"
                        else:
                            relation_str = RELATION_LABELS.get(pred_relation, f"rel_{pred_relation}")

                        pred_map[line_id] = {
                            "class": ID2LABEL.get(pred_class, f"cls_{pred_class}"),
                            "parent_id": pred_parent,
                            "relation": relation_str,
                        }

                    # 调试：打印部分 pred_map
                    sample_pred_map = {k: v for i, (k, v) in enumerate(pred_map.items()) if i < 5}
                    print(f"  pred_map sample (first 5): {sample_pred_map}")

                    # 调试：统计 parent_id 分布，特别是 -1 的情况
                    parent_counter = Counter(pred.line_parents)
                    print(f"  parent_id=-1 count: {parent_counter.get(-1, 0)}")
                    # 打印 parent_id=54 的元素
                    lines_with_parent_54 = [(line_id, pred_map[line_id]) for line_id in sorted_line_ids
                                            if pred_map[line_id]['parent_id'] == 54]
                    if lines_with_parent_54:
                        print(f"  Lines with parent_id=54 ({len(lines_with_parent_54)}):")
                        for line_id, info in lines_with_parent_54[:10]:
                            print(f"    line_id={line_id}: {info}")

                    # 将预测结果添加到原始数据中
                    output_data = []
                    for item in original_data:
                        # 复制原始数据
                        new_item = dict(item)
                        # 获取 line_id
                        line_id = item.get("line_id", item.get("id", -1))
                        if isinstance(line_id, str):
                            try:
                                line_id = int(line_id)
                            except ValueError:
                                line_id = -1
                        # 添加预测结果
                        if line_id in pred_map:
                            new_item.update(pred_map[line_id])
                        output_data.append(new_item)

                    # 保存为 {document_name}_infer.json
                    output_file = os.path.join(output_dir, f"{doc_name}_infer.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)

                    output_files.append(output_file)
                    print(f"  Saved: {output_file} ({len(output_data)} items)")

        # 计算推理时间
        end_time = time.time()
        inference_time = end_time - start_time

        # 保存元信息
        meta_file = os.path.join(output_dir, "meta.json")
        import datetime
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_documents": len(output_files),
            "total_chunks": total_pages,
            "total_lines": total_lines,
            "inference_time_seconds": round(inference_time, 2),
            "files": [os.path.basename(f) for f in output_files],
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"[Predictor] Inference Summary:")
        print(f"  Documents: {total_docs}")
        print(f"  Total chunks (≈pages): {total_pages}")
        print(f"  Total lines: {total_lines}")
        print(f"  Inference time: {inference_time:.2f}s")
        if total_docs > 0:
            print(f"  Avg time per doc: {inference_time/total_docs:.2f}s")
        if total_pages > 0:
            print(f"  Avg time per chunk: {inference_time/total_pages:.3f}s")
        print(f"  Output dir: {output_dir}")
        print(f"{'='*60}")
        self.model.train()
        return output_files

    # predict_from_dir 已删除，统一使用 HRDocDataLoader + predict_and_save

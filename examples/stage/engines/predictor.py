#!/usr/bin/env python
# coding=utf-8
"""
Predictor - 统一推理接口

支持页面级别和文档级别的推理，使用 Batch 抽象层隐藏差异。

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
            parent_logits, gru_hidden = self.model.stage3(
                line_features.unsqueeze(0),
                line_mask.unsqueeze(0),
                return_gru_hidden=True,
                cls_logits=cls_logits  # 传入分类 logits 用于 soft-mask
            )
            gru_hidden = gru_hidden[0]  # [L+1, gru_hidden_size]

            # 复用 tasks/parent_finding.py 的 decode 逻辑
            parent_preds = self.parent_task.decode(parent_logits, line_mask.unsqueeze(0))
            pred_parents = parent_preds[0].tolist()  # [L] -> list
        else:
            for child_idx in range(1, actual_num_lines):
                parent_candidates = line_features[:child_idx]
                child_feat = line_features[child_idx]
                scores = self.model.stage3(parent_candidates, child_feat)
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

            rel_logits = self.model.stage4(
                parent_feat.unsqueeze(0),
                child_feat.unsqueeze(0),
            )
            pred_relations[child_idx] = rel_logits.argmax(dim=1).item()

        # 构建输出
        sorted_line_ids = sorted(line_classes.keys())

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
    ) -> str:
        """
        对 dataloader 进行推理并保存结果

        Args:
            dataloader: DataLoader
            output_dir: 输出目录
            verbose: 是否显示进度条

        Returns:
            predictions_file: 保存的文件路径
        """
        import json
        import os
        from tqdm import tqdm
        from data.batch import wrap_batch

        # 标签映射
        try:
            from layoutlmft.data.labels import ID2LABEL
        except ImportError:
            ID2LABEL = {i: f"cls_{i}" for i in range(14)}

        RELATION_LABELS = {0: "connect", 1: "contain", 2: "equality"}

        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()

        all_predictions = []
        iterator = tqdm(dataloader, desc="Inference") if verbose else dataloader

        with torch.no_grad():
            for batch_idx, raw_batch in enumerate(iterator):
                batch = wrap_batch(raw_batch)
                batch = batch.to(self.device)

                for sample_idx, sample in enumerate(batch):
                    pred = self.predict(sample)

                    # 构建结果
                    sample_pred = {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "num_lines": pred.num_lines,
                        "lines": []
                    }

                    sorted_line_ids = sorted(pred.line_classes.keys())
                    for idx, line_id in enumerate(sorted_line_ids):
                        pred_class = pred.line_classes.get(line_id, 0)
                        pred_parent = pred.line_parents[idx] if idx < len(pred.line_parents) else -1
                        pred_relation = pred.line_relations[idx] if idx < len(pred.line_relations) else 0

                        sample_pred["lines"].append({
                            "line_id": line_id,
                            "pred_class": ID2LABEL.get(pred_class, f"cls_{pred_class}"),
                            "pred_class_id": pred_class,
                            "pred_parent": pred_parent,
                            "pred_relation": RELATION_LABELS.get(pred_relation, f"rel_{pred_relation}"),
                            "pred_relation_id": pred_relation,
                        })

                    all_predictions.append(sample_pred)

        # 保存结果
        predictions_file = os.path.join(output_dir, "predictions.json")
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)

        # 保存元信息
        meta_file = os.path.join(output_dir, "meta.json")
        import datetime
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "num_samples": len(all_predictions),
            "total_lines": sum(p["num_lines"] for p in all_predictions),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n[Predictor] Saved {len(all_predictions)} predictions to: {predictions_file}")
        self.model.train()
        return predictions_file

    def predict_from_dir(
        self,
        input_dir: str,
        output_dir: str,
        tokenizer,
        image_dir: str = None,
        verbose: bool = True,
    ) -> str:
        """
        从输入目录读取 JSON 文件进行推理（纯推理，无 GT）

        Args:
            input_dir: 输入目录（包含 JSON 文件）
            output_dir: 输出目录
            tokenizer: Tokenizer
            image_dir: 图片目录（默认为 input_dir 同级的 images 目录）
            verbose: 是否显示进度条

        Returns:
            output_dir: 输出目录路径
        """
        import json
        import os
        from glob import glob
        from tqdm import tqdm

        # 复用训练代码的 tokenize 函数和图片加载函数
        from data.hrdoc_data_loader import tokenize_page_with_line_boundary
        from layoutlmft.data.utils import load_image

        # 标签映射
        try:
            from layoutlmft.data.labels import ID2LABEL
        except ImportError:
            ID2LABEL = {i: f"cls_{i}" for i in range(14)}

        RELATION_LABELS = {0: "connect", 1: "contain", 2: "equality"}

        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()

        # 确定图片目录（默认为 input_dir 同级的 images 目录）
        if image_dir is None:
            # 尝试 input_dir/../images
            parent_dir = os.path.dirname(input_dir.rstrip("/"))
            image_dir = os.path.join(parent_dir, "images")

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        print(f"[Predictor] Image directory: {image_dir}")

        # 找到所有 JSON 文件
        json_files = glob(os.path.join(input_dir, "*.json"))
        if not json_files:
            print(f"[Predictor] No JSON files found in {input_dir}")
            return output_dir

        iterator = tqdm(json_files, desc="Inference") if verbose else json_files

        with torch.no_grad():
            for json_file in iterator:
                filename = os.path.basename(json_file)
                doc_name = filename.replace(".json", "")

                # 读取输入
                with open(json_file, "r", encoding="utf-8") as f:
                    input_data = json.load(f)

                if not input_data:
                    continue

                # 加载图片（复用训练代码的图片查找逻辑）
                # 尝试多种路径格式
                img_path = None
                page_num = 0  # 默认第 0 页
                for ext in [".png", ".jpg", ".jpeg"]:
                    # 格式1: {image_dir}/{doc_name}/{page_num}.png
                    candidate = os.path.join(image_dir, doc_name, f"{page_num}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                    # 格式2: {image_dir}/{doc_name}.png
                    candidate = os.path.join(image_dir, f"{doc_name}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                    # 格式3: {image_dir}/{doc_name}/{doc_name}_{page_num}.png
                    candidate = os.path.join(image_dir, doc_name, f"{doc_name}_{page_num}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

                if img_path is None:
                    raise FileNotFoundError(
                        f"Image not found for '{doc_name}'. Tried:\n"
                        f"  - {image_dir}/{doc_name}/{page_num}.png/jpg\n"
                        f"  - {image_dir}/{doc_name}.png/jpg\n"
                        f"  - {image_dir}/{doc_name}/{doc_name}_{page_num}.png/jpg"
                    )

                # 加载图片
                image, img_size = load_image(img_path)
                image_tensor = torch.tensor(image, device=self.device).unsqueeze(0).float()

                # 提取文本和 bbox
                tokens = [item.get("text", "") for item in input_data]
                bboxes = [item.get("box", [0, 0, 0, 0]) for item in input_data]
                # 推理时没有真实标签，用 0 填充
                labels = [0] * len(tokens)
                line_ids = list(range(len(tokens)))

                # 复用训练代码的 tokenize 函数
                chunks = tokenize_page_with_line_boundary(
                    tokenizer=tokenizer,
                    tokens=tokens,
                    bboxes=bboxes,
                    labels=labels,
                    line_ids=line_ids,
                    max_length=512,
                )

                if not chunks:
                    continue

                # 把所有 chunks 合并成一个 batch（和训练时一致）
                num_chunks = len(chunks)
                input_ids = torch.tensor([c["input_ids"] for c in chunks], device=self.device)
                bbox = torch.tensor([c["bbox"] for c in chunks], device=self.device)
                attention_mask = torch.tensor([c["attention_mask"] for c in chunks], device=self.device)
                chunk_line_ids = torch.tensor([c["line_ids"] for c in chunks], device=self.device)

                # 复制 image 到每个 chunk（和训练时一致）
                if num_chunks > 1:
                    batch_image = image_tensor.expand(num_chunks, -1, -1, -1)
                else:
                    batch_image = image_tensor

                # 一次性推理所有 chunks（line_pooling 会自动聚合跨 chunk 的行）
                pred = self._run_inference(input_ids, bbox, attention_mask, batch_image, chunk_line_ids)

                # 调试：打印预测统计
                from collections import Counter
                class_id_counts = Counter(pred.line_classes.values())
                class_name_counts = {ID2LABEL.get(k, f"cls_{k}"): v for k, v in class_id_counts.items()}
                print(f"  [{doc_name}] {len(pred.line_classes)} lines predicted: {dict(class_name_counts)}")

                # 构建输出（和 predict_and_save 保持一致）
                # pred.line_classes 的 key 是全局 line_id
                # pred.line_parents/line_relations 的索引是紧凑索引（0, 1, 2, ...）
                sorted_line_ids = sorted(pred.line_classes.keys())

                # 构建紧凑索引到全局 line_id 的映射
                # pred_parent 返回的是紧凑索引，需要映射回全局 line_id
                compact_to_global = {i: lid for i, lid in enumerate(sorted_line_ids)}

                # 为每个输入行设置预测结果
                output_data = []
                for item_idx, item in enumerate(input_data):
                    output_item = item.copy()

                    # 查找这个 item 对应的紧凑索引
                    if item_idx in pred.line_classes:
                        compact_idx = sorted_line_ids.index(item_idx)
                        pred_class = pred.line_classes[item_idx]
                        pred_parent_compact = pred.line_parents[compact_idx] if compact_idx < len(pred.line_parents) else -1
                        pred_relation = pred.line_relations[compact_idx] if compact_idx < len(pred.line_relations) else 0

                        # 将 parent 的紧凑索引映射回全局 line_id
                        if pred_parent_compact >= 0 and pred_parent_compact in compact_to_global:
                            pred_parent = compact_to_global[pred_parent_compact]
                        else:
                            pred_parent = pred_parent_compact  # -1 (ROOT) 保持不变

                        output_item["class"] = ID2LABEL.get(pred_class, f"cls_{pred_class}")
                        output_item["parent_id"] = pred_parent
                        output_item["relation"] = RELATION_LABELS.get(pred_relation, f"rel_{pred_relation}")
                    else:
                        # 该行没有预测结果（可能被跳过）
                        output_item["class"] = "unknown"
                        output_item["parent_id"] = -1
                        output_item["relation"] = "none"

                    output_data.append(output_item)

                # 保存到输出目录
                output_file = os.path.join(output_dir, filename)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n[Predictor] Processed {len(json_files)} files, saved to: {output_dir}")
        self.model.train()
        return output_dir

#!/usr/bin/env python
# coding=utf-8
"""
Inference Service - Core inference logic

Reuses existing inference code from examples/stage/.

Directory structure:
    data_dir_base/
    └── {task_id}/
        ├── {document_name}.json
        └── images/
            └── {document_name}/
                ├── 0.png
                └── 1.png
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple

import torch

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
EXAMPLES_ROOT = os.path.join(PROJECT_ROOT, "examples")
sys.path.insert(0, EXAMPLES_ROOT)
STAGE_ROOT = os.path.join(EXAMPLES_ROOT, "stage")
sys.path.insert(0, STAGE_ROOT)

from data.inference_data_loader import load_single_document
from data.hrdoc_data_loader import tokenize_page_with_line_boundary, get_label2id, get_id2label
from data.batch import wrap_batch
from joint_data_collator import HRDocDocumentLevelCollator
from layoutlmft.data.labels import ID2LABEL
from comp_hrdoc.utils.tree_utils import (
    build_tree_from_parents,
    format_toc_tree,
    resolve_ref_parents_and_relations,  # 反向转换: 格式B → 格式A
    flatten_full_tree_to_format_a,  # 完整树转扁平格式A
)

from .model_loader import get_model_loader

logger = logging.getLogger(__name__)


# Relation labels mapping
RELATION_LABELS = {0: "connect", 1: "contain", 2: "equality"}


class InferenceService:
    """Service for running inference on documents."""

    def __init__(self, data_dir_base: str = None):
        """
        Initialize inference service.

        Args:
            data_dir_base: Base directory for document data
                           Each document is at data_dir_base/{document_name}/
        """
        self.data_dir_base = data_dir_base
        self.label2id = get_label2id()
        self.id2label = get_id2label()

    def _get_task_dir(self, task_id: str) -> str:
        """
        Get task directory.

        Args:
            task_id: Task ID

        Returns:
            Path to task directory: data_dir_base/{task_id}/
        """
        if self.data_dir_base is None:
            raise ValueError("data_dir_base must be configured")
        return os.path.join(self.data_dir_base, task_id)

    def predict_single(
        self,
        task_id: str,
        document_name: str,
        return_original: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on a single document.

        Args:
            task_id: Task ID (folder under data_dir_base)
            document_name: Document name (without .json extension)
            return_original: If True, merge predictions with original JSON

        Returns:
            Dict with prediction results
        """
        start_time = time.time()

        # Resolve paths: data_dir_base/{task_id}/
        task_dir = self._get_task_dir(task_id)
        if not os.path.isdir(task_dir):
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        # JSON path: task_dir/{document_name}.json
        json_path = os.path.join(task_dir, f"{document_name}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Image directory: task_dir/images/
        img_dir = os.path.join(task_dir, "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Load document
        json_info = {
            "filepath": json_path,
            "filename": f"{document_name}.json",
            "doc_name": document_name,
        }
        doc_data = load_single_document(json_info, img_dir)
        if doc_data is None:
            raise ValueError(f"Failed to load document: {document_name}")

        # Get model loader
        loader = get_model_loader()
        if not loader.is_loaded:
            raise RuntimeError("Model not loaded. Initialize model first.")

        # Process document (tokenize)
        processed = self._process_document(doc_data, loader.tokenizer)
        if processed is None:
            raise ValueError(f"Failed to process document: {document_name}")

        # Create batch
        collator = HRDocDocumentLevelCollator(
            tokenizer=loader.tokenizer,
            max_length=512,
        )
        batch = collator([processed])

        # Run inference
        batch_wrapped = wrap_batch(batch)
        batch_wrapped = batch_wrapped.to(loader.device)

        with torch.no_grad():
            for sample in batch_wrapped:
                pred = loader.predictor.predict(sample)
                break  # Only one document

        inference_time = (time.time() - start_time) * 1000  # ms

        # Build result
        if return_original:
            return self._build_merged_result(
                document_name, pred, json_path, inference_time
            )
        else:
            return self._build_result(document_name, pred, inference_time)

    def _process_document(
        self,
        doc_data: Dict,
        tokenizer,
    ) -> Optional[Dict]:
        """
        Process a single document (tokenize all pages).

        Reuses logic from InferenceDataLoader._process_document.
        """
        document_name = doc_data["document_name"]
        pages = doc_data["pages"]

        all_chunks = []
        all_parent_ids = []
        all_relations = []

        for page in pages:
            page_number = page["page_number"]
            tokens = page["tokens"]
            bboxes = page["bboxes"]
            labels = page["ner_tags"]
            image = page["image"]
            line_ids = page["line_ids"]
            page_parent_ids = page["line_parent_ids"]
            page_relations = page["line_relations"]

            chunks = tokenize_page_with_line_boundary(
                tokenizer=tokenizer,
                tokens=tokens,
                bboxes=bboxes,
                labels=labels,
                line_ids=line_ids,
                max_length=512,
                label2id=self.label2id,
                image=image,
                page_number=page_number,
                label_all_tokens=True,
            )

            all_chunks.extend(chunks)
            all_parent_ids.extend(page_parent_ids)
            all_relations.extend(page_relations)

        if len(all_chunks) == 0:
            return None

        return {
            "document_name": document_name,
            "num_pages": len(pages),
            "chunks": all_chunks,
            "line_parent_ids": all_parent_ids,
            "line_relations": all_relations,
        }

    def _build_result(
        self,
        document_name: str,
        pred,
        inference_time: float,
    ) -> Dict[str, Any]:
        """Build prediction result dict."""
        sorted_line_ids = sorted(pred.line_classes.keys())

        results = []
        for idx, line_id in enumerate(sorted_line_ids):
            pred_class = pred.line_classes.get(line_id, 0)
            pred_parent = pred.line_parents[idx] if idx < len(pred.line_parents) else -1
            pred_relation = pred.line_relations[idx] if idx < len(pred.line_relations) else 0

            results.append({
                "line_id": line_id,
                "class_label": ID2LABEL.get(pred_class, f"cls_{pred_class}"),
                "class_id": pred_class,
                "parent_id": pred_parent,
                "relation": RELATION_LABELS.get(pred_relation, f"rel_{pred_relation}"),
                "relation_id": pred_relation,
            })

        return {
            "document_name": document_name,
            "num_lines": pred.num_lines,
            "results": results,
            "inference_time_ms": round(inference_time, 2),
        }

    def _build_merged_result(
        self,
        document_name: str,
        pred,
        json_path: str,
        inference_time: float,
    ) -> Dict[str, Any]:
        """Build result merged with original JSON data."""
        # Load original JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # Build prediction map
        sorted_line_ids = sorted(pred.line_classes.keys())
        pred_map = {}
        for idx, line_id in enumerate(sorted_line_ids):
            pred_class = pred.line_classes.get(line_id, 0)
            pred_parent = pred.line_parents[idx] if idx < len(pred.line_parents) else -1
            pred_relation = pred.line_relations[idx] if idx < len(pred.line_relations) else 0

            pred_map[line_id] = {
                "class": ID2LABEL.get(pred_class, f"cls_{pred_class}"),
                "parent_id": pred_parent,
                "relation": RELATION_LABELS.get(pred_relation, f"rel_{pred_relation}"),
            }

        # Merge with original data
        output_data = []
        for item in original_data:
            new_item = dict(item)
            line_id = item.get("line_id", item.get("id", -1))
            if isinstance(line_id, str):
                try:
                    line_id = int(line_id)
                except ValueError:
                    line_id = -1

            if line_id in pred_map:
                new_item.update(pred_map[line_id])

            output_data.append(new_item)

        return {
            "document_name": document_name,
            "num_lines": pred.num_lines,
            "inference_time_ms": round(inference_time, 2),
            "data": output_data,
        }


    def predict_with_construct(
        self,
        task_id: str,
        document_name: str,
        full_tree: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference with Construct model for TOC generation.

        流程：
        1. Stage 1: 提取 line_features + 分类
        2. 保存 features.pt 到 upload/{task_id}/
        3. Construct: 生成 TOC (toc_parent, toc_sibling)
        4. 保存 construct.json 到 upload/{task_id}/

        Args:
            task_id: Task ID (folder under data_dir_base)
            document_name: Document name (without .json extension)
            full_tree: 是否构建完整树（包含非 section 内容），默认 True

        Returns:
            Dict with construct results
        """
        import time
        start_time = time.time()

        # Resolve paths
        task_dir = self._get_task_dir(task_id)
        if not os.path.isdir(task_dir):
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        json_path = os.path.join(task_dir, f"{document_name}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        img_dir = os.path.join(task_dir, "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        # Load document
        json_info = {
            "filepath": json_path,
            "filename": f"{document_name}.json",
            "doc_name": document_name,
        }
        doc_data = load_single_document(json_info, img_dir)
        if doc_data is None:
            raise ValueError(f"Failed to load document: {document_name}")

        # Get model loader
        loader = get_model_loader()
        if not loader.is_loaded:
            raise RuntimeError("Model not loaded. Initialize model first.")

        # Process document
        processed = self._process_document(doc_data, loader.tokenizer)
        if processed is None:
            raise ValueError(f"Failed to process document: {document_name}")

        # Create batch
        collator = HRDocDocumentLevelCollator(
            tokenizer=loader.tokenizer,
            max_length=512,
        )
        batch = collator([processed])

        # Wrap and move to device
        batch_wrapped = wrap_batch(batch)
        batch_wrapped = batch_wrapped.to(loader.device)

        # Stage 1: Extract features only
        with torch.no_grad():
            for sample in batch_wrapped:
                features = loader.predictor.extract_features(sample)
                break

        # Save features.pt
        features_path = os.path.join(task_dir, "features.pt")
        torch.save({
            "line_features": features["line_features"].cpu(),
            "line_mask": features["line_mask"].cpu(),
            "line_classes": features["line_classes"],
            "num_lines": features["num_lines"],
            "line_ids": features["line_ids"],
        }, features_path)
        logger.info(f"Saved features to: {features_path}")

        # Construct inference (if model available)
        construct_result = None
        if loader.has_construct_model:
            construct_result = self._run_construct_inference(
                features, loader.construct_model, loader.device
            )

            # 读取原始 JSON 获取 text
            with open(json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

            # 构建 line_id -> text 映射
            line_text_map = {}
            for item in original_data:
                lid = item.get("line_id", item.get("id", -1))
                if isinstance(lid, str):
                    try:
                        lid = int(lid)
                    except ValueError:
                        continue
                line_text_map[lid] = item.get("text", "")

            # 合并 text 到 construct_result
            if construct_result and "predictions" in construct_result:
                for pred in construct_result["predictions"]:
                    pred["text"] = line_text_map.get(pred["line_id"], "")

                # 构建 section_ids 集合
                section_ids = set(pred["line_id"] for pred in construct_result["predictions"])

                # 准备全量数据（按 line_id 排序，作为阅读顺序）
                all_lines = []
                for item in original_data:
                    lid = item.get("line_id", item.get("id", -1))
                    if isinstance(lid, str):
                        try:
                            lid = int(lid)
                        except ValueError:
                            continue
                    all_lines.append({
                        "line_id": lid,
                        "text": item.get("text", ""),
                        "class": item.get("class", item.get("category", "")),
                    })
                all_lines.sort(key=lambda x: x["line_id"])

                if full_tree:
                    # full_tree=True: 返回包含所有行的扁平格式A
                    full_predictions = flatten_full_tree_to_format_a(
                        section_predictions=construct_result["predictions"],
                        all_lines=all_lines,
                        section_ids=section_ids,
                    )
                    construct_result["predictions"] = full_predictions

                # 转换为嵌套树结构（用于可视化/日志）
                toc_tree = build_tree_from_parents(
                    [p for p in construct_result["predictions"] if p.get("is_section", True)],
                    id_key="line_id",
                    parent_key="parent_id",
                )
                construct_result["toc_tree"] = toc_tree
                construct_result["full_tree"] = full_tree

            # Save construct.json
            construct_path = os.path.join(task_dir, "construct.json")
            with open(construct_path, 'w', encoding='utf-8') as f:
                json.dump(construct_result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved construct result to: {construct_path}")

            # 打印 TOC 树结构（使用嵌套的 toc_tree）
            tree_lines = format_toc_tree(toc_tree)
            logger.info(f"\n{'='*60}\n TOC Tree ({document_name})\n{'='*60}\n" + "\n".join(tree_lines))

        inference_time = (time.time() - start_time) * 1000

        return {
            "document_name": document_name,
            "num_lines": features["num_lines"],
            "features_path": features_path,
            "construct_result": construct_result,
            "inference_time_ms": round(inference_time, 2),
        }

    def _run_construct_inference(
        self,
        features: Dict,
        construct_model,
        device,
        section_label_id: int = 4,  # section 类的 label id
    ) -> Dict[str, Any]:
        """Run Construct model inference.

        只对 section 类的行进行 TOC 构建，与训练保持一致。
        """
        line_features = features["line_features"].to(device)
        line_mask = features["line_mask"].to(device)
        num_lines = features["num_lines"]
        line_ids = features["line_ids"]
        line_classes = features["line_classes"]

        # 过滤只保留 section 类 (与训练一致)
        section_indices = []  # 原始索引
        section_line_ids = []  # line_id
        for idx, line_id in enumerate(line_ids):
            if idx < num_lines:
                cls_id = line_classes.get(line_id, 0)
                if cls_id == section_label_id:
                    section_indices.append(idx)
                    section_line_ids.append(line_id)

        num_sections = len(section_indices)
        if num_sections == 0:
            logger.info("[Construct] No sections found, skipping TOC inference")
            return {
                "num_lines": 0,
                "num_sections": 0,
                "predictions": [],
            }

        # 提取 section 的特征
        section_features = line_features[section_indices]  # [S, H]
        section_features = section_features.unsqueeze(0)  # [1, S, H]
        section_mask = torch.ones(1, num_sections, dtype=torch.bool, device=device)

        # Categories (all sections)
        categories = torch.full((1, num_sections), section_label_id, dtype=torch.long, device=device)

        # Reading order (按 section 在文档中的顺序)
        reading_orders = torch.arange(num_sections, device=device).unsqueeze(0)

        # Run Construct model
        with torch.no_grad():
            outputs = construct_model(
                region_features=section_features,
                categories=categories,
                region_mask=section_mask,
                reading_orders=reading_orders,
            )

        # Decode predictions (格式B: 自指向方案)
        parent_preds = outputs["parent_logits"].argmax(dim=-1)[0].cpu().tolist()  # [S]
        sibling_preds = outputs["sibling_logits"].argmax(dim=-1)[0].cpu().tolist()  # [S]

        # 反向转换: 格式B → 格式A
        # 格式B: hierarchical_parent + sibling (自指向方案)
        # 格式A: ref_parent + relation (顶层节点 parent=-1)
        ref_parents, relations = resolve_ref_parents_and_relations(parent_preds, sibling_preds)

        # Build result - 映射回原始 line_id
        results = []
        for sec_idx, line_id in enumerate(section_line_ids):
            ref_parent_sec_idx = ref_parents[sec_idx]

            # 映射回 line_id (-1 表示顶层节点)
            parent_line_id = section_line_ids[ref_parent_sec_idx] if ref_parent_sec_idx >= 0 else -1

            results.append({
                "line_id": line_id,
                "parent_id": parent_line_id,
                "relation": relations[sec_idx],
                "section_index": sec_idx,  # section 空间的索引
            })

        logger.info(f"[Construct] TOC inference done: {num_sections} sections")

        return {
            "num_lines": num_lines,
            "num_sections": num_sections,
            "predictions": results,
        }


# Global service instance
_infer_service: Optional[InferenceService] = None


def get_infer_service(data_dir_base: str = None) -> InferenceService:
    """Get or create inference service instance."""
    global _infer_service
    if _infer_service is None:
        _infer_service = InferenceService(data_dir_base=data_dir_base)
    elif data_dir_base and _infer_service.data_dir_base != data_dir_base:
        _infer_service.data_dir_base = data_dir_base
    return _infer_service

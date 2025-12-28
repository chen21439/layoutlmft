#!/usr/bin/env python
# coding=utf-8
"""
Inference Service - Core inference logic

Reuses existing inference code from examples/stage/.

Directory structure:
    data_dir_base/
    └── {document_name}/
        ├── test/
        │   └── {document_name}.json
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

    def _get_data_dir(self, document_name: str) -> str:
        """
        Get data directory for a specific document.

        Args:
            document_name: Document name

        Returns:
            Path to document's data directory: data_dir_base/{document_name}/
        """
        if self.data_dir_base is None:
            raise ValueError("data_dir_base must be configured")
        return os.path.join(self.data_dir_base, document_name)

    def predict_single(
        self,
        document_name: str,
        return_original: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on a single document.

        Args:
            document_name: Document name (without .json extension)
            return_original: If True, merge predictions with original JSON

        Returns:
            Dict with prediction results
        """
        start_time = time.time()

        # Resolve paths: data_dir_base/{document_name}/
        data_dir = self._get_data_dir(document_name)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Document directory not found: {data_dir}")

        # JSON path: data_dir/test/{document_name}.json
        json_dir = os.path.join(data_dir, "test")
        if not os.path.isdir(json_dir):
            json_dir = data_dir

        json_path = os.path.join(json_dir, f"{document_name}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Image directory: data_dir/images/
        img_dir = os.path.join(data_dir, "images")
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

#!/usr/bin/env python
# coding=utf-8
"""
Inference Service - Core inference logic

使用 comp_hrdoc 的组件进行推理，与训练/评估代码保持一致。

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
from lxml import etree

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
EXAMPLES_ROOT = os.path.join(PROJECT_ROOT, "examples")
sys.path.insert(0, EXAMPLES_ROOT)
STAGE_ROOT = os.path.join(EXAMPLES_ROOT, "stage")
sys.path.insert(0, STAGE_ROOT)

# 数据加载仍使用 stage 目录的模块（训练也使用这些）
from data.inference_data_loader import load_single_document
from data.hrdoc_data_loader import tokenize_page_with_line_boundary, get_label2id, get_id2label
from joint_data_collator import HRDocDocumentLevelCollator
from layoutlmft.data.labels import ID2LABEL
from comp_hrdoc.utils.tree_utils import (
    build_tree_from_parents,
    build_doc_tree_with_nodes,
    format_toc_tree,
    format_tree_from_parents,
    flatten_full_tree_to_format_a,
    visualize_toc,
    tree_insertion_decode,
    resolve_ref_parents_and_relations,
)
from comp_hrdoc.metrics.classification import normalize_class

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

    def _find_document_name(self, task_id: str) -> str:
        """
        Auto-detect document name from task directory.

        Args:
            task_id: Task ID

        Returns:
            Document name (without .json extension)

        Raises:
            FileNotFoundError: If no json file found or multiple json files exist
        """
        task_dir = self._get_task_dir(task_id)
        if not os.path.isdir(task_dir):
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        # 查找所有 .json 文件（排除 _construct.json, _split_result.json 等生成文件）
        json_files = [
            f for f in os.listdir(task_dir)
            if f.endswith('.json')
            and not f.endswith('_construct.json')
            and not f.endswith('_split_result.json')
        ]

        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON file found in {task_dir}")
        if len(json_files) > 1:
            raise FileNotFoundError(f"Multiple JSON files found in {task_dir}: {json_files}. Please specify document_name.")

        # 返回不带 .json 后缀的文件名
        return json_files[0][:-5]

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

        # Run inference using comp_hrdoc components
        # 提取 line-level 特征（与训练代码一致）
        line_features, line_mask = loader.feature_extractor.extract_from_batch(batch)

        # 分类预测（与 train_doc.py 评估逻辑一致）
        num_lines = int(line_mask[0].sum().item())  # 只有一个文档
        valid_features = line_features[0, :num_lines]  # [num_lines, H]
        cls_logits = loader.feature_extractor.model.cls_head(valid_features)
        cls_preds = cls_logits.argmax(dim=-1)  # [num_lines]

        # line_id 就是索引（与训练数据格式一致）
        line_ids_list = list(range(num_lines))

        # 构建 pred 对象（模拟原有格式）
        class PredResult:
            pass
        pred = PredResult()
        pred.num_lines = num_lines
        pred.line_classes = {i: cls_preds[i].item() for i in range(num_lines)}
        pred.line_parents = [-1] * num_lines  # 单独分类没有父节点预测
        pred.line_relations = [0] * num_lines  # 默认 relation

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

    def _parse_bbox(self, bbox_str: str, page: int = 1) -> Optional[List[Dict]]:
        """
        解析 bbox 字符串为 coordinates 格式

        Args:
            bbox_str: bbox 字符串，格式如 "158.00,438.87,477.50,704.28"
            page: 页码，默认为 1

        Returns:
            coordinates 列表: [{"page": 1, "x0": 158.0, "y0": 438.87, "x1": 477.5, "y1": 704.28, "coord_origin": "TOPLEFT"}]
        """
        if not bbox_str:
            return None

        try:
            parts = bbox_str.replace(' ', ',').split(',')
            coords = [float(x.strip()) for x in parts if x.strip()]

            if len(coords) < 4:
                logger.warning(f"Invalid bbox format: {bbox_str}")
                return None

            return [{
                "page": page,
                "x0": coords[0],
                "y0": coords[1],
                "x1": coords[2],
                "y1": coords[3],
                "coord_origin": "TOPLEFT",
            }]
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse bbox '{bbox_str}': {e}")
            return None

    def _parse_table_xml(self, xml_text: str, table_index: int) -> List[Dict]:
        """
        解析 table 的 XML text 字段，拆分为多个 td/th 元素

        Args:
            xml_text: table 的 XML 内容
            table_index: 当前 table 的序号（全局计数，0-based）

        Returns:
            拆分后的 cell 列表，每个 cell 包含:
            {
                "element_id": "t002-r001-c001-p001",
                "text_preview": "项目编号：",
                "table_index": 0,
                "row_index": 0,
                "cell_index": 0,
                "is_header": True,  # th=True, td=False
                "coordinates": [{"page": 2, "x0": ..., "y0": ..., ...}],
            }
        """
        cells = []

        try:
            # 解析 XML
            root = etree.fromstring(xml_text.encode('utf-8'))

            # 从 <table> 标签获取 bbox 和 page
            table_bbox = root.get('bbox')
            table_page = int(root.get('page', '1'))
            table_coords = self._parse_bbox(table_bbox, table_page) if table_bbox else None

            # 遍历 <tr> 行
            for row_idx, tr in enumerate(root.findall('.//tr')):
                # 直接遍历 tr 的子元素，保持 DOM 顺序
                cell_idx = 0
                for cell_elem in tr:
                    # 只处理 th 和 td 标签
                    if cell_elem.tag not in ('th', 'td'):
                        continue

                    is_header = (cell_elem.tag == 'th')

                    # 提取所有 <p> 文本
                    texts = []
                    p_elements = cell_elem.findall('.//p')
                    element_id = None

                    for p in p_elements:
                        p_id = p.get('id')
                        if p_id and element_id is None:
                            element_id = p_id
                        text = ''.join(p.itertext()).strip()
                        if text:
                            texts.append(text)

                    # 合并文本
                    full_text = ' '.join(texts)

                    # 如果没有 element_id，生成一个默认的
                    if element_id is None:
                        element_id = f"t{table_index:03d}-r{row_idx:03d}-c{cell_idx:03d}"

                    # 构建 cell 结构
                    cell = {
                        "element_id": element_id,
                        "text_preview": full_text[:100] if len(full_text) > 100 else full_text,
                        "table_index": table_index,
                        "row_index": row_idx,
                        "cell_index": cell_idx,
                        "is_header": is_header,
                        "coordinates": table_coords,  # 使用 table 的坐标
                    }

                    cells.append(cell)
                    cell_idx += 1

        except etree.XMLSyntaxError as e:
            logger.error(f"Failed to parse table XML: {e}")
            logger.error(f"XML content: {xml_text[:200]}...")
        except Exception as e:
            logger.error(f"Unexpected error parsing table XML: {e}")

        return cells

    def _build_split_result(
        self,
        predictions: List[Dict],
        task_id: str,
        base_level: int = 1,
    ) -> Dict[str, Any]:
        """
        从扁平格式A构建 split_result 嵌套树格式

        使用 tree_utils.build_doc_tree_with_nodes 正确处理 relation:
        - contain: 真正的父子关系
        - equality: 兄弟关系（parent_id 指向左兄弟）

        Args:
            predictions: 扁平格式A的预测结果
            task_id: 任务ID
            base_level: 顶层 section 的 level 值

        Returns:
            split_result 格式的字典
        """
        if not predictions:
            return {"document": task_id, "total_elements": 0, "total_sections": 0, "sections": []}

        # 分离 section 和非 section
        sections = [p for p in predictions if p.get("is_section", False)]
        if not sections:
            return {"document": task_id, "total_elements": len(predictions), "total_sections": 0, "sections": []}

        # 构建 line_id -> section_index 映射
        section_line_ids = [sec["line_id"] for sec in sections]
        line_id_to_sec_idx = {lid: idx for idx, lid in enumerate(section_line_ids)}

        # 提取 parent_ids 和 relations（转换为 section_index 空间）
        parent_ids = []
        relations = []
        for sec in sections:
            parent_line_id = sec.get("parent_id", -1)
            relation = sec.get("relation", "contain")

            if parent_line_id == -1 or parent_line_id not in line_id_to_sec_idx:
                parent_ids.append(-1)
            else:
                parent_ids.append(line_id_to_sec_idx[parent_line_id])
            relations.append(relation)

        # 使用 tree_utils 构建树（正确处理 equality 关系）
        root_node, nodes = build_doc_tree_with_nodes(parent_ids, relations)

        # 构建 section_nodes 字典
        section_nodes = {}
        for sec_idx, sec in enumerate(sections):
            node_id = sec["line_id"]
            node = nodes[sec_idx]

            # 从 Node.parent 获取真正的层级父节点
            if node.parent is None or node.parent.name == 'ROOT':
                hierarchical_parent_id = -1
            else:
                parent_sec_idx = node.parent.info['index']
                hierarchical_parent_id = section_line_ids[parent_sec_idx]

            section_nodes[node_id] = {
                "title": sec.get("text", ""),
                "level": base_level + node.depth - 1,  # depth 从 1 开始
                "start_index": 0,
                "end_index": 0,
                "element_count": 0,
                "elements": [],
                "children": [],
                "_line_id": node_id,
                "_hierarchical_parent_id": hierarchical_parent_id,
            }

        # 构建父子关系
        roots = []
        for node_id, sec_node in section_nodes.items():
            parent_id = sec_node["_hierarchical_parent_id"]
            if parent_id == -1 or parent_id not in section_nodes:
                roots.append(sec_node)
            else:
                section_nodes[parent_id]["children"].append(sec_node)

        # 按阅读顺序分配 elements（处理 table 展开）
        all_items = sorted(predictions, key=lambda x: x.get("line_id", 0))
        current_section_id = None
        line_to_section = {}
        for item in all_items:
            line_id = item.get("line_id")
            if item.get("is_section", False):
                current_section_id = line_id
            line_to_section[line_id] = current_section_id

        element_seq_no = {}
        table_index_counter = 0  # 全局 table 计数器
        global_element_idx = 0

        for idx, item in enumerate(all_items):
            line_id = item.get("line_id")
            is_section = item.get("is_section", False)
            section_id = line_to_section.get(line_id) if not is_section else line_id

            if section_id is not None and section_id in section_nodes:
                seq = element_seq_no.get(section_id, 0)

                # 检查是否是 table 元素
                item_class = item.get("class", "").lower()
                if item_class == "table":
                    # 展开 table 为多个 cell
                    xml_text = item.get("text", "")
                    if xml_text.strip().startswith("<table"):
                        cells = self._parse_table_xml(xml_text, table_index_counter)
                        for cell in cells:
                            # 将 cell 转换为 element
                            element = {
                                "type": "table_cell",
                                "text_preview": cell["text_preview"],
                                "index": global_element_idx,
                                "element_id": cell["element_id"],
                                "seq_no": seq,
                                "coordinates": cell["coordinates"],
                                "table_index": cell["table_index"],
                                "row_index": cell["row_index"],
                                "cell_index": cell["cell_index"],
                                "is_header": cell["is_header"],
                            }
                            section_nodes[section_id]["elements"].append(element)
                            seq += 1
                            global_element_idx += 1
                        table_index_counter += 1
                    else:
                        # text 不是 XML，当作普通元素处理
                        element = self._build_element(item, global_element_idx, seq)
                        section_nodes[section_id]["elements"].append(element)
                        seq += 1
                        global_element_idx += 1
                else:
                    # 非 table 元素，正常处理
                    element = self._build_element(item, global_element_idx, seq)
                    section_nodes[section_id]["elements"].append(element)
                    seq += 1
                    global_element_idx += 1

                element_seq_no[section_id] = seq

        # 计算 element_count 和 start_index / end_index
        global_index = [0]
        def calc_indices(node):
            node["element_count"] = len(node["elements"])
            node["start_index"] = global_index[0]
            global_index[0] += len(node["elements"])
            for child in node["children"]:
                calc_indices(child)
            node["end_index"] = global_index[0]

        for root in roots:
            calc_indices(root)

        # 清理内部字段
        def clean_fields(node):
            node.pop("_line_id", None)
            node.pop("_hierarchical_parent_id", None)
            for child in node["children"]:
                clean_fields(child)

        for root in roots:
            clean_fields(root)

        return {
            "document": task_id,
            "total_elements": len(predictions),
            "total_sections": len(sections),
            "sections": roots,
        }

    def _build_element(self, line: Dict, index: int, seq_no: int) -> Dict:
        """
        构建单个 Element 结构

        注意：此方法处理非 table 元素，table 元素在 _build_split_result 中直接处理
        """
        line_id = line.get("line_id", index)
        text = line.get("text", "")
        cls = line.get("class", "")

        # 判断是否是 table_cell（预处理的 table_index 字段）
        is_table_cell = "table_index" in line
        element_type = "table_cell" if is_table_cell else "paragraph"
        prefix = "tc" if is_table_cell else "p"

        # element_id：如果是 table_cell 且有预设 element_id，使用它；否则生成
        if "element_id" in line:
            element_id = line["element_id"]
        else:
            element_id = f"{prefix}_{line_id:03d}"

        # 坐标转换
        location = line.get("location")
        coordinates = None
        if location:
            coordinates = []
            for loc in location:
                coordinates.append({
                    "x0": loc.get("l", loc.get("x0", 0)),
                    "y0": loc.get("t", loc.get("y0", 0)),
                    "x1": loc.get("r", loc.get("x1", 0)),
                    "y1": loc.get("b", loc.get("y1", 0)),
                    "page": loc.get("page", 1),
                    "originalText": text,
                })

        element = {
            "type": element_type,
            "text_preview": text[:100] if len(text) > 100 else text,
            "index": index,
            "element_id": element_id,
            "seq_no": seq_no,
            "coordinates": coordinates,
        }

        # 添加 table_cell 特有字段
        if is_table_cell:
            element["table_index"] = line.get("table_index", 0)
            element["row_index"] = line.get("row_index", 0)
            element["cell_index"] = line.get("cell_index", 0)
            element["is_header"] = line.get("is_header", False)

        return element

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
        4. 保存 {document_name}_construct.json 到 upload/{task_id}/

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

        # Stage 1: Extract features using comp_hrdoc components（与训练代码一致）
        # 如果启用 attention_pool_construct，需要返回 token-level hidden states
        use_attention_pool = loader.attention_pool_construct
        if use_attention_pool:
            line_features, line_mask, token_hidden_states, token_line_ids = \
                loader.feature_extractor.extract_from_batch(batch, return_token_hidden=True)
        else:
            line_features, line_mask = loader.feature_extractor.extract_from_batch(batch)

        # 分类预测（与 train_doc.py 评估逻辑一致）
        num_lines = int(line_mask[0].sum().item())  # 只有一个文档
        valid_features = line_features[0, :num_lines]  # [num_lines, H]
        cls_logits = loader.feature_extractor.model.cls_head(valid_features)
        cls_preds = cls_logits.argmax(dim=-1)  # [num_lines]

        # line_id 就是索引（与训练数据格式一致）
        line_ids_list = list(range(num_lines))

        # 构建 features 字典（与后续 construct 推理兼容）
        # 注意：去掉 batch 维度，因为推理时只有一个文档
        features = {
            "line_features": line_features[0],  # [max_lines, H]
            "line_mask": line_mask[0],  # [max_lines]
            "line_classes": {i: cls_preds[i].item() for i in range(num_lines)},
            "num_lines": num_lines,
            "line_ids": line_ids_list,
        }

        # 输出 Stage1 分类预测统计
        section_label_id = 4  # section 的 label id
        section_line_ids = [i for i, cls_id in features["line_classes"].items() if cls_id == section_label_id]
        logger.info(f"[Stage1] 分类预测: {num_lines} lines, {len(section_line_ids)} sections")
        logger.info(f"[Stage1] section line_ids: {section_line_ids}")

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
            # 传入 token 数据（用于 attention pool）
            token_data = None
            if use_attention_pool:
                token_data = {
                    "token_hidden_states": token_hidden_states,
                    "token_line_ids": token_line_ids,
                }
            construct_result = self._run_construct_inference(
                features, loader.construct_model, loader.device,
                token_data=token_data,
            )

            # 读取原始 JSON 获取 text
            with open(json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

            # 辅助函数：将原始 page + box 转换为标准 location 格式
            def build_location(item):
                """
                原始格式: {"page": "0", "box": [267, 72, 351, 86]}
                目标格式: [{"page": 1, "l": 267, "t": 72, "r": 351, "b": 86, "coord_origin": "TOPLEFT"}]
                """
                # 先检查是否已有标准 location 格式
                loc = item.get("location")
                if loc and isinstance(loc, list) and len(loc) > 0 and isinstance(loc[0], dict):
                    # 已是标准格式，确保有 coord_origin
                    result = []
                    for l in loc:
                        l = dict(l)
                        if "coord_origin" not in l:
                            l["coord_origin"] = "TOPLEFT"
                        result.append(l)
                    return result

                # 从 page + box 构建
                page = item.get("page")
                box = item.get("box", item.get("bbox"))

                if box is None:
                    return None

                # page 可能是字符串，转为整数
                if page is not None:
                    try:
                        page = int(page)
                    except (ValueError, TypeError):
                        page = 1
                else:
                    page = 1

                # box 格式: [l, t, r, b] 或 [x1, y1, x2, y2]
                if isinstance(box, list) and len(box) >= 4:
                    return [{
                        "page": page,
                        "l": float(box[0]),
                        "t": float(box[1]),
                        "r": float(box[2]),
                        "b": float(box[3]),
                        "coord_origin": "TOPLEFT",
                    }]

                return None

            # 构建 line_id -> (text, location) 映射
            line_info_map = {}
            # DEBUG: 打印第一个 item 的字段
            if original_data:
                logger.info(f"[DEBUG] First item keys: {list(original_data[0].keys())}")
                logger.info(f"[DEBUG] First item page={original_data[0].get('page')}, box={original_data[0].get('box')}")
                first_loc = build_location(original_data[0])
                logger.info(f"[DEBUG] build_location result: {first_loc}")
            for item in original_data:
                lid = item.get("line_id", item.get("id", -1))
                if isinstance(lid, str):
                    try:
                        lid = int(lid)
                    except ValueError:
                        continue
                line_info_map[lid] = {
                    "text": item.get("text", ""),
                    "location": build_location(item),
                }

            # 合并 text 和 location 到 construct_result
            if construct_result and "predictions" in construct_result:
                for pred in construct_result["predictions"]:
                    info = line_info_map.get(pred["line_id"], {})
                    pred["text"] = info.get("text", "")
                    pred["location"] = info.get("location")
                # DEBUG: 打印第一个 prediction 的 location
                if construct_result["predictions"]:
                    first_pred = construct_result["predictions"][0]
                    logger.info(f"[DEBUG] First prediction: line_id={first_pred.get('line_id')}, location={first_pred.get('location')}")

                # 构建 section_ids 集合
                section_ids = set(pred["line_id"] for pred in construct_result["predictions"])

                # 准备全量数据（按 line_id 排序，作为阅读顺序）
                # 使用 Stage1 预测的 class，而不是原始 JSON 的 class
                line_classes = features["line_classes"]  # {line_id: class_id}
                all_lines = []
                for item in original_data:
                    lid = item.get("line_id", item.get("id", -1))
                    if isinstance(lid, str):
                        try:
                            lid = int(lid)
                        except ValueError:
                            continue
                    # 使用 Stage1 预测的 class，回退到原始 JSON
                    pred_class_id = line_classes.get(lid)
                    if pred_class_id is not None:
                        pred_class = ID2LABEL.get(pred_class_id, f"cls_{pred_class_id}")
                    else:
                        pred_class = normalize_class(item.get("class", item.get("category", "")))
                    all_lines.append({
                        "line_id": lid,
                        "text": item.get("text", ""),
                        "class": pred_class,
                        "location": build_location(item),
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

            # Save {document_name}_construct.json
            construct_path = os.path.join(task_dir, f"{document_name}_construct.json")
            with open(construct_path, 'w', encoding='utf-8') as f:
                json.dump(construct_result, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved construct result to: {construct_path}")

            # Save {document_name}_split_result.json (嵌套树格式)
            try:
                split_result = self._build_split_result(
                    predictions=construct_result["predictions"],
                    task_id=f"task_{os.path.basename(task_dir)}",
                )
                split_result_path = os.path.join(task_dir, f"{document_name}_split_result.json")
                with open(split_result_path, 'w', encoding='utf-8') as f:
                    json.dump(split_result, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved split_result to: {split_result_path}")
            except Exception as e:
                logger.warning(f"Failed to save split_result: {e}")

            # 打印 TOC 树结构（使用 visualize_toc，和训练一致）
            format_b = construct_result.get("format_b", {})
            if format_b and format_b.get("pred_parents"):
                # 获取 section 文本（只取 section，和 format_b 对应）
                section_preds = [p for p in construct_result["predictions"] if p.get("is_section", True)]
                section_texts = [p.get("text", f"[{p['line_id']}]") for p in section_preds]
                vis_str = visualize_toc(
                    texts=section_texts,
                    pred_parents=format_b["pred_parents"],
                    gt_parents=None,  # 推理模式，无 ground truth
                    sample_id=document_name,
                    pred_siblings=format_b.get("pred_siblings"),
                )
                logger.info(vis_str)

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
        token_data: Dict = None,  # token-level 数据（用于 attention pool）
    ) -> Dict[str, Any]:
        """Run Construct model inference.

        只对 section 类的行进行 TOC 构建，与训练保持一致。

        Args:
            features: 包含 line_features, line_mask 等
            construct_model: Construct 模型
            device: 设备
            section_label_id: section 类的 label id
            token_data: token-level 数据（当 attention_pool_construct=True 时使用）
                - token_hidden_states: [total_chunks, seq_len, H]
                - token_line_ids: [total_chunks, seq_len]
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
        # 如果有 token_data，使用 attention pooling；否则使用 mean-pooled line_features
        section_tokens = None
        section_token_mask = None

        if token_data is not None:
            # 使用 token-level 特征 + AttentionPooling（与训练一致）
            from comp_hrdoc.models.modules.attention_pooling import extract_section_tokens

            token_hidden_states = token_data["token_hidden_states"].to(device)
            token_line_ids = token_data["token_line_ids"].to(device)

            section_line_indices = torch.tensor(section_line_ids, device=device)
            section_tokens, section_token_mask = extract_section_tokens(
                hidden_states=token_hidden_states,
                line_ids=token_line_ids,
                section_line_indices=section_line_indices,
                max_tokens_per_section=64,
            )
            # section_tokens: [num_sections, max_tokens, H]
            # section_token_mask: [num_sections, max_tokens]

            # 添加 batch 维度
            section_tokens = section_tokens.unsqueeze(0)  # [1, S, T, H]
            section_token_mask = section_token_mask.unsqueeze(0)  # [1, S, T]

            logger.info(f"[Construct] Using AttentionPooling: section_tokens shape={section_tokens.shape}")

        # Mean-pooled section features（用于非 attention pool 或作为 fallback）
        section_features = line_features[section_indices]  # [S, H]
        section_features = section_features.unsqueeze(0)  # [1, S, H]
        section_mask = torch.ones(1, num_sections, dtype=torch.bool, device=device)

        # Categories (all sections)
        categories = torch.full((1, num_sections), section_label_id, dtype=torch.long, device=device)

        # Reading order (按 section 在文档中的顺序)
        reading_orders = torch.arange(num_sections, device=device).unsqueeze(0)

        # Run Construct model
        with torch.no_grad():
            if section_tokens is not None:
                # 使用 AttentionPooling
                outputs = construct_model(
                    region_features=section_features,  # 作为 fallback
                    categories=categories,
                    region_mask=section_mask,
                    reading_orders=reading_orders,
                    section_tokens=section_tokens,
                    section_token_mask=section_token_mask,
                )
            else:
                # 使用 mean-pooled features
                outputs = construct_model(
                    region_features=section_features,
                    categories=categories,
                    region_mask=section_mask,
                    reading_orders=reading_orders,
                )

        # Decode predictions (格式B: 自指向方案)
        # 使用论文 Algorithm 1: Tree Insertion Algorithm 进行联合解码
        # 保证 sibling 约束：每个节点的左兄弟一定是其父节点的已有子节点
        parent_preds, sibling_preds = tree_insertion_decode(
            outputs["parent_logits"][0],  # [S, S]
            outputs["sibling_logits"][0],  # [S, S]
            debug=True,  # 启用调试日志
            section_line_ids=section_line_ids,  # 用于日志输出显示 line_id
        )

        # 反向转换: 格式B → 格式A
        # 格式B: hierarchical_parent + sibling (自指向方案)
        # 格式A: ref_parent + relation (顶层节点 parent=-1)
        ref_parents, relations = resolve_ref_parents_and_relations(
            parent_preds, sibling_preds, debug=True  # 启用调试日志
        )

        # [DEBUG] 打印每个节点的 B→A 转换详情
        logger.info(f"[Construct] Format B→A 转换详情 ({num_sections} sections):")
        for sec_idx in range(num_sections):
            line_id = section_line_ids[sec_idx]
            hp = parent_preds[sec_idx]  # hierarchical_parent (section index)
            ls = sibling_preds[sec_idx]  # left_sibling (section index)
            rp = ref_parents[sec_idx]  # ref_parent (section index)
            rel = relations[sec_idx]
            # 映射到 line_id 显示
            hp_lid = section_line_ids[hp] if hp != sec_idx else -1
            ls_lid = section_line_ids[ls] if ls != sec_idx else -1
            rp_lid = section_line_ids[rp] if rp >= 0 else -1
            logger.info(
                f"  [{sec_idx}] line_id={line_id}: "
                f"B(parent={hp_lid}, sibling={ls_lid}) -> "
                f"A(ref_parent={rp_lid}, rel={rel})"
            )

        # 安全检查：修复 Format A 中可能的互指问题
        # 注：使用 Tree Insertion Algorithm 后，Format B 应已保证一致性
        # 此检查作为防御性措施保留
        fixed_count = 0
        processed_pairs = set()
        for i, p in enumerate(ref_parents):
            if p >= 0 and p < len(ref_parents) and i != p:
                pair = (min(i, p), max(i, p))
                if pair not in processed_pairs and ref_parents[p] == i:
                    processed_pairs.add(pair)
                    lid_i = section_line_ids[i]
                    lid_p = section_line_ids[p]
                    # line_id 较小的那个设为 -1（提升为根节点）
                    if lid_i < lid_p:
                        ref_parents[i] = -1
                        relations[i] = "contain"
                        logger.warning(f"[Construct] 修复互指: sec[{i}](line_id={lid_i}).ref_parent={p} -> -1 (line_id较小)")
                    else:
                        ref_parents[p] = -1
                        relations[p] = "contain"
                        logger.warning(f"[Construct] 修复互指: sec[{p}](line_id={lid_p}).ref_parent={i} -> -1 (line_id较小)")
                    fixed_count += 1

        if fixed_count > 0:
            logger.info(f"[Construct] 修复了 {fixed_count} 对互指")

        # Build result - 映射回原始 line_id
        results = []
        for sec_idx, line_id in enumerate(section_line_ids):
            ref_parent_sec_idx = ref_parents[sec_idx]

            # 映射回 line_id (-1 表示顶层节点)
            parent_line_id = section_line_ids[ref_parent_sec_idx] if ref_parent_sec_idx >= 0 else -1

            # 格式 B 的预测值（用于调试 B→A 转换）
            hier_parent_sec_idx = parent_preds[sec_idx]
            left_sib_sec_idx = sibling_preds[sec_idx]
            # 映射到 line_id
            hier_parent_line_id = section_line_ids[hier_parent_sec_idx] if hier_parent_sec_idx != sec_idx else -1  # 自指向=ROOT
            left_sib_line_id = section_line_ids[left_sib_sec_idx] if left_sib_sec_idx != sec_idx else -1  # 自指向=无左兄弟

            results.append({
                "id": line_id,
                "line_id": line_id,
                "parent_id": parent_line_id,
                "relation": relations[sec_idx],
                "class": ID2LABEL.get(section_label_id, "section"),
                "section_index": sec_idx,  # section 空间的索引
                # 格式 B 预测值（调试用）
                "hierarchical_parent": hier_parent_line_id,  # 层级父节点（-1=ROOT）
                "left_sibling": left_sib_line_id,  # 左兄弟（-1=无）
            })

        # [DEBUG] 检测最终结果中的循环
        line_id_to_parent = {r["line_id"]: r["parent_id"] for r in results}
        for r in results:
            lid, pid = r["line_id"], r["parent_id"]
            if pid != -1 and pid in line_id_to_parent:
                if line_id_to_parent[pid] == lid:
                    logger.warning(f"[Construct] 最终结果互指: line_id={lid} <-> {pid}")

        logger.info(f"[Construct] TOC inference done: {num_sections} sections")

        return {
            "num_lines": num_lines,
            "num_sections": num_sections,
            "predictions": results,
            # Format B 数据（用于可视化）
            "format_b": {
                "pred_parents": parent_preds,
                "pred_siblings": sibling_preds,
                "section_line_ids": section_line_ids,
            },
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

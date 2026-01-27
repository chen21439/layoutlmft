#!/usr/bin/env python
# coding=utf-8
"""
Predict1 Router - /predict1 endpoint

返回 split_result 格式的嵌套树结构，用于下游 Pipeline 消费。

层级结构示例：
    [0] 第一章 总则              ← section, level=1
    [1] 本章规定适用于本项目。    ← 段落，属于 [0] 的 elements
    [2] 一、项目概况             ← section, level=2, 是 [0] 的 child
    [3] 项目名称：XXX采购项目     ← 段落，属于 [2] 的 elements
    [4] 二、投标要求             ← section, level=2, 是 [0] 的 child
    [5] 投标人应当具备资质。      ← 段落，属于 [4] 的 elements
    [6] 第二章 招标              ← section, level=1

    Section("第一章 总则"):
        elements: [[0], [1]]     ← 标题本身 + 下一个子章节之前的段落
        children: [[2], [4]]     ← 子章节

    Section("一、项目概况"):
        elements: [[2], [3]]     ← 标题本身 + 直属段落
        children: []
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException

from ..schemas import PredictRequest, ErrorResponse
from ..service.infer_service import get_infer_service
from ..service.model_loader import get_model_loader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict1", tags=["predict1"])


def convert_location_to_coordinates(
    location: Optional[List[Dict]],
    original_text: str = "",
) -> Optional[List[Dict]]:
    """
    将内部 location 格式转换为 split_result 的 coordinates 格式

    输入格式 (内部):
        [{"page": 1, "l": 72, "t": 100, "r": 200, "b": 120, "coord_origin": "TOPLEFT"}]

    输出格式 (split_result):
        [{"x0": 72, "y0": 100, "x1": 200, "y1": 120, "page": 1, "originalText": "..."}]
    """
    if not location:
        return None

    result = []
    for loc in location:
        coord = {
            "x0": loc.get("l", loc.get("x0", 0)),
            "y0": loc.get("t", loc.get("y0", 0)),
            "x1": loc.get("r", loc.get("x1", 0)),
            "y1": loc.get("b", loc.get("y1", 0)),
            "page": loc.get("page", 1),
            "originalText": original_text,
        }
        result.append(coord)

    return result


def build_element(
    line: Dict,
    index: int,
    seq_no: int,
) -> Dict:
    """
    构建单个 Element 结构

    Args:
        line: 原始行数据，包含 line_id, text, class, location 等
        index: 元素在全局的索引
        seq_no: 在当前 section 中的序号

    Returns:
        Element 字典
    """
    line_id = line.get("line_id", index)
    text = line.get("text", "")
    cls = line.get("class", "")

    # 判断元素类型
    is_table_cell = (
        "table_index" in line or
        "row_index" in line or
        cls in ("table", "table_cell")
    )
    element_type = "table_cell" if is_table_cell else "paragraph"

    # 生成 element_id
    prefix = "tc" if is_table_cell else "p"
    element_id = f"{prefix}_{line_id:03d}"

    # 坐标转换
    coordinates = convert_location_to_coordinates(line.get("location"), text)

    element = {
        "type": element_type,
        "text_preview": text[:100] if len(text) > 100 else text,
        "index": index,
        "element_id": element_id,
        "seq_no": seq_no,
        "coordinates": coordinates,
    }

    # 表格特有字段
    if is_table_cell:
        element["table_index"] = line.get("table_index", 0)
        element["row_index"] = line.get("row_index", 0)
        element["cell_index"] = line.get("cell_index", 0)
        element["row_start"] = line.get("row_start", line.get("row_index", 0))
        element["row_end"] = line.get("row_end", line.get("row_index", 0) + 1)
        element["col_start"] = line.get("col_start", line.get("cell_index", 0))
        element["col_end"] = line.get("col_end", line.get("cell_index", 0) + 1)

    return element


def build_nested_tree(
    predictions: List[Dict],
    id_key: str = "line_id",
    parent_key: str = "parent_id",
    base_level: int = 1,
) -> List[Dict]:
    """
    从扁平的格式A预测结果构建嵌套树结构

    核心逻辑：
    1. sections 根据 parent_id 构建父子关系 (children)
    2. 非 section 元素按阅读顺序挂载到"前面最近的 section"的 elements 中
    3. section 标题自身也作为 elements[0]

    Args:
        predictions: 格式A的预测结果列表
            每个: {"line_id": int, "parent_id": int, "text": str, "class": str, "is_section": bool, ...}
        id_key: 节点ID字段名
        parent_key: 父节点ID字段名
        base_level: 顶层 section 的 level 值（默认 1）

    Returns:
        嵌套树结构列表，符合 split_result 格式
    """
    if not predictions:
        return []

    # 分离 section 和非 section
    sections = [p for p in predictions if p.get("is_section", False)]

    if not sections:
        # 没有 section，返回空
        return []

    # ========== 1. 构建 section 节点和父子关系 ==========
    section_nodes = {}
    for sec in sections:
        node_id = sec[id_key]
        section_nodes[node_id] = {
            "title": sec.get("text", ""),
            "level": 0,  # 后面计算
            "start_index": 0,  # TODO: 待用户确认
            "end_index": 0,    # TODO: 待用户确认
            "element_count": 0,
            "elements": [],
            "children": [],
            # 内部字段
            "_line_id": node_id,
            "_parent_id": sec.get(parent_key, -1),
        }

    # 构建父子关系
    roots = []
    for sec in sections:
        node_id = sec[id_key]
        parent_id = sec.get(parent_key, -1)
        node = section_nodes[node_id]

        if parent_id == -1 or parent_id not in section_nodes:
            roots.append(node)
        else:
            section_nodes[parent_id]["children"].append(node)

    # ========== 2. 计算 level（树深度，从 1 开始）==========
    def calc_level(node: Dict, level: int):
        node["level"] = level
        for child in node["children"]:
            calc_level(child, level + 1)

    for root in roots:
        calc_level(root, base_level)

    # ========== 3. 按阅读顺序分配 elements ==========
    # 按 line_id 排序所有项
    all_items = sorted(predictions, key=lambda x: x.get(id_key, 0))

    # 构建 line_id -> "当前所属 section" 映射
    # 关键：非 section 元素属于它前面最近的 section
    current_section_id = None
    line_to_section = {}
    for item in all_items:
        line_id = item.get(id_key)
        if item.get("is_section", False):
            current_section_id = line_id
        line_to_section[line_id] = current_section_id

    # 分配 elements
    element_seq_no = {}  # section_id -> 当前 seq_no

    for idx, item in enumerate(all_items):
        line_id = item.get(id_key)
        is_section = item.get("is_section", False)

        if is_section:
            # section 标题作为 elements[0]
            section_id = line_id
            if section_id in section_nodes:
                element = build_element(item, idx, seq_no=0)
                section_nodes[section_id]["elements"].append(element)
                element_seq_no[section_id] = 1
        else:
            # 非 section：挂载到前面最近的 section
            section_id = line_to_section.get(line_id)
            if section_id is not None and section_id in section_nodes:
                seq = element_seq_no.get(section_id, 0)
                element = build_element(item, idx, seq_no=seq)
                section_nodes[section_id]["elements"].append(element)
                element_seq_no[section_id] = seq + 1

    # ========== 4. 计算 element_count 和 start_index / end_index ==========
    # element_count: 该 section 直属 elements 数量
    # start_index / end_index: 按 DFS 遍历顺序，表示该 section 在扁平数组中的范围
    #   start_index = 该 section 第一个 element 的全局索引
    #   end_index = 该 section 最后一个元素（含 children）之后的索引
    global_index = [0]  # 用列表包装以便在闭包中修改

    def calc_indices(node: Dict):
        node["element_count"] = len(node["elements"])
        node["start_index"] = global_index[0]

        # 累加当前 section 的 elements
        global_index[0] += len(node["elements"])

        # 递归处理 children
        for child in node["children"]:
            calc_indices(child)

        # end_index 是处理完所有 children 后的索引
        node["end_index"] = global_index[0]

    for root in roots:
        calc_indices(root)

    # ========== 5. 清理内部字段 ==========
    def clean_internal_fields(node: Dict):
        node.pop("_line_id", None)
        node.pop("_parent_id", None)
        for child in node["children"]:
            clean_internal_fields(child)

    for root in roots:
        clean_internal_fields(root)

    return roots


@router.post(
    "",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict document structure (split_result format)",
    description="Run inference and return nested tree structure in split_result format.",
)
async def predict1(request: PredictRequest):
    """
    Run inference and return split_result format.

    返回格式:
    {
        "document": "task_xxx",
        "total_elements": 100,
        "total_sections": 10,
        "sections": [
            {
                "title": "第一章 总则",
                "level": 1,
                "start_index": 0,
                "end_index": 6,
                "element_count": 2,
                "elements": [
                    {"type": "paragraph", "text_preview": "...", "index": 0, ...}
                ],
                "children": [
                    {"title": "一、项目概况", "level": 2, ...}
                ]
            }
        ]
    }
    """
    model_loader = get_model_loader()
    service = get_infer_service()

    # Auto-detect document_name if not provided
    document_name = document_name
    if not document_name:
        document_name = service._find_document_name(request.task_id)

    if not model_loader.is_loaded:
        logger.warning("Model not loaded, returning empty results")
        return {
            "document": f"task_{request.task_id}",
            "total_elements": 0,
            "total_sections": 0,
            "sections": [],
        }

    try:
        logger.info(f"[Predict1] task_id={request.task_id}, document={document_name}")

        # 使用与 /predict 相同的推理流程
        if model_loader.is_joint_training_model:
            # 联合训练模型
            logger.info("[Predict1] Using joint training model (Stage1 + Construct)")
            construct_result = service.predict_with_construct(
                task_id=request.task_id,
                document_name=document_name,
                full_tree=True,
            )
            logger.info(f"[Predict1] Done: {construct_result['num_lines']} lines, {construct_result['inference_time_ms']:.2f}ms")

            # 获取扁平格式A的预测结果
            predictions = construct_result.get("construct_result", {}).get("predictions", [])

        else:
            # 标准 JointModel
            result = service.predict_single(
                task_id=request.task_id,
                document_name=document_name,
                return_original=True,
            )
            logger.info(f"[Predict1] Stage done: {result['num_lines']} lines, {result['inference_time_ms']:.2f}ms")

            # 如果有 construct 模型，进行 TOC 构建
            if model_loader.has_construct_model:
                try:
                    construct_result = service.predict_with_construct(
                        task_id=request.task_id,
                        document_name=document_name,
                        full_tree=True,
                    )
                    predictions = construct_result.get("construct_result", {}).get("predictions", [])
                except Exception as e:
                    logger.warning(f"[Predict1] Construct failed: {e}")
                    predictions = result.get("data", [])
            else:
                predictions = result.get("data", [])

        # 构建嵌套树结构
        sections = build_nested_tree(predictions)

        # 统计
        total_sections = sum(1 for p in predictions if p.get("is_section", False))
        total_elements = len(predictions)

        # 构建返回结果
        result = {
            "document": f"task_{request.task_id}",
            "total_elements": total_elements,
            "total_sections": total_sections,
            "sections": sections,
        }

        # 保存到 upload/{task_id}/{document_name}_split_result.json
        try:
            task_dir = service._get_task_dir(request.task_id)
            split_result_path = os.path.join(task_dir, f"{document_name}_split_result.json")
            with open(split_result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"[Predict1] Saved split_result to: {split_result_path}")
        except Exception as e:
            logger.warning(f"[Predict1] Failed to save split_result: {e}")

        return result

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

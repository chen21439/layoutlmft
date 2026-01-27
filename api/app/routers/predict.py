#!/usr/bin/env python
# coding=utf-8
"""
Predict Router - /predict endpoint
"""

import json
import logging
import os
from fastapi import APIRouter, HTTPException

from ..schemas import (
    PredictRequest,
    PredictResponse,
    ErrorResponse,
)
from ..service.infer_service import get_infer_service
from ..service.model_loader import get_model_loader
from .predict1 import build_nested_tree

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post(
    "",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict document structure",
    description="Run inference on a single document and return predictions merged with original data.",
)
async def predict(request: PredictRequest):
    """
    Run inference on a single document.

    - **task_id**: Task ID (folder name under data_dir_base)
    - **document_name**: Document name (optional, auto-detect if not provided)
    """
    # Return empty results if model is not loaded (e.g., still training)
    model_loader = get_model_loader()
    service = get_infer_service()

    # Auto-detect document_name if not provided
    document_name = request.document_name
    if not document_name:
        document_name = service._find_document_name(request.task_id)

    if not model_loader.is_loaded:
        logger.warning("Model not loaded, returning empty results")
        return PredictResponse(
            document_name=document_name,
            num_lines=0,
            inference_time_ms=0.0,
            data=[],
        )

    try:
        logger.info(f"[Predict] task_id={request.task_id}, document={document_name}")

        if model_loader.is_joint_training_model:
            # 联合训练模型：直接使用 predict_with_construct（不需要 stage3/stage4）
            logger.info("[Predict] Using joint training model (Stage1 + Construct)")
            construct_result = service.predict_with_construct(
                task_id=request.task_id,
                document_name=document_name,
                full_tree=True,
            )
            logger.info(f"[Predict] Done: {construct_result['num_lines']} lines, {construct_result['inference_time_ms']:.2f}ms")

            # 从 construct_result 构建返回结果
            # predict_with_construct 的 predictions 已包含所有行（full_tree=True）
            predictions = construct_result.get("construct_result", {}).get("predictions", [])
            result_data = []
            for pred in predictions:
                result_data.append({
                    "line_id": pred.get("line_id"),
                    "text": pred.get("text", ""),
                    "class": pred.get("class", ""),
                    "parent_id": pred.get("parent_id", -1),
                    "relation": pred.get("relation", ""),
                    "location": pred.get("location"),
                })

            # 保存 split_result 格式
            try:
                sections = build_nested_tree(predictions)
                total_sections = sum(1 for p in predictions if p.get("is_section", False))
                split_result = {
                    "document": f"task_{request.task_id}",
                    "total_elements": len(predictions),
                    "total_sections": total_sections,
                    "sections": sections,
                }
                task_dir = service._get_task_dir(request.task_id)
                split_result_path = os.path.join(task_dir, f"{document_name}_split_result.json")
                with open(split_result_path, 'w', encoding='utf-8') as f:
                    json.dump(split_result, f, ensure_ascii=False, indent=2)
                logger.info(f"[Predict] Saved split_result to: {split_result_path}")
            except Exception as e:
                logger.warning(f"[Predict] Failed to save split_result: {e}")

            return PredictResponse(
                document_name=document_name,
                num_lines=construct_result["num_lines"],
                inference_time_ms=construct_result["inference_time_ms"],
                data=result_data,
            )
        else:
            # 标准 JointModel：原有流程
            # 1. Stage 1/3/4 推理
            result = service.predict_single(
                task_id=request.task_id,
                document_name=document_name,
                return_original=True,
            )
            logger.info(f"[Predict] Stage done: {result['num_lines']} lines, {result['inference_time_ms']:.2f}ms")

            # 2. Construct 推理（如果模型可用）
            if model_loader.has_construct_model:
                try:
                    construct_result = service.predict_with_construct(
                        task_id=request.task_id,
                        document_name=document_name,
                    )
                    logger.info(f"[Predict] Construct done: {construct_result['inference_time_ms']:.2f}ms")
                except Exception as e:
                    logger.warning(f"[Predict] Construct failed (non-fatal): {e}")

            return PredictResponse(
                document_name=result["document_name"],
                num_lines=result["num_lines"],
                inference_time_ms=result["inference_time_ms"],
                data=result["data"],
            )

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

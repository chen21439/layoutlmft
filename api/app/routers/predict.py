#!/usr/bin/env python
# coding=utf-8
"""
Predict Router - /predict endpoint
"""

import logging
from fastapi import APIRouter, HTTPException

from ..schemas import (
    PredictRequest,
    PredictResponse,
    ErrorResponse,
)
from ..service.infer_service import get_infer_service
from ..service.model_loader import get_model_loader

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
    - **document_name**: Document name (without .json extension)
    """
    # Return empty results if model is not loaded (e.g., still training)
    model_loader = get_model_loader()
    if not model_loader.is_loaded:
        logger.warning("Model not loaded, returning empty results")
        return PredictResponse(
            document_name=request.document_name,
            num_lines=0,
            inference_time_ms=0.0,
            data=[],
        )

    try:
        logger.info(f"[Predict] task_id={request.task_id}, document={request.document_name}")

        service = get_infer_service()

        # 1. 原有 Stage 1/3/4 推理
        result = service.predict_single(
            task_id=request.task_id,
            document_name=request.document_name,
            return_original=True,
        )
        logger.info(f"[Predict] Stage done: {result['num_lines']} lines, {result['inference_time_ms']:.2f}ms")

        # 2. Construct 推理（如果模型可用）
        if model_loader.has_construct_model:
            try:
                construct_result = service.predict_with_construct(
                    task_id=request.task_id,
                    document_name=request.document_name,
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

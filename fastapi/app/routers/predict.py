#!/usr/bin/env python
# coding=utf-8
"""
Predict Router - /predict endpoints
"""

import logging
from fastapi import APIRouter, HTTPException

from ..schemas import (
    PredictRequest,
    PredictResponse,
    PredictWithOriginalResponse,
    LineResult,
    ErrorResponse,
)
from ..service.infer_service import get_infer_service

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
    description="Run inference on a single document and return predictions.",
)
async def predict(request: PredictRequest):
    """
    Run inference on a single document.

    - **document_name**: Document name (folder name under data_dir_base)
    """
    try:
        service = get_infer_service()
        result = service.predict_single(
            document_name=request.document_name,
            return_original=False,
        )

        # Convert to response model
        line_results = [
            LineResult(
                line_id=r["line_id"],
                class_label=r["class_label"],
                class_id=r["class_id"],
                parent_id=r["parent_id"],
                relation=r["relation"],
                relation_id=r["relation_id"],
            )
            for r in result["results"]
        ]

        return PredictResponse(
            document_name=result["document_name"],
            num_lines=result["num_lines"],
            results=line_results,
            inference_time_ms=result["inference_time_ms"],
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


@router.post(
    "/with-original",
    response_model=PredictWithOriginalResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict and merge with original data",
    description="Run inference and return predictions merged with original JSON data.",
)
async def predict_with_original(request: PredictRequest):
    """
    Run inference and merge predictions with original JSON data.

    Returns the complete document data with prediction fields added.
    """
    try:
        service = get_infer_service()
        result = service.predict_single(
            document_name=request.document_name,
            return_original=True,
        )

        return PredictWithOriginalResponse(
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


@router.get(
    "/{document_name}",
    response_model=PredictResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict by document name (GET)",
    description="Convenience GET endpoint for prediction.",
)
async def predict_get(document_name: str):
    """GET endpoint for prediction (convenience)."""
    request = PredictRequest(document_name=document_name)
    return await predict(request)

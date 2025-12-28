#!/usr/bin/env python
# coding=utf-8
"""
Health Router - /health endpoints
"""

import logging
from fastapi import APIRouter

from ..schemas import HealthResponse
from ..service.model_loader import get_model_loader

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and model status.",
)
async def health_check():
    """
    Check service health.

    Returns model loading status and device information.
    """
    loader = get_model_loader()

    return HealthResponse(
        status="healthy" if loader.is_loaded else "degraded",
        model_loaded=loader.is_loaded,
        checkpoint=loader._checkpoint_path if loader.is_loaded else None,
        device=loader.device if loader.is_loaded else None,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness check",
    description="Check if service is ready to handle requests.",
)
async def readiness_check():
    """
    Check if service is ready.

    Returns 200 only if model is loaded and ready.
    """
    loader = get_model_loader()

    if not loader.is_loaded:
        return HealthResponse(
            status="not_ready",
            model_loaded=False,
            checkpoint=None,
            device=None,
        )

    return HealthResponse(
        status="ready",
        model_loaded=True,
        checkpoint=loader._checkpoint_path,
        device=loader.device,
    )

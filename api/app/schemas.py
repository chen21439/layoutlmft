#!/usr/bin/env python
# coding=utf-8
"""
Pydantic schemas for request/response models.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ==================== Request Models ====================

class PredictRequest(BaseModel):
    """Request model for /predict endpoint."""

    task_id: str = Field(
        ...,
        description="Task ID (folder name under data_dir_base)",
        example="task_001"
    )
    document_name: Optional[str] = Field(
        None,
        description="Document name (without .json extension). If not provided, auto-detect from task directory.",
        example="tender_doc_001"
    )


# ==================== Response Models ====================

class PredictResponse(BaseModel):
    """Response model for /predict endpoint."""

    document_name: str = Field(..., description="Document name")
    num_lines: int = Field(..., description="Total number of lines")
    inference_time_ms: float = Field(0.0, description="Inference time in milliseconds")
    data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Original JSON data merged with prediction results"
    )


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str = Field(..., description="Service status", example="healthy")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    checkpoint: Optional[str] = Field(None, description="Loaded checkpoint path")
    device: Optional[str] = Field(None, description="Device being used")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

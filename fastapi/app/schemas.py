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

    document_name: str = Field(
        ...,
        description="Document name (folder name under data_dir_base)",
        example="tender_doc_001"
    )


class PredictBatchRequest(BaseModel):
    """Request model for batch prediction."""

    document_names: List[str] = Field(
        ...,
        description="List of document names"
    )


# ==================== Response Models ====================

class LineResult(BaseModel):
    """Prediction result for a single line."""

    line_id: int = Field(..., description="Line ID")
    class_label: str = Field(..., description="Predicted class label (e.g., 'Title', 'Para')")
    class_id: int = Field(..., description="Predicted class ID")
    parent_id: int = Field(..., description="Predicted parent line ID (-1 for root)")
    relation: str = Field(..., description="Relation type (connect/contain/equality)")
    relation_id: int = Field(..., description="Relation type ID")


class PredictResponse(BaseModel):
    """Response model for /predict endpoint."""

    document_name: str = Field(..., description="Document name")
    num_lines: int = Field(..., description="Total number of lines")
    num_pages: int = Field(0, description="Number of pages processed")
    results: List[LineResult] = Field(default_factory=list, description="Prediction results per line")
    inference_time_ms: float = Field(0.0, description="Inference time in milliseconds")


class PredictWithOriginalResponse(BaseModel):
    """Response model that includes original JSON data merged with predictions."""

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

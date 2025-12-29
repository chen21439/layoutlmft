#!/usr/bin/env python
# coding=utf-8
"""
FastAPI Application - Document Structure Analysis Service

Usage:
    # Start server with config file
    python -m api.app.main --env test

    # Or with uvicorn directly
    ENV=test uvicorn api.app.main:app --host 0.0.0.0 --port 8000

    # Development mode with auto-reload
    ENV=test uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload

Configuration:
    Config is loaded from configs/{env}.yml
    - inference.checkpoint_path: Path to model checkpoint
    - inference.data_dir_base: Default data directory
    - gpu.cuda_visible_devices: GPU to use
"""

# ==================== GPU 设置（必须在 import torch 之前）====================
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
EXAMPLES_ROOT = os.path.join(PROJECT_ROOT, "examples")
sys.path.insert(0, EXAMPLES_ROOT)

# GPU 设置
from comp_hrdoc.utils.config import setup_environment
setup_environment()
# ==================== GPU 设置结束 ====================

import logging
import argparse
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 使用统一配置加载器
from configs.config_loader import get_config, Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_inference_config(config: Config) -> tuple:
    """
    Get inference configuration.

    Priority: environment variable > config file
    """
    # Checkpoint path
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        checkpoint_path = config.inference.checkpoint_path

    # Data directory base
    data_dir_base = os.environ.get("DATA_DIR_BASE")
    if not data_dir_base:
        data_dir_base = config.inference.data_dir_base

    return checkpoint_path, data_dir_base


# Global config
_config: Optional[Config] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Loads config and model on startup.
    """
    global _config

    # Load config (GPU already set by setup_environment)
    env = os.environ.get("ENV") or os.environ.get("COMP_HRDOC_ENV", "test")
    _config = get_config(env)
    logger.info(f"Loaded config for env: {env}")

    # Get inference config
    checkpoint_path, data_dir_base = get_inference_config(_config)

    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            from .service.model_loader import load_model
            from .service.infer_service import get_infer_service

            load_model(checkpoint_path, config=_config)
            if data_dir_base:
                get_infer_service(data_dir_base=data_dir_base)
                logger.info(f"Default data_dir_base: {data_dir_base}")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Service starting without model. Load model via /admin/load endpoint.")
    else:
        logger.warning("checkpoint_path not set. Model not loaded at startup.")
        logger.info("Set in config file or use /admin/load endpoint.")

    yield

    # Shutdown: cleanup if needed
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Document Structure Analysis API",
    description="""
API for document structure analysis using LayoutXLM-based model.

## Features
- Predict document structure (class, parent_id, relation) for each text line
- Support for single document inference
- Merge predictions with original JSON data

## Endpoints
- `POST /predict`: Predict document structure
- `GET /health`: Health check
- `GET /ready`: Readiness check

## Configuration
Config is loaded from `configs/{env}.yml`

Set `ENV` environment variable to specify environment (default: test)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from .routers import predict, health

app.include_router(health.router)
app.include_router(predict.router)


# Admin endpoints for runtime model management
@app.post("/admin/load", tags=["admin"])
async def load_model_endpoint(
    checkpoint_path: str,
    data_dir: str = None,
):
    """
    Load or reload model at runtime.

    - **checkpoint_path**: Path to model checkpoint
    - **data_dir**: Optional default data directory
    """
    try:
        from .service.model_loader import load_model
        from .service.infer_service import get_infer_service

        load_model(checkpoint_path, config=_config)
        if data_dir:
            get_infer_service(data_dir=data_dir)

        return {
            "status": "success",
            "message": f"Model loaded from {checkpoint_path}",
        }
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Document Structure Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/config", tags=["admin"])
async def get_current_config():
    """Get current configuration (for debugging)."""
    if _config is None:
        return {"error": "Config not loaded"}

    checkpoint_path, data_dir_base = get_inference_config(_config)
    return {
        "env": _config.env,
        "inference": {
            "checkpoint_path": checkpoint_path,
            "data_dir_base": data_dir_base,
        },
        "gpu": {
            "cuda_visible_devices": _config.gpu.cuda_visible_devices,
        },
    }


def main():
    """Entry point for running with python -m."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Document Structure Analysis API")
    parser.add_argument("--env", type=str, default="test", help="Environment (dev/test)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=9197, help="Port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Set environment for config loading
    os.environ["ENV"] = args.env

    uvicorn.run(
        "api.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

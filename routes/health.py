"""
Health and status route handlers.

This module provides endpoints for service health monitoring:
- /health: Basic health check
- /status: Detailed status with model information
- /: Root endpoint

These endpoints provide service monitoring and readiness information.
"""

import logging
from fastapi import APIRouter, Depends

from .models import HealthResponse, StatusResponse
from services import AIEngine

logger = logging.getLogger("glamify-ai")
router = APIRouter()


def get_ai_engine() -> AIEngine:
    """Dependency to get AIEngine instance."""
    from main import get_ai_engine as _get_ai_engine
    return _get_ai_engine()


@router.get("/", response_model=HealthResponse)
def read_root() -> HealthResponse:
    """
    Root endpoint health check.
    
    Simple health check endpoint that returns basic service status.
    Used for load balancer health checks and service discovery.
    
    Returns:
        HealthResponse with basic status information
    """
    return HealthResponse(
        status="ok",
        engine="Glamify-AI-Unified",
        ready=True
    )


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns basic health status for the service.
    Used by monitoring systems and load balancers.
    
    Returns:
        HealthResponse with health status
    """
    return HealthResponse(
        status="ok",
        engine="Glamify-AI-Unified", 
        ready=True
    )


@router.get("/status", response_model=StatusResponse)
def status_check(
    ai_engine: AIEngine = Depends(get_ai_engine)
) -> StatusResponse:
    """
    Detailed status endpoint.
    
    Returns comprehensive status information including:
    - Model loading status
    - Configuration summary
    - Startup information
    - Resource utilization
    
    Args:
        ai_engine: Injected AIEngine instance
        
    Returns:
        StatusResponse with detailed status information
    """
    try:
        # Get model status from AI engine
        model_status = ai_engine.model_status()
        
        # Get configuration summary
        from config import get_config
        config = get_config()
        config_summary = {
            "gpu_concurrency": config.app.gpu_concurrency,
            "flux2_backend": config.flux2.descriptor_backend,
            "analyze_enabled": True,
            "parser_enabled": config.analyze.enable_human_parser,
        }
        
        # Get startup information
        startup_info = {
            "preload_enabled": config.app.startup_background_preload,
            "ready": True,
        }
        
        return StatusResponse(
            status="ok",
            models=model_status,
            config=config_summary,
            startup=startup_info
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return StatusResponse(
            status="error",
            models={"error": str(e)},
            config={"error": "Configuration unavailable"},
            startup={"error": "Startup info unavailable"}
        )
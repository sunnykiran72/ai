"""
Route handlers for the Fashion AI API.

This module provides organized route handlers that delegate to service classes.
Each route module contains thin handlers that:
- Validate requests using Pydantic models
- Delegate business logic to service classes
- Format responses consistently
- Handle errors appropriately

Route modules:
- tryon.py: Virtual try-on endpoints
- analyze.py: Garment analysis endpoints  
- extract.py: Single garment extraction
- user_prep.py: User image preparation
- health.py: Health and status endpoints

Usage:
    from routes import tryon_router, analyze_router, extract_router
    
    app.include_router(tryon_router)
    app.include_router(analyze_router)
    app.include_router(extract_router)
"""

from .tryon import router as tryon_router
from .analyze import router as analyze_router
from .extract import router as extract_router
from .user_prep import router as user_prep_router
from .health import router as health_router
from .tryon_lab import router as tryon_lab_router
from .tryon_lora_multi_lab import router as tryon_lora_multi_lab_router
from .bulk_tryon_gallery import router as bulk_tryon_gallery_router
from .qwen_extract_outfit import router as qwen_extract_outfit_router
from .upscaler_test import router as upscaler_test_router
from .analyze_detection_lab import router as analyze_detection_lab_router
from .seedvr2_test import router as seedvr2_test_router

__all__ = [
    "tryon_router",
    "analyze_router", 
    "extract_router",
    "user_prep_router",
    "health_router",
    "tryon_lab_router",
    "tryon_lora_multi_lab_router",
    "bulk_tryon_gallery_router",
    "qwen_extract_outfit_router",
    "upscaler_test_router",
    "analyze_detection_lab_router",
    "seedvr2_test_router",
]

"""
Service layer module.

This module contains business logic orchestration components that coordinate
interactions between AI model runners, utilities, and external services.

Services follow dependency injection patterns and accept configuration and
model runners as constructor parameters.

Architecture:
    The service layer sits between route handlers and utilities, providing:
    - Business logic orchestration
    - Model runner coordination
    - Configuration-driven behavior
    - Error handling and logging
    - Resource management (GPU concurrency)

Usage Examples:
    # Initialize services with dependency injection
    from config import get_config
    from services import AIEngine, AnalyzeService, TryonService
    
    config = get_config()
    engine = AIEngine(config)
    analyze_service = AnalyzeService(engine, config.analyze)
    tryon_service = TryonService(engine, config.flux2)
    
    # Use services in route handlers
    result = await analyze_service.analyze_image(image)
    tryon_result = await tryon_service.try_on(user_image, garment_images)

Service Responsibilities:
    - AIEngine: AI model orchestrator and resource manager
    - AnalyzeService: Garment analysis and detection workflows
    - TryonService: Virtual try-on generation workflows
    - GarmentExtractionService: Single garment extraction
    - UserImageService: User image preparation and validation

Exports:
    AIEngine: AI model orchestrator
    AnalyzeService: Garment analysis workflows
    TryonService: Virtual try-on workflows
    GarmentExtractionService: Single garment extraction
    UserImageService: User image preparation
"""

from .ai_engine import AIEngine
from .analyze_service import AnalyzeService
from .tryon_service import TryonService
from .garment_extraction_service import GarmentExtractionService
from .user_image_service import UserImageService

__all__ = [
    "AIEngine",
    "AnalyzeService", 
    "TryonService",
    "GarmentExtractionService",
    "UserImageService",
]

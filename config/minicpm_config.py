"""
MiniCPM service configuration.

This module defines configuration for the MiniCPM vision-language model service including:
- Service URLs and timeouts
- Connection pooling settings
- Caching configuration
- Prompt templates for garment and person descriptions
- Image sizing parameters
- Token limits for generation
"""

import os
from pydantic import BaseModel, Field
from .base import env_int, env_float, env_bool


class MiniCPMConfig(BaseModel):
    """
    MiniCPM service configuration.
    
    This configuration manages all settings for the MiniCPM vision-language model service,
    which provides garment and person descriptions for virtual try-on workflows.
    """
    
    # Service URLs
    service_url: str = Field(
        default="http://127.0.0.1:8010",
        description="Base URL for MiniCPM service"
    )
    analyze_service_url: str = Field(
        default="",
        description="Optional separate URL for analyze endpoint (falls back to service_url if empty)"
    )
    
    # Timeouts
    timeout_s: int = Field(
        default=120,
        ge=5,
        description="Request timeout in seconds"
    )
    connect_timeout_s: float = Field(
        default=10.0,
        ge=1.0,
        description="Connection timeout in seconds"
    )
    
    # Token limits
    garment_max_new_tokens: int = Field(
        default=256,
        ge=64,
        description="Maximum new tokens for garment descriptions"
    )
    person_max_new_tokens: int = Field(
        default=140,
        ge=32,
        description="Maximum new tokens for person descriptions"
    )
    
    # Caching
    cache_enabled: bool = Field(
        default=True,
        description="Enable response caching"
    )
    cache_ttl_seconds: int = Field(
        default=900,
        ge=0,
        description="Cache time-to-live in seconds (0 = no expiration)"
    )
    cache_max_entries: int = Field(
        default=1024,
        ge=16,
        description="Maximum number of cache entries"
    )
    
    # Connection pooling
    pool_maxsize: int = Field(
        default=16,
        ge=4,
        description="Maximum connection pool size"
    )
    local_file_first: bool = Field(
        default=True,
        description="Prefer local file paths over URLs when available"
    )
    
    # Prompts
    garment_min_words: int = Field(
        default=10,
        ge=4,
        description="Minimum word count for garment descriptions"
    )
    garment_prompt: str = Field(
        default=(
            "Describe only the product garment for high-fidelity virtual try-on. "
            "Describe exactly one garment only and never describe any person, skin, hair, hands, legs, pose, room, props, or background objects. "
            "Ignore studio/background pixels, alpha-matte edges, and lighting shadows outside the garment fabric. "
            "Return one detailed line with schema: "
            "category=<dress|top|bottom|outerwear|set|unknown>; "
            "type=<specific garment type>; "
            "pattern=<solid|striped|floral|graphic|etc>; "
            "material=<fabric/material>; "
            "texture=<smooth|ribbed|knit|velvet|denim|etc>; "
            "fit=<slim|regular|relaxed|oversized|bodycon>; "
            "silhouette=<fit and shape>; "
            "length=<crop|hip|waist|midi|maxi|full>; "
            "construction=<neckline, collar, lapel, sleeve style/length, waist shaping, hem, rise, leg shape>; "
            "details=<buttons, zipper, pleats, ruffles, lace, embroidery, pockets, slit, straps, logo, hardware>; "
            "opening=<front/back/side closure if visible>; "
            "lining=<lined/unlined if visible>; "
            "sheer=<sheer/opaque if visible>; "
            "coverage=<body area to replace>; "
            "preserve=<garment structure and details must remain unchanged>. "
            "If the requested garment type is known, keep category/type locked to that garment only and ignore any other clothing visible in the crop. "
            "Use unknown when not visible."
        ),
        description="Prompt template for garment descriptions"
    )
    person_prompt: str = Field(
        default=(
            "Describe only the human subject for identity-preserving virtual try-on. "
            "Ignore the background entirely and do not describe the full outfit except where it overlaps or occludes the target garment region. "
            "Return one detailed line with schema: "
            "identity=<face traits, skin tone, hair style/color, age band>; "
            "body_pose=<posture, standing/sitting, limb position>; "
            "framing_lighting=<framing/crop, light direction/intensity>; "
            "occlusion=<hair/hands/accessories/objects overlapping the garment region>; "
            "outfit_boundary=<only the parts of the current outfit that overlap the target garment area>; "
            "preserve=<face identity, skin tone, hair, body proportions, pose>."
        ),
        description="Prompt template for person descriptions"
    )
    
    # Image sizing
    product_caption_max_side: int = Field(
        default=1024,
        ge=512,
        description="Maximum side length for product image captioning"
    )
    product_caption_min_side: int = Field(
        default=512,
        ge=256,
        description="Minimum side length for product image captioning"
    )
    user_caption_max_side: int = Field(
        default=1024,
        ge=512,
        description="Maximum side length for user image captioning"
    )
    user_caption_min_side: int = Field(
        default=512,
        ge=256,
        description="Minimum side length for user image captioning"
    )
    
    @classmethod
    def from_env(cls) -> "MiniCPMConfig":
        """
        Load configuration from environment variables.
        
        Returns:
            MiniCPMConfig instance populated from environment variables
        """
        # Service URLs
        raw_service_url = os.getenv("MINICPM_SERVICE_URL", "").strip().rstrip("/")
        service_url = raw_service_url or "http://127.0.0.1:8010"
        
        raw_analyze_service_url = os.getenv("ANALYZE_MINICPM_SERVICE_URL", "").strip().rstrip("/")
        analyze_service_url = raw_analyze_service_url
        
        # Timeouts - affected by low latency mode
        low_latency_mode = env_bool("FLUX2_LOW_LATENCY_MODE", "0")
        timeout_s = max(5, env_int(
            "MINICPM_SERVICE_TIMEOUT_S",
            60 if low_latency_mode else 120
        ))
        connect_timeout_s = max(1.0, env_float(
            "MINICPM_SERVICE_CONNECT_TIMEOUT_S",
            6.0 if low_latency_mode else 10.0
        ))
        
        # Token limits - affected by low latency mode
        garment_max_new_tokens = max(64, env_int(
            "MINICPM_SERVICE_GARMENT_MAX_NEW_TOKENS",
            256 if low_latency_mode else 256
        ))
        person_max_new_tokens = max(32, env_int(
            "MINICPM_SERVICE_PERSON_MAX_NEW_TOKENS",
            96 if low_latency_mode else 140
        ))
        
        # Caching - affected by low latency mode
        cache_ttl_seconds = max(0, env_int(
            "MINICPM_SERVICE_CACHE_TTL_SECONDS",
            1800 if low_latency_mode else 900
        ))
        
        # Prompts
        garment_prompt = os.getenv(
            "MINICPM_SERVICE_GARMENT_PROMPT",
            (
                "Describe only the product garment for high-fidelity virtual try-on. "
                "Describe exactly one garment only and never describe any person, skin, hair, hands, legs, pose, room, props, or background objects. "
                "Ignore studio/background pixels, alpha-matte edges, and lighting shadows outside the garment fabric. "
                "Return one detailed line with schema: "
                "category=<dress|top|bottom|outerwear|set|unknown>; "
                "type=<specific garment type>; "
                "pattern=<solid|striped|floral|graphic|etc>; "
                "material=<fabric/material>; "
                "texture=<smooth|ribbed|knit|velvet|denim|etc>; "
                "fit=<slim|regular|relaxed|oversized|bodycon>; "
                "silhouette=<fit and shape>; "
                "length=<crop|hip|waist|midi|maxi|full>; "
                "construction=<neckline, collar, lapel, sleeve style/length, waist shaping, hem, rise, leg shape>; "
                "details=<buttons, zipper, pleats, ruffles, lace, embroidery, pockets, slit, straps, logo, hardware>; "
                "opening=<front/back/side closure if visible>; "
                "lining=<lined/unlined if visible>; "
                "sheer=<sheer/opaque if visible>; "
                "coverage=<body area to replace>; "
                "preserve=<garment structure and details must remain unchanged>. "
                "If the requested garment type is known, keep category/type locked to that garment only and ignore any other clothing visible in the crop. "
                "Use unknown when not visible."
            )
        ).strip()
        
        person_prompt = os.getenv(
            "MINICPM_SERVICE_PERSON_PROMPT",
            (
                "Describe only the human subject for identity-preserving virtual try-on. "
                "Ignore the background entirely and do not describe the full outfit except where it overlaps or occludes the target garment region. "
                "Return one detailed line with schema: "
                "identity=<face traits, skin tone, hair style/color, age band>; "
                "body_pose=<posture, standing/sitting, limb position>; "
                "framing_lighting=<framing/crop, light direction/intensity>; "
                "occlusion=<hair/hands/accessories/objects overlapping the garment region>; "
                "outfit_boundary=<only the parts of the current outfit that overlap the target garment area>; "
                "preserve=<face identity, skin tone, hair, body proportions, pose>."
            )
        ).strip()
        
        return cls(
            # Service URLs
            service_url=service_url,
            analyze_service_url=analyze_service_url,
            
            # Timeouts
            timeout_s=timeout_s,
            connect_timeout_s=connect_timeout_s,
            
            # Token limits
            garment_max_new_tokens=garment_max_new_tokens,
            person_max_new_tokens=person_max_new_tokens,
            
            # Caching
            cache_enabled=env_bool("MINICPM_SERVICE_CACHE_ENABLED", "1"),
            cache_ttl_seconds=cache_ttl_seconds,
            cache_max_entries=max(16, env_int("MINICPM_SERVICE_CACHE_MAX_ENTRIES", 1024)),
            
            # Connection pooling
            pool_maxsize=max(4, env_int("MINICPM_SERVICE_POOL_MAXSIZE", 16)),
            local_file_first=env_bool("MINICPM_SERVICE_LOCAL_FILE_FIRST", "1"),
            
            # Prompts
            garment_min_words=max(4, env_int("MINICPM_SERVICE_GARMENT_MIN_WORDS", 10)),
            garment_prompt=garment_prompt,
            person_prompt=person_prompt,
            
            # Image sizing
            product_caption_max_side=max(512, env_int("FLUX2_MINICPM_PRODUCT_CAPTION_MAX_SIDE", 1024)),
            product_caption_min_side=max(256, env_int("FLUX2_MINICPM_PRODUCT_CAPTION_MIN_SIDE", 512)),
            user_caption_max_side=max(512, env_int("FLUX2_MINICPM_USER_CAPTION_MAX_SIDE", 1024)),
            user_caption_min_side=max(256, env_int("FLUX2_MINICPM_USER_CAPTION_MIN_SIDE", 512)),
        )

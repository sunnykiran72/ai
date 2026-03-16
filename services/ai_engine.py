"""
AI model orchestrator service.

This module provides the AIEngine class that manages all AI model runners
and provides unified access to ML capabilities.

Responsibilities:
- Initialize and manage AI model runners
- Provide lazy loading and caching of models
- Handle GPU concurrency and resource management
- Expose model status and health information
"""

from typing import Optional, Dict
import threading

from config import Config
from core.flux2_cvton_runner import Flux2CVTONRunner
from core.florence_runner import FlorenceRunner
from core.qwen25vl_runner import Qwen25VLRunner
from core.joycaption_runner import JoyCaptionRunner
from core.minicpm_runner import MiniCPMVRunner
from core.fashion_detection_runner import FashionDetectionRunner
from core.yolo_runner import YoloPersonDetectorRunner, YoloRunner
from core.human_parser_runner import HumanParserRunner
from core.openclip_runner import OpenCLIPRunner
from core.fashion_color_classifier_runner import FashionColorClassifierRunner
from core.garment_color_masker import GarmentColorMasker
from modules.wardrobe.yolo_cropper import YoloCropper
from modules.wardrobe.human_parser import HumanParser
from modules.wardrobe.cloth_detection import ClothDetector
from modules.vto.board_builder import BoardBuilder


class AIEngine:
    """
    Orchestrates AI model runners and provides unified access to ML capabilities.
    
    Responsibilities:
    - Initialize and manage AI model runners
    - Provide lazy loading and caching of models
    - Handle GPU concurrency and resource management
    - Expose model status and health information
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Detection models
        self.yolo_runner = YoloRunner()
        self.person_detector = YoloPersonDetectorRunner()
        self.fashion_detection_runner = FashionDetectionRunner()
        self.parser_runner = HumanParserRunner() if config.analyze.enable_human_parser else None
        
        # Wrappers
        self.yolo = YoloCropper(predictor=self.yolo_runner.predict)
        self.parser = HumanParser(parser_fn=self.parser_runner.parse) if self.parser_runner is not None else None
        self.cloth_detector = ClothDetector(
            legacy_detector=self.yolo,
            fashion_detector=self.fashion_detection_runner,
        )
        self.garment_color_masker = GarmentColorMasker(
            parser=self.parser,
            base_mask_fn=lambda image: self._get_heuristic_base_mask(image),
            skin_mask_fn=lambda rgb: self._skin_like_mask(rgb),
        )
        
        # Vision-language models
        self.florence = FlorenceRunner()
        self.qwen25vl = Qwen25VLRunner()
        self.joycaption = JoyCaptionRunner()
        self.minicpm = MiniCPMVRunner()
        
        # Classification models
        self.openclip = OpenCLIPRunner()
        self.fashion_basecolour = FashionColorClassifierRunner()
        
        # Generation models
        shared_flux2_config: Dict[str, object] = {}
        self._share_flux2_base_runner = bool(
            config.flux2.share_base_runner and config.analyze.flux_disable_lora
        )
        if self._share_flux2_base_runner:
            shared_flux2_config["runtime_lora_toggle"] = True
            shared_flux2_config["fuse_lora"] = False
        
        self.flux2 = Flux2CVTONRunner(config=shared_flux2_config)
        self._analyze_flux2: Optional[Flux2CVTONRunner] = None
        self._analyze_flux2_lock = threading.Lock()
        
        # Utilities
        self.board_builder = BoardBuilder()
    
    def get_flux2_for_analyze(self) -> Flux2CVTONRunner:
        """Get Flux2 runner for analyze endpoint (may be shared or isolated)."""
        # Shared-base mode keeps one FLUX pipeline in memory and toggles LoRA per request.
        if self._share_flux2_base_runner:
            return self.flux2
        # Strict isolation: /analyze can run on a separate no-LoRA runner so try-on remains unchanged.
        if not self.config.analyze.flux_disable_lora:
            return self.flux2
        if self._analyze_flux2 is None:
            with self._analyze_flux2_lock:
                if self._analyze_flux2 is None:
                    self._analyze_flux2 = Flux2CVTONRunner(
                        config={
                            "enable_lora": False,
                            "require_lora": False,
                            "fuse_lora": False,
                        }
                    )
        return self._analyze_flux2
    
    def ensure_vto_ready(self):
        """Preload models required for virtual try-on."""
        self.flux2.ensure_ready()
        if self.config.flux2.descriptor_compare:
            self.florence._ensure_loaded()
            if self.config.flux2.preload_qwen_with_flux2:
                self.qwen25vl.ensure_ready()
            if self.config.flux2.preload_joycaption_with_flux2:
                self.joycaption.ensure_ready()
            if self.config.flux2.preload_minicpm_with_flux2:
                self.minicpm.ensure_ready()
        elif self.config.flux2.descriptor_backend == "qwen2_5_vl":
            if self.config.flux2.preload_qwen_with_flux2:
                self.qwen25vl.ensure_ready()
        elif self.config.flux2.descriptor_backend == "joycaption":
            if self.config.flux2.preload_joycaption_with_flux2:
                self.joycaption.ensure_ready()
        elif self.config.flux2.descriptor_backend == "minicpm":
            if self.config.flux2.preload_minicpm_with_flux2:
                self.minicpm.ensure_ready()
        elif self.config.flux2.descriptor_backend == "minicpm_service":
            # External MiniCPM service handles descriptor generation.
            # Keep local descriptor runners unloaded for better VRAM headroom.
            pass
        else:
            self.florence._ensure_loaded()
    
    def ensure_analyze_ready(self):
        """Preload models required for garment analysis."""
        self.yolo_runner.ensure_ready()
        if self.parser_runner:
            self.parser_runner.ensure_ready()
        if self.config.analyze.preload_florence:
            self.florence._ensure_loaded()
        if self.config.analyze.preload_flux_runner:
            self.get_flux2_for_analyze().ensure_ready()
        if self.config.analyze.fashion_basecolour_trial_enabled:
            self.fashion_basecolour.ensure_ready()
    
    def model_status(self) -> Dict[str, object]:
        """Get current status of all AI models."""
        analyze_flux2 = self.flux2 if self._share_flux2_base_runner else self._analyze_flux2
        analyze_startup = dict(getattr(analyze_flux2, "_startup_metrics", {}) or {}) if analyze_flux2 else {}
        return {
            "flux2_loaded": self.flux2._pipeline is not None,
            "analyze_flux2_loaded": bool(analyze_flux2 and analyze_flux2._pipeline is not None),
            "analyze_flux2_isolated": bool(self.config.analyze.flux_disable_lora and not self._share_flux2_base_runner),
            "flux2_shared_base_runner": bool(self._share_flux2_base_runner),
            "flux2_runtime_lora_toggle": bool(getattr(self.flux2, "runtime_lora_toggle", False)),
            "analyze_flux2_lora_enabled": bool(
                analyze_startup.get("lora_enabled", False)
            ) if analyze_flux2 else False,
            "analyze_flux2_lora_loaded": bool(
                analyze_startup.get("lora_loaded", False)
            ) if analyze_flux2 else False,
            "florence_loaded": self.florence._model is not None,
            "qwen25vl_loaded": self.qwen25vl.is_loaded,
            "joycaption_loaded": self.joycaption.is_loaded,
            "minicpm_loaded": self.minicpm.is_loaded,
            "minicpm_model_id": str(getattr(self.minicpm, "model_id", "")),
            "openclip_loaded": self.openclip.is_loaded,
            "openclip_available": self.openclip.is_available,
            "fashion_basecolour_loaded": self.fashion_basecolour.is_loaded,
            "fashion_basecolour_available": self.fashion_basecolour.is_available,
            "fashion_basecolour_model_id": self.fashion_basecolour.model_id,
            "yolo_loaded": self.yolo_runner.is_loaded,
            "yolo_model_path": self.yolo_runner.model_path,
            "yolo_expected_label_family": self.yolo_runner.expected_label_family,
            "yolo_label_family": self.yolo_runner.label_family,
            "yolo_class_count": self.yolo_runner.class_count,
            "fashion_detection_loaded": self.fashion_detection_runner.is_loaded,
            "fashion_detection_model_path": self.fashion_detection_runner.model_path,
            "human_parser_loaded": bool(self.parser_runner and self.parser_runner.is_loaded),
        }
    
    def _get_heuristic_base_mask(self, image):
        """Get heuristic base mask for garment color masking."""
        # TODO: Implement the actual logic from main.py _get_heuristic_base_mask
        # This is a complex function that depends on BiRefNet, clean foreground detection, and rembg
        # For now, return None to maintain compatibility
        return None, {"source": "none"}
    
    def _skin_like_mask(self, rgb):
        """Get skin-like mask for color processing."""
        # TODO: Implement the actual logic from main.py _skin_like_mask
        # This function uses OpenCV for skin detection in HSV and YCrCb color spaces
        # For now, return empty mask to maintain compatibility
        import numpy as np
        if hasattr(rgb, 'shape') and len(rgb.shape) >= 2:
            h, w = rgb.shape[:2]
            return np.zeros((h, w), dtype=bool)
        return np.zeros((0, 0), dtype=bool)

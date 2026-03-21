"""
Unit tests for services/ai_engine.py AIEngine service.

Tests the AIEngine class that orchestrates AI model runners and provides
unified access to ML capabilities. Covers initialization, lazy loading,
model status reporting, and configuration handling.
"""

import os
import sys
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import threading

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ai_engine import AIEngine
from config import Config, AppConfig, Flux2Config, AnalyzeConfig, MiniCPMConfig, ColorConfig


class TestAIEngineInitialization(unittest.TestCase):
    """Tests for AIEngine initialization and model runner setup."""
    
    def setUp(self):
        """Set up test configuration and mocks."""
        # Create mock configuration
        self.mock_config = Mock(spec=Config)
        self.mock_config.app = Mock(spec=AppConfig)
        self.mock_config.flux2 = Mock(spec=Flux2Config)
        self.mock_config.analyze = Mock(spec=AnalyzeConfig)
        self.mock_config.minicpm = Mock(spec=MiniCPMConfig)
        self.mock_config.color = Mock(spec=ColorConfig)
        
        # Set default config values
        self.mock_config.analyze.enable_human_parser = True
        self.mock_config.flux2.share_base_runner = True
        self.mock_config.analyze.flux_disable_lora = True
    
    @patch('services.ai_engine.YoloRunner')
    @patch('services.ai_engine.YoloPersonDetectorRunner')
    @patch('services.ai_engine.FashionDetectionRunner')
    @patch('services.ai_engine.HumanParserRunner')
    @patch('services.ai_engine.FlorenceRunner')
    @patch('services.ai_engine.Qwen25VLRunner')
    @patch('services.ai_engine.JoyCaptionRunner')
    @patch('services.ai_engine.MiniCPMVRunner')
    @patch('services.ai_engine.OpenCLIPRunner')
    @patch('services.ai_engine.FashionColorClassifierRunner')
    @patch('services.ai_engine.Flux2CVTONRunner')
    @patch('services.ai_engine.YoloCropper')
    @patch('services.ai_engine.HumanParser')
    @patch('services.ai_engine.ClothDetector')
    @patch('services.ai_engine.GarmentColorMasker')
    @patch('services.ai_engine.BoardBuilder')
    def test_init_creates_all_model_runners(self, mock_board_builder, mock_garment_color_masker,
                                          mock_cloth_detector, mock_human_parser, mock_yolo_cropper,
                                          mock_flux2, mock_fashion_color, mock_openclip,
                                          mock_minicpm, mock_joycaption, mock_qwen25vl,
                                          mock_florence, mock_human_parser_runner,
                                          mock_fashion_detection, mock_person_detector, mock_yolo):
        """Test that AIEngine initializes all required model runners."""
        engine = AIEngine(self.mock_config)
        
        # Verify detection models are created
        mock_yolo.assert_called_once()
        mock_person_detector.assert_called_once()
        mock_fashion_detection.assert_called_once()
        mock_human_parser_runner.assert_called_once()
        
        # Verify vision-language models are created
        mock_florence.assert_called_once()
        mock_qwen25vl.assert_called_once()
        mock_joycaption.assert_called_once()
        mock_minicpm.assert_called_once()
        
        # Verify classification models are created
        mock_openclip.assert_called_once()
        mock_fashion_color.assert_called_once()
        
        # Verify generation models are created
        mock_flux2.assert_called_once()
        
        # Verify utilities are created
        mock_board_builder.assert_called_once()
        
        # Verify model runners are assigned to engine
        self.assertIsNotNone(engine.yolo_runner)
        self.assertIsNotNone(engine.person_detector)
        self.assertIsNotNone(engine.fashion_detection_runner)
        self.assertIsNotNone(engine.parser_runner)
        self.assertIsNotNone(engine.florence)
        self.assertIsNotNone(engine.qwen25vl)
        self.assertIsNotNone(engine.joycaption)
        self.assertIsNotNone(engine.minicpm)
        self.assertIsNotNone(engine.openclip)
        self.assertIsNotNone(engine.fashion_basecolour)
        self.assertIsNotNone(engine.flux2)
        self.assertIsNotNone(engine.board_builder)
    
    @patch('services.ai_engine.YoloRunner')
    @patch('services.ai_engine.HumanParserRunner')
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_init_without_human_parser(self, mock_flux2, mock_human_parser_runner, mock_yolo):
        """Test AIEngine initialization when human parser is disabled."""
        self.mock_config.analyze.enable_human_parser = False
        
        engine = AIEngine(self.mock_config)
        
        # Verify human parser is not created
        mock_human_parser_runner.assert_not_called()
        self.assertIsNone(engine.parser_runner)
        self.assertIsNone(engine.parser)
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_init_flux2_shared_base_runner_config(self, mock_flux2):
        """Test Flux2 configuration for shared base runner mode."""
        self.mock_config.flux2.share_base_runner = True
        self.mock_config.analyze.flux_disable_lora = True
        
        engine = AIEngine(self.mock_config)
        
        # Verify Flux2 is configured for shared base runner
        mock_flux2.assert_called_once_with(config={
            "runtime_lora_toggle": True,
            "fuse_lora": False
        })
        self.assertTrue(engine._share_flux2_base_runner)
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_init_flux2_isolated_runner_config(self, mock_flux2):
        """Test Flux2 configuration for isolated runner mode."""
        self.mock_config.flux2.share_base_runner = False
        self.mock_config.analyze.flux_disable_lora = True
        
        engine = AIEngine(self.mock_config)
        
        # Verify Flux2 is configured without shared runner options
        mock_flux2.assert_called_once_with(config={})
        self.assertFalse(engine._share_flux2_base_runner)


class TestAIEngineLazyLoading(unittest.TestCase):
    """Tests for AIEngine lazy loading behavior."""
    
    def setUp(self):
        """Set up test configuration and mocks."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.app = Mock(spec=AppConfig)
        self.mock_config.flux2 = Mock(spec=Flux2Config)
        self.mock_config.analyze = Mock(spec=AnalyzeConfig)
        self.mock_config.minicpm = Mock(spec=MiniCPMConfig)
        self.mock_config.color = Mock(spec=ColorConfig)
        
        # Set default config values
        self.mock_config.analyze.enable_human_parser = True
        self.mock_config.flux2.share_base_runner = True
        self.mock_config.analyze.flux_disable_lora = True
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_get_flux2_for_analyze_shared_runner(self, mock_flux2):
        """Test get_flux2_for_analyze returns shared runner when enabled."""
        self.mock_config.flux2.share_base_runner = True
        self.mock_config.analyze.flux_disable_lora = True
        
        engine = AIEngine(self.mock_config)
        result = engine.get_flux2_for_analyze()
        
        # Should return the main flux2 runner
        self.assertIs(result, engine.flux2)
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_get_flux2_for_analyze_lora_enabled(self, mock_flux2):
        """Test get_flux2_for_analyze returns main runner when LoRA is enabled."""
        self.mock_config.flux2.share_base_runner = False
        self.mock_config.analyze.flux_disable_lora = False
        
        engine = AIEngine(self.mock_config)
        result = engine.get_flux2_for_analyze()
        
        # Should return the main flux2 runner
        self.assertIs(result, engine.flux2)
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_get_flux2_for_analyze_isolated_runner_lazy_creation(self, mock_flux2):
        """Test get_flux2_for_analyze creates isolated runner lazily."""
        self.mock_config.flux2.share_base_runner = False
        self.mock_config.analyze.flux_disable_lora = True
        
        # Mock the Flux2CVTONRunner constructor to return different instances
        mock_main_runner = Mock()
        mock_analyze_runner = Mock()
        mock_flux2.side_effect = [mock_main_runner, mock_analyze_runner]
        
        engine = AIEngine(self.mock_config)
        
        # First call should create the isolated runner
        result1 = engine.get_flux2_for_analyze()
        self.assertIs(result1, mock_analyze_runner)
        
        # Second call should return the same isolated runner
        result2 = engine.get_flux2_for_analyze()
        self.assertIs(result2, mock_analyze_runner)
        self.assertIs(result1, result2)
        
        # Verify isolated runner was configured correctly
        self.assertEqual(mock_flux2.call_count, 2)  # Main + analyze runners
        mock_flux2.assert_any_call(config={
            "enable_lora": False,
            "require_lora": False,
            "fuse_lora": False,
        })
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_get_flux2_for_analyze_thread_safety(self, mock_flux2):
        """Test get_flux2_for_analyze is thread-safe for lazy creation."""
        self.mock_config.flux2.share_base_runner = False
        self.mock_config.analyze.flux_disable_lora = True
        
        # Mock the Flux2CVTONRunner constructor
        mock_main_runner = Mock()
        mock_analyze_runner = Mock()
        mock_flux2.side_effect = [mock_main_runner, mock_analyze_runner]
        
        engine = AIEngine(self.mock_config)
        
        results = []
        
        def get_runner():
            results.append(engine.get_flux2_for_analyze())
        
        # Create multiple threads to test thread safety
        threads = [threading.Thread(target=get_runner) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should get the same instance
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIs(result, results[0])


class TestAIEngineModelPreloading(unittest.TestCase):
    """Tests for AIEngine model preloading methods."""
    
    def setUp(self):
        """Set up test configuration and mocks."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.app = Mock(spec=AppConfig)
        self.mock_config.flux2 = Mock(spec=Flux2Config)
        self.mock_config.analyze = Mock(spec=AnalyzeConfig)
        self.mock_config.minicpm = Mock(spec=MiniCPMConfig)
        self.mock_config.color = Mock(spec=ColorConfig)
        
        # Set default config values
        self.mock_config.analyze.enable_human_parser = True
        self.mock_config.flux2.share_base_runner = True
        self.mock_config.analyze.flux_disable_lora = True
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    @patch('services.ai_engine.FlorenceRunner')
    @patch('services.ai_engine.Qwen25VLRunner')
    @patch('services.ai_engine.JoyCaptionRunner')
    @patch('services.ai_engine.MiniCPMVRunner')
    def test_ensure_vto_ready_descriptor_compare_mode(self, mock_minicpm, mock_joycaption,
                                                     mock_qwen25vl, mock_florence, mock_flux2):
        """Test ensure_vto_ready with descriptor compare mode enabled."""
        self.mock_config.flux2.descriptor_compare = True
        self.mock_config.flux2.preload_qwen_with_flux2 = True
        self.mock_config.flux2.preload_joycaption_with_flux2 = True
        self.mock_config.flux2.preload_minicpm_with_flux2 = True
        
        engine = AIEngine(self.mock_config)
        engine.ensure_vto_ready()
        
        # Verify flux2 is preloaded
        engine.flux2.ensure_ready.assert_called_once()
        
        # Verify florence is preloaded
        engine.florence._ensure_loaded.assert_called_once()
        
        # Verify optional models are preloaded based on config
        engine.qwen25vl.ensure_ready.assert_called_once()
        engine.joycaption.ensure_ready.assert_called_once()
        engine.minicpm.ensure_ready.assert_called_once()
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    @patch('services.ai_engine.Qwen25VLRunner')
    def test_ensure_vto_ready_qwen_backend(self, mock_qwen25vl, mock_flux2):
        """Test ensure_vto_ready with qwen2_5_vl backend."""
        self.mock_config.flux2.descriptor_compare = False
        self.mock_config.flux2.descriptor_backend = "qwen2_5_vl"
        self.mock_config.flux2.preload_qwen_with_flux2 = True
        
        engine = AIEngine(self.mock_config)
        engine.ensure_vto_ready()
        
        # Verify flux2 is preloaded
        engine.flux2.ensure_ready.assert_called_once()
        
        # Verify qwen is preloaded
        engine.qwen25vl.ensure_ready.assert_called_once()
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_ensure_vto_ready_minicpm_service_backend(self, mock_flux2):
        """Test ensure_vto_ready with minicpm_service backend (external service)."""
        self.mock_config.flux2.descriptor_compare = False
        self.mock_config.flux2.descriptor_backend = "minicpm_service"
        
        engine = AIEngine(self.mock_config)
        engine.ensure_vto_ready()
        
        # Verify flux2 is preloaded
        engine.flux2.ensure_ready.assert_called_once()
        
        # No local models should be preloaded for external service
        # (This is tested by not calling any other ensure_ready methods)
    
    @patch('services.ai_engine.YoloRunner')
    @patch('services.ai_engine.HumanParserRunner')
    @patch('services.ai_engine.FashionDetectionRunner')
    @patch('services.ai_engine.FlorenceRunner')
    @patch('services.ai_engine.Flux2CVTONRunner')
    @patch('services.ai_engine.FashionColorClassifierRunner')
    @patch('services.ai_engine.MiniCPMVRunner')
    def test_ensure_analyze_ready(self, mock_minicpm, mock_fashion_color, mock_flux2,
                                 mock_florence, mock_fashion_detection, mock_human_parser, mock_yolo):
        """Test ensure_analyze_ready preloads required models."""
        self.mock_config.analyze.preload_florence = True
        self.mock_config.analyze.preload_flux_runner = True
        self.mock_config.analyze.preload_minicpm = True
        self.mock_config.analyze.fashion_basecolour_trial_enabled = True
        
        engine = AIEngine(self.mock_config)
        
        # Mock get_flux2_for_analyze to return a mock runner
        mock_analyze_flux2 = Mock()
        with patch.object(engine, 'get_flux2_for_analyze', return_value=mock_analyze_flux2):
            engine.ensure_analyze_ready()
        
        # Verify required models are preloaded
        engine.yolo_runner.ensure_ready.assert_called_once()
        engine.fashion_detection_runner.ensure_ready.assert_called_once()
        engine.parser_runner.ensure_ready.assert_called_once()
        engine.florence._ensure_loaded.assert_called_once()
        engine.minicpm.ensure_ready.assert_called_once()
        mock_analyze_flux2.ensure_ready.assert_called_once()
        engine.fashion_basecolour.ensure_ready.assert_called_once()


class TestAIEngineModelStatus(unittest.TestCase):
    """Tests for AIEngine model status reporting."""
    
    def setUp(self):
        """Set up test configuration and mocks."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.app = Mock(spec=AppConfig)
        self.mock_config.flux2 = Mock(spec=Flux2Config)
        self.mock_config.analyze = Mock(spec=AnalyzeConfig)
        self.mock_config.minicpm = Mock(spec=MiniCPMConfig)
        self.mock_config.color = Mock(spec=ColorConfig)
        
        # Set default config values
        self.mock_config.analyze.enable_human_parser = True
        self.mock_config.flux2.share_base_runner = True
        self.mock_config.analyze.flux_disable_lora = True
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    @patch('services.ai_engine.FlorenceRunner')
    @patch('services.ai_engine.Qwen25VLRunner')
    @patch('services.ai_engine.JoyCaptionRunner')
    @patch('services.ai_engine.MiniCPMVRunner')
    @patch('services.ai_engine.OpenCLIPRunner')
    @patch('services.ai_engine.FashionColorClassifierRunner')
    @patch('services.ai_engine.YoloRunner')
    @patch('services.ai_engine.FashionDetectionRunner')
    @patch('services.ai_engine.HumanParserRunner')
    def test_model_status_all_loaded(self, mock_human_parser, mock_fashion_detection,
                                   mock_yolo, mock_fashion_color, mock_openclip,
                                   mock_minicpm, mock_joycaption, mock_qwen25vl,
                                   mock_florence, mock_flux2):
        """Test model_status when all models are loaded."""
        engine = AIEngine(self.mock_config)
        
        # Mock all models as loaded
        engine.flux2._pipeline = Mock()  # Loaded
        engine.florence._model = Mock()  # Loaded
        engine.qwen25vl.is_loaded = True
        engine.joycaption.is_loaded = True
        engine.minicpm.is_loaded = True
        engine.minicpm.model_id = "test-minicpm-model"
        engine.openclip.is_loaded = True
        engine.openclip.is_available = True
        engine.fashion_basecolour.is_loaded = True
        engine.fashion_basecolour.is_available = True
        engine.fashion_basecolour.model_id = "test-fashion-color-model"
        engine.yolo_runner.is_loaded = True
        engine.yolo_runner.model_path = "/path/to/yolo.pt"
        engine.yolo_runner.expected_label_family = "deepfashion2"
        engine.yolo_runner.label_family = "deepfashion2"
        engine.yolo_runner.class_count = 13
        engine.fashion_detection_runner.is_loaded = True
        engine.fashion_detection_runner.model_path = "/path/to/fashion.pt"
        engine.parser_runner.is_loaded = True
        
        status = engine.model_status()
        
        # Verify all models report as loaded
        self.assertTrue(status["flux2_loaded"])
        self.assertTrue(status["analyze_flux2_loaded"])
        self.assertTrue(status["florence_loaded"])
        self.assertTrue(status["qwen25vl_loaded"])
        self.assertTrue(status["joycaption_loaded"])
        self.assertTrue(status["minicpm_loaded"])
        self.assertEqual(status["minicpm_model_id"], "test-minicpm-model")
        self.assertTrue(status["openclip_loaded"])
        self.assertTrue(status["openclip_available"])
        self.assertTrue(status["fashion_basecolour_loaded"])
        self.assertTrue(status["fashion_basecolour_available"])
        self.assertEqual(status["fashion_basecolour_model_id"], "test-fashion-color-model")
        self.assertTrue(status["yolo_loaded"])
        self.assertEqual(status["yolo_model_path"], "/path/to/yolo.pt")
        self.assertEqual(status["yolo_expected_label_family"], "deepfashion2")
        self.assertEqual(status["yolo_label_family"], "deepfashion2")
        self.assertEqual(status["yolo_class_count"], 13)
        self.assertTrue(status["fashion_detection_loaded"])
        self.assertEqual(status["fashion_detection_model_path"], "/path/to/fashion.pt")
        self.assertTrue(status["human_parser_loaded"])
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_model_status_shared_runner_config(self, mock_flux2):
        """Test model_status reports shared runner configuration correctly."""
        self.mock_config.flux2.share_base_runner = True
        self.mock_config.analyze.flux_disable_lora = True
        
        engine = AIEngine(self.mock_config)
        engine.flux2._pipeline = Mock()
        engine.flux2.runtime_lora_toggle = True
        
        status = engine.model_status()
        
        # Verify shared runner configuration is reported
        self.assertTrue(status["flux2_shared_base_runner"])
        self.assertTrue(status["flux2_runtime_lora_toggle"])
        self.assertTrue(status["analyze_flux2_loaded"])  # Same as main flux2
        self.assertFalse(status["analyze_flux2_isolated"])
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_model_status_isolated_runner_config(self, mock_flux2):
        """Test model_status reports isolated runner configuration correctly."""
        self.mock_config.flux2.share_base_runner = False
        self.mock_config.analyze.flux_disable_lora = True
        
        # Mock separate runners
        mock_main_runner = Mock()
        mock_analyze_runner = Mock()
        mock_main_runner._pipeline = Mock()
        mock_analyze_runner._pipeline = None  # Not loaded
        mock_analyze_runner._startup_metrics = {}  # Fix for startup metrics
        mock_flux2.side_effect = [mock_main_runner, mock_analyze_runner]
        
        engine = AIEngine(self.mock_config)
        
        # Create the isolated runner
        engine.get_flux2_for_analyze()
        
        status = engine.model_status()
        
        # Verify isolated runner configuration is reported
        self.assertFalse(status["flux2_shared_base_runner"])
        self.assertTrue(status["analyze_flux2_isolated"])
        self.assertTrue(status["flux2_loaded"])  # Main runner loaded
        self.assertFalse(status["analyze_flux2_loaded"])  # Analyze runner not loaded
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_model_status_no_models_loaded(self, mock_flux2):
        """Test model_status when no models are loaded."""
        engine = AIEngine(self.mock_config)
        
        # Mock all models as not loaded using PropertyMock for properties
        engine.flux2._pipeline = None
        engine.florence._model = None
        
        # Use PropertyMock for properties that can't be set directly
        type(engine.qwen25vl).is_loaded = PropertyMock(return_value=False)
        type(engine.joycaption).is_loaded = PropertyMock(return_value=False)
        type(engine.minicpm).is_loaded = PropertyMock(return_value=False)
        type(engine.openclip).is_loaded = PropertyMock(return_value=False)
        type(engine.openclip).is_available = PropertyMock(return_value=False)
        type(engine.fashion_basecolour).is_loaded = PropertyMock(return_value=False)
        type(engine.fashion_basecolour).is_available = PropertyMock(return_value=False)
        type(engine.yolo_runner).is_loaded = PropertyMock(return_value=False)
        type(engine.fashion_detection_runner).is_loaded = PropertyMock(return_value=False)
        type(engine.parser_runner).is_loaded = PropertyMock(return_value=False)
        
        status = engine.model_status()
        
        # Verify all models report as not loaded
        self.assertFalse(status["flux2_loaded"])
        self.assertFalse(status["analyze_flux2_loaded"])
        self.assertFalse(status["florence_loaded"])
        self.assertFalse(status["qwen25vl_loaded"])
        self.assertFalse(status["joycaption_loaded"])
        self.assertFalse(status["minicpm_loaded"])
        self.assertFalse(status["openclip_loaded"])
        self.assertFalse(status["openclip_available"])
        self.assertFalse(status["fashion_basecolour_loaded"])
        self.assertFalse(status["fashion_basecolour_available"])
        self.assertFalse(status["yolo_loaded"])
        self.assertFalse(status["fashion_detection_loaded"])
        self.assertFalse(status["human_parser_loaded"])


class TestAIEngineHelperMethods(unittest.TestCase):
    """Tests for AIEngine helper methods."""
    
    def setUp(self):
        """Set up test configuration and mocks."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.app = Mock(spec=AppConfig)
        self.mock_config.flux2 = Mock(spec=Flux2Config)
        self.mock_config.analyze = Mock(spec=AnalyzeConfig)
        self.mock_config.minicpm = Mock(spec=MiniCPMConfig)
        self.mock_config.color = Mock(spec=ColorConfig)
        
        # Set default config values
        self.mock_config.analyze.enable_human_parser = True
        self.mock_config.flux2.share_base_runner = True
        self.mock_config.analyze.flux_disable_lora = True
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_get_heuristic_base_mask_placeholder(self, mock_flux2):
        """Test _get_heuristic_base_mask returns placeholder implementation."""
        engine = AIEngine(self.mock_config)
        
        # Test placeholder implementation
        result_mask, result_metadata = engine._get_heuristic_base_mask(Mock())
        
        self.assertIsNone(result_mask)
        self.assertEqual(result_metadata, {"source": "none"})
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_skin_like_mask_placeholder(self, mock_flux2):
        """Test _skin_like_mask returns placeholder implementation."""
        engine = AIEngine(self.mock_config)
        
        # Mock RGB image with shape
        mock_rgb = Mock()
        mock_rgb.shape = (100, 200, 3)
        
        result = engine._skin_like_mask(mock_rgb)
        
        # Verify result is a numpy array with correct shape
        import numpy as np
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 200))
        self.assertEqual(result.dtype, bool)
        # Verify it's all zeros (empty mask)
        self.assertFalse(result.any())
    
    @patch('services.ai_engine.Flux2CVTONRunner')
    def test_skin_like_mask_no_shape(self, mock_flux2):
        """Test _skin_like_mask handles input without shape attribute."""
        engine = AIEngine(self.mock_config)
        
        # Mock RGB without shape attribute
        mock_rgb = Mock(spec=[])  # No shape attribute
        
        result = engine._skin_like_mask(mock_rgb)
        
        # Verify result is a numpy array with empty shape
        import numpy as np
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0, 0))
        self.assertEqual(result.dtype, bool)


if __name__ == '__main__':
    unittest.main()

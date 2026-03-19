import unittest

from PIL import Image

from ai.core.flux2_cvton_runner import Flux2CVTONRunner


class _FakePipeResult:
    def __init__(self, image):
        self.images = [image]


class _FakePipe:
    def __init__(self):
        self.calls = []
        self.adapter_weights = []
        self.enabled = True

    def set_adapters(self, names, adapter_weights=None):
        self.calls.append(("set_adapters", list(names), list(adapter_weights or [])))
        self.adapter_weights.append(list(adapter_weights or []))

    def enable_lora(self):
        self.calls.append(("enable_lora",))
        self.enabled = True

    def disable_lora(self):
        self.calls.append(("disable_lora",))
        self.enabled = False

    def __call__(self, **kwargs):
        self.calls.append(("infer", sorted(kwargs.keys())))
        return _FakePipeResult(Image.new("RGB", (8, 8), color="white"))


class Flux2RuntimeLoraToggleTests(unittest.TestCase):
    def _make_runner(self):
        runner = Flux2CVTONRunner(
            config={
                "device": "cpu",
                "enable_lora": True,
                "fuse_lora": False,
                "runtime_lora_toggle": True,
            }
        )
        runner._pipeline = _FakePipe()
        runner._lora_loaded = True
        runner._supports_negative_prompt = False
        return runner

    def test_disable_then_enable_lora_runtime(self):
        runner = self._make_runner()

        disabled = runner._set_runtime_lora_state(False)
        enabled = runner._set_runtime_lora_state(True)

        self.assertFalse(disabled)
        self.assertTrue(enabled)
        self.assertIn(("disable_lora",), runner._pipeline.calls)
        self.assertIn(("enable_lora",), runner._pipeline.calls)
        self.assertIn([runner.lora_scale], runner._pipeline.adapter_weights)

    def test_run_tryon_exposes_request_level_lora_metadata(self):
        runner = self._make_runner()
        image = Image.new("RGB", (8, 8), color="black")

        result = runner.run_tryon(
            person_image=image,
            board_image=image,
            prompt="test",
            steps=1,
            seed=1,
            use_lora=False,
        )

        self.assertFalse(result["metadata"]["lora_requested"])
        self.assertFalse(result["metadata"]["lora_effective"])
        self.assertTrue(result["metadata"]["runtime_lora_toggle"])

    def test_disabling_fused_lora_is_rejected(self):
        runner = self._make_runner()
        runner._lora_fused = True

        with self.assertRaises(RuntimeError):
            runner._set_runtime_lora_state(False)

    def test_resolve_tryon_dimensions_preserves_source_aspect(self):
        runner = self._make_runner()
        image = Image.new("RGB", (640, 480), color="white")

        width, height = runner._resolve_tryon_dimensions(image)

        self.assertEqual((width, height), (640, 480))


if __name__ == "__main__":
    unittest.main()

from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from core.qwen_image_edit_runner import QwenImageEditRunner


class _FakePipeline:
    def __init__(self):
        self.lora_calls = []
        self.adapter_calls = []
        self.infer_calls = []
        self.transformer = object()

    def to(self, *_args, **_kwargs):
        return self

    def load_lora_weights(self, *args, **kwargs):
        self.lora_calls.append((args, kwargs))

    def set_adapters(self, *args, **kwargs):
        self.adapter_calls.append((args, kwargs))

    def __call__(self, **kwargs):
        self.infer_calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (32, 32), color="gray")])


def test_runner_omits_optional_guidance_and_negative_prompt(monkeypatch):
    fake_pipe = _FakePipeline()
    monkeypatch.setenv("QWEN_IMAGE_EDIT_ENABLE_LORA", "0")
    monkeypatch.setenv("QWEN_IMAGE_EDIT_DEVICE", "cpu")

    with patch("core.qwen_image_edit_runner.DiffusionPipeline.from_pretrained", return_value=fake_pipe):
        runner = QwenImageEditRunner(model_id="Qwen/Qwen-Image-Edit-2511")
        out, meta = runner.run_edit(
            Image.new("RGB", (32, 32), color="white"),
            prompt="Extract clothing",
            steps=6,
            guidance_scale=None,
            negative_prompt=None,
            seed=7,
        )

    assert out.size == (32, 32)
    assert meta["lora_loaded"] is False
    call = fake_pipe.infer_calls[0]
    assert "guidance_scale" not in call
    assert "negative_prompt" not in call


def test_runner_applies_guidance_and_negative_prompt_when_set(monkeypatch):
    fake_pipe = _FakePipeline()
    monkeypatch.setenv("QWEN_IMAGE_EDIT_ENABLE_LORA", "1")
    monkeypatch.setenv("QWEN_IMAGE_EDIT_DEVICE", "cpu")
    monkeypatch.setenv("QWEN_IMAGE_EDIT_LORA_SCALE", "0.9")

    with patch("core.qwen_image_edit_runner.DiffusionPipeline.from_pretrained", return_value=fake_pipe):
        runner = QwenImageEditRunner(
            model_id="Qwen/Qwen-Image-Edit-2511",
            lora_repo="prithivMLmods/QIE-2511-Extract-Outfit",
            lora_weight_name="QIE-2511-Extract-Outfit-4200.safetensors",
        )
        _out, meta = runner.run_edit(
            Image.new("RGB", (64, 64), color="white"),
            prompt="Extract the clothing and create a flat mockup.",
            steps=8,
            guidance_scale=3.2,
            negative_prompt="  blurry, low quality  ",
            seed=42,
        )

    assert len(fake_pipe.lora_calls) == 1
    call = fake_pipe.infer_calls[0]
    assert call["guidance_scale"] == 3.2
    assert call["negative_prompt"] == "blurry, low quality"
    assert meta["negative_prompt_supplied"] is True
    assert meta["guidance_scale"] == 3.2

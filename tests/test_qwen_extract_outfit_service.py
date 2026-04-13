from pathlib import Path

from PIL import Image

from core.qwen_extract_outfit_service import (
    DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT,
    build_qwen_extract_outfit_request,
    execute_qwen_extract_outfit_request,
)


class _FakeRunner:
    def __init__(self):
        self.calls = []

    def run_edit(
        self,
        image,
        *,
        prompt,
        steps,
        guidance_scale,
        negative_prompt,
        seed,
        output_width=None,
        output_height=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "size": tuple(image.size),
                "output_width": output_width,
                "output_height": output_height,
            }
        )
        return Image.new("RGB", (64, 64), "white"), {"lora_loaded": True}


def test_build_request_uses_alias_and_default_prompt():
    request = build_qwen_extract_outfit_request(
        prompt="  ",
        steps=30,
        seed=99,
        guidance_scale=2.0,
        guidance_scale_alias=3.1,
        negative_prompt="bad",
        negative_prompt_alias="  blurry, low quality  ",
        max_input_edge=1536,
        max_input_edge_alias=2048,
        output_max_edge=None,
        output_max_edge_alias=None,
        upload_output=False,
        include_base64=True,
    )
    assert request.prompt == DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT
    assert request.prompt_default_applied is True
    assert request.guidance_scale == 3.1
    assert request.negative_prompt == "blurry, low quality"
    assert request.max_input_edge == 2048
    assert request.output_max_edge == 2048
    assert request.upload_output is False
    assert request.include_base64 is True


def test_build_request_applies_default_guidance_scale():
    request = build_qwen_extract_outfit_request(
        prompt="extract outfit",
        steps=8,
        seed=42,
        guidance_scale=None,
        guidance_scale_alias=None,
        negative_prompt=None,
        negative_prompt_alias=None,
        max_input_edge=512,
        max_input_edge_alias=None,
        output_max_edge=None,
        output_max_edge_alias=None,
        upload_output=False,
        include_base64=False,
    )
    assert request.guidance_scale == 1.0


def test_execute_request_generates_payload(tmp_path: Path):
    request = build_qwen_extract_outfit_request(
        prompt="Extract only outfit",
        steps=28,
        seed=42,
        guidance_scale=None,
        guidance_scale_alias=None,
        negative_prompt=None,
        negative_prompt_alias=None,
        max_input_edge=512,
        max_input_edge_alias=None,
        output_max_edge=None,
        output_max_edge_alias=None,
        upload_output=False,
        include_base64=True,
    )
    fake_runner = _FakeRunner()
    source = Image.new("RGB", (800, 600), "gray")

    payload = execute_qwen_extract_outfit_request(
        request=request,
        source_image=source,
        runner=fake_runner,
        upload_image_fn=None,
        output_dir=str(tmp_path),
    )

    assert payload["output_url"] == ""
    assert isinstance(payload["image_base64"], str) and payload["image_base64"]
    assert payload["local_path"] == ""
    assert payload["metadata"]["prompt"] == "Extract only outfit"
    assert payload["metadata"]["max_input_edge"] == 512
    assert payload["metadata"]["max_output_edge"] == 512
    assert payload["metadata"]["input_size"] == {"width": 512, "height": 384}
    assert payload["metadata"]["requested_output_size"] == {"width": 512, "height": 384}
    assert fake_runner.calls[0]["size"] == (512, 384)
    assert fake_runner.calls[0]["guidance_scale"] == 1.0
    assert fake_runner.calls[0]["output_width"] == 512
    assert fake_runner.calls[0]["output_height"] == 384


def test_execute_request_can_save_local_when_enabled(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QWEN_EXTRACT_OUTFIT_SAVE_LOCAL", "1")
    request = build_qwen_extract_outfit_request(
        prompt="Extract only outfit",
        steps=8,
        seed=42,
        guidance_scale=1.0,
        guidance_scale_alias=None,
        negative_prompt=None,
        negative_prompt_alias=None,
        max_input_edge=512,
        max_input_edge_alias=None,
        output_max_edge=512,
        output_max_edge_alias=None,
        upload_output=False,
        include_base64=False,
    )
    fake_runner = _FakeRunner()
    source = Image.new("RGB", (800, 600), "gray")

    payload = execute_qwen_extract_outfit_request(
        request=request,
        source_image=source,
        runner=fake_runner,
        upload_image_fn=None,
        output_dir=str(tmp_path),
    )

    assert payload["local_path"]
    assert Path(payload["local_path"]).exists()

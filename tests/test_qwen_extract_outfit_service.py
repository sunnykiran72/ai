from pathlib import Path
from PIL import Image

from core.qwen_extract_outfit_service import (
    DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT,
    PROMPT_SOURCE_EXTRACTED_FALLBACK,
    PROMPT_SOURCE_INPUT_PARALLEL,
    PROMPT_SOURCE_QWEN_FASTPATH,
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
        return Image.new("RGB", (64, 96), "white"), {"lora_loaded": True}


class _FakeMiniCPM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.call_args = []

    def describe_garment(self, image, garment_type=None, prompt_override=None):
        self.call_args.append(
            {
                "size": tuple(image.size),
                "garment_type": garment_type,
                "prompt_override": prompt_override,
            }
        )
        idx = self.calls
        self.calls += 1
        if idx < len(self.responses):
            return str(self.responses[idx])
        return str(self.responses[-1]) if self.responses else ""


def _build_default_request(**kwargs):
    payload = dict(
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
        output_aspect_ratio=None,
        output_aspect_ratio_alias=None,
        upload_output=False,
        include_base64=False,
    )
    payload.update(kwargs)
    return build_qwen_extract_outfit_request(**payload)


def test_build_request_uses_alias_and_default_prompt():
    request = _build_default_request(
        prompt="  ",
        steps=30,
        seed=99,
        guidance_scale=2.0,
        guidance_scale_alias=3.1,
        negative_prompt="bad",
        negative_prompt_alias="  blurry, low quality  ",
        max_input_edge=1536,
        max_input_edge_alias=2048,
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


def test_build_request_keeps_guidance_none_by_default():
    request = _build_default_request()
    assert request.guidance_scale is None


def test_execute_request_generates_payload_with_parallel_prompt(tmp_path: Path):
    request = _build_default_request(include_base64=True)
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(
        [
            "Structured cotton t-shirt with crew neckline, short sleeves, straight hem, visible stitching, and smooth knit texture."
        ]
    )
    source = Image.new("RGB", (800, 600), "gray")

    payload = execute_qwen_extract_outfit_request(
        request=request,
        source_image=source,
        runner=fake_runner,
        minicpm_runner=fake_minicpm,
        upload_image_fn=None,
        output_dir=str(tmp_path),
    )

    assert payload["output_url"] == ""
    assert isinstance(payload["image_base64"], str) and payload["image_base64"]
    assert payload["local_path"] == ""
    assert payload["promptDescriptionSource"] == PROMPT_SOURCE_INPUT_PARALLEL
    assert payload["promptFallbackUsed"] is False
    assert payload["promptDescription"]
    assert payload["garmentCategorySubtype"] == "top"
    assert payload["minicpmJsonValid"] is False
    assert payload["minicpmJsonFallbackUsed"] is True
    assert payload["promptElapsedSeconds"] >= 0
    assert payload["garmentMetadata"]["schema_version"] == "garment_metadata.v1"
    assert payload["metadata"]["prompt"] == "Extract only outfit"
    assert payload["metadata"]["prompt_template"] == "Extract only outfit"
    assert payload["metadata"]["max_input_edge"] == 512
    assert payload["metadata"]["max_output_edge"] == 512
    assert payload["metadata"]["input_original_size"] == {"width": 800, "height": 600}
    assert payload["metadata"]["input_preprocessed_size"] == {"width": 512, "height": 384}
    assert payload["metadata"]["input_size"] == {"width": 512, "height": 384}
    assert payload["metadata"]["requested_output_size"] == {"width": 512, "height": 384}
    assert payload["metadata"]["requested_output_size_aligned"] == {"width": 512, "height": 384}
    assert fake_runner.calls[0]["size"] == (512, 384)
    assert fake_runner.calls[0]["guidance_scale"] is None
    assert fake_runner.calls[0]["output_width"] == 512
    assert fake_runner.calls[0]["output_height"] == 384
    assert fake_minicpm.calls == 1
    assert fake_minicpm.call_args[0]["size"] == (512, 384)
    assert fake_minicpm.call_args[0]["garment_type"] == "top"


def test_execute_request_uses_fallback_prompt_when_parallel_is_weak(tmp_path: Path):
    request = _build_default_request(include_base64=True)
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(
        [
            "none",
            "Fitted midi dress with square neckline, gathered waist seam, short puff sleeves, textured floral fabric, and clean hem finish.",
        ]
    )
    source = Image.new("RGB", (800, 600), "gray")

    payload = execute_qwen_extract_outfit_request(
        request=request,
        source_image=source,
        runner=fake_runner,
        minicpm_runner=fake_minicpm,
        upload_image_fn=None,
        output_dir=str(tmp_path),
    )

    assert payload["promptDescriptionSource"] == PROMPT_SOURCE_EXTRACTED_FALLBACK
    assert payload["promptFallbackUsed"] is True
    assert payload["promptDescription"]
    assert fake_minicpm.calls == 2
    assert fake_minicpm.call_args[0]["size"] == (512, 384)
    assert fake_minicpm.call_args[1]["size"] == (64, 96)
    assert fake_minicpm.call_args[0]["garment_type"] == "top"
    assert fake_minicpm.call_args[1]["garment_type"] == "top"


def test_execute_request_fails_when_both_prompt_attempts_are_unusable(tmp_path: Path):
    request = _build_default_request(include_base64=False)
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(["n/a", "no garment"])
    source = Image.new("RGB", (800, 600), "gray")

    payload = execute_qwen_extract_outfit_request(
        request=request,
        source_image=source,
        runner=fake_runner,
        minicpm_runner=fake_minicpm,
        upload_image_fn=None,
        output_dir=str(tmp_path),
    )
    assert payload["promptDescriptionSource"] == PROMPT_SOURCE_QWEN_FASTPATH
    assert payload["promptFallbackUsed"] is True
    assert "top garment with visible neckline" in payload["promptDescription"].lower()


def test_execute_request_can_save_local_when_enabled(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QWEN_EXTRACT_OUTFIT_SAVE_LOCAL", "1")
    request = _build_default_request(steps=8, output_max_edge=512)
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(
        [
            "Relaxed cotton shirt with collar stand, button placket, long sleeves, chest pocket, and curved hemline."
        ]
    )
    source = Image.new("RGB", (800, 600), "gray")

    payload = execute_qwen_extract_outfit_request(
        request=request,
        source_image=source,
        runner=fake_runner,
        minicpm_runner=fake_minicpm,
        upload_image_fn=None,
        output_dir=str(tmp_path),
    )

    assert payload["local_path"]
    assert Path(payload["local_path"]).exists()


def test_execute_request_prefers_valid_minicpm_json_contract(tmp_path: Path):
    request = _build_default_request(include_base64=False)
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(
        [
            """{
                "category_type": "one_shoulder_top",
                "garment_construction_prompt": "One-shoulder top with asymmetric upper edge, single strap configuration, fitted torso panel, and cropped hem endpoint."
            }"""
        ]
    )
    source = Image.new("RGB", (800, 600), "gray")

    payload = execute_qwen_extract_outfit_request(
        request=request,
        source_image=source,
        runner=fake_runner,
        minicpm_runner=fake_minicpm,
        upload_image_fn=None,
        output_dir=str(tmp_path),
    )

    assert payload["promptDescriptionSource"] == PROMPT_SOURCE_INPUT_PARALLEL
    assert payload["minicpmJsonValid"] is True
    assert payload["minicpmJsonFallbackUsed"] is False
    assert payload["garmentCategorySubtype"] == "one_shoulder_top"
    assert "one-shoulder top" in payload["promptDescription"].lower()


def test_execute_request_renders_subtype_placeholder_into_qwen_prompt(tmp_path: Path):
    request = _build_default_request(
        prompt="Extract mockup for {category_type} garment only",
        include_base64=False,
    )
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(
        [
            """{
                "category_type": "halter_top",
                "garment_construction_prompt": "Halter top with gathered neckline and cropped hem."
            }"""
        ]
    )
    source = Image.new("RGB", (800, 600), "gray")

    payload = execute_qwen_extract_outfit_request(
        request=request,
        source_image=source,
        runner=fake_runner,
        minicpm_runner=fake_minicpm,
        upload_image_fn=None,
        output_dir=str(tmp_path),
    )

    assert fake_runner.calls[0]["prompt"] == "Extract mockup for halter_top garment only"
    assert payload["metadata"]["prompt_template"] == "Extract mockup for {category_type} garment only"
    assert payload["metadata"]["prompt"] == "Extract mockup for halter_top garment only"

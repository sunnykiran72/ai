import io

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import routes.qwen_extract_outfit as route_mod
from core.qwen_extract_outfit_service import (
    DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT,
    PROMPT_GENERATION_FAILED_CODE,
    PROMPT_SOURCE_EXTRACTED_FALLBACK,
    PROMPT_SOURCE_INPUT_PARALLEL,
)

EXPECTED_DEFAULT_PROMPT = DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT


def _png_bytes(color: str = "white") -> bytes:
    image = Image.new("RGB", (64, 96), color=color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


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
        return Image.new("RGB", (80, 120), color="gray"), {"lora_loaded": True}


class _FakeMiniCPM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def describe_garment(self, image, garment_type=None, prompt_override=None):
        _ = image
        _ = garment_type
        _ = prompt_override
        idx = self.calls
        self.calls += 1
        if idx < len(self.responses):
            return str(self.responses[idx])
        return str(self.responses[-1]) if self.responses else ""


class _FakeAIEngine:
    def __init__(self, minicpm):
        self.minicpm = minicpm


def _build_client(fake_runner: _FakeRunner, fake_minicpm: _FakeMiniCPM):
    app = FastAPI()
    app.include_router(route_mod.router)
    app.state.ai_engine = _FakeAIEngine(fake_minicpm)
    route_mod._RUNNER = fake_runner
    route_mod._MINICPM_RUNNER = None
    return TestClient(app)


def test_lab_page_serves_html():
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(["Valid fitted top with sleeves, neckline, seams, and hem details."])
    client = _build_client(fake_runner, fake_minicpm)

    resp = client.get("/dev/qwen/extract-outfit-lab")
    assert resp.status_code == 200
    assert "Qwen Extract-Outfit Lab" in resp.text


def test_route_applies_default_prompt_when_blank():
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(
        ["Structured cotton t-shirt with crew neckline, short sleeves, straight hem, and visible stitch details."]
    )
    client = _build_client(fake_runner, fake_minicpm)

    resp = client.post(
        "/v1/qwen/extract-outfit",
        data={
            "prompt": "   ",
            "upload_output": "false",
            "include_base64": "false",
        },
        files={"file": ("input.png", _png_bytes(), "image/png")},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert payload["data"]["metadata"]["prompt"] == EXPECTED_DEFAULT_PROMPT
    assert payload["data"]["metadata"]["prompt_default_applied"] is True
    assert payload["data"]["promptDescriptionSource"] == PROMPT_SOURCE_INPUT_PARALLEL
    assert payload["data"]["promptFallbackUsed"] is False
    assert payload["data"]["promptDescription"]
    assert payload["data"]["garmentMetadata"]["schema_version"] == "garment_metadata.v1"
    assert fake_runner.calls[0]["prompt"] == EXPECTED_DEFAULT_PROMPT
    assert fake_runner.calls[0]["guidance_scale"] is None
    assert fake_runner.calls[0]["negative_prompt"] == ""
    assert fake_runner.calls[0]["output_width"] == 64
    assert fake_runner.calls[0]["output_height"] == 96


def test_route_normalizes_custom_prompt_and_alias_fields():
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(
        ["Boxy linen shirt with camp collar, front placket, short sleeves, chest pocket, and straight hemline."]
    )
    client = _build_client(fake_runner, fake_minicpm)

    resp = client.post(
        "/v1/qwen/extract-outfit",
        data={
            "prompt": "  Extract   only   outfit  flat mockup  ",
            "guidanceScale": "3.5",
            "negativePrompt": "  low quality, blurry   ",
            "upload_output": "false",
            "include_base64": "false",
        },
        files={"file": ("input.png", _png_bytes("blue"), "image/png")},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["metadata"]["prompt"] == "Extract only outfit flat mockup"
    assert payload["data"]["metadata"]["prompt_default_applied"] is False
    assert fake_runner.calls[0]["guidance_scale"] == 3.5
    assert fake_runner.calls[0]["negative_prompt"] == "low quality, blurry"


def test_route_uses_explicit_output_max_edge():
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(
        ["Tailored blazer with notch lapels, structured shoulders, button closure, welt pockets, and clean hem finish."]
    )
    client = _build_client(fake_runner, fake_minicpm)

    resp = client.post(
        "/v1/qwen/extract-outfit",
        data={
            "prompt": "extract outfit",
            "max_input_edge": "1536",
            "output_max_edge": "512",
            "upload_output": "false",
            "include_base64": "false",
        },
        files={"file": ("input.png", _png_bytes("green"), "image/png")},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["metadata"]["max_input_edge"] == 1536
    assert payload["data"]["metadata"]["max_output_edge"] == 512
    assert payload["data"]["metadata"]["input_size"] == {"width": 64, "height": 96}
    assert payload["data"]["metadata"]["requested_output_size"] == {"width": 64, "height": 96}


def test_route_uses_fallback_prompt_when_initial_prompt_is_weak():
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(
        [
            "none",
            "Pleated midi skirt with high waist, structured waistband, panel seams, soft drape, and clean stitched hemline.",
        ]
    )
    client = _build_client(fake_runner, fake_minicpm)

    resp = client.post(
        "/v1/qwen/extract-outfit",
        data={
            "prompt": "extract outfit",
            "upload_output": "false",
            "include_base64": "false",
        },
        files={"file": ("input.png", _png_bytes("green"), "image/png")},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["promptDescriptionSource"] == PROMPT_SOURCE_EXTRACTED_FALLBACK
    assert payload["data"]["promptFallbackUsed"] is True


def test_route_returns_prompt_generation_failed_when_both_attempts_fail():
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(["n/a", "unknown"])
    client = _build_client(fake_runner, fake_minicpm)

    resp = client.post(
        "/v1/qwen/extract-outfit",
        data={
            "prompt": "extract outfit",
            "upload_output": "false",
            "include_base64": "false",
        },
        files={"file": ("input.png", _png_bytes("green"), "image/png")},
    )

    assert resp.status_code == 502
    payload = resp.json()
    assert payload["detail"]["reason_codes"] == [PROMPT_GENERATION_FAILED_CODE]


def test_route_requires_uploaded_image():
    fake_runner = _FakeRunner()
    fake_minicpm = _FakeMiniCPM(["Valid garment prompt with enough detail and words to pass the quality gate."])
    client = _build_client(fake_runner, fake_minicpm)

    resp = client.post("/v1/qwen/extract-outfit", data={"prompt": "test"})
    assert resp.status_code == 422

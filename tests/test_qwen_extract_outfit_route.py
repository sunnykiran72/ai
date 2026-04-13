import io

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import routes.qwen_extract_outfit as route_mod
from core.qwen_extract_outfit_service import DEFAULT_QWEN_EXTRACT_OUTFIT_PROMPT

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


def _build_client(fake_runner: _FakeRunner):
    app = FastAPI()
    app.include_router(route_mod.router)
    route_mod._RUNNER = fake_runner
    return TestClient(app)


def test_lab_page_serves_html():
    fake = _FakeRunner()
    client = _build_client(fake)

    resp = client.get("/dev/qwen/extract-outfit-lab")
    assert resp.status_code == 200
    assert "Qwen Extract-Outfit Lab" in resp.text


def test_route_applies_default_prompt_when_blank():
    fake = _FakeRunner()
    client = _build_client(fake)

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
    assert fake.calls[0]["prompt"] == EXPECTED_DEFAULT_PROMPT
    assert fake.calls[0]["guidance_scale"] == 1.0
    assert fake.calls[0]["negative_prompt"] == ""
    assert fake.calls[0]["output_width"] == 64
    assert fake.calls[0]["output_height"] == 96


def test_route_normalizes_custom_prompt_and_alias_fields():
    fake = _FakeRunner()
    client = _build_client(fake)

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
    assert fake.calls[0]["guidance_scale"] == 3.5
    assert fake.calls[0]["negative_prompt"] == "low quality, blurry"


def test_route_uses_explicit_output_max_edge():
    fake = _FakeRunner()
    client = _build_client(fake)

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
    # Input stays original because max_input_edge is large; output target follows output_max_edge.
    assert payload["data"]["metadata"]["input_size"] == {"width": 64, "height": 96}
    assert payload["data"]["metadata"]["requested_output_size"] == {"width": 64, "height": 96}


def test_route_requires_uploaded_image():
    fake = _FakeRunner()
    client = _build_client(fake)

    resp = client.post("/v1/qwen/extract-outfit", data={"prompt": "test"})
    assert resp.status_code == 422

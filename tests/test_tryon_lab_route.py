import io
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from routes.tryon_lab import get_ai_engine, router


def _png_bytes(color: str) -> bytes:
    image = Image.new("RGB", (32, 48), color=color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class _Runner:
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def run_tryon(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "image": Image.new("RGB", (64, 96), color="gray"),
            "latency": 1.23,
            "metadata": {"runner": self.name},
        }


def _build_client():
    app = FastAPI()
    app.include_router(router)
    engine = SimpleNamespace(
        flux2=_Runner("tryon"),
        flux2_consistency=_Runner("consistency"),
    )
    app.dependency_overrides[get_ai_engine] = lambda: engine
    return TestClient(app), engine


def test_lab_page_exposes_lora_profile_selector():
    client, _engine = _build_client()

    response = client.get("/dev/flux2/consistency-lab")

    assert response.status_code == 200
    assert 'name="lora_profile"' in response.text
    assert "Flux2 LoRA Try-on Lab" in response.text


def test_tryon_profile_routes_to_flux2_tryon_runner_and_sets_tryon_prompt():
    client, engine = _build_client()

    response = client.post(
        "/dev/flux2/consistency-lab/run",
        data={
            "lora_profile": "tryon",
            "target_type": "top",
            "steps": "28",
            "seed": "42",
            "lora_scale": "1.0",
            "guidance_scale": "2.5",
            "garment_prompt_description": "black one-shoulder top",
            "user_prompt_description": "",
            "prompt": "",
        },
        files={
            "user_image": ("user.png", _png_bytes("white"), "image/png"),
            "garment_image": ("garment.png", _png_bytes("black"), "image/png"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["lab_lora_profile"] == "tryon"
    assert payload["metadata"]["prompt"].startswith("TRYON ")
    assert len(engine.flux2.calls) == 1
    assert len(engine.flux2_consistency.calls) == 0
    assert engine.flux2.calls[0]["lora_mode"] == "tryon"
    assert float(engine.flux2.calls[0]["guidance_scale"]) == 2.5


def test_consistency_profile_still_routes_to_consistency_runner():
    client, engine = _build_client()

    response = client.post(
        "/dev/flux2/consistency-lab/run",
        data={
            "lora_profile": "consistency",
            "target_type": "top",
            "source_worn_types": "top,bottom",
            "steps": "8",
            "seed": "42",
            "lora_scale": "0.4",
            "guidance_scale": "3.5",
            "garment_prompt_description": "black one-shoulder top",
            "user_prompt_description": "",
            "prompt": "",
        },
        files={
            "user_image": ("user.png", _png_bytes("white"), "image/png"),
            "garment_image": ("garment.png", _png_bytes("black"), "image/png"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["lab_lora_profile"] == "consistency"
    assert len(engine.flux2.calls) == 0
    assert len(engine.flux2_consistency.calls) == 1
    assert float(engine.flux2_consistency.calls[0]["guidance_scale"]) == 3.5

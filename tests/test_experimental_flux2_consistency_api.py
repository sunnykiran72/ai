import base64
import importlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def mod():
    return importlib.import_module("experimental_flux2_consistency_api")


def _jpeg_b64_size(payload: str) -> int:
    raw = base64.b64decode(payload.encode("utf-8"))
    return len(raw)


class _DummyPipe:
    def __init__(self):
        self.device = "cpu"
        self.last_scale = 0.8

    def set_adapters(self, _names, adapter_weights=None):
        if adapter_weights:
            self.last_scale = float(adapter_weights[0])

    def __call__(self, **kwargs):
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        # Deterministic by seed + lora scale.
        seed = int(kwargs["generator"].initial_seed())
        val = int((seed + int(self.last_scale * 1000)) % 255)
        image = Image.new("RGB", (width, height), color=(val, 20, 40))
        return SimpleNamespace(images=[image])


def test_valid_minimal_request_returns_200_and_image(mod, monkeypatch):
    monkeypatch.setattr(mod, "_ensure_pipeline", lambda: _DummyPipe())
    monkeypatch.setattr(mod, "_download_image", lambda _url: Image.new("RGB", (640, 960), color="white"))
    client = TestClient(mod.app)

    payload = {
        "user_image_url": "https://example.com/user.jpg",
        "garment_image_url": "https://example.com/garment.png",
    }
    response = client.post("/v1/flux2/consistency-tryon-test", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert isinstance(body["image_base64_jpeg"], str) and len(body["image_base64_jpeg"]) > 100
    assert _jpeg_b64_size(body["image_base64_jpeg"]) > 100


def test_invalid_url_returns_400(mod, monkeypatch):
    monkeypatch.setattr(mod, "_ensure_pipeline", lambda: _DummyPipe())
    client = TestClient(mod.app)
    payload = {
        "user_image_url": "not-a-valid-url",
        "garment_image_url": "https://example.com/garment.png",
    }
    response = client.post("/v1/flux2/consistency-tryon-test", json=payload)
    assert response.status_code == 400


def test_invalid_numeric_bounds_returns_422(mod):
    client = TestClient(mod.app)
    payload = {
        "user_image_url": "https://example.com/user.jpg",
        "garment_image_url": "https://example.com/garment.png",
        "steps": 2,
    }
    response = client.post("/v1/flux2/consistency-tryon-test", json=payload)
    assert response.status_code == 422


def test_determinism_same_seed_same_params(mod, monkeypatch):
    pipe = _DummyPipe()
    monkeypatch.setattr(mod, "_ensure_pipeline", lambda: pipe)
    monkeypatch.setattr(mod, "_download_image", lambda _url: Image.new("RGB", (512, 768), color="white"))
    client = TestClient(mod.app)

    payload = {
        "user_image_url": "https://example.com/user.jpg",
        "garment_image_url": "https://example.com/garment.png",
        "seed": 777,
        "lora_scale": 0.65,
        "steps": 8,
        "guidance_scale": 3.5,
        "output_width": 512,
        "output_height": 768,
    }
    r1 = client.post("/v1/flux2/consistency-tryon-test", json=payload)
    r2 = client.post("/v1/flux2/consistency-tryon-test", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["image_base64_jpeg"] == r2.json()["image_base64_jpeg"]


def test_param_effects_reflect_in_metadata(mod, monkeypatch):
    pipe = _DummyPipe()
    monkeypatch.setattr(mod, "_ensure_pipeline", lambda: pipe)
    monkeypatch.setattr(mod, "_download_image", lambda _url: Image.new("RGB", (640, 960), color="white"))
    client = TestClient(mod.app)

    payload = {
        "user_image_url": "https://example.com/user.jpg",
        "garment_image_url": "https://example.com/garment.png",
        "lora_scale": 1.25,
        "output_width": 768,
        "output_height": 1024,
    }
    response = client.post("/v1/flux2/consistency-tryon-test", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert float(body["metadata"]["lora_scale"]) == 1.25
    assert body["metadata"]["resolution"] == [768, 1024]


def test_download_failure_maps_to_400(mod, monkeypatch):
    monkeypatch.setattr(mod, "_ensure_pipeline", lambda: _DummyPipe())

    def _raise(_url):
        raise mod.HTTPException(status_code=400, detail="bad image")

    monkeypatch.setattr(mod, "_download_image", _raise)
    client = TestClient(mod.app)
    payload = {
        "user_image_url": "https://example.com/user.jpg",
        "garment_image_url": "https://example.com/garment.png",
    }
    response = client.post("/v1/flux2/consistency-tryon-test", json=payload)
    assert response.status_code == 400


def test_isolation_no_main_app_dependencies():
    source = Path("experimental_flux2_consistency_api.py").read_text(encoding="utf-8")
    forbidden = [
        "from config",
        "from services",
        "from routes",
        "get_config(",
        "load_dotenv(",
        "os.getenv(",
    ]
    for token in forbidden:
        assert token not in source

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

import importlib.util


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "install_qwen_extract_outfit.py"
_SPEC = importlib.util.spec_from_file_location("install_qwen_extract_outfit", str(_SCRIPT_PATH))
assert _SPEC and _SPEC.loader
install_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(install_mod)


def test_path_ready_detects_missing_model_shard():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "model_index.json").write_text("{}", encoding="utf-8")
        (root / "transformer").mkdir(parents=True, exist_ok=True)
        (root / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors"}}),
            encoding="utf-8",
        )
        assert install_mod.path_ready(root) is False


def test_path_ready_accepts_present_model_shard():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "model_index.json").write_text("{}", encoding="utf-8")
        (root / "transformer").mkdir(parents=True, exist_ok=True)
        (root / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "diffusion_pytorch_model-00001-of-00001.safetensors"}}),
            encoding="utf-8",
        )
        (root / "transformer" / "diffusion_pytorch_model-00001-of-00001.safetensors").write_bytes(b"ok")
        assert install_mod.path_ready(root) is True


def test_lora_path_ready_rejects_corrupt_safetensors():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        bad = root / "QIE-2511-Extract-Outfit-4200.safetensors"
        bad.write_bytes(b"\x00\x01\x02")
        assert install_mod.lora_path_ready(root, "QIE-2511-Extract-Outfit-4200.safetensors") is False


def test_lora_path_ready_accepts_valid_safetensors():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        good = root / "QIE-2511-Extract-Outfit-4200.safetensors"
        save_file({"x": torch.ones(1)}, str(good))
        assert install_mod.lora_path_ready(root, "QIE-2511-Extract-Outfit-4200.safetensors") is True


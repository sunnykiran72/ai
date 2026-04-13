#!/usr/bin/env python3
"""
Install and verify Qwen Image Edit base model + QIE Extract-Outfit LoRA.

Usage:
  python3 scripts/install_qwen_extract_outfit.py --env-file .env
  python3 scripts/install_qwen_extract_outfit.py --env-file .env --check-only
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


HF_ENV_KEYS = ("HUGGING_FACE_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_LORA_REPO = "prithivMLmods/QIE-2511-Extract-Outfit"
DEFAULT_LORA_WEIGHT = "QIE-2511-Extract-Outfit-4200.safetensors"
DEFAULT_MODEL_LOCAL_DIR = "/workspace/models/qwen-image-edit-2511"
DEFAULT_LORA_LOCAL_DIR = "/workspace/models/qwen-image-edit-lora/QIE-2511-Extract-Outfit"


def parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def write_env_file(path: Path, values: Dict[str, str]) -> None:
    keys = sorted(values.keys())
    lines = [f'{k}="{values[k]}"' if ";" in values[k] else f"{k}={values[k]}" for k in keys]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def choose_hf_token(env_values: Dict[str, str]) -> str:
    for key in HF_ENV_KEYS:
        val = env_values.get(key) or os.getenv(key)
        if val:
            return str(val).strip()
    return ""


def resolve_path(raw_path: str, project_root: Path) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def path_ready(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix.lower() in {".safetensors", ".bin", ".pt"}
    has_core_marker = (path / "model_index.json").exists() or (path / "config.json").exists()
    if not has_core_marker:
        return False

    # If a shard index exists, require all referenced shard files.
    index_candidates = [
        path / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        path / "transformer" / "diffusion_pytorch_model.fp16.safetensors.index.json",
    ]
    for idx_path in index_candidates:
        if not idx_path.exists():
            continue
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            weight_map = data.get("weight_map") or {}
            shard_names = sorted({str(v).strip() for v in weight_map.values() if str(v).strip()})
            if not shard_names:
                return False
            for shard in shard_names:
                shard_path = idx_path.parent / shard
                if (not shard_path.exists()) or shard_path.stat().st_size <= 0:
                    return False
            return True
        except Exception:
            return False

    return True


def _safetensors_valid(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".safetensors":
        return False
    if path.stat().st_size <= 0:
        return False
    try:
        from safetensors import safe_open  # type: ignore
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            return len(keys) > 0
    except Exception:
        # If safetensors parser is unavailable/fails, fall back to non-zero artifact check.
        return path.stat().st_size > 1024


def lora_path_ready(path: Path, weight_name: str) -> bool:
    if not path.exists():
        return False
    wanted = str(weight_name or "").strip()
    if path.is_file():
        if wanted and path.name != wanted:
            return False
        return _safetensors_valid(path)
    if wanted and (path / wanted).exists():
        return _safetensors_valid(path / wanted)
    for p in path.rglob("*.safetensors"):
        if p.is_file():
            return _safetensors_valid(p)
    return False


def print_status(rows: List[Tuple[str, Path, bool]]) -> None:
    print("\nModel status:")
    for name, path, ok in rows:
        status = "OK" if ok else "MISSING"
        print(f"- {name}: {status} ({path})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install/verify Qwen Image Edit + Extract-Outfit LoRA")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lora-repo", default=DEFAULT_LORA_REPO)
    parser.add_argument("--lora-weight-name", default=DEFAULT_LORA_WEIGHT)
    parser.add_argument("--model-local-dir", default=DEFAULT_MODEL_LOCAL_DIR)
    parser.add_argument("--lora-local-dir", default=DEFAULT_LORA_LOCAL_DIR)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    env_input = Path(args.env_file)
    if env_input.is_absolute():
        env_file = env_input
    elif env_input.parent != Path("."):
        env_file = (Path.cwd() / env_input).resolve()
    else:
        env_file = (project_root / env_input).resolve()

    env_values = parse_env_file(env_file)
    hf_token = choose_hf_token(env_values)
    if not hf_token:
        print("[warn] no Hugging Face token found; proceeding with anonymous downloads.")

    model_path = resolve_path(args.model_local_dir, project_root)
    lora_path = resolve_path(args.lora_local_dir, project_root)

    checks = [
        ("qwen-image-edit-2511", model_path, path_ready(model_path)),
        ("qie-extract-outfit-lora", lora_path, lora_path_ready(lora_path, args.lora_weight_name)),
    ]
    print_status(checks)

    missing = [name for name, _, ok in checks if not ok]
    if args.check_only:
        if missing:
            print(f"[warn] missing artifacts: {', '.join(missing)}")
            return 1
        print("[ok] all required model artifacts exist.")
        return 0

    from huggingface_hub import snapshot_download

    if args.force_redownload or not path_ready(model_path):
        print(f"[run] snapshot_download {args.model_id} -> {model_path}")
        model_path.mkdir(parents=True, exist_ok=True)
        kwargs = {"repo_id": args.model_id, "local_dir": str(model_path)}
        if hf_token:
            kwargs["token"] = hf_token
        snapshot_download(**kwargs)

    if args.force_redownload or not lora_path_ready(lora_path, args.lora_weight_name):
        print(f"[run] snapshot_download {args.lora_repo} -> {lora_path}")
        lora_path.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "repo_id": args.lora_repo,
            "local_dir": str(lora_path),
            "allow_patterns": ["*.safetensors", "*.json", "README.md", "*.txt", "ckpts/*"],
        }
        if hf_token:
            kwargs["token"] = hf_token
        snapshot_download(**kwargs)

    env_values["QWEN_IMAGE_EDIT_MODEL_ID"] = str(args.model_id)
    env_values["QWEN_IMAGE_EDIT_MODEL_PATH"] = str(model_path)
    env_values["QWEN_IMAGE_EDIT_LORA_REPO"] = str(args.lora_repo)
    env_values["QWEN_IMAGE_EDIT_LORA_PATH"] = str(lora_path)
    env_values["QWEN_IMAGE_EDIT_LORA_WEIGHT_NAME"] = str(args.lora_weight_name)
    env_values["QWEN_IMAGE_EDIT_ENABLE_LORA"] = env_values.get("QWEN_IMAGE_EDIT_ENABLE_LORA", "1") or "1"
    env_values["QWEN_IMAGE_EDIT_DEVICE"] = env_values.get("QWEN_IMAGE_EDIT_DEVICE", "auto") or "auto"
    if hf_token:
        env_values["HF_TOKEN"] = hf_token
        env_values["HUGGINGFACE_HUB_TOKEN"] = hf_token
    write_env_file(env_file, env_values)
    print(f"[ok] env updated: {env_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

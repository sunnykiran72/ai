#!/usr/bin/env python3
"""
Install and verify AI model weights for single-GPU Runpod deployments.

Usage:
  python3 ai/scripts/install_models.py --env-file ai/.env
  python3 ai/scripts/install_models.py --env-file ai/.env --check-only
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple


HF_ENV_KEYS = ("HUGGING_FACE_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
DEFAULT_FLUX2_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_FASHN_MODEL_ID = "fashn-ai/fashn-vton-1.5"
DEFAULT_FASHN_LOCAL_DIR = "/workspace/models/fashn"
DEFAULT_LORA_LOCAL_DIR = "/workspace/models/flux2-lora/fal-virtual-tryon"


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
            return val
    return ""


def resolve_path(raw_path: str, project_root: Path) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def looks_like_hf_repo_id(raw: str) -> bool:
    if not raw:
        return False
    if raw.startswith(".") or raw.startswith("/"):
        return False
    return "/" in raw


def path_ready(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix.lower() in {".safetensors", ".bin", ".pt"}
    for sentinel in (
        "model_index.json",
        "config.json",
        "README.md",
        "flux-2-klein-9b.safetensors",
        "flux-klein-tryon.safetensors",
        "model.safetensors",
    ):
        if (path / sentinel).exists():
            return True
    # Ignore metadata-only dirs (for example .cache) and require at least one non-hidden artifact.
    for child in path.iterdir():
        if not child.name.startswith("."):
            return True
    return False


def lora_path_ready(path: Path, weight_name: str) -> bool:
    if not path.exists():
        return False
    wanted = str(weight_name or "").strip()
    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            return False
        return (not wanted) or (path.name == wanted)

    if wanted and (path / wanted).exists():
        return True

    for fallback in ("flux-klein-tryon.safetensors", "flux-klein-tryon-comfy.safetensors"):
        if (path / fallback).exists():
            return True

    for p in path.rglob("*.safetensors"):
        if p.is_file():
            return True
    return False


def print_status(rows: List[Tuple[str, Path, bool]]) -> None:
    print("\nModel status:")
    for name, path, ok in rows:
        status = "OK" if ok else "MISSING"
        print(f"- {name}: {status} ({path})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install/verify FLUX2 + FASHN model weights")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--flux2-model-id", default=DEFAULT_FLUX2_MODEL_ID)
    parser.add_argument("--fashn-model-id", default=DEFAULT_FASHN_MODEL_ID)
    parser.add_argument("--fashn-local-dir", default=DEFAULT_FASHN_LOCAL_DIR)
    parser.add_argument("--lora-local-dir", default=DEFAULT_LORA_LOCAL_DIR)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    env_file_input = Path(args.env_file)
    if env_file_input.is_absolute():
        env_file = env_file_input
    else:
        if env_file_input.parent != Path("."):
            env_file = (Path.cwd() / env_file_input).resolve()
        else:
            env_file = (project_root / env_file_input).resolve()

    env_values = parse_env_file(env_file)
    hf_token = choose_hf_token(env_values)
    if not hf_token:
        print("[error] missing Hugging Face token in env or shell.")
        return 1

    flux2_path = resolve_path(env_values.get("FLUX2_MODEL_PATH", "./flux2-klein"), project_root)
    fashn_path = resolve_path(env_values.get("FASHN_V15_PATH", args.fashn_local_dir), project_root)
    lora_raw = env_values.get("FLUX2_LORA_PATH", "fal/flux-klein-9b-virtual-tryon-lora")
    lora_is_repo = looks_like_hf_repo_id(lora_raw)
    lora_path = resolve_path(args.lora_local_dir if lora_is_repo else lora_raw, project_root)
    lora_weight_name = env_values.get("FLUX2_LORA_WEIGHT_NAME", "flux-klein-tryon.safetensors")

    checks = [
        ("flux2-klein-9b", flux2_path, path_ready(flux2_path)),
        ("fashn-vton-1.5", fashn_path, path_ready(fashn_path)),
    ]
    if not args.skip_lora:
        checks.append(("flux2-tryon-lora", lora_path, lora_path_ready(lora_path, lora_weight_name)))
    print_status(checks)

    missing = [name for name, _, ok in checks if not ok]
    if args.check_only:
        if missing:
            print(f"[warn] missing artifacts: {', '.join(missing)}")
            return 1
        print("[ok] all required model artifacts exist.")
        return 0

    from huggingface_hub import snapshot_download

    if args.force_redownload or not path_ready(flux2_path):
        print(f"[run] snapshot_download {args.flux2_model_id} -> {flux2_path}")
        flux2_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.flux2_model_id,
            local_dir=str(flux2_path),
            token=hf_token,
            resume_download=True,
        )

    if args.force_redownload or not path_ready(fashn_path):
        print(f"[run] snapshot_download {args.fashn_model_id} -> {fashn_path}")
        fashn_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.fashn_model_id,
            local_dir=str(fashn_path),
            token=hf_token,
            resume_download=True,
        )

    if not args.skip_lora and (args.force_redownload or not lora_path_ready(lora_path, lora_weight_name)):
        lora_repo = lora_raw if lora_is_repo else ""
        if lora_repo:
            print(f"[run] snapshot_download {lora_repo} -> {lora_path}")
            lora_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=lora_repo,
                local_dir=str(lora_path),
                token=hf_token,
                resume_download=True,
                allow_patterns=["*.safetensors", "*.json", "README.md", "*.txt"],
            )
            env_values["FLUX2_LORA_PATH"] = str(lora_path)

    env_values["FLUX2_MODEL_PATH"] = str(flux2_path)
    env_values["FASHN_V15_PATH"] = str(fashn_path)
    env_values["FLUX2_ENABLE_LORA"] = env_values.get("FLUX2_ENABLE_LORA", "1") or "1"
    env_values["FLUX2_REQUIRE_LORA"] = env_values.get("FLUX2_REQUIRE_LORA", "1") or "1"
    env_values["FLUX2_LORA_AUTO_DOWNLOAD"] = env_values.get("FLUX2_LORA_AUTO_DOWNLOAD", "1") or "1"
    env_values["FLUX2_LORA_FALLBACK_REPO"] = env_values.get(
        "FLUX2_LORA_FALLBACK_REPO", "fal/flux-klein-9b-virtual-tryon-lora"
    ) or "fal/flux-klein-9b-virtual-tryon-lora"
    env_values["HF_TOKEN"] = hf_token
    env_values["HUGGINGFACE_HUB_TOKEN"] = hf_token
    write_env_file(env_file, env_values)
    print(f"[ok] env updated: {env_file}")

    checks = [
        ("flux2-klein-9b", flux2_path, path_ready(flux2_path)),
        ("fashn-vton-1.5", fashn_path, path_ready(fashn_path)),
    ]
    if not args.skip_lora:
        checks.append(("flux2-tryon-lora", lora_path, lora_path_ready(lora_path, lora_weight_name)))
    print_status(checks)
    if any(not ok for _, _, ok in checks):
        return 1
    print("[ok] model install complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Bootstrap script for fast AI service deployment across different GPU types.

Usage:
  python3 ai/scripts/bootstrap_deploy.py --profile h100_sxm --env-file ai/.env
"""

import argparse
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROFILE_DEFAULTS: Dict[str, Dict[str, str]] = {
    "h100_sxm": {
        "GPU_CONCURRENCY": "1",
        "PRELOAD": "1",
        "PRELOAD_ANALYZE": "0",
        "FLUX2_DEVICE": "cuda",
        "FLUX2_DTYPE": "bf16",
        "FLUX2_ENABLE_TF32": "1",
        "FLUX2_ENABLE_CHANNELS_LAST": "1",
        "FLUX2_COMPILE_MODE": "max-autotune",
        "FLUX2_COMPILE_VAE_DECODE": "0",
        "FLUX2_ALLOW_COMPILE_FALLBACK": "1",
        "FLUX2_WARMUPS": "1",
        "FLUX2_FUSE_LORA": "1",
        "FLUX2_WIDTH": "512",
        "FLUX2_HEIGHT": "768",
        "FLUX2_GUIDANCE_SCALE": "3.5",
    },
    "rtx_6000_ada": {
        "GPU_CONCURRENCY": "1",
        "PRELOAD": "1",
        "PRELOAD_ANALYZE": "0",
        "FLUX2_DEVICE": "cuda",
        "FLUX2_DTYPE": "fp16",
        "FLUX2_ENABLE_TF32": "1",
        "FLUX2_ENABLE_CHANNELS_LAST": "1",
        "FLUX2_COMPILE_MODE": "none",
        "FLUX2_COMPILE_VAE_DECODE": "0",
        "FLUX2_ALLOW_COMPILE_FALLBACK": "1",
        "FLUX2_WARMUPS": "1",
        "FLUX2_FUSE_LORA": "1",
        "FLUX2_WIDTH": "512",
        "FLUX2_HEIGHT": "768",
        "FLUX2_GUIDANCE_SCALE": "3.5",
    },
    "rtx_pro_6000_wk": {
        "GPU_CONCURRENCY": "1",
        "PRELOAD": "1",
        "PRELOAD_ANALYZE": "0",
        "FLUX2_DEVICE": "cuda",
        "FLUX2_DTYPE": "fp16",
        "FLUX2_ENABLE_TF32": "1",
        "FLUX2_ENABLE_CHANNELS_LAST": "1",
        "FLUX2_COMPILE_MODE": "reduce-overhead",
        "FLUX2_COMPILE_VAE_DECODE": "0",
        "FLUX2_ALLOW_COMPILE_FALLBACK": "1",
        "FLUX2_WARMUPS": "1",
        "FLUX2_FUSE_LORA": "1",
        "FLUX2_WIDTH": "512",
        "FLUX2_HEIGHT": "768",
        "FLUX2_GUIDANCE_SCALE": "3.5",
    },
}

HF_ENV_KEYS = ("HUGGING_FACE_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
REQUIRED_SECRETS = (
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_OUTPUT_CONTAINER",
)
DEFAULT_FLUX2_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_FASHN_MODEL_ID = "fashn-ai/fashn-vton-1.5"
DEFAULT_FASHN_LOCAL_DIR = "/workspace/models/fashn"
DEFAULT_LORA_LOCAL_DIR = "/workspace/models/flux2-lora/fal-virtual-tryon"
FLUX2_RUNTIME_PINS = (
    "huggingface-hub>=0.36,<1.0",
    "transformers==4.57.6",
    "hf_transfer",
)
DIFFUSERS_GIT_REF = "git+https://github.com/huggingface/diffusers.git"


def run(cmd: List[str], cwd: Path, extra_env: Optional[Dict[str, str]] = None) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"[run] {printable}")
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def choose_hf_token(env_data: Dict[str, str]) -> str:
    for key in HF_ENV_KEYS:
        val = env_data.get(key) or os.getenv(key)
        if val:
            return val
    return ""


def write_env_file(path: Path, values: Dict[str, str]) -> None:
    keys = sorted(values.keys())
    lines = [f'{k}="{values[k]}"' if ";" in values[k] else f"{k}={values[k]}" for k in keys]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def ensure_required(values: Dict[str, str]) -> Tuple[bool, List[str]]:
    missing = [k for k in REQUIRED_SECRETS if not values.get(k)]
    hf_token = choose_hf_token(values)
    if not hf_token:
        missing.append("HUGGING_FACE_KEY|HF_TOKEN|HUGGINGFACE_HUB_TOKEN")
    return (len(missing) == 0, missing)


def resolve_path(raw_path: str, project_root: Path) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def looks_like_hf_repo_id(raw: str) -> bool:
    value = (raw or "").strip()
    if not value:
        return False
    if value.startswith(".") or value.startswith("/"):
        return False
    return "/" in value


def install_requirements(python_bin: str, project_root: Path, skip_flash_attn: bool) -> None:
    run([python_bin, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], project_root)
    try:
        run([python_bin, "-m", "pip", "install", "-r", "requirements.txt"], project_root)
    except subprocess.CalledProcessError:
        if skip_flash_attn:
            raise
        print("[warn] requirements install failed; retrying without flash-attn.")
        req_path = project_root / "requirements.txt"
        filtered_lines = []
        for raw in req_path.read_text(encoding="utf-8").splitlines():
            if "flash-attn" in raw.replace(" ", ""):
                continue
            filtered_lines.append(raw)
        no_flash_path = project_root / ".requirements_no_flash_attn.txt"
        no_flash_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
        try:
            run([python_bin, "-m", "pip", "install", "-r", str(no_flash_path)], project_root)
        finally:
            try:
                no_flash_path.unlink()
            except Exception:
                pass

    if platform.system().lower() == "linux" and not skip_flash_attn:
        try:
            run(
                [python_bin, "-m", "pip", "install", "--no-build-isolation", "flash-attn"],
                project_root,
            )
        except subprocess.CalledProcessError:
            print("[warn] flash-attn install failed; continuing without it.")


def install_model_weights(
    env_values: Dict[str, str],
    project_root: Path,
    flux2_model_id: str,
    fashn_model_id: str,
    fashn_local_dir: str,
    lora_local_dir: str,
    skip_lora_prefetch: bool,
) -> None:
    hf_token = choose_hf_token(env_values)
    if not hf_token:
        print("[warn] HF token missing; skipping model weight installation.")
        return

    from huggingface_hub import snapshot_download

    flux2_path = resolve_path(env_values.get("FLUX2_MODEL_PATH", "./flux2-klein"), project_root)
    fashn_path = resolve_path(fashn_local_dir, project_root)

    flux2_path.mkdir(parents=True, exist_ok=True)
    fashn_path.mkdir(parents=True, exist_ok=True)

    print(f"[run] snapshot_download {flux2_model_id} -> {flux2_path}")
    snapshot_download(
        repo_id=flux2_model_id,
        local_dir=str(flux2_path),
        token=hf_token,
        resume_download=True,
    )

    print(f"[run] snapshot_download {fashn_model_id} -> {fashn_path}")
    snapshot_download(
        repo_id=fashn_model_id,
        local_dir=str(fashn_path),
        token=hf_token,
        resume_download=True,
    )
    env_values["FASHN_V15_PATH"] = str(fashn_path)

    if skip_lora_prefetch:
        return

    lora_path = (env_values.get("FLUX2_LORA_PATH") or "").strip()
    if not looks_like_hf_repo_id(lora_path):
        return

    lora_target = resolve_path(lora_local_dir, project_root)
    lora_target.mkdir(parents=True, exist_ok=True)
    print(f"[run] snapshot_download {lora_path} -> {lora_target}")
    snapshot_download(
        repo_id=lora_path,
        local_dir=str(lora_target),
        token=hf_token,
        resume_download=True,
    )
    env_values["FLUX2_LORA_PATH"] = str(lora_target)


def install_flux2_runtime_stack(python_bin: str, project_root: Path) -> None:
    # Required for FLUX.2-klein-9B + Florence coexistence.
    run([python_bin, "-m", "pip", "install", "-U", *FLUX2_RUNTIME_PINS], project_root)
    run([python_bin, "-m", "pip", "install", "-U", DIFFUSERS_GIT_REF], project_root)


def prefetch_models(python_bin: str, project_root: Path) -> None:
    code = """
from huggingface_hub import snapshot_download
models = [
    "microsoft/Florence-2-large",
    "mattmdjaga/segformer_b2_clothes",
]
for model in models:
    try:
        snapshot_download(repo_id=model)
        print(f"[ok] prefetched {model}")
    except Exception as err:
        print(f"[warn] failed prefetch for {model}: {err}")
"""
    run([python_bin, "-c", code], project_root)


def print_gpu_info(python_bin: str, project_root: Path) -> None:
    code = """
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("total_vram_gb:", round(props.total_memory / (1024**3), 2))
"""
    run([python_bin, "-c", code], project_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap deployment for AI service")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS.keys()), default="h100_sxm")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-flash-attn", action="store_true")
    parser.add_argument("--skip-model-weights", action="store_true")
    parser.add_argument("--skip-flux2-stack", action="store_true")
    parser.add_argument("--skip-lora-prefetch", action="store_true")
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--flux2-model-id", default=DEFAULT_FLUX2_MODEL_ID)
    parser.add_argument("--fashn-model-id", default=DEFAULT_FASHN_MODEL_ID)
    parser.add_argument("--fashn-local-dir", default=DEFAULT_FASHN_LOCAL_DIR)
    parser.add_argument("--lora-local-dir", default=DEFAULT_LORA_LOCAL_DIR)
    parser.add_argument("--require-azure-upload", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    env_file_input = Path(args.env_file)
    if env_file_input.is_absolute():
        env_file = env_file_input
    else:
        # If caller passes a path like "ai/.env" from repo root, resolve from CWD.
        # If caller passes a bare filename like ".env", resolve from project_root.
        if env_file_input.parent != Path("."):
            env_file = (Path.cwd() / env_file_input).resolve()
        else:
            env_file = (project_root / env_file_input).resolve()

    env_values = parse_env_file(env_file)
    env_values.update(PROFILE_DEFAULTS[args.profile])

    env_values.setdefault("AZURE_STORAGE_OUTPUT_CONTAINER", "wardrobe-outputs")
    env_values.setdefault("AZURE_STORAGE_INPUT_CONTAINER", "wardrobe-inputs")
    env_values.setdefault("FASHN_V15_PATH", args.fashn_local_dir)
    env_values["REQUIRE_AZURE_UPLOAD"] = "1" if args.require_azure_upload else env_values.get("REQUIRE_AZURE_UPLOAD", "0")

    # Keep FLUX2 model path absolute so startup behavior is stable regardless of CWD.
    flux2_path = resolve_path(env_values.get("FLUX2_MODEL_PATH", "./flux2-klein"), project_root)
    env_values["FLUX2_MODEL_PATH"] = str(flux2_path)

    # Mirror any available HF token key so libraries can pick it up consistently.
    hf_token = choose_hf_token(env_values)
    if hf_token:
        env_values["HF_TOKEN"] = hf_token
        env_values["HUGGINGFACE_HUB_TOKEN"] = hf_token

    write_env_file(env_file, env_values)
    print(f"[ok] wrote env file: {env_file}")

    valid, missing = ensure_required(env_values)
    if not valid:
        print("[warn] missing required secrets:")
        for item in missing:
            print(f"  - {item}")
        print("[warn] continue is allowed, but runtime calls may fail until these are set.")

    if not args.skip_install:
        install_requirements(args.python_bin, project_root, args.skip_flash_attn)
        if not args.skip_flux2_stack:
            install_flux2_runtime_stack(args.python_bin, project_root)

    print_gpu_info(args.python_bin, project_root)

    if not args.skip_model_weights:
        install_model_weights(
            env_values=env_values,
            project_root=project_root,
            flux2_model_id=args.flux2_model_id,
            fashn_model_id=args.fashn_model_id,
            fashn_local_dir=args.fashn_local_dir,
            lora_local_dir=args.lora_local_dir,
            skip_lora_prefetch=args.skip_lora_prefetch,
        )
        write_env_file(env_file, env_values)
        print(f"[ok] updated env file after model install: {env_file}")

    if not args.skip_prefetch:
        prefetch_models(args.python_bin, project_root)

    print("[ok] bootstrap complete")
    print(f"[next] start server: {args.python_bin} -m ai.main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

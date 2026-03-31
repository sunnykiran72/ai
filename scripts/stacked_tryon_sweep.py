#!/usr/bin/env python3
"""
Run a small stacked-LoRA sweep against /v1/flux2/tryon.

This script reuses the same request payload and varies only the LoRA scales.

Example:
  python3 ai/scripts/stacked_tryon_sweep.py \
    --base-url http://127.0.0.1:8000 \
    --payload-file /tmp/tryon_payload.json \
    --bfs-scales 0.35 0.50 0.65 0.80 \
    --tryon-scale 1.0
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


def _json_or_text(resp: requests.Response) -> Any:
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        return resp.json()
    return resp.text


def _load_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("payload-file must contain a JSON object")
    return payload


def _summarize_response(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"ok": False, "error": "non_json"}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    metadata = data.get("metadata") if isinstance(data, dict) else {}
    return {
        "ok": payload.get("status") == "success",
        "output_url": data.get("output_url") if isinstance(data, dict) else None,
        "latency": data.get("latency") if isinstance(data, dict) else None,
        "lora_mode": metadata.get("lora_requested_mode"),
        "lora_effective": metadata.get("lora_effective"),
        "bfs_scale": metadata.get("bfs_lora_scale_override"),
        "tryon_scale": metadata.get("lora_scale_override"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stacked LoRA sweep against /v1/flux2/tryon")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--payload-file", required=True, help="JSON file containing the request body")
    parser.add_argument("--bfs-scales", nargs="+", type=float, required=True, help="BFS scales to evaluate")
    parser.add_argument("--tryon-scale", type=float, default=1.0, help="Try-on LoRA scale to keep fixed")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", help="Optional directory to store raw responses")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    payload = _load_payload(Path(args.payload_file))
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for bfs_scale in args.bfs_scales:
        trial = copy.deepcopy(payload)
        trial["loraMode"] = "stacked"
        trial["loraScale"] = float(args.tryon_scale)
        trial["bfsLoraScale"] = float(bfs_scale)

        t0 = time.time()
        resp = requests.post(f"{base}/v1/flux2/tryon", json=trial, timeout=args.timeout)
        elapsed = time.time() - t0
        body = _json_or_text(resp)
        summary = _summarize_response(body)
        summary.update(
            {
                "http_status": resp.status_code,
                "elapsed": round(elapsed, 3),
                "trial": {
                    "loraMode": trial["loraMode"],
                    "loraScale": trial["loraScale"],
                    "bfsLoraScale": trial["bfsLoraScale"],
                },
            }
        )
        results.append(summary)

        print(json.dumps(summary, indent=2))

        if out_dir:
            out_path = out_dir / f"bfs_{str(bfs_scale).replace('.', '_')}.json"
            out_path.write_text(json.dumps(body, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

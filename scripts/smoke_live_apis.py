#!/usr/bin/env python3
"""
Live API smoke tester for Glamify AI service.

Example:
  python3 ai/scripts/smoke_live_apis.py \
    --base-url https://<your-service> \
    --analyze-image /path/to/top.png \
    --user-image-url https://.../user.jpg \
    --garment-image-url https://.../garment.jpg
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
from PIL import Image


def pretty(name: str, status: int, seconds: float, payload: Any) -> None:
    print(f"\n=== {name} ===")
    print(f"status: {status}")
    print(f"latency: {seconds:.3f}s")
    print(json.dumps(payload, indent=2)[:3000])

def _json_or_text(resp: requests.Response) -> Any:
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        return resp.json()
    return resp.text

def _extract_analyze_summary(payload: Dict[str, Any]) -> Tuple[str, str]:
    items = payload.get("items") or []
    if not items:
        return ("", "")
    first = items[0] or {}
    return (str(first.get("type", "")), str(first.get("type_source", "")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--analyze-image")
    parser.add_argument("--user-image-url", required=True)
    parser.add_argument("--garment-image-url", required=True)
    parser.add_argument("--garment-type", default="top")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--expect-hybrid", action="store_true")
    parser.add_argument("--expect-healthy", action="store_true")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    failures = []

    if args.analyze_image:
        analyze_path = Path(args.analyze_image)
        if not analyze_path.exists():
            raise FileNotFoundError(f"Analyze image not found: {analyze_path}")
    else:
        analyze_path = Path("/tmp/glamify_analyze_test.png")
        Image.new("RGB", (512, 768), (180, 80, 80)).save(analyze_path)

    # /health
    t0 = time.time()
    r = requests.get(f"{base}/health", timeout=args.timeout)
    health_payload = _json_or_text(r)
    pretty("/health", r.status_code, time.time() - t0, health_payload)
    if r.status_code != 200:
        failures.append("/health non-200")
    if args.expect_healthy and isinstance(health_payload, dict):
        if health_payload.get("status") != "healthy":
            failures.append("/health status != healthy")
        if not bool(health_payload.get("gpu_available")):
            failures.append("/health gpu_available is false")
    if args.expect_hybrid and isinstance(health_payload, dict):
        flags = health_payload.get("feature_flags", {})
        if not bool(flags.get("use_florence_hybrid_verify")):
            failures.append("feature flag use_florence_hybrid_verify is disabled")

    # /analyze
    with analyze_path.open("rb") as f:
        files = {"file": (analyze_path.name, f, "image/png")}
        data = {"type": args.garment_type}
        t1 = time.time()
        r2 = requests.post(f"{base}/analyze", files=files, data=data, timeout=args.timeout)
    payload2 = _json_or_text(r2)
    pretty("/analyze", r2.status_code, time.time() - t1, payload2)
    analyze_type = ""
    if r2.status_code != 200:
        failures.append("/analyze non-200")
    elif isinstance(payload2, dict):
        analyze_type, analyze_source = _extract_analyze_summary(payload2)
        if not payload2.get("items"):
            failures.append("/analyze returned empty items")
        if args.expect_hybrid and not bool(payload2.get("hybrid_verify_enabled")):
            failures.append("/analyze hybrid_verify_enabled is false")
        if args.expect_hybrid and analyze_source != "florence_hybrid":
            failures.append("/analyze first item type_source != florence_hybrid")
    else:
        failures.append("/analyze invalid JSON payload")

    # /analzye (legacy alias)
    with analyze_path.open("rb") as f:
        files = {"file": (analyze_path.name, f, "image/png")}
        data = {"type": args.garment_type}
        t1b = time.time()
        r2b = requests.post(f"{base}/analzye", files=files, data=data, timeout=args.timeout)
    payload2b = _json_or_text(r2b)
    pretty("/analzye", r2b.status_code, time.time() - t1b, payload2b)
    if r2b.status_code != 200:
        failures.append("/analzye non-200")
    elif isinstance(payload2b, dict):
        analzye_type, analzye_source = _extract_analyze_summary(payload2b)
        if args.expect_hybrid and not bool(payload2b.get("hybrid_verify_enabled")):
            failures.append("/analzye hybrid_verify_enabled is false")
        if args.expect_hybrid and analzye_source != "florence_hybrid":
            failures.append("/analzye first item type_source != florence_hybrid")
        if analyze_type and analzye_type and analyze_type != analzye_type:
            failures.append("/analyze and /analzye returned different first-item types")
    else:
        failures.append("/analzye invalid JSON payload")

    # /v1/flux2/tryon
    body = {
        "user_image_url": args.user_image_url,
        "garment_image_url": args.garment_image_url,
        "user_top_description": "black t-shirt",
        "steps": 6,
        "seed": 23,
    }
    t2 = time.time()
    r3 = requests.post(f"{base}/v1/flux2/tryon", json=body, timeout=args.timeout)
    payload3 = _json_or_text(r3)
    pretty("/v1/flux2/tryon", r3.status_code, time.time() - t2, payload3)
    if r3.status_code != 200:
        failures.append("/v1/flux2/tryon non-200")
    elif isinstance(payload3, dict):
        if payload3.get("status") != "success":
            failures.append("/v1/flux2/tryon status != success")
        if not payload3.get("result_url"):
            failures.append("/v1/flux2/tryon missing result_url")
    else:
        failures.append("/v1/flux2/tryon invalid JSON payload")

    if failures:
        print("\nSMOKE RESULT: FAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nSMOKE RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Benchmark FLUX try-on vs Qwen image editing on the same try-on cases.

This script compares:
1) Current /v1/flux2/tryon API (FLUX try-on LoRA path)
2) Qwen image edit API (DashScope-compatible)

It captures:
- request success/failure
- wall-clock latency
- model-reported generation latency (when available)
- optional type-hit proxy score using /analyze on generated outputs

Input cases file format (JSON):
[
  {
    "case_id": "single_top_01",
    "user_image": "https://.../user.png",
    "user_prompt_description": "front-facing full-body studio portrait",
    "source_worn_types": ["top", "bottom"],
    "products": [
      {
        "image": "https://.../top.png",
        "promptDescription": "red fitted sleeveless top",
        "targetType": "top"
      }
    ],
    "steps": 20,
    "seed": 42
  }
]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


DEFAULT_QWEN_MODEL = "qwen-image-2.0-pro"
DEFAULT_QWEN_ENDPOINT = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
DEFAULT_FLUX_ENDPOINT = "/v1/flux2/tryon"
DEFAULT_ANALYZE_ENDPOINT = "/analyze"
ALLOWED_TYPES = {"top", "bottom", "dress", "outer"}


class BenchmarkError(Exception):
    """Raised for benchmark input/runtime errors."""


@dataclass
class RequestResult:
    ok: bool
    status_code: int
    elapsed_s: float
    output_url: Optional[str]
    generation_s: Optional[float]
    error: Optional[str]
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "status_code": self.status_code,
            "elapsed_s": self.elapsed_s,
            "output_url": self.output_url,
            "generation_s": self.generation_s,
            "error": self.error,
            "raw": self.raw,
        }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _safe_join_url(base: str, endpoint: str) -> str:
    return f"{base.rstrip('/')}/{endpoint.lstrip('/')}"


def _normalize_type(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in ALLOWED_TYPES:
        return text
    return None


def _extract_requested_types(products: Sequence[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for item in products:
        kind = _normalize_type(item.get("targetType"))
        if not kind:
            prompt = str(item.get("promptDescription") or "").lower()
            if any(token in prompt for token in ("dress", "gown")):
                kind = "dress"
            elif any(token in prompt for token in ("pant", "trouser", "jean", "skirt", "short")):
                kind = "bottom"
            elif any(token in prompt for token in ("jacket", "coat", "blazer", "hoodie", "outer")):
                kind = "outer"
            else:
                kind = "top"
        if kind not in seen:
            seen.append(kind)
    return seen


def _extract_flux_output_url(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("result_url", "output_url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("result_url", "output_url"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_flux_generation_s(payload: Dict[str, Any]) -> Optional[float]:
    if isinstance(payload.get("data"), dict):
        data = payload["data"]
        timings = data.get("timings")
        if isinstance(timings, dict):
            for key in ("generation", "total"):
                value = timings.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        value = data.get("latency")
        if isinstance(value, (int, float)):
            return float(value)
    value = payload.get("latency")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_qwen_output_url(payload: Dict[str, Any]) -> Optional[str]:
    output = payload.get("output")
    if not isinstance(output, dict):
        return None
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    msg = first.get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if not isinstance(content, list):
        return None
    for entry in content:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if isinstance(image, str) and image.strip():
            return image.strip()
    return None


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _compact_json(payload: Dict[str, Any], limit: int = 2000) -> Dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= limit:
        return payload
    return {"truncated": True, "text": text[:limit]}


def _request_flux_tryon(
    *,
    base_url: str,
    endpoint: str,
    timeout_s: int,
    payload: Dict[str, Any],
) -> RequestResult:
    url = _safe_join_url(base_url, endpoint)
    started = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        elapsed = time.perf_counter() - started
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text[:2000]}
        ok = resp.status_code == 200 and str(body.get("status", "")).lower() == "success"
        return RequestResult(
            ok=ok,
            status_code=int(resp.status_code),
            elapsed_s=float(elapsed),
            output_url=_extract_flux_output_url(body) if ok else None,
            generation_s=_extract_flux_generation_s(body),
            error=None if ok else f"flux request failed (status={resp.status_code})",
            raw=_compact_json(body),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return RequestResult(
            ok=False,
            status_code=0,
            elapsed_s=float(elapsed),
            output_url=None,
            generation_s=None,
            error=f"{type(exc).__name__}: {exc}",
            raw={},
        )


def _build_qwen_prompt(case: Dict[str, Any], used_products: Sequence[Dict[str, Any]]) -> str:
    user_desc = " ".join(str(case.get("user_prompt_description") or "").split()).strip()
    product_chunks: List[str] = []
    for idx, product in enumerate(used_products, start=2):
        desc = " ".join(str(product.get("promptDescription") or "").split()).strip() or "garment"
        kind = _normalize_type(product.get("targetType")) or "garment"
        product_chunks.append(f"Image {idx}: {kind} -> {desc}")
    products_text = "; ".join(product_chunks) if product_chunks else "No garment references provided."
    identity_tail = (
        "Preserve the same person identity, face, body shape, pose, hair, camera framing, "
        "background, and lighting. Keep edits confined to clothing regions only."
    )
    if user_desc:
        return (
            "Identity-preserving virtual try-on image edit.\n"
            f"Person context: {user_desc}\n"
            f"Garment references: {products_text}\n"
            "Apply the garments from the reference images onto Image 1 naturally with realistic fit, "
            "fabric behavior, seams, and color fidelity.\n"
            f"{identity_tail}"
        )
    return (
        "Identity-preserving virtual try-on image edit.\n"
        f"Garment references: {products_text}\n"
        "Apply the garments from the reference images onto Image 1 naturally with realistic fit, "
        "fabric behavior, seams, and color fidelity.\n"
        f"{identity_tail}"
    )


def _request_qwen_edit(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    timeout_s: int,
    case: Dict[str, Any],
    qwen_max_inputs: int,
    qwen_size: Optional[str],
    qwen_prompt_extend: bool,
    qwen_watermark: bool,
) -> Tuple[RequestResult, Dict[str, Any]]:
    products = list(case.get("products") or [])
    max_products = max(0, int(qwen_max_inputs) - 1)
    used_products = products[:max_products]
    dropped = max(0, len(products) - len(used_products))

    content: List[Dict[str, Any]] = [{"image": case["user_image"]}]
    for product in used_products:
        content.append({"image": product["image"]})
    content.append({"text": _build_qwen_prompt(case, used_products)})

    body: Dict[str, Any] = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": {
            "n": 1,
            "watermark": bool(qwen_watermark),
            "prompt_extend": bool(qwen_prompt_extend),
        },
    }
    if qwen_size:
        body["parameters"]["size"] = qwen_size

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    started = time.perf_counter()
    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout_s)
        elapsed = time.perf_counter() - started
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw_text": resp.text[:2000]}
        ok = resp.status_code == 200 and _extract_qwen_output_url(payload) is not None
        req_result = RequestResult(
            ok=ok,
            status_code=int(resp.status_code),
            elapsed_s=float(elapsed),
            output_url=_extract_qwen_output_url(payload) if ok else None,
            generation_s=None,
            error=None if ok else f"qwen request failed (status={resp.status_code})",
            raw=_compact_json(payload),
        )
        extra = {
            "qwen_products_used": len(used_products),
            "qwen_products_dropped": dropped,
            "qwen_prompt": body["input"]["messages"][0]["content"][-1]["text"],
        }
        return req_result, extra
    except Exception as exc:
        elapsed = time.perf_counter() - started
        req_result = RequestResult(
            ok=False,
            status_code=0,
            elapsed_s=float(elapsed),
            output_url=None,
            generation_s=None,
            error=f"{type(exc).__name__}: {exc}",
            raw={},
        )
        extra = {
            "qwen_products_used": len(used_products),
            "qwen_products_dropped": dropped,
            "qwen_prompt": body["input"]["messages"][0]["content"][-1]["text"],
        }
        return req_result, extra


def _download_bytes(url: str, timeout_s: int) -> bytes:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.content


def _extract_analyze_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(body.get("data"), dict):
        return body["data"]
    return body


def _extract_first_item_type(analyze_payload: Dict[str, Any]) -> Optional[str]:
    item = analyze_payload.get("item")
    if isinstance(item, dict):
        kind = _normalize_type(item.get("type"))
        if kind:
            return kind
    items = analyze_payload.get("items")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict):
            return _normalize_type(first.get("type"))
    return None


def _analyze_type_hit_rate(
    *,
    base_url: str,
    endpoint: str,
    timeout_s: int,
    image_url: str,
    requested_types: Sequence[str],
) -> Dict[str, Any]:
    started = time.perf_counter()
    result: Dict[str, Any] = {
        "ok": False,
        "elapsed_s": None,
        "requested_types": list(requested_types),
        "hits": 0,
        "total": len(requested_types),
        "hit_rate": None,
        "checks": [],
        "error": None,
    }
    if not requested_types:
        result["ok"] = True
        result["elapsed_s"] = 0.0
        result["hits"] = 0
        result["total"] = 0
        result["hit_rate"] = None
        return result

    try:
        image_bytes = _download_bytes(image_url, timeout_s=timeout_s)
        analyze_url = _safe_join_url(base_url, endpoint)
        hits = 0
        checks = []
        for expected in requested_types:
            files = {"file": ("generated.png", image_bytes, "image/png")}
            data = {"type": expected}
            resp = requests.post(analyze_url, files=files, data=data, timeout=timeout_s)
            try:
                payload = resp.json()
            except Exception:
                payload = {"raw_text": resp.text[:1000]}
            parsed = _extract_analyze_payload(payload if isinstance(payload, dict) else {})
            found = _extract_first_item_type(parsed) if isinstance(parsed, dict) else None
            matched = bool(found == expected and resp.status_code == 200)
            if matched:
                hits += 1
            checks.append(
                {
                    "expected_type": expected,
                    "status_code": int(resp.status_code),
                    "found_type": found,
                    "matched": matched,
                }
            )

        elapsed = time.perf_counter() - started
        total = len(requested_types)
        result.update(
            {
                "ok": True,
                "elapsed_s": float(elapsed),
                "hits": int(hits),
                "total": int(total),
                "hit_rate": float(hits / total) if total else None,
                "checks": checks,
            }
        )
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - started
        result.update(
            {
                "ok": False,
                "elapsed_s": float(elapsed),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result


def _summarize_engine_runs(runs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    runs_list = list(runs)
    elapsed = [float(r["elapsed_s"]) for r in runs_list if r.get("ok")]
    generation = [
        float(r["generation_s"])
        for r in runs_list
        if r.get("ok") and isinstance(r.get("generation_s"), (int, float))
    ]
    hit_rates = [
        float(r["analyze"]["hit_rate"])
        for r in runs_list
        if r.get("ok")
        and isinstance(r.get("analyze"), dict)
        and isinstance(r["analyze"].get("hit_rate"), (int, float))
    ]
    total = len(runs_list)
    return {
        "success_count": len(elapsed),
        "total_count": total,
        "success_rate": float(len(elapsed) / total) if total else 0.0,
        "elapsed_s_mean": float(statistics.fmean(elapsed)) if elapsed else None,
        "elapsed_s_p50": _percentile(elapsed, 50.0),
        "elapsed_s_p95": _percentile(elapsed, 95.0),
        "generation_s_mean": float(statistics.fmean(generation)) if generation else None,
        "type_hit_rate_mean": float(statistics.fmean(hit_rates)) if hit_rates else None,
    }


def _flatten_engine_runs(all_runs: Sequence[Dict[str, Any]], engine_key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in all_runs:
        engine = row.get(engine_key) or {}
        if isinstance(engine, dict):
            out.append(engine)
    return out


def _build_markdown_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# FLUX vs Qwen Image Edit Benchmark ({result['generated_at_utc']})")
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Engine | Success | Mean (s) | P50 (s) | P95 (s) | Mean Type Hit Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for engine in ("flux", "qwen"):
        s = result["summary"].get(engine, {})
        lines.append(
            "| {engine} | {ok}/{total} ({rate:.1%}) | {mean} | {p50} | {p95} | {hit} |".format(
                engine=engine,
                ok=s.get("success_count", 0),
                total=s.get("total_count", 0),
                rate=float(s.get("success_rate", 0.0)),
                mean=f"{s['elapsed_s_mean']:.3f}" if isinstance(s.get("elapsed_s_mean"), (int, float)) else "-",
                p50=f"{s['elapsed_s_p50']:.3f}" if isinstance(s.get("elapsed_s_p50"), (int, float)) else "-",
                p95=f"{s['elapsed_s_p95']:.3f}" if isinstance(s.get("elapsed_s_p95"), (int, float)) else "-",
                hit=f"{s['type_hit_rate_mean']:.3f}" if isinstance(s.get("type_hit_rate_mean"), (int, float)) else "-",
            )
        )

    head = result["summary"].get("head_to_head", {})
    lines.append("")
    lines.append("## Head-to-Head")
    lines.append("")
    lines.append(f"- FLUX faster runs: {head.get('flux_faster_runs', 0)}")
    lines.append(f"- Qwen faster runs: {head.get('qwen_faster_runs', 0)}")
    lines.append(f"- Tied runs: {head.get('tied_runs', 0)}")

    lines.append("")
    lines.append("## Per Case")
    lines.append("")
    for case in result.get("cases", []):
        lines.append(f"### {case.get('case_id')}")
        lines.append("")
        lines.append(f"- Requested types: {', '.join(case.get('requested_types') or []) or '(none)'}")
        case_summary = case.get("summary", {})
        for engine in ("flux", "qwen"):
            s = case_summary.get(engine, {})
            lines.append(
                "- {engine}: success {ok}/{total}, mean {mean}s, p95 {p95}s, type hit mean {hit}".format(
                    engine=engine,
                    ok=s.get("success_count", 0),
                    total=s.get("total_count", 0),
                    mean=f"{s['elapsed_s_mean']:.3f}" if isinstance(s.get("elapsed_s_mean"), (int, float)) else "-",
                    p95=f"{s['elapsed_s_p95']:.3f}" if isinstance(s.get("elapsed_s_p95"), (int, float)) else "-",
                    hit=f"{s['type_hit_rate_mean']:.3f}" if isinstance(s.get("type_hit_rate_mean"), (int, float)) else "-",
                )
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _validate_case(case: Dict[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(case, dict):
        raise BenchmarkError(f"cases[{index}] must be an object")
    case_id = str(case.get("case_id") or f"case_{index+1}").strip()
    user_image = str(case.get("user_image") or "").strip()
    if not user_image.startswith(("http://", "https://")):
        raise BenchmarkError(f"{case_id}: user_image must be an http(s) URL")
    products = case.get("products")
    if not isinstance(products, list) or not products:
        raise BenchmarkError(f"{case_id}: products must be a non-empty list")

    normalized_products: List[Dict[str, Any]] = []
    for p_idx, product in enumerate(products):
        if not isinstance(product, dict):
            raise BenchmarkError(f"{case_id}: products[{p_idx}] must be an object")
        image = str(product.get("image") or "").strip()
        if not image.startswith(("http://", "https://")):
            raise BenchmarkError(f"{case_id}: products[{p_idx}].image must be an http(s) URL")
        prompt = " ".join(str(product.get("promptDescription") or "").split()).strip()
        if not prompt:
            raise BenchmarkError(f"{case_id}: products[{p_idx}].promptDescription is required")
        target = _normalize_type(product.get("targetType"))
        normalized_products.append(
            {
                "image": image,
                "promptDescription": prompt,
                "targetType": target,
            }
        )

    return {
        "case_id": case_id,
        "user_image": user_image,
        "user_prompt_description": " ".join(str(case.get("user_prompt_description") or "").split()).strip() or None,
        "source_worn_types": [t for t in (case.get("source_worn_types") or []) if _normalize_type(t)],
        "products": normalized_products,
        "steps": int(case.get("steps")) if case.get("steps") is not None else None,
        "seed": int(case.get("seed")) if case.get("seed") is not None else None,
        "guidanceScale": float(case.get("guidanceScale")) if case.get("guidanceScale") is not None else None,
        "loraScale": float(case.get("loraScale")) if case.get("loraScale") is not None else None,
    }


def _build_flux_payload(case: Dict[str, Any], *, mode: str, default_steps: int, default_seed: Optional[int]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "products": case["products"],
        "user_image": {
            "tryonImage": case["user_image"],
        },
        "mode": mode,
        "steps": int(case["steps"] if case["steps"] is not None else default_steps),
    }
    if case.get("user_prompt_description"):
        payload["user_image"]["promptDescription"] = case["user_prompt_description"]
    if case.get("source_worn_types"):
        payload["user_image"]["wornTypes"] = case["source_worn_types"]
    seed = case.get("seed") if case.get("seed") is not None else default_seed
    if seed is not None:
        payload["seed"] = int(seed)
    if case.get("guidanceScale") is not None:
        payload["guidanceScale"] = float(case["guidanceScale"])
    if case.get("loraScale") is not None:
        payload["loraScale"] = float(case["loraScale"])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark /v1/flux2/tryon against Qwen image editing on identical cases."
    )
    parser.add_argument("--cases-file", required=True, help="JSON file with benchmark cases")
    parser.add_argument("--output-dir", default="debug_comparisons/flux_vs_qwen_image_edit", help="Output directory")

    parser.add_argument("--flux-base-url", default=os.getenv("FLUX_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--flux-endpoint", default=DEFAULT_FLUX_ENDPOINT)
    parser.add_argument("--flux-mode", default="tryon-lora", choices=["tryon-lora", "consistency-lora"])
    parser.add_argument("--flux-default-steps", type=int, default=20)
    parser.add_argument("--flux-default-seed", type=int, default=None)

    parser.add_argument("--qwen-endpoint", default=os.getenv("QWEN_IMAGE_EDIT_ENDPOINT", DEFAULT_QWEN_ENDPOINT))
    parser.add_argument("--qwen-model", default=os.getenv("QWEN_IMAGE_EDIT_MODEL", DEFAULT_QWEN_MODEL))
    parser.add_argument("--dashscope-api-key", default=os.getenv("DASHSCOPE_API_KEY"))
    parser.add_argument("--qwen-max-input-images", type=int, default=3, help="Total input images for Qwen request (includes user image)")
    parser.add_argument("--qwen-size", default=None, help="Optional output size, e.g. 768*1024")
    parser.add_argument("--qwen-prompt-extend", action="store_true", default=True)
    parser.add_argument("--qwen-no-prompt-extend", dest="qwen_prompt_extend", action="store_false")
    parser.add_argument("--qwen-watermark", action="store_true", default=False)

    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds per request")
    parser.add_argument("--analyze-endpoint", default=DEFAULT_ANALYZE_ENDPOINT)
    parser.add_argument("--skip-analyze", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.dashscope_api_key:
        raise BenchmarkError("Missing DashScope key. Provide --dashscope-api-key or DASHSCOPE_API_KEY.")
    if args.repeats < 1:
        raise BenchmarkError("--repeats must be >= 1")
    if args.qwen_max_input_images < 2:
        raise BenchmarkError("--qwen-max-input-images must be >= 2 (user + at least one garment)")

    cases_path = Path(args.cases_file).expanduser().resolve()
    if not cases_path.exists():
        raise BenchmarkError(f"Cases file not found: {cases_path}")

    raw_cases = _read_json(cases_path)
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BenchmarkError("Cases file must be a non-empty JSON array")

    cases = [_validate_case(case, idx) for idx, case in enumerate(raw_cases)]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_case_results: List[Dict[str, Any]] = []
    all_pair_rows: List[Dict[str, Any]] = []

    for case in cases:
        requested_types = _extract_requested_types(case["products"])
        case_runs: List[Dict[str, Any]] = []
        flux_runs_flat: List[Dict[str, Any]] = []
        qwen_runs_flat: List[Dict[str, Any]] = []

        flux_payload = _build_flux_payload(
            case,
            mode=args.flux_mode,
            default_steps=int(args.flux_default_steps),
            default_seed=args.flux_default_seed,
        )

        for repeat_idx in range(1, int(args.repeats) + 1):
            flux_result = _request_flux_tryon(
                base_url=args.flux_base_url,
                endpoint=args.flux_endpoint,
                timeout_s=int(args.timeout),
                payload=flux_payload,
            )

            flux_analyze = None
            if flux_result.ok and flux_result.output_url and not args.skip_analyze:
                flux_analyze = _analyze_type_hit_rate(
                    base_url=args.flux_base_url,
                    endpoint=args.analyze_endpoint,
                    timeout_s=int(args.timeout),
                    image_url=flux_result.output_url,
                    requested_types=requested_types,
                )

            qwen_result, qwen_extra = _request_qwen_edit(
                endpoint=args.qwen_endpoint,
                api_key=args.dashscope_api_key,
                model=args.qwen_model,
                timeout_s=int(args.timeout),
                case=case,
                qwen_max_inputs=int(args.qwen_max_input_images),
                qwen_size=args.qwen_size,
                qwen_prompt_extend=bool(args.qwen_prompt_extend),
                qwen_watermark=bool(args.qwen_watermark),
            )

            qwen_analyze = None
            if qwen_result.ok and qwen_result.output_url and not args.skip_analyze:
                qwen_analyze = _analyze_type_hit_rate(
                    base_url=args.flux_base_url,
                    endpoint=args.analyze_endpoint,
                    timeout_s=int(args.timeout),
                    image_url=qwen_result.output_url,
                    requested_types=requested_types,
                )

            flux_row = flux_result.to_dict()
            flux_row["analyze"] = flux_analyze
            qwen_row = qwen_result.to_dict()
            qwen_row["analyze"] = qwen_analyze
            qwen_row["qwen_meta"] = qwen_extra

            comparison = {
                "latency_diff_s": (
                    float(flux_row["elapsed_s"]) - float(qwen_row["elapsed_s"])
                    if flux_result.ok and qwen_result.ok
                    else None
                ),
                "faster": None,
            }
            if flux_result.ok and qwen_result.ok:
                if abs(flux_row["elapsed_s"] - qwen_row["elapsed_s"]) <= 0.05:
                    comparison["faster"] = "tie"
                elif flux_row["elapsed_s"] < qwen_row["elapsed_s"]:
                    comparison["faster"] = "flux"
                else:
                    comparison["faster"] = "qwen"

            row = {
                "repeat": repeat_idx,
                "flux": flux_row,
                "qwen": qwen_row,
                "comparison": comparison,
            }
            case_runs.append(row)
            all_pair_rows.append(row)
            flux_runs_flat.append(flux_row)
            qwen_runs_flat.append(qwen_row)

        case_result = {
            "case_id": case["case_id"],
            "requested_types": requested_types,
            "input": case,
            "summary": {
                "flux": _summarize_engine_runs(flux_runs_flat),
                "qwen": _summarize_engine_runs(qwen_runs_flat),
            },
            "runs": case_runs,
        }
        all_case_results.append(case_result)

    flux_all = _flatten_engine_runs(all_pair_rows, "flux")
    qwen_all = _flatten_engine_runs(all_pair_rows, "qwen")

    flux_faster = 0
    qwen_faster = 0
    ties = 0
    for row in all_pair_rows:
        faster = ((row.get("comparison") or {}).get("faster"))
        if faster == "flux":
            flux_faster += 1
        elif faster == "qwen":
            qwen_faster += 1
        elif faster == "tie":
            ties += 1

    result = {
        "generated_at_utc": _now_iso(),
        "config": {
            "cases_file": str(cases_path),
            "flux_base_url": args.flux_base_url,
            "flux_endpoint": args.flux_endpoint,
            "flux_mode": args.flux_mode,
            "qwen_endpoint": args.qwen_endpoint,
            "qwen_model": args.qwen_model,
            "qwen_max_input_images": int(args.qwen_max_input_images),
            "qwen_size": args.qwen_size,
            "qwen_prompt_extend": bool(args.qwen_prompt_extend),
            "qwen_watermark": bool(args.qwen_watermark),
            "repeats": int(args.repeats),
            "timeout": int(args.timeout),
            "skip_analyze": bool(args.skip_analyze),
            "analyze_endpoint": args.analyze_endpoint,
        },
        "summary": {
            "flux": _summarize_engine_runs(flux_all),
            "qwen": _summarize_engine_runs(qwen_all),
            "head_to_head": {
                "flux_faster_runs": flux_faster,
                "qwen_faster_runs": qwen_faster,
                "tied_runs": ties,
            },
        },
        "cases": all_case_results,
    }

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"flux_vs_qwen_benchmark_{stamp}.json"
    md_path = output_dir / f"flux_vs_qwen_benchmark_{stamp}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(_build_markdown_report(result))

    print(f"[ok] wrote JSON report: {json_path}")
    print(f"[ok] wrote Markdown report: {md_path}")
    print(
        "[summary] flux success={}/{} | qwen success={}/{}".format(
            result["summary"]["flux"]["success_count"],
            result["summary"]["flux"]["total_count"],
            result["summary"]["qwen"]["success_count"],
            result["summary"]["qwen"]["total_count"],
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"[error] {exc}")
        raise SystemExit(2)

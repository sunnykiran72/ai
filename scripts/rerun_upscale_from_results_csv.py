#!/usr/bin/env python3
"""
Rerun SeedVR2 upscale only for existing bulk try-on result folders.

This script reads one or more `results.csv` files produced by
`run_bulk_tryon_upscale_from_prepare.py`, uses each row's `tryon_output_url`
as the new upscale input, and rewrites:

- results.csv
- results.jsonl
- summary.json
- report.html

in place, so the report points at the refreshed upscale outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import requests


def _load_bulk_helpers():
    script_path = Path(__file__).resolve().with_name("run_bulk_tryon_upscale_from_prepare.py")
    spec = importlib.util.spec_from_file_location("bulk_tryon_runner", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bulk = _load_bulk_helpers()
CaseResult = bulk.CaseResult
_safe_json = bulk._safe_json
_write_csv = bulk._write_csv
_build_summary = bulk._build_summary
_generate_html = bulk._generate_html
_build_upscale_filename = bulk._build_upscale_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun upscale only for existing bulk try-on result folders")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--results-csv", nargs="+", required=True, help="One or more bulk-run results.csv paths")
    parser.add_argument("--upscale-target-long-edge", type=int, default=2048)
    parser.add_argument("--output-filename-prefix", default="seedvr2_refresh")
    parser.add_argument("--timeout-seconds", type=float, default=920.0)
    parser.add_argument("--only-file", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    return parser.parse_args()


def _load_rows(results_csv: Path) -> List[Any]:
    rows: List[Any] = []
    with results_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for payload in reader:
            rows.append(
                CaseResult(
                    index=int(payload.get("index") or 0),
                    user_file=str(payload.get("user_file") or ""),
                    prepared_input_url=str(payload.get("prepared_input_url") or ""),
                    user_prompt=str(payload.get("user_prompt") or ""),
                    tryon_output_url=str(payload.get("tryon_output_url") or ""),
                    upscale_output_url=str(payload.get("upscale_output_url") or ""),
                    prompt_used=str(payload.get("prompt_used") or ""),
                    status=str(payload.get("status") or ""),
                    error=str(payload.get("error") or ""),
                    tryon_latency_s=float(payload.get("tryon_latency_s") or 0.0),
                    upscale_latency_s=float(payload.get("upscale_latency_s") or 0.0),
                    total_latency_s=float(payload.get("total_latency_s") or 0.0),
                )
            )
    rows.sort(key=lambda x: x.index)
    return rows


def _load_summary(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_args_from_summary(base_url: str, summary: Dict[str, Any]) -> SimpleNamespace:
    garment_mode = str(summary.get("garment_mode") or "dress")
    return SimpleNamespace(
        title=str(summary.get("title") or "Glamify Bulk Try-on + Upscale"),
        base_url=base_url,
        seed=int(summary.get("seed") or 0),
        steps=int(summary.get("steps") or 0),
        guidance_scale=float(summary.get("guidance_scale") or 0.0),
        lora_scale=float(summary.get("lora_scale") or 1.0),
        output_max_edge=int(summary.get("output_max_edge") or 1024),
        upscale_target_long_edge=int(summary.get("upscale_target_long_edge") or 2048),
        garment_mode=garment_mode,
        top_image_url=str(summary.get("top_image_url") or ""),
        bottom_image_url=str(summary.get("bottom_image_url") or ""),
        dress_image_url=str(summary.get("dress_image_url") or ""),
        top_prompt=str(summary.get("top_prompt") or ""),
        bottom_prompt=str(summary.get("bottom_prompt") or ""),
        dress_prompt=str(summary.get("dress_prompt") or ""),
    )


def _rewrite_jsonl(path: Path, rows: List[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")


def _should_process(row: Any, args: argparse.Namespace) -> bool:
    if row.status != "success":
        return False
    if not str(row.tryon_output_url or "").strip():
        return False
    only_file = str(args.only_file or "").strip()
    if only_file and str(row.user_file).strip() != only_file:
        return False
    if args.start_index > 0 and int(row.index) < int(args.start_index):
        return False
    if args.end_index > 0 and int(row.index) > int(args.end_index):
        return False
    return True


def _rerun_folder(results_csv_path: Path, args: argparse.Namespace) -> int:
    run_dir = results_csv_path.resolve().parent
    jsonl_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    html_path = run_dir / "report.html"

    rows = _load_rows(results_csv_path)
    summary = _load_summary(summary_path)
    helper_args = _build_args_from_summary(args.base_url.rstrip("/"), summary)
    session = requests.Session()
    updated = 0

    for row in rows:
        if not _should_process(row, args):
            continue

        request_payload = {
            "image_url": str(row.tryon_output_url),
            "target_long_edge": int(args.upscale_target_long_edge),
            "batch_size": 1,
            "use_persistent": True,
            "gpu_resident": True,
            "cache_models": True,
            "timeout_seconds": 900,
            "upload_to_storage": True,
            "output_filename": _build_upscale_filename(args.output_filename_prefix, int(row.index)),
        }

        started = time.time()
        response = session.post(
            f"{args.base_url.rstrip('/')}/v1/user-image/upscale",
            json=request_payload,
            timeout=float(args.timeout_seconds),
        )
        payload = _safe_json(response)
        if not response.ok:
            raise RuntimeError(
                f"Upscale rerun failed for index={row.index} file={row.user_file}: "
                f"status={response.status_code} payload={payload}"
            )

        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        new_url = str(output.get("url") or output.get("storage_url") or output.get("url_public") or "").strip()
        if not new_url:
            raise RuntimeError(f"Upscale rerun returned no output URL for index={row.index} file={row.user_file}")

        timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
        new_upscale_latency = float(timings.get("total_seconds") or timings.get("elapsed_seconds") or 0.0)
        if new_upscale_latency <= 0.0:
            new_upscale_latency = max(0.0, time.time() - started)

        old_url = row.upscale_output_url
        row.upscale_output_url = new_url
        row.upscale_latency_s = new_upscale_latency
        row.total_latency_s = float(row.tryon_latency_s) + float(new_upscale_latency)
        row.error = ""
        row.status = "success"
        updated += 1

        print(
            f"[{row.index:03d}] {row.user_file} -> refreshed "
            f"(old={old_url or 'n/a'} new={new_url} upscale={row.upscale_latency_s:.2f}s)"
        )
        sys.stdout.flush()

    _rewrite_jsonl(jsonl_path, rows)
    _write_csv(results_csv_path, rows, helper_args)
    new_summary = _build_summary(rows, helper_args)
    summary_path.write_text(json.dumps(new_summary, indent=2), encoding="utf-8")
    _generate_html(html_path, rows, new_summary)
    print(f"updated={updated} run_dir={run_dir}")
    return updated


def main() -> int:
    args = parse_args()
    total_updated = 0
    for raw_path in args.results_csv:
        results_csv_path = Path(raw_path)
        if not results_csv_path.exists():
            raise SystemExit(f"results.csv not found: {results_csv_path}")
        total_updated += _rerun_folder(results_csv_path, args)
    print(f"total_updated={total_updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Bulk runner for /dev/analyze-api-lab/run.

Runs analyze-lab for a folder of images and writes:
- per-case raw response JSON
- per-case flow JSON
- extracted output preview image
- preprocessed input preview image
- one aggregated HTML report
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".heic", ".heif"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "case"


def iter_images(input_dir: Path, recursive: bool = False) -> Iterable[Path]:
    if recursive:
        candidates = sorted(p for p in input_dir.rglob("*") if p.is_file())
    else:
        candidates = sorted(p for p in input_dir.iterdir() if p.is_file())
    for path in candidates:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def decode_data_url_to_file(data_url: str, destination: Path) -> Optional[Path]:
    text = str(data_url or "").strip()
    if not text or "," not in text or ";base64" not in text:
        return None
    _, b64 = text.split(",", 1)
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        return None
    destination.write_bytes(raw)
    return destination


def maybe_rel(path: Optional[Path], root: Path) -> str:
    if not path or not path.exists():
        return ""
    return path.relative_to(root).as_posix()


def run_case(
    *,
    base_url: str,
    image_path: Path,
    garment_type: str,
    timeout: int,
    headers: Dict[str, str],
    case_dir: Path,
) -> Dict[str, Any]:
    ensure_dir(case_dir)
    shutil.copy2(image_path, case_dir / image_path.name)

    mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    started = time.perf_counter()
    with image_path.open("rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/dev/analyze-api-lab/run",
            files={"file": (image_path.name, handle, mime)},
            data={"type": garment_type, "debug": "true"},
            headers=headers,
            timeout=timeout,
        )
    elapsed = round(time.perf_counter() - started, 4)

    payload: Dict[str, Any]
    parse_error = ""
    try:
        payload = response.json()
    except Exception as exc:
        parse_error = str(exc)
        payload = {"status": "error", "message": "Non-JSON response", "raw_text": response.text}

    raw_path = case_dir / "raw_response.json"
    write_json(
        raw_path,
        {
            "http_status": response.status_code,
            "headers": dict(response.headers),
            "elapsed_client_s": elapsed,
            "parse_error": parse_error,
            "payload": payload,
        },
    )

    flow = payload.get("flow") if isinstance(payload, dict) else {}
    if not isinstance(flow, dict):
        flow = {}
    write_json(case_dir / "flow.json", flow)

    analyze_response = payload.get("analyze_response") if isinstance(payload, dict) else {}
    if isinstance(analyze_response, dict):
        write_json(case_dir / "analyze_response.json", analyze_response)

    output_image = decode_data_url_to_file(
        str(payload.get("extracted_image_data_url") or ""),
        case_dir / "output_preview.png",
    )
    preprocessed_image = decode_data_url_to_file(
        str(payload.get("preprocessed_image_data_url") or ""),
        case_dir / "preprocessed_input.png",
    )

    prompt = flow.get("prompt") if isinstance(flow.get("prompt"), dict) else {}
    resolution = flow.get("resolution") if isinstance(flow.get("resolution"), dict) else {}
    flags = flow.get("flags") if isinstance(flow.get("flags"), dict) else {}
    timings = flow.get("timings") if isinstance(flow.get("timings"), dict) else {}
    selection = flow.get("selection") if isinstance(flow.get("selection"), dict) else {}

    summary = {
        "case_id": case_dir.name,
        "source_image": str(image_path),
        "http_status": int(response.status_code),
        "status": str(payload.get("status") or ""),
        "message": str(payload.get("message") or ""),
        "elapsed_client_s": elapsed,
        "selected_type": str(selection.get("selected_item_type") or selection.get("selected_type") or ""),
        "normalized_category_type": str(flags.get("normalized_category_type") or ""),
        "prompt_source": str(prompt.get("prompt_source") or ""),
        "prompt_description": str(prompt.get("prompt_description") or ""),
        "prompt_sent": str(prompt.get("prompt_sent") or ""),
        "input_original_size": resolution.get("input_original_size"),
        "input_preprocessed_size": resolution.get("input_preprocessed_size"),
        "requested_output_size_aligned": resolution.get("requested_output_size_aligned"),
        "qwen_output_size_raw": resolution.get("qwen_output_size_raw"),
        "final_output_size": resolution.get("final_output_size"),
        "minicpm_prompt_enabled": flags.get("minicpm_prompt_enabled"),
        "minicpm_json_valid": flags.get("minicpm_json_valid"),
        "minicpm_json_fallback_used": flags.get("minicpm_json_fallback_used"),
        "timings": timings,
        "paths": {
            "raw_response_json": str(raw_path),
            "flow_json": str(case_dir / "flow.json"),
            "analyze_response_json": str(case_dir / "analyze_response.json"),
            "source_image_copy": str(case_dir / image_path.name),
            "preprocessed_input_image": str(preprocessed_image) if preprocessed_image else "",
            "output_preview_image": str(output_image) if output_image else "",
        },
    }
    write_json(case_dir / "summary.json", summary)
    return summary


def build_html_report(rows: list[Dict[str, Any]], output_dir: Path, title: str) -> str:
    cards = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        source_image = Path(str(row.get("paths", {}).get("source_image_copy") or ""))
        preprocessed = Path(str(row.get("paths", {}).get("preprocessed_input_image") or ""))
        output_image = Path(str(row.get("paths", {}).get("output_preview_image") or ""))
        raw_json = Path(str(row.get("paths", {}).get("raw_response_json") or ""))
        flow_json = Path(str(row.get("paths", {}).get("flow_json") or ""))
        summary_json = output_dir / case_id / "summary.json"

        def img_block(label: str, p: Path) -> str:
            if not p.exists():
                return ""
            return (
                f'<div class="image-block"><div class="label">{html.escape(label)}</div>'
                f'<img src="{html.escape(maybe_rel(p, output_dir))}" alt="{html.escape(label)}"></div>'
            )

        def json_block(label: str, p: Path) -> str:
            if not p.exists():
                return ""
            content = p.read_text(encoding="utf-8", errors="replace")
            return (
                f'<details class="json-block"><summary>{html.escape(label)}</summary>'
                f"<pre>{html.escape(content)}</pre></details>"
            )

        cards.append(
            f"""
            <section class="card">
              <div class="card-header">
                <h2>{html.escape(case_id)}</h2>
                <div class="meta">
                  <span>http: {html.escape(str(row.get("http_status") or ""))}</span>
                  <span>selected: {html.escape(str(row.get("selected_type") or ""))}</span>
                  <span>category: {html.escape(str(row.get("normalized_category_type") or ""))}</span>
                  <span>source: {html.escape(str(row.get("prompt_source") or ""))}</span>
                  <span>client_s: {html.escape(str(row.get("elapsed_client_s") or ""))}</span>
                </div>
              </div>
              <div class="grid images">
                {img_block("Source", source_image)}
                {img_block("Preprocessed Input", preprocessed)}
                {img_block("Output", output_image)}
              </div>
              <div class="grid metrics">
                <div class="metric"><strong>input_original_size</strong><br>{html.escape(str(row.get("input_original_size") or ""))}</div>
                <div class="metric"><strong>input_preprocessed_size</strong><br>{html.escape(str(row.get("input_preprocessed_size") or ""))}</div>
                <div class="metric"><strong>requested_output_size_aligned</strong><br>{html.escape(str(row.get("requested_output_size_aligned") or ""))}</div>
                <div class="metric"><strong>qwen_output_size_raw</strong><br>{html.escape(str(row.get("qwen_output_size_raw") or ""))}</div>
                <div class="metric"><strong>final_output_size</strong><br>{html.escape(str(row.get("final_output_size") or ""))}</div>
                <div class="metric"><strong>minicpm_json_valid</strong><br>{html.escape(str(row.get("minicpm_json_valid") or ""))}</div>
                <div class="metric"><strong>minicpm_json_fallback_used</strong><br>{html.escape(str(row.get("minicpm_json_fallback_used") or ""))}</div>
              </div>
              <div class="text-block">
                <div class="label">Prompt Description</div>
                <pre>{html.escape(str(row.get("prompt_description") or ""))}</pre>
              </div>
              <div class="text-block">
                <div class="label">Prompt Sent</div>
                <pre>{html.escape(str(row.get("prompt_sent") or ""))}</pre>
              </div>
              {json_block("Summary JSON", summary_json)}
              {json_block("Flow JSON", flow_json)}
              {json_block("Raw Response JSON", raw_json)}
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #dbe3ef;
      --accent: #0b3b7a;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); color: var(--ink); }}
    .page {{ width: min(1700px, calc(100vw - 32px)); margin: 16px auto 28px; }}
    .hero {{ background: linear-gradient(135deg, #fff, #f4f8ff); border: 1px solid var(--line); border-radius: 14px; padding: 16px; margin-bottom: 14px; }}
    .hero h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .hero p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px; margin-bottom: 12px; }}
    .card-header {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; align-items: baseline; }}
    .card-header h2 {{ margin: 0; font-size: 18px; }}
    .meta {{ display: flex; gap: 8px; flex-wrap: wrap; color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .grid {{ display: grid; gap: 10px; }}
    .images {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); margin-top: 10px; }}
    .metrics {{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin-top: 10px; }}
    .image-block, .text-block, .metric, .json-block {{
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fbfdff;
      padding: 10px;
      margin-top: 10px;
    }}
    .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--accent); margin-bottom: 8px; font-weight: 700; }}
    img {{ width: 100%; height: auto; border-radius: 8px; background: #fff; border: 1px solid #edf2f8; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; color: #1e293b; }}
    details summary {{ cursor: pointer; font-size: 12px; font-weight: 700; color: #1e3a8a; }}
    @media (max-width: 720px) {{ .page {{ width: min(100vw - 12px, 1700px); }} }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <h1>{html.escape(title)}</h1>
      <p>Bulk /dev/analyze-api-lab/run report with per-image previews, flow metrics, timings, flags, and raw JSON.</p>
    </header>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk runner for /dev/analyze-api-lab/run.")
    parser.add_argument("--base-url", required=True, help="Base URL, e.g. https://...proxy.runpod.net")
    parser.add_argument("--input-dir", required=True, help="Input image directory")
    parser.add_argument("--garment-type", default="top", choices=["top", "bottom", "dress", "outer"])
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to tmp/analyze-lab-bulk/<timestamp>")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout seconds")
    parser.add_argument("--auth-token", default="", help="Bearer token without prefix")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively")
    parser.add_argument("--title", default="Analyze API Lab Bulk Report", help="Report title")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path("tmp") / "analyze-lab-bulk" / ts).resolve()
    )
    ensure_dir(output_dir)

    headers: Dict[str, str] = {}
    if args.auth_token:
        headers["Authorization"] = f"Bearer {args.auth_token}"

    images = list(iter_images(input_dir, recursive=bool(args.recursive)))
    if args.limit > 0:
        images = images[: args.limit]

    rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []

    for index, image_path in enumerate(images, start=1):
        case_id = f"{index:04d}_{slugify(image_path.stem)}"
        case_dir = ensure_dir(output_dir / case_id)
        try:
            row = run_case(
                base_url=args.base_url,
                image_path=image_path,
                garment_type=args.garment_type,
                timeout=int(args.timeout),
                headers=headers,
                case_dir=case_dir,
            )
            rows.append(row)
            print(f"[ok] {case_id} :: {image_path.name}")
        except Exception as exc:
            failure = {
                "case_id": case_id,
                "image": str(image_path),
                "error": str(exc),
            }
            failures.append(failure)
            write_json(case_dir / "error.json", failure)
            print(f"[fail] {case_id} :: {image_path.name} :: {exc}")

    summary = {
        "base_url": args.base_url.rstrip("/"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "garment_type": args.garment_type,
        "total_images": len(images),
        "success_count": len(rows),
        "failure_count": len(failures),
        "rows": rows,
        "failures": failures,
        "generated_at": ts,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.html").write_text(
        build_html_report(rows, output_dir=output_dir, title=args.title),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"report_html={output_dir / 'report.html'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())


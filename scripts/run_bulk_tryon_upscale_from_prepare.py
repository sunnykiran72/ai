#!/usr/bin/env python3
"""
Bulk try-on + SeedVR2 upscale runner using cached /user-image/prepare output.

Flow per case:
1) Read prepared user image URL + user prompt from results.csv
2) POST to /v1/flux2/tryon with top+bottom product references
   and a custom prompt override
4) POST resulting try-on output URL to /v1/user-image/upscale
5) Write resumable JSONL/CSV/summary and a 3-panel HTML report
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests


TRYON_PROMPT_TEMPLATE = (
    "TRYON {user_prompt}. Replace the entire outfit with a upper-garment : a strapless tube top. "
    "It features a vertical row of five buttons down the center front. The top has a fitted silhouette with vertical seams and a straight hemline at the bottom edge; "
    "lower-garment : Low-rise denim shorts feature a front waistband with a single visible button closure. "
    "One side displays an American flag print, while the another side is solid denim as shown in the reference images. "
    "Preserve exact person identity including face, hair color, eye direction, pose, footwear, and accessories. "
    "Preserve body geometry and silhouette: torso contour, waist-to-hip relationship, hip contour, glute projection, thigh volume, and leg length. "
    "Render both garments with accurate construction, fit, seam placement, drape, hem length, texture, pattern placement. "
    "Strictly remove any worn top, bottom, dress and outer garments. "
    "Preserve the exact garment color fidelity, garment structure, deisgn pattern and positions, sleeve construction, edge finish and fabric pattern layout as shown in references. "
    "Each garment must preserve only the design and construction of its own reference image. "
    "Maintain strict garment separation with no cross-garment inference, no feature borrowing, and no structural or visual mixing between upper and lower garments. "
    "Keep camera distance, background, and lighting consistent. "
    "The final image is a full body shot."
)


@dataclass
class CaseResult:
    index: int
    user_file: str
    prepared_input_url: str
    user_prompt: str
    tryon_output_url: str
    upscale_output_url: str
    prompt_used: str
    status: str
    error: str
    tryon_latency_s: float
    upscale_latency_s: float
    total_latency_s: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "user_file": self.user_file,
            "prepared_input_url": self.prepared_input_url,
            "user_prompt": self.user_prompt,
            "tryon_output_url": self.tryon_output_url,
            "upscale_output_url": self.upscale_output_url,
            "prompt_used": self.prompt_used,
            "status": self.status,
            "error": self.error,
            "tryon_latency_s": self.tryon_latency_s,
            "upscale_latency_s": self.upscale_latency_s,
            "total_latency_s": self.total_latency_s,
        }


def _normalize_worn_types(raw: object) -> List[str]:
    allowed = {"top", "bottom", "dress", "outer"}
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw_items = json.loads(text)
        except Exception:
            raw_items = [part.strip() for part in text.split(",")]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raw_items = [raw]
    normalized: List[str] = []
    for item in raw_items:
        token = str(item or "").strip().lower()
        if token in allowed and token not in normalized:
            normalized.append(token)
    return normalized


def _safe_json(resp: requests.Response) -> Dict[str, object]:
    try:
        payload = resp.json()
    except Exception:
        return {"_raw": resp.text}
    return payload if isinstance(payload, dict) else {"_payload": payload}


def _load_prepare_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cleaned: List[Dict[str, object]] = []
    for row in rows:
        if str(row.get("status") or "").strip().lower() != "success":
            continue
        url = str(row.get("output_url") or row.get("user_input_url") or row.get("url") or "").strip()
        prompt = str(row.get("prompt_description") or row.get("user_prompt") or row.get("promptDescription") or "").strip()
        if not url:
            continue
        cleaned.append(
            {
                "index": int(row.get("index") or len(cleaned) + 1),
                "user_file": str(row.get("image_name") or row.get("user_file") or "").strip(),
                "prepared_input_url": url,
                "user_prompt": prompt,
                "worn_types": _normalize_worn_types(row.get("worn_types")),
            }
        )
    cleaned.sort(key=lambda item: int(item["index"]))
    return cleaned


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return text.strip("._-") or "case"


def _build_upscale_filename(prefix: str, index: int) -> str:
    prefix_slug = _slugify(prefix)
    unique_suffix = uuid.uuid4().hex
    return f"{prefix_slug}_{index:03d}_{unique_suffix}.png"


def _resolve_output_dir(raw_output_dir: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    pod_pull_root = repo_root / "tmp" / "pod_pull"
    text = str(raw_output_dir or "").strip()
    if not text:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return pod_pull_root / f"bulk_tryon_{timestamp}"
    path = Path(text)
    if path.is_absolute():
        return path
    return pod_pull_root / path


def _write_debug_json(debug_dir: Optional[Path], name: str, payload: Dict[str, object]) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_existing(jsonl_path: Path) -> List[CaseResult]:
    rows: List[CaseResult] = []
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            rows.append(
                CaseResult(
                    index=int(payload["index"]),
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
    return rows


def _write_jsonl(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[CaseResult], args: argparse.Namespace) -> None:
    fields = [
        "index",
        "user_file",
        "prepared_input_url",
        "user_prompt",
        "tryon_output_url",
        "upscale_output_url",
        "prompt_used",
        "status",
        "error",
        "tryon_latency_s",
        "upscale_latency_s",
        "total_latency_s",
        "garment_mode",
        "top_image_url",
        "bottom_image_url",
        "dress_image_url",
        "top_prompt",
        "bottom_prompt",
        "dress_prompt",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = row.as_dict()
            payload.update(
                {
                    "garment_mode": args.garment_mode,
                    "top_image_url": args.top_image_url,
                    "bottom_image_url": args.bottom_image_url,
                    "dress_image_url": args.dress_image_url,
                    "top_prompt": args.top_prompt,
                    "bottom_prompt": args.bottom_prompt,
                    "dress_prompt": args.dress_prompt,
                }
            )
            writer.writerow(payload)


def _build_summary(rows: List[CaseResult], args: argparse.Namespace) -> Dict[str, object]:
    total = len(rows)
    success_rows = [row for row in rows if row.status == "success"]
    error_rows = [row for row in rows if row.status != "success"]
    avg_tryon = sum(row.tryon_latency_s for row in success_rows) / len(success_rows) if success_rows else 0.0
    avg_upscale = sum(row.upscale_latency_s for row in success_rows) / len(success_rows) if success_rows else 0.0
    avg_total = sum(row.total_latency_s for row in success_rows) / len(success_rows) if success_rows else 0.0
    return {
        "title": args.title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "lora_scale": args.lora_scale,
        "output_max_edge": args.output_max_edge,
        "upscale_target_long_edge": args.upscale_target_long_edge,
        "garment_mode": args.garment_mode,
        "top_image_url": args.top_image_url,
        "bottom_image_url": args.bottom_image_url,
        "dress_image_url": args.dress_image_url,
        "top_prompt": args.top_prompt,
        "bottom_prompt": args.bottom_prompt,
        "dress_prompt": args.dress_prompt,
        "total": total,
        "success": len(success_rows),
        "error": len(error_rows),
        "avg_tryon_latency_s": round(avg_tryon, 4),
        "avg_upscale_latency_s": round(avg_upscale, 4),
        "avg_total_latency_s": round(avg_total, 4),
    }


def _generate_html(path: Path, rows: List[CaseResult], summary: Dict[str, object]) -> None:
    if str(summary.get("garment_mode") or "top_bottom").strip().lower() == "dress":
        product_cards = [
            f"""
<figure class="product-card">
  <figcaption>Dress Reference</figcaption>
  <a href="{html.escape(str(summary.get('dress_image_url') or ''))}" target="_blank" rel="noopener">
    <img src="{html.escape(str(summary.get('dress_image_url') or ''))}" alt="dress garment" loading="lazy" />
  </a>
  <pre>{html.escape(str(summary.get('dress_prompt') or ''))}</pre>
</figure>
""",
        ]
    else:
        product_cards = [
            f"""
<figure class="product-card">
  <figcaption>Top Reference</figcaption>
  <a href="{html.escape(str(summary.get('top_image_url') or ''))}" target="_blank" rel="noopener">
    <img src="{html.escape(str(summary.get('top_image_url') or ''))}" alt="top garment" loading="lazy" />
  </a>
  <pre>{html.escape(str(summary.get('top_prompt') or ''))}</pre>
</figure>
""",
        f"""
<figure class="product-card">
  <figcaption>Bottom Reference</figcaption>
  <a href="{html.escape(str(summary.get('bottom_image_url') or ''))}" target="_blank" rel="noopener">
    <img src="{html.escape(str(summary.get('bottom_image_url') or ''))}" alt="bottom garment" loading="lazy" />
  </a>
  <pre>{html.escape(str(summary.get('bottom_prompt') or ''))}</pre>
</figure>
""",
        ]

    case_cards = []
    for row in rows:
        if row.status != "success" or not row.tryon_output_url:
            continue
        case_cards.append(
            f"""
<section class="case-card">
  <div class="head">
    <div>
      <div class="eyebrow">Case {row.index:03d}</div>
      <h3>{html.escape(row.user_file)}</h3>
    </div>
    <span class="badge">Success</span>
  </div>
  <div class="image-grid three">
    <figure>
      <figcaption>Prepared Input</figcaption>
      <a href="{html.escape(row.prepared_input_url)}" target="_blank" rel="noopener"><img src="{html.escape(row.prepared_input_url)}" alt="prepared input" loading="lazy" /></a>
    </figure>
    <figure>
      <figcaption>Try-on</figcaption>
      <a href="{html.escape(row.tryon_output_url)}" target="_blank" rel="noopener"><img src="{html.escape(row.tryon_output_url)}" alt="tryon output" loading="lazy" /></a>
    </figure>
    <figure>
      <figcaption>Upscaled</figcaption>
      <a href="{html.escape(row.upscale_output_url)}" target="_blank" rel="noopener"><img src="{html.escape(row.upscale_output_url)}" alt="upscaled output" loading="lazy" /></a>
    </figure>
  </div>
  <details class="prompt-details">
    <summary>View Prompt</summary>
    <div class="prompt-grid">
      <div class="prompt-item"><div class="prompt-label">User Prompt</div><pre>{html.escape(row.user_prompt or 'n/a')}</pre></div>
      <div class="prompt-item"><div class="prompt-label">Prompt Used</div><pre>{html.escape(row.prompt_used or 'n/a')}</pre></div>
      <div class="prompt-item"><div class="prompt-label">Try-on Latency</div><pre>{row.tryon_latency_s:.2f}s</pre></div>
      <div class="prompt-item"><div class="prompt-label">Upscale Latency</div><pre>{row.upscale_latency_s:.2f}s</pre></div>
    </div>
  </details>
</section>
"""
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(str(summary.get("title") or "Bulk Try-on + Upscale"))}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --border: #d8dee8;
      --text: #203040;
      --muted: #657487;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }}
    main {{ max-width: 1640px; margin: 0 auto; padding: 24px; }}
    .hero, .case-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 20px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 32px; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .summary-grid, .product-grid, .cases {{ display: grid; gap: 18px; }}
    .summary-grid {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 18px; }}
    .summary-item, .product-card {{ background: #fbfcff; border: 1px solid var(--border); border-radius: 16px; padding: 14px; }}
    .summary-item .label, .prompt-label, .eyebrow, figcaption {{ color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }}
    .summary-item .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .product-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 18px; }}
    .product-card img, .image-grid img {{ width: 100%; height: auto; border-radius: 12px; border: 1px solid var(--border); background: #fff; }}
    .product-card pre, .prompt-item pre {{ white-space: pre-wrap; word-break: break-word; margin: 10px 0 0; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; color: #334; }}
    .cases {{ margin-top: 24px; }}
    .head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    .head h3 {{ margin: 6px 0 0; font-size: 20px; }}
    .badge {{ background: #e8f4ea; color: #1f6b34; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; }}
    .image-grid {{ display: grid; gap: 18px; margin-top: 18px; }}
    .image-grid.three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .prompt-details {{ margin-top: 16px; }}
    .prompt-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 12px; }}
    .prompt-item {{ background: #fbfcff; border: 1px solid var(--border); border-radius: 14px; padding: 12px; }}
    @media (max-width: 1024px) {{
      .image-grid.three, .prompt-grid, .product-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(str(summary.get("title") or "Bulk Try-on + Upscale"))}</h1>
      <p>Prepared input, try-on output, and SeedVR2 upscaled result for each cached user case.</p>
      <div class="summary-grid">
        <div class="summary-item"><div class="label">Success</div><div class="value">{summary.get("success", 0)}</div></div>
        <div class="summary-item"><div class="label">Errors</div><div class="value">{summary.get("error", 0)}</div></div>
        <div class="summary-item"><div class="label">Seed</div><div class="value">{summary.get("seed", 0)}</div></div>
        <div class="summary-item"><div class="label">LoRA Scale</div><div class="value">{summary.get("lora_scale", 0)}</div></div>
        <div class="summary-item"><div class="label">Avg Try-on</div><div class="value">{summary.get("avg_tryon_latency_s", 0)}</div></div>
        <div class="summary-item"><div class="label">Avg Upscale</div><div class="value">{summary.get("avg_upscale_latency_s", 0)}</div></div>
      </div>
      <div class="product-grid">
        {''.join(product_cards)}
      </div>
    </section>
    <div class="cases">
      {''.join(case_cards)}
    </div>
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def _build_prompt(user_prompt: str, top_desc: str, bottom_desc: str) -> str:
    del top_desc, bottom_desc
    normalized_user = " ".join(str(user_prompt or "").split()).rstrip(" .!?").strip()
    normalized_user = normalized_user or "same person as shown in the reference image"
    return TRYON_PROMPT_TEMPLATE.format(user_prompt=normalized_user)


def _build_dress_prompt(user_prompt: str, dress_desc: str) -> str:
    normalized_user = " ".join(str(user_prompt or "").split()).rstrip(" .!?").strip()
    normalized_user = normalized_user or "same person as shown in the reference image"
    normalized_dress = " ".join(str(dress_desc or "").split()).strip()
    normalized_dress = normalized_dress or "dress garment"
    return (
        f"TRYON {normalized_user}. Replace the entire outfit with {normalized_dress} as shown in the reference images. "
        "Preserve exact person identity including face, hair color, eye direction, pose, footwear, and accessories. "
        "Preserve body geometry and silhouette: torso contour, waist-to-hip relationship, hip contour, glute projection, thigh volume, and leg length. "
        "Render garment with accurate construction, fit, seam placement, drape, hem length, texture, pattern placement. "
        "Strictly remove any worn top, bottom, dress and outer garments. "
        "Preserve the exact garment color fidelity, garment size, garment structure, deisgn patterns with positions and construction, edge finish and fabric pattern layout as shown in reference. "
        "Garment must preserve only the design and construction of its own reference image. "
        "no cross-garment inference, no feature borrowing, and no structural or visual mixing with garments. "
        "Keep camera distance, background, and lighting consistent. "
        "The final image is a full body shot."
    )


def _build_prompt_from_template(template: str, user_prompt: str) -> str:
    normalized_user = " ".join(str(user_prompt or "").split()).rstrip(" .!?").strip()
    normalized_user = normalized_user or "same person as shown in the reference image"
    return str(template).format(user_prompt=normalized_user)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk try-on + SeedVR2 upscale runner")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prepare-cache-csv", required=True)
    parser.add_argument("--garment-mode", choices=["top_bottom", "dress"], default="top_bottom")
    parser.add_argument("--top-image-url", default="")
    parser.add_argument("--bottom-image-url", default="")
    parser.add_argument("--dress-image-url", default="")
    parser.add_argument("--top-prompt", default="")
    parser.add_argument("--bottom-prompt", default="")
    parser.add_argument("--dress-prompt", default="")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--output-max-edge", type=int, default=1024)
    parser.add_argument("--upscale-target-long-edge", type=int, default=2048)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--only-file", default="")
    parser.add_argument("--debug-dump", action="store_true")
    parser.add_argument("--output-filename-prefix", default="seedvr2_tryon")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--title", default="Glamify Bulk Try-on + Upscale")
    parser.add_argument("--download-timeout", type=float, default=60.0)
    parser.add_argument("--prompt-template-file", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.garment_mode == "dress":
        if not str(args.dress_image_url or "").strip():
            raise SystemExit("--dress-image-url is required when --garment-mode dress")
        if not str(args.dress_prompt or "").strip():
            raise SystemExit("--dress-prompt is required when --garment-mode dress")
    else:
        if not str(args.top_image_url or "").strip():
            raise SystemExit("--top-image-url is required when --garment-mode top_bottom")
        if not str(args.bottom_image_url or "").strip():
            raise SystemExit("--bottom-image-url is required when --garment-mode top_bottom")
        if not str(args.top_prompt or "").strip():
            raise SystemExit("--top-prompt is required when --garment-mode top_bottom")
        if not str(args.bottom_prompt or "").strip():
            raise SystemExit("--bottom-prompt is required when --garment-mode top_bottom")
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = lambda idx: _build_upscale_filename(args.output_filename_prefix, idx)
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / "report.html"

    prepare_rows = _load_prepare_rows(Path(args.prepare_cache_csv))
    if args.start_index > 0:
        prepare_rows = [row for row in prepare_rows if int(row["index"]) >= int(args.start_index)]
    if args.end_index > 0:
        prepare_rows = [row for row in prepare_rows if int(row["index"]) <= int(args.end_index)]
    only_file = str(args.only_file or "").strip()
    if only_file:
        prepare_rows = [row for row in prepare_rows if str(row.get("user_file") or "").strip() == only_file]
    if args.max_cases > 0:
        prepare_rows = prepare_rows[: args.max_cases]

    rows = _load_existing(jsonl_path) if args.resume else []
    done_indexes = {row.index for row in rows}

    session = requests.Session()
    base = args.base_url.rstrip("/")
    prompt_template_override = ""
    if str(args.prompt_template_file or "").strip():
        prompt_template_override = Path(args.prompt_template_file).read_text(encoding="utf-8").strip()

    for item in prepare_rows:
        index = int(item["index"])
        if index in done_indexes:
            continue
        t0 = time.time()
        status = "error"
        error = ""
        tryon_output_url = ""
        upscale_output_url = ""
        prompt_used = ""
        tryon_latency = 0.0
        upscale_latency = 0.0

        try:
            if prompt_template_override:
                prompt_override = _build_prompt_from_template(prompt_template_override, str(item["user_prompt"]))
            elif args.garment_mode == "dress":
                prompt_override = _build_dress_prompt(str(item["user_prompt"]), args.dress_prompt)
            else:
                prompt_override = _build_prompt(str(item["user_prompt"]), args.top_prompt, args.bottom_prompt)
            debug_dir = None
            if args.debug_dump:
                debug_dir = output_dir / "debug" / f"{int(index):03d}_{_slugify(str(item['user_file']))}"
            if args.garment_mode == "dress":
                products = [
                    {
                        "image": args.dress_image_url,
                        "promptDescription": args.dress_prompt,
                        "targetType": "dress",
                    }
                ]
            else:
                products = [
                    {
                        "image": args.top_image_url,
                        "promptDescription": args.top_prompt,
                        "targetType": "top",
                    },
                    {
                        "image": args.bottom_image_url,
                        "promptDescription": args.bottom_prompt,
                        "targetType": "bottom",
                    },
                ]
            tryon_req = {
                "products": products,
                "user_image": {
                    "tryonImage": str(item["prepared_input_url"]),
                    "promptDescription": str(item["user_prompt"]),
                    "wornTypes": list(item.get("worn_types") or ["top", "bottom"]),
                },
                "mode": "tryon-lora",
                "steps": int(args.steps),
                "seed": int(args.seed),
                "guidanceScale": float(args.guidance_scale),
                "loraScale": float(args.lora_scale),
                "outputMaxEdge": int(args.output_max_edge),
                "prompt_override": prompt_override,
            }
            _write_debug_json(debug_dir, "01_tryon_request.json", tryon_req)

            tryon_resp = session.post(
                f"{base}/v1/flux2/tryon",
                json=tryon_req,
                timeout=420,
            )
            tryon_payload = _safe_json(tryon_resp)
            _write_debug_json(
                debug_dir,
                "02_tryon_response.json",
                {"status_code": tryon_resp.status_code, "payload": tryon_payload},
            )
            if not tryon_resp.ok:
                raise RuntimeError(f"tryon_failed status={tryon_resp.status_code} payload={tryon_payload}")

            tryon_data = tryon_payload.get("data") if isinstance(tryon_payload.get("data"), dict) else {}
            tryon_output_url = str(tryon_data.get("output_url") or "").strip()
            tryon_latency = float(tryon_data.get("latency") or 0.0)
            metadata = tryon_data.get("metadata") if isinstance(tryon_data.get("metadata"), dict) else {}
            prompt_used = str(metadata.get("prompt") or prompt_override).strip()
            if not tryon_output_url:
                raise RuntimeError("tryon_missing_output_url")

            upscale_req = {
                "image_url": tryon_output_url,
                "target_long_edge": int(args.upscale_target_long_edge),
                "batch_size": 1,
                "use_persistent": True,
                "gpu_resident": True,
                "cache_models": True,
                "timeout_seconds": 900,
                "upload_to_storage": True,
                "output_filename": output_filename(index),
            }
            _write_debug_json(debug_dir, "03_upscale_request.json", upscale_req)
            upscale_started = time.time()
            upscale_resp = session.post(f"{base}/v1/user-image/upscale", json=upscale_req, timeout=920)
            upscale_payload = _safe_json(upscale_resp)
            _write_debug_json(
                debug_dir,
                "04_upscale_response.json",
                {"status_code": upscale_resp.status_code, "payload": upscale_payload},
            )
            if not upscale_resp.ok:
                raise RuntimeError(f"upscale_failed status={upscale_resp.status_code} payload={upscale_payload}")

            upscale_data = upscale_payload.get("data") if isinstance(upscale_payload.get("data"), dict) else {}
            upscale_out = upscale_data.get("output") if isinstance(upscale_data.get("output"), dict) else {}
            upscale_output_url = str(upscale_out.get("url") or upscale_out.get("storage_url") or upscale_out.get("url_public") or "").strip()
            timings = upscale_data.get("timings") if isinstance(upscale_data.get("timings"), dict) else {}
            upscale_latency = float(timings.get("total_seconds") or timings.get("elapsed_seconds") or 0.0)
            if upscale_latency <= 0.0:
                upscale_latency = max(0.0, time.time() - upscale_started)
            if not upscale_output_url:
                raise RuntimeError("upscale_missing_output_url")

            status = "success"
        except Exception as exc:
            error = str(exc)

        total_latency = time.time() - t0
        row = CaseResult(
            index=index,
            user_file=str(item["user_file"]),
            prepared_input_url=str(item["prepared_input_url"]),
            user_prompt=str(item["user_prompt"]),
            tryon_output_url=tryon_output_url,
            upscale_output_url=upscale_output_url,
            prompt_used=prompt_used,
            status=status,
            error=error,
            tryon_latency_s=tryon_latency,
            upscale_latency_s=upscale_latency,
            total_latency_s=total_latency,
        )
        rows.append(row)
        rows.sort(key=lambda x: x.index)
        done_indexes.add(index)
        _write_jsonl(jsonl_path, row.as_dict())
        _write_csv(csv_path, rows, args)
        summary = _build_summary(rows, args)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _generate_html(html_path, rows, summary)
        print(
            f"[{row.index:03d}] {row.user_file} -> {row.status} "
            f"(tryon={row.tryon_latency_s:.2f}s upscale={row.upscale_latency_s:.2f}s total={row.total_latency_s:.2f}s)"
        )
        sys.stdout.flush()

    summary = _build_summary(rows, args)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows, args)
    _generate_html(html_path, rows, summary)
    print(f"summary_json={summary_path}")
    print(f"results_jsonl={jsonl_path}")
    print(f"results_csv={csv_path}")
    print(f"report_html={html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Temporary multi-product bulk try-on runner.

Uses prepared user cache CSV rows and posts them to /v1/flux2/tryon with a
fixed products list. Writes resumable JSONL/CSV/summary and a simple HTML view.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests


@dataclass
class CaseResult:
    index: int
    user_file: str
    user_input_url: str
    user_prompt: str
    worn_types: List[str]
    output_url: str
    prompt_used: str
    status: str
    error: str
    tryon_latency_s: float
    total_latency_s: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "user_file": self.user_file,
            "user_input_url": self.user_input_url,
            "user_prompt": self.user_prompt,
            "worn_types": list(self.worn_types or []),
            "output_url": self.output_url,
            "prompt_used": self.prompt_used,
            "status": self.status,
            "error": self.error,
            "tryon_latency_s": self.tryon_latency_s,
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
        url = str(row.get("user_input_url") or row.get("url") or "").strip()
        prompt = str(row.get("user_prompt") or row.get("promptDescription") or "").strip()
        if not url:
            continue
        cleaned.append(
            {
                "index": int(row.get("index") or len(cleaned) + 1),
                "user_file": str(row.get("user_file") or "").strip(),
                "user_input_url": url,
                "user_prompt": prompt,
                "worn_types": _normalize_worn_types(row.get("worn_types")),
            }
        )
    cleaned.sort(key=lambda item: int(item["index"]))
    return cleaned


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
                    user_input_url=str(payload.get("user_input_url") or ""),
                    user_prompt=str(payload.get("user_prompt") or ""),
                    worn_types=_normalize_worn_types(payload.get("worn_types")),
                    output_url=str(payload.get("output_url") or ""),
                    prompt_used=str(payload.get("prompt_used") or ""),
                    status=str(payload.get("status") or ""),
                    error=str(payload.get("error") or ""),
                    tryon_latency_s=float(payload.get("tryon_latency_s") or 0.0),
                    total_latency_s=float(payload.get("total_latency_s") or 0.0),
                )
            )
    return rows


def _write_jsonl(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[CaseResult], products: List[Dict[str, str]]) -> None:
    fields = [
        "index",
        "user_file",
        "user_input_url",
        "user_prompt",
        "worn_types",
        "output_url",
        "prompt_used",
        "status",
        "error",
        "tryon_latency_s",
        "total_latency_s",
        "products_json",
    ]
    products_json = json.dumps(products, ensure_ascii=False)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = row.as_dict()
            payload["worn_types"] = json.dumps(list(row.worn_types or []), ensure_ascii=False)
            payload["products_json"] = products_json
            writer.writerow(payload)


def _build_summary(rows: List[CaseResult], args: argparse.Namespace) -> Dict[str, object]:
    total = len(rows)
    success_rows = [row for row in rows if row.status == "success"]
    error_rows = [row for row in rows if row.status != "success"]
    avg_tryon = sum(row.tryon_latency_s for row in success_rows) / len(success_rows) if success_rows else 0.0
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
        "products": args.products,
        "total": total,
        "success": len(success_rows),
        "error": len(error_rows),
        "avg_tryon_latency_s": round(avg_tryon, 4),
        "avg_total_latency_s": round(avg_total, 4),
    }


def _generate_html(path: Path, rows: List[CaseResult], summary: Dict[str, object]) -> None:
    product_cards = []
    for product in summary.get("products", []):
        product_cards.append(
            f"""
<figure class="product-card">
  <figcaption>{html.escape(str(product.get('targetType') or 'garment').title())}</figcaption>
  <a href="{html.escape(str(product.get('image') or ''))}" target="_blank" rel="noopener">
    <img src="{html.escape(str(product.get('image') or ''))}" alt="garment" loading="lazy" />
  </a>
  <pre>{html.escape(str(product.get('promptDescription') or ''))}</pre>
</figure>
"""
        )
    case_cards = []
    for row in rows:
        if row.status != "success" or not row.output_url:
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
  <div class="image-grid">
    <figure>
      <figcaption>Input</figcaption>
      <a href="{html.escape(row.user_input_url)}" target="_blank" rel="noopener"><img src="{html.escape(row.user_input_url)}" alt="input" loading="lazy" /></a>
    </figure>
    <figure>
      <figcaption>Try-on</figcaption>
      <a href="{html.escape(row.output_url)}" target="_blank" rel="noopener"><img src="{html.escape(row.output_url)}" alt="tryon" loading="lazy" /></a>
    </figure>
  </div>
  <details class="prompt-details">
    <summary>View Prompt</summary>
    <div class="prompt-grid">
      <div class="prompt-item"><div class="prompt-label">User Prompt</div><pre>{html.escape(row.user_prompt or 'n/a')}</pre></div>
      <div class="prompt-item"><div class="prompt-label">Prepared Worn Types</div><pre>{html.escape(', '.join(row.worn_types or []) or 'n/a')}</pre></div>
      <div class="prompt-item prompt-item-wide"><div class="prompt-label">Final Try-on Prompt</div><pre>{html.escape(row.prompt_used or 'n/a')}</pre></div>
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
  <title>{html.escape(str(summary.get("title") or "Bulk Multi Try-on"))}</title>
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
    main {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    .hero, .case-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 20px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 32px; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .summary-grid, .product-grid, .cases {{ display: grid; gap: 18px; }}
    .summary-grid {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top: 18px; }}
    .summary-item, .product-card {{ background: #fbfcff; border: 1px solid var(--border); border-radius: 16px; padding: 14px; }}
    .summary-item .label, .prompt-label, .eyebrow, figcaption {{ color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }}
    .summary-item .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .product-grid {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); margin-top: 18px; }}
    .product-card img, .image-grid img {{ width: 100%; height: auto; border-radius: 12px; border: 1px solid var(--border); background: #fff; }}
    .product-card pre, .prompt-item pre {{ white-space: pre-wrap; word-break: break-word; margin: 10px 0 0; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; color: #334; }}
    .cases {{ margin-top: 24px; }}
    .head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    .head h3 {{ margin: 6px 0 0; font-size: 20px; }}
    .badge {{ background: #e8f4ea; color: #1f6b34; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; }}
    .image-grid {{ display: grid; gap: 18px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 18px; }}
    .prompt-details {{ margin-top: 16px; }}
    .prompt-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 12px; }}
    .prompt-item {{ background: #fbfcff; border: 1px solid var(--border); border-radius: 14px; padding: 12px; }}
    .prompt-item-wide {{ grid-column: 1 / -1; }}
    @media (max-width: 920px) {{
      .image-grid, .prompt-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(str(summary.get("title") or "Bulk Multi Try-on"))}</h1>
      <p>Multi-garment try-on run with cached prepared users.</p>
      <div class="summary-grid">
        <div class="summary-item"><div class="label">Success</div><div class="value">{summary.get("success", 0)}</div></div>
        <div class="summary-item"><div class="label">Errors</div><div class="value">{summary.get("error", 0)}</div></div>
        <div class="summary-item"><div class="label">Seed</div><div class="value">{summary.get("seed", 0)}</div></div>
        <div class="summary-item"><div class="label">Output Edge</div><div class="value">{summary.get("output_max_edge", 0)}</div></div>
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk multi-product try-on runner")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prepare-cache-csv", required=True)
    parser.add_argument("--products-json", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--output-max-edge", type=int, default=1280)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--title", default="Glamify Bulk Multi Tryon Testing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / "report.html"

    products = json.loads(Path(args.products_json).read_text(encoding="utf-8"))
    if not isinstance(products, list) or not products:
        raise SystemExit("products-json must contain a non-empty list")
    args.products = products

    prepare_rows = _load_prepare_rows(Path(args.prepare_cache_csv))
    if args.max_cases > 0:
        prepare_rows = prepare_rows[: args.max_cases]

    rows = _load_existing(jsonl_path) if args.resume else []
    done_indexes = {row.index for row in rows}

    session = requests.Session()
    base = args.base_url.rstrip("/")

    for item in prepare_rows:
        index = int(item["index"])
        if index in done_indexes:
            continue
        t0 = time.time()
        payload = {
            "mode": "tryon-lora",
            "steps": args.steps,
            "seed": args.seed,
            "guidanceScale": args.guidance_scale,
            "loraScale": args.lora_scale,
            "outputMaxEdge": args.output_max_edge,
            "user_image": {
                "tryonImage": item["user_input_url"],
                "promptDescription": item["user_prompt"],
                "wornTypes": item["worn_types"],
            },
            "products": products,
        }
        status = "error"
        error = ""
        output_url = ""
        prompt_used = ""
        tryon_latency = 0.0
        try:
            resp = session.post(f"{base}/v1/flux2/tryon", json=payload, timeout=420)
            data = _safe_json(resp)
            if resp.ok:
                result = data.get("data") if isinstance(data.get("data"), dict) else {}
                output_url = str(result.get("output_url") or "").strip()
                metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                prompt_used = str(metadata.get("prompt") or "").strip()
                tryon_latency = float(result.get("latency") or 0.0)
                status = "success" if output_url else "error"
                if status != "success":
                    error = "missing output_url"
            else:
                error = str(data)
        except Exception as exc:
            error = str(exc)
        total_latency = time.time() - t0
        row = CaseResult(
            index=index,
            user_file=str(item["user_file"]),
            user_input_url=str(item["user_input_url"]),
            user_prompt=str(item["user_prompt"]),
            worn_types=list(item["worn_types"]),
            output_url=output_url,
            prompt_used=prompt_used,
            status=status,
            error=error,
            tryon_latency_s=tryon_latency,
            total_latency_s=total_latency,
        )
        rows.append(row)
        rows.sort(key=lambda x: x.index)
        done_indexes.add(index)
        _write_jsonl(jsonl_path, row.as_dict())
        _write_csv(csv_path, rows, products)
        summary = _build_summary(rows, args)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _generate_html(html_path, rows, summary)
        print(
            f"[{row.index:03d}] {row.user_file} -> {row.status} "
            f"(tryon={row.tryon_latency_s:.2f}s total={row.total_latency_s:.2f}s)"
        )
        sys.stdout.flush()

    summary = _build_summary(rows, args)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows, products)
    _generate_html(html_path, rows, summary)
    print(f"summary_json={summary_path}")
    print(f"results_jsonl={jsonl_path}")
    print(f"results_csv={csv_path}")
    print(f"report_html={html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

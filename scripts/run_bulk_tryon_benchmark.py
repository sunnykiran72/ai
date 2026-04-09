#!/usr/bin/env python3
"""
Bulk virtual try-on benchmark pipeline:
1) Prepare user image prompt via /v1/user-image/prepare
2) Run /v1/flux2/tryon with a fixed garment
3) Save resumable JSONL/CSV and generate side-by-side HTML report
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
from typing import Dict, Iterable, List, Optional, Tuple

import requests


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


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
    prepare_latency_s: float
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
            "prepare_latency_s": self.prepare_latency_s,
            "tryon_latency_s": self.tryon_latency_s,
            "total_latency_s": self.total_latency_s,
        }


def _iter_user_images(users_dir: Path) -> Iterable[Path]:
    for path in sorted(users_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        yield path


def _safe_json(resp: requests.Response) -> Dict[str, object]:
    try:
        payload = resp.json()
    except Exception:
        return {"_raw": resp.text}
    return payload if isinstance(payload, dict) else {"_payload": payload}


def _normalize_worn_types(values: object) -> List[str]:
    allowed = {"top", "bottom", "outer", "dress"}
    normalized: List[str] = []
    if values is None:
        return normalized
    if isinstance(values, list):
        raw_items = values
    elif isinstance(values, str):
        text = values.strip()
        if not text:
            return normalized
        try:
            raw_items = json.loads(text)
        except Exception:
            raw_items = [part.strip() for part in text.split(",")]
    else:
        raw_items = [values]
    for raw in raw_items:
        token = str(raw or "").strip().lower()
        if token in allowed and token not in normalized:
            normalized.append(token)
    return normalized


def _extract_prepare_fields(payload: Dict[str, object]) -> Tuple[str, str, List[str]]:
    data = payload.get("data")
    if not isinstance(data, dict):
        data = payload
    url = str(data.get("url") or data.get("tryonImage") or "").strip()
    prompt = str(data.get("promptDescription") or "").strip()
    worn_types = _normalize_worn_types(data.get("wornTypes"))
    return url, prompt, worn_types


def _extract_tryon_fields(payload: Dict[str, object]) -> Tuple[str, str, float]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return "", "", 0.0
    output_url = str(data.get("output_url") or "").strip()
    metadata = data.get("metadata")
    prompt_used = ""
    if isinstance(metadata, dict):
        prompt_used = str(metadata.get("prompt") or "").strip()
    latency = float(data.get("latency") or 0.0)
    return output_url, prompt_used, latency


def _request_with_retry(
    fn,
    *,
    attempts: int = 4,
    sleep_seconds: float = 2.0,
) -> requests.Response:
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except requests.RequestException as err:
            last_err = err
            if attempt >= attempts:
                break
            time.sleep(sleep_seconds * attempt)
    assert last_err is not None
    raise last_err


def _write_jsonl(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[CaseResult], garment_url: str, garment_prompt: str, garment_type: str) -> None:
    fields = [
        "index",
        "user_file",
        "user_input_url",
        "output_url",
        "garment_url",
        "garment_prompt",
        "garment_type",
        "user_prompt",
        "worn_types",
        "prompt_used",
        "status",
        "error",
        "prepare_latency_s",
        "tryon_latency_s",
        "total_latency_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row.as_dict(),
                    "garment_url": garment_url,
                    "garment_type": garment_type,
                    "garment_prompt": garment_prompt,
                    "worn_types": json.dumps(list(row.worn_types or []), ensure_ascii=False),
                }
            )


def _generate_html(
    path: Path,
    rows: List[CaseResult],
    *,
    garment_url: str,
    garment_prompt: str,
    garment_type: str,
    summary: Dict[str, object],
) -> None:
    del garment_prompt

    visible_rows = [row for row in rows if row.status == "success" and row.output_url and row.user_input_url]
    cards: List[str] = []
    for row in visible_rows:
        worn_types_label = ", ".join(row.worn_types or []) or "n/a"
        prompt_block = ""
        if row.prompt_used or row.user_prompt or worn_types_label != "n/a":
            prompt_block = f"""
  <details class="prompt-details">
    <summary>View Prompt</summary>
    <div class="prompt-grid">
      <div class="prompt-item">
        <div class="prompt-label">User Prompt</div>
        <pre>{html.escape(row.user_prompt or "n/a")}</pre>
      </div>
      <div class="prompt-item">
        <div class="prompt-label">Prepared Worn Types</div>
        <pre>{html.escape(worn_types_label)}</pre>
      </div>
      <div class="prompt-item prompt-item-wide">
        <div class="prompt-label">Final Try-on Prompt</div>
        <pre>{html.escape(row.prompt_used or "n/a")}</pre>
      </div>
    </div>
  </details>
"""
        cards.append(
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
      <a href="{html.escape(row.user_input_url)}" target="_blank" rel="noopener">
        <img src="{html.escape(row.user_input_url)}" alt="input user" loading="lazy" />
      </a>
    </figure>
    <figure>
      <figcaption>Try-on</figcaption>
      <a href="{html.escape(row.output_url)}" target="_blank" rel="noopener">
        <img src="{html.escape(row.output_url)}" alt="tryon output" loading="lazy" />
      </a>
    </figure>
  </div>
  {prompt_block}
</section>
"""
        )

    summary_json = html.escape(json.dumps(summary, indent=2))
    cases_total = int(summary.get("cases_total") or 0)
    cases_success = int(summary.get("cases_success") or 0)
    cases_error = int(summary.get("cases_error") or 0)
    avg_tryon = summary.get("latency_avg_tryon_s")
    avg_total = summary.get("latency_avg_total_s")
    avg_tryon_label = f"{float(avg_tryon):.2f}s" if avg_tryon is not None else "n/a"
    avg_total_label = f"{float(avg_total):.2f}s" if avg_total is not None else "n/a"
    settings = summary.get("settings") if isinstance(summary.get("settings"), dict) else {}
    seed_label = html.escape(str(settings.get("seed") or "n/a"))
    max_edge_label = html.escape(str(settings.get("output_max_edge") or "n/a"))
    visible_label = len(visible_rows)
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Glamify Bulk Tryon Testing</title>
  <style>
    :root {{
      --bg: #f7f7f5;
      --surface: #ffffff;
      --surface-soft: #fbfbfa;
      --line: #e7e4df;
      --ink: #181713;
      --muted: #6b655c;
      --accent: #0f766e;
      --badge-bg: #eef7f5;
      --shadow: 0 18px 40px rgba(24, 23, 19, 0.06);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Instrument Sans", "Manrope", "Avenir Next", sans-serif;
      background:
        radial-gradient(circle at top left, #fffdfa 0%, transparent 30%),
        linear-gradient(180deg, #faf9f6 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 24px 18px 48px;
    }}
    .hero {{
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(180deg, #fff 0%, #fcfcfb 100%);
      padding: 22px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
    }}
    .hero-top {{
      display: grid;
      grid-template-columns: 1.35fr 300px;
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 38px;
      line-height: 1;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
      max-width: 880px;
    }}
    .garment-panel {{
      border: 1px solid var(--line);
      border-radius: 22px;
      background: var(--surface-soft);
      padding: 14px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(7, minmax(120px, 1fr));
      gap: 10px;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--surface);
      padding: 12px 14px;
    }}
    .chip .k {{
      display: block;
      margin-bottom: 4px;
      font-size: 11px;
      line-height: 1.2;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .chip .v {{
      display: block;
      font-size: 18px;
      line-height: 1.1;
      font-weight: 700;
      color: var(--ink);
    }}
    details {{
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--surface-soft);
      padding: 12px 14px;
    }}
    summary {{
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
      color: var(--ink);
    }}
    .summary-pre {{
      margin: 12px 0 0;
      padding: 16px;
      border-radius: 14px;
      background: #f2f1ed;
      color: #2a2824;
      font-size: 12px;
      line-height: 1.45;
      max-height: 260px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .case-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--surface);
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .eyebrow {{
      margin-bottom: 6px;
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .case-card h3 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
      word-break: break-word;
    }}
    .badge {{
      font-size: 11px;
      background: var(--badge-bg);
      color: var(--accent);
      border: 1px solid rgba(15, 118, 110, 0.18);
      border-radius: 999px;
      padding: 6px 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 10px;
      background: var(--surface-soft);
    }}
    figcaption {{
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    img {{
      width: 100%;
      height: 520px;
      object-fit: contain;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      display: block;
    }}
    a {{
      text-decoration: none;
      color: inherit;
    }}
    .prompt-details {{
      margin-top: 12px;
      border-radius: 18px;
      background: var(--surface-soft);
    }}
    .prompt-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .prompt-item {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--surface);
      padding: 10px;
    }}
    .prompt-item-wide {{
      grid-column: 1 / -1;
    }}
    .prompt-label {{
      margin-bottom: 6px;
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .prompt-item pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      color: #2a2824;
      max-height: 240px;
      overflow: auto;
    }}
    @media (max-width: 1280px) {{
      .hero-top {{
        grid-template-columns: 1fr;
      }}
      .summary-grid {{
        grid-template-columns: repeat(3, minmax(120px, 1fr));
      }}
      .gallery {{
        grid-template-columns: 1fr;
      }}
      img {{
        height: 420px;
      }}
      .prompt-grid {{
        grid-template-columns: 1fr;
      }}
      .prompt-item-wide {{
        grid-column: auto;
      }}
    }}
    @media (max-width: 780px) {{
      .wrap {{
        padding: 16px 12px 32px;
      }}
      .hero h1 {{
        font-size: 30px;
      }}
      .summary-grid {{
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }}
      .image-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-top">
        <div>
          <h1>Glamify Bulk Tryon Testing</h1>
          <p>
            Presentation gallery for the bulk try-on run. This page shows successful outputs only,
            so reviewing the dataset stays fast and visually clean.
          </p>
        </div>
        <aside class="garment-panel">
          <div class="eyebrow">Garment Reference</div>
          <a href="{html.escape(garment_url)}" target="_blank" rel="noopener">
            <img src="{html.escape(garment_url)}" alt="garment reference" />
          </a>
        </aside>
      </div>
      <div class="summary-grid">
        <div class="chip"><span class="k">Total</span><span class="v">{cases_total}</span></div>
        <div class="chip"><span class="k">Success</span><span class="v">{cases_success}</span></div>
        <div class="chip"><span class="k">Error</span><span class="v">{cases_error}</span></div>
        <div class="chip"><span class="k">Visible</span><span class="v">{visible_label}</span></div>
        <div class="chip"><span class="k">Seed</span><span class="v">{seed_label}</span></div>
        <div class="chip"><span class="k">Max Edge</span><span class="v">{max_edge_label}</span></div>
        <div class="chip"><span class="k">Avg Try-on</span><span class="v">{avg_tryon_label}</span></div>
      </div>
      <details>
        <summary>Run Summary JSON</summary>
        <pre class="summary-pre">{summary_json}</pre>
      </details>
    </section>
    <section class="gallery">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _build_summary(
    *,
    rows: List[CaseResult],
    base_url: str,
    users_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    ok_rows = [r for r in rows if r.status == "success"]
    err_rows = [r for r in rows if r.status != "success"]
    summary: Dict[str, object] = {
        "base_url": base_url,
        "users_dir": str(users_dir),
        "generated_at": datetime.now().isoformat(),
        "cases_total": len(rows),
        "cases_success": len(ok_rows),
        "cases_error": len(err_rows),
        "settings": {
            "steps": int(args.steps),
            "seed": int(args.seed),
            "guidance_scale": float(args.guidance_scale),
            "lora_scale": float(args.lora_scale),
            "output_max_edge": int(args.output_max_edge),
            "garment_type": args.garment_type,
            "garment_url": args.garment_url,
        },
    }
    if ok_rows:
        summary["latency_avg_prepare_s"] = round(sum(r.prepare_latency_s for r in ok_rows) / len(ok_rows), 4)
        summary["latency_avg_tryon_s"] = round(sum(r.tryon_latency_s for r in ok_rows) / len(ok_rows), 4)
        summary["latency_avg_total_s"] = round(sum(r.total_latency_s for r in ok_rows) / len(ok_rows), 4)
    if err_rows:
        summary["sample_errors"] = [r.error for r in err_rows[:10]]
    return summary


def _load_existing(jsonl_path: Path) -> Dict[str, Dict[str, object]]:
    by_file: Dict[str, Dict[str, object]] = {}
    if not jsonl_path.exists():
        return by_file
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        key = str(row.get("user_file") or "").strip()
        if key:
            by_file[key] = row
    return by_file


def _load_prepare_cache(jsonl_path: Path) -> Dict[str, Tuple[str, str, List[str]]]:
    cache: Dict[str, Tuple[str, str, List[str]]] = {}
    if not jsonl_path.exists():
        return cache
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if str(row.get("status") or "").strip().lower() != "success":
            continue
        user_file = str(row.get("user_file") or "").strip()
        user_input_url = str(row.get("user_input_url") or "").strip()
        user_prompt = str(row.get("user_prompt") or "").strip()
        worn_types = _normalize_worn_types(row.get("worn_types") or row.get("wornTypes"))
        if user_file and user_input_url and user_prompt:
            cache[user_file] = (user_input_url, user_prompt, worn_types)
    return cache


def _load_prepare_cache_csv(csv_path: Path) -> Dict[str, Tuple[str, str, List[str]]]:
    cache: Dict[str, Tuple[str, str, List[str]]] = {}
    if not csv_path.exists():
        return cache
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status") or "").strip().lower() != "success":
                continue
            user_file = str(row.get("user_file") or "").strip()
            user_input_url = str(row.get("user_input_url") or "").strip()
            user_prompt = str(row.get("user_prompt") or "").strip()
            worn_types = _normalize_worn_types(row.get("worn_types") or row.get("wornTypes"))
            if user_file and user_input_url and user_prompt:
                cache[user_file] = (user_input_url, user_prompt, worn_types)
    return cache


def run(args: argparse.Namespace) -> int:
    users_dir = Path(args.users_dir).expanduser().resolve()
    if not users_dir.exists():
        print(f"users_dir does not exist: {users_dir}", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"debug_outputs/bulk_tryon_{ts}").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "results.jsonl"
    csv_path = out_dir / "results.csv"
    summary_path = out_dir / "summary.json"
    html_path = out_dir / "report.html"

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    base = args.base_url.rstrip("/")
    prep_url = f"{base}/v1/user-image/prepare"
    tryon_url = f"{base}/v1/flux2/tryon"

    existing = _load_existing(jsonl_path) if args.resume else {}
    rows: List[CaseResult] = []
    if existing:
        for value in existing.values():
            rows.append(
                CaseResult(
                    index=int(value.get("index") or 0),
                    user_file=str(value.get("user_file") or ""),
                    user_input_url=str(value.get("user_input_url") or ""),
                    user_prompt=str(value.get("user_prompt") or ""),
                    worn_types=_normalize_worn_types(value.get("worn_types") or value.get("wornTypes")),
                    output_url=str(value.get("output_url") or ""),
                    prompt_used=str(value.get("prompt_used") or ""),
                    status=str(value.get("status") or "error"),
                    error=str(value.get("error") or ""),
                    prepare_latency_s=float(value.get("prepare_latency_s") or 0.0),
                    tryon_latency_s=float(value.get("tryon_latency_s") or 0.0),
                    total_latency_s=float(value.get("total_latency_s") or 0.0),
                )
            )

    files = list(_iter_user_images(users_dir))
    if args.max_cases and args.max_cases > 0:
        files = files[: args.max_cases]

    prepare_cache: Dict[str, Tuple[str, str, List[str]]] = {}
    if args.prepare_cache_jsonl:
        cache_path = Path(args.prepare_cache_jsonl).expanduser().resolve()
        if not cache_path.exists():
            print(f"prepare_cache_jsonl does not exist: {cache_path}", file=sys.stderr)
            return 2
        prepare_cache.update(_load_prepare_cache(cache_path))
        print(f"prepare_cache_jsonl={cache_path}")
        print(f"prepare_cache_entries={len(prepare_cache)}")
    if args.prepare_cache_csv:
        cache_path = Path(args.prepare_cache_csv).expanduser().resolve()
        if not cache_path.exists():
            print(f"prepare_cache_csv does not exist: {cache_path}", file=sys.stderr)
            return 2
        csv_cache = _load_prepare_cache_csv(cache_path)
        for user_file, value in csv_cache.items():
            if user_file in prepare_cache:
                existing_url, existing_prompt, existing_worn_types = prepare_cache[user_file]
                csv_url, csv_prompt, csv_worn_types = value
                prepare_cache[user_file] = (
                    existing_url or csv_url,
                    existing_prompt or csv_prompt,
                    existing_worn_types or csv_worn_types,
                )
            else:
                prepare_cache[user_file] = value
        print(f"prepare_cache_csv={cache_path}")
        print(f"prepare_cache_entries={len(prepare_cache)}")

    print(f"users_dir={users_dir}")
    print(f"cases_total={len(files)}")
    print(f"output_dir={out_dir}")
    print(f"base_url={base}")

    if args.garment_prompt_file:
        prompt_file = Path(args.garment_prompt_file).expanduser().resolve()
        if not prompt_file.exists():
            print(f"garment_prompt_file does not exist: {prompt_file}", file=sys.stderr)
            return 2
        args.garment_prompt = prompt_file.read_text(encoding="utf-8").strip()

    done_keys = {r.user_file for r in rows if r.user_file}
    next_index = max([r.index for r in rows], default=0) + 1

    for user_path in files:
        key = user_path.name
        if key in done_keys:
            continue

        started = time.time()
        prepare_latency = 0.0
        tryon_latency = 0.0
        worn_types: List[str] = []
        try:
            cached = prepare_cache.get(key)
            if cached:
                user_input_url, user_prompt, worn_types = cached
            else:
                if args.prepare_cache_only:
                    raise RuntimeError("prepare_cache_miss")
                with user_path.open("rb") as handle:
                    def _prep_call() -> requests.Response:
                        return session.post(
                            prep_url,
                            files={"file": (user_path.name, handle, "application/octet-stream")},
                            timeout=args.prepare_timeout,
                        )

                    prep_t0 = time.time()
                    prep_resp = _request_with_retry(_prep_call, attempts=args.retries)
                    prepare_latency = time.time() - prep_t0

                prep_payload = _safe_json(prep_resp)
                if prep_resp.status_code != 200:
                    raise RuntimeError(f"prepare_failed status={prep_resp.status_code} payload={prep_payload}")

                user_input_url, user_prompt, worn_types = _extract_prepare_fields(prep_payload)
                if not user_input_url:
                    raise RuntimeError(f"prepare_missing_url payload={prep_payload}")

            tryon_payload = {
                "products": [
                    {
                        "image": args.garment_url,
                        "promptDescription": args.garment_prompt,
                        "targetType": args.garment_type,
                    }
                ],
                "user_image": {
                    "tryonImage": user_input_url,
                    "promptDescription": user_prompt,
                },
                "mode": "tryon-lora",
                "steps": int(args.steps),
                "seed": int(args.seed),
                "guidanceScale": float(args.guidance_scale),
                "loraScale": float(args.lora_scale),
                "outputMaxEdge": int(args.output_max_edge),
            }
            if worn_types:
                tryon_payload["user_image"]["wornTypes"] = list(worn_types)

            def _tryon_call() -> requests.Response:
                return session.post(tryon_url, json=tryon_payload, timeout=args.tryon_timeout)

            tryon_t0 = time.time()
            tryon_resp = _request_with_retry(_tryon_call, attempts=args.retries)
            tryon_latency = time.time() - tryon_t0
            tryon_payload_resp = _safe_json(tryon_resp)
            if tryon_resp.status_code != 200:
                raise RuntimeError(f"tryon_failed status={tryon_resp.status_code} payload={tryon_payload_resp}")

            output_url, prompt_used, latency_api = _extract_tryon_fields(tryon_payload_resp)
            if latency_api > 0.0:
                tryon_latency = latency_api
            if not output_url:
                raise RuntimeError(f"tryon_missing_output payload={tryon_payload_resp}")

            row = CaseResult(
                index=next_index,
                user_file=key,
                user_input_url=user_input_url,
                user_prompt=user_prompt,
                worn_types=list(worn_types),
                output_url=output_url,
                prompt_used=prompt_used,
                status="success",
                error="",
                prepare_latency_s=prepare_latency,
                tryon_latency_s=tryon_latency,
                total_latency_s=time.time() - started,
            )
        except Exception as exc:
            row = CaseResult(
                index=next_index,
                user_file=key,
                user_input_url="",
                user_prompt="",
                worn_types=list(worn_types),
                output_url="",
                prompt_used="",
                status="error",
                error=str(exc),
                prepare_latency_s=prepare_latency,
                tryon_latency_s=tryon_latency,
                total_latency_s=time.time() - started,
            )

        rows.append(row)
        done_keys.add(key)
        _write_jsonl(jsonl_path, row.as_dict())
        rows.sort(key=lambda r: r.index)
        _write_csv(csv_path, rows, args.garment_url, args.garment_prompt, args.garment_type)
        progress_summary = _build_summary(rows=rows, base_url=base, users_dir=users_dir, args=args)
        summary_path.write_text(json.dumps(progress_summary, indent=2), encoding="utf-8")
        _generate_html(
            html_path,
            rows,
            garment_url=args.garment_url,
            garment_prompt=args.garment_prompt,
            garment_type=args.garment_type,
            summary=progress_summary,
        )
        print(
            f"[{row.index:03d}] {row.user_file} -> {row.status} "
            f"(prepare={row.prepare_latency_s:.2f}s tryon={row.tryon_latency_s:.2f}s total={row.total_latency_s:.2f}s)"
        )
        next_index += 1

    rows.sort(key=lambda r: r.index)
    _write_csv(csv_path, rows, args.garment_url, args.garment_prompt, args.garment_type)
    summary = _build_summary(rows=rows, base_url=base, users_dir=users_dir, args=args)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _generate_html(
        html_path,
        rows,
        garment_url=args.garment_url,
        garment_prompt=args.garment_prompt,
        garment_type=args.garment_type,
        summary=summary,
    )

    print(f"summary_json={summary_path}")
    print(f"results_jsonl={jsonl_path}")
    print(f"results_csv={csv_path}")
    print(f"report_html={html_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk try-on benchmark runner")
    parser.add_argument("--base-url", default="https://1r6ln3rbln3jhh-8000.proxy.runpod.net")
    parser.add_argument("--users-dir", default="tryon_dataset/users")
    parser.add_argument("--garment-url", required=True)
    parser.add_argument("--garment-type", default="dress", choices=["top", "bottom", "outer", "dress"])
    parser.add_argument("--garment-prompt", default="")
    parser.add_argument("--garment-prompt-file", default="", help="Optional path to read garment prompt text from file")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--output-max-edge", type=int, default=1024)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--prepare-timeout", type=float, default=90.0)
    parser.add_argument("--tryon-timeout", type=float, default=420.0)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--prepare-cache-jsonl", default="", help="Reuse prepared user_input_url/user_prompt from another run's results.jsonl")
    parser.add_argument("--prepare-cache-csv", default="", help="Reuse prepared user_input_url/user_prompt/worn_types from a CSV cache")
    parser.add_argument("--prepare-cache-only", action="store_true", help="Fail case when cache entry is missing instead of calling /v1/user-image/prepare")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.garment_prompt and not args.garment_prompt_file:
        parser.error("Provide either --garment-prompt or --garment-prompt-file")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

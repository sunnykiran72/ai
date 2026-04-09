#!/usr/bin/env python3
"""
Generate a local-only detailed HTML report for a single bulk try-on run.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _format_worn_types(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unavailable in this run"
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        cleaned = [str(item).strip() for item in parsed if str(item).strip()]
        return ", ".join(cleaned) if cleaned else "Unavailable in this run"
    return text


def _chip(label: str, value: str) -> str:
    return (
        '<div class="chip">'
        f'<span class="k">{html.escape(label)}</span>'
        f'<span class="v">{html.escape(value)}</span>'
        "</div>"
    )


def _build_html(*, title: str, rows: List[Dict[str, str]], summary: Dict[str, object]) -> str:
    success_rows = [row for row in rows if str(row.get("status") or "").strip().lower() == "success"]
    garment_url = str(summary.get("settings", {}).get("garment_url") or "").strip() if isinstance(summary.get("settings"), dict) else ""
    garment_type = str(summary.get("settings", {}).get("garment_type") or "").strip() if isinstance(summary.get("settings"), dict) else ""
    if not garment_url and success_rows:
        garment_url = str(success_rows[0].get("garment_url") or "").strip()
    if not garment_type and success_rows:
        garment_type = str(success_rows[0].get("garment_type") or "").strip()

    cards: List[str] = []
    for row in success_rows:
        user_file = str(row.get("user_file") or "").strip()
        input_url = str(row.get("user_input_url") or "").strip()
        output_url = str(row.get("output_url") or "").strip()
        user_prompt = str(row.get("user_prompt") or "").strip() or "n/a"
        garment_prompt = str(row.get("garment_prompt") or "").strip() or "n/a"
        prompt_used = str(row.get("prompt_used") or "").strip() or "n/a"
        selected_garment_type = str(row.get("garment_type") or "").strip() or "n/a"
        source_worn_types = _format_worn_types(str(row.get("worn_types") or "").strip())
        cards.append(
            f"""
<section class="case-card">
  <div class="head">
    <div>
      <div class="eyebrow">Case</div>
      <h3>{html.escape(user_file)}</h3>
    </div>
    <span class="badge">Success</span>
  </div>
  <div class="image-grid">
    <figure>
      <figcaption>Input</figcaption>
      <a href="{html.escape(input_url)}" target="_blank" rel="noopener">
        <img src="{html.escape(input_url)}" alt="input user" loading="lazy" />
      </a>
    </figure>
    <figure>
      <figcaption>Output</figcaption>
      <a href="{html.escape(output_url)}" target="_blank" rel="noopener">
        <img src="{html.escape(output_url)}" alt="try-on output" loading="lazy" />
      </a>
    </figure>
  </div>
  <details class="details-block">
    <summary>View Local Details</summary>
    <div class="meta-grid">
      <div class="meta-block">
        <div class="label">Prepared Source Garment Types</div>
        <pre>{html.escape(source_worn_types)}</pre>
      </div>
      <div class="meta-block">
        <div class="label">Selected Garment Type</div>
        <pre>{html.escape(selected_garment_type)}</pre>
      </div>
      <div class="meta-block">
        <div class="label">Prepared User Prompt</div>
        <pre>{html.escape(user_prompt)}</pre>
      </div>
      <div class="meta-block wide">
        <div class="label">Garment Prompt</div>
        <pre>{html.escape(garment_prompt)}</pre>
      </div>
      <div class="meta-block wide">
        <div class="label">Prepared User Image URL</div>
        <pre>{html.escape(input_url or "n/a")}</pre>
      </div>
      <div class="meta-block wide">
        <div class="label">Actual Try-On Prompt Used</div>
        <pre>{html.escape(prompt_used)}</pre>
      </div>
      <div class="meta-block wide">
        <div class="label">Output URL</div>
        <pre>{html.escape(output_url or "n/a")}</pre>
      </div>
    </div>
  </details>
</section>
"""
        )

    chips = [
        _chip("Cases", str(len(success_rows))),
        _chip("Garment", garment_type.title() if garment_type else "n/a"),
    ]
    settings = summary.get("settings") if isinstance(summary.get("settings"), dict) else {}
    if settings:
        chips.extend(
            [
                _chip("Seed", str(settings.get("seed", "n/a"))),
                _chip("Steps", str(settings.get("steps", "n/a"))),
                _chip("Guidance", str(settings.get("guidance_scale", "n/a"))),
                _chip("LoRA", str(settings.get("lora_scale", "n/a"))),
                _chip("Output Edge", str(settings.get("output_max_edge", "n/a"))),
            ]
        )

    garment_panel = ""
    if garment_url:
        garment_label = garment_type.title() if garment_type else "Garment"
        garment_panel = f"""
      <div class="garment-panel">
        <div class="eyebrow">Garment</div>
        <h2>{html.escape(garment_label)}</h2>
        <a href="{html.escape(garment_url)}" target="_blank" rel="noopener">
          <img src="{html.escape(garment_url)}" alt="{html.escape(garment_label)} reference" loading="lazy" />
        </a>
      </div>
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
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
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Instrument Sans", "Manrope", "Avenir Next", sans-serif;
      background:
        radial-gradient(circle at top left, #fffdfa 0%, transparent 30%),
        linear-gradient(180deg, #faf9f6 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1680px;
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
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
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
      max-width: 900px;
    }}
    .garment-panel {{
      width: 240px;
      flex: 0 0 240px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: var(--surface-soft);
      padding: 14px;
    }}
    .garment-panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      line-height: 1.15;
    }}
    .garment-panel img {{
      width: 100%;
      height: 280px;
      object-fit: contain;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 16px;
      display: block;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(7, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 18px;
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
    .details-block {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--surface-soft);
      padding: 12px 14px;
    }}
    .details-block summary {{
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
      color: var(--ink);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .meta-block {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fcfbf8;
      padding: 12px;
      min-width: 0;
    }}
    .meta-block.wide {{
      grid-column: 1 / -1;
    }}
    .label {{
      margin-bottom: 8px;
      font-size: 12px;
      line-height: 1.2;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .meta-block pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.5;
      color: #2a2824;
      max-height: 260px;
      overflow: auto;
      font-family: "SFMono-Regular", "Menlo", monospace;
    }}
    @media (max-width: 1280px) {{
      .summary-grid {{
        grid-template-columns: repeat(4, minmax(120px, 1fr));
      }}
      .gallery {{
        grid-template-columns: 1fr;
      }}
      img {{
        height: 420px;
      }}
    }}
    @media (max-width: 900px) {{
      .hero-top {{
        flex-direction: column;
      }}
      .garment-panel {{
        width: 100%;
        flex-basis: auto;
      }}
      .image-grid {{
        grid-template-columns: 1fr;
      }}
      .meta-grid {{
        grid-template-columns: 1fr;
      }}
      .summary-grid {{
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }}
      .wrap {{
        padding: 16px 12px 32px;
      }}
      .hero {{
        padding: 16px;
      }}
      .hero h1 {{
        font-size: 30px;
      }}
      img {{
        height: 340px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-top">
        <div>
          <h1>{html.escape(title)}</h1>
          <p>Local detailed review for a single try-on run. Each card shows the input, generated output, source garment types, prepared user prompt, garment prompt, and the final prompt sent to the try-on endpoint.</p>
        </div>
        {garment_panel}
      </div>
      <div class="summary-grid">
        {''.join(chips)}
      </div>
    </section>
    <div class="gallery">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a local detailed HTML report for a single try-on run.")
    parser.add_argument("--run-dir", required=True, help="Directory under debug_outputs containing results.csv")
    parser.add_argument("--title", default="Glamify Bulk Tryon Testing")
    parser.add_argument("--output-name", default="local_report.html")
    args = parser.parse_args()

    debug_outputs = Path(__file__).resolve().parent.parent / "debug_outputs"
    run_dir = (debug_outputs / str(args.run_dir).strip()).resolve()
    try:
        run_dir.relative_to(debug_outputs.resolve())
    except ValueError as exc:
        raise SystemExit(f"run_dir escapes debug_outputs: {run_dir}") from exc

    results_path = run_dir / "results.csv"
    summary_path = run_dir / "summary.json"
    if not results_path.exists():
        raise SystemExit(f"results.csv not found: {results_path}")

    rows = _read_rows(results_path)
    summary = _read_summary(summary_path)
    html_text = _build_html(title=args.title, rows=rows, summary=summary)
    output_path = run_dir / str(args.output_name).strip()
    output_path.write_text(html_text, encoding="utf-8")
    print(f"wrote_report={output_path}")
    print(f"cases_total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

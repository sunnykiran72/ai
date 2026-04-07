#!/usr/bin/env python3
"""Build side-by-side HTML report comparing two bulk try-on CSV runs."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("user_file") or row.get("index") or str(len(out) + 1)
        out[key] = row
    return out


def _img_cell(url: str, link_label: str) -> str:
    safe_url = html.escape(url or "")
    if not safe_url:
        return "<div class='missing'>N/A</div>"
    return (
        f"<img src='{safe_url}' loading='lazy' alt='img'/>"
        f"<div><a href='{safe_url}' target='_blank' rel='noopener'>{link_label}</a></div>"
    )


def build_report(
    *,
    csv_a: Path,
    csv_b: Path,
    label_a: str,
    label_b: str,
    output_html: Path,
) -> None:
    a = _read_csv(csv_a)
    b = _read_csv(csv_b)
    keys = sorted(set(a.keys()) | set(b.keys()))

    rows_html = []
    for key in keys:
        ra = a.get(key, {})
        rb = b.get(key, {})
        user_url = ra.get("user_input_url") or rb.get("user_input_url") or ""
        garment_url = ra.get("garment_url") or rb.get("garment_url") or ""
        out_a = ra.get("output_url", "")
        out_b = rb.get("output_url", "")
        prompt = rb.get("user_prompt") or ra.get("user_prompt") or ""
        status_a = ra.get("status", "missing")
        status_b = rb.get("status", "missing")

        rows_html.append(
            f"""
            <section class="case">
              <div class="head">
                <div class="id">{html.escape(key)}</div>
                <div class="status">{html.escape(label_a)}: {html.escape(status_a)} | {html.escape(label_b)}: {html.escape(status_b)}</div>
              </div>
              <div class="grid">
                <div class="cell"><h4>User Input</h4>{_img_cell(user_url, "open user")}</div>
                <div class="cell"><h4>{html.escape(label_a)}</h4>{_img_cell(out_a, "open output")}</div>
                <div class="cell"><h4>{html.escape(label_b)}</h4>{_img_cell(out_b, "open output")}</div>
                <div class="cell"><h4>Garment</h4>{_img_cell(garment_url, "open garment")}</div>
              </div>
              <details><summary>Prompt</summary><pre>{html.escape(prompt)}</pre></details>
            </section>
            """
        )

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Try-on Steps Comparison</title>
  <style>
    body {{
      margin: 0; padding: 20px; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #f5f7fb; color: #0f172a;
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 30px; }}
    .meta {{ margin-bottom: 20px; color: #475569; font-size: 14px; }}
    .case {{
      background: #fff; border: 1px solid #dbe5f0; border-radius: 12px; margin: 12px 0; padding: 12px;
    }}
    .head {{
      display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px;
    }}
    .id {{ font-weight: 700; font-size: 18px; }}
    .status {{ font-size: 13px; color: #64748b; }}
    .grid {{
      display: grid; gap: 12px; grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .cell {{
      border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px; background: #f8fafc;
    }}
    .cell h4 {{ margin: 4px 0 8px 0; font-size: 14px; color: #334155; }}
    img {{
      display: block; width: 100%; height: auto; border-radius: 8px; border: 1px solid #dbe4ef; background: #fff;
      min-height: 220px; object-fit: contain;
    }}
    a {{ font-size: 12px; color: #0f766e; text-decoration: none; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; }}
    .missing {{
      min-height: 220px; display: grid; place-items: center; border: 1px dashed #94a3b8; border-radius: 8px; color: #64748b; font-size: 13px;
    }}
    @media (max-width: 1400px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Try-on Steps Comparison</h1>
  <div class="meta">
    A = {html.escape(label_a)} ({html.escape(str(csv_a))})<br/>
    B = {html.escape(label_b)} ({html.escape(str(csv_b))})<br/>
    Cases: {len(keys)}
  </div>
  {''.join(rows_html)}
</body>
</html>"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_doc, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv-a", required=True, type=Path)
    p.add_argument("--csv-b", required=True, type=Path)
    p.add_argument("--label-a", default="steps-12")
    p.add_argument("--label-b", default="steps-28")
    p.add_argument("--output-html", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_report(
        csv_a=args.csv_a,
        csv_b=args.csv_b,
        label_a=args.label_a,
        label_b=args.label_b,
        output_html=args.output_html,
    )
    print(args.output_html)


if __name__ == "__main__":
    main()

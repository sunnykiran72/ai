#!/usr/bin/env python3
import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean


def _to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _status_class(status):
    return "ok" if status == "accepted" else "bad"


def _badge(label, ok):
    cls = "ok" if ok else "bad"
    return f"<span class='badge {cls}'>{html.escape(label)}</span>"


def _card(record, image_url):
    raw_status = (record.get("status") or "").strip()
    raw_result = (record.get("result") or "").strip().lower()
    status = raw_status
    if not status:
        if raw_result == "success":
            status = "accepted"
        elif raw_result in {"error", "exception"}:
            status = "rejected"
        elif raw_result == "open_failed":
            status = "open_failed"
    reason = (record.get("reason") or record.get("error") or "").strip()
    person_score = _to_float(record.get("person_score"))
    face_score = _to_float(record.get("face_score"))
    top_score = _to_float(record.get("top_score"))
    bottom_score = _to_float(record.get("bottom_score"))
    area_ratio = _to_float(record.get("person_area_ratio"))
    extent_ratio = _to_float(record.get("body_extent_ratio"))
    latency_ms = _to_float(record.get("latency_ms"))
    anchor_used = str(record.get("person_anchor_used") or "").strip()
    error = (record.get("error") or "").strip()

    decision = (
        "Accepted: single clear person, visible face, top section, bottom section, body extent passed."
        if status == "accepted"
        else f"Rejected: {reason or 'unknown_reject'}."
    )
    title = html.escape(record.get("filename") or "unknown")
    reason_view = html.escape(reason or "none")
    error_view = html.escape(error or "none")
    image_view = html.escape(image_url)

    return f"""
    <article class="card" data-status="{html.escape(status)}" data-reason="{reason_view}" data-filename="{title.lower()}">
      <div class="head">
        <h3>{title}</h3>
        {_badge(status, status == "accepted")}
      </div>
      <div class="img-wrap">
        <img loading="lazy" src="{image_view}" alt="{title}" />
      </div>
      <p class="decision">{html.escape(decision)}</p>
      <div class="metrics">
        <div><b>Reject reason:</b> {reason_view}</div>
        <div><b>Person score:</b> {person_score if person_score is not None else "n/a"}</div>
        <div><b>Face score:</b> {face_score if face_score is not None else "n/a"}</div>
        <div><b>Top score:</b> {top_score if top_score is not None else "n/a"}</div>
        <div><b>Bottom score:</b> {bottom_score if bottom_score is not None else "n/a"}</div>
        <div><b>Person area ratio:</b> {area_ratio if area_ratio is not None else "n/a"}</div>
        <div><b>Body extent ratio:</b> {extent_ratio if extent_ratio is not None else "n/a"}</div>
        <div><b>Person anchor used:</b> {html.escape(anchor_used or "n/a")}</div>
        <div><b>Latency (ms):</b> {latency_ms if latency_ms is not None else "n/a"}</div>
        <div><b>Error:</b> {error_view}</div>
      </div>
    </article>
    """


def build_report(csv_path: Path, summary_path: Path, images_dir: Path, output_path: Path, report_title: str):
    rows = list(csv.DictReader(csv_path.open()))
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    def norm_status(r):
        s = (r.get("status") or "").strip()
        if s:
            return s
        result = (r.get("result") or "").strip().lower()
        if result == "success":
            return "accepted"
        if result in {"error", "exception"}:
            return "rejected"
        if result == "open_failed":
            return "open_failed"
        return ""

    accepted = [r for r in rows if norm_status(r) == "accepted"]
    rejected = [r for r in rows if norm_status(r) == "rejected"]
    open_failed = [r for r in rows if norm_status(r) == "open_failed"]

    reasons = {}
    for r in rejected:
        reason = (r.get("reason") or "unknown_reject").strip() or "unknown_reject"
        reasons[reason] = reasons.get(reason, 0) + 1

    lat_values = [_to_float(r.get("latency_ms")) for r in rows]
    lat_values = [x for x in lat_values if x is not None]

    cards = []
    missing_count = 0
    for r in rows:
        image_path = images_dir / (r.get("filename") or "")
        if image_path.exists():
            image_url = image_path.as_uri()
        else:
            image_url = ""
            missing_count += 1
        cards.append(_card(r, image_url))

    reasons_items = "\n".join(
        f"<li><b>{html.escape(k)}</b>: {v}</li>" for k, v in sorted(reasons.items(), key=lambda x: x[0])
    ) or "<li>none</li>"

    summary_json = html.escape(json.dumps(summary, indent=2))
    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{html.escape(report_title)}</title>
  <style>
    :root {{
      --bg: #0b1018;
      --panel: #131d2b;
      --line: #27384f;
      --txt: #e7eefb;
      --muted: #9fb1cc;
      --ok: #1db36b;
      --bad: #d64b62;
      --chip: #1d2b3f;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at top right, #1a2a43 0%, var(--bg) 45%); color: var(--txt); font-family: "Segoe UI", Arial, sans-serif; }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    h2 {{ margin: 0 0 8px; font-size: 19px; }}
    .panel {{ background: color-mix(in hsl, var(--panel) 88%, #000); border: 1px solid var(--line); border-radius: 12px; padding: 14px; margin-bottom: 14px; }}
    .muted {{ color: var(--muted); }}
    .mono {{ font-family: Menlo, Consolas, monospace; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; margin-top: 10px; }}
    .stat {{ background: #0d1624; border: 1px solid var(--line); border-radius: 10px; padding: 10px; }}
    .controls {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
    .controls input, .controls select {{ background: #0d1624; color: var(--txt); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; margin-top: 14px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 10px; }}
    .head {{ display: flex; justify-content: space-between; gap: 8px; align-items: start; margin-bottom: 8px; }}
    .head h3 {{ margin: 0; font-size: 14px; line-height: 1.35; word-break: break-word; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; background: var(--chip); border: 1px solid var(--line); }}
    .badge.ok {{ color: #7de8b3; border-color: #2f7352; background: #12271d; }}
    .badge.bad {{ color: #ff9faf; border-color: #7a3341; background: #29131a; }}
    .img-wrap {{ border-radius: 8px; overflow: hidden; border: 1px solid var(--line); background: #0a111b; min-height: 160px; }}
    .img-wrap img {{ width: 100%; height: auto; display: block; }}
    .decision {{ margin: 8px 0; color: #d7e6ff; font-size: 13px; }}
    .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 10px; font-size: 12px; color: #cfdbf1; }}
    .legend li {{ margin: 5px 0; }}
    pre {{ margin: 6px 0 0; background: #09111c; border: 1px solid var(--line); border-radius: 8px; padding: 10px; font-size: 12px; overflow: auto; }}
    .hidden {{ display: none !important; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(report_title)}</h1>
    <p class="muted">Built from: <span class="mono">{html.escape(str(csv_path))}</span></p>

    <section class="panel">
      <h2>Summary</h2>
      <div class="stats">
        <div class="stat"><b>Total rows</b><div>{len(rows)}</div></div>
        <div class="stat"><b>Accepted</b><div>{len(accepted)}</div></div>
        <div class="stat"><b>Rejected</b><div>{len(rejected)}</div></div>
        <div class="stat"><b>Open failed</b><div>{len(open_failed)}</div></div>
        <div class="stat"><b>Missing local image files</b><div>{missing_count}</div></div>
        <div class="stat"><b>Avg latency (ms)</b><div>{round(mean(lat_values), 2) if lat_values else "n/a"}</div></div>
      </div>
      <p><b>Reject reason split</b></p>
      <ul class="legend">{reasons_items}</ul>
      <details><summary>Raw summary JSON</summary><pre>{summary_json}</pre></details>
    </section>

    <section class="panel">
      <h2>Metric Meaning</h2>
      <ul class="legend">
        <li><b>person_score:</b> confidence of selected primary person detection.</li>
        <li><b>face_score:</b> confidence of selected face detection tied to that person.</li>
        <li><b>top_score:</b> confidence of top-section detection (or proxy inferred from face).</li>
        <li><b>bottom_score:</b> confidence of bottom-section detection (or proxy inferred from person bbox).</li>
        <li><b>person_area_ratio:</b> person bbox area divided by full image area.</li>
        <li><b>body_extent_ratio:</b> dominant body span ratio vs image dimension (vertical or horizontal).</li>
        <li><b>latency_ms:</b> detector gate runtime for this image.</li>
        <li><b>Accepted:</b> one-shot detector found single clear person + visible face + top + bottom + sufficient body extent.</li>
        <li><b>Rejected:</b> failed one or more hard checks (for example <span class="mono">no_person</span> or <span class="mono">multiple_people</span>).</li>
      </ul>
    </section>

    <section class="panel">
      <h2>Filter</h2>
      <div class="controls">
        <input id="search" placeholder="Search filename..." />
        <select id="status">
          <option value="all">All statuses</option>
          <option value="accepted">Accepted</option>
          <option value="rejected">Rejected</option>
          <option value="open_failed">Open failed</option>
        </select>
        <select id="reason">
          <option value="all">All reasons</option>
          <option value="no_person">no_person</option>
          <option value="multiple_people">multiple_people</option>
          <option value="face_hidden">face_hidden</option>
          <option value="top_section_not_visible">top_section_not_visible</option>
          <option value="bottom_section_not_visible">bottom_section_not_visible</option>
          <option value="body_not_clear">body_not_clear</option>
        </select>
      </div>
    </section>

    <section class="grid" id="cards">
      {''.join(cards)}
    </section>
  </div>
  <script>
    const search = document.getElementById('search');
    const status = document.getElementById('status');
    const reason = document.getElementById('reason');
    const cards = Array.from(document.querySelectorAll('.card'));

    function applyFilters() {{
      const q = (search.value || '').toLowerCase().trim();
      const st = status.value;
      const rs = reason.value;
      cards.forEach(card => {{
        const filename = card.dataset.filename || '';
        const cStatus = card.dataset.status || '';
        const cReason = card.dataset.reason || '';
        const okSearch = !q || filename.includes(q);
        const okStatus = st === 'all' || cStatus === st;
        const okReason = rs === 'all' || cReason === rs;
        card.classList.toggle('hidden', !(okSearch && okStatus && okReason));
      }});
    }}

    search.addEventListener('input', applyFilters);
    status.addEventListener('change', applyFilters);
    reason.addEventListener('change', applyFilters);
  </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html, encoding="utf-8")
    return {
        "rows": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "open_failed": len(open_failed),
        "missing_local_images": missing_count,
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Build HTML report for DINO gate evaluation CSV.")
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV.")
    parser.add_argument("--summary", required=True, help="Path to summary JSON.")
    parser.add_argument("--images-dir", required=True, help="Directory containing image files.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    parser.add_argument("--title", default="GroundingDINO Gate Report", help="Report title.")
    args = parser.parse_args()

    result = build_report(
        csv_path=Path(args.csv),
        summary_path=Path(args.summary),
        images_dir=Path(args.images_dir),
        output_path=Path(args.out),
        report_title=args.title,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

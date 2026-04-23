#!/usr/bin/env python3
"""
Batch runner for /v1/user-image/prepare.

This script is intentionally independent from the try-on pipeline. It builds the
current combined user-image set, calls the prepare API for each image, stores
raw response data, captures failures, and emits resumable JSONL/CSV plus a
simple HTML review page.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".heif"}
DEFAULT_EXCLUDES = {
    "d59b4890-cd3b-4863-b7d4-243fbea10610.jpg",
    "pexels-_ofarias-g-530112232-16986947.jpg",
    "pexels-aminnaderloei-31897926.jpg",
    "pexels-asio-9688592.jpg",
    "pexels-breno-cardoso-149064345-15728306.jpg",
    "pexels-breno-cardoso-149064345-17110661.jpg",
    "pexels-colordragon-24813292.jpg",
    "pexels-cristian-rojas-10041266.jpg",
    "pexels-cristian-rojas-7535460.jpg",
    "pexels-dan-borges-1755177.jpg",
    "pexels-high-rollick-studio-553091901-32504521.jpg",
    "pexels-israwmx-19899329.jpg",
    "pexels-kushan-perera-95408363-13953101.jpg",
    "pexels-ladoo-6675408.jpg",
    "pexels-mehmet-altintas-392989477-31615337.jpg",
    "pexels-moises-caro-photographer-296475578-14603123.jpg",
    "pexels-moises-caro-photographer-296475578-14944705.jpg",
    "pexels-shazardr-13262282.jpg",
    "pexels-wolfart-34750099.jpg",
}


@dataclass
class ImageCase:
    index: int
    image_name: str
    image_path: str
    width: int
    height: int
    image_format: str
    source_group: str


@dataclass
class PrepareResult:
    index: int
    image_name: str
    image_path: str
    width: int
    height: int
    image_format: str
    source_group: str
    status: str
    error: str
    prepare_latency_s: float
    output_url: str
    prompt_description: str
    worn_types: List[str]
    person_type: str
    age_band: str
    body_build: str
    hair_style: str
    hair_length: str
    hair_color: str
    head_covering: str
    raw_response_path: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "image_name": self.image_name,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "image_format": self.image_format,
            "source_group": self.source_group,
            "status": self.status,
            "error": self.error,
            "prepare_latency_s": self.prepare_latency_s,
            "output_url": self.output_url,
            "prompt_description": self.prompt_description,
            "worn_types": list(self.worn_types or []),
            "person_type": self.person_type,
            "age_band": self.age_band,
            "body_build": self.body_build,
            "hair_style": self.hair_style,
            "hair_length": self.hair_length,
            "hair_color": self.hair_color,
            "head_covering": self.head_covering,
            "raw_response_path": self.raw_response_path,
        }


def _safe_json(resp: requests.Response) -> Dict[str, object]:
    try:
        payload = resp.json()
    except Exception:
        return {"_raw_text": resp.text}
    return payload if isinstance(payload, dict) else {"_payload": payload}


def _normalize_worn_types(values: object) -> List[str]:
    allowed = {"top", "bottom", "outer", "dress"}
    out: List[str] = []
    if values is None:
        return out
    if isinstance(values, list):
        raw_items = values
    else:
        raw_items = [values]
    for item in raw_items:
        token = str(item or "").strip().lower()
        if token in allowed and token not in out:
            out.append(token)
    return out


def _extract_prepare_fields(payload: Dict[str, object]) -> Dict[str, object]:
    data = payload.get("data")
    if not isinstance(data, dict):
        data = payload

    minicpm = data.get("minicpmOutput")
    parsed_json: Dict[str, object] = {}
    if isinstance(minicpm, dict) and isinstance(minicpm.get("parsed_json"), dict):
        parsed_json = dict(minicpm.get("parsed_json") or {})
    user_metadata = parsed_json.get("user_metadata")
    if not isinstance(user_metadata, dict):
        user_metadata = {}

    return {
        "output_url": str(data.get("url") or data.get("tryonImage") or "").strip(),
        "prompt_description": str(data.get("promptDescription") or "").strip(),
        "worn_types": _normalize_worn_types(data.get("wornTypes")),
        "person_type": str(parsed_json.get("person_type") or user_metadata.get("person_label") or "").strip(),
        "age_band": str(parsed_json.get("age_band") or user_metadata.get("age_hint") or "").strip(),
        "body_build": str(parsed_json.get("body_build") or user_metadata.get("body_build") or "").strip(),
        "hair_style": str(parsed_json.get("hair_style") or "").strip(),
        "hair_length": str(parsed_json.get("hair_length") or "").strip(),
        "hair_color": str(parsed_json.get("hair_color") or "").strip(),
        "head_covering": str(parsed_json.get("head_covering") or "").strip(),
    }


def build_cases(
    *,
    repo_root: Path,
    min_new_set_height: int,
    excludes: set[str],
    combined_dir: Optional[Path] = None,
    combined_min_height: int = 0,
) -> List[ImageCase]:
    cases: List[ImageCase] = []
    if combined_dir is not None:
        next_index = 1
        for path in sorted(combined_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    image_format = str(image.format or path.suffix.lstrip(".")).upper()
            except Exception:
                continue
            if combined_min_height and height < combined_min_height:
                continue
            cases.append(
                ImageCase(
                    index=next_index,
                    image_name=path.name,
                    image_path=str(path.resolve()),
                    width=width,
                    height=height,
                    image_format=image_format,
                    source_group=combined_dir.name,
                )
            )
            next_index += 1
        return cases

    datasets = [
        ("users", repo_root / "tryon_dataset" / "users", True),
        ("new_set", repo_root / "tryon_dataset" / "new_set", False),
    ]
    next_index = 1
    for source_group, folder, apply_excludes in datasets:
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            if apply_excludes and path.name in excludes:
                continue
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    image_format = str(image.format or path.suffix.lstrip(".")).upper()
            except Exception:
                continue
            if source_group == "new_set" and height < min_new_set_height:
                continue
            cases.append(
                ImageCase(
                    index=next_index,
                    image_name=path.name,
                    image_path=str(path.resolve()),
                    width=width,
                    height=height,
                    image_format=image_format,
                    source_group=source_group,
                )
            )
            next_index += 1
    return cases


def write_manifest(path: Path, cases: List[ImageCase]) -> None:
    payload = [
        {
            "index": case.index,
            "image_name": case.image_name,
            "image_path": case.image_path,
            "width": case.width,
            "height": case.height,
            "image_format": case.image_format,
            "source_group": case.source_group,
        }
        for case in cases
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_zip(path: Path, cases: List[ImageCase]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for case in cases:
            arcname = f"{case.source_group}/{case.image_name}"
            archive.write(case.image_path, arcname=arcname)


def load_existing(jsonl_path: Path) -> Dict[str, Dict[str, object]]:
    if not jsonl_path.exists():
        return {}
    by_name: Dict[str, Dict[str, object]] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        key = str(row.get("image_name") or "").strip()
        if key:
            by_name[key] = row
    return by_name


def _request_with_retry(fn, *, attempts: int, sleep_seconds: float) -> requests.Response:
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


def write_jsonl(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[PrepareResult]) -> None:
    fields = [
        "index",
        "image_name",
        "image_path",
        "width",
        "height",
        "image_format",
        "source_group",
        "status",
        "error",
        "prepare_latency_s",
        "output_url",
        "prompt_description",
        "worn_types",
        "person_type",
        "age_band",
        "body_build",
        "hair_style",
        "hair_length",
        "hair_color",
        "head_covering",
        "raw_response_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = row.as_dict()
            data["worn_types"] = json.dumps(data["worn_types"], ensure_ascii=False)
            writer.writerow(data)


def build_summary(*, rows: List[PrepareResult], args: argparse.Namespace) -> Dict[str, object]:
    ok = [row for row in rows if row.status == "success"]
    err = [row for row in rows if row.status != "success"]
    summary: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(),
        "base_url": args.base_url,
        "cases_total": len(rows),
        "cases_success": len(ok),
        "cases_error": len(err),
        "settings": {
            "min_new_set_height": args.min_new_set_height,
            "prompt_override": args.prompt_override or "",
            "timeout": args.timeout,
            "retries": args.retries,
        },
    }
    if ok:
        summary["avg_prepare_latency_s"] = round(sum(row.prepare_latency_s for row in ok) / len(ok), 4)
    if err:
        summary["sample_errors"] = [row.error for row in err[:20]]
    return summary


def generate_html(path: Path, rows: List[PrepareResult], summary: Dict[str, object]) -> None:
    cards: List[str] = []
    for row in rows:
        response_label = row.status.title()
        badge_class = "ok" if row.status == "success" else "error"
        image_path = Path(row.image_path)
        image_src = image_path.as_posix()
        raw_path = row.raw_response_path
        cards.append(
            f"""
<section class="card">
  <div class="head">
    <div>
      <div class="eyebrow">Case {row.index:03d}</div>
      <h3>{html.escape(row.image_name)}</h3>
    </div>
    <span class="badge {badge_class}">{html.escape(response_label)}</span>
  </div>
  <div class="grid">
    <figure>
      <figcaption>Source</figcaption>
      <img src="file://{html.escape(image_src)}" alt="source image" loading="lazy" />
    </figure>
    <div class="meta">
      <div><strong>Source Group:</strong> {html.escape(row.source_group)}</div>
      <div><strong>Size:</strong> {row.width}x{row.height}</div>
      <div><strong>Format:</strong> {html.escape(row.image_format)}</div>
      <div><strong>Latency:</strong> {row.prepare_latency_s:.2f}s</div>
      <div><strong>Output URL:</strong> <pre>{html.escape(row.output_url or "")}</pre></div>
      <div><strong>Prompt:</strong> <pre>{html.escape(row.prompt_description or "")}</pre></div>
      <div><strong>Worn Types:</strong> <pre>{html.escape(', '.join(row.worn_types or []))}</pre></div>
      <div><strong>Person Type:</strong> <pre>{html.escape(row.person_type or "")}</pre></div>
      <div><strong>Age Band:</strong> <pre>{html.escape(row.age_band or "")}</pre></div>
      <div><strong>Body Build:</strong> <pre>{html.escape(row.body_build or "")}</pre></div>
      <div><strong>Error:</strong> <pre>{html.escape(row.error or "")}</pre></div>
      <div><strong>Raw JSON:</strong> <pre>{html.escape(raw_path)}</pre></div>
    </div>
  </div>
</section>
"""
        )

    summary_json = html.escape(json.dumps(summary, indent=2))
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>User Prepare Batch</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f6f5f1; color: #161616; margin: 0; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    .hero, .card {{ background: #fff; border: 1px solid #ddd8cf; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }}
    .hero {{ padding: 20px; margin-bottom: 18px; }}
    .card {{ padding: 16px; margin-bottom: 18px; }}
    .eyebrow {{ font-size: 11px; text-transform: uppercase; color: #6d665e; margin-bottom: 6px; }}
    h1, h3 {{ margin: 0; }}
    .chips {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }}
    .chip {{ border: 1px solid #e5e0d7; border-radius: 14px; padding: 12px; }}
    .chip .k {{ font-size: 11px; text-transform: uppercase; color: #6d665e; display: block; margin-bottom: 4px; }}
    .chip .v {{ font-size: 20px; font-weight: 700; }}
    .head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }}
    .badge {{ padding: 6px 10px; border-radius: 999px; font-size: 11px; text-transform: uppercase; font-weight: 700; }}
    .badge.ok {{ background: #ecf8f2; color: #0c7a43; }}
    .badge.error {{ background: #fff1f1; color: #b3261e; }}
    .grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 16px; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 11px; text-transform: uppercase; color: #6d665e; margin-bottom: 8px; }}
    img {{ width: 100%; height: 480px; object-fit: contain; border: 1px solid #e5e0d7; border-radius: 14px; background: #fafafa; }}
    .meta pre {{ white-space: pre-wrap; word-break: break-word; background: #f7f6f2; padding: 8px; border-radius: 10px; }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} .chips {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Batch</div>
      <h1>User Image Prepare Report</h1>
      <div class="chips">
        <div class="chip"><span class="k">Total</span><span class="v">{summary.get("cases_total", 0)}</span></div>
        <div class="chip"><span class="k">Success</span><span class="v">{summary.get("cases_success", 0)}</span></div>
        <div class="chip"><span class="k">Error</span><span class="v">{summary.get("cases_error", 0)}</span></div>
        <div class="chip"><span class="k">Avg Latency</span><span class="v">{summary.get("avg_prepare_latency_s", "n/a")}</span></div>
      </div>
      <details style="margin-top:16px;">
        <summary>Summary JSON</summary>
        <pre>{summary_json}</pre>
      </details>
    </section>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch /v1/user-image/prepare runner")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt-override", default="")
    parser.add_argument("--combined-dir", default="", help="Optional folder of images to use directly instead of rebuilding users + new_set")
    parser.add_argument("--combined-min-height", type=int, default=0, help="Optional height filter when --combined-dir is used")
    parser.add_argument("--min-new-set-height", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-zip", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or repo_root / "tmp" / f"user_prepare_batch_{ts}").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(
        repo_root=repo_root,
        min_new_set_height=int(args.min_new_set_height),
        excludes=set(DEFAULT_EXCLUDES),
        combined_dir=Path(args.combined_dir).expanduser().resolve() if args.combined_dir else None,
        combined_min_height=int(args.combined_min_height),
    )
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest_path, cases)
    if args.write_zip:
        write_zip(out_dir / "images.zip", cases)

    jsonl_path = out_dir / "results.jsonl"
    csv_path = out_dir / "results.csv"
    summary_path = out_dir / "summary.json"
    html_path = out_dir / "report.html"

    existing = load_existing(jsonl_path) if args.resume else {}
    rows: List[PrepareResult] = []
    for payload in existing.values():
        rows.append(
            PrepareResult(
                index=int(payload.get("index") or 0),
                image_name=str(payload.get("image_name") or ""),
                image_path=str(payload.get("image_path") or ""),
                width=int(payload.get("width") or 0),
                height=int(payload.get("height") or 0),
                image_format=str(payload.get("image_format") or ""),
                source_group=str(payload.get("source_group") or ""),
                status=str(payload.get("status") or "error"),
                error=str(payload.get("error") or ""),
                prepare_latency_s=float(payload.get("prepare_latency_s") or 0.0),
                output_url=str(payload.get("output_url") or ""),
                prompt_description=str(payload.get("prompt_description") or ""),
                worn_types=_normalize_worn_types(payload.get("worn_types")),
                person_type=str(payload.get("person_type") or ""),
                age_band=str(payload.get("age_band") or ""),
                body_build=str(payload.get("body_build") or ""),
                hair_style=str(payload.get("hair_style") or ""),
                hair_length=str(payload.get("hair_length") or ""),
                hair_color=str(payload.get("hair_color") or ""),
                head_covering=str(payload.get("head_covering") or ""),
                raw_response_path=str(payload.get("raw_response_path") or ""),
            )
        )

    done = {row.image_name for row in rows}
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    endpoint = args.base_url.rstrip("/") + "/v1/user-image/prepare"

    print(f"output_dir={out_dir}")
    print(f"cases_total={len(cases)}")
    print(f"endpoint={endpoint}")

    for case in cases:
        if case.image_name in done:
            continue
        started = time.time()
        raw_payload: Dict[str, object] = {}
        try:
            with open(case.image_path, "rb") as handle:
                def _call() -> requests.Response:
                    data = {}
                    if args.prompt_override:
                        data["prompt_override"] = args.prompt_override
                    return session.post(
                        endpoint,
                        files={"file": (case.image_name, handle, "application/octet-stream")},
                        data=data,
                        timeout=args.timeout,
                    )

                response = _request_with_retry(_call, attempts=args.retries, sleep_seconds=args.retry_sleep)
            latency = time.time() - started
            raw_payload = _safe_json(response)
            raw_path = raw_dir / f"{case.index:03d}_{case.image_name}.json"
            raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")

            if response.status_code != 200:
                raise RuntimeError(f"prepare_failed status={response.status_code}")

            fields = _extract_prepare_fields(raw_payload)
            if not fields["output_url"]:
                raise RuntimeError("prepare_missing_output_url")

            row = PrepareResult(
                index=case.index,
                image_name=case.image_name,
                image_path=case.image_path,
                width=case.width,
                height=case.height,
                image_format=case.image_format,
                source_group=case.source_group,
                status="success",
                error="",
                prepare_latency_s=latency,
                output_url=str(fields["output_url"]),
                prompt_description=str(fields["prompt_description"]),
                worn_types=list(fields["worn_types"]),
                person_type=str(fields["person_type"]),
                age_band=str(fields["age_band"]),
                body_build=str(fields["body_build"]),
                hair_style=str(fields["hair_style"]),
                hair_length=str(fields["hair_length"]),
                hair_color=str(fields["hair_color"]),
                head_covering=str(fields["head_covering"]),
                raw_response_path=str(raw_path),
            )
        except Exception as exc:
            latency = time.time() - started
            raw_path = raw_dir / f"{case.index:03d}_{case.image_name}.json"
            if raw_payload:
                raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            row = PrepareResult(
                index=case.index,
                image_name=case.image_name,
                image_path=case.image_path,
                width=case.width,
                height=case.height,
                image_format=case.image_format,
                source_group=case.source_group,
                status="error",
                error=str(exc),
                prepare_latency_s=latency,
                output_url="",
                prompt_description="",
                worn_types=[],
                person_type="",
                age_band="",
                body_build="",
                hair_style="",
                hair_length="",
                hair_color="",
                head_covering="",
                raw_response_path=str(raw_path),
            )

        rows.append(row)
        done.add(case.image_name)
        write_jsonl(jsonl_path, row.as_dict())
        rows.sort(key=lambda item: item.index)
        write_csv(csv_path, rows)
        summary = build_summary(rows=rows, args=args)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        generate_html(html_path, rows, summary)
        print(f"[{row.index:03d}] {row.image_name} -> {row.status} ({row.prepare_latency_s:.2f}s)")

    rows.sort(key=lambda item: item.index)
    write_csv(csv_path, rows)
    summary = build_summary(rows=rows, args=args)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generate_html(html_path, rows, summary)

    print(f"manifest_json={manifest_path}")
    print(f"results_jsonl={jsonl_path}")
    print(f"results_csv={csv_path}")
    print(f"summary_json={summary_path}")
    print(f"report_html={html_path}")
    if args.write_zip:
        print(f"images_zip={out_dir / 'images.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

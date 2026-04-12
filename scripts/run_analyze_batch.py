#!/usr/bin/env python3
"""
Batch /analyze tester for local garment fixture folders.

This script is designed for prompt and extraction review on curated garment
images. It can:

1. Probe an image with /analyze to inspect detected items.
2. Auto-pick the requested garment type from multi-item responses.
3. Re-run /analyze on the selected crop when selection is needed.
4. Save raw responses, prompt fields, timings, and extracted images for review.

Typical usage:

  python3 scripts/run_analyze_batch.py \
    --base-url http://127.0.0.1:8000 \
    --input-dir test_garments \
    --types top dress
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".heic", ".heif"}
SUPPORTED_TARGET_TYPES = ("top", "bottom", "dress", "outer")


def normalize_type(value: Optional[str]) -> str:
    raw = " ".join(str(value or "").split()).strip().lower()
    if raw in {"top", "shirt", "blouse", "tee", "t-shirt", "tshirt", "bra", "bralette", "corset"}:
        return "top"
    if raw in {"bottom", "pants", "trousers", "jeans", "shorts", "skirt"}:
        return "bottom"
    if raw in {"dress", "gown", "one-piece", "one piece"}:
        return "dress"
    if raw in {"outer", "outerwear", "coat", "jacket", "blazer"}:
        return "outer"
    if "dress" in raw or "gown" in raw:
        return "dress"
    if any(token in raw for token in ("top", "shirt", "blouse", "bra", "bralette", "corset", "cami")):
        return "top"
    if any(token in raw for token in ("pant", "trouser", "jean", "short", "skirt", "bottom")):
        return "bottom"
    if any(token in raw for token in ("coat", "jacket", "blazer", "outer")):
        return "outer"
    return raw


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "case"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_image_paths(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_multipart_response(body: bytes, content_type: str) -> Dict[str, Any]:
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    metadata: Dict[str, Any] | None = None
    binary_parts: list[dict[str, Any]] = []

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        filename_match = re.search(r'filename="([^"]+)"', disposition)
        name = name_match.group(1) if name_match else ""
        filename = filename_match.group(1) if filename_match else ""
        payload = part.get_payload(decode=True) or b""
        part_type = str(part.get_content_type() or "application/octet-stream")

        if name == "metadata":
            try:
                metadata = json.loads(payload.decode("utf-8"))
            except Exception:
                metadata = {"raw_text": payload.decode("utf-8", errors="replace")}
            continue

        binary_parts.append(
            {
                "name": name or "file",
                "filename": filename or (name or "file.bin"),
                "content_type": part_type,
                "bytes": payload,
            }
        )

    return {
        "metadata": metadata,
        "binary_parts": binary_parts,
    }


def parse_http_response(resp: requests.Response) -> Dict[str, Any]:
    content_type = resp.headers.get("content-type", "")
    parsed: Dict[str, Any] = {
        "status_code": resp.status_code,
        "content_type": content_type,
        "headers": dict(resp.headers),
    }

    if content_type.startswith("multipart/form-data"):
        multipart = parse_multipart_response(resp.content, content_type)
        parsed["kind"] = "multipart"
        parsed.update(multipart)
        return parsed

    if content_type.startswith("application/json"):
        parsed["kind"] = "json"
        parsed["metadata"] = resp.json()
        parsed["binary_parts"] = []
        return parsed

    parsed["kind"] = "text"
    parsed["metadata"] = {"raw_text": resp.text}
    parsed["binary_parts"] = []
    return parsed


def extract_payload_data(parsed: Dict[str, Any]) -> Dict[str, Any]:
    metadata = parsed.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    if isinstance(metadata.get("data"), dict):
        return dict(metadata["data"])
    return dict(metadata)


def selection_required(parsed: Dict[str, Any]) -> bool:
    data = extract_payload_data(parsed)
    return bool(data.get("selection_required")) or str(data.get("status", "")).lower() == "selection_required"


def candidate_score(item: Dict[str, Any]) -> tuple[float, float]:
    confidence = item.get("confidence") if isinstance(item.get("confidence"), dict) else {}
    hybrid = float(confidence.get("hybrid") or 0.0)
    florence = float(confidence.get("florence_type") or 0.0)
    yolo = float(confidence.get("yolo") or 0.0)
    area = 0.0
    crop = item.get("crop") if isinstance(item.get("crop"), dict) else {}
    width = int(crop.get("width") or 0)
    height = int(crop.get("height") or 0)
    if width > 0 and height > 0:
        area = float(width * height)
    return (hybrid or florence or yolo, area)


def choose_item_for_type(items: list[Dict[str, Any]], target_type: str) -> Optional[Dict[str, Any]]:
    desired = normalize_type(target_type)
    matching = [item for item in items if normalize_type(item.get("type")) == desired]
    if not matching:
        return None
    matching.sort(key=candidate_score, reverse=True)
    return dict(matching[0])


def derive_target_types_from_probe(parsed: Dict[str, Any], allowed_types: Iterable[str]) -> list[str]:
    allowed = {normalize_type(value) for value in allowed_types}
    data = extract_payload_data(parsed)
    discovered: list[str] = []

    def add(candidate: Any) -> None:
        normalized = normalize_type(candidate)
        if normalized in allowed and normalized not in discovered:
            discovered.append(normalized)

    items = data.get("items") if isinstance(data.get("items"), list) else []
    for item in items:
        if isinstance(item, dict):
            add(item.get("type"))
            add(item.get("garment_type"))
            add(item.get("style"))

    selected_item = data.get("selected_item") if isinstance(data.get("selected_item"), dict) else {}
    add(data.get("selected_type"))
    add(data.get("garment_type"))
    add(data.get("clothing_type"))
    add(data.get("primary_category_key"))
    add(selected_item.get("type"))
    add(selected_item.get("garment_type"))
    add(selected_item.get("style"))
    add(selected_item.get("primary_category_key"))

    return discovered


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_binary_parts(case_dir: Path, binary_parts: list[Dict[str, Any]]) -> dict[str, str]:
    ensure_dir(case_dir)
    saved: dict[str, str] = {}
    for index, part in enumerate(binary_parts):
        filename = slugify(str(part.get("filename") or f"part_{index}.bin"))
        if "." not in filename:
            filename = f"{filename}.bin"
        path = case_dir / filename
        path.write_bytes(bytes(part.get("bytes") or b""))
        saved[str(part.get("name") or f"part_{index}")] = str(path)
    return saved


def maybe_download_file(url: str, destination: Path, timeout: int, headers: Dict[str, str]) -> Optional[Path]:
    clean_url = str(url or "").strip()
    if not clean_url:
        return None
    resp = requests.get(clean_url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    destination.write_bytes(resp.content)
    return destination


def analyze_file(
    *,
    base_url: str,
    image_path: Path,
    garment_type: Optional[str],
    timeout: int,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    with image_path.open("rb") as handle:
        files = {"file": (image_path.name, handle, "application/octet-stream")}
        data: Dict[str, str] = {}
        if garment_type:
            data["type"] = garment_type
        started = time.perf_counter()
        resp = requests.post(f"{base_url}/analyze", files=files, data=data, headers=headers, timeout=timeout)
    parsed = parse_http_response(resp)
    parsed["latency_s"] = round(time.perf_counter() - started, 4)
    return parsed


def write_prompt_fields(case_dir: Path, selected_item: Dict[str, Any], payload_data: Dict[str, Any]) -> None:
    fields = {
        "prompt_description.txt": str(
            selected_item.get("promptDescription")
            or payload_data.get("promptDescription")
            or ""
        ).strip(),
        "minicpm_description.txt": str(selected_item.get("minicpm_description") or "").strip(),
        "base_garment_prompt.txt": str(selected_item.get("baseGarmentPrompt") or "").strip(),
        "prompt_source.txt": str(selected_item.get("promptDescriptionSource") or "").strip(),
    }
    for filename, text in fields.items():
        if text:
            (case_dir / filename).write_text(text + "\n", encoding="utf-8")


def build_summary_row(case_id: str, target_type: str, source_image: Path, probe: Dict[str, Any], final: Dict[str, Any]) -> Dict[str, Any]:
    probe_data = extract_payload_data(probe)
    final_data = extract_payload_data(final)
    selected_item = final_data.get("selected_item") if isinstance(final_data.get("selected_item"), dict) else {}
    return {
        "case_id": case_id,
        "target_type": target_type,
        "source_image": str(source_image),
        "probe_status_code": probe.get("status_code"),
        "probe_selection_required": selection_required(probe),
        "final_status_code": final.get("status_code"),
        "final_selection_required": selection_required(final),
        "selected_type": str(final_data.get("selected_type") or selected_item.get("type") or ""),
        "prompt_description": str(
            selected_item.get("promptDescription")
            or final_data.get("promptDescription")
            or ""
        ).strip(),
        "output_image_url": str(
            final_data.get("output_image_url")
            or selected_item.get("outputImage")
            or selected_item.get("url")
            or ""
        ).strip(),
        "probe_latency_s": probe.get("latency_s"),
        "final_latency_s": final.get("latency_s"),
    }


def build_markdown_summary(rows: list[Dict[str, Any]]) -> str:
    lines = [
        "# Analyze Batch Summary",
        "",
        "| Case | Target | Final Type | Probe Sel | Final Sel | Final Status | Prompt |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        prompt = str(row.get("prompt_description") or "").replace("\n", " ").strip()
        if len(prompt) > 90:
            prompt = prompt[:87] + "..."
        lines.append(
            "| {case_id} | {target_type} | {selected_type} | {probe_selection_required} | {final_selection_required} | {final_status_code} | {prompt} |".format(
                case_id=row.get("case_id", ""),
                target_type=row.get("target_type", ""),
                selected_type=row.get("selected_type", ""),
                probe_selection_required=row.get("probe_selection_required", ""),
                final_selection_required=row.get("final_selection_required", ""),
                final_status_code=row.get("final_status_code", ""),
                prompt=prompt,
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_html_report(rows: list[Dict[str, Any]], output_dir: Path) -> str:
    cards: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        target_type = str(row.get("target_type") or "")
        selected_type = str(row.get("selected_type") or "")
        case_dir = output_dir / target_type / case_id

        source_image = case_dir / Path(str(row.get("source_image") or "")).name
        selected_input = case_dir / "selected_input.png"
        output_image = case_dir / "output_image.png"
        prompt_description = case_dir / "prompt_description.txt"
        minicpm_description = case_dir / "minicpm_description.txt"
        base_prompt = case_dir / "base_garment_prompt.txt"
        case_summary = case_dir / "case_summary.json"

        def rel(path: Path) -> str:
            return path.relative_to(output_dir).as_posix()

        def maybe_block(label: str, path: Path) -> str:
            if not path.exists():
                return ""
            return (
                f'<div class="text-block"><div class="label">{label}</div>'
                f"<pre>{path.read_text(encoding='utf-8', errors='replace')}</pre></div>"
            )

        def maybe_img(label: str, path: Path) -> str:
            if not path.exists():
                return ""
            return (
                f'<div class="image-block"><div class="label">{label}</div>'
                f'<img src="{rel(path)}" alt="{label}"></div>'
            )

        summary_text = ""
        if case_summary.exists():
            summary_text = json.dumps(json.loads(case_summary.read_text(encoding="utf-8")), indent=2)

        cards.append(
            f"""
            <section class="card">
              <div class="card-header">
                <h2>{case_id}</h2>
                <div class="meta">
                  <span>target: {target_type}</span>
                  <span>selected: {selected_type}</span>
                  <span>status: {row.get("final_status_code", "")}</span>
                </div>
              </div>
              <div class="grid images">
                {maybe_img("Source Image", source_image)}
                {maybe_img("Selected Input Crop", selected_input)}
                {maybe_img("Output Image", output_image)}
              </div>
              <div class="grid text">
                {maybe_block("Prompt Description", prompt_description)}
                {maybe_block("MiniCPM Description", minicpm_description)}
                {maybe_block("Base Garment Prompt", base_prompt)}
                <div class="text-block"><div class="label">Case Summary</div><pre>{summary_text}</pre></div>
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analyze Batch Report</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --card: #fffdf8;
      --ink: #181512;
      --muted: #6f655c;
      --line: #dfd6ca;
      --accent: #8f3d28;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(143,61,40,0.08), transparent 28%),
        linear-gradient(180deg, #f8f3ec 0%, var(--bg) 100%);
    }}
    .page {{
      width: min(1500px, calc(100vw - 40px));
      margin: 24px auto 48px;
    }}
    .hero {{
      padding: 28px 30px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,248,241,0.92));
      border-radius: 22px;
      box-shadow: 0 18px 50px rgba(60, 39, 24, 0.08);
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      margin-bottom: 20px;
      box-shadow: 0 14px 36px rgba(28, 19, 12, 0.06);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}
    .card-header h2 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.1;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .images {{
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      margin-bottom: 18px;
    }}
    .text {{
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .image-block, .text-block {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.75);
      padding: 14px;
    }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--accent);
      margin-bottom: 10px;
      font-weight: 700;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid rgba(0,0,0,0.05);
      background: white;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #2c241d;
    }}
    @media (max-width: 720px) {{
      .page {{ width: min(100vw - 20px, 1500px); }}
      .hero {{ padding: 20px; }}
      .card {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <h1>Analyze Batch Report</h1>
      <p>Review of /analyze runs including source image, selected crop, generated prompt text, and final extracted garment output.</p>
    </header>
    {"".join(cards)}
  </main>
</body>
</html>
"""


def run_case(
    *,
    base_url: str,
    source_image: Path,
    target_type: str,
    case_dir: Path,
    timeout: int,
    headers: Dict[str, str],
    skip_probe: bool,
    probe_parsed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_dir(case_dir)
    shutil.copy2(source_image, case_dir / source_image.name)

    if probe_parsed is not None:
        write_json(case_dir / "probe_response.json", probe_parsed.get("metadata"))
        probe_saved = save_binary_parts(case_dir / "probe_parts", probe_parsed.get("binary_parts", []))
        if probe_saved:
            write_json(case_dir / "probe_parts.json", probe_saved)
        probe_data = extract_payload_data(probe_parsed)
        items = probe_data.get("items") if isinstance(probe_data.get("items"), list) else []
        chosen_item = choose_item_for_type(items, target_type)
        if chosen_item and chosen_item.get("url"):
            selected_input = case_dir / "selected_input.png"
            maybe_download_file(str(chosen_item.get("url")), selected_input, timeout, headers)
            write_json(case_dir / "selected_probe_item.json", chosen_item)
            working_input = selected_input
        else:
            working_input = source_image
    elif skip_probe:
        probe_parsed = {"status_code": 0, "latency_s": 0.0, "metadata": {}, "binary_parts": []}
        working_input = source_image
        chosen_item = None
    else:
        probe_parsed = analyze_file(
            base_url=base_url,
            image_path=source_image,
            garment_type=None,
            timeout=timeout,
            headers=headers,
        )
        write_json(case_dir / "probe_response.json", probe_parsed.get("metadata"))
        probe_saved = save_binary_parts(case_dir / "probe_parts", probe_parsed.get("binary_parts", []))
        if probe_saved:
            write_json(case_dir / "probe_parts.json", probe_saved)

        probe_data = extract_payload_data(probe_parsed)
        items = probe_data.get("items") if isinstance(probe_data.get("items"), list) else []
        chosen_item = choose_item_for_type(items, target_type)
        if chosen_item and chosen_item.get("url"):
            selected_input = case_dir / "selected_input.png"
            maybe_download_file(str(chosen_item.get("url")), selected_input, timeout, headers)
            write_json(case_dir / "selected_probe_item.json", chosen_item)
            working_input = selected_input
        else:
            working_input = source_image

    final_parsed = analyze_file(
        base_url=base_url,
        image_path=working_input,
        garment_type=target_type,
        timeout=timeout,
        headers=headers,
    )
    write_json(case_dir / "final_response.json", final_parsed.get("metadata"))

    saved_binaries = save_binary_parts(case_dir, final_parsed.get("binary_parts", []))
    if saved_binaries:
        write_json(case_dir / "binary_parts.json", saved_binaries)

    final_data = extract_payload_data(final_parsed)
    selected_item = final_data.get("selected_item") if isinstance(final_data.get("selected_item"), dict) else {}
    write_prompt_fields(case_dir, selected_item, final_data)

    output_url = str(
        final_data.get("output_image_url")
        or selected_item.get("outputImage")
        or selected_item.get("url")
        or ""
    ).strip()
    if output_url:
        try:
            maybe_download_file(output_url, case_dir / "output_image.png", timeout, headers)
        except Exception as exc:
            write_json(case_dir / "output_image_download_error.json", {"url": output_url, "error": str(exc)})

    summary = build_summary_row(case_dir.name, target_type, source_image, probe_parsed, final_parsed)
    details = {
        "summary": summary,
        "source_image": str(source_image),
        "working_input": str(working_input),
        "chosen_probe_item": chosen_item,
        "probe_response_path": str(case_dir / "probe_response.json"),
        "final_response_path": str(case_dir / "final_response.json"),
    }
    write_json(case_dir / "case_summary.json", details)
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch tester for /analyze garment fixtures.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Glamify API base URL.")
    parser.add_argument(
        "--input-dir",
        default="/Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/test_garments",
        help="Directory of local garment fixture images.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["top", "dress"],
        choices=["top", "bottom", "dress", "outer"],
        help="Requested garment types to test.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to tmp/analyze-batch/<timestamp>.",
    )
    parser.add_argument("--timeout", type=int, default=240, help="HTTP timeout in seconds.")
    parser.add_argument("--auth-token", default="", help="Bearer token. Optional.")
    parser.add_argument("--skip-probe", action="store_true", help="Skip the untyped probe step.")
    parser.add_argument(
        "--auto-detect-types",
        action="store_true",
        help="Probe each image once and run only the garment types detected in that image.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of input images to process.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        Path("/Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/tmp") / "analyze-batch" / timestamp
    )
    ensure_dir(output_dir)

    headers: Dict[str, str] = {}
    if args.auth_token:
        headers["Authorization"] = f"Bearer {args.auth_token}"

    images = list(iter_image_paths(input_dir))
    if args.limit > 0:
        images = images[: args.limit]

    all_rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []

    for image_path in images:
        image_probe: Optional[Dict[str, Any]] = None
        target_types = list(args.types)

        if args.auto_detect_types:
            image_probe = analyze_file(
                base_url=args.base_url.rstrip("/"),
                image_path=image_path,
                garment_type=None,
                timeout=args.timeout,
                headers=headers,
            )
            target_types = derive_target_types_from_probe(image_probe, args.types)
            if not target_types:
                failure = {
                    "case_id": f"{slugify(image_path.stem)}__no-detected-type",
                    "image": str(image_path),
                    "target_type": "",
                    "error": "No supported garment type detected from probe response",
                }
                failures.append(failure)
                continue

        for target_type in target_types:
            case_id = f"{slugify(image_path.stem)}__{target_type}"
            case_dir = ensure_dir(output_dir / target_type / case_id)
            try:
                details = run_case(
                    base_url=args.base_url.rstrip("/"),
                    source_image=image_path,
                    target_type=target_type,
                    case_dir=case_dir,
                    timeout=args.timeout,
                    headers=headers,
                    skip_probe=bool(args.skip_probe),
                    probe_parsed=image_probe,
                )
                all_rows.append(details["summary"])
            except Exception as exc:
                failure = {
                    "case_id": case_id,
                    "image": str(image_path),
                    "target_type": target_type,
                    "error": str(exc),
                }
                failures.append(failure)
                write_json(case_dir / "error.json", failure)

    summary = {
        "base_url": args.base_url.rstrip("/"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "types": list(args.types),
        "auto_detect_types": bool(args.auto_detect_types),
        "cases": all_rows,
        "failures": failures,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "SUMMARY.md").write_text(build_markdown_summary(all_rows), encoding="utf-8")
    (output_dir / "report.html").write_text(build_html_report(all_rows, output_dir), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

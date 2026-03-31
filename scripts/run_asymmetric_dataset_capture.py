#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from pathlib import Path
from typing import Dict, Optional

import requests


def infer_garment_type(path: Path) -> Optional[str]:
    name = path.stem.lower()
    if "dress" in name:
        return "dress"
    return "top"


def parse_multipart_json(resp: requests.Response) -> Dict[str, object]:
    ctype = resp.headers.get("content-type", "")
    match = re.search(r"boundary=([^;]+)", ctype)
    if not match:
        try:
            return resp.json()
        except Exception:
            return {"raw_text": resp.text[:20000]}

    boundary = match.group(1).strip().strip('"').encode("utf-8")
    marker = b"--" + boundary
    for raw in resp.content.split(marker):
        part = raw.strip()
        if not part or part == b"--":
            continue
        if part.startswith(b"--"):
            part = part[2:]
        if part.startswith(b"\r\n"):
            part = part[2:]
        head, sep, body = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers = head.decode("utf-8", errors="ignore").split("\r\n")
        disp = next((value for value in headers if value.lower().startswith("content-disposition:")), "")
        if 'name="metadata"' not in disp:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]
        try:
            return json.loads(body.decode("utf-8", errors="ignore"))
        except Exception:
            return {"raw_text": body.decode("utf-8", errors="ignore")[:20000]}
    return {"raw_text": resp.text[:20000]}


def run_case(
    *,
    base_url: str,
    token: str,
    image_path: Path,
    garment_type: str,
) -> Dict[str, object]:
    headers = {}
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"

    with image_path.open("rb") as handle:
        content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        files = {"file": (image_path.name, handle, content_type)}
        data = {"type": garment_type}
        resp = requests.post(f"{base_url.rstrip('/')}/analyze", headers=headers, files=files, data=data, timeout=3600)

    payload = parse_multipart_json(resp)
    data_block = payload.get("data") if isinstance(payload, dict) else {}
    selected_item = data_block.get("selected_item") if isinstance(data_block, dict) else {}
    return {
        "file": image_path.name,
        "garment_type": garment_type,
        "status_code": resp.status_code,
        "result": data_block.get("result") if isinstance(data_block, dict) else None,
        "selected_type": data_block.get("selected_type") if isinstance(data_block, dict) else None,
        "debug_artifact_capture_path": (
            selected_item.get("debugArtifactCapturePath") if isinstance(selected_item, dict) else None
        ),
        "reason_codes": data_block.get("reason_codes") if isinstance(data_block, dict) else None,
        "response": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run /analyze over asymmetric_dataset and capture top/dress artifacts.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Analyze API base URL")
    parser.add_argument("--token", default="", help="Bearer token or raw JWT")
    parser.add_argument("--dataset-dir", default="asymmetric_dataset", help="Dataset directory")
    parser.add_argument(
        "--output",
        default="asymmetric_dataset/debug_capture/run_summary.json",
        help="Path to write the run summary JSON",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    image_paths = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    results = []
    for image_path in image_paths:
        garment_type = infer_garment_type(image_path)
        if garment_type not in {"top", "dress"}:
            continue
        print(f"[run] {image_path.name} -> type={garment_type}")
        results.append(
            run_case(
                base_url=args.base_url,
                token=args.token,
                image_path=image_path,
                garment_type=garment_type,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote summary to {output_path}")


if __name__ == "__main__":
    main()

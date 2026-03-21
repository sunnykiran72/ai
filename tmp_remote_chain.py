import json
import os
import re
import sys
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8000"
TOKEN = os.environ.get("ANALYZE_TOKEN", "").strip()
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

ANALYZE_IMAGE = Path("/tmp/analyze_input.png")
USER_IMAGE = Path("/tmp/userprep_input.png")


def parse_multipart(resp):
    ctype = resp.headers.get("content-type", "")
    m = re.search(r'boundary=([^;]+)', ctype)
    if not m:
        return None
    boundary = m.group(1).strip().strip('"').encode("utf-8")
    marker = b"--" + boundary
    parts = {}
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
        if body.endswith(b"\r\n"):
            body = body[:-2]
        headers = head.decode("utf-8", errors="ignore").split("\r\n")
        disp = next((h for h in headers if h.lower().startswith("content-disposition:")), "")
        nm = re.search(r'name="([^"]+)"', disp)
        fn = re.search(r'filename="([^"]+)"', disp)
        if not nm:
            continue
        parts[nm.group(1)] = {
            "filename": fn.group(1) if fn else None,
            "bytes": body,
            "headers": headers,
        }
    return parts


def do_analyze(path: Path):
    with path.open("rb") as f:
        r = requests.post(
            f"{BASE}/analyze",
            headers=HEADERS,
            files={"file": (path.name, f, "image/png")},
            data={"type": "top"},
            timeout=3600,
        )
    print("ANALYZE_STATUS", r.status_code)
    print("ANALYZE_CTYPE", r.headers.get("content-type", ""))
    parts = parse_multipart(r)
    if parts is None:
        print(r.text[:20000])
        return None
    md = json.loads(parts["metadata"]["bytes"].decode("utf-8", errors="ignore"))
    print("ANALYZE_METADATA")
    print(json.dumps(md, indent=2)[:40000])
    return md


def do_prepare(path: Path):
    with path.open("rb") as f:
        r = requests.post(
            f"{BASE}/v1/user-image/prepare",
            files={"file": (path.name, f, "image/png")},
            timeout=1800,
        )
    print("PREPARE_STATUS", r.status_code)
    body = r.json()
    print("PREPARE_JSON")
    print(json.dumps(body, indent=2)[:40000])
    return body


def do_tryon(user_url, user_prompt, garment_url, garment_prompt, garment_type):
    payload = {
        "products": [
            {
                "image": garment_url,
                "promptDescription": garment_prompt,
                "targetType": garment_type,
            }
        ],
        "user_image": {
            "tryonImage": user_url,
            "promptDescription": user_prompt,
        },
        "steps": 20,
        "seed": 42,
    }
    r = requests.post(f"{BASE}/v1/flux2/tryon", json=payload, timeout=3600)
    print("TRYON_STATUS", r.status_code)
    body = r.json()
    print("TRYON_JSON")
    print(json.dumps(body, indent=2)[:40000])
    return body


def main():
    if not ANALYZE_IMAGE.exists():
        raise FileNotFoundError(ANALYZE_IMAGE)
    if not USER_IMAGE.exists():
        raise FileNotFoundError(USER_IMAGE)
    if not HEADERS["Authorization"].strip():
        raise RuntimeError("ANALYZE_TOKEN is required")

    ana = do_analyze(ANALYZE_IMAGE)
    prep = do_prepare(USER_IMAGE)
    if not ana or not prep:
        sys.exit(1)

    ana_data = ana.get("data", {}) if isinstance(ana, dict) else {}
    sel = ana_data.get("selected_item") or {}
    user_data = prep.get("data", {}) if isinstance(prep, dict) else {}

    user_url = user_data.get("url")
    user_prompt = user_data.get("promptDescription") or ""
    garment_url = sel.get("output_image_url") or ana_data.get("output_image_url") or sel.get("url")
    garment_prompt = sel.get("promptDescription") or ana_data.get("promptDescription") or sel.get("description") or ""
    garment_type = sel.get("type") or ana_data.get("selected_type") or "top"

    print("CHAIN_INPUTS")
    print(json.dumps(
        {
            "user_url": user_url,
            "user_prompt": user_prompt,
            "garment_url": garment_url,
            "garment_prompt": garment_prompt,
            "garment_type": garment_type,
        },
        indent=2,
    )[:20000])

    if user_url and garment_url:
        do_tryon(user_url, user_prompt, garment_url, garment_prompt, garment_type)
    else:
        print("Missing URLs, not running tryon")


if __name__ == "__main__":
    main()

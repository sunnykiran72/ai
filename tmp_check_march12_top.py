import json
import re
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8000"
IMAGE = Path("/tmp/analysis_top_source_20260312.png")


def parse_multipart(resp):
    ctype = resp.headers.get("content-type", "")
    m = re.search(r"boundary=([^;]+)", ctype)
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


def main():
    if not IMAGE.exists():
        raise FileNotFoundError(IMAGE)

    headers = {"Authorization": "Bearer local-test-token"}
    print("health=", requests.get(f"{BASE}/health", timeout=30).text)
    with IMAGE.open("rb") as handle:
        files = {"file": (IMAGE.name, handle, "image/png")}
        data = {"type": "top"}
        resp = requests.post(f"{BASE}/analyze", headers=headers, files=files, data=data, timeout=1200)

    print("status=", resp.status_code)
    print("content-type=", resp.headers.get("content-type", ""))
    parts = parse_multipart(resp)
    if parts is None:
        print(resp.text[:12000])
        return

    metadata = json.loads(parts["metadata"]["bytes"].decode("utf-8", errors="ignore"))
    print(json.dumps(metadata, indent=2)[:20000])
    data = metadata.get("data", {}) if isinstance(metadata, dict) else {}
    print("output_image_url=", data.get("output_image_url"))
    print("selected_type=", data.get("selected_type"))
    print("promptDescription=", data.get("promptDescription"))
    selected_item = data.get("selected_item") or {}
    if isinstance(selected_item, dict):
        print("selected_item.promptDescription=", selected_item.get("promptDescription"))
        print("selected_item.baseGarmentPrompt=", selected_item.get("baseGarmentPrompt"))
        print("selected_item.extractionAvoidClause=", selected_item.get("extractionAvoidClause"))
        print("selected_item.minicpm_description=", selected_item.get("minicpm_description"))


if __name__ == "__main__":
    main()

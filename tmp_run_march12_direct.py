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
    with IMAGE.open("rb") as handle:
        files = {"file": (IMAGE.name, handle, "image/png")}
        data = {"type": "top"}
        resp = requests.post(f"{BASE}/analyze", headers=headers, files=files, data=data, timeout=3600)

    print("status=", resp.status_code)
    print("content-type=", resp.headers.get("content-type", ""))
    parts = parse_multipart(resp)
    print("parts=", None if parts is None else list(parts.keys()))
    if parts is None:
        print(resp.text[:20000])
        return

    metadata = json.loads(parts["metadata"]["bytes"].decode("utf-8", errors="ignore"))
    print(json.dumps(metadata, indent=2)[:30000])


if __name__ == "__main__":
    main()

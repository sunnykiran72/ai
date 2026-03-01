import json
import re
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000/analyze"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiIwNDM3OTY3OC00N2E2LTQwYWYtODU1Ni05YTZhMTU5MjQyYmUiLCJhdXRoVHlwZSI6IlBIT05FIiwidG9rZW5faWQiOiI2M2Q4NWFhYi1lMTFjLTQ1NmItOTM3OS03M2ViMzJhMzZiOWMiLCJleHAiOjE3NzIzNTYwODB9.AUmcPiI1cJwE076u2ZC0DTCzx4JVIwm3o5KvDzG52wo"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

CASES = {
    "glamify_png": "https://www.glamifyfashion.com/cdn/shop/files/29_7acf02c7-3a70-4af6-bb5b-48511538a93b.png?v=1740137765",
    "fashionloft_jpg": "https://thefashionloft.in/wp-content/uploads/2025/03/engin-akyurt-jaZoffxg1yc-unsplash-768x1152.jpg",
}

OUT = Path("/workspace/hybrid_vto_v1/ai/debug_latest_user_urls")
OUT.mkdir(parents=True, exist_ok=True)


def parse_multipart(resp):
    ctype = resp.headers.get("content-type", "")
    m = re.search(r"boundary=([^;]+)", ctype)
    if not m:
        return {}
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


def save_resp(case_dir: Path, tag: str, resp):
    (case_dir / f"{tag}_status.txt").write_text(str(resp.status_code))
    parts = parse_multipart(resp)
    if "metadata" not in parts:
        (case_dir / f"{tag}_raw.txt").write_text(resp.text[:20000])
        return {"status": resp.status_code, "error": "no_metadata_part"}

    md = json.loads(parts["metadata"]["bytes"].decode("utf-8", errors="ignore"))
    (case_dir / f"{tag}_metadata.json").write_text(json.dumps(md, indent=2))

    for name, p in parts.items():
        if name == "metadata":
            continue
        fn = p.get("filename") or f"{name}.png"
        (case_dir / f"{tag}_{fn}").write_bytes(p["bytes"])

    data = md.get("data", {}) if isinstance(md, dict) else {}
    out_url = data.get("output_image_url") or data.get("cloth_url")
    if out_url:
        try:
            r = requests.get(out_url, timeout=60)
            if r.ok and r.content:
                (case_dir / f"{tag}_output_image_url.png").write_bytes(r.content)
        except Exception:
            pass

    return {
        "status": resp.status_code,
        "result": data.get("result"),
        "reason_codes": data.get("reason_codes"),
        "total_garments_found": data.get("total_garments_found"),
        "selected_type": data.get("selected_type"),
        "selected_item": {
            "type": (data.get("selected_item") or {}).get("type"),
            "bbox": (data.get("selected_item") or {}).get("bbox"),
            "detection_source": (data.get("selected_item") or {}).get("detection_source"),
            "type_source": (data.get("selected_item") or {}).get("type_source"),
            "vton_crop_bbox": (data.get("selected_item") or {}).get("vton_crop_bbox"),
            "vton_crop_mode": (data.get("selected_item") or {}).get("vton_crop_mode"),
            "tighten_adjustment": (data.get("selected_item") or {}).get("tighten_adjustment"),
            "output_image_url": (data.get("selected_item") or {}).get("output_image_url"),
        },
        "item_breakdown": [
            {
                "rank": it.get("rank"),
                "type": it.get("type"),
                "bbox": it.get("bbox"),
                "detection_source": it.get("detection_source"),
                "type_source": it.get("type_source"),
                "tighten_adjustment": it.get("tighten_adjustment"),
            }
            for it in (data.get("item_breakdown") or [])
        ],
        "output_image_url": data.get("output_image_url"),
    }


def run_case(case, url):
    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)

    img_r = requests.get(url, timeout=60)
    img_r.raise_for_status()
    img_bytes = img_r.content
    suffix = ".jpg"
    ctype = (img_r.headers.get("content-type") or "").lower()
    if "png" in ctype or ".png" in url.lower():
        suffix = ".png"

    (case_dir / f"input{suffix}").write_bytes(img_bytes)

    def post(tag, data=None):
        files = {
            "image": (
                f"input{suffix}",
                img_bytes,
                "image/png" if suffix == ".png" else "image/jpeg",
            )
        }
        r = requests.post(BASE, headers=HEADERS, files=files, data=data or {}, timeout=300)
        return save_resp(case_dir, tag, r)

    return {
        "step1": post("step1"),
        "top": post("top", {"type": "top"}),
        "bottom": post("bottom", {"type": "bottom"}),
    }


summary = {}
for case, url in CASES.items():
    summary[case] = run_case(case, url)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"ARTIFACT_DIR={OUT}")

"""
Response payload builders for the /analyze API.
Ported from tryon_tryoff_v2/src/api_features/response_payloads.py
"""
import json
import uuid
from typing import Dict, List, Optional

from fastapi.responses import JSONResponse, Response


def build_error_payload(
    title: str,
    description: str,
    reason_codes: List[str],
    status_code: int = 400,
    result: str = "REJECTED",
    message: str = "",
) -> Dict[str, object]:
    return {
        "status": status_code,
        "data": {
            "result": result,
            "title": title,
            "description": description,
            "reason_codes": reason_codes,
        },
        "message": message,
    }


def build_success_payload(data: Dict[str, object], message: str = "", status_code: int = 200) -> Dict[str, object]:
    normalized_data = dict(data or {})
    if "result" not in normalized_data:
        normalized_data["result"] = "ACCEPTED" if 200 <= int(status_code) < 300 else "REJECTED"
    return {
        "status": status_code,
        "data": normalized_data,
        "message": message,
    }


def build_multipart_parts(
    *,
    items: List[Dict[str, object]],
    cloth_url: Optional[str] = None,
) -> Dict[str, object]:
    parts: List[Dict[str, object]] = []
    for item in items:
        part: Dict[str, object] = {
            "part_name": f"item_{int(item.get('rank', item.get('garment_id', 0)))}",
            "kind": "crop",
            "rank": int(item.get("rank", item.get("garment_id", 0))),
            "mime_type": "image/png",
            "type": str(item.get("garment_type", item.get("type", "all"))),
        }
        if item.get("image_url"):
            part["image_url"] = item["image_url"]
        parts.append(part)

    if cloth_url:
        parts.append({
            "part_name": "extracted_cloth",
            "kind": "cloth",
            "mime_type": "image/png",
            "image_url": cloth_url,
        })

    return {
        "format": "multipart-data",
        "parts": parts,
    }


def json_response(payload: Dict[str, object]) -> JSONResponse:
    status_code = int(payload.get("status", 200))
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def multipart_form_response(
    payload: Dict[str, object],
    *,
    binary_parts: Optional[List[Dict[str, object]]] = None,
) -> Response:
    boundary = f"glamify-boundary-{uuid.uuid4().hex}"
    body = bytearray()

    def _append_text_part(name: str, text: str, content_type: str) -> None:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(text.encode("utf-8"))
        body.extend(b"\r\n")

    def _append_binary_part(name: str, filename: str, content_type: str, data: bytes) -> None:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(data)
        body.extend(b"\r\n")

    metadata_json = json.dumps(payload, ensure_ascii=False)
    _append_text_part("metadata", metadata_json, "application/json")

    for part in (binary_parts or []):
        data = part.get("bytes")
        if not isinstance(data, (bytes, bytearray)):
            continue
        _append_binary_part(
            str(part.get("name", "file")),
            str(part.get("filename", "file.bin")),
            str(part.get("content_type", "application/octet-stream")),
            bytes(data),
        )

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    status_code = int(payload.get("status", 200))
    return Response(
        content=bytes(body),
        status_code=status_code,
        media_type=f"multipart/form-data; boundary={boundary}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

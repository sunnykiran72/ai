#!/usr/bin/env python3
"""Batch-evaluate the user image eligibility gate on local images.

This script runs the local person detector + face detector path used by the
user-image preparation pipeline, but it disables MiniCPM and keeps the result
fully local. It prints per-image accept/reject results and an optional accuracy
summary when expected labels are supplied.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.yolo_runner import YoloPersonDetectorRunner  # noqa: E402
from utils.user_preparation import _opencv_face_detector, prepare_user_image_core  # noqa: E402


@dataclass
class GateResult:
    path: Path
    verdict: str
    reason: str
    face_visible: Optional[bool]
    full_body_visible: Optional[bool]


def _expand_inputs(paths: List[str], globs: List[str]) -> List[Path]:
    items: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            items.extend(sorted(x for x in p.iterdir() if x.is_file()))
        elif p.exists():
            items.append(p)
        else:
            raise FileNotFoundError(f"Path does not exist: {raw}")

    for pattern in globs:
        items.extend(sorted(Path(p) for p in glob.glob(pattern)))

    seen = set()
    unique: List[Path] = []
    for p in items:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _load_labels(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Labels file must be a JSON object mapping image paths to 'accept'/'reject'.")
    labels: Dict[str, str] = {}
    for key, value in payload.items():
        label = str(value).strip().lower()
        if label not in {"accept", "reject"}:
            raise ValueError(f"Invalid label for {key!r}: {value!r}")
        labels[str(key)] = label
    return labels


def _classify_image(
    path: Path,
    *,
    person_detector_fn,
) -> GateResult:
    image = Image.open(path).convert("RGB")
    result = prepare_user_image_core(
        image,
        person_detector_fn=person_detector_fn,
        face_detector_fn=_opencv_face_detector,
        verifier_fn=None,
        description_fn=None,
        upload_fn=lambda _bytes: "dry-run://prepared",
        blur_check_enabled=False,
        verification_required=False,
    )

    if "error" in result:
        face_meta = result.get("meta", {}).get("face", {})
        body_visibility = face_meta.get("body_visibility", {})
        face_detect_meta = face_meta.get("detect", {})
        return GateResult(
            path=path,
            verdict="reject",
            reason=str(result.get("error") or result.get("message") or "unknown"),
            face_visible=bool(face_detect_meta.get("count", 0)) if isinstance(face_detect_meta, dict) else None,
            full_body_visible=bool(body_visibility.get("visible")) if isinstance(body_visibility, dict) else None,
        )

    face_meta = result.get("meta", {}).get("face", {})
    body_visibility = face_meta.get("body_visibility", {})
    face_detect_meta = face_meta.get("detect", {})
    return GateResult(
        path=path,
        verdict="accept",
        reason="ok",
        face_visible=bool(face_detect_meta.get("count", 0)) if isinstance(face_detect_meta, dict) else None,
        full_body_visible=bool(body_visibility.get("visible")) if isinstance(body_visibility, dict) else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local user-image gate on a batch of images.")
    parser.add_argument("paths", nargs="*", help="Image files or directories to evaluate.")
    parser.add_argument(
        "--glob",
        dest="globs",
        action="append",
        default=[],
        help="Glob pattern to expand into image files. Can be used multiple times.",
    )
    parser.add_argument(
        "--labels",
        help="Optional JSON file mapping image paths to expected labels ('accept' or 'reject').",
    )
    args = parser.parse_args()

    inputs = _expand_inputs(args.paths, args.globs)
    if not inputs:
        print("No input images found.", file=sys.stderr)
        return 2

    labels = _load_labels(args.labels)

    runner = YoloPersonDetectorRunner()

    def person_detector_fn(image, conf=0.25, iou=0.45):
        return runner.predict(image, conf=conf, iou=iou)

    results: List[GateResult] = []
    correct = 0
    total_labeled = 0

    print("image\tverdict\treason\tface_visible\tfull_body_visible\texpected")
    for path in inputs:
        res = _classify_image(path, person_detector_fn=person_detector_fn)
        results.append(res)
        expected = labels.get(str(path)) or labels.get(path.name) or labels.get(path.as_posix())
        if expected is not None:
            total_labeled += 1
            if expected == res.verdict:
                correct += 1
        print(
            f"{path}\t{res.verdict}\t{res.reason}\t"
            f"{str(res.face_visible).lower() if res.face_visible is not None else 'unknown'}\t"
            f"{str(res.full_body_visible).lower() if res.full_body_visible is not None else 'unknown'}\t"
            f"{expected or '-'}"
        )

    if total_labeled:
        accuracy = correct / total_labeled
        print(f"\naccuracy={accuracy:.3f} ({correct}/{total_labeled})")
    else:
        accept_count = sum(1 for r in results if r.verdict == "accept")
        reject_count = len(results) - accept_count
        print(f"\naccepted={accept_count} rejected={reject_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

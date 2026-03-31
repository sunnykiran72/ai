"""
Identity dataset v1 helpers.

This module encodes the 20-shot per-person capture template and provides
helpers to generate/validate a manifest for the pilot dataset:
75 people x 20 shots = 1500 images.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ShotTemplateRow:
    shot_id: str
    pose_id: str
    pose_description: str
    angle_id: str
    angle_description: str
    background_id: str
    lighting_id: str
    garment_slot: str


SHOT_TEMPLATE: Tuple[ShotTemplateRow, ...] = (
    ShotTemplateRow("01", "P1", "Standing neutral, arms relaxed", "A1", "Front", "B1", "L1", "G_core_1"),
    ShotTemplateRow("02", "P1", "Standing neutral, arms relaxed", "A2", "30° left", "B1", "L1", "G_core_1"),
    ShotTemplateRow("03", "P1", "Standing neutral, arms relaxed", "A3", "30° right", "B1", "L1", "G_core_1"),
    ShotTemplateRow("04", "P1", "Standing neutral, arms relaxed", "A4", "Near profile", "B1", "L2", "G_core_1"),
    ShotTemplateRow("05", "P2", "Standing contrapposto (weight shift)", "A1", "Front", "B2", "L1", "G_core_2"),
    ShotTemplateRow("06", "P2", "Standing contrapposto (weight shift)", "A2", "30° left", "B2", "L1", "G_core_2"),
    ShotTemplateRow("07", "P2", "Standing contrapposto (weight shift)", "A3", "30° right", "B2", "L2", "G_core_2"),
    ShotTemplateRow("08", "P2", "Standing contrapposto (weight shift)", "A4", "Near profile", "B2", "L2", "G_core_2"),
    ShotTemplateRow("09", "P3", "Seated upright, knees forward", "A1", "Front", "B1", "L2", "G_rot_1"),
    ShotTemplateRow("10", "P3", "Seated upright, knees forward", "A2", "30° left", "B1", "L2", "G_rot_1"),
    ShotTemplateRow("11", "P3", "Seated upright, knees forward", "A3", "30° right", "B3", "L1", "G_rot_1"),
    ShotTemplateRow("12", "P3", "Seated upright, knees forward", "A4", "Near profile", "B3", "L1", "G_rot_1"),
    ShotTemplateRow("13", "P4", "Seated side posture", "A1", "Front", "B2", "L2", "G_rot_2"),
    ShotTemplateRow("14", "P4", "Seated side posture", "A2", "30° left", "B2", "L2", "G_rot_2"),
    ShotTemplateRow("15", "P4", "Seated side posture", "A3", "30° right", "B3", "L1", "G_rot_2"),
    ShotTemplateRow("16", "P4", "Seated side posture", "A4", "Near profile", "B3", "L2", "G_rot_2"),
    ShotTemplateRow("17", "P5", "Motion pose (step/walk freeze)", "A1", "Front", "B1", "L1", "G_rot_3"),
    ShotTemplateRow("18", "P5", "Motion pose (step/walk freeze)", "A2", "30° left", "B1", "L2", "G_rot_3"),
    ShotTemplateRow("19", "P5", "Motion pose (step/walk freeze)", "A3", "30° right", "B2", "L1", "G_rot_3"),
    ShotTemplateRow("20", "P5", "Motion pose (step/walk freeze)", "A4", "Near profile", "B2", "L2", "G_rot_3"),
)

REQUIRED_FIELDS: Tuple[str, ...] = (
    "person_id",
    "shot_id",
    "pose_id",
    "angle_id",
    "background_id",
    "lighting_id",
    "garment_id",
    "garment_type",
    "camera_distance_class",
    "quality_flags",
)

MANIFEST_FIELDS: Tuple[str, ...] = (
    "person_id",
    "split",
    "shot_id",
    "pose_id",
    "pose_description",
    "angle_id",
    "angle_description",
    "background_id",
    "lighting_id",
    "garment_slot",
    "garment_id",
    "garment_type",
    "camera_distance_class",
    "quality_flags",
)


def generate_person_ids(num_people: int = 75, prefix: str = "person") -> List[str]:
    if num_people <= 0:
        raise ValueError("num_people must be > 0")
    clean_prefix = str(prefix or "").strip() or "person"
    return [f"{clean_prefix}_{idx:03d}" for idx in range(1, num_people + 1)]


def assign_person_splits(
    person_ids: Sequence[str],
    train_count: int = 60,
    val_count: int = 8,
    test_count: int = 7,
) -> Dict[str, str]:
    if len(person_ids) != (train_count + val_count + test_count):
        raise ValueError(
            "split counts must sum to number of people: "
            f"{train_count}+{val_count}+{test_count}!={len(person_ids)}"
        )
    split_by_person: Dict[str, str] = {}
    for idx, person_id in enumerate(person_ids):
        if idx < train_count:
            split_by_person[person_id] = "train"
        elif idx < train_count + val_count:
            split_by_person[person_id] = "val"
        else:
            split_by_person[person_id] = "test"
    return split_by_person


def build_manifest_rows(
    person_ids: Sequence[str],
    split_by_person: Dict[str, str],
    camera_distance_class: str = "full_body_mid_distance",
    default_quality_flags: str = "[]",
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for person_id in person_ids:
        split_name = str(split_by_person.get(person_id) or "").strip()
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"missing/invalid split for person_id={person_id}")
        for shot in SHOT_TEMPLATE:
            rows.append(
                {
                    "person_id": person_id,
                    "split": split_name,
                    "shot_id": shot.shot_id,
                    "pose_id": shot.pose_id,
                    "pose_description": shot.pose_description,
                    "angle_id": shot.angle_id,
                    "angle_description": shot.angle_description,
                    "background_id": shot.background_id,
                    "lighting_id": shot.lighting_id,
                    "garment_slot": shot.garment_slot,
                    "garment_id": f"{person_id}__{shot.garment_slot}",
                    "garment_type": "tbd",
                    "camera_distance_class": camera_distance_class,
                    "quality_flags": default_quality_flags,
                }
            )
    return rows


def validate_manifest_rows(rows: Sequence[Dict[str, str]]) -> List[str]:
    errors: List[str] = []
    if not rows:
        return ["manifest has no rows"]

    required = set(REQUIRED_FIELDS)
    template_shot_ids = {s.shot_id for s in SHOT_TEMPLATE}
    template_by_shot = {s.shot_id: s for s in SHOT_TEMPLATE}
    person_to_shots: Dict[str, List[str]] = defaultdict(list)
    person_to_splits: Dict[str, set] = defaultdict(set)

    for idx, row in enumerate(rows, start=1):
        missing = [field for field in required if not str(row.get(field, "")).strip()]
        if missing:
            errors.append(f"row {idx}: missing required fields: {', '.join(missing)}")
            continue

        person_id = str(row["person_id"]).strip()
        shot_id = str(row["shot_id"]).strip()
        split = str(row.get("split", "")).strip()

        if shot_id not in template_shot_ids:
            errors.append(f"row {idx}: invalid shot_id={shot_id}")
            continue

        # Check template consistency for template-defined fields.
        expected = template_by_shot[shot_id]
        if str(row["pose_id"]).strip() != expected.pose_id:
            errors.append(f"row {idx}: pose_id mismatch for shot {shot_id}")
        if str(row["angle_id"]).strip() != expected.angle_id:
            errors.append(f"row {idx}: angle_id mismatch for shot {shot_id}")
        if str(row["background_id"]).strip() != expected.background_id:
            errors.append(f"row {idx}: background_id mismatch for shot {shot_id}")
        if str(row["lighting_id"]).strip() != expected.lighting_id:
            errors.append(f"row {idx}: lighting_id mismatch for shot {shot_id}")

        if split and split not in {"train", "val", "test"}:
            errors.append(f"row {idx}: invalid split={split}")
        if split:
            person_to_splits[person_id].add(split)

        person_to_shots[person_id].append(shot_id)

        # quality_flags must be a JSON array string.
        quality_flags = str(row.get("quality_flags", "")).strip()
        try:
            parsed = json.loads(quality_flags)
            if not isinstance(parsed, list):
                errors.append(f"row {idx}: quality_flags must decode to a JSON array")
        except Exception:
            errors.append(f"row {idx}: quality_flags must be valid JSON array string")

    for person_id, shots in person_to_shots.items():
        counts = Counter(shots)
        if len(shots) != len(SHOT_TEMPLATE):
            errors.append(
                f"person_id={person_id}: expected {len(SHOT_TEMPLATE)} rows, got {len(shots)}"
            )
        missing = sorted(template_shot_ids.difference(counts.keys()))
        if missing:
            errors.append(f"person_id={person_id}: missing shot_ids={','.join(missing)}")
        duplicate = sorted(shot for shot, n in counts.items() if n > 1)
        if duplicate:
            errors.append(f"person_id={person_id}: duplicate shot_ids={','.join(duplicate)}")

    for person_id, splits in person_to_splits.items():
        if len(splits) > 1:
            errors.append(f"person_id={person_id}: appears in multiple splits={sorted(splits)}")

    return errors


def write_csv(path: Path, rows: Iterable[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return [dict(r) for r in reader]


def shot_template_rows() -> List[Dict[str, str]]:
    return [
        {
            "shot_id": s.shot_id,
            "pose_id": s.pose_id,
            "pose_description": s.pose_description,
            "angle_id": s.angle_id,
            "angle_description": s.angle_description,
            "background_id": s.background_id,
            "lighting_id": s.lighting_id,
            "garment_slot": s.garment_slot,
        }
        for s in SHOT_TEMPLATE
    ]


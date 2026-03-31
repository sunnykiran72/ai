#!/usr/bin/env python3
"""
Identity dataset v1 CLI.

Commands:
- generate: build a manifest template (default 75 people x 20 shots)
- validate: validate a filled manifest against the v1 template contract
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

# Ensure repo root is importable when running as a direct script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.identity_dataset_v1 import (
    MANIFEST_FIELDS,
    assign_person_splits,
    build_manifest_rows,
    generate_person_ids,
    read_csv,
    shot_template_rows,
    validate_manifest_rows,
    write_csv,
)


SHOT_TEMPLATE_FIELDS = (
    "shot_id",
    "pose_id",
    "pose_description",
    "angle_id",
    "angle_description",
    "background_id",
    "lighting_id",
    "garment_slot",
)


def _build_summary(split_by_person: Dict[str, str], total_rows: int) -> Dict[str, object]:
    train = sum(1 for _, split in split_by_person.items() if split == "train")
    val = sum(1 for _, split in split_by_person.items() if split == "val")
    test = sum(1 for _, split in split_by_person.items() if split == "test")
    return {
        "num_people": len(split_by_person),
        "rows_total": total_rows,
        "shots_per_person": 20,
        "split_counts": {"train": train, "val": val, "test": test},
    }


def cmd_generate(args: argparse.Namespace) -> int:
    person_ids = generate_person_ids(num_people=args.num_people, prefix=args.person_prefix)
    split_by_person = assign_person_splits(
        person_ids,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
    )
    rows = build_manifest_rows(
        person_ids,
        split_by_person,
        camera_distance_class=args.camera_distance_class,
        default_quality_flags=args.default_quality_flags,
    )

    manifest_path = Path(args.output_manifest_csv).resolve()
    shot_template_path = Path(args.output_shot_template_csv).resolve()
    split_roster_path = Path(args.output_split_roster_csv).resolve()
    summary_path = Path(args.output_summary_json).resolve()

    write_csv(manifest_path, rows, MANIFEST_FIELDS)
    write_csv(shot_template_path, shot_template_rows(), SHOT_TEMPLATE_FIELDS)
    split_rows = [{"person_id": person_id, "split": split} for person_id, split in split_by_person.items()]
    write_csv(split_roster_path, split_rows, ("person_id", "split"))

    summary = _build_summary(split_by_person, len(rows))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Generated manifest: {manifest_path}")
    print(f"Generated shot template: {shot_template_path}")
    print(f"Generated split roster: {split_roster_path}")
    print(f"Generated summary: {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest_csv).resolve()
    rows = read_csv(manifest_path)
    errors = validate_manifest_rows(rows)
    if errors:
        print(f"Manifest INVALID: {manifest_path}")
        for err in errors:
            print(f"- {err}")
        return 1
    print(f"Manifest valid: {manifest_path}")
    print(f"Rows: {len(rows)}")
    unique_people = len({str(r.get('person_id', '')).strip() for r in rows})
    print(f"Unique people: {unique_people}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identity dataset v1 generator/validator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="generate v1 manifest template")
    gen.add_argument("--num-people", type=int, default=75)
    gen.add_argument("--person-prefix", type=str, default="person")
    gen.add_argument("--train-count", type=int, default=60)
    gen.add_argument("--val-count", type=int, default=8)
    gen.add_argument("--test-count", type=int, default=7)
    gen.add_argument("--camera-distance-class", type=str, default="full_body_mid_distance")
    gen.add_argument("--default-quality-flags", type=str, default="[]")
    gen.add_argument(
        "--output-manifest-csv",
        type=str,
        default="docs/identity_dataset_manifest_v1_template.csv",
    )
    gen.add_argument(
        "--output-shot-template-csv",
        type=str,
        default="docs/identity_dataset_v1_shot_template.csv",
    )
    gen.add_argument(
        "--output-split-roster-csv",
        type=str,
        default="docs/identity_dataset_v1_person_splits.csv",
    )
    gen.add_argument(
        "--output-summary-json",
        type=str,
        default="docs/identity_dataset_v1_summary.json",
    )
    gen.set_defaults(func=cmd_generate)

    val = sub.add_parser("validate", help="validate a manifest")
    val.add_argument("manifest_csv", type=str)
    val.set_defaults(func=cmd_validate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

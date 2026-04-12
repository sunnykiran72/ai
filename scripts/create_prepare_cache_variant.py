#!/usr/bin/env python3
"""
Build a derived prepare-cache CSV by replacing the user prompt text.

This is intended for try-on A/B tests where the prepared image URL and worn types
must remain unchanged while only the promptDescription changes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_MINIMAL_PROMPT = "A young person for tryon shown in the reference image."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a derived prepare-cache CSV with a prompt override.")
    parser.add_argument("--input-csv", required=True, help="Path to source prepare-cache results.csv")
    parser.add_argument("--output-csv", required=True, help="Path to write derived cache CSV")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_MINIMAL_PROMPT,
        help="Constant user prompt to write into every successful cache row",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Override prompts for all rows, including error rows. Default only updates success rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not input_csv.exists():
        raise SystemExit(f"input csv not found: {input_csv}")

    prompt = " ".join(str(args.prompt or "").split()).strip()
    if not prompt:
        raise SystemExit("prompt must not be empty")
    if prompt[-1] not in ".!?":
        prompt = f"{prompt}."

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise SystemExit(f"input csv has no header: {input_csv}")
        if "user_prompt" not in fieldnames:
            raise SystemExit("input csv is missing required 'user_prompt' column")

        rows = list(reader)

    updated = 0
    with output_csv.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            should_override = args.all_rows or str(row.get("status") or "").strip().lower() == "success"
            if should_override:
                row["user_prompt"] = prompt
                updated += 1
            writer.writerow(row)

    print(f"input_csv={input_csv}")
    print(f"output_csv={output_csv}")
    print(f"prompt={prompt}")
    print(f"rows_total={len(rows)}")
    print(f"rows_updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

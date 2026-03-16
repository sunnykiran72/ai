#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import urllib.request
from typing import Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.image_ops import rgb_to_lab, delta_e_cie76

MEODAI_CSV_URL = "https://raw.githubusercontent.com/meodai/color-names/master/src/colornames.csv"
MEODAI_BESTOF_JSON_URL = "https://unpkg.com/color-name-list@14.28.0/dist/colornames.bestof.json"
MEODAI_SHORT_JSON_URL = "https://unpkg.com/color-name-list@14.28.0/dist/colornames.short.json"
XKCD_RGB_URL = "https://xkcd.com/color/rgb.txt"
COLOR_PEDIA_PARQUET_URL = "https://huggingface.co/datasets/boltuix/color-pedia/resolve/main/color_pedia.parquet"


def _download_if_missing(url: str, dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        urllib.request.urlretrieve(url, dest_path)
    return dest_path


def _hex_to_rgb(hex_value: str) -> Tuple[int, int, int]:
    token = hex_value.strip().lstrip("#")
    if len(token) != 6:
        raise ValueError(f"Expected 6-char hex, got {hex_value!r}")
    return tuple(int(token[i : i + 2], 16) for i in (0, 2, 4))


def _load_meodai(csv_path: str) -> List[Tuple[str, np.ndarray]]:
    entries: List[Tuple[str, np.ndarray]] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() in {"name", "color"}:
                continue
            name = row[0].strip()
            hex_value = row[1].strip() if len(row) > 1 else ""
            if not name or not hex_value.startswith("#"):
                continue
            try:
                rgb = np.array([_hex_to_rgb(hex_value)], dtype=np.uint8)
            except ValueError:
                continue
            lab = rgb_to_lab(rgb)[0]
            entries.append((name, lab))
    return entries


def _load_meodai_json(json_path: str) -> List[Tuple[str, np.ndarray]]:
    entries: List[Tuple[str, np.ndarray]] = []
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        return entries
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        hex_value = str(item.get("hex") or "").strip()
        if not name or not hex_value.startswith("#"):
            continue
        try:
            rgb = np.array([_hex_to_rgb(hex_value)], dtype=np.uint8)
        except ValueError:
            continue
        lab = rgb_to_lab(rgb)[0]
        entries.append((name, lab))
    return entries


def _load_xkcd_rgb(txt_path: str) -> List[Tuple[str, np.ndarray]]:
    entries: List[Tuple[str, np.ndarray]] = []
    with open(txt_path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) < 2:
                continue
            try:
                name = parts[0].strip()
                hex_value = parts[1].strip()
            except Exception:
                continue
            if not name or not hex_value.startswith("#"):
                continue
            try:
                rgb = np.array([_hex_to_rgb(hex_value)], dtype=np.uint8)
            except ValueError:
                continue
            lab = rgb_to_lab(rgb)[0]
            entries.append((name, lab))
    return entries


def _load_color_pedia(parquet_path: str) -> List[Tuple[str, np.ndarray]]:
    entries: List[Tuple[str, np.ndarray]] = []
    try:
        import polars as pl
    except Exception:
        return entries
    try:
        df = pl.read_parquet(parquet_path)
    except Exception:
        return entries
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("color name") or cols.get("color_name") or cols.get("name")
    hex_col = cols.get("hex code") or cols.get("hex_code") or cols.get("hex")
    if not name_col or not hex_col:
        return entries
    for name, hex_value in df.select([name_col, hex_col]).iter_rows():
        if not isinstance(name, str) or not isinstance(hex_value, str):
            continue
        name = name.strip()
        hex_value = hex_value.strip()
        if not name or not hex_value.startswith("#"):
            continue
        try:
            rgb = np.array([_hex_to_rgb(hex_value)], dtype=np.uint8)
        except ValueError:
            continue
        lab = rgb_to_lab(rgb)[0]
        entries.append((name, lab))
    return entries


def _nearest_name(hex_value: str, entries: List[Tuple[str, np.ndarray]]) -> str:
    rgb = np.array([_hex_to_rgb(hex_value)], dtype=np.uint8)
    lab = rgb_to_lab(rgb)[0]
    best_name = ""
    best_dist = float("inf")
    for name, lab_ref in entries:
        dist = delta_e_cie76(lab, lab_ref)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _load_palette_from_metadata(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    item = (payload.get("data") or {}).get("selected_item") or {}
    color = (item.get("metadata") or {}).get("garmentMetadata", {}).get("color", {})
    hexes = color.get("dominant_hexes") or []
    return [str(h).strip() for h in hexes if str(h).strip()]


def _iter_hexes(args: argparse.Namespace) -> Iterable[str]:
    if args.hex:
        for hx in args.hex:
            yield hx
    for meta in args.metadata or []:
        for hx in _load_palette_from_metadata(meta):
            yield hx


def main() -> int:
    parser = argparse.ArgumentParser(description="Map hex colors to descriptive names.")
    parser.add_argument(
        "--list",
        choices=["meodai", "meodai_bestof", "meodai_short", "xkcd", "color_pedia"],
        default="meodai_bestof",
    )
    parser.add_argument("--hex", nargs="*", default=[])
    parser.add_argument("--metadata", nargs="*", default=[])
    parser.add_argument("--cache-dir", default="/tmp/color_name_lists")
    args = parser.parse_args()

    if not args.hex and not args.metadata:
        print("Provide --hex values or --metadata JSON paths.", file=sys.stderr)
        return 2

    if args.list == "meodai":
        csv_path = os.path.join(args.cache_dir, "meodai_colornames.csv")
        _download_if_missing(MEODAI_CSV_URL, csv_path)
        entries = _load_meodai(csv_path)
    elif args.list == "meodai_bestof":
        json_path = os.path.join(args.cache_dir, "meodai_colornames.bestof.json")
        _download_if_missing(MEODAI_BESTOF_JSON_URL, json_path)
        entries = _load_meodai_json(json_path)
    elif args.list == "meodai_short":
        json_path = os.path.join(args.cache_dir, "meodai_colornames.short.json")
        _download_if_missing(MEODAI_SHORT_JSON_URL, json_path)
        entries = _load_meodai_json(json_path)
    elif args.list == "xkcd":
        txt_path = os.path.join(args.cache_dir, "xkcd_rgb.txt")
        _download_if_missing(XKCD_RGB_URL, txt_path)
        entries = _load_xkcd_rgb(txt_path)
    else:
        parquet_path = os.path.join(args.cache_dir, "color_pedia.parquet")
        _download_if_missing(COLOR_PEDIA_PARQUET_URL, parquet_path)
        entries = _load_color_pedia(parquet_path)

    seen = []
    for hx in _iter_hexes(args):
        token = str(hx).strip().upper()
        if not token:
            continue
        if not token.startswith("#"):
            token = f"#{token}"
        seen.append(token)

    for token in seen:
        name = _nearest_name(token, entries)
        print(f"{token} -> {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

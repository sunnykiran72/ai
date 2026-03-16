#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
import urllib.request
from typing import Dict, Iterable, List, Tuple

import colorsys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.image_ops import rgb_to_lab, delta_e_cie76

MEODAI_BESTOF_JSON_URL = "https://unpkg.com/color-name-list@14.28.0/dist/colornames.bestof.json"
XKCD_RGB_URL = "https://xkcd.com/color/rgb.txt"
COLOR_PEDIA_PARQUET_URL = "https://huggingface.co/datasets/boltuix/color-pedia/resolve/main/color_pedia.parquet"
NTC_PBI_URL = "https://raw.githubusercontent.com/tajmone/name-that-color/master/ntc.colors-names.pbi"


def _download_if_missing(url: str, dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        urllib.request.urlretrieve(url, dest_path)
    return dest_path


def _hex_to_rgb(hex_value: str) -> Tuple[int, int, int]:
    token = hex_value.strip().lstrip("#")
    return tuple(int(token[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _load_meodai_bestof(json_path: str) -> List[Tuple[str, np.ndarray]]:
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
        rgb = np.array([_hex_to_rgb(hex_value)], dtype=np.uint8)
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
            name = parts[0].strip()
            hex_value = parts[1].strip()
            if not name or not hex_value.startswith("#"):
                continue
            rgb = np.array([_hex_to_rgb(hex_value)], dtype=np.uint8)
            lab = rgb_to_lab(rgb)[0]
            entries.append((name, lab))
    return entries


def _load_color_pedia(parquet_path: str) -> List[Tuple[str, np.ndarray]]:
    entries: List[Tuple[str, np.ndarray]] = []
    try:
        import polars as pl
    except Exception:
        return entries
    df = pl.read_parquet(parquet_path)
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
        rgb = np.array([_hex_to_rgb(hex_value)], dtype=np.uint8)
        lab = rgb_to_lab(rgb)[0]
        entries.append((name, lab))
    return entries


def _load_ntc(pbi_path: str) -> List[Tuple[str, np.ndarray]]:
    entries: List[Tuple[str, np.ndarray]] = []
    with open(pbi_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("Data.s"):
                continue
            if "," not in line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 2:
                continue
            hex_token = parts[0].split("\"")[-2].strip()
            name = parts[1].split("\"")[1].strip()
            if not hex_token or not name:
                continue
            hex_value = f"#{hex_token}"
            rgb = np.array([_hex_to_rgb(hex_value)], dtype=np.uint8)
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


def _generate_garment_hexes(count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    buckets = [
        ("neutral", (0, 360), (0.0, 0.12), (0.15, 0.95), 18),
        ("earth", (20, 60), (0.25, 0.55), (0.25, 0.65), 15),
        ("pastel", (0, 360), (0.15, 0.45), (0.70, 0.90), 16),
        ("jewel", (0, 360), (0.55, 0.80), (0.30, 0.55), 14),
        ("denim", (200, 240), (0.25, 0.55), (0.25, 0.60), 12),
        ("greens", (80, 160), (0.20, 0.60), (0.25, 0.70), 10),
        ("reds", (330, 20), (0.30, 0.70), (0.30, 0.70), 10),
        ("yellows", (40, 70), (0.20, 0.55), (0.60, 0.90), 5),
    ]
    colors: List[str] = []
    for _, hue_range, sat_range, light_range, num in buckets:
        for _ in range(num):
            h_low, h_high = hue_range
            if h_low <= h_high:
                hue = rng.uniform(h_low, h_high)
            else:
                hue = rng.uniform(h_low, 360.0) if rng.random() < 0.5 else rng.uniform(0.0, h_high)
            sat = rng.uniform(*sat_range)
            light = rng.uniform(*light_range)
            r, g, b = colorsys.hls_to_rgb(hue / 360.0, light, sat)
            rgb = (int(r * 255), int(g * 255), int(b * 255))
            colors.append(_rgb_to_hex(rgb))
    while len(colors) < count:
        hue = rng.uniform(0.0, 360.0)
        sat = rng.uniform(0.15, 0.60)
        light = rng.uniform(0.25, 0.80)
        r, g, b = colorsys.hls_to_rgb(hue / 360.0, light, sat)
        colors.append(_rgb_to_hex((int(r * 255), int(g * 255), int(b * 255))))
    return colors[:count]


def _expected_descriptor(hex_value: str) -> str:
    r, g, b = _hex_to_rgb(hex_value)
    lab = rgb_to_lab(np.array([[r, g, b]], dtype=np.uint8))[0]
    l_star = lab[0] * 100.0 / 255.0
    a_star = lab[1] - 128.0
    b_star = lab[2] - 128.0
    chroma = float((a_star * a_star + b_star * b_star) ** 0.5)
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    hue = h * 360.0
    if chroma < 4.5:
        if l_star < 12.0:
            return "black"
        if l_star < 26.0:
            return "charcoal"
        if l_star < 52.0:
            return "gray"
        if l_star < 74.0:
            return "silver"
        if l_star < 88.0:
            return "off-white"
        return "white"
    if chroma < 9.5:
        if a_star <= -2.0 and b_star >= 4.0:
            if l_star >= 72.0:
                return "light grayish green"
            if l_star >= 58.0:
                return "grayish green"
            return "dark grayish green"
        if b_star >= 6.0 and l_star >= 56.0:
            return "beige"
        if b_star <= -4.0 and l_star >= 52.0:
            return "cool gray"

    light = "light" if l >= 0.70 else ("dark" if l <= 0.35 else "mid")
    tone = "muted" if s < 0.35 else ("vivid" if s > 0.65 else "soft")
    if 345 <= hue or hue < 15:
        base = "red"
    elif hue < 35:
        base = "orange"
    elif hue < 55:
        base = "yellow"
    elif hue < 80:
        base = "yellow-green"
    elif hue < 150:
        base = "green"
    elif hue < 200:
        base = "teal"
    elif hue < 250:
        base = "blue"
    elif hue < 290:
        base = "purple"
    elif hue < 330:
        base = "magenta"
    else:
        base = "rose"
    return f"{light} {tone} {base}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare color name models on random garment colors.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="/tmp/color_name_lists")
    parser.add_argument("--output", default="/tmp/color_name_compare.csv")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    meodai_path = _download_if_missing(
        MEODAI_BESTOF_JSON_URL,
        os.path.join(cache_dir, "meodai_colornames.bestof.json"),
    )
    xkcd_path = _download_if_missing(
        XKCD_RGB_URL,
        os.path.join(cache_dir, "xkcd_rgb.txt"),
    )
    color_pedia_path = _download_if_missing(
        COLOR_PEDIA_PARQUET_URL,
        os.path.join(cache_dir, "color_pedia.parquet"),
    )
    ntc_path = _download_if_missing(
        NTC_PBI_URL,
        os.path.join(cache_dir, "ntc.colors-names.pbi"),
    )

    lists = {
        "meodai_bestof": _load_meodai_bestof(meodai_path),
        "xkcd": _load_xkcd_rgb(xkcd_path),
        "color_pedia": _load_color_pedia(color_pedia_path),
        "ntc": _load_ntc(ntc_path),
    }

    colors = _generate_garment_hexes(args.count, args.seed)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hex", "expected", "meodai_bestof", "xkcd", "color_pedia", "ntc"])
        for hx in colors:
            expected = _expected_descriptor(hx)
            row = [
                hx,
                expected,
                _nearest_name(hx, lists["meodai_bestof"]),
                _nearest_name(hx, lists["xkcd"]),
                _nearest_name(hx, lists["color_pedia"]),
                _nearest_name(hx, lists["ntc"]),
            ]
            writer.writerow(row)

    print(f"Wrote {args.output} with {len(colors)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

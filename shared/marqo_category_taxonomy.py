"""
CSV-backed category taxonomy resolver for Marqo subcategory matching.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class CategoryRow:
    id: str
    key: str
    label: str
    parent_id: str
    is_active: bool


@dataclass(frozen=True)
class MarqoCandidate:
    key: str
    label: str
    parent_key: str
    parent_label: str


def _normalize_key(value: str) -> str:
    return " ".join(str(value or "").replace("-", "_").split()).strip().lower()


def _parse_active(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_LANE_PARENT_KEYS: Dict[str, tuple[str, ...]] = {
    "top": ("tops", "layering_pieces"),
    "outer": ("outerwear", "office_wear_formal"),
    "bottom": ("bottoms", "skirts", "activewear_sportswear"),
    "dress": ("dresses", "loungewear", "sets_one_pieces"),
}


def normalize_lane_parent_keys(raw: Optional[Mapping[str, str]]) -> Dict[str, tuple[str, ...]]:
    resolved: Dict[str, tuple[str, ...]] = {}
    for lane, defaults in DEFAULT_LANE_PARENT_KEYS.items():
        raw_value = str((raw or {}).get(lane, "") or "").strip()
        if not raw_value:
            resolved[lane] = tuple(_normalize_key(x) for x in defaults)
            continue
        parent_keys = []
        for part in raw_value.split(","):
            key = _normalize_key(part)
            if key:
                parent_keys.append(key)
        resolved[lane] = tuple(parent_keys) if parent_keys else tuple(_normalize_key(x) for x in defaults)
    return resolved


class MarqoCategoryTaxonomy:
    def __init__(self, *, rows: Iterable[CategoryRow], lane_parent_keys: Mapping[str, tuple[str, ...]]):
        self._rows = list(rows)
        self._row_by_id: Dict[str, CategoryRow] = {}
        self._row_by_key: Dict[str, CategoryRow] = {}
        self._children_by_parent_id: Dict[str, List[CategoryRow]] = {}
        self._lane_parent_keys = dict(lane_parent_keys)
        self._index()

    def _index(self) -> None:
        for row in self._rows:
            self._row_by_id[row.id] = row
            self._row_by_key[row.key] = row
        for row in self._rows:
            if row.parent_id:
                self._children_by_parent_id.setdefault(row.parent_id, []).append(row)
        for children in self._children_by_parent_id.values():
            children.sort(key=lambda r: r.label.lower())

    @property
    def has_rows(self) -> bool:
        return bool(self._rows)

    def top_level_keys(self) -> List[str]:
        return sorted(
            [row.key for row in self._rows if not row.parent_id],
            key=lambda key: str(self._row_by_key.get(key).label if key in self._row_by_key else key).lower(),
        )

    def children_for_primary_key(self, primary_key: str) -> List[MarqoCandidate]:
        normalized_primary = _normalize_key(primary_key)
        parent = self._row_by_key.get(normalized_primary)
        if parent is None:
            return []
        children = self._children_by_parent_id.get(parent.id, [])
        return [
            MarqoCandidate(
                key=child.key,
                label=child.label,
                parent_key=parent.key,
                parent_label=parent.label,
            )
            for child in children
        ]

    def candidates_for_lane(
        self,
        lane: str,
        *,
        preferred_primary_key: Optional[str] = None,
    ) -> List[MarqoCandidate]:
        preferred = _normalize_key(preferred_primary_key or "")
        if preferred:
            preferred_candidates = self.children_for_primary_key(preferred)
            if preferred_candidates:
                return preferred_candidates

        normalized_lane = _normalize_key(lane)
        if normalized_lane not in {"top", "bottom", "dress", "outer"}:
            normalized_lane = "top"
        parent_keys = self._lane_parent_keys.get(normalized_lane, ())
        rendered: List[MarqoCandidate] = []
        seen = set()
        for parent_key in parent_keys:
            for candidate in self.children_for_primary_key(parent_key):
                if candidate.key in seen:
                    continue
                seen.add(candidate.key)
                rendered.append(candidate)
        return rendered


def _read_rows(csv_path: str) -> List[CategoryRow]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows: List[CategoryRow] = []
        for payload in reader:
            row = CategoryRow(
                id=str(payload.get("id") or "").strip(),
                key=_normalize_key(str(payload.get("key") or "")),
                label=" ".join(str(payload.get("label") or "").split()).strip(),
                parent_id=str(payload.get("parent_id") or "").strip(),
                is_active=_parse_active(str(payload.get("is_active") or "")),
            )
            if not row.id or not row.key:
                continue
            if not row.is_active:
                continue
            rows.append(row)
    return rows


@lru_cache(maxsize=8)
def load_marqo_taxonomy(
    *,
    csv_path: str,
    lane_top_parents: str = "",
    lane_bottom_parents: str = "",
    lane_dress_parents: str = "",
    lane_outer_parents: str = "",
) -> MarqoCategoryTaxonomy:
    rows = _read_rows(csv_path)
    lane_parent_keys = normalize_lane_parent_keys(
        {
            "top": lane_top_parents,
            "bottom": lane_bottom_parents,
            "dress": lane_dress_parents,
            "outer": lane_outer_parents,
        }
    )
    return MarqoCategoryTaxonomy(rows=rows, lane_parent_keys=lane_parent_keys)

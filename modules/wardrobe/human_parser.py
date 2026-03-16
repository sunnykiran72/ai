import logging
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("glamify-ai")

class HumanParser:
    """
    Stateless parsing utilities.
    Model execution is injected via parser_fn to keep modules/ pure.
    """
    
    def __init__(
        self,
        parser_fn: Optional[Callable[[Image.Image], np.ndarray]] = None,
        labels: Optional[Dict[str, int]] = None,
    ):
        self._parser_fn = parser_fn

        # Default labels (LIP-style) plus fashn-ai aliases for backward compatibility.
        # Runtime labels from model config are merged automatically.
        self.labels = self._normalize_labels(labels or {
            # LIP / CIHP style ids
            "background": 0,
            "hat": 1,
            "hair": 2,
            "glove": 3,
            "sunglasses": 4,
            "upper_clothes": 5,
            "dress": 6,
            "coat": 7,
            "socks": 8,
            "pants": 9,
            "jumpsuit": 10,
            "scarf": 11,
            "skirt": 12,
            "face": 13,
            "left_arm": 14,
            "right_arm": 15,
            "left_leg": 16,
            "right_leg": 17,
            "left_shoe": 18,
            "right_shoe": 19,
            # fashn-ai aliases (kept distinct)
            "fashn_top": 3,
            "fashn_dress": 4,
            "fashn_skirt": 5,
            "fashn_pants": 6,
            "fashn_belt": 7,
            "fashn_bag": 8,
            "fashn_hat": 9,
            "fashn_scarf": 10,
            "fashn_glasses": 11,
            "fashn_arms": 12,
            "fashn_hands": 13,
            "fashn_legs": 14,
            "fashn_feet": 15,
            "fashn_torso": 16,
            "fashn_jewelry": 17,
        })

    @staticmethod
    def _norm_label(name: str) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    def _normalize_labels(self, labels: Dict[str, int]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k, v in (labels or {}).items():
            try:
                out[self._norm_label(k)] = int(v)
            except Exception:
                continue
        return out

    def _runtime_labels(self) -> Dict[str, int]:
        merged = dict(self.labels)
        runner = getattr(self._parser_fn, "__self__", None)
        runtime = getattr(runner, "label2id", None)
        if isinstance(runtime, dict) and runtime:
            merged.update(self._normalize_labels(runtime))
        return merged

    def _ids_for_aliases(self, labels: Dict[str, int], aliases: list[str]) -> list[int]:
        ids = []
        for alias in aliases:
            key = self._norm_label(alias)
            if key in labels:
                ids.append(int(labels[key]))
        return sorted(set(ids))

    def category_ids(self, category: str) -> list[int]:
        labels = self._runtime_labels()
        c = self._norm_label(category)
        alias_map = {
            "top": [
                "top", "upper", "upper_clothes", "upper-clothes", "upperclothes",
                "fashn_top",
            ],
            "outer": [
                "outer", "outerwear", "coat", "jacket", "blazer",
                "top", "upper", "upper_clothes", "upper-clothes", "upperclothes",
                "fashn_top",
            ],
            "bottom": [
                "bottom", "pants", "trousers", "skirt", "shorts", "belt",
                "fashn_pants", "fashn_skirt", "fashn_belt",
            ],
            "dress": ["dress", "fashn_dress"],
            "body": [
                "hat", "hair", "sunglasses", "glasses", "face",
                "left_arm", "right_arm", "left_leg", "right_leg",
                "left_shoe", "right_shoe", "bag", "scarf", "jewelry",
                "fashn_hat", "fashn_scarf", "fashn_glasses", "fashn_arms",
                "fashn_hands", "fashn_legs", "fashn_feet", "fashn_torso", "fashn_jewelry",
            ],
            "garment_fallback": [
                "top", "upper", "upper_clothes", "upper-clothes", "upperclothes",
                "dress", "pants", "skirt", "belt", "scarf",
                "fashn_top", "fashn_dress", "fashn_pants", "fashn_skirt", "fashn_belt", "fashn_scarf",
            ],
            "kill_fallback": [
                "background", "hat", "hair", "sunglasses", "glasses", "face",
                "left_arm", "right_arm", "left_leg", "right_leg", "left_shoe", "right_shoe",
                "bag", "scarf", "jewelry",
                "fashn_hat", "fashn_scarf", "fashn_glasses", "fashn_arms",
                "fashn_hands", "fashn_legs", "fashn_feet", "fashn_torso", "fashn_jewelry",
            ],
        }
        ids = self._ids_for_aliases(labels, alias_map.get(c, [c]))
        if not ids and c in labels:
            ids = [int(labels[c])]
        return sorted(set(ids))

    def parse(self, image: Image.Image) -> np.ndarray:
        """
        Executes parser_fn and returns a segmentation map.
        """
        if self._parser_fn is None:
            raise RuntimeError("HumanParser parser_fn is not configured.")

        parsing = self._parser_fn(image)
        if not isinstance(parsing, np.ndarray):
            raise TypeError("HumanParser parser_fn must return numpy.ndarray.")
        return parsing

    def get_mask_for_category(self, parsing: np.ndarray, category: str) -> np.ndarray:
        """
        Extracts a binary mask for 'top', 'bottom', 'dress', etc.
        """
        target_ids = self.category_ids(category)
        return np.isin(parsing, target_ids)

    def build_category_masks(self, parsing: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "top": self.get_mask_for_category(parsing, "top"),
            "bottom": self.get_mask_for_category(parsing, "bottom"),
            "dress": self.get_mask_for_category(parsing, "dress"),
            "outer": self.get_mask_for_category(parsing, "outer"),
        }

    def get_garment_mask(self, parsing: np.ndarray) -> np.ndarray:
        garment_ids = self.category_ids("garment_fallback")
        return np.isin(parsing, garment_ids)

    def get_occluder_mask(self, parsing: np.ndarray) -> np.ndarray:
        occluder_ids = self.category_ids("kill_fallback")
        return np.isin(parsing, occluder_ids)

if __name__ == "__main__":
    # Isolated Test
    logging.basicConfig(level=logging.INFO)
    parser = HumanParser()
    print("Human Parser Initialized.")

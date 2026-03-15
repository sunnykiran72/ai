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

        # Default labels for fashn-ai/fashn-human-parser; runtime labels from model config
        # are merged automatically to support alternate parsers.
        self.labels = self._normalize_labels(labels or {
            "background": 0,
            "face": 1,
            "hair": 2,
            "top": 3,
            "dress": 4,
            "skirt": 5,
            "pants": 6,
            "belt": 7,
            "bag": 8,
            "hat": 9,
            "scarf": 10,
            "glasses": 11,
            "arms": 12,
            "hands": 13,
            "legs": 14,
            "feet": 15,
            "torso": 16,
            "jewelry": 17,
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
            "top": ["top", "upper", "upper_clothes", "upper-clothes"],
            "outer": ["outer", "outerwear", "coat", "jacket", "blazer", "top", "upper", "upper_clothes", "upper-clothes"],
            "bottom": ["bottom", "pants", "trousers", "skirt", "shorts", "belt"],
            "dress": ["dress"],
            "body": [
                "hat", "hair", "sunglasses", "glasses", "face", "torso", "arms", "hands", "legs", "feet",
                "left_arm", "right_arm", "left_leg", "right_leg", "left_shoe", "right_shoe", "bag", "scarf", "jewelry"
            ],
            "garment_fallback": ["top", "upper", "upper_clothes", "upper-clothes", "dress", "pants", "skirt", "belt", "scarf"],
            "kill_fallback": [
                "background", "hat", "hair", "sunglasses", "glasses", "face", "torso", "arms", "hands", "legs", "feet",
                "left_arm", "right_arm", "left_leg", "right_leg", "left_shoe", "right_shoe", "bag", "scarf", "jewelry"
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

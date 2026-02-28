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

        # Standard Segformer labels
        self.labels = labels or {
            "background": 0, "hat": 1, "hair": 2, "glove": 3, "sunglasses": 4,
            "upper": 5, "dress": 6, "coat": 7, "socks": 8, "pants": 9,
            "jumpsuit": 10, "scarf": 11, "skirt": 12, "face": 13,
            "left_arm": 14, "right_arm": 15, "left_leg": 16, "right_leg": 17,
            "left_shoe": 18, "right_shoe": 19
        }

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
        target_ids = []
        if category == "top": target_ids = [5, 7, 11] # upper, coat, scarf
        elif category == "bottom": target_ids = [9, 12] # pants, skirt
        elif category == "dress": target_ids = [6, 10, 7, 11, 12] # dress, jumpsuit, coat, scarf, skirt (handles misclass)
        elif category == "outer": target_ids = [7] # coat
        else:
            # Check individual labels
            if category in self.labels:
                target_ids = [self.labels[category]]
                
        return np.isin(parsing, target_ids)

    def build_category_masks(self, parsing: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "top": self.get_mask_for_category(parsing, "top"),
            "bottom": self.get_mask_for_category(parsing, "bottom"),
            "dress": self.get_mask_for_category(parsing, "dress"),
            "outer": self.get_mask_for_category(parsing, "outer"),
        }

    def get_garment_mask(self, parsing: np.ndarray) -> np.ndarray:
        garment_ids = [5, 6, 7, 9, 10, 11, 12]
        return np.isin(parsing, garment_ids)

    def get_occluder_mask(self, parsing: np.ndarray) -> np.ndarray:
        # Items that should be removed to isolate the garment:
        # 1: hat, 2: hair, 3: glove, 4: sunglasses, 8: socks, 13: face, 
        # 14-15: arms, 16-17: legs, 18-19: shoes
        occluder_ids = [1, 2, 3, 4, 8, 13, 14, 15, 16, 17, 18, 19]
        return np.isin(parsing, occluder_ids)

if __name__ == "__main__":
    # Isolated Test
    logging.basicConfig(level=logging.INFO)
    parser = HumanParser()
    print("Human Parser Initialized.")

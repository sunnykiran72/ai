import os
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from ai.shared.image_ops import (
    resize_mask_to_image, binary_open, binary_close, build_soft_alpha
)

logger = logging.getLogger("glamify-ai")

@dataclass
class YoloCropperConfig:
    use_human_parser: bool = True
    min_component_area_ratio: float = 0.001
    min_item_width_px: int = 32
    min_item_height_px: int = 32
    min_garment_overlap_ratio: float = 0.15
    yolo_min_conf: float = 0.25
    yolo_iou: float = 0.45
    enable_waist_split: bool = True
    mask_refinement_enabled: bool = True
    dress_second_component_enable: bool = True
    
class YoloCropper:
    """
    Stateless garment detector/cropper logic.
    Actual model execution is injected via predictor to keep modules/ pure.
    """
    
    def __init__(
        self,
        config: Optional[YoloCropperConfig] = None,
        predictor: Optional[Callable[[np.ndarray, float, float], Any]] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or YoloCropperConfig()
        self.predictor = predictor
        # Retained only for backwards compatibility in logs/config.
        self.model_path = model_path or os.getenv("YOLO_MODEL_PATH", "yolov8x-seg.pt")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def detect_instances(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Base detection with heuristic filtering.
        """
        if self.predictor is None:
            raise RuntimeError("YoloCropper predictor is not configured.")

        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        
        prediction = self.predictor(
            rgb,
            self.config.yolo_min_conf,
            self.config.yolo_iou,
        )
        result = prediction[0] if isinstance(prediction, (list, tuple)) else prediction

        if result is None:
            return []

        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None or boxes is None:
            return []

        mask_data = self._to_numpy(getattr(masks, "data", []))
        box_data = self._to_numpy(getattr(boxes, "data", []))
        if len(mask_data) == 0 or len(box_data) == 0:
            return []

        names = getattr(result, "names", {})
        instances = []
        
        for idx, raw_mask in enumerate(mask_data):
            if idx >= len(box_data):
                break

            # Resize mask to original image size
            mask = resize_mask_to_image(raw_mask, w, h) > 0.5
            area = int(mask.sum())
            if area < (h * w * self.config.min_component_area_ratio):
                continue
                
            conf = float(box_data[idx, 4])
            cls_id = int(box_data[idx, 5])
            label = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else str(cls_id)
            
            # Bounding box from mask
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            
            instances.append({
                "mask": mask,
                "bbox": bbox,
                "area": area,
                "confidence": conf,
                "label": label,
                "class_id": cls_id,
                "source": "yolo"
            })
            
        return instances

    def refine_mask(self, mask: np.ndarray, garment_type: str = "all") -> np.ndarray:
        """
        Refines a binary mask by keeping only the largest connected components.
        """
        if not self.config.mask_refinement_enabled:
            return mask
            
        import cv2
        binary = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return mask
            
        # Keep largest component
        best_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        refined = (labels == best_id)
        
        # Morphological clean up
        refined = binary_open(refined, 3)
        refined = binary_close(refined, 3)
        
        return refined

    def get_crops(self, image: Image.Image, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes instances into finalized crops with metadata.
        """
        items = []
        for idx, inst in enumerate(instances):
            mask = inst["mask"]
            bbox = inst["bbox"]
            
            # Simple padding logic for now (can expand to v2 logic)
            pad = 20
            x0, y0, x1, y1 = bbox
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(image.width, x1 + pad)
            y1 = min(image.height, y1 + pad)
            
            crop_img = image.crop((x0, y0, x1, y1))
            crop_mask = mask[y0:y1, x0:x1]
            
            # Alpha-isolated preview
            alpha = build_soft_alpha(crop_mask)
            rgba = np.dstack([np.asarray(crop_img.convert("RGB")), alpha])
            isolated_img = Image.fromarray(rgba)
            
            items.append({
                "id": idx,
                "label": inst["label"],
                "confidence": inst["confidence"],
                "image": crop_img,
                "isolated_image": isolated_img,
                "bbox": [x0, y0, x1, y1],
                "mask": crop_mask
            })
        return items

if __name__ == "__main__":
    # Isolated Test
    logging.basicConfig(level=logging.INFO)
    cropper = YoloCropper()
    print("YOLO High-Fidelity Cropper Initialized.")

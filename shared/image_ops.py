import io
import base64
import numpy as np
from PIL import Image
from typing import Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

def resize_mask_to_image(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    if cv2 is not None:
        return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    pil = Image.fromarray((mask.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    pil = pil.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0

def binary_open(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    # Fallback to simple mask ops or return as is
    return mask

def binary_close(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return mask

def build_soft_alpha(mask: np.ndarray, feather_px: int = 2) -> np.ndarray:
    """
    Creates a feathered alpha mask from a binary mask.
    """
    mask_u8 = (mask.astype(np.uint8) * 255)
    if cv2 is not None and feather_px > 0:
        kernel_size = (feather_px * 2) + 1
        alpha = cv2.GaussianBlur(mask_u8, (kernel_size, kernel_size), 0)
        return alpha
    return mask_u8

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def download_image(url: str, timeout: int = 15, preserve_alpha: bool = False) -> Image.Image:
    import requests
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content))
    if preserve_alpha:
        return image.convert("RGBA")
    return image.convert("RGB")

def bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union_area = area_a + area_b - inter_area
    return inter_area / max(1, union_area)

def rgb_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    """
    Converts an [N, 3] or [H, W, 3] RGB array (0-255) to CIELAB.
    """
    if cv2 is not None:
        # OpenCV expects a 3D array for cvtColor, reshape if N, 3
        is_1d = (rgb_array.ndim == 2)
        if is_1d:
            inp = rgb_array.reshape(1, -1, 3).astype(np.uint8)
        else:
            inp = rgb_array.astype(np.uint8)
        lab = cv2.cvtColor(inp, cv2.COLOR_RGB2LAB)
        if is_1d:
            return lab.reshape(-1, 3).astype(np.float32)
        return lab.astype(np.float32)
    
    # Simple fallback if cv2 not available (not perceptual uniform but better than raw RGB)
    return rgb_array.astype(np.float32)

def delta_e_cie76(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """
    Euclidean distance in CIELAB space.
    Approximation of perceptual color difference.
    """
    # Just Euclidean distance in Lab space
    diff = lab_a.astype(np.float32) - lab_b.astype(np.float32)
    return float(np.sqrt(np.sum(diff**2)))

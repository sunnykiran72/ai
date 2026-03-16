"""
Image preprocessing utilities for the fashion analysis system.

This module contains functions for image manipulation, cropping, resizing,
mask operations, background removal, and image enhancement.
"""

import io
import logging
import os
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from utils.validation import normalize_garment_type

logger = logging.getLogger(__name__)

# Configuration constants
ANALYZE_GARMENT_ALPHA_THRESHOLD = max(0, min(255, int(os.getenv("ANALYZE_GARMENT_ALPHA_THRESHOLD", "12"))))
ANALYZE_GARMENT_WHITE_THRESHOLD = max(200, min(255, int(os.getenv("ANALYZE_GARMENT_WHITE_THRESHOLD", "246"))))
ANALYZE_GARMENT_ENHANCE_ENABLED = os.getenv("ANALYZE_GARMENT_ENHANCE_ENABLED", "1") == "1"
ANALYZE_GARMENT_ENHANCE_SHARPNESS = float(os.getenv("ANALYZE_GARMENT_ENHANCE_SHARPNESS", "1.22"))
ANALYZE_GARMENT_ENHANCE_CONTRAST = float(os.getenv("ANALYZE_GARMENT_ENHANCE_CONTRAST", "1.08"))
ANALYZE_GARMENT_ENHANCE_COLOR = float(os.getenv("ANALYZE_GARMENT_ENHANCE_COLOR", "1.04"))
ANALYZE_GARMENT_ENHANCE_BRIGHTNESS = float(os.getenv("ANALYZE_GARMENT_ENHANCE_BRIGHTNESS", "1.02"))
ANALYZE_GARMENT_ENHANCE_LIGHTING_AUTO = os.getenv("ANALYZE_GARMENT_ENHANCE_LIGHTING_AUTO", "0") == "1"
ANALYZE_BIREFNET_MODEL_ID = os.getenv("ANALYZE_BIREFNET_MODEL_ID", "ZhengPeng7/BiRefNet").strip()
ANALYZE_BIREFNET_INPUT_SIZE = max(512, int(os.getenv("ANALYZE_BIREFNET_INPUT_SIZE", "1024")))
USER_PREP_BG_BACKEND = "birefnet"

# Global model cache
_birefnet_model_cache = None
_rembg_session_cache = None


def crop_image(image: Image.Image, bbox: List[int]) -> Image.Image:
    """
    Crop image to specified bounding box.
    
    Args:
        image: Input PIL Image
        bbox: Bounding box as [x0, y0, x1, y1]
        
    Returns:
        Cropped PIL Image
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.width, x1)
    y1 = min(image.height, y1)
    
    if x1 <= x0 or y1 <= y0:
        return image
        
    return image.crop((x0, y0, x1, y1))


def resize_image(image: Image.Image, max_side: int, min_side: int) -> Image.Image:
    """
    Resize image while preserving aspect ratio and detail.
    Never upscales small images.
    
    Args:
        image: Input PIL Image
        max_side: Maximum dimension for longest side
        min_side: Minimum dimension for shortest side
        
    Returns:
        Resized PIL Image
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w <= 0 or h <= 0:
        return rgb

    longest = max(w, h)
    shortest = min(w, h)
    scale = min(1.0, float(max_side) / float(longest))

    # Prevent over-downscaling thin details
    if shortest > min_side and (shortest * scale) < min_side:
        scale = min(1.0, float(min_side) / float(shortest))

    if scale >= 0.999:
        return rgb

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return rgb.resize((new_w, new_h), Image.BICUBIC)


def pad_image(image: Image.Image, aspect_w: int, aspect_h: int) -> Image.Image:
    """
    Pad image to fit specified aspect ratio.
    
    Args:
        image: Input PIL Image
        aspect_w: Target aspect ratio width
        aspect_h: Target aspect ratio height
        
    Returns:
        Padded PIL Image
    """
    iw, ih = image.size
    if iw < 1 or ih < 1:
        return image
    
    target_ratio = float(aspect_w) / float(aspect_h)
    cur_ratio = float(iw) / float(ih)
    
    if abs(cur_ratio - target_ratio) < 1e-4:
        return image

    if cur_ratio > target_ratio:
        out_w = iw
        out_h = int(round(iw / target_ratio))
    else:
        out_h = ih
        out_w = int(round(ih * target_ratio))

    out_w = max(out_w, iw)
    out_h = max(out_h, ih)
    
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", (out_w, out_h), (255, 255, 255, 0))
    else:
        canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    
    paste_x = (out_w - iw) // 2
    paste_y = (out_h - ih) // 2
    canvas.paste(image, (paste_x, paste_y), image if image.mode == "RGBA" else None)
    return canvas


def bbox_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        bbox1: First bounding box as [x0, y0, x1, y1]
        bbox2: Second bounding box as [x0, y0, x1, y1]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return float(inter_area) / float(union_area)


def expand_bbox(bbox: List[int], width: int, height: int, pad: int) -> List[int]:
    """
    Expand bounding box by padding amount, clamped to image bounds.
    
    Args:
        bbox: Bounding box as [x0, y0, x1, y1]
        width: Image width
        height: Image height
        pad: Padding amount in pixels
        
    Returns:
        Expanded bounding box
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width, x1 + pad),
        min(height, y1 + pad),
    ]


def merge_bboxes(bboxes: List[List[int]]) -> Optional[List[int]]:
    """
    Merge multiple bounding boxes into a single encompassing box.
    
    Args:
        bboxes: List of bounding boxes as [x0, y0, x1, y1]
        
    Returns:
        Merged bounding box or None if input is empty
    """
    if not bboxes:
        return None
    
    x_mins = [bbox[0] for bbox in bboxes]
    y_mins = [bbox[1] for bbox in bboxes]
    x_maxs = [bbox[2] for bbox in bboxes]
    y_maxs = [bbox[3] for bbox in bboxes]
    
    return [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Dilate binary mask using morphological operations.
    
    Args:
        mask: Binary mask as numpy array
        iterations: Number of dilation iterations
        
    Returns:
        Dilated mask
    """
    arr = np.asarray(mask).astype(bool)
    if int(iterations) <= 0:
        return arr
    
    try:
        import cv2
        dilated = cv2.dilate(
            arr.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=max(1, int(iterations)),
        )
        return dilated > 0
    except Exception:
        return arr


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Erode binary mask using morphological operations.
    
    Args:
        mask: Binary mask as numpy array
        iterations: Number of erosion iterations
        
    Returns:
        Eroded mask
    """
    arr = np.asarray(mask).astype(bool)
    if int(iterations) <= 0:
        return arr
    
    try:
        import cv2
        eroded = cv2.erode(
            arr.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=max(1, int(iterations)),
        )
        return eroded > 0
    except Exception:
        return arr


def feather_mask_edges(mask: np.ndarray, feather_px: int = 1) -> np.ndarray:
    """
    Apply feathering to mask edges for smooth transitions.
    
    Args:
        mask: Binary mask as numpy array
        feather_px: Feathering radius in pixels
        
    Returns:
        Feathered mask with soft edges
    """
    if feather_px <= 0:
        return mask.astype(np.uint8)
    
    try:
        from scipy.ndimage import distance_transform_edt
        
        # Convert to binary
        binary_mask = np.asarray(mask).astype(bool)
        
        # Calculate distance from edges
        dist_inside = distance_transform_edt(binary_mask)
        dist_outside = distance_transform_edt(~binary_mask)
        
        # Create soft transition
        soft_mask = np.zeros_like(mask, dtype=np.float32)
        soft_mask[binary_mask] = np.minimum(1.0, dist_inside[binary_mask] / feather_px)
        soft_mask[~binary_mask] = np.maximum(0.0, 1.0 - dist_outside[~binary_mask] / feather_px)
        
        return (soft_mask * 255).astype(np.uint8)
    except Exception:
        return mask.astype(np.uint8)


def apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Apply mask to image, setting masked areas to transparent.
    
    Args:
        image: Input PIL Image
        mask: Binary mask as numpy array
        
    Returns:
        Masked image with alpha channel
    """
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba).copy()
    alpha = arr[:, :, 3]
    
    keep = np.asarray(mask).astype(bool)
    if keep.shape != alpha.shape:
        keep = np.zeros_like(alpha, dtype=bool)
    
    alpha[~keep] = 0
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def remove_background_rembg(image_bytes: bytes) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """
    Remove background using REMBG library.
    
    Args:
        image_bytes: Input image as bytes
        
    Returns:
        Tuple of (output_bytes, metadata_dict)
    """
    session = _get_rembg_session()
    if not session:
        return None, {"mode": "rembg_unavailable"}
    
    try:
        from rembg import remove

        output = remove(
            image_bytes,
            session=session,
            alpha_matting=False,
            only_mask=False,
            post_process_mask=False,
            force_return_bytes=True,
        )
        out_img = Image.open(io.BytesIO(output)).convert("RGBA")
        out = io.BytesIO()
        out_img.save(out, format="PNG")
        alpha = np.asarray(out_img)[:, :, 3]
        return out.getvalue(), {
            "mode": "rembg_isnet_general_use",
            "original_size": {"width": int(out_img.width), "height": int(out_img.height)},
            "output_size": {"width": int(out_img.width), "height": int(out_img.height)},
            "opaque_pixels": int(np.sum(alpha >= 250)),
            "transparent_pixels": int(np.sum(alpha <= 4)),
        }
    except Exception as rembg_err:
        logger.warning(f"rembg background removal failed: {rembg_err}")
        return None, {"mode": "rembg_failed", "error": str(rembg_err)}


def remove_background_birefnet(image_bytes: bytes) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """
    Remove background using BiRefNet model.
    
    Args:
        image_bytes: Input image as bytes
        
    Returns:
        Tuple of (output_bytes, metadata_dict)
    """
    bundle = _get_birefnet_model()
    if not bundle:
        return None, {"mode": "birefnet_unavailable"}
    
    try:
        from torchvision import transforms

        model = bundle["model"]
        device = bundle["device"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size
        
        transform = transforms.Compose([
            transforms.Resize((ANALYZE_BIREFNET_INPUT_SIZE, ANALYZE_BIREFNET_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(tensor)
        
        if isinstance(preds, (list, tuple)):
            pred = preds[-1]
        elif isinstance(preds, dict):
            pred = preds.get("pred") or preds.get("out") or next(iter(preds.values()))
        else:
            pred = preds
        
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        
        pred = torch.sigmoid(pred)
        if pred.ndim == 4:
            pred = pred[0, 0]
        elif pred.ndim == 3:
            pred = pred[0]
        
        mask = pred.detach().float().cpu().numpy()
        mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        alpha = Image.fromarray(mask_u8, mode="L").resize((orig_w, orig_h), Image.LANCZOS)

        rgba = image.convert("RGBA")
        rgba.putalpha(alpha)
        out = io.BytesIO()
        rgba.save(out, format="PNG")
        alpha_arr = np.asarray(alpha)
        
        return out.getvalue(), {
            "mode": "birefnet",
            "model_id": ANALYZE_BIREFNET_MODEL_ID,
            "input_size": ANALYZE_BIREFNET_INPUT_SIZE,
            "original_size": {"width": int(orig_w), "height": int(orig_h)},
            "output_size": {"width": int(orig_w), "height": int(orig_h)},
            "opaque_pixels": int(np.sum(alpha_arr >= 250)),
            "transparent_pixels": int(np.sum(alpha_arr <= 4)),
        }
    except Exception as birefnet_err:
        logger.warning(f"BiRefNet background removal failed: {birefnet_err}")
        return None, {"mode": "birefnet_failed", "error": str(birefnet_err)}


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Enhance garment image with brightness, contrast, color, and sharpness adjustments.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Enhanced PIL Image
    """
    if not ANALYZE_GARMENT_ENHANCE_ENABLED:
        return image

    rgba = image.convert("RGBA")
    alpha = rgba.split()[-1]
    
    # Standard enhancements
    # 1. Lighting Balance (Fixes the 'half section shadow' issue)
    if ANALYZE_GARMENT_ENHANCE_LIGHTING_AUTO:
        try:
            # Convert to HSV to ensure hue-safe brightness adjustments
            hsv = np.asarray(rgba.convert("HSV")).astype(np.float32)
            v_chan = hsv[:, :, 2]
            mask = np.asarray(alpha) > 10
            
            if np.any(mask):
                h, w = v_chan.shape[0], v_chan.shape[1]
                
                # --- HORIZONTAL CORRECTION ---
                h_brightness = np.zeros(w)
                for x in range(w):
                    col_mask = mask[:, x]
                    if np.any(col_mask):
                        h_brightness[x] = np.mean(v_chan[col_mask, x])
                
                win = max(5, w // 10)
                h_brightness = np.convolve(h_brightness, np.ones(win)/win, mode="same")
                left_avg = np.mean(h_brightness[:w//2][h_brightness[:w//2] > 0])
                right_avg = np.mean(h_brightness[w//2:][h_brightness[w//2:] > 0])
                
                if not np.isnan(left_avg) and not np.isnan(right_avg) and left_avg > 2 and right_avg > 2:
                    h_ratio = right_avg / left_avg
                    if abs(1.0 - h_ratio) > 0.10:
                        logger.info(f"Applying horizontal lighting correction: ratio={h_ratio:.2f}")
                        h_gradient = np.linspace(h_ratio, 1.0, w)
                        v_chan *= h_gradient

                # --- VERTICAL CORRECTION (Gamma-based) ---
                v_brightness = np.zeros(h)
                for y in range(h):
                    row_mask = mask[y, :]
                    if np.any(row_mask):
                        v_brightness[y] = np.mean(v_chan[y, row_mask])
                
                v_win = max(5, h // 10)
                v_brightness = np.convolve(v_brightness, np.ones(v_win)/v_win, mode="same")
                
                top_zone = np.mean(v_brightness[:h//6][v_brightness[:h//6] > 0])
                bot_zone = np.mean(v_brightness[h//2:][v_brightness[h//2:] > 0])
                
                if not np.isnan(top_zone) and not np.isnan(bot_zone) and top_zone > 2 and bot_zone > 2:
                    v_ratio = bot_zone / top_zone
                    if v_ratio > 1.15: # Only lift if top is 15%+ darker
                        logger.info(f"Applying vertical gamma-lift (HSV): ratio={v_ratio:.2f}")
                        # Damping the ratio to prevent over-lifting and 'washing out' colors
                        gamma = 1.0 / (v_ratio ** 0.4) 
                        v_gamma_grad = np.linspace(gamma, 1.0, h).reshape(-1, 1)
                        v_chan = 255.0 * (v_chan / 255.0) ** v_gamma_grad

                # --- GLOBAL LIFT FOR DARK FABRICS ---
                avg_v = np.mean(v_chan[mask])
                if avg_v < 55:
                    logger.info(f"Applying deep shadow boost (HSV avg={avg_v:.1f})")
                    # Subtle 10% lift for very dark pixels only
                    v_chan = np.where(v_chan < 50, v_chan * 1.10, v_chan)

                hsv[:, :, 2] = np.clip(v_chan, 0, 255)
                # Convert back to RGB via HSV
                rgba = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB").convert("RGBA")
                rgba.putalpha(alpha)
        except Exception as e:
            logger.error(f"Adaptive lighting correction failed: {e}", exc_info=True)
            # Fallback to original image if correction fails to avoid bad artifacts
            rgba = image.convert("RGBA")

    # 2. PIL Image Enhancements
    res = rgba.convert("RGB")
    if ANALYZE_GARMENT_ENHANCE_BRIGHTNESS != 1.0:
        res = ImageEnhance.Brightness(res).enhance(ANALYZE_GARMENT_ENHANCE_BRIGHTNESS)
    if ANALYZE_GARMENT_ENHANCE_CONTRAST != 1.0:
        res = ImageEnhance.Contrast(res).enhance(ANALYZE_GARMENT_ENHANCE_CONTRAST)
    if ANALYZE_GARMENT_ENHANCE_COLOR != 1.0:
        res = ImageEnhance.Color(res).enhance(ANALYZE_GARMENT_ENHANCE_COLOR)
    if ANALYZE_GARMENT_ENHANCE_SHARPNESS != 1.0:
        # Sharpness can cause halos if too high; use UnsharpMask for better results
        res = ImageEnhance.Sharpness(res).enhance(ANALYZE_GARMENT_ENHANCE_SHARPNESS)
        res = res.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3))

    res.putalpha(alpha)
    return res


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """
    Adjust image brightness.
    
    Args:
        image: Input PIL Image
        factor: Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
        
    Returns:
        Brightness-adjusted image
    """
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """
    Adjust image contrast.
    
    Args:
        image: Input PIL Image
        factor: Contrast factor (1.0 = no change, >1.0 = more contrast, <1.0 = less contrast)
        
    Returns:
        Contrast-adjusted image
    """
    return ImageEnhance.Contrast(image).enhance(factor)


# Additional utility functions for bbox operations
def bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    """
    Extract bounding box from binary mask.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Bounding box as [x0, y0, x1, y1] or None if mask is empty
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_y_overlap_ratio(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate vertical overlap ratio between two bounding boxes.
    
    Args:
        bbox1: First bounding box as [x0, y0, x1, y1]
        bbox2: Second bounding box as [x0, y0, x1, y1]
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    ay0, ay1 = int(bbox1[1]), int(bbox1[3])
    by0, by1 = int(bbox2[1]), int(bbox2[3])
    inter = max(0, min(ay1, by1) - max(ay0, by0))
    ha = max(1, ay1 - ay0)
    hb = max(1, by1 - by0)
    return float(inter) / float(max(1, min(ha, hb)))


def bbox_x_overlap_ratio(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate horizontal overlap ratio between two bounding boxes.
    
    Args:
        bbox1: First bounding box as [x0, y0, x1, y1]
        bbox2: Second bounding box as [x0, y0, x1, y1]
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    ax0, ax1 = int(bbox1[0]), int(bbox1[2])
    bx0, bx1 = int(bbox2[0]), int(bbox2[2])
    inter = max(0, min(ax1, bx1) - max(ax0, bx0))
    wa = max(1, ax1 - ax0)
    wb = max(1, bx1 - bx0)
    return float(inter) / float(max(1, min(wa, wb)))


def bbox_x_gap(bbox1: List[int], bbox2: List[int]) -> int:
    """
    Calculate horizontal gap between two bounding boxes.
    
    Args:
        bbox1: First bounding box as [x0, y0, x1, y1]
        bbox2: Second bounding box as [x0, y0, x1, y1]
        
    Returns:
        Gap in pixels (0 if overlapping)
    """
    ax0, ax1 = int(bbox1[0]), int(bbox1[2])
    bx0, bx1 = int(bbox2[0]), int(bbox2[2])
    if ax1 >= bx0 and bx1 >= ax0:
        return 0
    if ax1 < bx0:
        return int(bx0 - ax1)
    return int(ax0 - bx1)


def square_bbox_from_bbox(bbox: List[int], image_width: int, image_height: int, padding_ratio: float) -> List[int]:
    """
    Create square bounding box from rectangular bbox.
    
    Args:
        bbox: Input bounding box as [x0, y0, x1, y1]
        image_width: Image width
        image_height: Image height
        padding_ratio: Padding ratio (0.0 to 0.8)
        
    Returns:
        Square bounding box
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    safe_pad = max(0.0, min(0.8, float(padding_ratio)))
    side = int(round(max(bw, bh) * (1.0 + (2.0 * safe_pad))))
    side = max(2, side)

    cx = float(x0 + x1) / 2.0
    cy = float(y0 + y1) / 2.0
    sx0 = int(round(cx - (side / 2.0)))
    sy0 = int(round(cy - (side / 2.0)))
    return [sx0, sy0, sx0 + side, sy0 + side]


def crop_square_with_padding(
    image: Image.Image,
    square_bbox: List[int],
    fill_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Crop image to square with padding if needed.
    
    Args:
        image: Input PIL Image
        square_bbox: Square bounding box as [x0, y0, x1, y1]
        fill_rgb: Fill color for padding areas
        
    Returns:
        Square cropped image
    """
    x0, y0, x1, y1 = [int(v) for v in square_bbox]
    side = max(2, int(x1 - x0), int(y1 - y0))
    canvas = Image.new("RGB", (side, side), fill_rgb)

    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(image.width, x1)
    iy1 = min(image.height, y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return canvas

    src_crop = image.crop((ix0, iy0, ix1, iy1)).convert("RGB")
    paste_x = int(ix0 - x0)
    paste_y = int(iy0 - y0)
    canvas.paste(src_crop, (paste_x, paste_y))
    return canvas


def expand_bbox_by_type(
    bbox: List[int],
    garment_type: str,
    image_width: int,
    image_height: int,
) -> List[int]:
    """
    Expand bounding box based on garment type with type-specific padding.
    
    Args:
        bbox: Input bounding box as [x0, y0, x1, y1]
        garment_type: Type of garment ('top', 'bottom', 'dress', etc.)
        image_width: Image width
        image_height: Image height
        
    Returns:
        Expanded bounding box
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    gt = normalize_garment_type(garment_type) or "top"
    if gt == "top":
        # Parser top masks frequently truncate one-sleeve/long-sleeve regions.
        # Keep richer context for captioning + flux garment reconstruction.
        x_pad = int(bw * 0.12)
        y_pad_top = int(bh * 0.05)
        y_pad_bottom = int(bh * 0.36)
        if y0 < int(image_height * 0.38):
            y_pad_bottom = max(y_pad_bottom, int(image_height * 0.20))
    elif gt == "bottom":
        x_pad = int(bw * 0.03)
        y_pad_top = int(bh * 0.06)
        y_pad_bottom = int(bh * 0.04)
    elif gt == "dress":
        x_pad = int(bw * 0.03)
        y_pad_top = int(bh * 0.02)
        y_pad_bottom = int(bh * 0.04)
    else:
        x_pad = int(bw * 0.03)
        y_pad_top = int(bh * 0.02)
        y_pad_bottom = int(bh * 0.04)

    ex0 = max(0, x0 - x_pad)
    ey0 = max(0, y0 - y_pad_top)
    ex1 = min(image_width, x1 + x_pad)
    ey1 = min(image_height, y1 + y_pad_bottom)
    return [ex0, ey0, ex1, ey1]


def expand_bbox_with_ratios(
    bbox: List[int],
    image_width: int,
    image_height: int,
    x_pad_ratio: float,
    y_pad_top_ratio: float,
    y_pad_bottom_ratio: float,
) -> List[int]:
    """
    Expand bounding box with custom padding ratios.
    
    Args:
        bbox: Input bounding box as [x0, y0, x1, y1]
        image_width: Image width
        image_height: Image height
        x_pad_ratio: Horizontal padding ratio
        y_pad_top_ratio: Top padding ratio
        y_pad_bottom_ratio: Bottom padding ratio
        
    Returns:
        Expanded bounding box
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    x_pad = int(bw * max(0.0, x_pad_ratio))
    y_pad_top = int(bh * max(0.0, y_pad_top_ratio))
    y_pad_bottom = int(bh * max(0.0, y_pad_bottom_ratio))
    return [
        max(0, x0 - x_pad),
        max(0, y0 - y_pad_top),
        min(image_width, x1 + x_pad),
        min(image_height, y1 + y_pad_bottom),
    ]


def robust_bbox_from_component_mask(component_mask: np.ndarray, fallback_bbox: List[int]) -> List[int]:
    """
    Extract robust bounding box from component mask, ignoring sparse outlier pixels.
    
    Args:
        component_mask: Component mask as numpy array
        fallback_bbox: Fallback bounding box if extraction fails
        
    Returns:
        Robust bounding box
    """
    try:
        mask = np.asarray(component_mask).astype(bool)
        if mask.ndim != 2 or mask.size == 0 or int(mask.sum()) <= 0:
            return [int(v) for v in fallback_bbox]

        h, w = mask.shape
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)

        # Keep rows/cols that have meaningful support, not isolated specks.
        row_min = max(1, int(w * 0.006))
        col_min = max(1, int(h * 0.006))

        valid_rows = np.where(row_counts >= row_min)[0]
        valid_cols = np.where(col_counts >= col_min)[0]
        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return [int(v) for v in fallback_bbox]

        x0 = int(valid_cols.min())
        y0 = int(valid_rows.min())
        x1 = int(valid_cols.max()) + 1
        y1 = int(valid_rows.max()) + 1
        if x1 <= x0 or y1 <= y0:
            return [int(v) for v in fallback_bbox]
        return [x0, y0, x1, y1]
    except Exception:
        return [int(v) for v in fallback_bbox]


def extract_alpha_mask(image: Image.Image, threshold: int = 24) -> Optional[np.ndarray]:
    """
    Extract foreground mask from alpha channel.
    
    Args:
        image: Input PIL Image
        threshold: Alpha threshold for foreground detection
        
    Returns:
        Binary mask or None if not meaningful
    """
    try:
        rgba = image.convert("RGBA")
        alpha = np.array(rgba, dtype=np.uint8)[:, :, 3]
        mask = alpha >= int(max(1, threshold))
        keep = int(np.sum(mask))
        if keep < 64:
            return None
        coverage = float(keep) / float(mask.shape[0] * mask.shape[1])
        if coverage >= 0.985:
            # Mostly opaque image; alpha likely not meaningful for foreground isolation.
            return None
        return mask
    except Exception:
        return None


def connected_support_mask(base_mask: np.ndarray, support_mask: np.ndarray, dilate_iters: int = 3) -> np.ndarray:
    """
    Create connected support mask by dilating base mask and finding connected components.
    
    Args:
        base_mask: Base binary mask
        support_mask: Support binary mask
        dilate_iters: Number of dilation iterations
        
    Returns:
        Connected support mask
    """
    base = np.asarray(base_mask).astype(bool)
    sup = np.asarray(support_mask).astype(bool)
    if base.shape != sup.shape:
        return np.zeros_like(base, dtype=bool)
    if not np.any(base) or not np.any(sup):
        return np.zeros_like(base, dtype=bool)
    
    try:
        from scipy.ndimage import binary_dilation, label

        seed = binary_dilation(base, iterations=max(1, int(dilate_iters)))
        labeled, num_labels = label(sup)
        connected = np.zeros_like(sup, dtype=bool)
        for comp_id in range(1, int(num_labels) + 1):
            comp = labeled == comp_id
            if np.any(comp & seed):
                connected |= comp
        return connected
    except Exception:
        # Conservative fallback: keep only direct overlap.
        return sup & base


def crop_rgba_with_mask(
    image: Image.Image,
    object_mask: np.ndarray,
    bbox: List[int],
    x_pad_ratio: float = 0.03,
    y_pad_top_ratio: float = 0.02,
    y_pad_bottom_ratio: float = 0.04,
) -> Tuple[Image.Image, List[int]]:
    """
    Crop RGBA image with mask applied and padding.
    
    Args:
        image: Input PIL Image
        object_mask: Object mask as numpy array
        bbox: Bounding box as [x0, y0, x1, y1]
        x_pad_ratio: Horizontal padding ratio
        y_pad_top_ratio: Top padding ratio
        y_pad_bottom_ratio: Bottom padding ratio
        
    Returns:
        Tuple of (cropped_image, expanded_bbox)
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    ex0 = max(0, x0 - int(bw * max(0.0, x_pad_ratio)))
    ey0 = max(0, y0 - int(bh * max(0.0, y_pad_top_ratio)))
    ex1 = min(image.width, x1 + int(bw * max(0.0, x_pad_ratio)))
    ey1 = min(image.height, y1 + int(bh * max(0.0, y_pad_bottom_ratio)))
    if ex1 <= ex0 or ey1 <= ey0:
        ex0, ey0, ex1, ey1 = x0, y0, x1, y1

    rgba = image.convert("RGBA")
    arr = np.asarray(rgba).copy()
    alpha = arr[:, :, 3]
    keep = np.asarray(object_mask).astype(bool)
    if keep.shape != alpha.shape:
        keep = np.zeros_like(alpha, dtype=bool)
    alpha[~keep] = 0
    arr[:, :, 3] = alpha
    masked = Image.fromarray(arr, mode="RGBA")
    crop = masked.crop((ex0, ey0, ex1, ey1))
    return crop, [int(ex0), int(ey0), int(ex1), int(ey1)]


def mask_connected_components(mask: np.ndarray, min_pixels: int) -> List[Dict[str, Any]]:
    """
    Find connected components in binary mask.
    
    Args:
        mask: Binary mask as numpy array
        min_pixels: Minimum pixels for valid component
        
    Returns:
        List of component dictionaries with bbox, area, and mask
    """
    clean = np.asarray(mask).astype(bool)
    if clean.size == 0:
        return []
    if int(clean.sum()) < max(1, int(min_pixels)):
        return []

    try:
        import cv2
    except Exception:
        cv2 = None

    out: List[Dict[str, Any]] = []
    if cv2 is None:
        bbox = bbox_from_mask(clean)
        if bbox is None:
            return []
        out.append({"bbox": bbox, "area": int(clean.sum()), "mask": clean})
        return out

    num_l, labels_l, stats_l, _ = cv2.connectedComponentsWithStats((clean.astype(np.uint8) * 255), connectivity=8)
    for i in range(1, int(num_l)):
        area = int(stats_l[i, cv2.CC_STAT_AREA])
        if area < max(1, int(min_pixels)):
            continue
        x = int(stats_l[i, cv2.CC_STAT_LEFT])
        y = int(stats_l[i, cv2.CC_STAT_TOP])
        w = int(stats_l[i, cv2.CC_STAT_WIDTH])
        h = int(stats_l[i, cv2.CC_STAT_HEIGHT])
        if w < 8 or h < 8:
            continue
        out.append(
            {
                "bbox": [x, y, x + w, y + h],
                "area": area,
                "mask": labels_l == i,
            }
        )
    out.sort(key=lambda item: int(item.get("area", 0)), reverse=True)
    return out


def content_bbox_from_image(image: Image.Image) -> Tuple[int, int, int, int]:
    """
    Extract content bounding box from image using alpha or color thresholding.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Content bounding box as (x0, y0, x1, y1)
    """
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba)
    alpha = arr[:, :, 3]
    content = alpha > ANALYZE_GARMENT_ALPHA_THRESHOLD
    if not np.any(content):
        rgb = arr[:, :, :3]
        content = np.any(rgb < ANALYZE_GARMENT_WHITE_THRESHOLD, axis=2)

    ys, xs = np.where(content)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, rgba.width, rgba.height)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


# Helper functions for model caching

def _get_rembg_session():
    """Get cached REMBG session."""
    global _rembg_session_cache
    if _rembg_session_cache is not None:
        return _rembg_session_cache
    
    try:
        from rembg import new_session
        _rembg_session_cache = new_session("isnet-general-use")
        return _rembg_session_cache
    except Exception as e:
        logger.warning(f"Failed to initialize REMBG session: {e}")
        return None


def _get_birefnet_model():
    """Get cached BiRefNet model."""
    global _birefnet_model_cache
    if _birefnet_model_cache is not None:
        return _birefnet_model_cache
    
    try:
        from transformers import AutoModelForImageSegmentation
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageSegmentation.from_pretrained(
            ANALYZE_BIREFNET_MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        
        _birefnet_model_cache = {"model": model, "device": device}
        return _birefnet_model_cache
    except Exception as e:
        logger.warning(f"Failed to initialize BiRefNet model: {e}")
        return None
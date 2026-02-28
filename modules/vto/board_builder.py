import io
import math
import logging
import numpy as np
from PIL import Image, ImageOps
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger("glamify-ai")

class BoardBuilder:
    """
    Arranges multiple garment images into a single 1024x1024 collage board.
    Optimized for Flux 2.0 CVTON context.
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (1024, 1024)):
        self.canvas_size = canvas_size
        self.background_color = (246, 246, 246)
        self.card_color = (255, 255, 255)
        self.padding = 20

    def get_layout_rects(self, count: int) -> List[Tuple[int, int, int, int]]:
        """
        Deterministic outfit-board zones:
        - left top: primary top
        - left bottom: primary bottom
        - right column: outerwear and others
        """
        cw, ch = self.canvas_size
        p = self.padding
        
        if count <= 0: return []
        if count == 1: return [(p, p, cw - 2 * p, ch - 2 * p)]
        
        # Split canvas into left (primary) and right (secondary)
        left_w = int((cw - 3 * p) * 0.65)
        right_w = cw - 3 * p - left_w
        
        if count == 2:
            h_each = (ch - 3 * p) // 2
            return [
                (p, p, left_w, h_each),           # Top item
                (p, 2 * p + h_each, left_w, h_each) # Bottom item
            ]
            
        # 3 or more items
        left_top_h = int((ch - 3 * p) * 0.5)
        left_bottom_h = ch - 3 * p - left_top_h
        
        rects = [
            (p, p, left_w, left_top_h),
            (p, 2 * p + left_top_h, left_w, left_bottom_h)
        ]
        
        right_count = count - 2
        right_h_each = (ch - (right_count + 1) * p) // right_count
        for i in range(right_count):
            rects.append((2 * p + left_w, p + i * (right_h_each + p), right_w, right_h_each))
            
        return rects

    def resize_to_fit(self, image: Image.Image, size: Tuple[int, int], background: Tuple[int, int, int]) -> Image.Image:
        target_w, target_h = size
        canvas = Image.new("RGB", (target_w, target_h), background)
        
        # Inner padding inside the card
        inner_p = 12
        max_w = max(1, target_w - 2 * inner_p)
        max_h = max(1, target_h - 2 * inner_p)
        
        img_w, img_h = image.size
        scale = min(max_w / img_w, max_h / img_h)
        nw, nh = max(1, int(img_w * scale)), max(1, int(img_h * scale))
        
        resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
        
        # Center on card
        ox = (target_w - nw) // 2
        oy = (target_h - nh) // 2
        
        if resized.mode == "RGBA":
            canvas.paste(resized.convert("RGB"), (ox, oy), resized.split()[-1])
        else:
            canvas.paste(resized.convert("RGB"), (ox, oy))
            
        return canvas

    def build_board(self, images: List[Image.Image]) -> Image.Image:
        """
        Creates the collage.
        """
        canvas = Image.new("RGB", self.canvas_size, self.background_color)
        rects = self.get_layout_rects(len(images))
        
        for img, rect in zip(images, rects):
            x, y, w, h = rect
            card = self.resize_to_fit(img, (w, h), self.card_color)
            canvas.paste(card, (x, y))
            
        return canvas

if __name__ == "__main__":
    # Test logic
    builder = BoardBuilder()
    print("Board Builder Initialized.")

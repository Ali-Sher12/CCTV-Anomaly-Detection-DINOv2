"""mask_utils.py — Unified zone class mask processing.

Zone classes:
  0 = IGNORE (Black)
  1 = MEDIUM (Green)
  2 = HIGH   (Red)

Implements:
- Nearest reference color Euclidean classification (MASK-02)
- Nearest-neighbor resizing (MASK-03)
- Cell dominant class reduction with tie-break HIGH > MEDIUM > IGNORE (MASK-04)
- Resilient fallback to all-MEDIUM on missing/corrupt mask (MASK-08)
"""

import os
import cv2
import numpy as np

ZONE_IGNORE = 0
ZONE_MEDIUM = 1
ZONE_HIGH = 2

# Reference colors in BGR (as returned by cv2.imread)
COLOR_BLACK_BGR = np.array([0, 0, 0], dtype=np.float32)
COLOR_GREEN_BGR = np.array([0, 255, 0], dtype=np.float32)
COLOR_RED_BGR   = np.array([0, 0, 255], dtype=np.float32)

# Reference colors in RGB (for PIL / MaskEditor)
COLOR_BLACK_RGB = (0, 0, 0)
COLOR_GREEN_RGB = (0, 255, 0)
COLOR_RED_RGB   = (255, 0, 0)


def classify_pixels_to_classes(bgr_image: np.ndarray) -> np.ndarray:
    """Classify each pixel in a BGR image to the nearest reference color class.
    
    Returns uint8 array of shape (H, W) with values in {0, 1, 2}.
    """
    img_f = bgr_image.astype(np.float32)
    # Compute squared Euclidean distances to the 3 reference colors
    d_black = np.sum((img_f - COLOR_BLACK_BGR) ** 2, axis=2)  # class 0
    d_green = np.sum((img_f - COLOR_GREEN_BGR) ** 2, axis=2)  # class 1
    d_red   = np.sum((img_f - COLOR_RED_BGR) ** 2, axis=2)    # class 2

    stacked = np.stack([d_black, d_green, d_red], axis=-1)  # shape (H, W, 3)
    # argmin gives index 0 (black), 1 (green), or 2 (red)
    return np.argmin(stacked, axis=-1).astype(np.uint8)


def reduce_pixel_classes_to_grid(pixel_classes: np.ndarray,
                                grid_h: int, grid_w: int) -> np.ndarray:
    """Reduce (H, W) pixel class array to (grid_h, grid_w) by dominant class.
    
    Tie-break order: HIGH (2) > MEDIUM (1) > IGNORE (0).
    """
    h, w = pixel_classes.shape[:2]
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    y_edges = np.linspace(0, h, grid_h + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_w + 1, dtype=int)

    for r in range(grid_h):
        y1, y2 = y_edges[r], y_edges[r + 1]
        for c in range(grid_w):
            x1, x2 = x_edges[c], x_edges[c + 1]
            cell = pixel_classes[y1:y2, x1:x2]
            if cell.size == 0:
                grid[r, c] = ZONE_MEDIUM
                continue

            count_0 = np.count_nonzero(cell == ZONE_IGNORE)
            count_1 = np.count_nonzero(cell == ZONE_MEDIUM)
            count_2 = np.count_nonzero(cell == ZONE_HIGH)

            # Tie-break: HIGH > MEDIUM > IGNORE
            max_count = max(count_0, count_1, count_2)
            if count_2 == max_count:
                grid[r, c] = ZONE_HIGH
            elif count_1 == max_count:
                grid[r, c] = ZONE_MEDIUM
            else:
                grid[r, c] = ZONE_IGNORE

    return grid


def build_zone_class_grid(mask_path: str,
                          frame_w: int,
                          frame_h: int,
                          grid_w: int,
                          grid_h: int) -> tuple[np.ndarray, str]:
    """Load mask from disk, resize with NEAREST, and compute zone_class_grid.
    
    Returns (zone_class_grid, warning_message).
    If mask is missing or corrupt, returns all-MEDIUM grid and warning.
    """
    warning = ""
    mask_bgr = None
    if mask_path and os.path.exists(mask_path):
        try:
            mask_bgr = cv2.imread(mask_path)
        except Exception as e:
            warning = f"Mask unavailable ({e}) — using all-MEDIUM zone"
            mask_bgr = None

    if mask_bgr is None or mask_bgr.size == 0:
        if not warning:
            warning = "Mask unavailable — using all-MEDIUM zone"
        grid = np.full((grid_h, grid_w), ZONE_MEDIUM, dtype=np.uint8)
        return grid, warning

    # Resize mask to camera frame size using nearest-neighbor
    if (mask_bgr.shape[1] != frame_w) or (mask_bgr.shape[0] != frame_h):
        mask_bgr = cv2.resize(mask_bgr, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    pixel_classes = classify_pixels_to_classes(mask_bgr)
    grid = reduce_pixel_classes_to_grid(pixel_classes, grid_h, grid_w)
    return grid, warning


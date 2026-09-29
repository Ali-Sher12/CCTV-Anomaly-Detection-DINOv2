import pytest
import numpy as np
import cv2
import os
import tempfile
from mask_utils import (
    ZONE_IGNORE, ZONE_MEDIUM, ZONE_HIGH,
    classify_pixels_to_classes,
    reduce_pixel_classes_to_grid,
    build_zone_class_grid
)

def compute_tier_for_scores(score_grid: np.ndarray,
                           zone_class_grid: np.ndarray,
                           thresholds: dict) -> np.ndarray:
    """Pure tier computation function matching MASK-01 / MASK-05."""
    tier_grid = np.full(score_grid.shape, "NORMAL", dtype=object)
    
    med_mask = (zone_class_grid == ZONE_MEDIUM)
    high_mask = (zone_class_grid == ZONE_HIGH)
    # IGNORE mask is excluded from any tier contribution (remains NORMAL)

    tier_grid[med_mask & (score_grid >= thresholds["ALERT"])] = "ALERT"
    tier_grid[med_mask & (score_grid >= thresholds["CRITICAL"])] = "CRITICAL"

    tier_grid[high_mask & (score_grid >= thresholds["HIGH_PRIORITY_ALERT"])] = "ALERT"
    tier_grid[high_mask & (score_grid >= thresholds["HIGH_PRIORITY_CRITICAL"])] = "CRITICAL"

    return tier_grid


def test_mask_worked_example_table():
    thresholds = {
        "ALERT": 45.0,
        "CRITICAL": 50.0,
        "HIGH_PRIORITY_ALERT": 35.0,
        "HIGH_PRIORITY_CRITICAL": 40.0,
    }

    # 5 rows corresponding to scores [30, 37, 42, 47, 52]
    # 3 columns: Col 0 = RED (HIGH=2), Col 1 = GREEN (MEDIUM=1), Col 2 = BLACK (IGNORE=0)
    scores = np.array([
        [30.0, 30.0, 30.0],
        [37.0, 37.0, 37.0],
        [42.0, 42.0, 42.0],
        [47.0, 47.0, 47.0],
        [52.0, 52.0, 52.0],
    ])

    zone_grid = np.array([
        [ZONE_HIGH, ZONE_MEDIUM, ZONE_IGNORE],
        [ZONE_HIGH, ZONE_MEDIUM, ZONE_IGNORE],
        [ZONE_HIGH, ZONE_MEDIUM, ZONE_IGNORE],
        [ZONE_HIGH, ZONE_MEDIUM, ZONE_IGNORE],
        [ZONE_HIGH, ZONE_MEDIUM, ZONE_IGNORE],
    ], dtype=np.uint8)

    tiers = compute_tier_for_scores(scores, zone_grid, thresholds)

    # Red / HIGH column (col 0):
    assert tiers[0, 0] == "NORMAL"
    assert tiers[1, 0] == "ALERT"
    assert tiers[2, 0] == "CRITICAL"
    assert tiers[3, 0] == "CRITICAL"
    assert tiers[4, 0] == "CRITICAL"

    # Green / MEDIUM column (col 1):
    assert tiers[0, 1] == "NORMAL"
    assert tiers[1, 1] == "NORMAL"
    assert tiers[2, 1] == "NORMAL"
    assert tiers[3, 1] == "ALERT"
    assert tiers[4, 1] == "CRITICAL"

    # Black / IGNORE column (col 2):
    assert tiers[0, 2] == "NORMAL"
    assert tiers[1, 2] == "NORMAL"
    assert tiers[2, 2] == "NORMAL"
    assert tiers[3, 2] == "NORMAL"
    assert tiers[4, 2] == "NORMAL"


def test_nearest_color_classification():
    # Slightly off colors should snap to nearest reference
    # BGR format
    bgr = np.zeros((3, 1, 3), dtype=np.uint8)
    bgr[0, 0] = [10, 10, 240]   # Near Red BGR (0, 0, 255)
    bgr[1, 0] = [10, 245, 10]   # Near Green BGR (0, 255, 0)
    bgr[2, 0] = [5, 5, 5]       # Near Black BGR (0, 0, 0)

    classes = classify_pixels_to_classes(bgr)
    assert classes[0, 0] == ZONE_HIGH
    assert classes[1, 0] == ZONE_MEDIUM
    assert classes[2, 0] == ZONE_IGNORE


def test_dominant_class_tie_breaking():
    # 2x2 patch with 1 High, 1 Med, 0 Ignore -> Tie between High and Med -> High wins
    patch_tie = np.array([
        [ZONE_HIGH, ZONE_MEDIUM],
        [ZONE_IGNORE, 99] # dummy
    ], dtype=np.uint8)
    # 1 High, 1 Med, 1 Ignore
    cell = np.array([[ZONE_HIGH, ZONE_MEDIUM, ZONE_IGNORE]], dtype=np.uint8)
    grid = reduce_pixel_classes_to_grid(cell, 1, 1)
    assert grid[0, 0] == ZONE_HIGH  # HIGH > MEDIUM > IGNORE

    # 1 Med, 1 Ignore
    cell2 = np.array([[ZONE_MEDIUM, ZONE_IGNORE]], dtype=np.uint8)
    grid2 = reduce_pixel_classes_to_grid(cell2, 1, 1)
    assert grid2[0, 0] == ZONE_MEDIUM # MEDIUM > IGNORE


def test_all_black_mask_always_normal():
    thresholds = {"ALERT": 45.0, "CRITICAL": 50.0, "HIGH_PRIORITY_ALERT": 35.0, "HIGH_PRIORITY_CRITICAL": 40.0}
    scores = np.full((14, 14), 100.0) # Huge anomaly score everywhere
    zone_grid = np.full((14, 14), ZONE_IGNORE, dtype=np.uint8)
    tiers = compute_tier_for_scores(scores, zone_grid, thresholds)
    assert np.all(tiers == "NORMAL")


def test_missing_or_corrupt_mask_fallback():
    grid, warning = build_zone_class_grid("non_existent_path.png", 640, 480, 14, 14)
    assert grid.shape == (14, 14)
    assert np.all(grid == ZONE_MEDIUM)
    assert "Mask unavailable" in warning

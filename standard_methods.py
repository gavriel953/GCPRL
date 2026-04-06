"""
Standard contrast enhancement methods — parameter-coupled to GCPRL settings.

DESIGN PRINCIPLE (v2):
  All methods receive the same (k, stretch) parameters that the user set for
  GCPRL and interpret them in a consistent, proportional way:

    k=0.5  → subtle enhancement for every method
    k=1.5  → moderate enhancement for every method
    k=3.0  → aggressive enhancement for every method

  This means the comparison grid shows a fair apples-to-apples view of what
  each algorithm does at the SAME enhancement "budget", not each at its own
  personal optimum.

Parameter mapping:
  HE       — applies a partial blend: output = lerp(original, full_HE, k/3.0)
              So k=3 → full HE; k=1 → 33% toward HE.
  CLAHE    — clip_limit mapped from k: clip = 1.0 + (k-0.5) * 2.0
              k=0.5→1.0, k=1.5→3.0, k=3.0→6.0
              grid_size driven by window_size for consistency.
  Min-Max  — applies a partial stretch: lerp(original, fully_stretched, k/3.0)
              so k=3 → full stretch; k=1 → 33%.
"""

import numpy as np
import cv2
import time
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# ─────────────────── helpers ───────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _is_color(image: np.ndarray) -> bool:
    return len(image.shape) == 3


def _apply_on_luminance(image: np.ndarray, fn) -> np.ndarray:
    """Apply fn to L channel in LAB space (preserves hue/saturation)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_out = fn(l)
    return cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────── public methods ────────────────────────────────────────

def histogram_equalization(
    image: np.ndarray,
    k: float = 1.5,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """
    Partial histogram equalization blended by k.

    blend = clip(k / 3.0, 0, 1)
    output = (1-blend)*original + blend*full_HE

    k=0.5 → 17% toward HE (very subtle)
    k=1.5 → 50% toward HE (moderate)
    k=3.0 → 100% toward HE (full equalization)

    Args:
        image: uint8 grayscale or BGR image.
        k: Enhancement strength from GCPRL slider (0.5–3.0).

    Returns:
        (enhanced_uint8, processing_time_seconds)
    """
    t0 = time.time()
    blend = float(np.clip(k / 3.0, 0.0, 1.0))

    def _partial_he(gray):
        full_he = cv2.equalizeHist(gray)
        out = (1.0 - blend) * gray.astype(np.float32) + blend * full_he.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    if _is_color(image):
        enhanced = _apply_on_luminance(image, _partial_he)
    else:
        enhanced = _partial_he(image)

    return enhanced, time.time() - t0


def clahe_enhancement(
    image: np.ndarray,
    k: float = 1.5,
    window_size: int = 7,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """
    CLAHE with clip_limit and grid_size derived from GCPRL parameters.

    clip_limit = 1.0 + (k - 0.5) * 2.0
      k=0.5 → clip=1.0, k=1.5 → clip=3.0, k=3.0 → clip=6.0

    grid_size tiles = round(max_dim / window_size), clamped to 4–16.

    Args:
        image: uint8 grayscale or BGR image.
        k: Enhancement strength from GCPRL slider.
        window_size: GCPRL variance window (used to derive grid tile count).

    Returns:
        (enhanced_uint8, processing_time_seconds)
    """
    t0 = time.time()
    clip_limit = float(1.0 + (k - 0.5) * 2.0)   # [1.0, 6.0]
    # Larger window → fewer, larger tiles (coarser local adaptation)
    tile = max(4, min(16, round(512 / (window_size * 8))))
    grid = (tile, tile)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)

    def _apply(gray):
        return clahe.apply(gray)

    if _is_color(image):
        enhanced = _apply_on_luminance(image, _apply)
    else:
        enhanced = _apply(image)

    logger.debug(f"CLAHE: clip={clip_limit:.2f}, grid={grid}")
    return enhanced, time.time() - t0


def min_max_stretching(
    image: np.ndarray,
    k: float = 1.5,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """
    Partial linear min-max stretch blended by k.

    blend = clip(k / 3.0, 0, 1)
    output = (1-blend)*original + blend*full_stretch

    k=0.5 → 17% stretch (very subtle)
    k=1.5 → 50% stretch (moderate)
    k=3.0 → 100% stretch (full range)

    Args:
        image: uint8 grayscale or BGR image.
        k: Enhancement strength from GCPRL slider.

    Returns:
        (enhanced_uint8, processing_time_seconds)
    """
    t0 = time.time()
    blend = float(np.clip(k / 3.0, 0.0, 1.0))

    def _partial_stretch(ch):
        mn, mx = float(ch.min()), float(ch.max())
        if mx == mn:
            return ch.copy()
        full = ((ch.astype(np.float32) - mn) / (mx - mn) * 255.0)
        out = (1.0 - blend) * ch.astype(np.float32) + blend * full
        return np.clip(out, 0, 255).astype(np.uint8)

    if _is_color(image):
        channels = cv2.split(image)
        enhanced = cv2.merge([_partial_stretch(c) for c in channels])
    else:
        enhanced = _partial_stretch(image)

    return enhanced, time.time() - t0


def apply_all_standard_methods(
    image: np.ndarray,
    k: float = 1.5,
    window_size: int = 7,
    **kwargs,
) -> dict:
    """
    Apply all standard methods using the same k and window_size as GCPRL.

    Args:
        image: Input uint8 image.
        k: Enhancement strength (same value user set for GCPRL).
        window_size: Variance window (same value user set for GCPRL).

    Returns:
        Dict with keys 'he', 'clahe', 'minmax', each containing
        {'image': ndarray, 'time': float}.
    """
    he_img,   he_t   = histogram_equalization(image, k=k)
    clahe_img, cl_t  = clahe_enhancement(image, k=k, window_size=window_size)
    mm_img,   mm_t   = min_max_stretching(image, k=k)

    logger.info(f"Standard methods applied: k={k}, window={window_size}")
    return {
        'he':     {'image': he_img,    'time': round(he_t, 4)},
        'clahe':  {'image': clahe_img, 'time': round(cl_t, 4)},
        'minmax': {'image': mm_img,    'time': round(mm_t, 4)},
    }

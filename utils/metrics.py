"""
Medical image quality metrics for evaluating contrast enhancement.

Implements CNR, Entropy, Edge Preservation Index, and Brightness Preservation Score.
"""

import numpy as np
import cv2
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def contrast_to_noise_ratio(image: np.ndarray) -> float:
    """
    Compute Contrast-to-Noise Ratio (CNR).

    CNR = |mean_signal - mean_background| / std_background

    Approximated by comparing bright and dark regions of the image.

    Args:
        image: Grayscale or BGR uint8 image.

    Returns:
        CNR value (higher = better contrast).
    """
    gray = ensure_gray(image).astype(np.float32)
    threshold = np.median(gray)

    signal_region = gray[gray > threshold]
    background_region = gray[gray <= threshold]

    if len(signal_region) == 0 or len(background_region) == 0:
        return 0.0

    mean_signal = signal_region.mean()
    mean_bg = background_region.mean()
    std_bg = background_region.std() + 1e-8

    cnr = abs(mean_signal - mean_bg) / std_bg
    return round(float(cnr), 4)


def image_entropy(image: np.ndarray) -> float:
    """
    Compute Shannon entropy of the image histogram.

    Higher entropy indicates more information content / complexity.

    Args:
        image: Grayscale or BGR uint8 image.

    Returns:
        Entropy value in bits (0 to ~8 for uint8 images).
    """
    gray = ensure_gray(image)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-12
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return round(float(entropy), 4)


def edge_preservation_index(
    original: np.ndarray,
    enhanced: np.ndarray
) -> float:
    """
    Compute Edge Preservation Index (EPI).

    EPI = correlation between edge maps of original and enhanced image.
    Value in [0, 1]; higher = better edge preservation.

    Args:
        original: Original grayscale or BGR uint8 image.
        enhanced: Enhanced grayscale or BGR uint8 image.

    Returns:
        EPI value in [0, 1].
    """
    orig_gray = ensure_gray(original).astype(np.float32)
    enh_gray = ensure_gray(enhanced).astype(np.float32)

    orig_edges = cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0) ** 2 + \
                 cv2.Sobel(orig_gray, cv2.CV_32F, 0, 1) ** 2
    enh_edges = cv2.Sobel(enh_gray, cv2.CV_32F, 1, 0) ** 2 + \
                cv2.Sobel(enh_gray, cv2.CV_32F, 0, 1) ** 2

    orig_flat = orig_edges.flatten()
    enh_flat = enh_edges.flatten()

    corr = np.corrcoef(orig_flat, enh_flat)[0, 1]
    epi = max(0.0, min(1.0, float(corr)))
    return round(epi, 4)


def brightness_preservation_score(
    original: np.ndarray,
    enhanced: np.ndarray
) -> float:
    """
    Compute brightness preservation score (1 - normalized mean difference).

    Score in [0, 1]; 1.0 = perfect brightness preservation.

    Args:
        original: Original uint8 image.
        enhanced: Enhanced uint8 image.

    Returns:
        Brightness preservation score in [0, 1].
    """
    orig_mean = ensure_gray(original).astype(np.float32).mean()
    enh_mean = ensure_gray(enhanced).astype(np.float32).mean()
    score = 1.0 - abs(orig_mean - enh_mean) / 255.0
    return round(float(score), 4)


def compute_all_metrics(
    original: np.ndarray,
    enhanced: np.ndarray,
    processing_time: float = 0.0
) -> dict:
    """
    Compute all quality metrics for an enhanced image.

    Args:
        original: Original uint8 image.
        enhanced: Enhanced uint8 image.
        processing_time: Time taken to process (seconds).

    Returns:
        Dictionary of all metrics.
    """
    return {
        'cnr_original': contrast_to_noise_ratio(original),
        'cnr_enhanced': contrast_to_noise_ratio(enhanced),
        'entropy_original': image_entropy(original),
        'entropy_enhanced': image_entropy(enhanced),
        'edge_preservation': edge_preservation_index(original, enhanced),
        'brightness_preservation': brightness_preservation_score(original, enhanced),
        'processing_time': round(processing_time, 4),
    }


def compute_metrics_set(original: np.ndarray, results: dict) -> dict:
    """
    Compute metrics for a set of enhanced images (for comparison view).

    Args:
        original: Original image.
        results: Dict of {method_name: {'image': ndarray, 'time': float}}.

    Returns:
        Dict of {method_name: metrics_dict}.
    """
    metrics_out = {}
    for method, data in results.items():
        img = data['image']
        t = data.get('time', 0.0)
        metrics_out[method] = {
            'cnr': contrast_to_noise_ratio(img),
            'entropy': image_entropy(img),
            'edge_preservation': edge_preservation_index(original, img),
            'brightness': brightness_preservation_score(original, img),
            'processing_time': t,
        }
    return metrics_out

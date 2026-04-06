"""
Image utility functions for loading, saving, and converting medical images.
"""

import os
import uuid
import base64
import logging
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.dcm'}


def load_image(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load an image from any supported format.

    Handles PNG, JPEG, TIFF, BMP via OpenCV, and DICOM via pydicom.

    Args:
        filepath: Path to the image file.

    Returns:
        Tuple of (image_bgr_or_gray_uint8, metadata_dict).
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.dcm':
        from utils.dicom_handler import load_dicom
        img, meta = load_dicom(filepath)
        return img, meta

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Try with Pillow as fallback
        pil_img = Image.open(filepath).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if img is None:
        raise ValueError(f"Could not load image: {filepath}")

    # Handle 16-bit images
    if img.dtype == np.uint16:
        # Min-max normalization preserves structure better than just dropping 8 bits
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    # Handle RGBA
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    meta = {
        'width': w,
        'height': h,
        'channels': channels,
        'dtype': str(img.dtype),
        'modality': 'Unknown',
    }

    return img, meta


def save_image(image: np.ndarray, output_dir: str, suffix: str = '') -> str:
    """
    Save an image to the output directory with a unique filename.

    Args:
        image: Image array (uint8).
        output_dir: Directory to save to.
        suffix: Optional suffix for filename.

    Returns:
        Filename (not full path) of saved image.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{suffix}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    logger.debug(f"Saved image: {filepath}")
    return filename


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert an image array to a base64-encoded PNG string for inline display.

    Args:
        image: Grayscale or BGR uint8 array.

    Returns:
        Base64-encoded string (without data URI prefix).
    """
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def generate_difference_map(
    original: np.ndarray,
    enhanced: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate absolute difference map and colorized heatmap.

    Args:
        original: Original image (uint8).
        enhanced: Enhanced image (uint8).
        colormap: OpenCV colormap constant.

    Returns:
        Tuple of (diff_gray_uint8, heatmap_bgr_uint8).
    """
    def to_gray(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    orig_g = to_gray(original).astype(np.float32)
    enh_g = to_gray(enhanced).astype(np.float32)

    diff = np.abs(enh_g - orig_g)
    diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, colormap)

    return diff_norm, heatmap


def compute_histogram(image: np.ndarray, bins: int = 256) -> Tuple[list, list]:
    """
    Compute image histogram for a grayscale or color image.

    Args:
        image: Input uint8 image.
        bins: Number of histogram bins.

    Returns:
        Tuple of (bin_edges_list, counts_list) for JSON serialization.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    counts, edges = np.histogram(gray.flatten(), bins=bins, range=(0, 255))
    return edges[:-1].tolist(), counts.tolist()


def resize_for_display(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """
    Resize image for display, maintaining aspect ratio.

    Args:
        image: Input image.
        max_dim: Maximum dimension (width or height).

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

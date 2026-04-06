"""
GCPRL Auto-Optimizer: analyze image statistics and predict optimal parameters.

Strategy:
  1. Extract 6 image statistics (contrast range, std, variance, entropy, noise, brightness).
  2. Map statistics → parameters via interpretable rules derived from grid-search benchmarks.
  3. Return params + a plain-English rationale explaining every choice.

Rules summary (derived from exhaustive benchmarks):
  k       ← driven by contrast_range: low range → high k (need aggressive push toward HE)
  window  ← driven by noise_estimate: noisy image → larger window to smooth variance map
  alpha   ← driven by mean_variance + noise: rich structure + low noise → more injection
  stretch ← driven by entropy: compressed histogram (low entropy) → stretch further
"""

import numpy as np
import cv2
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def analyze_image(image: np.ndarray) -> dict:
    """
    Extract statistical descriptors from a grayscale or BGR image.

    Returns a dict with:
      contrast_range  — (max-min)/255, measures dynamic range usage [0,1]
      std_norm        — std/255, overall intensity spread [0,1]
      mean_norm       — mean/255, overall brightness [0,1]
      entropy         — Shannon entropy of 256-bin histogram [0,8]
      mean_variance   — average local variance (7-px window) [0,~0.1]
      noise_estimate  — std of Laplacian / 255, high-freq content [0,~0.3]
      percentile_range— (p99-p1)/255, robust contrast range [0,1]
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    f = gray.astype(np.float32)

    # Dynamic range
    contrast_range   = float(f.max() - f.min()) / 255.0
    p1, p99          = np.percentile(f, 1), np.percentile(f, 99)
    percentile_range = float(p99 - p1) / 255.0
    std_norm         = float(f.std()) / 255.0
    mean_norm        = float(f.mean()) / 255.0

    # Histogram entropy
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    hist_p = hist.astype(np.float64) / (hist.sum() + 1e-9)
    hist_p = hist_p[hist_p > 0]
    entropy = float(-np.sum(hist_p * np.log2(hist_p)))

    # Local variance (structural richness)
    fn = f / 255.0
    k7 = np.ones((7, 7), dtype=np.float32) / 49.0
    mu  = cv2.filter2D(fn, -1, k7, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.filter2D(fn**2, -1, k7, borderType=cv2.BORDER_REFLECT)
    mean_variance = float(np.maximum(mu2 - mu**2, 0.0).mean())

    # Noise estimate via Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise_estimate = float(lap.std()) / 255.0

    return {
        'contrast_range':   round(contrast_range, 4),
        'percentile_range': round(percentile_range, 4),
        'std_norm':         round(std_norm, 4),
        'mean_norm':        round(mean_norm, 4),
        'entropy':          round(entropy, 4),
        'mean_variance':    round(mean_variance, 6),
        'noise_estimate':   round(noise_estimate, 4),
    }


def predict_params(stats: dict) -> Tuple[dict, dict]:
    """
    Map image statistics to optimal GCPRL parameters with rationale.

    Args:
        stats: Output of analyze_image().

    Returns:
        (params_dict, rationale_dict) where rationale explains each choice.
    """
    cr  = stats['contrast_range']
    pr  = stats['percentile_range']
    sv  = stats['std_norm']
    mv  = stats['mean_variance']
    ent = stats['entropy']
    nz  = stats['noise_estimate']
    mn  = stats['mean_norm']

    rationale = {}

    # ── k: enhancement strength ───────────────────────────────────────────────
    # Low percentile range → image is severely compressed → push hard toward HE
    if pr < 0.10:
        k = 3.0
        rationale['k'] = f"Severely compressed range ({pr:.2f}) — maximum k=3.0 to aggressively expand contrast."
    elif pr < 0.25:
        k = 2.5
        rationale['k'] = f"Narrow intensity range ({pr:.2f}) — high k=2.5 to strongly expand the histogram."
    elif pr < 0.45:
        k = 2.0
        rationale['k'] = f"Moderate range ({pr:.2f}) — k=2.0 for balanced enhancement."
    elif pr < 0.65:
        k = 1.5
        rationale['k'] = f"Good range ({pr:.2f}) — k=1.5 for gentle refinement."
    else:
        k = 1.2
        rationale['k'] = f"Wide range ({pr:.2f}) — minimal k=1.2, image already well-exposed."

    # ── window_size: variance map smoothing ───────────────────────────────────
    # More noise → bigger window to prevent noise from being treated as structure
    if nz > 0.12:
        win = 13
        rationale['window_size'] = f"High noise ({nz:.3f}) — large window=13 to smooth variance map over noise."
    elif nz > 0.07:
        win = 11
        rationale['window_size'] = f"Moderate noise ({nz:.3f}) — window=11 for noise-robust variance."
    elif nz > 0.04:
        win = 9
        rationale['window_size'] = f"Low-moderate noise ({nz:.3f}) — window=9."
    else:
        win = 7
        rationale['window_size'] = f"Clean image ({nz:.3f}) — default window=7 for precise local structure."

    # ── local_alpha: residual local contrast injection ────────────────────────
    # Rich structure + low noise → inject more local contrast safely
    # High noise → reduce alpha to avoid amplifying noise
    if mv > 0.012 and nz < 0.06:
        alpha = 0.45
        rationale['local_alpha'] = f"Rich structure (var={mv:.4f}) and clean — α=0.45 for strong local boost."
    elif mv > 0.005 and nz < 0.10:
        alpha = 0.35
        rationale['local_alpha'] = f"Moderate structure (var={mv:.4f}) — α=0.35 balanced injection."
    elif nz > 0.10:
        alpha = 0.20
        rationale['local_alpha'] = f"Noisy image ({nz:.3f}) — reduced α=0.20 to avoid noise amplification."
    else:
        alpha = 0.25
        rationale['local_alpha'] = f"Low local structure (var={mv:.4f}) — conservative α=0.25."

    # ── stretch: CDF target palette extent ───────────────────────────────────
    # Low entropy = histogram is concentrated/spiked → need to spread further
    if ent < 3.5:
        stretch = 0.99
        rationale['stretch'] = f"Very low entropy ({ent:.2f}) — stretch=0.99 for maximum histogram expansion."
    elif ent < 5.0:
        stretch = 0.97
        rationale['stretch'] = f"Low entropy ({ent:.2f}) — stretch=0.97 for strong histogram spread."
    elif ent < 6.5:
        stretch = 0.95
        rationale['stretch'] = f"Moderate entropy ({ent:.2f}) — stretch=0.95 standard setting."
    else:
        stretch = 0.90
        rationale['stretch'] = f"High entropy ({ent:.2f}) — stretch=0.90, histogram already rich."

    params = {
        'k':            k,
        'window_size':  win,
        'local_alpha':  alpha,
        'stretch':      stretch,
        'preserve_diagnostic': True,
    }

    return params, rationale


def auto_optimize(image: np.ndarray) -> Tuple[dict, dict, dict]:
    """
    Analyze an image and return optimal GCPRL parameters.

    Args:
        image: Input grayscale or BGR uint8 image.

    Returns:
        (params, rationale, stats) — all dicts.
    """
    stats = analyze_image(image)
    params, rationale = predict_params(stats)
    logger.info(f"Auto-optimize: stats={stats} → params={params}")
    return params, rationale, stats

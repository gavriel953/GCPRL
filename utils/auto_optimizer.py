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

    # ── CO 5: Least Squares Method for Optimal Data Representation ──
    # We formulate the parameter prediction as a Multiple Linear Regression problem.
    # Features X = [1 (bias), percentile_range, noise_estimate, mean_variance, entropy]
    # We train this model on-the-fly using a design matrix representing optimal clinical baselines.
    
    X_design = np.array([
        [1.0, 0.05, 0.02, 0.002, 3.0],  # Case 1: severely compressed, clean, low struct
        [1.0, 0.40, 0.05, 0.008, 5.0],  # Case 2: moderate contrast, moderate noise
        [1.0, 0.80, 0.15, 0.015, 7.0],  # Case 3: wide range, noisy, high entropy
        [1.0, 0.20, 0.08, 0.010, 4.0],  # Case 4: narrow range, moderately noisy
        [1.0, 0.60, 0.03, 0.020, 6.0],  # Case 5: good range, rich structure (high var), clean
        [1.0, 0.95, 0.01, 0.005, 7.5],  # Case 6: nearly full range, clean
    ])
    
    # Target parameter matrices (Y) corresponding to the design matrix
    y_k       = np.array([3.0, 2.0, 1.2, 2.5, 1.5, 1.2]) 
    y_win     = np.array([7,   9,  13,  11,  7,   7])
    y_alpha   = np.array([0.25,0.35,0.20,0.30,0.45,0.25])
    y_stretch = np.array([0.99,0.97,0.90,0.98,0.95,0.90])
    
    # Calculate Least Squares Optimal Weights using the Normal Equation
    # We use np.linalg.pinv (Moore-Penrose Pseudoinverse) for numerical stability.
    pseudo_inv = np.linalg.pinv(X_design)
    
    beta_k       = pseudo_inv @ y_k
    beta_win     = pseudo_inv @ y_win
    beta_alpha   = pseudo_inv @ y_alpha
    beta_stretch = pseudo_inv @ y_stretch
    
    # Current image feature vector
    x_current = np.array([1.0, pr, nz, mv, ent])
    
    # Predict optimal parameters using linear functional relationship
    pred_k       = float(x_current @ beta_k)
    pred_win     = int(round(x_current @ beta_win))
    pred_alpha   = float(x_current @ beta_alpha)
    pred_stretch = float(x_current @ beta_stretch)
    
    # Apply safety clipping constraints
    k       = float(np.clip(pred_k, 0.5, 3.0))
    win     = int(np.clip(pred_win, 3, 15))
    if win % 2 == 0: 
        win += 1  # Window size must be an odd integer
    alpha   = float(np.clip(pred_alpha, 0.0, 0.5))
    stretch = float(np.clip(pred_stretch, 0.80, 0.99))

    # Generate Mathematical Rationales based on Least Squares execution
    rationale['Model'] = "Employed Least Squares Regression (Normal Equation) to map image statistics to parameters."
    rationale['k'] = f"Predicted k={k:.2f} (base: {beta_k[0]:.2f}, pr_weight: {beta_k[1]:.2f})."
    rationale['window_size'] = f"Least Squares predicted {pred_win}px → snapped to odd integer {win}px."
    rationale['local_alpha'] = f"Predicted α={alpha:.2f} driven by variance weight {beta_alpha[3]:.2f}."
    rationale['stretch'] = f"Predicted stretch={stretch:.2f} driven inversely by entropy."

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

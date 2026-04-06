"""
GCPRL Medical Image Enhancement - Flask Application Entry Point.

Provides RESTful API endpoints for image upload, GCPRL enhancement,
standard method comparison, difference mapping, and report generation.
"""

import os
import sys
import uuid
import json
import logging
from datetime import datetime

# Ensure project root is on path for utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import (
    Flask, request, jsonify, render_template, send_from_directory, session
)
from werkzeug.utils import secure_filename
import numpy as np
import cv2

from config import Config
from gcprl_core import gcprl_enhance, gcprl_enhance_color
from standard_methods import apply_all_standard_methods
from utils.image_utils import (
    load_image, save_image, image_to_base64,
    generate_difference_map, compute_histogram, resize_for_display
)
from utils.metrics import compute_all_metrics, compute_metrics_set
from utils.auto_optimizer import auto_optimize

# ──────────────────────────── App setup ─────────────────────────────────────

app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(
    level=getattr(logging, app.config.get('LOG_LEVEL', 'DEBUG')),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# In-memory job store (use Redis/DB in production)
job_store: dict = {}


# ──────────────────────────── Helpers ───────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'])


def get_upload_path(filename: str) -> str:
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)


def encode_image(image: np.ndarray) -> str:
    """Resize for display and encode as base64 PNG."""
    display = resize_for_display(image, max_dim=900)
    return image_to_base64(display)


# ──────────────────────────── Routes ────────────────────────────────────────

@app.route('/')
def index():
    """Main application interface."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle image upload.

    Returns:
        JSON with job_id, original image (base64), metadata, and histograms.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: '
                                  f'{", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = get_upload_path(unique_name)
    file.save(filepath)

    try:
        image, meta = load_image(filepath)
    except Exception as e:
        logger.error(f"Load error: {e}")
        return jsonify({'error': f'Could not load image: {str(e)}'}), 422

    job_id = uuid.uuid4().hex
    job_store[job_id] = {
        'original_path': filepath,
        'original_filename': unique_name,
        'meta': meta,
        'created': datetime.utcnow().isoformat(),
        'enhanced_path': None,
        'results': {},
    }

    # Histogram
    bin_edges, counts = compute_histogram(image)

    logger.info(f"Upload job {job_id}: {filename} {meta.get('width')}x{meta.get('height')}")
    return jsonify({
        'job_id': job_id,
        'original_image': encode_image(image),
        'metadata': meta,
        'histogram': {'edges': bin_edges, 'counts': counts},
    })


@app.route('/enhance', methods=['POST'])
def enhance():
    """
    Apply GCPRL enhancement to the uploaded image.

    Request JSON:
        job_id (str): Job identifier from /upload.
        k (float): Enhancement strength (0.5-3.0).
        window_size (int): Variance window (3-15, odd).
        preserve_diagnostic (bool): Preserve brightness.

    Returns:
        JSON with enhanced image (base64), metrics, and histogram.
    """
    data = request.get_json(force=True)
    job_id = data.get('job_id')

    if not job_id or job_id not in job_store:
        return jsonify({'error': 'Invalid or expired job_id'}), 404

    k = float(data.get('k', app.config['DEFAULT_K']))
    window_size = int(data.get('window_size', app.config['DEFAULT_WINDOW_SIZE']))
    preserve = bool(data.get('preserve_diagnostic', True))
    local_alpha = float(data.get('local_alpha', 0.30))
    stretch = float(data.get('stretch', 0.95))

    # Clamp parameters
    k = max(app.config['MIN_K'], min(app.config['MAX_K'], k))
    window_size = max(app.config['MIN_WINDOW'], min(app.config['MAX_WINDOW'], window_size))
    if window_size % 2 == 0:
        window_size += 1
    local_alpha = max(0.0, min(0.5, local_alpha))
    stretch = max(0.8, min(0.99, stretch))

    job = job_store[job_id]
    try:
        original, _ = load_image(job['original_path'])
    except Exception as e:
        return jsonify({'error': f'Could not reload original: {str(e)}'}), 500

    try:
        if len(original.shape) == 3:
            enhanced, gcprl_meta = gcprl_enhance_color(original, k, window_size, preserve, local_alpha, stretch)
        else:
            enhanced, gcprl_meta = gcprl_enhance(original, k, window_size, preserve, local_alpha, stretch)
    except Exception as e:
        logger.exception("GCPRL enhancement failed")
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500

    # Save enhanced
    enh_filename = save_image(enhanced, app.config['UPLOAD_FOLDER'], '_gcprl')
    job['enhanced_path'] = get_upload_path(enh_filename)
    job['enhanced_filename'] = enh_filename
    job['results']['gcprl'] = {'image': enhanced, 'time': gcprl_meta['processing_time_s']}

    # Metrics
    metrics = compute_all_metrics(original, enhanced, gcprl_meta['processing_time_s'])
    metrics.update(gcprl_meta)
    job['metrics'] = metrics

    # Histograms
    _, orig_counts = compute_histogram(original)
    _, enh_counts = compute_histogram(enhanced)
    bin_edges, _ = compute_histogram(original)

    job['last_k'] = k
    job['last_window'] = window_size
    logger.info(f"Job {job_id} enhanced: k={k}, window={window_size}")
    return jsonify({
        'enhanced_image': encode_image(enhanced),
        'metrics': metrics,
        'histogram': {
            'edges': bin_edges,
            'original_counts': orig_counts,
            'enhanced_counts': enh_counts,
        },
        'enhanced_filename': enh_filename,
    })


@app.route('/compare_standard', methods=['POST'])
def compare_standard():
    """
    Compare GCPRL with standard enhancement methods (HE, CLAHE, Min-Max).

    Request JSON:
        job_id (str): Must have an enhanced result already.

    Returns:
        JSON with base64 images and metrics for all methods.
    """
    data = request.get_json(force=True)
    job_id = data.get('job_id')

    if not job_id or job_id not in job_store:
        return jsonify({'error': 'Invalid or expired job_id'}), 404

    job = job_store[job_id]
    if not job.get('enhanced_path'):
        return jsonify({'error': 'Please enhance the image with GCPRL first'}), 400

    try:
        original, _ = load_image(job['original_path'])
        enhanced, _ = load_image(job['enhanced_path'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Apply standard methods
    # Use same k and window the user set for GCPRL — consistent comparison
    k_used = job.get('last_k', 2.0)
    win_used = job.get('last_window', 7)
    std_results = apply_all_standard_methods(original, k=k_used, window_size=win_used)

    # Build response
    images_b64 = {
        'original': encode_image(original),
        'gcprl': encode_image(enhanced),
        'he': encode_image(std_results['he']['image']),
        'clahe': encode_image(std_results['clahe']['image']),
        'minmax': encode_image(std_results['minmax']['image']),
    }

    # Metrics for each
    all_results = {
        'gcprl': {'image': enhanced, 'time': job.get('metrics', {}).get('processing_time', 0)},
        **{k: v for k, v in std_results.items()},
    }
    metrics_set = compute_metrics_set(original, all_results)
    # Also compute original metrics
    from utils.metrics import contrast_to_noise_ratio, image_entropy
    metrics_set['original'] = {
        'cnr': contrast_to_noise_ratio(original),
        'entropy': image_entropy(original),
        'edge_preservation': 1.0,
        'brightness': 1.0,
        'processing_time': 0.0,
    }

    # Save std results
    for method, res in std_results.items():
        fname = save_image(res['image'], app.config['UPLOAD_FOLDER'], f'_{method}')
        job[f'{method}_filename'] = fname

    return jsonify({
        'images': images_b64,
        'metrics': metrics_set,
    })


@app.route('/difference_map', methods=['POST'])
def difference_map():
    """
    Generate difference heatmap between original and enhanced images.

    Request JSON:
        job_id (str): Must have an enhanced result.
        colormap (str): 'jet' | 'hot' | 'viridis' (default: 'jet').

    Returns:
        JSON with diff_gray and heatmap base64 images, plus histogram.
    """
    data = request.get_json(force=True)
    job_id = data.get('job_id')
    colormap_name = data.get('colormap', 'jet')

    COLORMAPS = {'jet': cv2.COLORMAP_JET, 'hot': cv2.COLORMAP_HOT,
                 'viridis': cv2.COLORMAP_VIRIDIS, 'plasma': cv2.COLORMAP_PLASMA}
    colormap = COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)

    if not job_id or job_id not in job_store:
        return jsonify({'error': 'Invalid or expired job_id'}), 404

    job = job_store[job_id]
    if not job.get('enhanced_path'):
        return jsonify({'error': 'Please enhance the image first'}), 400

    try:
        original, _ = load_image(job['original_path'])
        enhanced, _ = load_image(job['enhanced_path'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    diff_gray, heatmap = generate_difference_map(original, enhanced, colormap)

    # Overlay: blend heatmap onto original (50% opacity)
    if len(original.shape) == 2:
        orig_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        orig_color = original.copy()
    overlay = cv2.addWeighted(orig_color, 0.5, heatmap, 0.5, 0)

    # Diff histogram
    diff_edges, diff_counts = compute_histogram(diff_gray)

    # Save heatmap
    heatmap_fname = save_image(heatmap, app.config['UPLOAD_FOLDER'], '_heatmap')
    job['heatmap_filename'] = heatmap_fname

    return jsonify({
        'diff_gray': encode_image(diff_gray),
        'heatmap': encode_image(heatmap),
        'overlay': encode_image(overlay),
        'diff_histogram': {'edges': diff_edges, 'counts': diff_counts},
        'max_diff': int(diff_gray.max()),
        'mean_diff': round(float(diff_gray.mean()), 2),
    })


@app.route('/auto_optimize', methods=['POST'])
def auto_optimize_endpoint():
    """
    Analyze the uploaded image and return optimal GCPRL parameters.

    Request JSON:
        job_id (str): Job identifier from /upload.

    Returns:
        JSON with recommended params, per-param rationale, and image stats.
    """
    data = request.get_json(force=True)
    job_id = data.get('job_id')

    if not job_id or job_id not in job_store:
        return jsonify({'error': 'Invalid or expired job_id'}), 404

    job = job_store[job_id]
    try:
        original, _ = load_image(job['original_path'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    try:
        params, rationale, stats = auto_optimize(original)
    except Exception as e:
        logger.exception("Auto-optimize failed")
        return jsonify({'error': str(e)}), 500

    logger.info(f"Auto-optimize job {job_id}: {params}")
    return jsonify({
        'params': params,
        'rationale': rationale,
        'stats': stats,
    })


@app.route('/download/<filename>')
def download(filename):
    """Download a processed image file."""
    safe_name = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name,
                                as_attachment=True)


@app.route('/metrics/<job_id>')
def get_metrics(job_id):
    """Return stored metrics for a job."""
    if job_id not in job_store:
        return jsonify({'error': 'Job not found'}), 404
    job = job_store[job_id]
    return jsonify(job.get('metrics', {}))


@app.route('/report/<job_id>')
def generate_report(job_id):
    """Generate and download a PDF report for a job."""
    if job_id not in job_store:
        return jsonify({'error': 'Job not found'}), 404

    job = job_store[job_id]
    if not job.get('enhanced_path'):
        return jsonify({'error': 'No enhanced image available'}), 400

    try:
        from utils.report_generator import generate_pdf_report
        pdf_filename = generate_pdf_report(
            job_id=job_id,
            original_path=job['original_path'],
            enhanced_path=job['enhanced_path'],
            metrics=job.get('metrics', {}),
            output_dir=app.config['UPLOAD_FOLDER'],
        )
        return send_from_directory(app.config['UPLOAD_FOLDER'], pdf_filename,
                                    as_attachment=True)
    except Exception as e:
        logger.exception("Report generation failed")
        return jsonify({'error': str(e)}), 500


# ──────────────────────────── Run ───────────────────────────────────────────

if __name__ == '__main__':
    logger.info("Starting GCPRL Medical Image Enhancement Server...")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.run(debug=True, host='0.0.0.0', port=5005, threaded=True)

# GCPRL Medical Image Enhancement

**Globally Coupled Pixel Ranking Linearization (GCPRL)** — A Flask web application for contrast enhancement of medical images (X-rays, MRIs, CT scans), with side-by-side comparison, multi-method benchmarking, and difference visualization.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py

# 3. Open browser
open http://localhost:5000
```

---

## File Structure

```
gcprl_medical_app/
├── app.py                    # Flask application & API endpoints
├── gcprl_core.py             # GCPRL algorithm (6-step pipeline)
├── standard_methods.py       # HE, CLAHE, Min-Max implementations
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── static/
│   ├── css/style.css         # Medical-grade dark/light theme UI
│   ├── js/main.js            # Application logic (AJAX, UI, charts)
│   ├── js/image_comparison.js # Interactive before/after slider widget
│   └── uploads/              # Temporary processed images
├── templates/
│   └── index.html            # Single-page application template
└── utils/
    ├── __init__.py
    ├── dicom_handler.py      # DICOM loading & metadata extraction
    ├── image_utils.py        # Load/save/encode/histogram utilities
    ├── metrics.py            # CNR, Entropy, EPI, Brightness metrics
    └── report_generator.py   # PDF report generation (reportlab)
```

---

## GCPRL Algorithm

The 6-step pipeline in `gcprl_core.py`:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `global_pixel_ranking` | Rank every pixel globally, normalize to [0,1] |
| 2 | `compute_local_variance_map` | Sliding-window variance with configurable size |
| 3 | `variance_modulated_rank_redistribution` | Spread ranks in high-detail regions (k-controlled) |
| 4 | `construct_synthetic_target` | Map modulated ranks → target intensities via original CDF |
| 5 | `fit_global_affine_map` | Fit I' = aI + b via least squares |
| 6 | `apply_affine_transform` | Apply transform and clip to valid range |

Key property: **edges and diagnostically significant regions** (high local variance) receive proportionally more contrast enhancement than homogeneous areas, preserving subtle pathological markers.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/upload` | POST | Upload image (multipart/form-data, field: `file`) |
| `/enhance` | POST | GCPRL enhancement (`job_id`, `k`, `window_size`, `preserve_diagnostic`) |
| `/compare_standard` | POST | Run HE, CLAHE, Min-Max on same image (`job_id`) |
| `/difference_map` | POST | Generate heatmap diff (`job_id`, `colormap`) |
| `/download/<filename>` | GET | Download processed image |
| `/metrics/<job_id>` | GET | Retrieve stored metrics JSON |
| `/report/<job_id>` | GET | Download PDF summary report |

### Example API Usage

```python
import requests

# Upload
with open('chest_xray.png', 'rb') as f:
    r = requests.post('http://localhost:5000/upload', files={'file': f})
job_id = r.json()['job_id']

# Enhance
r = requests.post('http://localhost:5000/enhance', json={
    'job_id': job_id, 'k': 1.5, 'window_size': 7, 'preserve_diagnostic': True
})
print(r.json()['metrics'])  # CNR, entropy, affine coefficients, timing

# Compare
r = requests.post('http://localhost:5000/compare_standard', json={'job_id': job_id})
metrics_table = r.json()['metrics']  # Per-method metrics dict

# Difference map
r = requests.post('http://localhost:5000/difference_map', json={'job_id': job_id, 'colormap': 'hot'})
```

---

## Three Comparison Features

### Button 1 — ⟺ Interactive Comparison Slider
After enhancement, an interactive drag slider appears between the original and enhanced image panels. Drag the divider to reveal either side — ideal for quickly spotting subtle changes.

### Button 2 — ⊞ Compare with Standard Methods
Processes the image with HE, CLAHE, and Min-Max Stretching and displays:
- A 5-card comparison grid (Original / GCPRL / HE / CLAHE / Min-Max)
- A sortable metrics table showing CNR, Entropy, Edge Preservation, Brightness Preservation, and Processing Time
- Best values highlighted in green per metric

### Button 3 — ◈ Show Difference Map
Generates three views:
- **Grayscale absolute difference** |Enhanced − Original|
- **Colorized heatmap** (choose JET / HOT / VIRIDIS / PLASMA)
- **50% blend overlay** on the original image
- Difference histogram showing pixel change distribution
- Summary stats: max and mean pixel difference

---

## Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Enhancement Strength (k) | 0.5 – 3.0 | 1.0 | Higher = more contrast boost in detail regions |
| Variance Window Size | 3 – 15 (odd) | 7 | Larger = more spatial context for variance |
| Preserve Diagnostic Brightness | bool | true | Clamps affine slope to [0.5, 2.5] |

---

## Supported Formats

- **PNG, JPEG, TIFF, BMP** — via OpenCV + Pillow
- **DICOM (.dcm)** — via pydicom, with automatic windowing and metadata extraction (Modality, PatientID, StudyDate, WindowCenter/Width)

---

## Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **CNR** | \|μ_signal − μ_bg\| / σ_bg | Higher = better lesion/background separation |
| **Entropy** | −Σ p·log₂(p) | Higher = more information content |
| **Edge Preservation Index (EPI)** | corr(∇original, ∇enhanced) | 1.0 = perfect structural preservation |
| **Brightness Preservation** | 1 − \|μ_orig − μ_enh\| / 255 | 1.0 = mean brightness unchanged |

---

## Production Deployment

```bash
# Using gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 "app:app"

# Environment variables
export SECRET_KEY="your-production-secret"
export FLASK_ENV=production
```

For production, replace the in-memory `job_store` dictionary in `app.py` with Redis or a database.

---

## Dependencies

```
flask>=2.3.0          # Web framework
numpy>=1.24.0         # Numerical operations
opencv-python>=4.8.0  # Image I/O and processing
pydicom>=2.4.0        # DICOM format support
pillow>=10.0.0        # Additional image format support
scikit-image>=0.21.0  # Image utilities
matplotlib>=3.7.0     # (optional) plotting
reportlab>=4.0.0      # PDF report generation
scipy>=1.11.0         # Scientific computing
werkzeug>=2.3.0       # Flask utilities
```

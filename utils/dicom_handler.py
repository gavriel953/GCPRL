"""
DICOM file handler for reading and extracting medical imaging metadata.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, dict as Dict

logger = logging.getLogger(__name__)


def load_dicom(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load a DICOM file and extract image array and metadata.

    Args:
        filepath: Path to .dcm file.

    Returns:
        Tuple of (image_uint8, metadata_dict).
        image_uint8 is a normalized uint8 grayscale image ready for display.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom is required for DICOM support. Install with: pip install pydicom")

    ds = pydicom.dcmread(filepath)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Handle multi-frame DICOM (take middle frame)
    if pixel_array.ndim == 3:
        pixel_array = pixel_array[pixel_array.shape[0] // 2]

    # Apply windowing if available
    wc = getattr(ds, 'WindowCenter', None)
    ww = getattr(ds, 'WindowWidth', None)
    if wc is not None and ww is not None:
        if hasattr(wc, '__iter__'):
            wc = float(wc[0])
            ww = float(ww[0])
        else:
            wc = float(wc)
            ww = float(ww)
        lower = wc - ww / 2
        upper = wc + ww / 2
        pixel_array = np.clip(pixel_array, lower, upper)

    # Normalize to [0, 255]
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
        pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)

    image_uint8 = pixel_array.astype(np.uint8)

    # Extract metadata
    metadata = {
        'modality': str(getattr(ds, 'Modality', 'Unknown')),
        'patient_id': str(getattr(ds, 'PatientID', 'N/A')),
        'study_date': str(getattr(ds, 'StudyDate', 'N/A')),
        'institution': str(getattr(ds, 'InstitutionName', 'N/A')),
        'series_description': str(getattr(ds, 'SeriesDescription', 'N/A')),
        'rows': int(getattr(ds, 'Rows', image_uint8.shape[0])),
        'columns': int(getattr(ds, 'Columns', image_uint8.shape[1])),
        'bits_allocated': int(getattr(ds, 'BitsAllocated', 8)),
        'photometric': str(getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')),
    }

    logger.info(f"DICOM loaded: {filepath} | modality={metadata['modality']}")
    return image_uint8, metadata


def is_dicom(filepath: str) -> bool:
    """Check if a file is a valid DICOM file."""
    try:
        import pydicom
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception:
        return False

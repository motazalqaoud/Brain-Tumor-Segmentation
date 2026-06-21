"""
dicom_loader.py
---------------
Load DICOM series into numpy arrays with proper metadata handling.

Author: Motaz Alqaoud, PhD
GitHub: https://github.com/motazalqaoud

Clinical note:
    DICOM files contain critical metadata (voxel spacing, orientation, 
    patient position) that most tutorials ignore. In surgical navigation,
    this metadata is the difference between a correct and a dangerous result.
"""

import os
import numpy as np
import pydicom
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings


def load_dicom_series(folder_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load a DICOM series from a folder into a 3D numpy array.

    Args:
        folder_path: Path to folder containing .dcm files

    Returns:
        volume: 3D numpy array (slices, height, width)
        metadata: dict with spacing, orientation, and other DICOM tags

    Example:
        >>> volume, meta = load_dicom_series("path/to/dicom/")
        >>> print(volume.shape, meta['spacing'])
    """
    folder = Path(folder_path)
    dcm_files = sorted(folder.glob("*.dcm"))

    if not dcm_files:
        dcm_files = sorted(folder.glob("*.DCM"))

    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {folder_path}")

    # Read all slices
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        slices.append(ds)

    # Sort slices by ImagePositionPatient (Z position)
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        warnings.warn("ImagePositionPatient not found. Using file order.")

    # Stack into 3D volume
    volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)

    # Apply rescale slope/intercept (Hounsfield units for CT)
    ref = slices[0]
    if hasattr(ref, 'RescaleSlope') and hasattr(ref, 'RescaleIntercept'):
        volume = volume * float(ref.RescaleSlope) + float(ref.RescaleIntercept)

    # Extract metadata
    metadata = _extract_metadata(slices)

    return volume, metadata


def _extract_metadata(slices: list) -> Dict:
    """Extract clinically relevant metadata from DICOM slices."""
    ref = slices[0]
    metadata = {}

    # Voxel spacing (critical for measurements)
    try:
        row_spacing, col_spacing = float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1])
    except AttributeError:
        row_spacing, col_spacing = 1.0, 1.0
        warnings.warn("PixelSpacing not found. Defaulting to 1.0mm.")

    # Slice thickness
    try:
        slice_thickness = float(ref.SliceThickness)
    except AttributeError:
        if len(slices) > 1:
            try:
                z0 = float(slices[0].ImagePositionPatient[2])
                z1 = float(slices[1].ImagePositionPatient[2])
                slice_thickness = abs(z1 - z0)
            except Exception:
                slice_thickness = 1.0
        else:
            slice_thickness = 1.0

    metadata['spacing'] = (slice_thickness, row_spacing, col_spacing)
    metadata['n_slices'] = len(slices)
    metadata['shape'] = (len(slices), int(ref.Rows), int(ref.Columns))

    # Modality (MRI, CT, US, etc.)
    metadata['modality'] = getattr(ref, 'Modality', 'Unknown')

    # Patient orientation
    try:
        metadata['orientation'] = list(ref.ImageOrientationPatient)
    except AttributeError:
        metadata['orientation'] = None

    # Study/Series info (anonymized-safe tags)
    metadata['series_description'] = getattr(ref, 'SeriesDescription', '')
    metadata['study_date'] = getattr(ref, 'StudyDate', '')

    return metadata


def apply_window(volume: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    Apply CT windowing (Hounsfield unit clipping).
    
    Common windows:
        Brain:    center=40,  width=80
        Lung:     center=-600, width=1500
        Bone:     center=400, width=1800
        Soft tissue: center=50, width=400

    Args:
        volume: 3D numpy array in Hounsfield units
        window_center: center of the window
        window_width: width of the window

    Returns:
        Windowed volume normalized to [0, 1]
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(volume, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed.astype(np.float32)


def get_volume_stats(volume: np.ndarray, metadata: Dict) -> Dict:
    """
    Print clinically useful statistics about a loaded volume.

    Args:
        volume: 3D numpy array
        metadata: metadata dict from load_dicom_series

    Returns:
        stats dict
    """
    spacing = metadata.get('spacing', (1, 1, 1))
    physical_size = (
        volume.shape[0] * spacing[0],
        volume.shape[1] * spacing[1],
        volume.shape[2] * spacing[2],
    )

    stats = {
        'shape_voxels': volume.shape,
        'physical_size_mm': physical_size,
        'voxel_spacing_mm': spacing,
        'intensity_min': float(volume.min()),
        'intensity_max': float(volume.max()),
        'intensity_mean': float(volume.mean()),
        'modality': metadata.get('modality', 'Unknown'),
    }
    return stats

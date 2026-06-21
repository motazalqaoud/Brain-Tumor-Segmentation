"""
nifti_loader.py
---------------
Load NIfTI files (.nii, .nii.gz) with full metadata and orientation handling.

Author: Motaz Alqaoud, PhD
GitHub: https://github.com/motazalqaoud

Clinical note:
    NIfTI is the standard format in research MRI (FSL, FreeSurfer, 3D Slicer).
    The affine matrix encodes the mapping from voxel indices to real-world 
    RAS coordinates — essential for any surgical navigation system.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Dict, Optional


def load_nifti(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a NIfTI file and return volume, affine, and metadata.

    Args:
        file_path: Path to .nii or .nii.gz file

    Returns:
        volume:   3D (or 4D) numpy array
        affine:   4x4 affine matrix (voxel → RAS world coordinates)
        metadata: dict with voxel spacing, shape, and header info

    Example:
        >>> volume, affine, meta = load_nifti("brain_mri.nii.gz")
        >>> print(volume.shape, meta['spacing'])
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")

    img = nib.load(str(path))
    volume = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header

    metadata = _extract_nifti_metadata(header, affine, volume.shape)

    return volume, affine, metadata


def _extract_nifti_metadata(header, affine: np.ndarray, shape: tuple) -> Dict:
    """Extract clinically relevant metadata from NIfTI header."""
    # Voxel dimensions (zooms = spacing in mm)
    zooms = header.get_zooms()
    spacing = tuple(float(z) for z in zooms[:3])

    metadata = {
        'spacing': spacing,
        'shape': shape,
        'affine': affine,
        'dim_info': str(header.get_dim_info()),
        'data_type': str(header.get_data_dtype()),
        'physical_size_mm': tuple(s * sp for s, sp in zip(shape[:3], spacing)),
    }

    # Orientation (RAS, LAS, etc.)
    try:
        orientation = nib.aff2axcodes(affine)
        metadata['orientation'] = ''.join(orientation)
    except Exception:
        metadata['orientation'] = 'Unknown'

    return metadata


def reorient_to_ras(volume: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorient a NIfTI volume to standard RAS orientation.
    
    RAS = Right, Anterior, Superior — the standard for neuroimaging.
    Critical for: multi-modal registration, atlas alignment, surgical planning.

    Args:
        volume: input 3D array
        affine: 4x4 affine matrix

    Returns:
        reoriented volume and updated affine
    """
    import nibabel as nib

    img = nib.Nifti1Image(volume, affine)
    img_ras = nib.as_closest_canonical(img)

    return img_ras.get_fdata(dtype=np.float32), img_ras.affine


def voxel_to_world(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel indices to world (RAS) coordinates using the affine matrix.

    Args:
        voxel_coords: array of shape (N, 3) — voxel [i, j, k] indices
        affine: 4x4 affine matrix

    Returns:
        world_coords: array of shape (N, 3) in mm

    Clinical use:
        Converting a tumor centroid in voxels to real-world OR coordinates.
    """
    voxel_coords = np.array(voxel_coords)
    if voxel_coords.ndim == 1:
        voxel_coords = voxel_coords[np.newaxis, :]

    ones = np.ones((voxel_coords.shape[0], 1))
    voxel_h = np.hstack([voxel_coords, ones])
    world_h = (affine @ voxel_h.T).T

    return world_h[:, :3]


def world_to_voxel(world_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert world (RAS) coordinates to voxel indices.

    Args:
        world_coords: array of shape (N, 3) in mm
        affine: 4x4 affine matrix

    Returns:
        voxel_coords: array of shape (N, 3) as integer indices
    """
    inv_affine = np.linalg.inv(affine)
    world_coords = np.array(world_coords)
    if world_coords.ndim == 1:
        world_coords = world_coords[np.newaxis, :]

    ones = np.ones((world_coords.shape[0], 1))
    world_h = np.hstack([world_coords, ones])
    voxel_h = (inv_affine @ world_h.T).T

    return np.round(voxel_h[:, :3]).astype(int)


def save_nifti(volume: np.ndarray, affine: np.ndarray, output_path: str,
               reference_header=None) -> None:
    """
    Save a numpy array as a NIfTI file.

    Args:
        volume: 3D numpy array
        affine: 4x4 affine matrix
        output_path: output .nii or .nii.gz path
        reference_header: optional header to copy metadata from
    """
    if reference_header is not None:
        img = nib.Nifti1Image(volume.astype(np.float32), affine, header=reference_header)
    else:
        img = nib.Nifti1Image(volume.astype(np.float32), affine)

    nib.save(img, output_path)
    print(f"Saved NIfTI: {output_path}")

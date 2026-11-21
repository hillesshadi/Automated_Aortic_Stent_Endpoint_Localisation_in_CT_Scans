#!/usr/bin/env python3
"""
Classical (non-deep-learning) aorta/stent segmentation helpers for the
Mini CT Viewer.

No trained model weights exist anywhere in this workspace, so segmentation
here is HU-threshold + connected-component + shape-filter based rather than
a neural network. It needs no training data and runs immediately on any
loaded CT series.

Fully automated: the spine is located first as the densest landmark in the
posterior-central part of the frame, the aorta is then found anterior and
image-right of the spine (radiological convention: image-right = patient's
anatomical left) and grown from there, and the stent is isolated by a metal
HU threshold constrained to the aorta mask. No mouse interaction is needed.

Pure numpy/scipy - no Qt, no torch. Keeps the viewer's UI code free of
image-processing detail and this module free of GUI detail.
"""

import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    center_of_mass,
    label,
)


# --------------------------------------------------------------------------- #
#  Spine landmark detection
# --------------------------------------------------------------------------- #
def detect_spine(volume_hu, bone_hu=150.0) -> Tuple[int, int]:
    """
    Locate the vertebral body as the largest dense (bone-HU) blob in a
    posterior-central region of interest, averaged over a handful of slices
    around the volume's vertical center for robustness.

    Returns (row, col) in full-image voxel coordinates.
    """
    z_total, h, w = volume_hu.shape
    z_center = z_total // 2
    z_samples = sorted(set(
        int(np.clip(z_center + off, 0, z_total - 1))
        for off in (-10, -5, 0, 5, 10)
    ))

    r0, r1 = int(0.40 * h), int(0.95 * h)
    c0, c1 = int(0.25 * w), int(0.75 * w)

    centroids = []
    for z in z_samples:
        roi = volume_hu[z, r0:r1, c0:c1]
        candidate = roi >= bone_hu
        if not candidate.any():
            continue
        labeled, num = label(candidate)
        if num == 0:
            continue
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        best = int(np.argmax(sizes))
        if sizes[best] < 20:
            continue
        rr, cc = center_of_mass(labeled == best)
        centroids.append((rr + r0, cc + c0))

    if not centroids:
        # Fallback: typical vertebral body position - lower-center of frame.
        return (int(0.65 * h), int(0.5 * w))

    rows, cols = zip(*centroids)
    return (int(round(np.mean(rows))), int(round(np.mean(cols))))


# --------------------------------------------------------------------------- #
#  Aorta seed localisation (anterior + image-right of the spine)
# --------------------------------------------------------------------------- #
def locate_aorta_seed(volume_hu, spine_rc) -> Optional[Tuple[int, int, int]]:
    """
    Search a small box anterior (smaller row index) and image-right (larger
    column index - the patient's anatomical left in radiological convention)
    of the spine, across a handful of central slices, for the largest
    roughly-round vessel/blood-pool blob. Returns (z, row, col) or None.
    """
    z_total, h, w = volume_hu.shape
    z_center = z_total // 2
    z_samples = sorted(set(
        int(np.clip(z_center + off, 0, z_total - 1))
        for off in (-15, -8, 0, 8, 15)
    ))

    spine_row, spine_col = spine_rc
    band_rows = max(20, int(0.16 * h))
    band_cols = max(20, int(0.22 * w))
    margin = max(4, int(0.015 * w))

    r0 = max(0, spine_row - band_rows)
    r1 = min(h, spine_row + int(0.03 * h))
    c0 = min(w, spine_col + margin)
    c1 = min(w, spine_col + margin + band_cols)
    
    if c1 - c0 < 5 or r1 - r0 < 5:
        return None

    best = None  # (score, z, row, col)
    for z in z_samples:
        roi = volume_hu[z, r0:r1, c0:c1].astype(np.float32)
        candidate = (roi >= -50.0) & (roi <= 400.0)
        if not candidate.any():
            continue
        labeled, num = label(candidate)

        for comp_id in range(1, num + 1):
            comp = labeled == comp_id
            area = int(comp.sum())
            if area < 40 or area > 0.5 * comp.size:
                continue

            eroded = binary_erosion(comp, iterations=2) ^ comp
            perimeter = max(1, int(eroded.sum()))
            circularity = (4.0 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.1: circularity = 1.0

            score = area * circularity       
            # coords = np.argwhere(comp)
            # spread = coords.std(axis=0)
            # roundness = float(spread.min() / (spread.max() + 1e-6))
            # score = area * roundness
            if best is None or score > best[0]:
                rr, cc = center_of_mass(comp)
                best = (score, z, int(round(rr)) + r0, int(round(cc)) + c0)

    if best is None:
        return None
    _, z, row, col = best
    return (z, row, col)


# --------------------------------------------------------------------------- #
#  Aorta segmentation: fully automated (spine landmark -> ROI -> grow)
# --------------------------------------------------------------------------- #
def auto_segment_aorta(volume_hu, tolerance=120):
    """
    Fully automated aorta segmentation: locate the spine, find the aorta
    seed anterior + image-right of it, then threshold to an HU band around
    that seed's value and keep the connected component containing the seed -
    all restricted to a row/col window around the seed (full Z range, since
    the descending aorta runs the length of the scan). This keeps the
    threshold/connected-component arrays proportional to the vessel's
    footprint instead of the whole 512x512 frame: on a memory-constrained
    machine, a (946, 512, 512) labeled array is a ~1GB allocation, which is
    exactly the kind of MemoryError this whole feature is meant to avoid.

    Returns (sub_mask, bbox, seed): sub_mask is cropped to bbox
    (z0, z1, y0, y1, x0, x1) in the *original* volume's coordinates; seed is
    (z, row, col) or None if no plausible aorta location was found.
    """
    shape = volume_hu.shape
    h, w = shape[1], shape[2]

    spine_rc = detect_spine(volume_hu)
    seed = locate_aorta_seed(volume_hu, spine_rc)
    if seed is None:
        return np.zeros((0, 0, 0), dtype=np.uint8), (0, 0, 0, 0, 0, 0), None

    z, row, col = seed
    row_band = max(80, int(0.20 * h))
    col_band = max(80, int(0.20 * w))
    y0, y1 = max(0, row - row_band), min(h, row + row_band)
    x0, x1 = max(0, col - col_band), min(w, col + col_band)
    bbox = (0, shape[0], y0, y1, x0, x1)

    sub_volume = volume_hu[:, y0:y1, x0:x1]
    seed_hu = float(volume_hu[z, row, col])
    lo, hi = seed_hu - tolerance, seed_hu + tolerance

    candidate = (sub_volume >= lo) & (sub_volume <= hi)
    labeled, _num = label(candidate)
    seed_label = labeled[z, row - y0, col - x0]
    if seed_label == 0:
        return np.zeros_like(candidate, dtype=np.uint8), bbox, seed

    sub_mask = (labeled == seed_label).astype(np.uint8)
    return postprocess_aorta_mask(sub_mask), bbox, seed


def postprocess_aorta_mask(mask):
    """Fill holes per axial slice, keep the largest 3-D component, light dilation."""
    filled = np.zeros_like(mask, dtype=np.uint8)
    for z in range(mask.shape[0]):
        filled[z] = binary_fill_holes(mask[z]).astype(np.uint8)

    labeled, num = label(filled)
    if num == 0:
        return filled
    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    largest = np.argmax(sizes) + 1
    cleaned = (labeled == largest).astype(np.uint8)

    return binary_dilation(cleaned, iterations=2).astype(np.uint8)


def embed_mask(shape, sub_mask, bbox, memmap_path=None):
    """
    Place a cropped mask into a full-volume-shaped array for display/indexing
    (e.g. `aorta_mask[idx]` per slice). Backed by a memmap file when a scratch
    path is given, so the full-size allocation (hundreds of MB for a large
    series) never has to land on the RAM heap in one contiguous block.
    """
    if memmap_path is not None:
        full = np.memmap(memmap_path, mode="w+", dtype=np.uint8, shape=shape)
    else:
        full = np.zeros(shape, dtype=np.uint8)
    z0, z1, y0, y1, x0, x1 = bbox
    if sub_mask.size:
        full[z0:z1, y0:y1, x0:x1] = sub_mask
    return full


# --------------------------------------------------------------------------- #
#  Stent segmentation: HU threshold inside aorta + shape filter
# --------------------------------------------------------------------------- #

from scipy.ndimage import binary_closing
def segment_stent(volume_hu, aorta_mask, hu_threshold=1200, dilate=3):
    """
    Threshold for bright metal, restricted to a dilated aorta mask - the
    stent frame sits at/near the vessel wall, not necessarily in the lumen
    center, so the aorta mask alone (undilated) would clip it.

    `aorta_mask` may be the small `auto_segment_aorta` sub_mask or a
    full-volume mask; either way the search is cropped to its own bounding
    box (+ padding for the dilation) rather than the whole volume.

    Returns (sub_mask, bbox) in the same convention as `auto_segment_aorta`.
    """
    shape = volume_hu.shape
    coords = np.argwhere(aorta_mask)
    if coords.size == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8), (0, 0, 0, 0, 0, 0)

    pad = dilate + 2
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + 1 + pad, shape)
    z0, y0, x0 = (int(v) for v in lo)
    z1, y1, x1 = (int(v) for v in hi)
    bbox = (z0, z1, y0, y1, x0, x1)

    sub_volume = volume_hu[z0:z1, y0:y1, x0:x1]
    sub_aorta = aorta_mask[z0:z1, y0:y1, x0:x1]
    region = binary_dilation(sub_aorta.astype(bool), iterations=dilate)
    # sub_mask = ((sub_volume >= hu_threshold) & region).astype(np.uint8)
# Base thresholding
    raw_metal = (sub_volume >= hu_threshold) & region
    closed_metal = np.zeros_like(raw_metal, dtype=np.uint8)
    
    # Structure element for closing gaps between individual struts (5x5 circular matrix template)
    struct = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]
    ], dtype=bool)
    
    # Process slice-by-slice along the local Z-axis
    for z in range(raw_metal.shape[0]):
        closed_metal[z] = binary_closing(raw_metal[z], structure=struct)

    return closed_metal, bbox
    # return sub_mask, bbox


def shape_filter_stent(
    sub_mask,
    bbox,
    spacing_zyx=(1.0, 1.0, 1.0),
    min_vol_mm3=20.0,
    max_vol_mm3=50_000.0,
    min_aspect=2.0,
    hu_volume=None,
    hu_threshold=None,
):
    """
    Keep only connected components that plausibly look like stent wire/mesh:
    within a volume range and elongated (tubular). Optionally also require
    a minimum mean HU (redundant with the pre-threshold, but useful if a
    caller passes in a raw mask built some other way).

    Operates on the cropped `(sub_mask, bbox)` pair returned by
    `segment_stent` - `hu_volume`, if given, must already be cropped to the
    same shape as `sub_mask` (e.g. `volume_hu[z0:z1, y0:y1, x0:x1]`).

    Returns (final_sub_mask, bbox).
    """
    if sub_mask.size == 0:
        return sub_mask, bbox

    voxel_vol_mm3 = float(np.prod(spacing_zyx))
    labeled, num = label(sub_mask)
    if num == 0:
        return sub_mask.astype(np.uint8), bbox

    final_mask = np.zeros_like(sub_mask, dtype=np.uint8)
    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        vol_mm3 = comp.sum() * voxel_vol_mm3
        if not (min_vol_mm3 <= vol_mm3 <= max_vol_mm3):
            continue

        coords = np.argwhere(comp)
        if len(coords) < 5:
            continue
        cov = np.cov((coords * np.array(spacing_zyx)).T)
        try:
            eigvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
            aspect = 0.0 if eigvals[2] < 1e-9 else np.sqrt(eigvals[0] / eigvals[2])
        except Exception:
            aspect = 0.0
        if aspect < min_aspect:
            continue

        if hu_volume is not None and hu_threshold is not None:
            if hu_volume[comp].mean() < hu_threshold:
                continue

        final_mask[comp] = 1

    return final_mask, bbox


# --------------------------------------------------------------------------- #
#  Stent geometry: PCA endpoints (summary stats), per-slice centroids,
#  and Z-scan endpoints (first/last slice where the stent consistently
#  appears - used for the start/end table rows and CSV export)
#
#  All three take the cropped (sub_mask, bbox) pair and return coordinates
#  already offset back into the original volume's voxel space, so callers
#  never need to know about the crop.
# --------------------------------------------------------------------------- #
def stent_endpoints(sub_mask, bbox, spacing_zyx=(1.0, 1.0, 1.0)) -> Dict:
    """
    PCA the stent voxel cloud, project onto its principal axis, and take the
    extreme-projection voxels as the two endpoints (start/end of the tube).

    Used for the summary length/volume line; see `stent_z_endpoints` for the
    first/last-slice endpoints used in the table and CSV export.
    """
    if sub_mask.size == 0 or sub_mask.sum() == 0:
        return {}

    z0, _z1, y0, _y1, x0, _x1 = bbox
    coords_vox = np.argwhere(sub_mask > 0)
    coords_mm = coords_vox * np.array(spacing_zyx)

    centroid_mm = coords_mm.mean(axis=0)
    centered = coords_mm - centroid_mm
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]

    proj = centered @ principal
    i_start, i_end = np.argmin(proj), np.argmax(proj)

    start_mm, end_mm = coords_mm[i_start], coords_mm[i_end]
    length_mm = float(np.linalg.norm(end_mm - start_mm))
    voxel_vol_mm3 = float(np.prod(spacing_zyx))
    offset = np.array([z0, y0, x0])

    return {
        "start_vox": tuple(int(v) for v in coords_vox[i_start] + offset),
        "end_vox": tuple(int(v) for v in coords_vox[i_end] + offset),
        "start_mm": tuple(float(v) for v in start_mm),
        "end_mm": tuple(float(v) for v in end_mm),
        "length_mm": length_mm,
        "volume_mm3": float(sub_mask.sum() * voxel_vol_mm3),
        "voxel_count": int(sub_mask.sum()),
    }


def per_slice_centroids(sub_mask, bbox, spacing_zyx=(1.0, 1.0, 1.0)) -> List[Dict]:
    """Centroid (row, col) in original-volume voxel space for every z-slice containing stent voxels."""
    if sub_mask.size == 0:
        return []
    z0, _z1, y0, _y1, x0, _x1 = bbox
    rows = []
    for z_local in range(sub_mask.shape[0]):
        sl = sub_mask[z_local]
        if sl.sum() == 0:
            continue
        r, c = center_of_mass(sl)
        rows.append({
            "z_index": z_local + z0,
            "row": float(r) + y0,
            "col": float(c) + x0,
            "voxel_count": int(sl.sum()),
        })
    return rows


def stent_z_endpoints(sub_mask, bbox, spacing_zyx=(1.0, 1.0, 1.0), min_voxels=5) -> Dict:
    """
    Scan the stent mask along Z and find the first slice index (z1) where the
    mesh consistently appears (voxel count at or above a small noise floor)
    and the last slice index (z2) where it ends, plus each boundary slice's
    (row, col) centroid. This is the geometry used for the Stent Table's
    Start/End rows and the CSV export.
    """
    per_slice = per_slice_centroids(sub_mask, bbox, spacing_zyx)
    consistent = [s for s in per_slice if s["voxel_count"] >= min_voxels]
    if not consistent:
        return {}

    first, last = consistent[0], consistent[-1]
    voxel_vol_mm3 = float(np.prod(spacing_zyx))
    return {
        "z1": first["z_index"], "row1": first["row"], "col1": first["col"],
        "z2": last["z_index"], "row2": last["row"], "col2": last["col"],
        "voxel_count": int(sub_mask.sum()),
        "volume_mm3": float(sub_mask.sum() * voxel_vol_mm3),
    }


def write_stent_endpoints_csv(row: Dict, path="stent_endpoints.csv") -> str:
    """Write a single-row CSV with the resolved start/end patient-space coordinates."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


# --------------------------------------------------------------------------- #
#  Pixel -> patient coordinate transform (standard DICOM formula)
# --------------------------------------------------------------------------- #
def voxel_to_patient(position, row, col, row_cosine, col_cosine, row_spacing, col_spacing):
    """
    P = IPP + col * col_spacing * RowCosine + row * row_spacing * ColCosine

    `row_cosine`/`col_cosine` are ImageOrientationPatient's first/second
    3-vectors (direction the row/column index advances in patient space).
    """
    position = np.asarray(position, dtype=float)
    row_cosine = np.asarray(row_cosine, dtype=float)
    col_cosine = np.asarray(col_cosine, dtype=float)
    return position + col * col_spacing * row_cosine + row * row_spacing * col_cosine


# --------------------------------------------------------------------------- #
#  Overlay compositing (grayscale -> RGB with translucent colored masks)
# --------------------------------------------------------------------------- #
def blend_overlay(gray_u8, layers):
    """
    layers: list of (mask_slice_bool, (r, g, b), alpha) to composite on top
    of the grayscale image, in order.
    """
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)
    for mask_slice, color, alpha in layers:
        if mask_slice is None or not mask_slice.any():
            continue
        color_arr = np.array(color, dtype=np.float32)
        m = mask_slice.astype(bool)
        rgb[m] = rgb[m] * (1 - alpha) + color_arr * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)

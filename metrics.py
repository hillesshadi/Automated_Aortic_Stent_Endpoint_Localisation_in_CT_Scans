#!/usr/bin/env python3
"""
Quantitative evaluation metrics for the Mini CT Viewer.

Two independent metric families, both pure functions (no file I/O, no Qt,
no torch):

  enhancement_metrics(raw, processed)   - image-quality comparison of a
                                           denoised/enhanced 2-D slice
                                           against the raw slice it came from
                                           (PSNR, MSE, SSIM, FSIM)

  segmentation_metrics(pred, gt, ...)   - overlap/distance comparison of a
                                           predicted binary mask against a
                                           ground-truth mask (DSC, IoU, HD)

MSE/PSNR/SSIM reuse skimage (as does the sibling aorta_stent_pipeline
project's enhancement_evaluation.py); FSIM has no ready-made implementation
in that project or in skimage, so it's implemented here from phase
congruency (via the `phasepack` port of Kovesi's algorithm) and gradient
magnitude, combined per Zhang et al. (2011) "FSIM: A Feature Similarity
Index for Image Quality Assessment".
"""

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.metrics import (
    mean_squared_error as _skimage_mse,
    peak_signal_noise_ratio as _skimage_psnr,
    structural_similarity as _skimage_ssim,
)
try:
    from phasepack import phasecong
    _PHASEPACK_ERROR = None
except Exception as _exc:                                      # noqa: BLE001
    # Only compute_fsim() needs this. Importing it at module scope meant a
    # missing optional dependency also broke dice/iou/hausdorff, which need
    # nothing but numpy and scipy -- so segmentation metrics failed for a
    # reason that had nothing to do with them.
    phasecong = None
    _PHASEPACK_ERROR = "%s: %s" % (type(_exc).__name__, _exc)

EPS = 1e-7


# --------------------------------------------------------------------------- #
#  Enhancement / image-quality metrics
# --------------------------------------------------------------------------- #
def fsim_available():
    """Whether FSIM can be computed (it needs the optional phasepack)."""
    return phasecong is not None


def _phase_congruency_energy(img_f64, nscale=2, norient=4):
    """Mean phase-congruency map across orientations (Kovesi's algorithm)."""
    if phasecong is None:
        raise RuntimeError("phasepack is not installed (%s)" % _PHASEPACK_ERROR)
    _m, _mn, _ori, _ft, pc, _eo, _t = phasecong(
        img_f64, nscale=nscale, norient=norient,
        minWaveLength=6, mult=2.0, sigmaOnf=0.55)
    return np.mean(np.stack(pc, axis=0), axis=0)


def compute_fsim(ref, test, data_range=255.0) -> float:
    """
    Feature Similarity Index: combines phase-congruency similarity (a
    contrast-invariant proxy for perceptually significant structure) with
    gradient-magnitude similarity.
    """
    if phasecong is None:
        return float('nan')      # optional dependency absent

    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    pc1 = _phase_congruency_energy(ref)
    pc2 = _phase_congruency_energy(test)

    gy1, gx1 = np.gradient(ref)
    gy2, gx2 = np.gradient(test)
    g1 = np.sqrt(gx1 ** 2 + gy1 ** 2)
    g2 = np.sqrt(gx2 ** 2 + gy2 ** 2)

    t1 = 0.85
    t2 = 160.0 * (data_range / 255.0) ** 2
    s_pc = (2 * pc1 * pc2 + t1) / (pc1 ** 2 + pc2 ** 2 + t1)
    s_g = (2 * g1 * g2 + t2) / (g1 ** 2 + g2 ** 2 + t2)
    s_l = s_pc * s_g

    pc_max = np.maximum(pc1, pc2)
    denom = np.sum(pc_max)
    if denom < EPS:
        return 1.0
    return float(np.sum(s_l * pc_max) / denom)


def enhancement_metrics(raw_slice, processed_slice, data_range=255.0) -> Dict[str, float]:
    """
    Compare a processed (denoised/enhanced) 2-D slice against the raw slice
    it was derived from.

    Parameters
    ----------
    raw_slice, processed_slice : 2-D arrays, same shape (uint8 display slices)
    """
    ref = raw_slice.astype(np.float64)
    test = processed_slice.astype(np.float64)

    mse = float(_skimage_mse(ref, test))
    psnr = float(_skimage_psnr(ref, test, data_range=data_range)) if mse > EPS else float("inf")
    ssim = float(_skimage_ssim(ref, test, data_range=data_range))
    fsim = compute_fsim(ref, test, data_range=data_range)

    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim, "FSIM": fsim}


# --------------------------------------------------------------------------- #
#  Segmentation validation metrics
# --------------------------------------------------------------------------- #
def dice(pred, gt) -> float:
    """Dice Similarity Coefficient = 2|P n G| / (|P|+|G|)."""
    p, g = pred.astype(bool).ravel(), gt.astype(bool).ravel()
    tp = (p & g).sum()
    return float(2 * tp / (p.sum() + g.sum() + EPS))


def iou(pred, gt) -> float:
    """Intersection over Union (Jaccard) = |P n G| / |P u G|."""
    p, g = pred.astype(bool).ravel(), gt.astype(bool).ravel()
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter / (union + EPS))


def hausdorff_distance(pred, gt, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """
    Hausdorff Distance (mm): the largest of the two nearest-surface-point
    distances (pred->gt and gt->pred), computed via Euclidean distance
    transforms from each mask's border.
    """
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return float("inf")

    pred_border = pred_b ^ binary_erosion(pred_b)
    gt_border = gt_b ^ binary_erosion(gt_b)

    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)

    d_p2g = dt_gt[pred_border]
    d_g2p = dt_pred[gt_border]
    return float(max(d_p2g.max(), d_g2p.max()))


def segmentation_metrics(pred, gt, spacing_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """DSC, IoU, and Hausdorff Distance (mm) of a predicted mask vs. ground truth."""
    return {
        "DSC": dice(pred, gt),
        "IoU": iou(pred, gt),
        "HD_mm": hausdorff_distance(pred, gt, spacing_zyx),
    }

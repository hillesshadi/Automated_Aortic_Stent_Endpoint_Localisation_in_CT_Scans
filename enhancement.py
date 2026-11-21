#!/usr/bin/env python3
"""
2-D image enhancement / denoising / edge-detection filter suite for the
Mini CT Viewer.

Every filter operates on a single 2-D uint8 slice (the already
window/level-mapped image the viewport is about to draw) and returns a
uint8 slice of the same shape - never the full 3-D volume, so applying a
filter while scrubbing through a large series has a flat, small memory
footprint regardless of how many slices the series has.

Usage
-----
    import enhancement as enh
    filtered = enh.FILTERS["Gaussian"](u8_slice, sigma=1.5)
"""

import numpy as np
import cv2
from scipy.ndimage import (
    gaussian_filter as _nd_gaussian,
    gaussian_gradient_magnitude,
    median_filter as _nd_median,
)
from scipy.signal import wiener as _sig_wiener
from skimage.restoration import denoise_nl_means, denoise_wavelet, estimate_sigma


def _to_float01(img_u8):
    return img_u8.astype(np.float32) / 255.0


def _to_u8(img_float01):
    return np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Spatial filters
# --------------------------------------------------------------------------- #
def gaussian_filter2d(img_u8, sigma=1.0):
    out = _nd_gaussian(img_u8.astype(np.float32), sigma=float(sigma))
    return np.clip(out, 0, 255).astype(np.uint8)


def median_filter2d(img_u8, size=3):
    size = int(size) | 1  # force odd
    return _nd_median(img_u8, size=size)


def bilateral_filter2d(img_u8, sigma_color=50.0, sigma_space=50.0):
    return cv2.bilateralFilter(
        img_u8, d=0, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))


# --------------------------------------------------------------------------- #
#  Advanced denoising
# --------------------------------------------------------------------------- #
def nlm_filter2d(img_u8, h=10.0):
    f = _to_float01(img_u8)
    sigma_est = float(np.mean(estimate_sigma(f)))
    denoised = denoise_nl_means(
        f, h=float(h) / 255.0, fast_mode=True,
        patch_size=5, patch_distance=6, sigma=sigma_est)
    return _to_u8(denoised)


def wavelet_denoise2d(img_u8, sigma=0.1):
    f = _to_float01(img_u8)
    denoised = denoise_wavelet(f, sigma=float(sigma), rescale_sigma=True)
    return _to_u8(denoised)


def wiener_filter2d(img_u8, size=5):
    size = int(size) | 1
    out = _sig_wiener(img_u8.astype(np.float32), mysize=size)
    return np.nan_to_num(out, nan=0.0).clip(0, 255).astype(np.uint8)


def eadtv_denoise2d(img_u8, lam=0.15, iterations=50, alpha=0.5, dt=0.1):
    """
    Edge-Adaptive (Dual) Total Variation denoising, adapted per-2D-slice
    from the volumetric EADTV implementation in aorta_stent_pipeline's
    preprocessing.py (dual/Chambolle projected-gradient TV minimisation with
    a locally adaptive weight w(x) = exp(-alpha * |grad(u)|) that relaxes
    smoothing near edges).
    """
    f = _to_float01(img_u8).astype(np.float64)
    u = f.copy()
    px = np.zeros_like(u)
    py = np.zeros_like(u)

    def edge_weight(img):
        grad = gaussian_gradient_magnitude(img, sigma=1.0)
        return np.exp(-alpha * grad)

    w = edge_weight(u)
    for it in range(int(iterations)):
        if it % 10 == 0 and it > 0:
            w = edge_weight(u)
        gx = np.roll(u, -1, axis=1) - u
        gy = np.roll(u, -1, axis=0) - u
        px_new = px + dt * w * gx
        py_new = py + dt * w * gy
        norm = np.maximum(1.0, np.sqrt(px_new ** 2 + py_new ** 2))
        px, py = px_new / norm, py_new / norm
        dx = px - np.roll(px, 1, axis=1)
        dy = py - np.roll(py, 1, axis=0)
        u = f - lam * (dx + dy)

    return _to_u8(np.clip(u, 0.0, 1.0))


# --------------------------------------------------------------------------- #
#  Variational & local contrast enhancement
# --------------------------------------------------------------------------- #
def clahe2d(img_u8, clip_limit=2.0):
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    return clahe.apply(img_u8)


# --------------------------------------------------------------------------- #
#  Edge extraction
# --------------------------------------------------------------------------- #
def sobel_edges2d(img_u8):
    gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = mag / (mag.max() + 1e-6) * 255.0
    return mag.astype(np.uint8)


def laplacian_edges2d(img_u8):
    lap = np.abs(cv2.Laplacian(img_u8, cv2.CV_32F, ksize=3))
    lap = lap / (lap.max() + 1e-6) * 255.0
    return lap.astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Registry - drives the Enhancement panel's filter dropdown directly.
#  Each params tuple is (name, min, max, step, default) for building spinboxes.
# --------------------------------------------------------------------------- #
FILTERS = {
    "Gaussian": gaussian_filter2d,
    "Median": median_filter2d,
    "Bilateral": bilateral_filter2d,
    "Non-Local Means": nlm_filter2d,
    "Wavelet Denoising": wavelet_denoise2d,
    "Wiener": wiener_filter2d,
    "EADTV": eadtv_denoise2d,
    "CLAHE": clahe2d,
    "Sobel Edges": sobel_edges2d,
    "Laplacian Edges": laplacian_edges2d,
}

FILTER_PARAMS = {
    "Gaussian": [("sigma", 0.1, 5.0, 0.1, 1.0)],
    "Median": [("size", 3, 15, 2, 3)],
    "Bilateral": [("sigma_color", 1, 150, 1, 50), ("sigma_space", 1, 150, 1, 50)],
    "Non-Local Means": [("h", 1, 30, 1, 10)],
    "Wavelet Denoising": [("sigma", 0.01, 1.0, 0.01, 0.1)],
    "Wiener": [("size", 3, 15, 2, 5)],
    "EADTV": [("lam", 0.01, 1.0, 0.01, 0.15), ("iterations", 5, 200, 5, 50)],
    "CLAHE": [("clip_limit", 0.5, 10.0, 0.5, 2.0)],
    "Sobel Edges": [],
    "Laplacian Edges": [],
}

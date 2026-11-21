#!/usr/bin/env python3
"""
2-D figure output for batch runs -- matplotlib on Agg, no Qt and no OpenGL.

These are the same views the desktop panel draws, rendered straight to PNG.
The drawing is deliberately re-implemented here rather than imported from
gui/widgets/mpr_panel.py: that module subclasses QWidget and pins the QtAgg
backend at import time, so importing it on a compute node would pull in Qt and
fail exactly where this pipeline must not.
"""

import os

import matplotlib
matplotlib.use("Agg")                 # before pyplot; no display is required
import numpy as np                                              # noqa: E402
from matplotlib.figure import Figure                            # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg     # noqa: E402
from matplotlib.patches import Circle                           # noqa: E402

from . import paths  # noqa: F401

BG = "#0b0e14"
FG = "#93a2c0"
PLANES = ("axial", "sagittal", "coronal")


def _save(fig, path, dpi=140):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    return path


def _window(hu, wc, ww):
    lo, hi = wc - ww / 2.0, wc + ww / 2.0
    if hi <= lo:
        hi = lo + 1.0
    out = (np.asarray(hu, np.float32) - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)


def _mask_rgba(mask, rgb, alpha):
    m = np.asarray(mask, bool)
    out = np.zeros(m.shape + (4,), np.float32)
    out[m, 0], out[m, 1], out[m, 2] = rgb
    out[m, 3] = alpha
    return out


def _blank(ax, text):
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, ha="center", va="center", color="#3a465e",
            transform=ax.transAxes, fontsize=9)


def _cursor_for_station(ctx, station):
    nz, ny, nx = ctx.volume.shape
    if ctx.resampler is None:
        return (nz // 2, ny // 2, nx // 2)
    from stent_endpoint_sop import world_to_idx
    i = int(np.clip(station, 0, len(ctx.resampler) - 1))
    ix, iy, iz = world_to_idx(ctx.resampler.points[i], ctx.geom)
    return (int(np.clip(round(iz), 0, nz - 1)),
            int(np.clip(round(iy), 0, ny - 1)),
            int(np.clip(round(ix), 0, nx - 1)))


# --------------------------------------------------------------------------- #
#  MPR montage
# --------------------------------------------------------------------------- #
def mpr_montage(ctx, path, station=None, wc=300, ww=600, lumen_hu=200,
                fov_mm=60, res_mm=0.4, size=(11, 9)):
    """The 2x2 MPR grid at one station, as a single PNG."""
    import mpr as mpr_mod

    if station is None:
        station = (len(ctx.resampler) // 2) if ctx.resampler is not None else 0
    cursor = _cursor_for_station(ctx, station)

    fig = Figure(figsize=size, facecolor=BG)
    rects = {"axial": [0.0, 0.5, 0.5, 0.5], "sagittal": [0.5, 0.5, 0.5, 0.5],
             "coronal": [0.0, 0.0, 0.5, 0.5], "orthogonal": [0.5, 0.0, 0.5, 0.5]}

    for plane in PLANES:
        ax = fig.add_axes(rects[plane])
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        raw = mpr_mod.plane_slice(ctx.volume, plane, cursor)
        aspect = mpr_mod.plane_aspect(plane, ctx.spacing_zyx)
        ax.imshow(_window(raw, wc, ww), cmap="gray", vmin=0, vmax=255,
                  aspect=aspect, interpolation="bilinear", origin="upper")
        ax.set_adjustable("datalim")
        if ctx.aorta_mask is not None:
            ax.imshow(_mask_rgba(mpr_mod.plane_slice(ctx.aorta_mask, plane, cursor),
                                 (0.86, 0.26, 0.28), 0.30), aspect=aspect,
                      interpolation="nearest", origin="upper")
        if ctx.stent_mask is not None:
            ax.imshow(_mask_rgba(mpr_mod.plane_slice(ctx.stent_mask, plane, cursor),
                                 (0.30, 0.95, 1.00), 0.75), aspect=aspect,
                      interpolation="nearest", origin="upper")
        row, col = mpr_mod.plane_crosshair(plane, cursor)
        ax.axhline(row, color="#00e5ff", lw=0.7, alpha=0.8)
        ax.axvline(col, color="#00e5ff", lw=0.7, alpha=0.8)
        ax.text(0.012, 0.978, "%s  z=%d y=%d x=%d" % (plane.capitalize(), *cursor),
                transform=ax.transAxes, ha="left", va="top", color=FG,
                fontsize=8, bbox=dict(facecolor=BG, edgecolor="none", alpha=0.65))

    ax = fig.add_axes(rects["orthogonal"])
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    if ctx.resampler is None:
        _blank(ax, "no centerline")
    else:
        from .centerline import cross_section_at
        patch, extent, metrics, info = cross_section_at(
            ctx, station, fov_mm=fov_mm, res_mm=res_mm, lumen_hu=lumen_hu)
        lo, hi = extent
        ax.imshow(_window(patch, wc, ww), cmap="gray", vmin=0, vmax=255,
                  extent=[lo, hi, hi, lo], interpolation="bilinear")
        ax.set_adjustable("datalim")
        ax.axhline(0, color="#ffd24a", lw=0.7, alpha=0.85)
        ax.axvline(0, color="#ffd24a", lw=0.7, alpha=0.85)
        d = metrics.get("diameter_mm", 0.0)
        if d > 0:
            ax.add_patch(Circle((0, 0), d / 2.0, fill=False, ec="#6ee7a0",
                                lw=1.0, alpha=0.9))
        label = ("Orthogonal  %.1f mm along  ·  lumen %.1f mm"
                 % (info["arc_mm"], d))
        if metrics.get("n_components", 1) > 1:
            label += "  (plane crosses %dx)" % metrics["n_components"]
        ax.text(0.012, 0.978, label, transform=ax.transAxes, ha="left",
                va="top", color=FG, fontsize=8,
                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.65))
    return _save(fig, path)


def cross_section_strip(ctx, path, n=8, wc=300, ww=600, lumen_hu=200,
                        fov_mm=60, res_mm=0.4):
    """Evenly spaced orthogonal cross-sections along the whole vessel."""
    if ctx.resampler is None:
        return None
    from .centerline import cross_section_at

    total = len(ctx.resampler)
    idx = np.unique(np.linspace(0, total - 1, min(n, total)).astype(int))
    cols = min(4, len(idx))
    rows = int(np.ceil(len(idx) / cols))
    fig = Figure(figsize=(3.0 * cols, 3.1 * rows), facecolor=BG)

    for k, station in enumerate(idx):
        ax = fig.add_subplot(rows, cols, k + 1)
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            patch, extent, metrics, info = cross_section_at(
                ctx, int(station), fov_mm=fov_mm, res_mm=res_mm,
                lumen_hu=lumen_hu)
        except Exception as exc:                               # noqa: BLE001
            _blank(ax, str(exc)[:40])
            continue
        lo, hi = extent
        ax.imshow(_window(patch, wc, ww), cmap="gray", vmin=0, vmax=255,
                  extent=[lo, hi, hi, lo], interpolation="bilinear")
        d = metrics.get("diameter_mm", 0.0)
        if d > 0:
            ax.add_patch(Circle((0, 0), d / 2.0, fill=False, ec="#6ee7a0",
                                lw=1.0, alpha=0.9))
        ax.set_title("%.0f mm  ·  %.1f mm" % (info["arc_mm"], d),
                     color=FG, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def diameter_profile(profile, path, specs=None):
    """Lumen diameter against arc length, with the device diameter marked."""
    if not profile:
        return None
    fig = Figure(figsize=(10, 4), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#2a3347")
    ax.tick_params(colors=FG, labelsize=8)
    ax.set_xlabel("distance along centerline (mm)", color=FG, fontsize=9)
    ax.set_ylabel("lumen diameter (mm)", color=FG, fontsize=9)

    arc = np.asarray(profile["arc_mm"], float)
    dia = np.asarray(profile["diameter_mm"], float)
    ax.plot(arc, dia, color="#6ee7a0", lw=1.4, label="measured")

    multi = np.asarray(profile.get("multi_lumen", np.zeros_like(arc)), bool)
    if multi.any():
        # Where the flat plane re-crosses a bend, the centre lumen was used;
        # mark those so the reader can see which points were disambiguated.
        ax.plot(arc[multi], dia[multi], "o", ms=3, color="#ffcc66",
                label="plane crossed the vessel more than once")
    if specs and specs.get("diameter_mm"):
        ax.axhline(specs["diameter_mm"], color="#4ddcff", ls="--", lw=1.0,
                   label="device %.0f mm" % specs["diameter_mm"])
    legend = ax.legend(fontsize=8, facecolor="#12161f", edgecolor="#2a3347",
                       labelcolor="#c8d3e6")
    legend.get_frame().set_alpha(0.9)
    fig.tight_layout()
    return _save(fig, path)


def hu_histograms(report, path):
    """HU distribution per structure, on a log count axis."""
    structures = (report or {}).get("structures", {})
    if not structures:
        return None
    colors = {"aorta": "#e04447", "stent": "#4ddcff", "whole_volume": "#93a2c0"}
    fig = Figure(figsize=(9, 4.5), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#2a3347")
    ax.tick_params(colors=FG, labelsize=8)
    ax.set_xlabel("Hounsfield units", color=FG, fontsize=9)
    ax.set_ylabel("voxels (log)", color=FG, fontsize=9)

    drawn = False
    for name, entry in structures.items():
        h = entry.get("histogram")
        if not h or not h.get("n"):
            continue
        edges = np.asarray(h["edges"], float)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centres, np.maximum(np.asarray(h["counts"], float), 0.5),
                where="mid", color=colors.get(name, FG), lw=1.2,
                label="%s (n=%d)" % (name.replace("_", " "), h["n"]))
        drawn = True
    if not drawn:
        return None
    ax.set_yscale("log")
    legend = ax.legend(fontsize=8, facecolor="#12161f", edgecolor="#2a3347",
                       labelcolor="#c8d3e6")
    legend.get_frame().set_alpha(0.9)
    fig.tight_layout()
    return _save(fig, path)

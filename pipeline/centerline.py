#!/usr/bin/env python3
"""
Centerline stage: skeletonise the aorta mask, take the longest geodesic path,
resample it to a uniform step, and build the orthogonal-resampling frames the
2-D cross-section panel scrubs along.

Both halves are existing code:
    stentlib.centerline.aorta_centerline   skeleton -> longest path -> spline
                                           resample, returning world mm
    stentlib.centerline.orient_root_to_distal
                                           proximal-first ordering
    oblique_mpr.CenterlineResampler        arc-length + parallel-transport
                                           frames for perpendicular cuts
"""

import numpy as np

from . import paths  # noqa: F401
from .context import CaseContext


class CenterlineError(RuntimeError):
    pass


def extract_centerline(ctx: CaseContext, step_mm=1.0, orient=True, progress=None):
    """Fill ctx.centerline and ctx.resampler from ctx.aorta_mask."""
    from stentlib.centerline import aorta_centerline, orient_root_to_distal

    if ctx.aorta_mask is None or not ctx.aorta_mask.any():
        raise CenterlineError(
            "No aorta mask.\n\nRun Preprocessing first — the centerline is "
            "the skeleton of the segmented aorta, so it needs that mask.")
    if ctx.geom is None:
        raise CenterlineError("Volume geometry is missing; reload the case.")

    if progress:
        progress(15, "Skeletonising aorta and searching longest path…")
    cl = aorta_centerline(ctx.aorta_mask, ctx.geom, float(step_mm))
    if cl is None or len(cl) < 4:
        raise CenterlineError(
            "Centerline extraction produced no usable path.\n\n"
            "That usually means the aorta mask is fragmented or too small. "
            "Check the Aorta HU tolerance and re-run Preprocessing.")

    rule = "none"
    if orient:
        if progress:
            progress(55, "Orienting proximal → distal…")
        # No heart mask here (that comes from TotalSegmentator in the full
        # pipeline), so this falls back to the anterior/posterior then world-Z
        # rules built into the same function.
        cl, rule = orient_root_to_distal(cl, None, ctx.geom)

    if progress:
        progress(75, "Building orthogonal frames…")
    import oblique_mpr as om
    try:
        resampler = om.CenterlineResampler(cl, step_mm=float(step_mm))
    except om.CenterlineError as exc:
        raise CenterlineError(str(exc)) from exc

    ctx.centerline = np.asarray(cl, float)
    ctx.resampler = resampler
    ctx.metrics["centerline_points"] = int(len(resampler))
    ctx.metrics["centerline_length_mm"] = float(resampler.total_length_mm)
    ctx.metrics["centerline_orientation_rule"] = rule

    if progress:
        progress(100, "Centerline: %d points, %.1f mm (orientation: %s)"
                 % (len(resampler), resampler.total_length_mm, rule))
    return ctx


def cross_section_at(ctx: CaseContext, index, fov_mm=60.0, res_mm=0.4,
                     lumen_hu=200.0):
    """Perpendicular cross-section plus lumen measurement at a centerline index.

    Returns (patch, extent, metrics, info) where `info` carries the local
    tangent, obliquity and curvature radius for the panel's readout.
    """
    import oblique_mpr as om
    if ctx.resampler is None:
        raise CenterlineError("No centerline — run Extract Centerline first.")

    rs = ctx.resampler
    i = int(np.clip(index, 0, len(rs) - 1))
    patch, extent = rs.cross_section(ctx.volume.astype(np.float32), ctx.geom, i,
                                     fov_mm=float(fov_mm), res_mm=float(res_mm))
    metrics = om.lumen_metrics(patch, float(res_mm), hu_threshold=float(lumen_hu))

    t = rs.tangents[i]
    curvature = rs.curvature_radius_mm()[i]
    info = {
        "index": i,
        "arc_mm": float(rs.arclength[i]),
        "point": rs.points[i].tolist(),
        "tangent": [float(v) for v in t],
        # Angle between the vessel and the scanner's axial plane: 0 means an
        # axial slice would already be orthogonal, so this quantifies how much
        # this view differs from plain MPR at this location.
        "obliquity_deg": float(np.degrees(np.arccos(
            float(np.clip(abs(t[2]), 0.0, 1.0))))),
        "curvature_radius_mm": float(curvature) if np.isfinite(curvature) else None,
    }
    return patch, extent, metrics, info


def diameter_profile(ctx: CaseContext, fov_mm=60.0, res_mm=0.5, lumen_hu=200.0,
                     stride=1, progress=None, should_stop=None):
    """Lumen diameter along the whole centerline.

    Uses the centre-connected component at each station (see
    oblique_mpr.lumen_metrics): a plane perpendicular to a bend re-crosses the
    same vessel further along it, and summing every bright pixel there would
    overstate the diameter substantially.
    """
    import oblique_mpr as om
    if ctx.resampler is None:
        raise CenterlineError("No centerline — run Extract Centerline first.")

    rs = ctx.resampler
    idx = list(range(0, len(rs), max(1, int(stride))))
    vol = ctx.volume.astype(np.float32)
    arc, diam, multi = [], [], []

    for n, i in enumerate(idx):
        if should_stop and should_stop():
            raise CenterlineError("Cancelled.")
        patch, _ = rs.cross_section(vol, ctx.geom, i, fov_mm=float(fov_mm),
                                    res_mm=float(res_mm))
        m = om.lumen_metrics(patch, float(res_mm), hu_threshold=float(lumen_hu))
        arc.append(float(rs.arclength[i]))
        diam.append(float(m["diameter_mm"]))
        multi.append(int(m["n_components"]) > 1)
        if progress and (n % 8 == 0 or n == len(idx) - 1):
            progress(int(100 * (n + 1) / len(idx)),
                     "Profiling %d/%d stations" % (n + 1, len(idx)))

    arc, diam = np.asarray(arc), np.asarray(diam)
    ctx.metrics["diameter_mean_mm"] = float(np.mean(diam)) if len(diam) else 0.0
    ctx.metrics["diameter_min_mm"] = float(np.min(diam)) if len(diam) else 0.0
    ctx.metrics["diameter_max_mm"] = float(np.max(diam)) if len(diam) else 0.0
    return {"arc_mm": arc, "diameter_mm": diam, "multi_lumen": np.asarray(multi)}

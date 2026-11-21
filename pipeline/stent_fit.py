#!/usr/bin/env python3
"""
Stent-fitting / verification stage.

Two independent descriptions of the same device, deliberately kept separate
because they answer different questions and can disagree:

  * PCA endpoints (segmentation.stent_endpoints) -- the principal axis of the
    metal voxel cloud and its extreme projections. A pure shape measurement.

  * Centerline landing zones -- the same endpoints projected onto the aortic
    centerline, giving the arc position of each crown and therefore the
    covered segment in vessel coordinates rather than in straight-line space.

The straight-line PCA length is always <= the along-vessel span, and the gap
between them is a direct measure of how curved the stented segment is; a large
divergence is reported rather than averaged away.
"""

import numpy as np

from . import paths  # noqa: F401
from .context import CaseContext

# The pipeline's own plausibility window for a deployed device. Outside this,
# the fit is reported but flagged rather than silently trusted.
MIN_PLAUSIBLE_MM = 30.0
MAX_PLAUSIBLE_MM = 400.0


class StentFitError(RuntimeError):
    pass


def _bbox_of(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]),
            int(lo[2]), int(hi[2]))


def fit_stent(ctx: CaseContext, progress=None):
    """Fit the stent and verify it against the centerline. Fills ctx.stent_result."""
    import segmentation as seg

    if ctx.stent_mask is None or not ctx.stent_mask.any():
        raise StentFitError(
            "No stent mask.\n\nRun Preprocessing first, and if it still finds "
            "no metal, lower the Stent HU threshold — a bare-metal frame can "
            "sit well below the 1200 HU default on a low-kVp series.")

    if progress:
        progress(15, "Principal-axis fit of the metal cloud…")

    bbox = _bbox_of(ctx.stent_mask)
    z0, z1, y0, y1, x0, x1 = bbox
    sub = ctx.stent_mask[z0:z1, y0:y1, x0:x1]

    pca = seg.stent_endpoints(sub, bbox, ctx.spacing_zyx)
    if not pca:
        raise StentFitError("The stent mask was too small to fit an axis to.")

    if progress:
        progress(45, "Slice-wise extent…")
    zends = seg.stent_z_endpoints(sub, bbox, ctx.spacing_zyx)

    result = {
        "length_mm": float(pca.get("length_mm", 0.0)),
        "volume_mm3": float(pca.get("volume_mm3", 0.0)),
        "voxel_count": int(pca.get("voxel_count", 0)),
        "start_vox": pca.get("start_vox"),
        "end_vox": pca.get("end_vox"),
        "z_first": zends.get("z1"),
        "z_last": zends.get("z2"),
        "warnings": [],
    }

    # ---- verification against the centerline --------------------------------
    if ctx.resampler is not None and ctx.geom is not None:
        if progress:
            progress(70, "Projecting crowns onto the centerline…")
        try:
            result.update(_landing_zones(ctx, pca))
        except Exception as exc:                               # noqa: BLE001
            result["warnings"].append("Centerline projection failed: %s" % exc)
    else:
        result["warnings"].append(
            "No centerline yet — run Extract Centerline for landing zones "
            "and along-vessel length.")

    # ---- plausibility -------------------------------------------------------
    length = result["length_mm"]
    if not (MIN_PLAUSIBLE_MM <= length <= MAX_PLAUSIBLE_MM):
        result["warnings"].append(
            "Implausible stent length %.1f mm (expected %.0f–%.0f mm). The "
            "metal threshold may be catching calcium or clipping the frame."
            % (length, MIN_PLAUSIBLE_MM, MAX_PLAUSIBLE_MM))

    ctx.stent_result = result
    ctx.metrics["stent_length_mm"] = length
    if progress:
        progress(100, "Stent: %.1f mm, %.0f mm³, %d voxels"
                 % (length, result["volume_mm3"], result["voxel_count"]))
    return result


def _landing_zones(ctx: CaseContext, pca):
    """Arc positions of the two crowns along the centerline.

    stent_endpoints returns endpoints in voxel indices and in
    spacing-scaled mm, but that mm value is *index * spacing*, not world
    coordinates -- it ignores the volume origin and direction. The centerline
    is in world/LPS mm, so the voxel indices are converted properly here
    instead of comparing the two millimetre spaces directly.
    """
    from stentlib.centerline import zyx_to_world

    start_vox = np.asarray(pca["start_vox"], float).reshape(1, 3)
    end_vox = np.asarray(pca["end_vox"], float).reshape(1, 3)
    world = zyx_to_world(np.vstack([start_vox, end_vox]), ctx.geom)

    rs = ctx.resampler
    pts = rs.points
    out = {}
    arcs = []
    for label, w in (("proximal", world[0]), ("distal", world[1])):
        d = np.linalg.norm(pts - w[None, :], axis=1)
        i = int(np.argmin(d))
        arcs.append(float(rs.arclength[i]))
        out["%s_arc_mm" % label] = float(rs.arclength[i])
        out["%s_index" % label] = i
        out["%s_offset_mm" % label] = float(d[i])
        out["%s_world" % label] = [float(v) for v in w]

    lo, hi = min(arcs), max(arcs)
    out["along_vessel_length_mm"] = hi - lo
    out["covered_fraction"] = ((hi - lo) / rs.total_length_mm
                               if rs.total_length_mm else 0.0)

    straight = float(pca.get("length_mm", 0.0))
    if straight > 1e-6:
        out["tortuosity"] = (hi - lo) / straight
    return out


def export_results(ctx: CaseContext, out_dir, progress=None):
    """Write a JSON summary, a centerline CSV, and the stent-endpoint CSV.

    Uses segmentation.write_stent_endpoints_csv for the endpoint row so the
    column layout matches what the desktop viewer and the batch pipeline
    already produce, rather than inventing a third format.
    """
    import csv
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    case = ctx.case_id or "case"
    written = []

    if progress:
        progress(20, "Writing summary JSON…")
    summary = {
        "case_id": case,
        "source_path": ctx.source_path,
        "source_kind": ctx.source_kind,
        "shape_zyx": list(ctx.volume.shape) if ctx.has_volume else None,
        "spacing_zyx": list(ctx.spacing_zyx) if ctx.spacing_zyx else None,
        "metrics": ctx.metrics,
        "stent": ctx.stent_result,
        "notes": ctx.notes,
    }
    path = os.path.join(out_dir, "%s_summary.json" % case)
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    written.append(path)

    if ctx.has_centerline:
        if progress:
            progress(55, "Writing centerline CSV…")
        path = os.path.join(out_dir, "%s_centerline.csv" % case)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "arc_mm", "x", "y", "z"])
            rs = ctx.resampler
            for i, p in enumerate(rs.points):
                w.writerow([i, "%.4f" % rs.arclength[i],
                            "%.4f" % p[0], "%.4f" % p[1], "%.4f" % p[2]])
        written.append(path)

        if progress:
            progress(75, "Writing centerline VTK…")
        path = os.path.join(out_dir, "%s_centerline.vtk" % case)
        _write_polyline_vtk(path, ctx.resampler.points)
        written.append(path)

    if ctx.stent_result:
        if progress:
            progress(90, "Writing stent endpoints CSV…")
        import segmentation as seg
        row = {"case_id": case}
        row.update({k: v for k, v in ctx.stent_result.items()
                    if not isinstance(v, (list, dict))})
        path = os.path.join(out_dir, "%s_stent_endpoints.csv" % case)
        seg.write_stent_endpoints_csv(row, path)
        written.append(path)

    if progress:
        progress(100, "Exported %d file(s) to %s" % (len(written), out_dir))
    return written


def _write_polyline_vtk(path, points):
    """Legacy ASCII VTK PolyData, the format oblique_mpr.load_centerline_points
    and the batch pipeline both already read."""
    pts = np.asarray(points, float).reshape(-1, 3)
    with open(path, "w", newline="\n") as fh:
        fh.write("# vtk DataFile Version 3.0\ncenterline\nASCII\n"
                 "DATASET POLYDATA\n")
        fh.write("POINTS %d float\n" % len(pts))
        for p in pts:
            fh.write("%.4f %.4f %.4f\n" % (p[0], p[1], p[2]))
        fh.write("LINES 1 %d\n%d %s\n"
                 % (len(pts) + 1, len(pts), " ".join(str(i) for i in range(len(pts)))))
    return path

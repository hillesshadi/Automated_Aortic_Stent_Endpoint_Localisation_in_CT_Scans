#!/usr/bin/env python3
"""
Evaluation suite: image-enhancement quality, segmentation agreement, and
morphometry, in the tabular form the GUI renders and exports.

The mathematics lives in ct_viewer/metrics.py (MSE / PSNR / SSIM / FSIM, Dice,
IoU, Hausdorff). This module decides *what to measure on*, which is the part
that is specific to this pipeline:

  * Enhancement metrics are computed per-slice on a spread of slices rather
    than on one, and reported as mean +/- sd. A single mid-volume slice is not
    representative -- noise and contrast vary strongly from the aortic root to
    the iliacs.

  * Dice and IoU need two masks. This pipeline has no ground truth of its own,
    so they are reported only against an explicit reference: either a
    segmentation the user loads, or the previous run's mask held on the
    context (which measures threshold stability, not accuracy). When neither
    exists the report says so instead of returning a number that would read as
    an accuracy figure.
"""

import csv
import json
import os
from datetime import datetime

import numpy as np

from . import paths  # noqa: F401
from .context import CaseContext

HU_RANGE = (-1024.0, 3071.0)
HU_BINS = 64


class MetricsError(RuntimeError):
    pass


# Which pip package provides which import name, so a failure names the thing
# that is actually missing rather than a guess.
_PACKAGE_FOR = {
    "skimage": "scikit-image",
    "phasepack": "phasepack",
    "scipy": "scipy",
    "cv2": "opencv-python",
}


def _metrics_mod():
    try:
        import metrics as m
        return m
    except ImportError as exc:                                 # noqa: BLE001
        missing = getattr(exc, "name", None) or ""
        package = _PACKAGE_FOR.get(missing, missing or "the missing module")
        raise MetricsError(
            "ct_viewer/metrics.py is not importable: %s\n\n"
            "Install the missing dependency:\n    pip install %s"
            % (exc, package)) from exc


# --------------------------------------------------------------------------- #
#  Image enhancement
# --------------------------------------------------------------------------- #
def enhancement_report(ctx: CaseContext, filter_name, filter_params, wc, ww,
                       n_slices=9, progress=None, should_stop=None):
    """MSE / PSNR / SSIM (and FSIM) of `filter_name` against the raw slices.

    Both images are windowed to uint8 first, because the filters in
    enhancement.py are uint8-only and PSNR is only meaningful against a stated
    data range -- here 255 by construction.
    """
    from .preprocess import apply_filter_2d, window_to_u8

    if not ctx.has_volume:
        raise MetricsError("Load a volume first.")
    if not filter_name or filter_name == "None":
        raise MetricsError(
            "No filter selected.\n\nChoose a filter in the Filtering / "
            "enhancement group -- enhancement metrics compare a filtered "
            "image against the raw one, so there is nothing to compare yet.")

    mm = _metrics_mod()
    nz = ctx.volume.shape[0]
    # Interior slices only: the first and last few of a CT series are often
    # padding or partial anatomy and would skew the mean.
    lo, hi = int(0.15 * nz), int(0.85 * nz)
    idx = np.unique(np.linspace(lo, max(lo + 1, hi - 1),
                                min(n_slices, max(1, hi - lo))).astype(int))

    rows = []
    for k, z in enumerate(idx):
        if should_stop and should_stop():
            raise MetricsError("Cancelled.")
        raw_u8 = window_to_u8(ctx.volume[int(z)], wc, ww)
        out_u8 = apply_filter_2d(raw_u8, filter_name, filter_params)
        vals = mm.enhancement_metrics(raw_u8, out_u8, data_range=255.0)
        vals["slice"] = int(z)
        rows.append(vals)
        if progress:
            progress(int(100.0 * (k + 1) / len(idx)),
                     "Slice %d/%d (z=%d)" % (k + 1, len(idx), int(z)))

    keys = [k for k in ("MSE", "PSNR", "SSIM", "FSIM") if k in rows[0]]
    summary = {}
    for key in keys:
        series = np.asarray([r[key] for r in rows], float)
        series = series[np.isfinite(series)]
        if series.size:
            summary[key] = {"mean": float(series.mean()),
                            "sd": float(series.std()),
                            "min": float(series.min()),
                            "max": float(series.max())}

    report = {
        "kind": "enhancement",
        "filter": filter_name,
        "filter_params": dict(filter_params or {}),
        "window": {"wc": int(wc), "ww": int(ww)},
        "n_slices": len(rows),
        "per_slice": rows,
        "summary": summary,
    }
    ctx.metrics["enhancement"] = report
    return report


# --------------------------------------------------------------------------- #
#  Segmentation and morphometry
# --------------------------------------------------------------------------- #
def _mask_extent(m):
    """(dz, dy, dx) voxel span of a mask, without materialising its indices.

    np.nonzero on a 254-million-voxel mask allocates three int64 arrays of that
    length -- about 6 GB, and minutes of work. Reducing with any() along the
    other two axes gives the same answer in a single pass and constant memory.
    """
    spans = []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        present = np.any(m, axis=others)
        idx = np.flatnonzero(present)
        spans.append(0 if idx.size == 0 else int(idx[-1] - idx[0] + 1))
    return spans


def morphometry(mask, volume, spacing_zyx, whole=False):
    """Voxel count, physical volume, bounding box and HU statistics of a mask.

    `whole=True` marks the degenerate all-true mask. Indexing a volume with it
    copies the entire array; on a 1167x467x467 series that is a needless half
    a gigabyte, so the statistics are taken from the volume directly.
    """
    dz, dy, dx = spacing_zyx
    if whole:
        n = int(volume.size)
        vals = volume
        extent = list(volume.shape)
    else:
        m = np.asarray(mask, bool)
        n = int(m.sum())
        vals = volume[m] if n else None
        extent = _mask_extent(m) if n else None

    out = {
        "voxels": n,
        "volume_mm3": n * dz * dy * dx,
        "volume_ml": n * dz * dy * dx / 1000.0,
    }
    if not n:
        out.update({"hu_mean": float("nan"), "hu_sd": float("nan"),
                    "hu_min": float("nan"), "hu_max": float("nan"),
                    "bbox_mm": None})
        return out

    out["bbox_mm"] = [float(extent[0] * dz), float(extent[1] * dy),
                      float(extent[2] * dx)]
    # float64 accumulators on an int16 source, without a float64 *copy* of it.
    out.update({"hu_mean": float(np.mean(vals, dtype=np.float64)),
                "hu_sd": float(np.std(vals, dtype=np.float64)),
                "hu_min": float(vals.min()), "hu_max": float(vals.max())})
    return out


def hu_histogram(volume, mask=None, bins=HU_BINS, hu_range=HU_RANGE,
                 max_samples=40000000):
    """Counts per HU bin, over the whole volume or inside a mask.

    The source is left in its native dtype -- converting a 254-million-voxel
    int16 volume to float64 first allocates 2 GB and dominated the runtime.
    Beyond `max_samples` the data is strided rather than copied; regular
    subsampling does not change the shape of a histogram, and the stride is
    recorded so the counts stay interpretable.
    """
    if mask is not None:
        data = volume[np.asarray(mask, bool)]
    else:
        data = np.ravel(volume)

    stride = 1
    if data.size > max_samples:
        stride = int(np.ceil(data.size / float(max_samples)))
        data = data[::stride]
    if data.size == 0:
        edges = np.linspace(hu_range[0], hu_range[1], bins + 1)
        return {"counts": [0] * bins, "edges": edges.tolist(), "n": 0}
    counts, edges = np.histogram(data, bins=bins, range=hu_range)
    out = {"counts": counts.astype(int).tolist(),
           "edges": edges.tolist(),
           "n": int(data.size),
           "clipped_low": int((data < hu_range[0]).sum()),
           "clipped_high": int((data > hu_range[1]).sum())}
    if stride > 1:
        out["subsampled_every"] = stride
        out["represents"] = int(data.size * stride)
    return out


def segmentation_report(ctx: CaseContext, reference=None, progress=None,
                        should_stop=None):
    """Morphometry + HU histograms for aorta, stent and whole volume.

    `reference` is an optional {"aorta": mask, "stent": mask} of ground-truth
    or comparison masks; ctx.reference_mask is picked up automatically when
    no explicit one is passed. Dice/IoU/Hausdorff appear only for structures a
    reference supplies, and `agreement_source` records what the comparison was
    against so a threshold-stability number is never mistaken for an accuracy
    one.

    Self-comparison is refused rather than reported. When the working aorta
    mask *is* the reference (ctx.aorta_mask_source == "reference", which is
    what happens when --reference is supplied and preprocessing is skipped),
    Dice and IoU are 1.0 and Hausdorff is 0 by construction. Printing those as
    an accuracy result would be actively misleading, so they are omitted and
    the note says why. To get a real accuracy figure, segment the volume as
    well -- run_headless.py --resegment, or Recompute in the GUI.
    """
    if not ctx.has_volume:
        raise MetricsError("Load a volume first.")

    # A supplied reference is a perfectly good segmentation to measure; not
    # having run the seed search is no reason to refuse to report on it.
    if ctx.aorta_mask is None and ctx.reference_mask is not None:
        ctx.use_reference_as_aorta()

    if ctx.aorta_mask is None and ctx.stent_mask is None:
        raise MetricsError(
            "No segmentation yet.\n\nRun Preprocessing (or Recompute "
            "Segmentation) first, or supply one with --reference.")

    spacing = ctx.spacing_zyx
    structures = {}
    # No all-true array: morphometry(whole=True) reads the volume
    # directly, so allocating a 254 MB boolean here would be pure waste.
    targets = [("aorta", ctx.aorta_mask), ("stent", ctx.stent_mask),
               ("whole_volume", None)]

    for k, (name, mask) in enumerate(targets):
        if should_stop and should_stop():
            raise MetricsError("Cancelled.")
        whole = (name == "whole_volume")
        if mask is None and not whole:
            continue
        entry = morphometry(mask, ctx.volume, spacing, whole=whole)
        entry["histogram"] = hu_histogram(ctx.volume,
                                          None if whole else mask)
        structures[name] = entry
        if progress:
            progress(int(70.0 * (k + 1) / len(targets)), "Measured %s" % name)

    if reference is None and ctx.reference_mask is not None:
        reference = {"aorta": ctx.reference_mask,
                     "source": ctx.reference_name or "reference mask"}

    agreement, source, skipped = {}, "none", None
    if reference:
        mm = _metrics_mod()
        for name in ("aorta", "stent"):
            ref = reference.get(name)
            cur = getattr(ctx, "%s_mask" % name, None)
            if ref is None or cur is None:
                continue
            ref = np.asarray(ref, bool)
            if ref.shape != cur.shape:
                raise MetricsError(
                    "Reference %s mask is %s but the case is %s -- they must "
                    "come from the same volume." % (name, ref.shape, cur.shape))
            if name == "aorta" and ctx.aorta_mask_source == "reference":
                # Same array on both sides: DSC 1.0, IoU 1.0, HD 0. Reporting
                # that as accuracy would be a fabricated result.
                skipped = ("the working aorta mask IS the reference, so "
                           "Dice/IoU/Hausdorff would be 1.0/1.0/0 by "
                           "construction; re-run with --resegment (or "
                           "Recompute in the GUI) to compare an independent "
                           "segmentation against it")
                continue
            agreement[name] = mm.segmentation_metrics(cur, ref, spacing)
        source = reference.get("source", "reference masks")
    if progress:
        progress(100, "Segmentation metrics done.")

    report = {
        "kind": "segmentation",
        "case_id": ctx.case_id,
        "spacing_zyx": [float(v) for v in spacing],
        "structures": structures,
        "agreement": agreement,
        "agreement_source": source,
        "aorta_mask_source": ctx.aorta_mask_source or "segmented",
        "agreement_note": (
            "Dice/IoU/Hausdorff compare against %s." % source if agreement
            else "Dice/IoU/Hausdorff not reported: %s." % skipped if skipped
            else "No reference mask supplied, so Dice/IoU/Hausdorff are not "
                 "reported -- they need a second segmentation to compare "
                 "against."),
    }
    ctx.metrics["segmentation"] = report
    return report


def snapshot_masks(ctx: CaseContext, label="previous run"):
    """Copy the current masks so a later run can be compared against them."""
    out = {"source": label}
    if ctx.aorta_mask is not None:
        out["aorta"] = ctx.aorta_mask.copy()
    if ctx.stent_mask is not None:
        out["stent"] = ctx.stent_mask.copy()
    return out if len(out) > 1 else None


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #
def _flatten(report):
    """Report -> flat (section, metric, value) rows for the CSV."""
    rows = []
    if report.get("kind") == "enhancement":
        for key, stats in report.get("summary", {}).items():
            for stat, value in stats.items():
                rows.append(("enhancement/%s" % report.get("filter", "?"),
                             "%s_%s" % (key, stat), value))
        for r in report.get("per_slice", []):
            for key in ("MSE", "PSNR", "SSIM", "FSIM"):
                if key in r:
                    rows.append(("enhancement/slice_%d" % r["slice"], key,
                                 r[key]))
    else:
        for name, entry in report.get("structures", {}).items():
            for key, value in entry.items():
                if key == "histogram":
                    continue
                if isinstance(value, list):
                    value = " x ".join("%.2f" % v for v in value)
                rows.append((name, key, value))
        for name, entry in report.get("agreement", {}).items():
            for key, value in entry.items():
                rows.append(("%s/agreement" % name, key, value))
    return rows


def export_metrics(reports, out_dir, case_id, progress=None):
    """Write every report to <out_dir> as one JSON plus flat CSV tables."""
    reports = [r for r in reports if r]
    if not reports:
        raise MetricsError("Nothing to export -- compute some metrics first.")
    os.makedirs(out_dir, exist_ok=True)
    stem = case_id or "case"
    written = []

    payload = {
        "case_id": case_id,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "reports": reports,
    }
    jpath = os.path.join(out_dir, "%s_metrics.json" % stem)
    with open(jpath, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_safe)
    written.append(jpath)

    cpath = os.path.join(out_dir, "%s_metrics.csv" % stem)
    with open(cpath, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["case_id", "report", "section", "metric", "value"])
        for report in reports:
            for section, metric, value in _flatten(report):
                w.writerow([case_id, report.get("kind", "?"), section, metric,
                            value])
    written.append(cpath)

    # Histograms are long and rectangular, so they get their own table rather
    # than being folded into the key/value CSV above.
    hist_rows = []
    for report in reports:
        for name, entry in report.get("structures", {}).items():
            h = entry.get("histogram")
            if not h:
                continue
            edges, counts = h["edges"], h["counts"]
            for i, count in enumerate(counts):
                hist_rows.append([case_id, name, edges[i], edges[i + 1], count])
    if hist_rows:
        hpath = os.path.join(out_dir, "%s_hu_histograms.csv" % stem)
        with open(hpath, "w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["case_id", "structure", "hu_low", "hu_high", "count"])
            w.writerows(hist_rows)
        written.append(hpath)

    if progress:
        progress(100, "Wrote %d metric file(s)." % len(written))
    return written


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

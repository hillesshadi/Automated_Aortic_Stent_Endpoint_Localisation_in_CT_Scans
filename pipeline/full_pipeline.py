#!/usr/bin/env python3
"""
End-to-end automated pipeline: raw DICOM/NIfTI -> segmentation -> centerline
-> surface meshes -> stent endpoints -> structured JSON.

This sequences the same stage functions the GUI's individual "Run
Preprocessing" / "Extract Centerline" / "Fit Stent" / "Export Results" buttons
call (preprocess.run_preprocessing, centerline.extract_centerline,
stent_fit.fit_stent, stent_specs.landmark_table, stent_fit.export_results) and
adds surface-mesh generation from surfaces.py, so a raw volume can be taken
straight to every derived artifact in one call instead of four manual steps.

Progress is reported as a single continuous 0-100 sweep divided into four
bands, one per stage, so the caller can drive one progress bar across the
whole run rather than resetting it between stages. Each stage transition is
reported through a message beginning "Step N/4: " so a caller logging those
messages gets a readable record of which stage is running.
"""

import os

from . import paths  # noqa: F401
from . import centerline as cl_stage
from . import preprocess
from . import stent_fit
from . import stent_specs
from . import surfaces
from .context import CaseContext

#: (label, band_start_pct, band_end_pct)
_STAGES = (
    ("Segmenting the aorta and stent metal", 0, 25),
    ("Extracting the centerline", 25, 50),
    ("Generating surface meshes", 50, 75),
    ("Detecting stent endpoints and writing metadata", 75, 100),
)


class FullPipelineError(RuntimeError):
    pass


def run_full_pipeline(ctx: CaseContext, params, out_dir, specs=None,
                      mesh_formats=("stl", "vtk"), target_triangles=150000,
                      tube_radius_mm=1.2, progress=None, should_stop=None):
    """Segmentation -> centerline -> meshes -> stent JSON, from a raw volume.

    `ctx` must already have a volume loaded (io_data.load_dicom_series /
    load_image_file). Everything after that -- the aorta and stent masks, the
    centerline, the meshes and the stent-endpoint JSON -- is produced here and
    written under `out_dir`. Returns a dict describing what was produced, for
    the caller to load into its own views without re-reading the files back
    from disk.
    """
    if not ctx.has_volume:
        raise FullPipelineError(
            "Load a DICOM series or a volume file before running the full "
            "pipeline.")

    os.makedirs(out_dir, exist_ok=True)
    case = ctx.case_id or "case"

    def check():
        if should_stop and should_stop():
            raise FullPipelineError("Cancelled.")

    def band(index):
        label, lo, hi = _STAGES[index]
        if progress:
            progress(lo, "Step %d/4: %s…" % (index + 1, label))

        def report(pct, msg):
            if progress:
                progress(lo + int((hi - lo) * max(0, min(100, pct)) / 100.0),
                         msg)
            check()
        return report

    written = []

    # ---- 1/4: segmentation ------------------------------------------------ #
    report = band(0)
    preprocess.run_preprocessing(ctx, params, progress=report,
                                 should_stop=should_stop)
    check()

    # ---- 2/4: centerline --------------------------------------------------- #
    report = band(1)
    if ctx.aorta_mask is None or not ctx.aorta_mask.any():
        raise FullPipelineError(
            "Segmentation produced no aorta mask; the centerline stage has "
            "nothing to skeletonise. Adjust the Aorta HU tolerance and try "
            "again.")
    cl_stage.extract_centerline(ctx, step_mm=params.get("cl_step", 1.0),
                                orient=params.get("cl_orient", True),
                                progress=report)
    check()

    # ---- 3/4: surface meshes ------------------------------------------------ #
    report = band(2)
    formats = [f.strip().lower().lstrip(".") for f in mesh_formats if f.strip()]
    mask_names = [(n, m) for n, m in
                  (("aorta", ctx.aorta_mask), ("stent", ctx.stent_mask))
                  if m is not None and m.any()]
    n_parts = len(mask_names) + (1 if ctx.resampler is not None else 0) or 1
    done = 0

    for name, mask in mask_names:
        poly = surfaces.build_surface(mask, ctx.geom, target_triangles)
        done += 1
        if poly is None:
            report(int(100 * done / n_parts), "%s mask produced no surface" % name)
            continue
        for ext in formats:
            path = os.path.join(out_dir, "%s_%s.%s" % (case, name, ext))
            written.append(surfaces.write_mesh(poly, path))
        report(int(100 * done / n_parts),
              "%s surface: %d points" % (name, poly.GetNumberOfPoints()))

    if ctx.resampler is not None:
        cl_poly = surfaces.centerline_polydata(ctx.resampler.points)
        tube = surfaces.tube_polydata(ctx.resampler.points,
                                      radius_mm=tube_radius_mm)
        for ext in formats:
            path = os.path.join(out_dir, "%s_centerline.%s" % (case, ext))
            poly = cl_poly if ext in ("vtk", "vtp") else tube
            if poly is not None:
                written.append(surfaces.write_mesh(poly, path))
        done += 1
        report(int(100 * done / n_parts), "centerline written")
    check()

    # ---- 4/4: stent endpoints + structured JSON ---------------------------- #
    report = band(3)
    stent_result = None
    if ctx.stent_mask is not None and ctx.stent_mask.any():
        stent_result = stent_fit.fit_stent(
            ctx, progress=lambda p, m: report(int(0.4 * p), m))
    else:
        ctx.note("no stent metal found above the HU threshold; stent "
                 "endpoint detection skipped")
    check()

    landmarks = None
    if ctx.resampler is not None:
        try:
            landmarks = stent_specs.landmark_table(
                ctx, specs, lumen_hu=params.get("lumen_hu", 200))
        except stent_specs.SpecError as exc:
            ctx.note("landmark table skipped: %s" % exc)
        if landmarks is not None:
            written.extend(stent_specs.export_landmarks(landmarks, out_dir, case))
    report(70, "landmarks ready" if landmarks else "no landmarks")

    written.extend(stent_fit.export_results(
        ctx, out_dir, progress=lambda p, m: report(70 + int(0.3 * p), m)))

    if progress:
        progress(100, "Step 4/4: pipeline complete — %d file(s) written."
                 % len(written))

    return {
        "written": written,
        "out_dir": out_dir,
        "case_id": case,
        "stent_result": stent_result,
        "landmarks": landmarks,
    }

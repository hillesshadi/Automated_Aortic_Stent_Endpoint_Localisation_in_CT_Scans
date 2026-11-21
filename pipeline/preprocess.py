#!/usr/bin/env python3
"""
Preprocessing stage: optional EADTV/denoise filtering, then aorta and stent
segmentation by Hounsfield-unit thresholding.

Nothing here is new mathematics. The filters are ct_viewer/enhancement.py's
existing registry (including its EADTV implementation, the edge-adaptive dual
total-variation denoiser), and the segmentation is ct_viewer/segmentation.py's
existing seed-and-threshold aorta search plus its metal threshold. This module
only sequences them, converts between the 2-D slice API the filters expose and
the 3-D volume the GUI holds, and reports progress.
"""

import numpy as np

from . import paths  # noqa: F401
from .context import CaseContext


class PreprocessError(RuntimeError):
    pass


# A thoracoabdominal aorta holds roughly 150-400 mL. These bounds are
# deliberately loose -- they are not a measurement, they are a leak detector.
# An HU band grown from a seed that landed in soft tissue connects the whole
# torso and comes out at thousands of millilitres; that mask then produces a
# nonsense stent length and a surface with no anatomical meaning, so it must be
# rejected rather than returned.
MIN_AORTA_ML = 8.0
MAX_AORTA_ML = 900.0
MAX_AORTA_FRACTION = 0.06        # of the whole scan


def available_filters():
    """(names, params) from the enhancement registry, for building controls.

    Returning the registry itself rather than a hard-coded list means a filter
    added to enhancement.py shows up in the GUI without touching this file.
    """
    try:
        import enhancement as enh
        return sorted(enh.FILTERS), dict(enh.FILTER_PARAMS)
    except ImportError:
        return [], {}


def apply_filter_2d(slice_u8, filter_name, params=None):
    """Run one registered 2-D filter on a uint8 slice."""
    if not filter_name or filter_name == "None":
        return slice_u8
    import enhancement as enh
    fn = enh.FILTERS.get(filter_name)
    if fn is None:
        raise PreprocessError("Unknown filter: %s" % filter_name)
    return fn(slice_u8, **(params or {}))


def window_to_u8(hu, wc, ww):
    """HU -> uint8 for display and for the 2-D filters, which are uint8-only."""
    lo, hi = wc - ww / 2.0, wc + ww / 2.0
    if hi <= lo:
        hi = lo + 1.0
    out = (np.asarray(hu, np.float32) - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)


def filter_volume(volume, filter_name, params, wc, ww, progress=None,
                  should_stop=None):
    """Apply a 2-D filter slice-by-slice across the whole volume.

    The enhancement filters are deliberately 2-D and uint8 (see that module's
    docstring: a flat memory footprint while scrubbing). Running one over a
    volume therefore means windowing to uint8 per slice, which is lossy with
    respect to the original HU -- so the filtered result is returned as a
    separate uint8 array for display and measurement, and the caller keeps the
    original int16 volume for anything quantitative.
    """
    if not filter_name or filter_name == "None":
        return None
    nz = volume.shape[0]
    out = np.empty((nz,) + volume.shape[1:], np.uint8)
    for z in range(nz):
        if should_stop and should_stop():
            raise PreprocessError("Cancelled.")
        out[z] = apply_filter_2d(window_to_u8(volume[z], wc, ww),
                                 filter_name, params)
        if progress and (z % 16 == 0 or z == nz - 1):
            progress(int(100 * (z + 1) / nz), "Filtering slice %d/%d" % (z + 1, nz))
    return out


def _bbox_around(shape, coords, margin_vox, full_z=True):
    """Tight (z0,z1,y0,y1,x0,x1) around voxel coords, padded by `margin_vox`.

    Restricting the threshold and connected-component arrays to this box is
    what keeps memory sane: a 1167x467x467 volume labelled in full is a ~1 GB
    int32 allocation, and seeded growth only ever needs the neighbourhood the
    vessel actually occupies.
    """
    nz, ny, nx = shape
    c = np.asarray(coords, int).reshape(-1, 3)          # (z, y, x)
    if full_z:
        z0, z1 = 0, nz
    else:
        z0 = max(0, int(c[:, 0].min()) - margin_vox)
        z1 = min(nz, int(c[:, 0].max()) + margin_vox + 1)
    y0 = max(0, int(c[:, 1].min()) - margin_vox)
    y1 = min(ny, int(c[:, 1].max()) + margin_vox + 1)
    x0 = max(0, int(c[:, 2].min()) - margin_vox)
    x1 = min(nx, int(c[:, 2].max()) + margin_vox + 1)
    return (z0, z1, y0, y1, x0, x1)


def grow_from_seeds(volume, seeds, tolerance=120.0, margin_vox=40,
                    full_z=False, hu_range=None):
    """Region-grow an HU band around explicit seed voxels.

    `seeds` are (z, y, x) voxel triples. The band is centred on the *median*
    seed intensity rather than one seed's value, so a seed that happens to
    land on a calcification or the vessel wall cannot drag the window off the
    blood pool.

    Returns (sub_mask, bbox) with sub_mask cropped to bbox, matching the
    convention of segmentation.auto_segment_aorta.
    """
    from scipy.ndimage import label
    import segmentation as seg

    seeds = np.asarray(seeds, int).reshape(-1, 3)
    nz, ny, nx = volume.shape
    seeds = seeds[(seeds[:, 0] >= 0) & (seeds[:, 0] < nz)
                  & (seeds[:, 1] >= 0) & (seeds[:, 1] < ny)
                  & (seeds[:, 2] >= 0) & (seeds[:, 2] < nx)]
    if len(seeds) == 0:
        return np.zeros((0, 0, 0), np.uint8), (0, 0, 0, 0, 0, 0)

    bbox = _bbox_around(volume.shape, seeds, margin_vox, full_z=full_z)
    z0, z1, y0, y1, x0, x1 = bbox
    sub = volume[z0:z1, y0:y1, x0:x1]

    if hu_range is not None:
        # An explicit band from the operator. Used verbatim: the point of
        # setting it by hand is that the seed's own intensity should not move
        # it, which is what happens on a series whose contrast phase puts the
        # lumen outside the usual window.
        lo, hi = float(hu_range[0]), float(hu_range[1])
    else:
        hu = np.array([float(volume[z, y, x]) for z, y, x in seeds])
        centre = float(np.median(hu))
        lo, hi = centre - tolerance, centre + tolerance

    candidate = (sub >= lo) & (sub <= hi)
    if not candidate.any():
        return np.zeros(sub.shape, np.uint8), bbox

    labeled, _num = label(candidate)
    keep = set()
    for z, y, x in seeds:
        lab = int(labeled[z - z0, y - y0, x - x0])
        if lab:
            keep.add(lab)
    if not keep:
        return np.zeros(sub.shape, np.uint8), bbox

    mask = np.isin(labeled, list(keep)).astype(np.uint8)
    return seg.postprocess_aorta_mask(mask), bbox


def _centerline_seeds(ctx: CaseContext, n=9):
    """Voxel seeds sampled along the centerline, if one is loaded.

    The centerline runs down the middle of the lumen by construction, so its
    points are the most reliable seeds available -- far better than any
    landmark search. Several are taken along its length so one point sitting
    in a calcified or stented segment cannot decide the HU band alone.
    """
    points = None
    if ctx.resampler is not None and len(ctx.resampler) >= 2:
        points = np.asarray(ctx.resampler.points, float)
    elif ctx.centerline is not None and len(ctx.centerline) >= 2:
        points = np.asarray(ctx.centerline, float)
    if points is None:
        return None

    from stent_endpoint_sop import world_to_idx
    nz, ny, nx = ctx.volume.shape
    idx = np.unique(np.linspace(0, len(points) - 1,
                                min(n, len(points))).astype(int))
    seeds = []
    for i in idx:
        try:
            ix, iy, iz = world_to_idx(points[i], ctx.geom)
        except Exception:                                      # noqa: BLE001
            continue
        z, y, x = int(round(iz)), int(round(iy)), int(round(ix))
        if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx:
            seeds.append((z, y, x))
    return seeds or None


def _locate_seed_wide(volume, hu_window=(-100.0, 600.0), z_fraction=(0.2, 0.8),
                      n_slices=11, min_area=40):
    """Seed search over the whole scan length with a widened HU window.

    segmentation.locate_aorta_seed samples five slices within +/-15 of the
    volume centre and accepts only -50..400 HU. On a long scan -- a 700 mm
    whole-body series is ~1167 slices at 0.6 mm -- those five slices cover
    18 mm, and if the aorta is not clearly round there, or the contrast phase
    puts the lumen outside that window, it returns None. That is the failure
    this tier exists to recover from, and it is why simply widening
    `tolerance` does not help: tolerance is applied only after a seed exists.
    """
    from scipy.ndimage import label, center_of_mass
    import segmentation as seg

    nz, ny, nx = volume.shape
    lo, hi = hu_window
    z_samples = np.unique(np.linspace(int(z_fraction[0] * nz),
                                      int(z_fraction[1] * nz),
                                      n_slices).astype(int))
    best = None
    for z in z_samples:
        z = int(np.clip(z, 0, nz - 1))
        spine_row, spine_col = seg.detect_spine(volume[max(0, z - 5):z + 6])
        sl = volume[z]
        # Anterior of the spine, and to *either* side of it: which side the
        # aorta appears on depends on patient orientation as stored, and a
        # wrong guess is exactly what makes the narrow search fail.
        r0 = max(0, spine_row - int(0.30 * ny))
        r1 = min(ny, spine_row + int(0.05 * ny))
        c0 = max(0, spine_col - int(0.25 * nx))
        c1 = min(nx, spine_col + int(0.25 * nx))
        if r1 - r0 < 5 or c1 - c0 < 5:
            continue
        roi = sl[r0:r1, c0:c1]
        candidate = (roi >= lo) & (roi <= hi)
        if not candidate.any():
            continue
        labeled, num = label(candidate)
        if num == 0:
            continue
        for comp_id in range(1, num + 1):
            comp = labeled == comp_id
            area = int(comp.sum())
            if area < min_area or area > 0.25 * comp.size:
                continue
            coords = np.argwhere(comp)
            spread = coords.std(axis=0)
            roundness = float(spread.min() / (spread.max() + 1e-6))
            if roundness < 0.45:            # aortic cross-sections are round
                continue
            score = area * roundness
            if best is None or score > best[0]:
                rr, cc = center_of_mass(comp)
                best = (score, z, int(round(rr)) + r0, int(round(cc)) + c0)
    if best is None:
        return None
    return (best[1], best[2], best[3])


def _implausible(mask_voxels, ctx):
    """Reason the candidate mask cannot be an aorta, or None if it may be.

    Checked after every growth attempt. Without this the first tier that
    produces *any* mask wins, and a leak into soft tissue is indistinguishable
    from success -- which is exactly how a 237 mL aorta came back as 2598 mL.
    """
    dz, dy, dx = ctx.spacing_zyx
    ml = mask_voxels * dz * dy * dx / 1000.0
    total = float(ctx.volume.size)
    if mask_voxels == 0:
        return "empty"
    if ml < MIN_AORTA_ML:
        return "only %.1f mL (below %.0f mL)" % (ml, MIN_AORTA_ML)
    if ml > MAX_AORTA_ML:
        return "%.0f mL (above %.0f mL -- leaked into surrounding tissue)" % (
            ml, MAX_AORTA_ML)
    if mask_voxels / total > MAX_AORTA_FRACTION:
        return "%.1f%% of the whole scan (leaked)" % (100.0 * mask_voxels / total)
    return None


def segment_aorta(ctx: CaseContext, tolerance=120, progress=None,
                  allow_fallback=True, hu_range=None):
    """Seed-and-threshold aorta segmentation; fills ctx.aorta_mask.

    Three tiers, tried in order, because the single-shot landmark search fails
    on perfectly good data -- long scans, an unusual contrast phase, or a
    patient orientation the search does not expect:

      1. the standard spine-relative search at the requested tolerance;
      2. a widened seed search across the whole scan length, then wider HU
         tolerances;
      3. seeds taken straight from the centerline, when one is loaded.

    On tier 2: re-running tier 1 with a larger `tolerance` alone does nothing
    when the *seed* was not found, because tolerance is applied only after
    seed detection to size the growth band. So tier 2 re-runs the seed search
    itself with a widened HU window and slices spread over the whole volume,
    and only then tries wider tolerances.
    """
    import segmentation as seg
    if not ctx.has_volume:
        raise PreprocessError("Load a volume before segmenting.")

    attempts = []
    dz, dy, dx = ctx.spacing_zyx
    voxel_ml = dz * dy * dx / 1000.0

    def _accept(sub, bbox, seed, how):
        """Adopt a candidate mask if it is anatomically plausible."""
        n = int(np.asarray(sub).sum())
        bad = _implausible(n, ctx)
        if bad is not None:
            attempts.append("%s: %s" % (how, bad))
            return None
        full = np.zeros(ctx.volume.shape, bool)
        z0, z1, y0, y1, x0, x1 = bbox
        full[z0:z1, y0:y1, x0:x1] = np.asarray(sub).astype(bool)
        ctx.aorta_mask, ctx.aorta_seed = full, seed
        ctx.aorta_mask_source = "segmented"
        ctx.note("aorta segmented via %s (%d vox, %.1f mL)"
                 % (how, n, n * voxel_ml))
        if progress:
            progress(100, "Aorta: %d voxels (%.1f mL) via %s"
                     % (n, n * voxel_ml, how))
        return full

    # ---- explicit HU band ------------------------------------------------ #
    # When the operator has set a band by hand, that band is the answer: run
    # it once against the best seeds available and report honestly if it
    # yields nothing, rather than quietly sweeping other tolerances and
    # returning a mask the stated thresholds did not produce.
    if hu_range is not None:
        lo, hi = float(hu_range[0]), float(hu_range[1])
        seeds = _centerline_seeds(ctx)
        if seeds is None:
            wide = _locate_seed_wide(ctx.volume, hu_window=(lo, hi))
            seeds = [wide] if wide else None
        if seeds:
            if progress:
                progress(40, "Growing the explicit band %d-%d HU..." % (lo, hi))
            sub, bbox = grow_from_seeds(ctx.volume, seeds, margin_vox=40,
                                        full_z=False, hu_range=(lo, hi))
            got = _accept(sub, bbox, tuple(seeds[len(seeds) // 2]),
                          "explicit band %d-%d HU" % (lo, hi))
            if got is not None:
                return got
        else:
            attempts.append("explicit band %d-%d HU: no seed to grow from"
                            % (lo, hi))
        ctx.aorta_mask, ctx.aorta_seed = None, None
        raise PreprocessError(
            "The explicit HU band %d-%d produced no plausible aorta:\n\n%s\n\n"
            "Widen the band, or clear the explicit-band checkbox to let the "
            "automatic search choose one."
            % (lo, hi, "\n".join("  - " + a for a in attempts)))

    # ---- tier 0: centerline seeds ---------------------------------------- #
    # Tried first when a centerline is loaded. It runs down the middle of the
    # lumen by construction, which makes it a far better seed than any
    # landmark search -- and its points bound the ROI, so a leak cannot spread
    # across the whole torso the way a full-Z growth can.
    seeds = _centerline_seeds(ctx)
    if seeds:
        if progress:
            progress(15, "Growing from %d centerline seeds..." % len(seeds))
        for tol in (tolerance, 200, 300, 400):
            sub, bbox = grow_from_seeds(ctx.volume, seeds, tolerance=tol,
                                        margin_vox=40, full_z=False)
            got = _accept(sub, bbox, tuple(seeds[len(seeds) // 2]),
                          "centerline seeds, %d HU" % tol)
            if got is not None:
                return got

    # ---- tier 1: the standard landmark search ---------------------------- #
    if progress:
        progress(35, "Locating spine and aortic seed...")
    sub, bbox, seed = seg.auto_segment_aorta(ctx.volume, tolerance=int(tolerance))
    if seed is not None:
        got = _accept(sub, bbox, seed, "landmark search, %d HU" % tolerance)
        if got is not None:
            return got
    else:
        attempts.append("landmark search at %d HU: no seed found" % tolerance)

    if not allow_fallback:
        ctx.aorta_mask, ctx.aorta_seed = None, None
        raise PreprocessError(
            "No plausible aorta segmentation and fallbacks are disabled.\n\n"
            + "\n".join("  - " + a for a in attempts))

    # ---- tier 2: wider tolerances on the landmark seed ------------------- #
    if progress:
        progress(55, "Retrying with wider HU tolerances...")
    for tol in (200, 300, 400):
        sub, bbox, seed = seg.auto_segment_aorta(ctx.volume, tolerance=tol)
        if seed is None:
            attempts.append("landmark search at %d HU: no seed found" % tol)
            continue
        got = _accept(sub, bbox, seed, "landmark search, %d HU" % tol)
        if got is not None:
            return got

    # ---- tier 3: widened seed search over the whole scan ----------------- #
    # Last resort, and the most leak-prone: it grows over the full Z range, so
    # every candidate here is volume-checked before it is accepted.
    if progress:
        progress(75, "Retrying the seed search over the whole scan length...")
    wide_seed = _locate_seed_wide(ctx.volume)
    if wide_seed is not None:
        for tol in (80, tolerance, 200):
            sub, bbox = grow_from_seeds(ctx.volume, [wide_seed], tolerance=tol,
                                        margin_vox=60, full_z=True)
            got = _accept(sub, bbox, wide_seed,
                          "widened seed search, %d HU" % tol)
            if got is not None:
                return got
    else:
        attempts.append("widened seed search: no round vessel found")

    ctx.aorta_mask, ctx.aorta_seed = None, None
    raise PreprocessError(
        "No plausible aorta segmentation, after %d attempts:\n\n%s\n\n"
        "Every candidate was either empty or far too large to be an aorta "
        "(the bounds are %.0f-%.0f mL). Options: supply a segmentation with "
        "--reference; supply a centerline with --centerline so its points can "
        "seed the growth; or check that this series is a contrast-enhanced "
        "CTA with calibrated Hounsfield units."
        % (len(attempts), "\n".join("  - " + a for a in attempts),
           MIN_AORTA_ML, MAX_AORTA_ML))


def segment_stent(ctx: CaseContext, hu_threshold=1200, dilate=3, progress=None):
    """Threshold bright metal inside a dilated aorta ROI; fills ctx.stent_mask."""
    import segmentation as seg
    if ctx.aorta_mask is None:
        raise PreprocessError("Segment the aorta before the stent — the metal "
                              "search is restricted to a dilated aorta ROI.")
    if progress:
        progress(20, "Thresholding metal above %d HU…" % hu_threshold)

    sub, bbox = seg.segment_stent(ctx.volume, ctx.aorta_mask,
                                  hu_threshold=int(hu_threshold),
                                  dilate=int(dilate))
    if sub.size == 0 or not sub.any():
        ctx.stent_mask = None
        if progress:
            progress(100, "No metal found above %d HU" % hu_threshold)
        return None

    full = np.zeros(ctx.volume.shape, bool)
    z0, z1, y0, y1, x0, x1 = bbox
    full[z0:z1, y0:y1, x0:x1] = sub.astype(bool)
    ctx.stent_mask = full
    if progress:
        progress(100, "Stent: %d voxels" % int(full.sum()))
    return full


def initial_surface(ctx: CaseContext, hu=150.0, max_dim=192,
                    progress=None, should_stop=None):
    """Coarse isosurface mask of the raw volume, for the view right after load.

    This exists so a freshly loaded case shows something in 3-D immediately,
    before any segmentation has been run. It is deliberately a plain HU
    threshold on a decimated grid -- it is an orientation aid, not a
    measurement, and the aorta surface from run_preprocessing replaces it.

    Decimating by striding keeps voxel (0,0,0) at the same physical point, so
    the origin is unchanged and only the spacing scales; that keeps the preview
    in the same world coordinates as everything drawn beside it.
    """
    if not ctx.has_volume:
        raise PreprocessError("Load a volume first.")
    if progress:
        progress(10, "Building preview surface…")

    nz, ny, nx = ctx.volume.shape
    step = max(1, int(np.ceil(max(nz, ny, nx) / float(max_dim))))
    sub = ctx.volume[::step, ::step, ::step]
    if should_stop and should_stop():
        raise PreprocessError("Cancelled.")

    mask = sub >= float(hu)
    if progress:
        progress(70, "Preview: %d voxels above %d HU (stride %d)"
                 % (int(mask.sum()), int(hu), step))
    if not mask.any():
        return None

    origin, spacing, direction = ctx.geom
    geom = (origin, tuple(float(v) * step for v in spacing), direction)
    if progress:
        progress(100, "Preview surface ready.")
    return {"mask": mask, "geom": geom, "stride": step, "hu": float(hu)}


def run_preprocessing(ctx: CaseContext, params, progress=None, should_stop=None):
    """Full preprocessing stage, as the 'Run Preprocessing' button invokes it.

    A reference segmentation short-circuits the aorta stage: there is no point
    searching for a seed when a mask has already been supplied, and the search
    failing would then abort a run that had everything it needed. Pass
    params["resegment"]=True to segment anyway -- which is what you want when
    the reference is ground truth and you are measuring accuracy against it,
    since otherwise the two masks are the same array and Dice is 1.0 by
    construction.
    """
    if not ctx.has_volume:
        raise PreprocessError("Load a DICOM series or a volume file first.")

    def step(pct, msg):
        if progress:
            progress(pct, msg)
        if should_stop and should_stop():
            raise PreprocessError("Cancelled.")

    resegment = bool(params.get("resegment", False))
    if ctx.has_reference and not resegment:
        ctx.use_reference_as_aorta()
        step(50, "Aorta: %d voxels (from reference mask%s)"
             % (int(ctx.aorta_mask.sum()),
                " " + ctx.reference_name if ctx.reference_name else ""))
    else:
        step(5, "Segmenting aorta (HU tolerance %d)…"
             % params.get("aorta_tolerance", 120))
        segment_aorta(ctx, params.get("aorta_tolerance", 120),
                      progress=lambda p, m: step(5 + int(0.45 * p), m),
                      allow_fallback=params.get("allow_fallback", True),
                      hu_range=params.get("aorta_hu_range"))

    step(55, "Segmenting stent metal…")
    segment_stent(ctx, params.get("stent_hu", 1200), params.get("stent_dilate", 3),
                  progress=lambda p, m: step(55 + int(0.40 * p), m))

    aorta_n = int(ctx.aorta_mask.sum()) if ctx.aorta_mask is not None else 0
    stent_n = int(ctx.stent_mask.sum()) if ctx.stent_mask is not None else 0
    ctx.metrics["aorta_voxels"] = aorta_n
    ctx.metrics["stent_voxels"] = stent_n
    dz, dy, dx = ctx.spacing_zyx
    ctx.metrics["aorta_volume_ml"] = aorta_n * dz * dy * dx / 1000.0

    step(100, "Preprocessing done — aorta %d vox (%.1f mL), stent %d vox"
         % (aorta_n, ctx.metrics["aorta_volume_ml"], stent_n))
    return ctx

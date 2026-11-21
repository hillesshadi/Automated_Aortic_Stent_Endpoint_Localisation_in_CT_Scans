#!/usr/bin/env python3
"""
Local (non-HPC) driver for the Mogon aortic centerline / stent-graft endpoint
pipeline, for use from the desktop viewer (ct_viewer.py).

This is a thin wrapper, not a reimplementation: the actual geometry work is
still done by process_aorta_stents.process_one_case() and stentlib/, exactly
as it runs on Mogon (see src/run_single_case.slurm and src/env.sh). The two
differences from the HPC job are:

  * no SLURM/module/venv bootstrap -- this just calls the same Python
    functions in-process, using whatever interpreter runs ct_viewer.py.
  * output goes under <app_dir>/output/hpc_pipeline/<case_id>/ instead of the
    shared /fshpc/... paths in env.sh.

Requires the same third-party packages the HPC job needs: SimpleITK,
numpy, scipy, scikit-image, and the TotalSegmentator pip package (with its
model weights -- several GB, downloaded automatically on first run if
TOTALSEG_HOME/nnUNet_results isn't already pointed at an existing cache).
TotalSegmentator also needs a working PyTorch install; it runs on CPU if no
GPU is available, just much slower (easily 10-20 minutes per case on CPU
alone, vs. under a minute on the A100 nodes it was tuned for on Mogon).
"""
import os
import sys

import numpy as np

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class PipelineImportError(RuntimeError):
    """Raised when a required pipeline dependency isn't importable."""


def _desktop_thread_limit():
    """Leave at least 2 logical cores free for the OS and the Qt UI thread
    (never below 1). The HPC job (env.sh) pins every numeric lib to 1 thread
    because it has a dedicated GPU node and runs cases in parallel across a
    job array; here there's no GPU and only one case runs at a time, so a
    higher cap is fine, but leaving the whole machine uncapped is what makes
    a laptop stop responding for the minutes TotalSegmentator's CPU inference
    takes."""
    total = os.cpu_count() or 4
    return max(1, total - 2)


def _apply_desktop_safety_limits():
    """Cap BLAS/OpenMP/ITK thread pools before the pipeline runs. These env
    vars are read by PyTorch, SimpleITK/ITK, and nnU-Net's own preprocessing
    at startup, and are inherited by the TotalSegmentator subprocess launched
    from stentlib.io_dicom.run_totalsegmentator -- setting them here, before
    that subprocess is spawned, is enough to cover it too."""
    n = str(_desktop_thread_limit())
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
        os.environ[var] = n
    # nnU-Net's augmentation worker pool is training-only, but some versions
    # spawn it regardless unless this is set; each worker is an extra process
    # holding its own copy of the volume, which is a memory risk on a laptop.
    os.environ.setdefault("nnUNet_n_proc_DA", "0")

    # SimpleITK/ITK also cache their thread count in-process the first time a
    # filter runs, so the env var above only reaches ITK code loaded *after*
    # this point. Set it directly too, covering the resampling this same
    # process does for the canonical volume.
    try:
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(n))
    except ImportError:
        pass


def _import_pipeline():
    try:
        from process_aorta_stents import AdaptiveConfig, process_one_case
        from stent_html_export import ExportError, export_case
    except ImportError as exc:
        raise PipelineImportError(
            "Could not import the aortic centerline / stent pipeline "
            "(src/process_aorta_stents.py and friends). Missing dependency: "
            f"{exc}. This feature needs the same packages as the Mogon "
            "pipeline: SimpleITK, numpy, scipy, scikit-image, and the "
            "'TotalSegmentator' pip package. Install them in this same "
            "Python environment, e.g.:\n"
            "  pip install SimpleITK scikit-image TotalSegmentator"
        ) from exc
    return AdaptiveConfig, process_one_case, ExportError, export_case


def case_output_root(app_dir, case_id):
    return os.path.join(app_dir, "output", "hpc_pipeline", case_id)


def run_local_case(dicom_dir, app_dir, force_resegment=False, iso_spacing_mm=0.6,
                    ts_fast=True):
    """Run the full local-equivalent pipeline on one DICOM series folder.

    Returns a dict:
        rec            -- the per-case record (status, stent_length_mm, ...)
        case_id
        case_dir        -- output/hpc_pipeline/<case_id>
        html_path       -- interactive HTML, or None if no mesh was produced
        hu_volume       -- np.int16 array (z, y, x), the canonical resampled CT
        spacing_zyx     -- (dz, dy, dx) mm, isotropic
        origin_xyz      -- SimpleITK origin of the canonical volume (LPS mm)
        aorta_mask      -- bool array matching hu_volume.shape, or None

    Never raises for a pipeline failure (mirrors process_one_case: a failed
    case comes back with rec["status"] describing why, not an exception).
    Only missing dependencies or a bad dicom_dir raise.
    """
    _apply_desktop_safety_limits()
    AdaptiveConfig, process_one_case, ExportError, export_case = _import_pipeline()

    if not os.path.isdir(dicom_dir):
        raise ValueError(f"Not a folder: {dicom_dir}")

    case_id = os.path.basename(os.path.normpath(dicom_dir))
    case_dir = case_output_root(app_dir, case_id)
    canonical_dir = os.path.join(case_dir, "canonical")
    seg_cache_dir = os.path.join(case_dir, "seg_cache")
    mesh_dir = os.path.join(case_dir, "mesh_centerlines")
    for d in (canonical_dir, seg_cache_dir, mesh_dir):
        os.makedirs(d, exist_ok=True)

    cfg = AdaptiveConfig(
        work_dir=canonical_dir,
        seg_cache_dir=seg_cache_dir,
        mesh_centerline_dir=mesh_dir,
        sop_png_dir="qc_sop",
        write_enhanced_volume=False,
        write_mesh_centerline=True,
        write_stl_meshes=True,
        write_qc=True,
        force_resegment=force_resegment,
        # Mogon's default (0.6mm) is tuned for an A100 node's memory; on a
        # long-FOV series (chest+abdomen CTA, hundreds of slices) resampling
        # the whole volume to 0.6mm isotropic can ask SimpleITK for a single
        # allocation of several GB, which fails on a typical desktop. Callers
        # on constrained hardware should pass a coarser value (e.g. 1.0-1.5).
        iso_spacing_mm=iso_spacing_mm,
        # TotalSegmentator's full-resolution model on a CPU-only desktop is
        # what turns a long-FOV (1000+ slice) series into an OOM: --fast (the
        # 3mm model) is the single biggest lever for keeping that step inside
        # a laptop's memory. Mogon's GPU jobs never set this (ts_fast defaults
        # to False in Config) and keep running full-resolution.
        ts_fast=ts_fast,
    )

    rec = process_one_case(cfg, dicom_dir)

    html_path = None
    case_mesh_dir = os.path.join(mesh_dir, case_id)
    if rec.get("artifacts", {}).get("aorta_stl"):
        try:
            html_path, _size = export_case(
                case_mesh_dir, case_id,
                os.path.join(case_dir, f"{case_id}_interactive.html"),
                ct_dirs=[canonical_dir],
            )
        except ExportError as exc:
            rec.setdefault("sop_reject_reasons", "")
            rec["html_export_error"] = str(exc)

    hu_volume = None
    spacing_zyx = None
    origin_xyz = None
    aorta_mask = None
    canonical_nifti = os.path.join(canonical_dir, case_id + "_canonical.nii.gz")
    if os.path.isfile(canonical_nifti):
        import SimpleITK as sitk
        img = sitk.ReadImage(canonical_nifti)
        hu_volume = sitk.GetArrayFromImage(img).astype(np.int16)
        sx, sy, sz = img.GetSpacing()
        spacing_zyx = (sz, sy, sx)
        origin_xyz = img.GetOrigin()

        aorta_path = os.path.join(seg_cache_dir, case_id, "aorta.nii.gz")
        if os.path.isfile(aorta_path):
            aorta_img = sitk.ReadImage(aorta_path)
            aorta_img = sitk.Resample(
                aorta_img, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
            aorta_mask = sitk.GetArrayFromImage(aorta_img).astype(bool)

    return {
        "rec": rec,
        "case_id": case_id,
        "case_dir": case_dir,
        "html_path": html_path,
        "hu_volume": hu_volume,
        "spacing_zyx": spacing_zyx,
        "origin_xyz": origin_xyz,
        "aorta_mask": aorta_mask,
    }

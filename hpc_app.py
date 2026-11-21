#!/usr/bin/env python3
"""
hpc_app.py -- browser-based GUI for the aortic stent-graft pipeline, built to
run on a Mogon compute/login node and be reached through an SSH tunnel.

    streamlit run hpc_app.py --server.port 8501 --server.address 127.0.0.1

See hpc/launch_gui.sh for the Mogon-side launcher and the exact SSH
port-forwarding command.

WHAT THIS REUSES (it is a front end, not a reimplementation)
-----------------------------------------------------------
    ct_viewer/mpr.py            axis-aligned MPR plane extraction + aspect
    ct_viewer/segmentation.py   aorta seed/threshold, stent HU threshold,
                                endpoints, overlay compositing
    ct_viewer/enhancement.py    the 2-D filter registry (FILTERS/FILTER_PARAMS)
                                drives the filter controls directly
    ct_viewer/metrics.py        PSNR/SSIM/FSIM for filtered-vs-raw comparison
    ct_viewer/centerline_pipeline.py  run_local_case() -- in-process pipeline
    src/oblique_mpr.py          centerline-orthogonal resampling (new; wraps
                                stent_endpoint_sop.sample_plane)
    src/stent_html_export.py    the interactive 3-D HTML that gets embedded
    src/stent_pipeline.py       what the Slurm batch path invokes

WHAT IT DELIBERATELY DOES NOT IMPORT
------------------------------------
ct_viewer/ct_viewer.py. That module imports PyQt5, a Qt5Agg matplotlib
backend and VTK at module scope, so importing it on a headless compute node
fails before any of its functions become reachable. The only thing this app
needs from it is window_slice(), eight lines of pure numpy, reproduced below
as _window_slice() rather than dragging a GUI toolkit into a web server.
"""

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime

import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
#  Path bootstrap -- must happen before any project import.
#
#  ct_viewer/*.py use flat imports of each other ("import mpr"), so their own
#  directory has to be on sys.path, not just the repo root. src/ likewise.
#  Note ct_viewer/centerline_pipeline.py computes its _SRC_DIR as
#  <ct_viewer>/src, which does not exist in this layout -- putting the real
#  src/ on sys.path here is what makes its pipeline import resolve anyway.
# --------------------------------------------------------------------------- #
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
VIEWER_DIR = os.path.join(APP_DIR, "ct_viewer")
OUTPUT_DIR = os.path.join(APP_DIR, "output")
HPC_DIR = os.path.join(APP_DIR, "hpc")
for _d in (SRC_DIR, VIEWER_DIR):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

st.set_page_config(page_title="Aortic Stent-Graft — Mogon",
                   page_icon="🫀", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
#  Guarded imports. A missing heavy dependency must degrade one feature, not
#  blank the whole page -- the MPR/oblique/filter views are useful on their own
#  on a node where TotalSegmentator or SimpleITK is not installed.
# --------------------------------------------------------------------------- #
def _try(name, fn):
    try:
        return fn(), None
    except Exception as exc:                                   # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


def _imp_mpr():
    import mpr
    return mpr


def _imp_seg():
    import segmentation
    return segmentation


def _imp_enh():
    import enhancement
    return enhancement


def _imp_metrics():
    import metrics
    return metrics


def _imp_oblique():
    import oblique_mpr
    return oblique_mpr


def _imp_sitk():
    import SimpleITK
    return SimpleITK


def _imp_pipeline():
    import centerline_pipeline
    return centerline_pipeline


mpr, ERR_MPR = _try("mpr", _imp_mpr)
seg, ERR_SEG = _try("segmentation", _imp_seg)
enh, ERR_ENH = _try("enhancement", _imp_enh)
qm, ERR_METRICS = _try("metrics", _imp_metrics)
om, ERR_OBLIQUE = _try("oblique_mpr", _imp_oblique)
sitk, ERR_SITK = _try("SimpleITK", _imp_sitk)
clpipe, ERR_PIPE = _try("centerline_pipeline", _imp_pipeline)

WL_PRESETS = {
    "CT Angio (aorta)": (300, 600),
    "Soft tissue": (40, 400),
    "Lung": (-600, 1500),
    "Bone": (300, 1500),
    "Stent / metal": (1500, 2500),
}
MAX_EMBED_MB = 40.0     # above this an inline HTML embed wedges the browser


# --------------------------------------------------------------------------- #
#  Display helpers
# --------------------------------------------------------------------------- #
def _window_slice(hu_slice, wc, ww):
    """HU -> uint8 via window centre/width.

    Behaviourally identical to ct_viewer.window_slice; duplicated here only
    because importing that module pulls in PyQt5 (see module docstring).
    """
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    if hi <= lo:
        hi = lo + 1.0
    out = (hu_slice.astype(np.float32) - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)


def _apply_aspect(img, aspect):
    """Stretch rows so a plane with anisotropic spacing is drawn undistorted.

    st.image has no pixel-aspect argument, so the correction has to be baked
    into the array. Skipped when aspect is ~1 to avoid a pointless resample.
    """
    if aspect is None or abs(aspect - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    new_h = max(1, int(round(h * float(aspect))))
    try:
        import cv2
        interp = cv2.INTER_AREA if new_h < h else cv2.INTER_LINEAR
        return cv2.resize(img, (w, new_h), interpolation=interp)
    except Exception:                                          # noqa: BLE001
        idx = np.clip((np.arange(new_h) / float(aspect)).astype(int), 0, h - 1)
        return img[idx]


def _draw_crosshair(rgb, row, col, color=(0, 255, 255)):
    """One-pixel crosshair, drawn after any aspect correction."""
    out = rgb.copy()
    h, w = out.shape[:2]
    r = int(np.clip(row, 0, h - 1))
    c = int(np.clip(col, 0, w - 1))
    out[r, :, :] = color
    out[:, c, :] = color
    return out


def _to_rgb(u8):
    return np.stack([u8, u8, u8], axis=-1)


def _default_case_id(path):
    """Best guess at the pipeline's case ID for `path`.

    A DICOM directory is already named after the case, but a NIfTI is not:
    the pipeline writes "<case>_canonical.nii.gz", so the bare basename would
    be "6634138-2_canonical.nii.gz" and would never match the
    "<case>_centerline.vtk" it is being paired with. Strip the extension and
    the known suffix so the centerline lookup lands by default.
    """
    if not path:
        return ""
    base = os.path.basename(os.path.normpath(path))
    if os.path.isdir(path):
        return base
    for ext in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    for suffix in ("_canonical", "_ct", "_volume"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


_IMG_WIDTH_KW = None


def show_image(container, img, caption=None):
    """st.image sized to the column, across Streamlit versions.

    `use_container_width` is deprecated in favour of `width="stretch"`, but the
    replacement does not exist in older releases and the cluster's Streamlit
    version is not pinned by this repo. Probe once, then reuse whichever the
    installed version accepts.
    """
    global _IMG_WIDTH_KW
    if _IMG_WIDTH_KW is None:
        try:
            container.image(img, caption=caption, width="stretch")
            _IMG_WIDTH_KW = "width"
            return
        except TypeError:
            _IMG_WIDTH_KW = "use_container_width"
    if _IMG_WIDTH_KW == "width":
        container.image(img, caption=caption, width="stretch")
    else:
        container.image(img, caption=caption, use_container_width=True)


def embed_html(html, height):
    """Render a self-contained HTML scene inline.

    st.components.v1.html is deprecated in favour of st.iframe, which older
    Streamlit releases do not have. Prefer the new API, fall back to the old
    one, so this works on whatever version the cluster venv happens to carry.
    """
    if hasattr(st, "iframe"):
        # Note the signatures differ beyond the name: st.iframe takes
        # (src, *, width, height, tab_index) and has no `scrolling`, so the
        # old call cannot simply be forwarded. It does accept a raw HTML
        # string as `src`, which is what these self-contained scenes are.
        st.iframe(html, height=height)
    else:
        st.components.v1.html(html, height=height, scrolling=False)


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def _dir_signature(path):
    """Cheap cache key that still changes when the data does."""
    try:
        stat = os.stat(path)
        return (path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (path, 0, 0)


@st.cache_data(show_spinner="Reading volume…", max_entries=3)
def load_volume(path, _sig):
    """Read a DICOM directory or a NIfTI/NRRD file into (volume, spacing, geom).

    SimpleITK handles both, and yields the origin/spacing/direction triple that
    oblique_mpr (and stent_endpoint_sop.sample_plane underneath it) expect for
    world-space resampling -- so one reader covers both the axis-aligned and
    the centerline-orthogonal views.
    """
    if sitk is None:
        raise RuntimeError("SimpleITK is not available: %s" % ERR_SITK)

    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesIDs(path)
        if not ids:
            raise RuntimeError("no DICOM series found in %s" % path)
        # Longest series = the diagnostic acquisition, not a scout/localizer.
        best, best_files = None, []
        for sid in ids:
            files = reader.GetGDCMSeriesFileNames(path, sid)
            if len(files) > len(best_files):
                best, best_files = sid, files
        reader.SetFileNames(best_files)
        img = reader.Execute()
        label = "%s (%d slices)" % (os.path.basename(path), len(best_files))
    else:
        img = sitk.ReadImage(path)
        label = os.path.basename(path)

    vol = sitk.GetArrayFromImage(img)
    if vol.ndim != 3:
        raise RuntimeError("expected a 3-D volume, got shape %s" % (vol.shape,))
    vol = vol.astype(np.int16, copy=False)

    sx, sy, sz = img.GetSpacing()
    geom = (tuple(img.GetOrigin()), (sx, sy, sz), tuple(img.GetDirection()))
    return vol, (sz, sy, sx), geom, label


@st.cache_data(show_spinner="Segmenting aorta (HU threshold)…", max_entries=2)
def segment_aorta_cached(_vol, tolerance, sig):
    """seg.auto_segment_aorta returns a cropped sub-mask + bbox; expand it once
    here so every viewport can index it like the volume."""
    sub, bbox, seed = seg.auto_segment_aorta(_vol, tolerance=tolerance)
    if sub.size == 0:
        return None, None, None
    full = np.zeros(_vol.shape, dtype=bool)
    z0, z1, y0, y1, x0, x1 = bbox
    full[z0:z1, y0:y1, x0:x1] = sub.astype(bool)
    return full, bbox, seed


@st.cache_data(show_spinner="Thresholding stent metal…", max_entries=2)
def segment_stent_cached(_vol, _aorta_mask, hu_threshold, dilate, sig):
    sub, bbox = seg.segment_stent(_vol, _aorta_mask,
                                  hu_threshold=int(hu_threshold),
                                  dilate=int(dilate))
    if sub.size == 0:
        return None, None
    full = np.zeros(_vol.shape, dtype=bool)
    z0, z1, y0, y1, x0, x1 = bbox
    full[z0:z1, y0:y1, x0:x1] = sub.astype(bool)
    return full, bbox


# --------------------------------------------------------------------------- #
#  Slurm
# --------------------------------------------------------------------------- #
JOBS_FILE = os.path.join(OUTPUT_DIR, "gui_jobs.json")


def _have(cmd):
    from shutil import which
    return which(cmd) is not None


def load_jobs():
    try:
        with open(JOBS_FILE, "r") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


def save_jobs(jobs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = JOBS_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(jobs, fh, indent=2)
    os.replace(tmp, JOBS_FILE)


def build_job_script(case_dir, work_dir, seg_cache_dir, account, partition,
                     gres, cpus, mem, walltime, job_name):
    """A self-contained sbatch script for one case.

    env.sh is intentionally NOT sourced. It runs `module purge`, and its line
    41 (MIGRATION_REPORTS_DIR="${DATA_ROOT}/...") dereferences a variable it
    never sets -- under its own `set -uo pipefail` that aborts sourcing before
    the venv is activated. Paths are passed explicitly here instead, and the
    venv is activated directly if one is configured.
    """
    venv = os.environ.get("STENT_VENV", "")
    activate = ('source "%s/bin/activate"' % venv) if venv else \
               '# STENT_VENV not set; using the interpreter already on PATH'
    gres_line = ("#SBATCH --gres=%s" % gres) if gres else ""
    part_line = ("#SBATCH --partition=%s" % partition) if partition else ""
    acct_line = ("#SBATCH --account=%s" % account) if account else ""
    return """#!/bin/bash
#SBATCH --job-name={job_name}
{acct_line}
{part_line}
{gres_line}
#SBATCH --time={walltime}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --output={work_dir}/%x-%j.out
#SBATCH --error={work_dir}/%x-%j.err

set -uo pipefail

export OMP_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}
export nnUNet_n_proc_DA=0
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

{activate}

cd {src_dir} || exit 1
echo "=== case {case_dir} @ $(date '+%F %T') ==="
python stent_pipeline.py \\
    --single-case {case_dir} \\
    --work-dir {work_dir} \\
    --seg-cache-dir {seg_cache_dir}
echo "=== done rc=$? @ $(date '+%F %T') ==="
""".format(job_name=job_name, acct_line=acct_line, part_line=part_line,
           gres_line=gres_line, walltime=walltime, cpus=cpus, mem=mem,
           work_dir=shlex.quote(work_dir), activate=activate,
           src_dir=shlex.quote(SRC_DIR), case_dir=shlex.quote(case_dir),
           seg_cache_dir=shlex.quote(seg_cache_dir))


def submit_job(script_text, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, "gui_job_%s.slurm"
                        % datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(path, "w", newline="\n") as fh:
        fh.write(script_text)
    proc = subprocess.run(["sbatch", path], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    # "Submitted batch job 12345"
    jid = proc.stdout.strip().split()[-1]
    return jid, path


def job_state(job_id):
    """Live state via squeue, falling back to the accounting DB once the job
    leaves the queue (squeue only knows about pending/running jobs)."""
    if _have("squeue"):
        p = subprocess.run(["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                           capture_output=True, text=True)
        s = p.stdout.strip()
        if s:
            return s.split()[0]
    if _have("sacct"):
        p = subprocess.run(
            ["sacct", "-j", str(job_id), "-n", "-P", "-o", "State"],
            capture_output=True, text=True)
        for line in p.stdout.splitlines():
            line = line.strip()
            if line:
                return line.split()[0]
    return "UNKNOWN"


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🫀 Aortic Stent-Graft")
st.sidebar.caption("Mogon HPC · web front end")

missing = [(n, e) for n, e in (("mpr", ERR_MPR), ("segmentation", ERR_SEG),
                               ("enhancement", ERR_ENH), ("metrics", ERR_METRICS),
                               ("oblique_mpr", ERR_OBLIQUE),
                               ("SimpleITK", ERR_SITK)) if e]
if missing:
    with st.sidebar.expander("⚠️ %d module(s) unavailable" % len(missing)):
        for n, e in missing:
            st.write("**%s** — %s" % (n, e))

st.sidebar.header("1 · Dataset")
default_root = os.environ.get("STENT_DICOM_ROOT", OUTPUT_DIR)
src_kind = st.sidebar.radio("Source", ["DICOM directory", "NIfTI / NRRD file"],
                            horizontal=True)

picked = None
browse_root = st.sidebar.text_input("Browse under", value=default_root,
                                    help="A directory on the cluster filesystem "
                                         "to list candidates from.")
if os.path.isdir(browse_root):
    try:
        if src_kind == "DICOM directory":
            opts = sorted(d for d in os.listdir(browse_root)
                          if os.path.isdir(os.path.join(browse_root, d)))
        else:
            opts = sorted(f for f in os.listdir(browse_root)
                          if f.lower().endswith((".nii", ".nii.gz", ".nrrd", ".mha")))
    except OSError as exc:
        opts, _ = [], st.sidebar.error("Cannot list: %s" % exc)
    if opts:
        chosen = st.sidebar.selectbox("Found %d" % len(opts), ["—"] + opts)
        if chosen != "—":
            picked = os.path.join(browse_root, chosen)
    else:
        st.sidebar.info("Nothing matching in that folder.")
else:
    st.sidebar.warning("Not a directory.")

manual = st.sidebar.text_input("…or full path", value=picked or "")
data_path = manual.strip() or picked

st.sidebar.header("2 · Display")
preset = st.sidebar.selectbox("Window preset", list(WL_PRESETS))
_wc, _ww = WL_PRESETS[preset]
wc = st.sidebar.slider("Window centre (HU)", -1000, 3000, _wc, 10)
ww = st.sidebar.slider("Window width (HU)", 1, 4000, _ww, 10)

st.sidebar.header("3 · Segmentation")
aorta_tol = st.sidebar.slider("Aorta HU tolerance", 20, 400, 120, 10,
                              help="Half-width of the HU band grown from the "
                                   "auto-located aortic seed.")
stent_hu = st.sidebar.slider("Stent HU threshold", 500, 3000, 1200, 50,
                             help="Metal is thresholded above this, inside a "
                                  "dilated aorta ROI.")
stent_dilate = st.sidebar.slider("Aorta ROI dilation (vox)", 0, 10, 3, 1)
do_seg = st.sidebar.checkbox("Run segmentation", value=False)

st.sidebar.header("4 · Filter")
filter_names = ["None"] + (sorted(enh.FILTERS) if enh else [])
filter_name = st.sidebar.selectbox("2-D filter", filter_names)
filter_kwargs = {}
if enh and filter_name != "None":
    for pname, pmin, pmax, pstep, pdef in enh.FILTER_PARAMS.get(filter_name, []):
        if isinstance(pdef, int) and isinstance(pstep, int):
            filter_kwargs[pname] = st.sidebar.slider(
                pname, int(pmin), int(pmax), int(pdef), int(pstep))
        else:
            filter_kwargs[pname] = st.sidebar.slider(
                pname, float(pmin), float(pmax), float(pdef), float(pstep))

st.sidebar.header("5 · Centerline")
cl_step = st.sidebar.slider("Sampling step (mm)", 0.25, 5.0, 1.0, 0.25)
cl_fov = st.sidebar.slider("Cross-section FOV (mm)", 20, 160, 60, 5)
cl_res = st.sidebar.select_slider("Cross-section resolution (mm/px)",
                                  [0.2, 0.25, 0.3, 0.4, 0.5, 0.8], value=0.4)

# --------------------------------------------------------------------------- #
#  Load
# --------------------------------------------------------------------------- #
volume = spacing = geom = None
label = ""
if data_path and os.path.exists(data_path):
    try:
        volume, spacing, geom, label = load_volume(data_path,
                                                   _dir_signature(data_path))
    except Exception as exc:                                   # noqa: BLE001
        st.error("Could not load **%s**\n\n`%s`" % (data_path, exc))
elif data_path:
    st.error("Path does not exist: `%s`" % data_path)

if volume is None:
    st.title("Aortic Stent-Graft — Mogon web GUI")
    st.info("Pick a DICOM directory or a NIfTI file in the sidebar to begin.")
    st.markdown(
        "This front end reuses the existing pipeline modules directly "
        "(`ct_viewer/mpr.py`, `segmentation.py`, `enhancement.py`, "
        "`metrics.py`, `src/stent_endpoint_sop.py`, `src/stent_html_export.py`). "
        "It does **not** import `ct_viewer.py`, which requires PyQt5 and a "
        "display and so cannot run on a headless compute node.")
    st.stop()

nz, ny, nx = volume.shape
if "cursor" not in st.session_state or st.session_state.get("cur_for") != data_path:
    st.session_state.cursor = (nz // 2, ny // 2, nx // 2)
    st.session_state.cur_for = data_path

st.title(label or "Volume")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Shape (z,y,x)", "%d×%d×%d" % (nz, ny, nx))
c2.metric("Spacing (mm)", "%.2f×%.2f×%.2f" % spacing)
c3.metric("HU range", "%d … %d" % (int(volume.min()), int(volume.max())))
c4.metric("Extent (mm)", "%.0f×%.0f×%.0f"
          % (nz * spacing[0], ny * spacing[1], nx * spacing[2]))

# --------------------------------------------------------------------------- #
#  Optional segmentation
# --------------------------------------------------------------------------- #
aorta_mask = stent_mask = None
if do_seg and seg is not None:
    sig = (data_path, aorta_tol)
    try:
        aorta_mask, _bbox, seed = segment_aorta_cached(volume, aorta_tol, sig)
        if aorta_mask is None:
            st.warning("No plausible aorta seed found — try a wider HU tolerance.")
        else:
            stent_mask, _sb = segment_stent_cached(
                volume, aorta_mask, stent_hu, stent_dilate,
                (data_path, aorta_tol, stent_hu, stent_dilate))
    except Exception as exc:                                   # noqa: BLE001
        st.error("Segmentation failed: `%s`" % exc)

tab_mpr, tab_obl, tab_3d, tab_jobs = st.tabs(
    ["🧭 MPR (axial / sagittal / coronal)",
     "⟂ Orthogonal to centerline",
     "🧊 Interactive 3-D",
     "🚀 Run / Slurm"])


# --------------------------------------------------------------------------- #
#  TAB 1 -- synchronized MPR
# --------------------------------------------------------------------------- #
with tab_mpr:
    if mpr is None:
        st.error("mpr.py unavailable: %s" % ERR_MPR)
    else:
        z, y, x = st.session_state.cursor
        s1, s2, s3 = st.columns(3)
        z = s1.slider("Axial (z)", 0, nz - 1, int(z))
        y = s2.slider("Coronal (y)", 0, ny - 1, int(y))
        x = s3.slider("Sagittal (x)", 0, nx - 1, int(x))
        cursor = mpr.clamp_cursor((z, y, x), volume.shape)
        st.session_state.cursor = cursor

        show_cross = st.checkbox("Crosshair", value=True)
        show_ov = st.checkbox("Overlay segmentation", value=bool(aorta_mask is not None))

        cols = st.columns(3)
        for col, plane in zip(cols, mpr.PLANES):
            raw = mpr.plane_slice(volume, plane, cursor)
            u8 = _window_slice(raw, wc, ww)

            if enh is not None and filter_name != "None":
                try:
                    u8 = enh.FILTERS[filter_name](u8, **filter_kwargs)
                except Exception as exc:                       # noqa: BLE001
                    col.warning("Filter failed: %s" % exc)

            if show_ov and seg is not None and (aorta_mask is not None
                                                or stent_mask is not None):
                layers = []
                if aorta_mask is not None:
                    layers.append((mpr.plane_slice(aorta_mask, plane, cursor),
                                   (255, 80, 80), 0.35))
                if stent_mask is not None:
                    layers.append((mpr.plane_slice(stent_mask, plane, cursor),
                                   (80, 255, 255), 0.65))
                rgb = seg.blend_overlay(u8, layers)
            else:
                rgb = _to_rgb(u8)

            aspect = mpr.plane_aspect(plane, spacing)
            rgb = _apply_aspect(rgb, aspect)
            if show_cross:
                r, c = mpr.plane_crosshair(plane, cursor)
                if aspect and abs(aspect - 1.0) > 1e-3:
                    r = int(round(r * aspect))
                rgb = _draw_crosshair(rgb, r, c)

            col.caption("**%s** — %s" % (plane.capitalize(),
                                         "×".join(str(v) for v in raw.shape)))
            show_image(col, rgb)

        hu = int(volume[cursor])
        st.caption("Cursor (z=%d, y=%d, x=%d) · **%d HU**" % (*cursor, hu))

        if qm is not None and filter_name != "None":
            with st.expander("Filtered-vs-raw image quality (this axial slice)"):
                try:
                    raw_ax = _window_slice(
                        mpr.plane_slice(volume, "axial", cursor), wc, ww)
                    filt_ax = enh.FILTERS[filter_name](raw_ax, **filter_kwargs)
                    m = qm.enhancement_metrics(raw_ax, filt_ax)
                    mc = st.columns(4)
                    for (k, v), cc in zip(m.items(), mc):
                        cc.metric(k, "∞" if np.isinf(v) else "%.4f" % v)
                except Exception as exc:                       # noqa: BLE001
                    st.warning("Metrics unavailable: %s" % exc)


# --------------------------------------------------------------------------- #
#  TAB 2 -- centerline-orthogonal viewer
# --------------------------------------------------------------------------- #
with tab_obl:
    st.markdown(
        "Each frame is resampled on the plane **perpendicular to the local "
        "centerline tangent**, so a vessel that runs obliquely to the scanner "
        "axes is measured across its true circular section rather than the "
        "elongated one an axial cut would give.")

    if om is None:
        st.error("oblique_mpr unavailable: %s" % ERR_OBLIQUE)
    else:
        default_case_dir = os.path.join(OUTPUT_DIR, "hpc_pipeline")
        cl_dir = st.text_input(
            "Centerline artifacts directory",
            value=st.session_state.get("cl_dir", default_case_dir),
            help="Directory holding <case>_centerline.vtk/.csv, as written by "
                 "the pipeline's mesh_centerlines stage.")
        st.session_state.cl_dir = cl_dir
        cl_case = st.text_input("Case ID", value=_default_case_id(data_path),
                                help="Used to find <case>_centerline.vtk/.csv "
                                     "in the directory above.")

        if not cl_dir or not os.path.isdir(cl_dir):
            st.info("Point this at a directory containing a centerline, or run "
                    "the pipeline first from the **Run / Slurm** tab.")
        else:
            try:
                pts = om.load_centerline_points(cl_dir, cl_case)
                rs = om.CenterlineResampler(pts, step_mm=float(cl_step))

                st.success("Centerline: %d points · %.1f mm long · resampled "
                           "every %.2f mm" % (len(pts), rs.total_length_mm, cl_step))

                pos = st.slider("Position along centerline (mm from proximal end)",
                                0.0, float(max(rs.total_length_mm, 0.01)),
                                float(rs.total_length_mm) / 2.0, float(cl_step))
                i = rs.index_at_mm(pos)

                patch, extent = rs.cross_section(
                    volume.astype(np.float32), geom, i,
                    fov_mm=float(cl_fov), res_mm=float(cl_res))
                pu8 = _window_slice(patch, wc, ww)
                if enh is not None and filter_name != "None":
                    try:
                        pu8 = enh.FILTERS[filter_name](pu8, **filter_kwargs)
                    except Exception:                          # noqa: BLE001
                        pass

                prgb = _to_rgb(pu8)
                mid = prgb.shape[0] // 2
                prgb = _draw_crosshair(prgb, mid, prgb.shape[1] // 2,
                                       color=(255, 200, 0))

                left, right = st.columns([2, 1])
                show_image(left, prgb,
                           caption="Perpendicular cross-section · FOV %d mm · "
                                   "%.2f mm/px" % (cl_fov, cl_res))
                p = rs.points[i]
                t = rs.tangents[i]
                right.metric("Arc position", "%.1f mm" % rs.arclength[i])
                right.metric("Sample index", "%d / %d" % (i, len(rs) - 1))
                right.write("**World point (LPS mm)**")
                right.code("x %8.2f\ny %8.2f\nz %8.2f" % (p[0], p[1], p[2]))
                right.write("**Tangent**")
                right.code("%.3f, %.3f, %.3f" % (t[0], t[1], t[2]))
                obliquity = float(np.degrees(np.arccos(
                    np.clip(abs(t[2]), 0.0, 1.0))))
                right.metric("Angle to axial plane", "%.1f°" % obliquity,
                             help="0° means the vessel runs straight through "
                                  "the scanner's z axis, so an axial slice "
                                  "would already be orthogonal. Larger values "
                                  "are where this view differs most from MPR.")

                st.divider()
                lm_hu = st.slider("Lumen threshold (HU)", -200, 1500, 200, 10,
                                  key="lumen_hu",
                                  help="Contrast-filled lumen is bright; this "
                                       "separates it from the vessel wall.")
                lm = om.lumen_metrics(patch, float(cl_res), hu_threshold=lm_hu)
                rcurv = rs.curvature_radius_mm()[i]
                m1, m2, m3 = st.columns(3)
                m1.metric("Lumen diameter", "%.1f mm" % lm["diameter_mm"],
                          help="Equivalent circular diameter of the connected "
                               "region containing the plane centre.")
                m2.metric("Lumen area", "%.0f mm²" % lm["area_mm2"])
                m3.metric("Curvature radius",
                          "∞" if not np.isfinite(rcurv) else "%.0f mm" % rcurv)

                if lm["n_components"] > 1:
                    # Expected geometry, not a defect: a flat plane cuts a
                    # bent vessel more than once once the FOV approaches the
                    # local radius of curvature. Measured on a phantom, summing
                    # every bright pixel here instead of the centre component
                    # overstated an 18 mm lumen as 25 mm.
                    st.info(
                        "This plane crosses the vessel **%d** times "
                        "(%.0f mm² outside the central lumen) — the bend here "
                        "curves back through the same plane (curvature radius "
                        "%.0f mm vs %d mm FOV). The measurement above uses "
                        "only the region at the crosshair; narrow the FOV to "
                        "exclude the rest."
                        % (lm["n_components"], lm["extra_area_mm2"],
                           rcurv if np.isfinite(rcurv) else 0.0, cl_fov))
            except om.CenterlineError as exc:
                st.warning("No usable centerline: %s" % exc)
            except Exception as exc:                           # noqa: BLE001
                st.error("Cross-section failed: `%s`" % exc)


# --------------------------------------------------------------------------- #
#  TAB 3 -- interactive 3-D
# --------------------------------------------------------------------------- #
with tab_3d:
    st.markdown("Interactive 3-D scenes exported by `src/stent_html_export.py` "
                "(aorta surface, centerline tube, stent, crown endpoints).")
    search_dirs = [OUTPUT_DIR, os.path.join(OUTPUT_DIR, "hpc_pipeline")]
    found = []
    for root in search_dirs:
        if os.path.isdir(root):
            for dirpath, _dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith(".html"):
                        found.append(os.path.join(dirpath, fn))
    found = sorted(set(found))

    if not found:
        st.info("No exported `.html` scenes under `output/` yet. Run a case "
                "from the **Run / Slurm** tab — the pipeline writes one per "
                "case via `export_case()`.")
    else:
        pick = st.selectbox("Scene (%d found)" % len(found), found,
                            format_func=lambda p: os.path.relpath(p, APP_DIR))
        size_mb = os.path.getsize(pick) / 1e6
        st.caption("`%s` — %.1f MB" % (os.path.relpath(pick, APP_DIR), size_mb))

        if size_mb > MAX_EMBED_MB:
            # These exports embed geometry as base64; a full-skeleton scene can
            # reach 100 MB, and inlining that into the page will hang the tab
            # rather than render slowly. Offer it as a download instead.
            st.warning(
                "This scene is %.0f MB, above the %.0f MB inline-embed limit. "
                "Embedding it would stall the browser tab. Download it and "
                "open it directly, or re-export with `--no-skeleton` for a "
                "much smaller file." % (size_mb, MAX_EMBED_MB))
            with open(pick, "rb") as fh:
                st.download_button("⬇ Download scene", fh,
                                   file_name=os.path.basename(pick),
                                   mime="text/html")
        else:
            height = st.slider("Viewport height (px)", 400, 1200, 720, 20)
            with open(pick, "r", encoding="utf-8", errors="replace") as fh:
                html = fh.read()
            embed_html(html, height)


# --------------------------------------------------------------------------- #
#  TAB 4 -- execution
# --------------------------------------------------------------------------- #
with tab_jobs:
    st.markdown("Run the full pipeline (`process_one_case` → centerline, crown "
                "detection, STL + interactive HTML) either here in the web "
                "process or as an asynchronous Slurm job.")

    has_sbatch = _have("sbatch")
    mode = st.radio(
        "Execution mode",
        ["In-process (blocking)", "Slurm batch (sbatch)"],
        horizontal=True,
        index=1 if has_sbatch else 0,
        help=None if has_sbatch else
             "sbatch was not found on PATH — this is not a Slurm submit host, "
             "so only in-process execution is available here.")

    if mode.startswith("Slurm") and not has_sbatch:
        st.warning("`sbatch` is not on PATH. Run this app on a Mogon login "
                   "node (or any submit host) to enable batch submission.")

    case_dir = st.text_input("Case DICOM directory", value=data_path or "")
    work_dir = st.text_input("Work / output directory",
                             value=os.path.join(OUTPUT_DIR, "hpc_pipeline"))
    seg_cache = st.text_input("Segmentation cache",
                              value=os.path.join(OUTPUT_DIR, "seg_cache"))

    if mode.startswith("In-process"):
        colA, colB = st.columns(2)
        iso = colA.slider("Isotropic resample (mm)", 0.4, 2.0, 1.0, 0.1,
                          help="0.6 mm matches the A100 job; coarser values "
                               "keep the resample inside a smaller memory "
                               "budget on a CPU node.")
        ts_fast = colB.checkbox("TotalSegmentator --fast (3 mm model)", value=True)
        force = colB.checkbox("Force re-segmentation", value=False)

        if st.button("▶ Run now", type="primary", disabled=not case_dir):
            if clpipe is None:
                st.error("centerline_pipeline unavailable: %s" % ERR_PIPE)
            else:
                prog = st.progress(0.0, text="Starting…")
                try:
                    prog.progress(0.1, text="Running pipeline (this can take "
                                            "many minutes on CPU)…")
                    res = clpipe.run_local_case(
                        case_dir, APP_DIR, force_resegment=force,
                        iso_spacing_mm=float(iso), ts_fast=bool(ts_fast))
                    prog.progress(1.0, text="Done")
                    rec = res.get("rec", {})
                    st.success("Status: **%s**" % rec.get("status", "?"))
                    if res.get("html_path"):
                        st.info("Interactive scene: `%s`" % res["html_path"])
                    st.json(rec, expanded=False)
                except Exception as exc:                       # noqa: BLE001
                    prog.empty()
                    st.error("Pipeline failed: `%s`" % exc)
    else:
        st.subheader("Slurm resources")
        r1, r2, r3 = st.columns(3)
        account = r1.text_input("Account", value=os.environ.get("SLURM_ACCOUNT", "ki-adies"))
        partition = r2.text_input("Partition", value=os.environ.get("SLURM_PARTITION_GPU", "a100ai"))
        gres = r3.text_input("GRES", value=os.environ.get("SLURM_GRES",
                                                          "gpu:nvidia_a100-sxm4-80gb:1"))
        r4, r5, r6 = st.columns(3)
        cpus = r4.number_input("CPUs", 1, 64, 8)
        mem = r5.text_input("Memory", value="32G")
        walltime = r6.text_input("Walltime", value="02:00:00")

        script = build_job_script(case_dir, work_dir, seg_cache, account,
                                  partition, gres, int(cpus), mem, walltime,
                                  "stent-gui")
        with st.expander("Preview generated sbatch script"):
            st.code(script, language="bash")

        if st.button("🚀 Submit to Slurm", type="primary",
                     disabled=not (case_dir and has_sbatch)):
            try:
                jid, spath = submit_job(script, work_dir)
                jobs = load_jobs()
                jobs.insert(0, {"job_id": jid, "case": case_dir,
                                "submitted": datetime.now().isoformat(timespec="seconds"),
                                "script": spath, "work_dir": work_dir})
                save_jobs(jobs)
                st.success("Submitted job **%s**" % jid)
            except Exception as exc:                           # noqa: BLE001
                st.error("Submission failed: `%s`" % exc)

    st.divider()
    st.subheader("Tracked jobs")
    jobs = load_jobs()
    if not jobs:
        st.caption("No jobs submitted from this GUI yet.")
    else:
        if st.button("🔄 Refresh states"):
            st.rerun()
        term = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
                "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}
        for j in jobs[:15]:
            state = job_state(j["job_id"]) if has_sbatch else "n/a"
            done = any(state.startswith(t) for t in term)
            icon = ("✅" if state.startswith("COMPLETED")
                    else "❌" if done else "⏳")
            with st.expander("%s  **%s** · %s · %s"
                             % (icon, j["job_id"], state,
                                os.path.basename(j["case"]))):
                st.write("Submitted:", j["submitted"])
                st.write("Work dir:", "`%s`" % j["work_dir"])
                if state in ("RUNNING", "PENDING"):
                    st.progress(0.5 if state == "RUNNING" else 0.1,
                                text=state.title())
                elif state.startswith("COMPLETED"):
                    st.progress(1.0, text="Completed")
                out = os.path.join(j["work_dir"],
                                   "stent-gui-%s.out" % j["job_id"])
                if os.path.isfile(out):
                    with open(out, "r", errors="replace") as fh:
                        tail = fh.read()[-4000:]
                    st.code(tail or "(empty)", language="text")
                else:
                    st.caption("No log yet at `%s`" % out)

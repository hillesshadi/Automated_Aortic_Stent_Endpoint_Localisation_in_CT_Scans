#!/usr/bin/env python3
"""
Headless batch runner for the aortic centerline & stent verification pipeline.

Runs the whole pipeline from a terminal with no X server, no $DISPLAY and no
Qt, which is what a Slurm compute node gives you. Writes meshes, screenshots,
figures, metrics and a machine-readable summary to an output directory.

    python run_headless.py --volume CASE_canonical.nii.gz --out results/CASE
    python run_headless.py --dicom /scratch/cases/CASE --out results/CASE \
        --specs device.json --stages all

    srun --gres=gpu:1 --pty python run_headless.py --dicom ... --out ...
    sbatch hpc/run_headless.slurm /scratch/cases/CASE results/CASE

Exit codes
    0  every requested stage completed
    1  the run failed (bad input, or a stage raised)
    2  the run completed but 3-D screenshots were skipped: no offscreen
       OpenGL backend worked. Meshes, figures and metrics are still written.

Import order matters. pipeline.offscreen.configure_headless() sets the
environment that VTK, PyVista and matplotlib read *when they are first
imported*, so it runs before any of them is touched -- hence the deferred
imports further down this file rather than at the top.
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import offscreen                                  # noqa: E402

STAGES = ("preprocess", "centerline", "stent", "profile", "metrics")


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
class Log:
    """Timestamped, line-buffered logging.

    Slurm captures stdout through a pipe, which is block-buffered by default;
    without the explicit flush a job that dies leaves an empty .out file and
    the failure looks like it happened somewhere it did not.
    """

    def __init__(self, path=None, quiet=False):
        self.t0 = time.time()
        self.quiet = quiet
        self.handle = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, text, level="info"):
        line = "[%7.1fs] %-5s %s" % (time.time() - self.t0, level.upper(), text)
        if not self.quiet or level in ("error", "warn"):
            try:
                sys.stdout.write(line + "\n")
            except UnicodeEncodeError:
                # Belt and braces: configure_headless() reconfigures stdout with
                # errors="replace", but a caller that constructs this Log
                # without going through it would still be exposed.
                enc = getattr(sys.stdout, "encoding", None) or "ascii"
                sys.stdout.write(
                    line.encode(enc, "replace").decode(enc, "replace") + "\n")
            sys.stdout.flush()
        if self.handle:
            self.handle.write(line + "\n")
            self.handle.flush()

    def warn(self, text):
        self("%s" % text, "warn")

    def error(self, text):
        self("%s" % text, "error")

    def close(self):
        if self.handle:
            self.handle.close()


# --------------------------------------------------------------------------- #
#  Runner
# --------------------------------------------------------------------------- #
class HeadlessRun:
    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.out = os.path.abspath(args.out)
        os.makedirs(self.out, exist_ok=True)
        self.ctx = None
        self.specs = None
        self.profile_data = None
        self.written = []
        self.summary = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "argv": sys.argv[1:],
            "stages": {},
            "outputs": [],
            "warnings": [],
        }

    # -- helpers ---------------------------------------------------------- #
    def path(self, name):
        return os.path.join(self.out, "%s_%s" % (self.case_id, name))

    @property
    def case_id(self):
        return (self.ctx.case_id if self.ctx and self.ctx.case_id
                else self.args.case_id or "case")

    def record(self, path):
        if path:
            self.written.append(path)
            self.summary["outputs"].append(os.path.basename(path))
            self.log("wrote %s" % os.path.basename(path))
        return path

    def warn(self, text):
        self.log.warn(text)
        self.summary["warnings"].append(text)

    def stage(self, name, fn):
        t = time.time()
        self.log("--- %s ---" % name)
        try:
            fn()
        except Exception as exc:                               # noqa: BLE001
            self.summary["stages"][name] = {
                "ok": False, "error": str(exc),
                "seconds": round(time.time() - t, 2)}
            raise
        self.summary["stages"][name] = {
            "ok": True, "seconds": round(time.time() - t, 2)}

    # -- stages ------------------------------------------------------------ #
    def load(self):
        from pipeline import io_data
        from pipeline.context import CaseContext

        if self.args.dicom:
            result = io_data.load_dicom_series(
                self.args.dicom, progress=lambda m: self.log(m))
            src, kind = self.args.dicom, "dicom"
        else:
            result = io_data.load_image_file(
                self.args.volume, progress=lambda m: self.log(m))
            src, kind = self.args.volume, "image"

        self.ctx = CaseContext(
            case_id=self.args.case_id or result["case_id"],
            source_path=src, source_kind=kind, volume=result["volume"],
            spacing_zyx=result["spacing_zyx"], geom=result["geom"])
        self.log(self.ctx.describe())
        self.summary["case_id"] = self.ctx.case_id
        self.summary["source"] = src
        self.summary["volume"] = {
            "shape_zyx": list(self.ctx.volume.shape),
            "spacing_zyx_mm": [float(v) for v in self.ctx.spacing_zyx],
            "hu_range": [int(self.ctx.volume.min()), int(self.ctx.volume.max())],
        }

        if self.args.centerline:
            pts = io_data.load_centerline_file(self.args.centerline)
            import oblique_mpr as om
            self.ctx.centerline = pts
            self.ctx.resampler = om.CenterlineResampler(
                pts, step_mm=self.args.cl_step)
            self.log("centerline from file: %d points, %.1f mm"
                     % (len(pts), self.ctx.resampler.total_length_mm))

        if self.args.reference:
            self._load_reference()

        if self.args.specs:
            from pipeline import stent_specs
            self.specs = stent_specs.load_specs(self.args.specs)
            self.log("specs: %s" % self.specs.get("device", self.args.specs))
            self.summary["specs"] = {k: v for k, v in self.specs.items()
                                     if k != "source_file"}

    def _load_reference(self):
        """Adopt a pre-computed segmentation as the working aorta mask.

        Done at load time rather than inside the metrics stage so that every
        downstream stage -- centerline, stent, profile, metrics -- sees a valid
        mask even when `preprocess` is not in --stages. Without this, supplying
        a perfectly good segmentation still produced "no aorta mask" warnings
        and a metrics failure.
        """
        import SimpleITK as sitk
        path = self.args.reference
        if not os.path.isfile(path):
            raise FileNotFoundError("reference mask not found: %s" % path)
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = arr.astype(bool)
        if arr.shape != self.ctx.volume.shape:
            self.warn("reference mask %s is %s but the case is %s; ignoring it"
                      % (os.path.basename(path), arr.shape,
                         self.ctx.volume.shape))
            return
        self.ctx.reference_mask = arr
        self.ctx.reference_name = os.path.basename(path)
        self.ctx.use_reference_as_aorta()
        self.log("reference mask %s: %d voxels -- adopted as the aorta mask"
                 % (self.ctx.reference_name, int(arr.sum())))
        self.summary["reference_mask"] = {
            "file": path, "voxels": int(arr.sum()),
            "used_as_aorta_mask": not self.args.resegment,
        }

    def preprocess(self):
        from pipeline import preprocess
        params = {
            "aorta_tolerance": self.args.aorta_tol,
            "stent_hu": self.args.stent_hu,
            "stent_dilate": self.args.stent_dilate,
            "resegment": self.args.resegment,
        }
        preprocess.run_preprocessing(
            self.ctx, params, progress=lambda p, m: self.log("  %3d%%  %s" % (p, m)))
        m = self.ctx.metrics
        self.summary["segmentation"] = {
            "aorta_voxels": m.get("aorta_voxels"),
            "aorta_volume_ml": round(m.get("aorta_volume_ml", 0.0), 2),
            "stent_voxels": m.get("stent_voxels"),
            "aorta_mask_source": self.ctx.aorta_mask_source or "segmented",
        }
        for note in self.ctx.notes:
            self.log("  " + note)

    def centerline(self):
        from pipeline import centerline as cl
        if self.ctx.resampler is not None and self.args.centerline:
            self.log("using the centerline loaded from file; skipping extraction")
            return
        cl.extract_centerline(
            self.ctx, step_mm=self.args.cl_step, orient=True,
            progress=lambda p, m: self.log("  %3d%%  %s" % (p, m)))
        m = self.ctx.metrics
        self.summary["centerline"] = {
            "points": m.get("centerline_points"),
            "length_mm": round(m.get("centerline_length_mm", 0.0), 1),
            "orientation": m.get("centerline_orientation_rule"),
        }

    def stent(self):
        from pipeline import stent_fit
        result = stent_fit.fit_stent(self.ctx)
        self.summary["stent"] = {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in result.items()
            if not isinstance(v, (list, dict)) or k == "warnings"}
        for w in result.get("warnings", []):
            self.warn("stent fit: %s" % w)
        self.log("stent %.1f mm straight-line, %.1f mm along vessel"
                 % (result.get("length_mm", 0.0),
                    result.get("along_vessel_length_mm", 0.0)))

        if self.specs or self.ctx.resampler is not None:
            from pipeline import stent_specs
            table = stent_specs.landmark_table(
                self.ctx, self.specs, lumen_hu=self.args.lumen_hu)
            for w in table["summary"].get("warnings", []):
                self.warn("landmarks: %s" % w)
            self.summary["landmarks"] = table["summary"]
            for p in stent_specs.export_landmarks(table, self.out, self.case_id):
                self.record(p)

    def profile(self):
        from pipeline import centerline as cl
        self.profile_data = cl.diameter_profile(
            self.ctx, fov_mm=self.args.fov, res_mm=max(0.5, self.args.res),
            lumen_hu=self.args.lumen_hu, stride=self.args.profile_stride,
            progress=lambda p, m: self.log("  %3d%%  %s" % (p, m)))
        m = self.ctx.metrics
        self.summary["diameter"] = {
            "mean_mm": round(m.get("diameter_mean_mm", 0.0), 2),
            "min_mm": round(m.get("diameter_min_mm", 0.0), 2),
            "max_mm": round(m.get("diameter_max_mm", 0.0), 2),
        }

    def metrics(self):
        from pipeline import metrics_suite
        reports = []
        if self.args.filter and self.args.filter != "None":
            reports.append(metrics_suite.enhancement_report(
                self.ctx, self.args.filter, {}, self.args.wc, self.args.ww,
                n_slices=self.args.metric_slices,
                progress=lambda p, m: self.log("  %3d%%  %s" % (p, m))))
        seg = metrics_suite.segmentation_report(
            self.ctx, progress=lambda p, m: self.log("  %3d%%  %s" % (p, m)))
        reports.append(seg)
        if seg.get("agreement"):
            for name, vals in seg["agreement"].items():
                self.log("  %s agreement: DSC %.4f  IoU %.4f"
                         % (name, vals.get("DSC", float("nan")),
                            vals.get("IoU", float("nan"))))
        else:
            self.warn(seg.get("agreement_note", "no agreement metrics"))
        self.summary["metrics"] = {
            "structures": {k: {kk: vv for kk, vv in v.items() if kk != "histogram"}
                           for k, v in seg["structures"].items()},
            "agreement": seg.get("agreement", {}),
            "agreement_note": seg.get("agreement_note"),
        }
        for p in metrics_suite.export_metrics(reports, self.out, self.case_id):
            self.record(p)
        self._seg_report = seg

    # -- outputs ----------------------------------------------------------- #
    def export_meshes(self):
        """Surfaces to disk. Needs no OpenGL, so it always runs."""
        from pipeline import surfaces
        formats = [f.strip().lower() for f in self.args.mesh_format.split(",")
                   if f.strip()]
        built = {}

        for name, mask, in (("aorta", self.ctx.aorta_mask),
                            ("stent", self.ctx.stent_mask)):
            if mask is None:
                continue
            poly = surfaces.build_surface(mask, self.ctx.geom,
                                          self.args.target_triangles)
            if poly is None:
                self.warn("%s mask produced no surface" % name)
                continue
            built[name] = poly
            stats = surfaces.surface_stats(poly)
            self.log("%s surface: %d points, %d triangles"
                     % (name, stats["points"], stats["triangles"]))
            self.summary.setdefault("meshes", {})[name] = stats
            for ext in formats:
                self.record(surfaces.write_mesh(
                    poly, self.path("%s.%s" % (name, ext.lstrip(".")))))

        if self.ctx.resampler is not None:
            # A polyline cannot be stored in STL or PLY, so the centerline is
            # written as a tube for those and as a true polyline for .vtk/.vtp.
            tube = surfaces.tube_polydata(self.ctx.resampler.points,
                                          radius_mm=self.args.tube_radius)
            built["centerline"] = tube
            for ext in formats:
                target = self.path("centerline.%s" % ext.lstrip("."))
                if ext.lstrip(".") in ("vtk", "vtp"):
                    poly = surfaces.centerline_polydata(self.ctx.resampler.points)
                else:
                    poly = tube
                self.record(surfaces.write_mesh(poly, target))
        self._surfaces = built

    def export_figures(self):
        """2-D PNGs via matplotlib Agg. Also needs no OpenGL."""
        from pipeline import report
        wc, ww = self.args.wc, self.args.ww
        try:
            self.record(report.mpr_montage(
                self.ctx, self.path("mpr.png"), wc=wc, ww=ww,
                lumen_hu=self.args.lumen_hu, fov_mm=self.args.fov,
                res_mm=self.args.res))
        except Exception as exc:                               # noqa: BLE001
            self.warn("MPR montage failed: %s" % exc)

        if self.ctx.resampler is not None:
            try:
                self.record(report.cross_section_strip(
                    self.ctx, self.path("cross_sections.png"),
                    n=self.args.strip_sections, wc=wc, ww=ww,
                    lumen_hu=self.args.lumen_hu, fov_mm=self.args.fov,
                    res_mm=self.args.res))
            except Exception as exc:                           # noqa: BLE001
                self.warn("cross-section strip failed: %s" % exc)
        if self.profile_data is not None:
            self.record(report.diameter_profile(
                self.profile_data, self.path("diameter_profile.png"), self.specs))
        seg = getattr(self, "_seg_report", None)
        if seg is not None:
            self.record(report.hu_histograms(seg, self.path("hu_histograms.png")))

    def export_screenshots(self):
        """3-D PNGs. The only step that needs a working GL context."""
        backend = offscreen.backend()
        if backend is None:
            self.warn("no offscreen OpenGL backend; 3-D screenshots skipped")
            self.summary["render_backend"] = None
            return False

        from pipeline import offscreen as off
        self.summary["render_backend"] = backend
        try:
            renderer = off.OffscreenRenderer(
                size=(self.args.width, self.args.height))
        except (SystemError, RuntimeError) as exc:
            # SystemError is what a VTK wrapper raises when its C++ side never
            # came up -- the shape of a GL init failure that did not abort.
            self.warn("could not create an offscreen render window (%s); "
                      "3-D screenshots skipped" % exc)
            return False
        except Exception as exc:                               # noqa: BLE001
            self.warn("offscreen renderer unavailable (%s); 3-D screenshots "
                      "skipped" % exc)
            return False
        try:
            surfaces_built = getattr(self, "_surfaces", {})
            renderer.add_surface("aorta", surfaces_built.get("aorta"), opacity=0.40)
            renderer.add_surface("stent", surfaces_built.get("stent"), opacity=1.0)
            renderer.add_surface("centerline", surfaces_built.get("centerline"),
                                 opacity=1.0)
            renderer.reset_camera(zoom=self.args.zoom)
            shot = renderer.screenshot(self.path("3d.png"))
            if shot is None:
                self.warn("3-D render came back blank; screenshot not written")
                return False
            self.record(shot)
            if self.args.orbit > 1:
                for p in renderer.orbit(self.path("3d_view%d.png"),
                                        n=self.args.orbit,
                                        zoom=self.args.zoom):
                    self.record(p)
            return True
        except (SystemError, RuntimeError) as exc:
            self.warn("3-D rendering failed (%s); screenshots skipped, every "
                      "other output was still written" % exc)
            return False
        finally:
            # Never let teardown turn a completed run into a failed job.
            try:
                renderer.close()
            except Exception as exc:                           # noqa: BLE001
                self.warn("render window cleanup raised (%s); ignored" % exc)

    def export_results(self):
        from pipeline import stent_fit
        try:
            for p in stent_fit.export_results(self.ctx, self.out):
                self.record(p)
        except Exception as exc:                               # noqa: BLE001
            self.warn("result export failed: %s" % exc)

    def write_summary(self, status):
        self.summary["status"] = status
        self.summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.summary["slurm"] = {
            k: os.environ[k] for k in
            ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
             "SLURM_CPUS_ON_NODE", "CUDA_VISIBLE_DEVICES")
            if k in os.environ}
        self.summary["environment_notes"] = offscreen.notes()
        path = os.path.join(self.out, "%s_run_summary.json" % self.case_id)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary, handle, indent=2, default=str)
        self.log("wrote %s" % os.path.basename(path))
        return path


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="run_headless.py",
        description="Headless batch run of the aortic centerline & stent "
                    "pipeline (no X server, no Qt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = p.add_argument_group("input")
    # Not required at the parser level: --check-render is a diagnostic that
    # needs no data, and demanding a volume for it made the documented
    # self-check impossible to run. main() enforces the requirement instead.
    g = src.add_mutually_exclusive_group(required=False)
    g.add_argument("--dicom", metavar="DIR", help="DICOM series directory")
    g.add_argument("--volume", metavar="FILE", help="NIfTI/NRRD/MHA volume")
    src.add_argument("--centerline", metavar="FILE",
                     help="use this centerline (.vtk/.vtp/.csv) instead of "
                          "extracting one")
    src.add_argument("--specs", metavar="FILE",
                     help="stent specification JSON")
    src.add_argument("--reference", metavar="FILE",
                     help="reference aorta mask for Dice/IoU")
    src.add_argument("--case-id", metavar="ID",
                     help="override the case ID used in output filenames")

    out = p.add_argument_group("output")
    out.add_argument("--out", metavar="DIR", default="results",
                     help="output directory")
    out.add_argument("--mesh-format", default="stl,vtk",
                     help="comma-separated: stl, ply, vtk, vtp, obj")
    out.add_argument("--width", type=int, default=1400,
                     help="screenshot width in pixels")
    out.add_argument("--height", type=int, default=1100,
                     help="screenshot height in pixels")
    out.add_argument("--zoom", type=float, default=1.0,
                     help="extra camera zoom after framing (>1 tightens)")
    out.add_argument("--orbit", type=int, default=4,
                     help="extra screenshots around the object (1 = none)")
    out.add_argument("--strip-sections", type=int, default=8,
                     help="cross-sections in the strip figure")
    out.add_argument("--log-file", metavar="FILE", help="also write the log here")
    out.add_argument("--quiet", action="store_true",
                     help="only print warnings and errors")

    par = p.add_argument_group("parameters")
    par.add_argument("--stages", default="all",
                     help="comma-separated subset of: %s. Mesh export, 2-D "
                          "figures and screenshots always run."
                          % ", ".join(STAGES))
    par.add_argument("--resegment", action="store_true",
                     help="segment the volume even when --reference is given "
                          "(needed for a real Dice/IoU: otherwise the working "
                          "mask IS the reference and agreement is 1.0 by "
                          "construction)")
    par.add_argument("--aorta-tol", type=int, default=120)
    par.add_argument("--stent-hu", type=int, default=1200)
    par.add_argument("--stent-dilate", type=int, default=3)
    par.add_argument("--cl-step", type=float, default=1.0)
    par.add_argument("--fov", type=float, default=60.0)
    par.add_argument("--res", type=float, default=0.4)
    par.add_argument("--lumen-hu", type=int, default=200)
    par.add_argument("--wc", type=int, default=300)
    par.add_argument("--ww", type=int, default=600)
    par.add_argument("--filter", default="None",
                     help="enhancement filter for MSE/PSNR/SSIM, e.g. EADTV")
    par.add_argument("--metric-slices", type=int, default=9)
    par.add_argument("--profile-stride", type=int, default=2)
    par.add_argument("--target-triangles", type=int, default=150000)
    par.add_argument("--tube-radius", type=float, default=1.2)

    ren = p.add_argument_group("rendering")
    ren.add_argument("--no-screenshots", action="store_true",
                     help="skip 3-D rendering entirely (meshes still written)")
    ren.add_argument("--no-xvfb", action="store_true",
                     help="do not start Xvfb when probing backends")
    ren.add_argument("--backend", choices=offscreen.BACKENDS,
                     help="force a render backend instead of probing")
    ren.add_argument("--check-render", action="store_true",
                     help="probe the render backends, report, and exit")
    ren.add_argument("--probe-in-process", action="store_true",
                     help="smoke-test render backends inline instead of in a "
                          "child process (faster, but a backend that aborts "
                          "rather than raising then kills the job)")
    ren.add_argument("--exit-mode", choices=("auto", "normal", "hard"),
                     default="auto",
                     help="auto: skip interpreter finalization when a GL "
                          "window was created, so a driver that crashes in its "
                          "destructor cannot turn a finished run into exit "
                          "139; normal: always finalize; hard: always skip")
    return p


def _leave(args, code):
    """Return `code`, or exit immediately without running finalizers.

    Every output file is written and both streams flushed before this is
    reached, so skipping finalization loses nothing. What it avoids is a VTK
    render window crashing inside its destructor during interpreter shutdown:
    the work is already done, but the process dies with SIGSEGV and Slurm
    records exit 139, i.e. a failed job. try/except cannot help there -- a
    segmentation fault never returns control to Python -- so the only reliable
    fix is not to run the destructor at all.

    Only taken when a GL window was actually created; a run with
    --no-screenshots exits normally.
    """
    mode = getattr(args, "exit_mode", "auto")
    if mode == "hard" or (mode == "auto" and offscreen.created_window()):
        offscreen.hard_exit(code)
    return code


def main(argv=None):
    # Environment first: VTK/PyVista/matplotlib read this when imported.
    offscreen.configure_headless(verbose=False)

    args = build_parser().parse_args(argv)
    log = Log(args.log_file, args.quiet)
    for line in offscreen.notes():
        log(line)
    log("headless mode: DISPLAY=%r PYVISTA_OFF_SCREEN=%r"
        % (os.environ.get("DISPLAY"), os.environ.get("PYVISTA_OFF_SCREEN")))

    if args.check_render:
        log("libOSMesa available: %s" % offscreen.has_osmesa())
        backend = offscreen.select_backend(
            allow_xvfb=not args.no_xvfb, verbose=False,
            in_process=args.probe_in_process)
        for line in offscreen.notes():
            log(line)
        log("selected backend: %s" % backend)
        offscreen.safe_cleanup()
        log.close()
        return _leave(args, 0 if backend else 2)

    if not args.dicom and not args.volume:
        log.error("one of --dicom or --volume is required "
                  "(only --check-render may be run without an input)")
        return 1
    if args.dicom and not os.path.isdir(args.dicom):
        log.error("not a directory: %s" % args.dicom)
        return 1
    if args.volume and not os.path.isfile(args.volume):
        log.error("file not found: %s" % args.volume)
        return 1
    if args.reference and not os.path.isfile(args.reference):
        log.error("reference mask not found: %s" % args.reference)
        return 1
    if args.centerline and not os.path.isfile(args.centerline):
        log.error("centerline not found: %s" % args.centerline)
        return 1

    stages = (list(STAGES) if args.stages == "all"
              else [s.strip() for s in args.stages.split(",") if s.strip()])
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        log.error("unknown stage(s): %s" % ", ".join(unknown))
        log.error("  selectable stages: %s" % ", ".join(STAGES))
        log.error("  mesh export, 2-D figures and screenshots are not "
                  "selectable -- they always run on whatever the stages "
                  "produced")
        return 1

    run = HeadlessRun(args, log)
    rendered = False
    try:
        run.stage("load", run.load)
        if "preprocess" in stages:
            run.stage("preprocess", run.preprocess)
        elif run.ctx.reference_mask is not None:
            # preprocess was excluded from --stages, but a reference mask is
            # available; adopt it so the later stages are not starved.
            run.ctx.use_reference_as_aorta()
            run.log("preprocess skipped; using the reference mask as the "
                    "aorta segmentation (%d voxels)"
                    % int(run.ctx.aorta_mask.sum()))
        if "centerline" in stages and run.ctx.aorta_mask is not None:
            run.stage("centerline", run.centerline)
        elif "centerline" in stages:
            run.warn("centerline stage skipped: no aorta mask")
        if "stent" in stages and run.ctx.stent_mask is not None:
            run.stage("stent", run.stent)
        elif "stent" in stages:
            run.warn("stent stage skipped: no stent mask")
        if "profile" in stages and run.ctx.resampler is not None:
            run.stage("profile", run.profile)
        if "metrics" in stages:
            run.stage("metrics", run.metrics)

        run.stage("meshes", run.export_meshes)
        run.stage("figures", run.export_figures)
        run.stage("results", run.export_results)

        if args.no_screenshots:
            run.log("3-D screenshots disabled by --no-screenshots")
            rendered = True          # not a degraded run; it was requested
        else:
            if args.backend:
                offscreen._STATE["backend"] = args.backend
                run.log("render backend forced to %s" % args.backend)
            else:
                offscreen.select_backend(allow_xvfb=not args.no_xvfb,
                                         verbose=False,
                                         in_process=args.probe_in_process)
                for line in offscreen.notes():
                    run.log(line)
            rendered = run.stage("screenshots", run.export_screenshots) or \
                offscreen.backend() is not None

    except Exception as exc:                                   # noqa: BLE001
        log.error("%s: %s" % (type(exc).__name__, exc))
        for line in traceback.format_exc().rstrip().splitlines():
            log.error("  " + line)
        run.write_summary("failed")
        log.close()
        offscreen.safe_cleanup()
        return _leave(args, 1)

    status = "ok" if rendered else "ok-without-screenshots"
    run.write_summary(status)
    log("%d file(s) written to %s" % (len(run.written), run.out))
    log("status: %s" % status)
    log.close()
    offscreen.safe_cleanup()
    return _leave(args, 0 if rendered else 2)


if __name__ == "__main__":
    sys.exit(main())

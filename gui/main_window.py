#!/usr/bin/env python3
"""
MainWindow: layout, worker orchestration, and error presentation.

Threading contract, which the rest of the app depends on:
  * exactly one PipelineWorker runs at a time (self._worker), so a second
    click cannot race the first;
  * workers never touch widgets -- they hand back a result object and the
    slots here apply it on the GUI thread;
  * every stage failure arrives as `failed` and becomes a QMessageBox with the
    exception text, while the full traceback goes to the log panel rather than
    into a dialog nobody can read.

Layout: controls and log on the left, the 3-D view in the centre, and a tabbed
right-hand pane (MPR / Stent / Metrics). The 3-D view is outside the tabs on
purpose -- it is the reference view, and losing sight of it while reading a
landmark coordinate defeats the point of the landmark.
"""

import os
import traceback

from .qt_compat import QtCore, QtGui, QtWidgets, Qt
from .workers import LogBridge, PipelineWorker
from .widgets.control_panel import ControlPanel
from .widgets.log_panel import LogPanel
from .widgets.metrics_panel import MetricsPanel
from .widgets.mpr_panel import MPRPanel
from .widgets.stent_panel import StentPanel
from .widgets.viewer_3d import Viewer3D

from pipeline import paths
from pipeline.context import CaseContext
from pipeline import (io_data, preprocess, centerline as cl_stage, stent_fit,
                      stent_specs, metrics_suite, full_pipeline)

DARK_QSS = """
QWidget { background:#0e1219; color:#c8d3e6; font-size:12px; }
QGroupBox { border:1px solid #2a3347; border-radius:5px; margin-top:14px;
            padding-top:8px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px;
                   color:#7fd1ff; }
QPushButton { background:#1b2333; border:1px solid #2f3b52; border-radius:4px;
              padding:5px 10px; }
QPushButton:hover:!disabled { background:#243049; border-color:#3d7fb8; }
QPushButton:disabled { color:#55607a; background:#151a25; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background:#151a25; border:1px solid #2a3347; border-radius:3px;
    padding:3px 5px; }
QSlider::groove:horizontal { height:4px; background:#2a3347; border-radius:2px; }
QSlider::handle:horizontal { background:#3d7fb8; width:12px; margin:-5px 0;
                             border-radius:6px; }
QProgressBar { border:1px solid #2a3347; border-radius:3px; text-align:center;
               background:#151a25; height:18px; }
QProgressBar::chunk { background:#3d7fb8; border-radius:2px; }
QSplitter::handle { background:#1b2333; }
QScrollArea { border:none; }
QTabWidget::pane { border:1px solid #2a3347; }
QTabBar::tab { background:#151a25; padding:5px 12px; border:1px solid #2a3347; }
QTabBar::tab:selected { background:#243049; color:#7fd1ff; }
QTableWidget { background:#12161f; alternate-background-color:#151a25;
               gridline-color:#2a3347; border:1px solid #2a3347; }
QHeaderView::section { background:#1b2333; color:#7fd1ff; padding:4px;
                       border:none; border-right:1px solid #2a3347; }
QTableWidget::item:selected { background:#28527a; }
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aortic Centerline & Stent Verification")
        self.resize(1720, 1000)

        self.ctx = CaseContext()
        self._worker = None
        self._specs = None            # parsed stent specification JSON
        self._manual_anchors = {}     # operator-picked arc positions
        self._reference_masks = None  # for Dice/IoU against a previous run

        self.log = LogPanel()
        self.controls = ControlPanel()
        self.viewer = Viewer3D()
        self.mpr = MPRPanel()
        self.stent = StentPanel()
        self.metrics = MetricsPanel()

        self._build_layout()
        self._connect()
        self._install_logging()

        self.statusBar().showMessage("Ready")
        self.log.ok("Aortic stent GUI ready.")
        self._report_environment()

    # ------------------------------------------------------------------ init
    def _build_layout(self):
        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(self.controls, 3)
        left_lay.addWidget(self.log, 2)
        left.setMinimumWidth(340)
        left.setMaximumWidth(480)

        view3d = QtWidgets.QGroupBox("3-D view")
        v3 = QtWidgets.QVBoxLayout(view3d)
        v3.setContentsMargins(2, 12, 2, 2)
        v3.addWidget(self.viewer)
        bar = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("Reset camera")
        reset.clicked.connect(self.viewer.reset_camera)
        bar.addWidget(reset)
        shot = QtWidgets.QPushButton("Save 3-D snapshot…")
        shot.clicked.connect(self._save_snapshot)
        bar.addWidget(shot)
        bar.addStretch(1)
        v3.addLayout(bar)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.mpr, "2-D MPR")
        self.tabs.addTab(self.stent, "Stent")
        self.tabs.addTab(self.metrics, "Metrics")

        right = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        right.addWidget(view3d)
        right.addWidget(self.tabs)
        right.setSizes([700, 780])

        main = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(left)
        main.addWidget(right)
        main.setSizes([380, 1340])
        self.setCentralWidget(main)

        # Live coordinate strip, permanent so a transient status message
        # cannot overwrite it.
        self.coord_label = QtWidgets.QLabel("—")
        self.coord_label.setStyleSheet(
            "color:#7fd1ff; font-family:Consolas,monospace; padding-right:8px;")
        self.statusBar().addPermanentWidget(self.coord_label)

        self._build_menu()

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for label, kind in (("Open DICOM directory…", "dicom"),
                            ("Open volume file…", "image"),
                            ("Open surface mesh…", "mesh"),
                            ("Open centerline…", "centerline")):
            act = QtGui.QAction(label, self)
            act.triggered.connect(lambda _c=False, k=kind: self._browse_and_load(k))
            m.addAction(act)
        m.addSeparator()
        act = QtGui.QAction("Load reference segmentation…", self)
        act.triggered.connect(self.on_load_reference)
        m.addAction(act)
        act = QtGui.QAction("Load stent specs JSON…", self)
        act.triggered.connect(self.on_load_specs)
        m.addAction(act)
        m.addSeparator()
        act = QtGui.QAction("Export results…", self)
        act.triggered.connect(self.on_export)
        m.addAction(act)
        act = QtGui.QAction("Export metrics…", self)
        act.triggered.connect(self.on_export_metrics)
        m.addAction(act)
        m.addSeparator()
        act = QtGui.QAction("Quit", self)
        act.triggered.connect(self.close)
        m.addAction(act)

        h = self.menuBar().addMenu("&Help")
        act = QtGui.QAction("Environment report", self)
        act.triggered.connect(self._report_environment)
        h.addAction(act)

    def _connect(self):
        c = self.controls
        c.load_requested.connect(self.on_load)
        c.preprocess_requested.connect(self.on_preprocess)
        c.centerline_requested.connect(self.on_centerline)
        c.stent_requested.connect(self.on_stent)
        c.export_requested.connect(self.on_export)
        c.cancel_requested.connect(self.on_cancel)
        c.display_changed.connect(self.on_display_changed)
        c.apply_thresholds.connect(self.on_apply_thresholds)
        c.recompute_requested.connect(self.on_recompute)
        c.profile_requested.connect(self.on_profile)
        c.full_pipeline_requested.connect(self.on_full_pipeline)

        self.mpr.station_changed.connect(self._on_station_changed)
        self.mpr.cursor_info.connect(self._on_cursor_info)
        self.mpr.cursor_picked.connect(self._on_cursor_picked)
        self.viewer.cursor_info.connect(self._on_cursor_info)

        s = self.stent
        s.specs_requested.connect(self.on_load_specs)
        s.template_requested.connect(self.on_save_template)
        s.compute_requested.connect(self.on_landmarks)
        s.export_requested.connect(self.on_export_landmarks)
        s.pick_requested.connect(self.on_pick_anchor)
        s.station_requested.connect(self.on_goto_station)

        self.metrics.enhancement_requested.connect(self.on_enhancement_metrics)
        self.metrics.segmentation_requested.connect(self.on_segmentation_metrics)
        self.metrics.export_requested.connect(self.on_export_metrics)

    def _install_logging(self):
        self.bridge = LogBridge()
        self.bridge.message.connect(self.log.append)
        self.bridge.install()

    def _report_environment(self):
        from .qt_compat import QT_API
        self.log.info("Qt binding: %s" % QT_API)
        for name, mod in (("SimpleITK", "SimpleITK"), ("VTK", "vtk"),
                          ("scikit-image", "skimage"), ("SciPy", "scipy"),
                          ("OpenCV", "cv2")):
            try:
                m = __import__(mod)
                self.log.info("  %-13s %s" % (name, getattr(m, "__version__", "?")))
            except ImportError:
                self.log.warn("  %-13s not installed" % name)

    # ----------------------------------------------------------------- helpers
    def _busy(self, busy, message=""):
        self.controls.set_busy(busy)
        if busy:
            self.statusBar().showMessage(message or "Working…")
            QtWidgets.QApplication.setOverrideCursor(
                QtGui.QCursor(Qt.CursorShape.BusyCursor))
        else:
            self.statusBar().showMessage("Ready")
            QtWidgets.QApplication.restoreOverrideCursor()

    def _run(self, name, fn, *args, on_done=None, **kwargs):
        """Start one background task. Refuses to start a second."""
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Busy",
                "'%s' is still running.\n\nWait for it to finish, or press "
                "Cancel running task." % self._worker.name)
            return None

        self.log.command("▶ %s" % name)
        self.log.reset_progress(name)
        self._busy(True, name)

        worker = PipelineWorker(name, fn, *args, **kwargs)
        worker.progress.connect(self.log.set_progress)
        worker.progress.connect(self._log_stage_progress)
        worker.failed.connect(self._on_failed)
        worker.finished_ok.connect(
            lambda n, r, cb=on_done: self._on_done(n, r, cb))
        self._worker = worker
        worker.start()
        return worker

    def _on_done(self, name, result, callback):
        self._busy(False)
        self.log.set_progress(100, "%s — done" % name)
        self.log.ok("✔ %s finished." % name)
        if callback:
            try:
                callback(result)
            except Exception as exc:                           # noqa: BLE001
                self._show_error("Displaying results failed", exc,
                                 traceback.format_exc())

    def _log_stage_progress(self, _pct, message):
        # The progress bar already shows every update; only a stage boundary
        # (run_full_pipeline's "Step N/4: ..." messages) is worth a permanent
        # line in the scrolling log too.
        if message.startswith("Step "):
            self.log.info(message)

    def _on_failed(self, name, message, tb):
        self._busy(False)
        self.log.reset_progress("failed")
        self.log.error("✖ %s failed: %s" % (name, message))
        if tb:
            for line in tb.rstrip().splitlines():
                self.log.error("   " + line)
        if message.strip() == "Cancelled.":
            self.statusBar().showMessage("Cancelled")
            return
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle("%s failed" % name)
        box.setText(message)
        box.setInformativeText("The full traceback is in the status log.")
        if tb:
            box.setDetailedText(tb)
        box.exec()

    def _show_error(self, title, exc, tb=""):
        self.log.error("%s: %s" % (title, exc))
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(str(exc))
        if tb:
            box.setDetailedText(tb)
        box.exec()

    def _warn(self, title, text):
        QtWidgets.QMessageBox.warning(self, title, text)

    def _display_params(self):
        p = self.controls.params()
        p["show_aorta"] = self.controls.show_aorta.isChecked()
        p["show_stent"] = self.controls.show_stent.isChecked()
        p["show_centerline"] = self.controls.show_centerline.isChecked()
        return p

    def _on_cursor_info(self, text):
        self.coord_label.setText(text)

    def _on_cursor_picked(self, z, y, x):
        self.log.info("Picked voxel [%d, %d, %d] — HU %d"
                      % (x, y, z, int(self.ctx.volume[z, y, x])))

    # -------------------------------------------------------------- loading
    def _browse_and_load(self, kind):
        if kind == "dicom":
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select DICOM series directory")
        elif kind == "mesh":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select surface mesh", "",
                "Meshes (*.stl *.vtk *.vtp *.ply *.obj);;All files (*)")
        elif kind == "centerline":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select centerline", "",
                "Centerline (*.vtk *.vtp *.csv *.json);;All files (*)")
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select volume", "",
                "Images (*.nii *.nii.gz *.nrrd *.mha);;All files (*)")
        if path:
            self.on_load(kind, path)

    def on_load(self, kind, path):
        if not path:
            self._warn("No path", "Choose a %s first."
                       % ("directory" if kind == "dicom" else "file"))
            return
        if not os.path.exists(path):
            self._warn("Not found", "This path does not exist:\n\n%s" % path)
            return

        if kind == "dicom":
            self._run("Load DICOM series", io_data.load_dicom_series, path,
                      on_done=lambda r: self._apply_volume(r, path, "dicom"))
        elif kind == "image":
            self._run("Load volume", io_data.load_image_file, path,
                      on_done=lambda r: self._apply_volume(r, path, "image"))
        elif kind == "mesh":
            self._run("Load mesh", io_data.load_surface_mesh, path,
                      on_done=lambda r: self._apply_mesh(r, path))
        elif kind == "centerline":
            self._run("Load centerline", io_data.load_centerline_file, path,
                      on_done=lambda r: self._apply_centerline_file(r, path))

    def _apply_volume(self, result, path, kind):
        self.ctx = CaseContext(
            case_id=result["case_id"], source_path=path, source_kind=kind,
            volume=result["volume"], spacing_zyx=result["spacing_zyx"],
            geom=result["geom"])
        self._manual_anchors = {}
        self._reference_masks = None
        self.log.ok(self.ctx.describe())
        self.viewer.clear()
        self.viewer.set_probe_volume(self.ctx.volume, self.ctx.geom)
        self.mpr.set_context(self.ctx, self._display_params())
        self.stent.set_table(None)
        self.controls.set_volume_loaded(True)
        self.setWindowTitle("Aortic Centerline & Stent Verification — %s"
                            % self.ctx.case_id)

        params = self.controls.params()
        if params.get("auto_render", True):
            # Queued so the 2-D views paint first; the preview is a courtesy,
            # and the operator should not wait on it to see the slices.
            QtCore.QTimer.singleShot(0, self._auto_preview)

    def _auto_preview(self):
        hu = self.controls.params().get("preview_hu", 150)
        self._run("Preview surface", preprocess.initial_surface, self.ctx,
                  hu=hu, on_done=self._apply_preview)

    def _apply_preview(self, result):
        if not result:
            self.log.warn("Preview surface: nothing above %d HU — lower the "
                          "Preview HU value." % self.controls.preview_hu.value())
            return
        self.viewer.set_mask_surface("preview", result["mask"], result["geom"],
                                     opacity=0.35, color=(0.62, 0.66, 0.78))
        self.viewer.reset_camera()
        self.log.info("Preview surface at %d HU (stride %d)."
                      % (result["hu"], result["stride"]))

    def _apply_mesh(self, result, path):
        verts, faces = result.get("vertices"), result.get("faces")
        if faces is None:
            self.log.warn("%s holds no triangles (it may be a polyline); "
                          "loading it as a centerline instead."
                          % os.path.basename(path))
            lines = result.get("lines") or []
            if lines:
                import numpy as np
                verts = verts[np.asarray(max(lines, key=len), np.int64)]
            self._apply_centerline_file(verts, path)
            return
        self.ctx.surface_mesh = (verts, faces)
        self.viewer.set_visible("preview", False)
        self.viewer.set_surface_mesh("aorta", verts, faces, opacity=0.45)
        self.viewer.reset_camera()
        self.log.ok("Mesh: %d vertices, %d triangles" % (len(verts), len(faces)))

    def _apply_centerline_file(self, points, path):
        import numpy as np
        import oblique_mpr as om
        pts = np.asarray(points, float).reshape(-1, 3)
        if len(pts) < 2:
            raise ValueError("Centerline needs at least 2 points.")
        step = self.controls.cl_step.value()
        self.ctx.centerline = pts
        self.ctx.resampler = om.CenterlineResampler(pts, step_mm=step)
        self.log.ok("Centerline from %s: %d points, %.1f mm"
                    % (os.path.basename(path), len(pts),
                       self.ctx.resampler.total_length_mm))
        self._refresh_views()

    # ---------------------------------------------------------- full pipeline
    def on_full_pipeline(self):
        """Raw volume -> segmentation -> centerline -> meshes -> stent JSON.

        One background task chaining the same four stages the individual
        buttons run, so a case that has nothing precomputed yet needs one
        click instead of four. Uses whatever thresholds are currently set in
        the Segmentation / Centerline groups, so adjusting them before
        pressing this has the same effect as before Recompute.
        """
        if not self.ctx.has_volume:
            self._warn("No volume", "Load a DICOM series or a volume file "
                       "first — the full pipeline starts from the raw "
                       "volume.")
            return
        params = self.controls.params()
        out_dir = paths.ensure_output_dir(
            "full_pipeline", self.ctx.case_id or "case")
        self.log.info("Full pipeline output: %s" % out_dir)
        self._reference_masks = metrics_suite.snapshot_masks(
            self.ctx, "previous segmentation")
        self._run("Run Full Pipeline", full_pipeline.run_full_pipeline,
                  self.ctx, params, out_dir, specs=self._specs,
                  on_done=self._after_full_pipeline)

    def _after_full_pipeline(self, result):
        self.log.ok("Full pipeline: %d file(s) written to %s"
                    % (len(result["written"]), result["out_dir"]))
        for f in result["written"]:
            self.log.info("  wrote %s" % os.path.basename(f))

        # Point the manual-override fields at what was just generated, so the
        # operator can see -- and later reload -- exactly what the pipeline
        # wrote, without disturbing the in-memory results already applied
        # below.
        case = result["case_id"]
        mesh_path = os.path.join(result["out_dir"], "%s_aorta.stl" % case)
        if os.path.exists(mesh_path):
            self.controls.mesh_row.set_path(mesh_path)
        centerline_path = os.path.join(result["out_dir"],
                                       "%s_centerline.vtk" % case)
        if os.path.exists(centerline_path):
            self.controls.centerline_row.set_path(centerline_path)

        stent_result = result.get("stent_result")
        if stent_result:
            self.log.ok("Stent length %.1f mm (straight-line), volume %.0f mm³"
                        % (stent_result.get("length_mm", 0.0),
                           stent_result.get("volume_mm3", 0.0)))
            for w in stent_result.get("warnings", []):
                self.log.warn("  " + w)

        table = result.get("landmarks")
        if table:
            self.stent.set_table(table)
            self.stent.set_anchor_note(table["summary"].get("anchor", "?"))
            self.log.ok("Landmarks: %d rows, anchored on %s."
                        % (len(table["rows"]), table["summary"].get("anchor")))
            for w in table["summary"].get("warnings", []):
                self.log.warn("  " + w)

        self.mpr.set_context(self.ctx, self._display_params())
        self._refresh_views(reset_camera=True)
        self.tabs.setCurrentWidget(self.stent if table else self.mpr)

    # -------------------------------------------------------------- stages
    def on_preprocess(self):
        if not self.ctx.has_volume:
            self._warn("No volume", "Load a DICOM series or a volume file first.")
            return
        # Keep the outgoing masks so the next segmentation report can quantify
        # how much the thresholds moved the result.
        self._reference_masks = metrics_suite.snapshot_masks(
            self.ctx, "previous segmentation")
        params = self.controls.params()
        self._run("Run Preprocessing", preprocess.run_preprocessing,
                  self.ctx, params, on_done=lambda _r: self._after_preprocess())

    def on_recompute(self):
        """Re-segment with the current thresholds and rebuild the 3-D surfaces."""
        if not self.ctx.has_volume:
            self._warn("No volume", "Load a DICOM series or a volume file first.")
            return
        self.log.info("Recompute: aorta tol %d · stent %d HU · dilate %d"
                      % (self.controls.aorta_tol.value(),
                         self.controls.stent_hu.value(),
                         self.controls.stent_dilate.value()))
        self.on_preprocess()

    def _after_preprocess(self):
        m = self.ctx.metrics
        self.log.ok("Aorta %d vox (%.1f mL) · stent %d vox"
                    % (m.get("aorta_voxels", 0), m.get("aorta_volume_ml", 0.0),
                       m.get("stent_voxels", 0)))
        # The coarse preview has served its purpose once a real mask exists.
        self.viewer.set_visible("preview", False)
        self._refresh_views(reset_camera=True)

    def on_centerline(self):
        if self.ctx.aorta_mask is None:
            self._warn("No aorta mask",
                       "Run Preprocessing first.\n\nThe centerline is the "
                       "skeleton of the segmented aorta, so it needs that mask.")
            return
        p = self.controls.params()
        self._run("Extract Centerline", cl_stage.extract_centerline,
                  self.ctx, step_mm=p["cl_step"], orient=p["cl_orient"],
                  on_done=lambda _r: self._after_centerline())

    def _after_centerline(self):
        m = self.ctx.metrics
        self.log.ok("Centerline: %d points, %.1f mm (orientation: %s)"
                    % (m.get("centerline_points", 0),
                       m.get("centerline_length_mm", 0.0),
                       m.get("centerline_orientation_rule", "?")))
        self.mpr.set_context(self.ctx, self._display_params())
        self._refresh_views(reset_camera=True)

    def on_stent(self):
        if self.ctx.stent_mask is None:
            self._warn("No stent mask",
                       "Run Preprocessing first.\n\nIf it still finds no "
                       "metal, lower the Stent HU threshold — a bare-metal "
                       "frame can sit well below the 1200 HU default on a "
                       "low-kVp series.")
            return
        self._run("Fit Stent", stent_fit.fit_stent, self.ctx,
                  on_done=lambda r: self._after_stent(r))

    def _after_stent(self, result):
        self.log.ok("Stent length %.1f mm (straight-line), volume %.0f mm³"
                    % (result.get("length_mm", 0.0),
                       result.get("volume_mm3", 0.0)))
        if "along_vessel_length_mm" in result:
            self.log.info("  along vessel %.1f mm · covers %.0f%% of the "
                          "centerline · tortuosity %.2f"
                          % (result["along_vessel_length_mm"],
                             100.0 * result.get("covered_fraction", 0.0),
                             result.get("tortuosity", 1.0)))
        for w in result.get("warnings", []):
            self.log.warn("  " + w)
        self._refresh_views()
        if self.ctx.resampler is not None:
            self.on_landmarks(quiet=True)

    def on_profile(self):
        if self.ctx.resampler is None:
            self._warn("No centerline", "Run Extract Centerline first.")
            return
        p = self.controls.params()
        self._run("Diameter profile", cl_stage.diameter_profile, self.ctx,
                  fov_mm=p["cl_fov"], res_mm=max(0.5, p["cl_res"]),
                  lumen_hu=p["lumen_hu"], stride=2,
                  on_done=self._after_profile)

    def _after_profile(self, profile):
        m = self.ctx.metrics
        self.log.ok("Lumen diameter along vessel: mean %.1f mm, "
                    "range %.1f–%.1f mm"
                    % (m.get("diameter_mean_mm", 0.0),
                       m.get("diameter_min_mm", 0.0),
                       m.get("diameter_max_mm", 0.0)))
        n_multi = int(profile["multi_lumen"].sum())
        if n_multi:
            self.log.info("  %d of %d stations had the plane crossing the "
                          "vessel more than once (tight bends); the centre "
                          "lumen was measured at each."
                          % (n_multi, len(profile["multi_lumen"])))
        self.mpr.set_profile(profile)

    # --------------------------------------------------------- stent specs
    def on_load_specs(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Upload stent specs JSON", "", "JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            self._specs = stent_specs.load_specs(path)
        except stent_specs.SpecError as exc:
            self._show_error("Stent specs rejected", exc)
            return
        self.stent.set_specs(self._specs)
        self.log.ok("Stent specs: %s — ⌀%.1f mm, %.0f mm, landing %.0f/%.0f mm"
                    % (self._specs.get("device", os.path.basename(path)),
                       self._specs["diameter_mm"],
                       self._specs["target_length_mm"],
                       self._specs["proximal_land_mm"],
                       self._specs["distal_land_mm"]))
        self.tabs.setCurrentWidget(self.stent)
        if self.ctx.resampler is not None:
            self.on_landmarks(quiet=True)

    def on_load_reference(self):
        """Load a pre-computed segmentation as reference / working aorta mask.

        Serves two purposes at once, which is why the provenance is tracked:
        it gives Dice/IoU something to compare against, and it lets a case
        whose automatic seed search fails still be measured, meshed and
        reported. Which of the two applies depends on the "Re-segment" toggle,
        and the metrics report says which it used.
        """
        if not self.ctx.has_volume:
            self._warn("No volume", "Load the CT volume first — the reference "
                                    "mask has to match its shape.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load reference segmentation", "",
            "Masks (*.nii *.nii.gz *.nrrd *.mha);;All files (*)")
        if not path:
            return
        try:
            import SimpleITK as sitk
            import numpy as np
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            arr = arr.astype(bool)
        except Exception as exc:                               # noqa: BLE001
            self._show_error("Could not read the reference mask", exc,
                             traceback.format_exc())
            return
        if arr.shape != self.ctx.volume.shape:
            self._warn("Shape mismatch",
                       "The reference mask is %s but this case is %s.\n\n"
                       "They must come from the same volume."
                       % (arr.shape, self.ctx.volume.shape))
            return

        self.ctx.reference_mask = arr
        self.ctx.reference_name = os.path.basename(path)
        if self.ctx.aorta_mask is None:
            self.ctx.use_reference_as_aorta()
            self.log.ok("Reference %s: %d voxels — adopted as the aorta mask."
                        % (self.ctx.reference_name, int(arr.sum())))
        else:
            self.log.ok("Reference %s: %d voxels — will be used for "
                        "Dice/IoU." % (self.ctx.reference_name, int(arr.sum())))
        self._refresh_views(reset_camera=True)

    def on_save_template(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save stent specs template", "stent_specs.json",
            "JSON (*.json)")
        if not path:
            return
        try:
            stent_specs.write_template(path)
            self.log.ok("Wrote template %s" % path)
        except Exception as exc:                               # noqa: BLE001
            self._show_error("Could not write template", exc,
                             traceback.format_exc())

    def on_pick_anchor(self, which):
        if which == "clear":
            self._manual_anchors = {}
            self.stent.set_anchor_note("fitted stent crowns")
            self.log.info("Manual stent anchors cleared.")
            return
        if self.ctx.resampler is None:
            self._warn("No centerline", "Run Extract Centerline first.")
            return
        arc = float(self.ctx.resampler.arclength[
            min(self.mpr.station(), len(self.ctx.resampler) - 1)])
        self._manual_anchors["%s_arc_mm" % which] = arc
        self.log.ok("%s anchor set at %.1f mm along the centerline."
                    % (which.capitalize(), arc))
        have = [k.split("_")[0] for k in self._manual_anchors]
        self.stent.set_anchor_note(
            "manual (%s)" % ", ".join(sorted(have))
            if len(have) < 2 else "manual picks")
        if len(self._manual_anchors) == 2:
            self.on_landmarks(quiet=True)

    def on_landmarks(self, quiet=False):
        if self.ctx.resampler is None:
            if not quiet:
                self._warn("No centerline",
                           "Run Extract Centerline first — landmarks are "
                           "positions along it.")
            return
        if self._specs is None and not quiet:
            self._warn(
                "No specification",
                "Upload a stent specs JSON first.\n\nLanding-zone landmarks "
                "need proximal_land_mm and distal_land_mm; press "
                "'Save template…' for a file with every expected field.")
            return
        try:
            table = stent_specs.landmark_table(
                self.ctx, self._specs,
                lumen_hu=self.controls.lumen_hu.value(),
                manual=self._manual_anchors)
        except stent_specs.SpecError as exc:
            if not quiet:
                self._show_error("Landmarks failed", exc)
            return
        except Exception as exc:                               # noqa: BLE001
            self._show_error("Landmarks failed", exc, traceback.format_exc())
            return

        self.stent.set_table(table)
        self.stent.set_anchor_note(table["summary"].get("anchor", "?"))
        self.log.ok("Landmarks: %d rows, anchored on %s."
                    % (len(table["rows"]), table["summary"].get("anchor")))
        for w in table["summary"].get("warnings", []):
            self.log.warn("  " + w)

    def on_export_landmarks(self):
        table = self.ctx.metrics.get("stent_landmarks")
        if not table:
            self._warn("Nothing to export", "Compute landmarks first.")
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export landmarks to…", paths.ensure_output_dir())
        if not out:
            return
        try:
            files = stent_specs.export_landmarks(table, out,
                                                 self.ctx.case_id)
        except Exception as exc:                               # noqa: BLE001
            self._show_error("Export failed", exc, traceback.format_exc())
            return
        for f in files:
            self.log.info("  wrote %s" % os.path.basename(f))
        QtWidgets.QMessageBox.information(
            self, "Export complete",
            "Wrote %d file(s) to:\n\n%s" % (len(files), out))

    def on_goto_station(self, index):
        if self.ctx.resampler is None:
            return
        self.mpr.slider.setValue(int(index))

    # ------------------------------------------------------------- metrics
    def on_enhancement_metrics(self):
        if not self.ctx.has_volume:
            self._warn("No volume", "Load a volume first.")
            return
        name, params = self.controls.filter_settings()
        p = self.controls.params()
        self._run("Enhancement metrics", metrics_suite.enhancement_report,
                  self.ctx, name, params, p["wc"], p["ww"],
                  on_done=self._after_enhancement)

    def _after_enhancement(self, report):
        self.metrics.set_enhancement(report)
        s = report.get("summary", {})
        self.log.ok("%s over %d slices: PSNR %.2f dB · SSIM %.4f · MSE %.3f"
                    % (report.get("filter"), report.get("n_slices", 0),
                       s.get("PSNR", {}).get("mean", float("nan")),
                       s.get("SSIM", {}).get("mean", float("nan")),
                       s.get("MSE", {}).get("mean", float("nan"))))
        self.tabs.setCurrentWidget(self.metrics)

    def on_segmentation_metrics(self):
        if not self.ctx.has_volume:
            self._warn("No volume", "Load a volume first.")
            return
        self._run("Segmentation metrics", metrics_suite.segmentation_report,
                  self.ctx, self._reference_masks,
                  on_done=self._after_segmentation)

    def _after_segmentation(self, report):
        self.metrics.set_segmentation(report)
        for name, entry in report.get("structures", {}).items():
            self.log.info("  %-13s %8d vox · %6.1f mL · HU %.0f ± %.0f"
                          % (name, entry["voxels"], entry["volume_ml"],
                             entry["hu_mean"], entry["hu_sd"]))
        if report.get("agreement"):
            for name, vals in report["agreement"].items():
                self.log.ok("  %s agreement: DSC %.4f · IoU %.4f"
                            % (name, vals.get("DSC", float("nan")),
                               vals.get("IoU", float("nan"))))
            self.log.info("  " + report.get("agreement_note", ""))
        else:
            self.log.warn("  " + report.get("agreement_note", ""))
        self.tabs.setCurrentWidget(self.metrics)

    def on_export_metrics(self):
        reports = [self.ctx.metrics.get("enhancement"),
                   self.ctx.metrics.get("segmentation")]
        if not any(reports):
            self._warn("Nothing to export",
                       "Compute enhancement or segmentation metrics first.")
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export metrics to…", paths.ensure_output_dir())
        if not out:
            return
        self._run("Export metrics", metrics_suite.export_metrics, reports,
                  out, self.ctx.case_id,
                  on_done=lambda files: self._after_export(files, out))

    # -------------------------------------------------------------- export
    def on_export(self):
        if not self.ctx.has_volume:
            self._warn("Nothing to export", "Load a case first.")
            return
        default = paths.ensure_output_dir("gui_exports",
                                          self.ctx.case_id or "case")
        out = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export results to…", default)
        if not out:
            return
        self._run("Export Results", stent_fit.export_results, self.ctx, out,
                  on_done=lambda files: self._after_export(files, out))

    def _after_export(self, files, out):
        for f in files:
            self.log.info("  wrote %s" % os.path.basename(f))
        QtWidgets.QMessageBox.information(
            self, "Export complete",
            "Wrote %d file(s) to:\n\n%s" % (len(files), out))

    def on_cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self.log.warn("Cancel requested — the stage will stop at its next "
                          "checkpoint.")
            self._worker.request_stop()

    # -------------------------------------------------------------- display
    def on_display_changed(self):
        params = self._display_params()
        self.mpr.set_params(params)
        self.viewer.set_visible("aorta", params["show_aorta"])
        self.viewer.set_visible("stent", params["show_stent"])
        self.viewer.set_visible("centerline", params["show_centerline"])

    def on_apply_thresholds(self):
        """Redraw the 2-D views with the current window / lumen threshold.

        Deliberately does not re-segment: that is what Recompute is for.
        """
        params = self._display_params()
        self.log.info("Applied window %d/%d HU · lumen threshold %d HU"
                      % (params["wc"], params["ww"], params["lumen_hu"]))
        self.mpr.set_params(params)

    def _on_station_changed(self, index):
        if self.ctx.resampler is not None and 0 <= index < len(self.ctx.resampler):
            self.viewer.set_marker(self.ctx.resampler.points[index])

    def _refresh_views(self, reset_camera=False):
        params = self._display_params()

        # The coarse load-time preview is superseded the moment a real
        # segmentation exists, and leaving it on does active harm: it is a
        # raw HU isosurface, so on a contrast study it wraps the vessel in
        # bone and table, and ResetCamera then frames *that* rather than the
        # anatomy. Retiring it here rather than in each handler means every
        # path that produces a mask -- Preprocess, Recompute, Run Full
        # Pipeline -- gets it, which is what the full-pipeline path missed.
        if self.ctx.aorta_mask is not None:
            self.viewer.set_visible("preview", False)

        try:
            if self.ctx.aorta_mask is not None:
                self.viewer.set_mask_surface("aorta", self.ctx.aorta_mask,
                                             self.ctx.geom, opacity=0.40)
            if self.ctx.stent_mask is not None:
                self.viewer.set_mask_surface("stent", self.ctx.stent_mask,
                                             self.ctx.geom, opacity=1.0)
            if self.ctx.resampler is not None:
                self.viewer.set_centerline(self.ctx.resampler.points)
                self.viewer.set_marker(
                    self.ctx.resampler.points[
                        min(self.mpr.station(), len(self.ctx.resampler) - 1)])
        except Exception as exc:                               # noqa: BLE001
            self.log.error("3-D update failed: %s" % exc)

        self.viewer.set_visible("aorta", params["show_aorta"])
        self.viewer.set_visible("stent", params["show_stent"])
        self.viewer.set_visible("centerline", params["show_centerline"])
        if reset_camera:
            self.viewer.reset_camera()
        self.mpr.set_context(self.ctx, params)

    def _save_snapshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 3-D snapshot", "%s_3d.png" % (self.ctx.case_id or "view"),
            "PNG image (*.png)")
        if not path:
            return
        try:
            self.viewer.screenshot(path)
            self.log.ok("Saved %s" % path)
        except Exception as exc:                               # noqa: BLE001
            self._show_error("Snapshot failed", exc, traceback.format_exc())

    # ----------------------------------------------------------------- close
    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            answer = QtWidgets.QMessageBox.question(
                self, "Task running",
                "'%s' is still running.\n\nQuit anyway?" % self._worker.name,
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.request_stop()
            self._worker.wait(3000)
        try:
            self.bridge.uninstall()
        except Exception:                                      # noqa: BLE001
            pass
        # Release every GL resource before Qt tears the widgets down. An
        # interactor that outlives its widget, or a PyVista plotter left for
        # the garbage collector, crashes during interpreter shutdown -- after
        # the user has already closed the window and believes they are done.
        try:
            self.viewer.close_viewer()
        except Exception as exc:                               # noqa: BLE001
            self.log.warn("3-D viewer cleanup raised (%s); ignored" % exc)
        try:
            from pipeline import offscreen
            offscreen.safe_cleanup()
        except Exception:                                      # noqa: BLE001
            pass
        event.accept()

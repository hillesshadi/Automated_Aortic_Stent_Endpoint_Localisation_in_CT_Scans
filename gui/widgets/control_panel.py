#!/usr/bin/env python3
"""
Left-hand control panel: file selectors, parameters, and the stage buttons.

Two things here are deliberate rather than incidental:

  * The filter list and its per-filter parameter spin boxes are built from
    enhancement.FILTER_PARAMS at construction time rather than hard-coded, so a
    filter added to that registry appears here with correctly ranged controls
    without this file changing.

  * HU controls do not apply themselves. Window centre/width, the aorta
    tolerance, the stent threshold and the lumen threshold all sit behind
    "Apply Thresholds" (redraw the 2-D views) and "Recompute Segmentation"
    (re-run segmentation and rebuild the 3-D surfaces). Re-segmenting a volume
    on every keystroke of a spin box is both slow and misleading -- you cannot
    tell which value produced what you are looking at. Pending edits are
    signalled by enabling the buttons, so the deferred model stays visible.
"""

import os

from ..qt_compat import QtWidgets, Qt, Signal
from pipeline.preprocess import available_filters

WL_PRESETS = {
    "CT Angio (aorta)": (300, 600),
    "Soft tissue": (40, 400),
    "Lung": (-600, 1500),
    "Bone": (300, 1500),
    "Stent / metal": (1500, 2500),
}
PENDING_QSS = ("QPushButton{background:#3d5a2a;border-color:#6ea83d;"
               "color:#dff5c8;font-weight:600;}")


class _FileRow(QtWidgets.QWidget):
    """A labelled path field with a Browse button."""

    changed = Signal(str)

    def __init__(self, label, mode, filters="", parent=None):
        super().__init__(parent)
        self.mode = mode                      # "dir" | "file"
        self.filters = filters
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(QtWidgets.QLabel(label))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("(none selected)")
        self.edit.editingFinished.connect(
            lambda: self.changed.emit(self.edit.text().strip()))
        row.addWidget(self.edit, 1)
        btn = QtWidgets.QPushButton("Browse…")
        btn.setFixedWidth(78)
        btn.clicked.connect(self._browse)
        row.addWidget(btn)
        lay.addLayout(row)

    def _browse(self):
        start = self.edit.text().strip() or os.getcwd()
        if self.mode == "dir":
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select DICOM series directory", start)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select file", start, self.filters)
        if path:
            self.edit.setText(path)
            self.changed.emit(path)

    def path(self):
        return self.edit.text().strip()

    def set_path(self, value):
        self.edit.setText(value or "")


FULL_PIPELINE_QSS = (
    "QPushButton{background:#1f6b3d;border:1px solid #3fae6f;color:#eafff2;"
    "font-weight:700;} QPushButton:hover:!disabled{background:#268049;"
    "border-color:#4fd18a;} QPushButton:disabled{color:#55607a;"
    "background:#151a25;border-color:#2a3347;}")


class ControlPanel(QtWidgets.QScrollArea):
    load_requested = Signal(str, str)     # (kind, path)
    preprocess_requested = Signal()
    centerline_requested = Signal()
    stent_requested = Signal()
    export_requested = Signal()
    cancel_requested = Signal()
    display_changed = Signal()            # cheap: overlay visibility only
    apply_thresholds = Signal()           # redraw 2-D with new window/threshold
    recompute_requested = Signal()        # re-segment and rebuild 3-D surfaces
    profile_requested = Signal()
    full_pipeline_requested = Signal()    # raw volume -> every derived artifact

    def __init__(self, parent=None):
        super().__init__(parent)
        self._volume_loaded = False
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        self.setWidget(inner)
        lay = QtWidgets.QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        lay.addWidget(self._data_group())
        lay.addWidget(self._display_group())
        lay.addWidget(self._segmentation_group())
        lay.addWidget(self._filter_group())
        lay.addWidget(self._centerline_group())
        lay.addWidget(self._actions_group())
        lay.addStretch(1)
        self._set_pending(False)

    # ------------------------------------------------------------------ data
    def _data_group(self):
        box = QtWidgets.QGroupBox("1 · Data")
        lay = QtWidgets.QVBoxLayout(box)
        lay.setSpacing(6)

        self.dicom_row = _FileRow("DICOM series directory", "dir")
        lay.addWidget(self.dicom_row)
        btn = QtWidgets.QPushButton("Load DICOM series")
        btn.clicked.connect(
            lambda: self.load_requested.emit("dicom", self.dicom_row.path()))
        lay.addWidget(btn)

        lay.addWidget(self._sep())

        self.image_row = _FileRow(
            "Volume file (.nii/.nii.gz/.nrrd)", "file",
            "Images (*.nii *.nii.gz *.nrrd *.mha *.mhd);;All files (*)")
        lay.addWidget(self.image_row)
        btn = QtWidgets.QPushButton("Load volume file")
        btn.clicked.connect(
            lambda: self.load_requested.emit("image", self.image_row.path()))
        lay.addWidget(btn)

        lay.addWidget(self._sep())

        self.btn_full_pipeline = QtWidgets.QPushButton("▶ Run Full Pipeline")
        self.btn_full_pipeline.setStyleSheet(FULL_PIPELINE_QSS)
        self.btn_full_pipeline.setMinimumHeight(34)
        self.btn_full_pipeline.setEnabled(False)
        self.btn_full_pipeline.setToolTip(
            "From the loaded DICOM series or volume, run every stage in "
            "sequence:\n"
            "  1. Segment the aorta and stent metal\n"
            "  2. Extract the centerline\n"
            "  3. Generate surface meshes (STL/VTK)\n"
            "  4. Detect stent endpoints and write the JSON metadata\n\n"
            "Uses the segmentation thresholds set below. Results are written "
            "to output/full_pipeline/<case id>/ and loaded into the views "
            "automatically when done.")
        self.btn_full_pipeline.clicked.connect(
            self.full_pipeline_requested.emit)
        lay.addWidget(self.btn_full_pipeline)

        lay.addWidget(self._sep())

        self.mesh_row = _FileRow(
            "Surface mesh (.stl/.vtk/.vtp/.ply/.obj)", "file",
            "Meshes (*.stl *.vtk *.vtp *.ply *.obj);;All files (*)")
        lay.addWidget(self.mesh_row)
        btn = QtWidgets.QPushButton("Load surface mesh")
        btn.clicked.connect(
            lambda: self.load_requested.emit("mesh", self.mesh_row.path()))
        lay.addWidget(btn)

        self.centerline_row = _FileRow(
            "Centerline file (.vtk/.vtp/.csv/.json) — optional", "file",
            "Centerline (*.vtk *.vtp *.csv *.json);;All files (*)")
        lay.addWidget(self.centerline_row)
        btn = QtWidgets.QPushButton("Load centerline")
        btn.clicked.connect(
            lambda: self.load_requested.emit("centerline",
                                             self.centerline_row.path()))
        lay.addWidget(btn)

        self.auto_render = QtWidgets.QCheckBox(
            "Render a preview surface as soon as a volume loads")
        self.auto_render.setChecked(True)
        self.auto_render.setToolTip(
            "Coarse isosurface of the raw HU volume, so a freshly loaded case "
            "is visible in 3-D before any segmentation has run.")
        lay.addWidget(self.auto_render)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Preview HU"))
        self.preview_hu = QtWidgets.QSpinBox()
        self.preview_hu.setRange(-1000, 3000)
        self.preview_hu.setValue(150)
        self.preview_hu.setSingleStep(25)
        row.addWidget(self.preview_hu, 1)
        lay.addLayout(row)
        return box

    # --------------------------------------------------------------- display
    def _display_group(self):
        box = QtWidgets.QGroupBox("2 · Display / windowing")
        outer = QtWidgets.QVBoxLayout(box)
        form = QtWidgets.QFormLayout()
        outer.addLayout(form)

        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(WL_PRESETS))
        self.preset.currentTextChanged.connect(self._apply_preset)
        form.addRow("Preset", self.preset)

        self.wc = QtWidgets.QSpinBox()
        self.wc.setRange(-1024, 3071)
        self.wc.setValue(300)
        self.wc.setSingleStep(10)
        self.wc.valueChanged.connect(self._mark_pending)
        form.addRow("Window centre (HU)", self.wc)

        self.ww = QtWidgets.QSpinBox()
        self.ww.setRange(1, 4000)
        self.ww.setValue(600)
        self.ww.setSingleStep(10)
        self.ww.valueChanged.connect(self._mark_pending)
        form.addRow("Window width (HU)", self.ww)

        self.lumen_hu = QtWidgets.QSpinBox()
        self.lumen_hu.setRange(-200, 1500)
        self.lumen_hu.setValue(200)
        self.lumen_hu.setSingleStep(10)
        self.lumen_hu.setToolTip(
            "Contrast threshold used to measure the lumen in the orthogonal "
            "cross-section.")
        self.lumen_hu.valueChanged.connect(self._mark_pending)
        form.addRow("Lumen HU threshold", self.lumen_hu)

        row = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("Apply Thresholds")
        self.btn_apply.setToolTip(
            "Redraw the 2-D views with the window and lumen threshold above.")
        self.btn_apply.setMinimumHeight(28)
        self.btn_apply.clicked.connect(self._on_apply)
        row.addWidget(self.btn_apply, 1)
        self.btn_recompute = QtWidgets.QPushButton("Recompute")
        self.btn_recompute.setToolTip(
            "Re-run segmentation with the thresholds below and rebuild the "
            "3-D surface meshes.")
        self.btn_recompute.setMinimumHeight(28)
        self.btn_recompute.clicked.connect(self._on_recompute)
        row.addWidget(self.btn_recompute, 1)
        outer.addLayout(row)

        outer.addWidget(self._sep())
        for attr, label in (("show_aorta", "Aorta overlay / surface"),
                            ("show_stent", "Stent overlay / surface"),
                            ("show_centerline", "Centerline")):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            # Visibility is a toggle on an existing actor, so it stays live.
            cb.toggled.connect(lambda _v: self.display_changed.emit())
            setattr(self, attr, cb)
            outer.addWidget(cb)
        return box

    def _apply_preset(self, name):
        if name in WL_PRESETS:
            wc, ww = WL_PRESETS[name]
            self.wc.setValue(wc)
            self.ww.setValue(ww)   # each fires _mark_pending

    # ---------------------------------------------------------- segmentation
    def _segmentation_group(self):
        box = QtWidgets.QGroupBox("3 · Segmentation thresholds")
        outer = QtWidgets.QVBoxLayout(box)
        form = QtWidgets.QFormLayout()
        outer.addLayout(form)

        self.aorta_tol = QtWidgets.QSpinBox()
        self.aorta_tol.setRange(20, 400)
        self.aorta_tol.setValue(120)
        self.aorta_tol.setSingleStep(10)
        self.aorta_tol.setToolTip(
            "Half-width of the HU band grown from the auto-located aortic "
            "seed. Widen if the aorta is missed, narrow if it leaks into "
            "adjacent structures.")
        self.aorta_tol.valueChanged.connect(self._mark_pending)
        form.addRow("Aorta HU tolerance", self.aorta_tol)

        self.aorta_explicit = QtWidgets.QCheckBox(
            "Use an explicit aorta HU band instead of seed \u00b1 tolerance")
        self.aorta_explicit.setToolTip(
            "Off: the band is centred on the detected seed's own intensity.\n"
            "On: the band below is used verbatim, which is what you want when "
            "the contrast phase puts the lumen outside the usual window.")
        self.aorta_explicit.toggled.connect(self._on_explicit_band)
        form.addRow(self.aorta_explicit)

        self.aorta_hu_lo = QtWidgets.QSpinBox()
        self.aorta_hu_lo.setRange(-1024, 3071)
        self.aorta_hu_lo.setValue(150)
        self.aorta_hu_lo.setSingleStep(10)
        self.aorta_hu_lo.setEnabled(False)
        self.aorta_hu_lo.valueChanged.connect(self._mark_pending)
        form.addRow("Aorta HU lower", self.aorta_hu_lo)

        self.aorta_hu_hi = QtWidgets.QSpinBox()
        self.aorta_hu_hi.setRange(-1024, 3071)
        self.aorta_hu_hi.setValue(450)
        self.aorta_hu_hi.setSingleStep(10)
        self.aorta_hu_hi.setEnabled(False)
        self.aorta_hu_hi.valueChanged.connect(self._mark_pending)
        form.addRow("Aorta HU upper", self.aorta_hu_hi)

        self.stent_hu = QtWidgets.QSpinBox()
        self.stent_hu.setRange(300, 3000)
        self.stent_hu.setValue(1200)
        self.stent_hu.setSingleStep(50)
        self.stent_hu.setToolTip(
            "Metal is thresholded above this, inside a dilated aorta ROI.")
        self.stent_hu.valueChanged.connect(self._mark_pending)
        form.addRow("Stent HU threshold", self.stent_hu)

        self.stent_dilate = QtWidgets.QSpinBox()
        self.stent_dilate.setRange(0, 12)
        self.stent_dilate.setValue(3)
        self.stent_dilate.setToolTip(
            "The frame sits at the vessel wall, so the undilated lumen mask "
            "would clip it.")
        self.stent_dilate.valueChanged.connect(self._mark_pending)
        form.addRow("Aorta ROI dilation (vox)", self.stent_dilate)

        self.resegment = QtWidgets.QCheckBox(
            "Re-segment even when a reference mask is loaded")
        self.resegment.setChecked(False)
        self.resegment.setToolTip(
            "Off: a loaded reference mask is used as the aorta segmentation "
            "directly.\nOn: the volume is segmented anyway, so Dice/IoU "
            "against the reference measure something real -- otherwise the "
            "two masks are the same array and agreement is 1.0 by "
            "construction.")
        self.resegment.toggled.connect(self._mark_pending)
        outer.addWidget(self.resegment)

        note = QtWidgets.QLabel(
            "<i>Changes take effect on <b>Recompute</b>.</i>")
        note.setStyleSheet("color:#7f8ba3; font-size:11px;")
        outer.addWidget(note)
        return box

    def _on_explicit_band(self, on):
        self.aorta_hu_lo.setEnabled(on)
        self.aorta_hu_hi.setEnabled(on)
        self.aorta_tol.setEnabled(not on)
        self._mark_pending()

    # --------------------------------------------------------------- filters
    def _filter_group(self):
        box = QtWidgets.QGroupBox("4 · Filtering / enhancement")
        lay = QtWidgets.QVBoxLayout(box)

        names, params = available_filters()
        self._filter_params_spec = params

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Filter"))
        self.filter_name = QtWidgets.QComboBox()
        self.filter_name.addItem("None")
        self.filter_name.addItems(names)
        self.filter_name.currentTextChanged.connect(self._rebuild_filter_params)
        row.addWidget(self.filter_name, 1)
        lay.addLayout(row)

        if not names:
            lay.addWidget(QtWidgets.QLabel(
                "<i>enhancement.py unavailable — filters disabled</i>"))

        self.filter_params_host = QtWidgets.QWidget()
        self.filter_params_form = QtWidgets.QFormLayout(self.filter_params_host)
        self.filter_params_form.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.filter_params_host)
        self._param_widgets = {}
        return box

    def _rebuild_filter_params(self, name):
        while self.filter_params_form.rowCount():
            self.filter_params_form.removeRow(0)
        self._param_widgets = {}
        for pname, pmin, pmax, pstep, pdef in \
                self._filter_params_spec.get(name, []):
            if isinstance(pdef, int) and isinstance(pstep, int):
                w = QtWidgets.QSpinBox()
                w.setRange(int(pmin), int(pmax))
                w.setSingleStep(int(pstep))
                w.setValue(int(pdef))
            else:
                w = QtWidgets.QDoubleSpinBox()
                w.setRange(float(pmin), float(pmax))
                w.setSingleStep(float(pstep))
                w.setDecimals(3)
                w.setValue(float(pdef))
            self.filter_params_form.addRow(pname, w)
            self._param_widgets[pname] = w

    # ------------------------------------------------------------ centerline
    def _centerline_group(self):
        box = QtWidgets.QGroupBox("5 · Centerline & cross-section")
        form = QtWidgets.QFormLayout(box)

        self.cl_step = QtWidgets.QDoubleSpinBox()
        self.cl_step.setRange(0.25, 5.0)
        self.cl_step.setSingleStep(0.25)
        self.cl_step.setValue(1.0)
        self.cl_step.setSuffix(" mm")
        self.cl_step.setToolTip("Arc-length sampling step along the centerline.")
        form.addRow("Sampling step", self.cl_step)

        self.cl_smooth = QtWidgets.QCheckBox("Orient root → distal")
        self.cl_smooth.setChecked(True)
        self.cl_smooth.setToolTip(
            "Direction of travel along the extracted centerline. Stent "
            "start/end come from the specs JSON or manual picks in the Stent "
            "tab, not from this.")
        form.addRow(self.cl_smooth)

        self.cl_fov = QtWidgets.QSpinBox()
        self.cl_fov.setRange(20, 200)
        self.cl_fov.setValue(60)
        self.cl_fov.setSingleStep(5)
        self.cl_fov.setSuffix(" mm")
        self.cl_fov.valueChanged.connect(self._mark_pending)
        form.addRow("Cross-section FOV", self.cl_fov)

        self.cl_res = QtWidgets.QDoubleSpinBox()
        self.cl_res.setRange(0.1, 1.0)
        self.cl_res.setSingleStep(0.05)
        self.cl_res.setValue(0.4)
        self.cl_res.setSuffix(" mm/px")
        self.cl_res.valueChanged.connect(self._mark_pending)
        form.addRow("Cross-section resolution", self.cl_res)

        btn = QtWidgets.QPushButton("Compute diameter profile")
        btn.clicked.connect(self.profile_requested.emit)
        form.addRow(btn)
        return box

    # --------------------------------------------------------------- actions
    def _actions_group(self):
        box = QtWidgets.QGroupBox("6 · Run")
        lay = QtWidgets.QVBoxLayout(box)

        self.btn_preprocess = QtWidgets.QPushButton("① Run Preprocessing")
        self.btn_preprocess.clicked.connect(self.preprocess_requested.emit)
        lay.addWidget(self.btn_preprocess)

        self.btn_centerline = QtWidgets.QPushButton("② Extract Centerline")
        self.btn_centerline.clicked.connect(self.centerline_requested.emit)
        lay.addWidget(self.btn_centerline)

        self.btn_stent = QtWidgets.QPushButton("③ Fit Stent")
        self.btn_stent.clicked.connect(self.stent_requested.emit)
        lay.addWidget(self.btn_stent)

        self.btn_export = QtWidgets.QPushButton("④ Export Results")
        self.btn_export.clicked.connect(self.export_requested.emit)
        lay.addWidget(self.btn_export)

        self.btn_cancel = QtWidgets.QPushButton("Cancel running task")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        lay.addWidget(self.btn_cancel)

        for b in (self.btn_preprocess, self.btn_centerline,
                  self.btn_stent, self.btn_export):
            b.setMinimumHeight(30)
        return box

    # ------------------------------------------------- deferred-apply state
    def _mark_pending(self, *_args):
        self._set_pending(True)

    def _set_pending(self, pending):
        """Highlight Apply/Recompute while edited values are not yet in effect."""
        self._pending = bool(pending)
        style = PENDING_QSS if pending else ""
        self.btn_apply.setStyleSheet(style)
        self.btn_recompute.setStyleSheet(style)
        suffix = " •" if pending else ""
        self.btn_apply.setText("Apply Thresholds" + suffix)
        self.btn_recompute.setText("Recompute" + suffix)

    def _on_apply(self):
        self._set_pending(False)
        self.apply_thresholds.emit()

    def _on_recompute(self):
        self._set_pending(False)
        self.recompute_requested.emit()

    # ------------------------------------------------------------------ misc
    @staticmethod
    def _sep():
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return line

    def set_busy(self, busy):
        for b in (self.btn_preprocess, self.btn_centerline, self.btn_stent,
                  self.btn_export, self.btn_apply, self.btn_recompute,
                  self.btn_full_pipeline):
            b.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        if not busy:
            self.btn_full_pipeline.setEnabled(self._volume_loaded)

    def set_volume_loaded(self, loaded):
        """Gate 'Run Full Pipeline' on a raw volume actually being loaded."""
        self._volume_loaded = bool(loaded)
        self.btn_full_pipeline.setEnabled(self._volume_loaded)

    # -- parameter access --------------------------------------------------- #
    def filter_settings(self):
        name = self.filter_name.currentText()
        params = {k: w.value() for k, w in self._param_widgets.items()}
        return name, params

    def params(self):
        return {
            "aorta_tolerance": self.aorta_tol.value(),
            "aorta_hu_range": ((self.aorta_hu_lo.value(),
                                self.aorta_hu_hi.value())
                               if self.aorta_explicit.isChecked() else None),
            "stent_hu": self.stent_hu.value(),
            "stent_dilate": self.stent_dilate.value(),
            "cl_step": self.cl_step.value(),
            "cl_orient": self.cl_smooth.isChecked(),
            "cl_fov": self.cl_fov.value(),
            "cl_res": self.cl_res.value(),
            "lumen_hu": self.lumen_hu.value(),
            "resegment": self.resegment.isChecked(),
            "wc": self.wc.value(),
            "ww": self.ww.value(),
            "preview_hu": self.preview_hu.value(),
            "auto_render": self.auto_render.isChecked(),
        }

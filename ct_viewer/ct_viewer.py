#!/usr/bin/env python3
"""
Mini CT Viewer
A lightweight DICOM/NIfTI viewer with synchronized Multi-Planar
Reconstruction (MPR) and 3D volume rendering.

Features
--------
- Load a DICOM series from a folder (recursive scan) or a NIfTI file
- Auto-group by series and sort slices by ImagePositionPatient (falls back to InstanceNumber)
- Synchronized 2x2 viewport grid: Axial (XY), Sagittal (YZ), Coronal (XZ), 3D
- Interactive crosshair: click/drag in any 2D viewport moves the shared 3D
  cursor and updates the other two planes + the 3D cursor marker (see mpr.py)
- 3D volume ray-casting with optional aorta/stent segmentation isosurfaces
  (see volume3d.py, VTK-based)
- Basic metadata panel (PatientID, Slice Thickness, Image Position, ...)
- Window/Level (brightness/contrast) via sliders, drag, and CT presets
- Zoom and pan, per 2D viewport
- Live HU readout on mouse hover, in any 2D viewport
- Switch between series in a folder, or load a different folder

Mouse controls (2D viewports)
------------------------------
- Left click / drag  : move the shared 3D crosshair cursor
- Shift + Left drag  : adjust window/level (brightness/contrast)
- Wheel              : scroll through that plane's slices
- Ctrl + Wheel       : zoom
- Middle drag        : pan
- Right drag         : zoom (drag up = zoom in)

The 3D viewport uses VTK's default trackball camera controls (left drag to
orbit, wheel to zoom, middle drag to pan). Right-click without dragging opens
a menu to save the current 3D view as PNG/JPG.

Dependencies: pydicom, numpy, PyQt5, vtk, nibabel (for NIfTI support)
    pip install pydicom numpy PyQt5 vtk nibabel
Run:
    python ct_viewer.py
"""

import os
import shutil
import sys
import tempfile
import threading
import uuid

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

try:
    import nibabel as nib
except ImportError:  # NIfTI support is optional
    nib = None

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QThread

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import enhancement as enh
import metrics as qm
import segmentation as seg
import mpr
from volume3d import Volume3DWidget


# --------------------------------------------------------------------------- #
#  Frozen-build path helpers
#
#  Running from source, "here" is this file's folder. Running as a
#  PyInstaller build, bundled data lives under sys._MEIPASS (onefile:
#  a temp extraction dir; onedir: the folder next to the exe) while the
#  exe itself lives in sys.executable's folder - those two only differ
#  for onefile, so resource_path()/app_dir() are kept separate.
# --------------------------------------------------------------------------- #
def app_dir():
    """Folder containing the running exe (or this script, from source)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    """Resolve a path to a bundled resource (icon, sample data, ...) that
    works both from source and from a frozen onefile/onedir build."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


def output_dir():
    """Writable folder for files the app produces (e.g. stent CSV export).
    Prefers the exe's own folder; falls back to Documents if that folder
    isn't writable (e.g. an install under Program Files)."""
    candidate = app_dir()
    try:
        probe = os.path.join(candidate, ".write_test")
        with open(probe, "w") as fh:
            fh.write("")
        os.remove(probe)
        return candidate
    except OSError:
        fallback = os.path.join(os.path.expanduser("~"), "Documents", "Mini CT Viewer")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def app_icon():
    icon_path = resource_path("icon.ico")
    if os.path.isfile(icon_path):
        return QtGui.QIcon(icon_path)
    return QtGui.QIcon()


# --------------------------------------------------------------------------- #
#  CT window/level presets (Window Width, Window Center)
# --------------------------------------------------------------------------- #
PRESETS = {
    "Soft Tissue":  (400, 40),
    "Abdomen":      (350, 50),
    "Lung":         (1500, -600),
    "Bone":         (1500, 500),
    "Brain":        (80, 40),
    "Mediastinum":  (350, 50),
    "Angio":        (600, 300),
    "Stent/Metal":  (2000, 450),
}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _str(ds, tag, default="-"):
    """Safely read a DICOM tag as a string."""
    val = getattr(ds, tag, None)
    if val is None or val == "":
        return default
    return str(val)


def _first(value):
    """DICOM WindowCenter/Width may be a single value or a list."""
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        return float(value[0])
    return float(value)


def compute_sort_value(ds):
    """
    Project ImagePositionPatient onto the slice normal so slices order
    correctly regardless of patient orientation. Fall back to InstanceNumber.
    """
    try:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        ipp = [float(x) for x in ds.ImagePositionPatient]
        row = np.array(iop[0:3], dtype=float)
        col = np.array(iop[3:6], dtype=float)
        normal = np.cross(row, col)
        return float(np.dot(np.array(ipp, dtype=float), normal))
    except Exception:
        try:
            return float(ds.InstanceNumber)
        except Exception:
            return 0.0


def window_slice(hu_slice, wc, ww):
    """Apply window center/width and return a uint8 grayscale image."""
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    if hi <= lo:
        hi = lo + 1.0
    out = (hu_slice.astype(np.float32) - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)


def to_pixmap(u8, photometric):
    """Convert a uint8 grayscale array to a QPixmap (handles MONOCHROME1)."""
    if photometric == "MONOCHROME1":
        u8 = 255 - u8
    u8 = np.ascontiguousarray(u8)
    h, w = u8.shape
    qimg = QtGui.QImage(u8.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg)  # fromImage copies the buffer


# --------------------------------------------------------------------------- #
#  Folder scanning  ->  series catalogue
# --------------------------------------------------------------------------- #
class SeriesEntry:
    def __init__(self, uid):
        self.uid = uid
        self.files = []           # list of (sort_value, instance_number, path, n_frames)
        self.modality = "?"
        self.description = ""
        self.patient_id = ""

    @property
    def n_slices(self):
        return sum(max(1, f[3]) for f in self.files)

    def label(self):
        desc = self.description or "(no description)"
        return f"[{self.modality}] {desc}  -  {self.n_slices} slices"


def scan_folder(folder):
    """Walk a folder, return {series_uid: SeriesEntry} for all image series."""
    series = {}
    for root, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            # Must look like an image
            if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
                continue
            uid = getattr(ds, "SeriesInstanceUID", None)
            if uid is None:
                uid = path  # treat orphan file as its own "series"

            entry = series.get(uid)
            if entry is None:
                entry = SeriesEntry(uid)
                entry.modality = _str(ds, "Modality", "?")
                entry.description = _str(ds, "SeriesDescription", "")
                entry.patient_id = _str(ds, "PatientID", "")
                series[uid] = entry

            n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
            inst = int(getattr(ds, "InstanceNumber", 0) or 0)
            entry.files.append((compute_sort_value(ds), inst, path, n_frames))

    # sort slices within each series
    for entry in series.values():
        entry.files.sort(key=lambda t: (t[0], t[1]))
    return series


# --------------------------------------------------------------------------- #
#  Background loader (reads pixel data, builds an HU volume)
# --------------------------------------------------------------------------- #
class SeriesLoader(QThread):
    progress = pyqtSignal(int, int)          # current, total
    finished_ok = pyqtSignal(dict)           # result dict
    failed = pyqtSignal(str)

    def __init__(self, entry, memmap_path):
        super().__init__()
        self.entry = entry
        self.memmap_path = memmap_path

    def run(self):
        try:
            positions = []
            meta_ds = None
            total = len(self.entry.files)
            total_frames = sum(max(1, nf) for _sv, _inst, _p, nf in self.entry.files)

            # Back the volume with a disk-mapped file instead of an in-RAM
            # array: a 946-slice 512x512 int16 volume is ~500MB, which alone
            # can MemoryError on a machine with only a couple GB free. Each
            # slice is still written in one iterative pass as it's read (same
            # loop as before) - only the backing store changed, so indexing
            # (self.volume[i]) behaves identically for the rest of the app.
            first_ds = pydicom.dcmread(self.entry.files[0][2], force=True)
            common_shape = apply_modality_lut(first_ds.pixel_array, first_ds).shape[-2:]
            volume = np.memmap(self.memmap_path, mode="w+", dtype=np.int16,
                                shape=(total_frames,) + common_shape)
            write_idx = 0

            for i, (_sv, _inst, path, _nf) in enumerate(self.entry.files):
                ds = pydicom.dcmread(path, force=True)
                if meta_ds is None:
                    meta_ds = ds

                arr = ds.pixel_array
                # Apply rescale slope/intercept -> Hounsfield Units
                hu = apply_modality_lut(arr, ds).astype(np.float32)

                ipp = getattr(ds, "ImagePositionPatient", None)
                frames = hu if hu.ndim == 3 else hu[None]

                for f in range(frames.shape[0]):
                    frame = frames[f]
                    if frame.shape != common_shape or write_idx >= volume.shape[0]:
                        continue  # skip odd-shaped or unexpected extra frames
                    volume[write_idx] = frame.astype(np.int16)
                    positions.append(list(ipp) if ipp is not None else [0, 0, f])
                    write_idx += 1

                self.progress.emit(i + 1, total)

            if write_idx == 0:
                self.failed.emit("No readable slices in this series.")
                return

            volume = volume[:write_idx]

            # Default window/level
            try:
                wc = _first(meta_ds.WindowCenter)
                ww = _first(meta_ds.WindowWidth)
            except Exception:
                # Subsampled percentile - a full-volume np.percentile() would
                # transiently need a multi-GB float64 copy to sort.
                lo, hi = np.percentile(volume[::4, ::2, ::2], [1, 99])
                wc = float((lo + hi) / 2.0)
                ww = float(max(hi - lo, 1.0))

            # Pixel geometry
            spacing = getattr(meta_ds, "PixelSpacing", [1.0, 1.0])
            try:
                row_sp, col_sp = float(spacing[0]), float(spacing[1])
            except Exception:
                row_sp = col_sp = 1.0

            iop = getattr(meta_ds, "ImageOrientationPatient", None)
            iop = [float(v) for v in iop] if iop is not None else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

            result = {
                "volume": volume,
                "memmap_path": self.memmap_path,
                "positions": positions,
                "iop": iop,
                "photometric": _str(meta_ds, "PhotometricInterpretation",
                                    "MONOCHROME2"),
                "wc": wc, "ww": ww,
                "row_spacing": row_sp, "col_spacing": col_sp,
                "meta": {
                    "PatientID": _str(meta_ds, "PatientID"),
                    "PatientName": _str(meta_ds, "PatientName"),
                    "Modality": _str(meta_ds, "Modality"),
                    "StudyDescription": _str(meta_ds, "StudyDescription"),
                    "SeriesDescription": _str(meta_ds, "SeriesDescription"),
                    "Manufacturer": _str(meta_ds, "Manufacturer"),
                    "Rows": _str(meta_ds, "Rows"),
                    "Columns": _str(meta_ds, "Columns"),
                    "SliceThickness": _str(meta_ds, "SliceThickness"),
                    "PixelSpacing": f"{row_sp:.3f} / {col_sp:.3f}",
                    "NumSlices": str(volume.shape[0]),
                },
            }
            self.finished_ok.emit(result)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
#  NIfTI (.nii / .nii.gz) loader - produces the same result dict as
#  SeriesLoader so the rest of the app (rendering, W/L, segmentation, ...)
#  doesn't need to know which format was loaded.
# --------------------------------------------------------------------------- #
class NiftiLoader(QThread):
    progress = pyqtSignal(int, int)          # current, total
    finished_ok = pyqtSignal(dict)           # result dict
    failed = pyqtSignal(str)

    def __init__(self, path, memmap_path):
        super().__init__()
        self.path = path
        self.memmap_path = memmap_path

    def run(self):
        try:
            if nib is None:
                self.failed.emit(
                    "NIfTI support requires the 'nibabel' package.\n"
                    "Install it with:  pip install nibabel")
                return

            img = nib.load(self.path)
            img = nib.as_closest_canonical(img)  # consistent RAS+ orientation
            data = img.get_fdata(dtype=np.float32)
            if data.ndim == 4:
                data = data[..., 0]  # first volume/timepoint only

            zooms = img.header.get_zooms()[:3]
            col_sp, row_sp, dz = float(zooms[0]), float(zooms[1]), float(zooms[2] or 1.0)

            # NIfTI (RAS+) axes are (X, Y, Z) -> viewer expects slice-major
            # (Z, row, col). Flip the row axis so anterior renders at the
            # top of the image, matching the axial layout used elsewhere.
            volume_data = np.transpose(data, (2, 1, 0))
            volume_data = np.flip(volume_data, axis=1)
            n = volume_data.shape[0]

            volume = np.memmap(self.memmap_path, mode="w+", dtype=np.int16,
                                shape=volume_data.shape)
            for i in range(n):
                volume[i] = np.clip(volume_data[i], -32768, 32767).astype(np.int16)
                self.progress.emit(i + 1, n)

            positions = [[0.0, 0.0, float(i * dz)] for i in range(n)]

            step = max(1, n // 40)
            lo, hi = np.percentile(volume[::step], [1, 99])
            wc = float((lo + hi) / 2.0)
            ww = float(max(hi - lo, 1.0))

            fname = os.path.basename(self.path)
            result = {
                "volume": volume,
                "memmap_path": self.memmap_path,
                "positions": positions,
                "iop": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                "photometric": "MONOCHROME2",
                "wc": wc, "ww": ww,
                "row_spacing": row_sp, "col_spacing": col_sp,
                "meta": {
                    "PatientID": "-",
                    "PatientName": "-",
                    "Modality": "CT (NIfTI)",
                    "StudyDescription": "-",
                    "SeriesDescription": fname,
                    "Manufacturer": "-",
                    "Rows": str(volume.shape[1]),
                    "Columns": str(volume.shape[2]),
                    "SliceThickness": f"{dz:.3f}",
                    "PixelSpacing": f"{row_sp:.3f} / {col_sp:.3f}",
                    "NumSlices": str(volume.shape[0]),
                },
            }
            self.finished_ok.emit(result)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
#  Generic background worker (segmentation / export run off the UI thread)
# --------------------------------------------------------------------------- #
class Worker(QThread):
    progress = pyqtSignal(int, int)     # current, total
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            result = self.fn(self.progress.emit)
            self.finished_ok.emit(result)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
#  Enhancement filters (Non-Local Means, Wavelet, EADTV, ...) can take
#  seconds per slice - running them inline in _redraw_plane() used to block
#  the UI thread on every slice scroll / parameter tweak while a filter was
#  active. This persistent background thread always works on the *latest*
#  submitted request only (older, superseded requests are dropped rather
#  than queued), so scrubbing/tweaking stays responsive: the UI shows the
#  raw slice immediately and swaps in the filtered result once it's ready.
# --------------------------------------------------------------------------- #
class FilterWorker(QThread):
    ready = pyqtSignal(int, dict)     # request_id, {plane: filtered_u8}
    failed = pyqtSignal(int, str)     # request_id, message

    def __init__(self):
        super().__init__()
        self._mutex = QtCore.QMutex()
        self._cond = QtCore.QWaitCondition()
        self._pending = None
        self._running = True

    def submit(self, request_id, filter_name, kwargs, raw_by_plane):
        self._mutex.lock()
        self._pending = (request_id, filter_name, dict(kwargs), raw_by_plane)
        self._cond.wakeOne()
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._cond.wakeOne()
        self._mutex.unlock()
        self.wait()

    def run(self):
        while True:
            self._mutex.lock()
            while self._pending is None and self._running:
                self._cond.wait(self._mutex)
            if not self._running:
                self._mutex.unlock()
                return
            request_id, filter_name, kwargs, raw_by_plane = self._pending
            self._pending = None
            self._mutex.unlock()

            try:
                fn = enh.FILTERS[filter_name]
                out = {plane: fn(raw, **kwargs) for plane, raw in raw_by_plane.items()}
                self.ready.emit(request_id, out)
            except Exception as exc:
                self.failed.emit(request_id, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
#  Slice view  (zoom / pan / window-level drag / HU hover / scroll)
# --------------------------------------------------------------------------- #
class SliceView(QtWidgets.QGraphicsView):
    hovered = pyqtSignal(object)             # (row, col) or None
    wlDragged = pyqtSignal(int, int)         # dx, dy
    sliceScrolled = pyqtSignal(int)          # +1 / -1
    picked = pyqtSignal(int, int)            # (row, col) - crosshair navigation

    def __init__(self):
        super().__init__()
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.pix_item = QtWidgets.QGraphicsPixmapItem()
        self.pix_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self.pix_item)

        # Crosshair lines are children of pix_item so they live in the same
        # untransformed pixel coordinate space as the image and automatically
        # inherit its non-square-pixel aspect scaling. A cosmetic (width 0)
        # pen keeps them a crisp 1 device-pixel line at any zoom level.
        self.hline = QtWidgets.QGraphicsLineItem(self.pix_item)
        self.vline = QtWidgets.QGraphicsLineItem(self.pix_item)
        crosshair_pen = QtGui.QPen(QtGui.QColor(255, 220, 0, 200))
        crosshair_pen.setWidth(0)
        self.hline.setPen(crosshair_pen)
        self.vline.setPen(crosshair_pen)
        self.hline.setZValue(10)
        self.vline.setZValue(10)
        self.hline.hide()
        self.vline.hide()

        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(15, 15, 18)))
        self.setMouseTracking(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        self._has_image = False
        self._last = None
        self._btn = None
        self._left_mode = None  # "pick" or "wl", decided when LMB is pressed

    def set_pixmap(self, pixmap, pixel_aspect=1.0):
        first = not self._has_image
        self.pix_item.setPixmap(pixmap)
        # account for non-square pixels by scaling the item vertically
        self.pix_item.setTransform(QtGui.QTransform().scale(1.0, pixel_aspect))
        self._scene.setSceneRect(self.pix_item.mapRectToScene(
            self.pix_item.boundingRect()))
        self._has_image = True
        if first:
            self.fit()

    def fit(self):
        if self._has_image:
            self.fitInView(self.pix_item, Qt.KeepAspectRatio)

    def set_crosshair(self, row, col):
        """Draw the crosshair at (row, col) in this view's pixel space -
        i.e. where the other two MPR planes currently intersect it."""
        if not self._has_image:
            return
        pm = self.pix_item.pixmap()
        w, h = pm.width(), pm.height()
        self.hline.setLine(0, row, w, row)
        self.vline.setLine(col, 0, col, h)
        self.hline.show()
        self.vline.show()

    # ---- HU readout -------------------------------------------------------- #
    def _item_rc(self, view_pos):
        """Map a view position to (row, col) in image pixel space, or None."""
        if not self._has_image:
            return None
        item_pt = self.pix_item.mapFromScene(self.mapToScene(view_pos))
        x, y = int(item_pt.x()), int(item_pt.y())
        pm = self.pix_item.pixmap()
        if 0 <= x < pm.width() and 0 <= y < pm.height():
            return (y, x)
        return None

    def _emit_hover(self, view_pos):
        self.hovered.emit(self._item_rc(view_pos))

    # ---- events ------------------------------------------------------------ #
    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            factor = 1.25 if e.angleDelta().y() > 0 else 0.8
            self.scale(factor, factor)
        else:
            self.sliceScrolled.emit(1 if e.angleDelta().y() < 0 else -1)
        e.accept()

    def mousePressEvent(self, e):
        self._last = e.pos()
        self._btn = e.button()
        if e.button() == Qt.LeftButton:
            # Plain left click/drag navigates the shared 3D cursor (crosshair);
            # Shift+left keeps the original window/level drag behaviour.
            if e.modifiers() & Qt.ShiftModifier:
                self._left_mode = "wl"
            else:
                self._left_mode = "pick"
                rc = self._item_rc(e.pos())
                if rc is not None:
                    self.picked.emit(*rc)
        elif e.button() == Qt.MiddleButton:
            self.setCursor(Qt.ClosedHandCursor)
        elif e.button() == Qt.RightButton:
            self.setCursor(Qt.SizeVerCursor)
        e.accept()

    def mouseMoveEvent(self, e):
        self._emit_hover(e.pos())
        if self._last is not None and self._btn is not None:
            dx = e.x() - self._last.x()
            dy = e.y() - self._last.y()
            if self._btn == Qt.LeftButton:
                if self._left_mode == "wl":
                    self.wlDragged.emit(dx, dy)
                else:
                    rc = self._item_rc(e.pos())
                    if rc is not None:
                        self.picked.emit(*rc)
            elif self._btn == Qt.MiddleButton:
                h, v = self.horizontalScrollBar(), self.verticalScrollBar()
                h.setValue(h.value() - dx)
                v.setValue(v.value() - dy)
            elif self._btn == Qt.RightButton:
                factor = 1.0 + (-dy) * 0.005
                if factor > 0:
                    self.scale(factor, factor)
            self._last = e.pos()
        e.accept()

    def mouseReleaseEvent(self, e):
        self._btn = None
        self._last = None
        self._left_mode = None
        self.setCursor(Qt.ArrowCursor)
        e.accept()

    def leaveEvent(self, e):
        self.hovered.emit(None)
        super().leaveEvent(e)


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini CT Viewer")
        self.resize(1500, 820)

        # state
        self.series_catalogue = {}     # uid -> SeriesEntry
        self.current_dicom_dir = None  # folder passed to open_folder(), if any
        self.volume = None
        self.positions = []
        self.iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.photometric = "MONOCHROME2"
        self.row_spacing = 1.0
        self.col_spacing = 1.0
        self.dz = 1.0

        # Shared 3D cursor (z, y, x) - the single voxel index every viewport
        # is centred on. z indexes the axial slice stack, y is the row axis
        # within a slice (sagittal position), x is the column axis (coronal
        # position). See mpr.py for how a plane's 2D slice / crosshair /
        # click position are derived from this triple.
        self.cursor_z = 0
        self.cursor_y = 0
        self.cursor_x = 0
        self.wc = 40.0
        self.ww = 400.0
        self.loader = None
        self._active_worker = None
        self._meta_dict = {}

        # segmentation state
        self.aorta_mask = None
        self.stent_mask = None
        self.stent_report = None
        self.gt_mask = None

        # enhancement state
        self.active_filter = "None"
        self._filter_spinboxes = {}
        self._last_raw_u8 = None
        self._last_filtered_u8 = None

        # Heavy filters (NLM, Wavelet, EADTV, ...) run on this persistent
        # background thread instead of inline in _redraw_plane() - see
        # FilterWorker. request_id lets a late result from a superseded
        # request be recognised and dropped.
        self._filter_worker = FilterWorker()
        self._filter_worker.ready.connect(self._on_filtered_ready)
        self._filter_worker.failed.connect(self._on_filter_failed)
        self._filter_worker.start()
        self._filter_request_id = 0
        self._filter_pending_cursor = None
        self._filter_pending_raws = {}

        # Per-session scratch dir for memmap-backed volumes (requirement #1):
        # deleting individual files while numpy still holds them mmap'd is
        # unreliable on Windows, so the whole dir is removed on close instead.
        self._scratch_dir = tempfile.mkdtemp(prefix="ctviewer_")

        self._build_ui()

    def closeEvent(self, event):
        self._filter_worker.stop()
        shutil.rmtree(self._scratch_dir, ignore_errors=True)
        super().closeEvent(event)

    @property
    def spacing_zyx(self):
        return (self.dz, self.row_spacing, self.col_spacing)

    # ------------------------------------------------------------------ UI -- #
    def _build_ui(self):
        self.view_axial = SliceView()
        self.view_sagittal = SliceView()
        self.view_coronal = SliceView()
        self.view3d = Volume3DWidget()

        # views dict drives all the plane-generic wiring below (see mpr.py
        # for how a plane name maps to slice indexing / crosshair / picking)
        self.planes = {
            "axial": self.view_axial,
            "sagittal": self.view_sagittal,
            "coronal": self.view_coronal,
        }
        for plane, view in self.planes.items():
            view.hovered.connect(lambda rc, p=plane: self._on_hover(p, rc))
            view.wlDragged.connect(self._on_wl_drag)
            view.sliceScrolled.connect(lambda delta, p=plane: self._on_scroll(p, delta))
            view.picked.connect(lambda row, col, p=plane: self._on_picked(p, row, col))

        mpr_grid = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(mpr_grid)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(2)
        grid.addWidget(self._make_quadrant("Axial (XY)", self.view_axial), 0, 0)
        grid.addWidget(self._make_quadrant("Sagittal (YZ)", self.view_sagittal), 0, 1)
        grid.addWidget(self._make_quadrant("Coronal (XZ)", self.view_coronal), 1, 0)
        grid.addWidget(self._make_quadrant("3D View", self.view3d), 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        panel = self._build_side_panel()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(panel)

        results = self._build_results_panel()

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(mpr_grid)
        splitter.addWidget(results)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 780, 380])
        self.setCentralWidget(splitter)

        self.status = self.statusBar()
        self.lbl_hu = QtWidgets.QLabel("HU: -")
        self.lbl_slice = QtWidgets.QLabel("Cursor: -")
        self.lbl_pos = QtWidgets.QLabel("Pos: -")
        for w in (self.lbl_hu, self.lbl_slice, self.lbl_pos):
            self.status.addPermanentWidget(w)

    @staticmethod
    def _make_quadrant(title, widget):
        """Wrap a viewport widget with a small title bar for the 2x2 grid."""
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lbl = QtWidgets.QLabel(title)
        lbl.setStyleSheet(
            "background:#2a2a2e; color:#ccc; padding:2px 6px; font-size:10px;")
        lay.addWidget(lbl)
        lay.addWidget(widget, 1)
        return box

    def _build_side_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        btn_open = QtWidgets.QPushButton("Open...")
        btn_open.clicked.connect(self.open_data)
        lay.addWidget(btn_open)

        lay.addWidget(QtWidgets.QLabel("Series:"))
        self.cmb_series = QtWidgets.QComboBox()
        self.cmb_series.currentIndexChanged.connect(self._on_series_changed)
        lay.addWidget(self.cmb_series)

        # MPR slice sliders - one per orthogonal plane, all driving the same
        # shared (z, y, x) cursor (see _set_cursor).
        lay.addWidget(self._hline())
        lay.addWidget(QtWidgets.QLabel("MPR Position"))
        slice_form = QtWidgets.QFormLayout()
        self.sld_axial = QtWidgets.QSlider(Qt.Horizontal)
        self.sld_axial.valueChanged.connect(self._on_axial_slider)
        self.sld_sagittal = QtWidgets.QSlider(Qt.Horizontal)
        self.sld_sagittal.valueChanged.connect(self._on_sagittal_slider)
        self.sld_coronal = QtWidgets.QSlider(Qt.Horizontal)
        self.sld_coronal.valueChanged.connect(self._on_coronal_slider)
        slice_form.addRow("Axial (Z)", self.sld_axial)
        slice_form.addRow("Sagittal (X)", self.sld_sagittal)
        slice_form.addRow("Coronal (Y)", self.sld_coronal)
        lay.addLayout(slice_form)

        # Window / Level
        lay.addWidget(self._hline())
        lay.addWidget(QtWidgets.QLabel("Window / Level"))
        form = QtWidgets.QFormLayout()
        self.spn_wc = QtWidgets.QSpinBox()
        self.spn_wc.setRange(-2000, 4000)
        self.spn_wc.valueChanged.connect(self._on_wc_spin)
        self.spn_ww = QtWidgets.QSpinBox()
        self.spn_ww.setRange(1, 6000)
        self.spn_ww.valueChanged.connect(self._on_ww_spin)
        form.addRow("Center (WC)", self.spn_wc)
        form.addRow("Width (WW)", self.spn_ww)
        lay.addLayout(form)

        grid = QtWidgets.QGridLayout()
        for i, name in enumerate(PRESETS):
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)

        # View buttons
        lay.addWidget(self._hline())
        b_fit = QtWidgets.QPushButton("Fit / Reset Zoom (2D views)")
        b_fit.clicked.connect(self._fit_all_2d)
        lay.addWidget(b_fit)

        # 3D view controls
        lay.addWidget(self._hline())
        lay.addWidget(QtWidgets.QLabel("3D View"))
        self.chk_3d_volume = QtWidgets.QCheckBox("Show volume render")
        self.chk_3d_volume.setChecked(True)
        self.chk_3d_volume.toggled.connect(self.view3d.set_volume_visible)
        lay.addWidget(self.chk_3d_volume)

        # Enhancement
        lay.addWidget(self._hline())
        lay.addWidget(QtWidgets.QLabel("Enhancement"))
        self.cmb_filter = QtWidgets.QComboBox()
        self.cmb_filter.addItem("None")
        self.cmb_filter.addItems(sorted(enh.FILTERS.keys()))
        self.cmb_filter.currentTextChanged.connect(self._on_filter_changed)
        lay.addWidget(self.cmb_filter)

        self.filter_param_form = QtWidgets.QFormLayout()
        lay.addLayout(self.filter_param_form)

        # Segmentation
        lay.addWidget(self._hline())
        lay.addWidget(QtWidgets.QLabel("Segmentation (fully automated)"))

        form_aorta = QtWidgets.QFormLayout()
        self.spn_aorta_tol = QtWidgets.QSpinBox()
        self.spn_aorta_tol.setRange(10, 1000)
        self.spn_aorta_tol.setValue(120)
        form_aorta.addRow("Aorta HU tolerance (±)", self.spn_aorta_tol)
        lay.addLayout(form_aorta)

        form_stent = QtWidgets.QFormLayout()
        self.spn_stent_hu = QtWidgets.QSpinBox()
        self.spn_stent_hu.setRange(-1000, 3000)
        self.spn_stent_hu.setValue(700)
        self.spn_stent_minvol = QtWidgets.QDoubleSpinBox()
        self.spn_stent_minvol.setRange(0, 100000)
        self.spn_stent_minvol.setValue(20)
        self.spn_stent_maxvol = QtWidgets.QDoubleSpinBox()
        self.spn_stent_maxvol.setRange(0, 1000000)
        self.spn_stent_maxvol.setValue(50000)
        self.spn_stent_aspect = QtWidgets.QDoubleSpinBox()
        self.spn_stent_aspect.setRange(1.0, 20.0)
        self.spn_stent_aspect.setSingleStep(0.5)
        self.spn_stent_aspect.setValue(2.0)
        form_stent.addRow("Stent HU threshold (metal)", self.spn_stent_hu)
        form_stent.addRow("Min volume (mm³)", self.spn_stent_minvol)
        form_stent.addRow("Max volume (mm³)", self.spn_stent_maxvol)
        form_stent.addRow("Min aspect ratio", self.spn_stent_aspect)
        lay.addLayout(form_stent)

        self.btn_auto_seg = QtWidgets.QPushButton("Auto-Segment Aorta + Stent")
        self.btn_auto_seg.clicked.connect(self._run_auto_segmentation)
        lay.addWidget(self.btn_auto_seg)

        self.chk_aorta_overlay = QtWidgets.QCheckBox("Show aorta overlay (red)")
        self.chk_aorta_overlay.setChecked(True)
        self.chk_aorta_overlay.toggled.connect(self._on_aorta_overlay_toggled)
        self.chk_stent_overlay = QtWidgets.QCheckBox("Show stent overlay (cyan)")
        self.chk_stent_overlay.setChecked(True)
        self.chk_stent_overlay.toggled.connect(self._on_stent_overlay_toggled)
        lay.addWidget(self.chk_aorta_overlay)
        lay.addWidget(self.chk_stent_overlay)

        btn_load_gt = QtWidgets.QPushButton("Load Ground Truth Mask (.npy)...")
        btn_load_gt.clicked.connect(self._load_ground_truth)
        lay.addWidget(btn_load_gt)

        # HPC (Mogon) pipeline, run locally
        lay.addWidget(self._hline())
        lay.addWidget(QtWidgets.QLabel("Aortic Centerline + Stent (HPC pipeline)"))
        hpc_lbl = QtWidgets.QLabel(
            "Full pipeline (TotalSegmentator + centerline + crown geometry), "
            "the same one used on Mogon. Needs a loaded DICOM folder. Slow "
            "(minutes) and needs TotalSegmentator installed.")
        hpc_lbl.setWordWrap(True)
        hpc_lbl.setStyleSheet("color: #999;")
        lay.addWidget(hpc_lbl)
        form_hpc = QtWidgets.QFormLayout()
        self.spn_hpc_iso_spacing = QtWidgets.QDoubleSpinBox()
        self.spn_hpc_iso_spacing.setRange(0.4, 2.0)
        self.spn_hpc_iso_spacing.setSingleStep(0.1)
        self.spn_hpc_iso_spacing.setValue(0.6)
        self.spn_hpc_iso_spacing.setToolTip(
            "Resample spacing for the pipeline's own canonical volume, mm. "
            "0.6mm matches Mogon but can exhaust memory on a long-FOV series "
            "on a desktop machine -- raise it (e.g. 1.0-1.5) if the run "
            "fails with a memory allocation error.")
        form_hpc.addRow("Resample spacing (mm)", self.spn_hpc_iso_spacing)
        lay.addLayout(form_hpc)
        self.btn_hpc_pipeline = QtWidgets.QPushButton(
            "Extract Aortic Centerline + Stent...")
        self.btn_hpc_pipeline.clicked.connect(self._run_hpc_centerline_pipeline)
        lay.addWidget(self.btn_hpc_pipeline)

        # Export
        lay.addWidget(self._hline())
        lay.addWidget(QtWidgets.QLabel("Export"))
        btn_export = QtWidgets.QPushButton("Export Slices as PNG/JPG...")
        btn_export.clicked.connect(self._export_images)
        lay.addWidget(btn_export)

        # Metadata
        lay.addWidget(self._hline())
        lay.addWidget(QtWidgets.QLabel("Metadata"))
        self.meta = QtWidgets.QTextEdit()
        self.meta.setReadOnly(True)
        self.meta.setFont(QtGui.QFont("Monospace", 9))
        lay.addWidget(self.meta, 1)

        help_lbl = QtWidgets.QLabel(
            "Click/L-drag: move crosshair  |  Wheel: scroll this plane\n"
            "Shift+L-drag: window/level  |  Ctrl+Wheel: zoom\n"
            "M-drag: pan  |  R-drag: zoom  |  Arrows: axial slice\n"
            "3D view: L-drag orbit, wheel zoom, M-drag pan (VTK controls)\n"
            "3D view: right-click (no drag) to save as PNG/JPG")
        help_lbl.setStyleSheet("color:#888; font-size:10px;")
        lay.addWidget(help_lbl)
        return w

    @staticmethod
    def _hline():
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    def _build_results_panel(self):
        tabs = QtWidgets.QTabWidget()

        # ---- Histogram tab --------------------------------------------------
        hist_tab = QtWidgets.QWidget()
        hist_lay = QtWidgets.QVBoxLayout(hist_tab)
        self.cmb_hist_source = QtWidgets.QComboBox()
        self.cmb_hist_source.addItems(
            ["Whole Volume", "Aorta", "Stent", "Current Slice (raw vs filtered)"])
        self.cmb_hist_source.currentIndexChanged.connect(self._update_histogram)
        hist_lay.addWidget(self.cmb_hist_source)

        self.hist_figure = Figure(figsize=(4, 3))
        self.hist_canvas = FigureCanvas(self.hist_figure)
        self.hist_ax = self.hist_figure.add_subplot(111)
        hist_lay.addWidget(self.hist_canvas, 1)
        tabs.addTab(hist_tab, "Histogram")

        # ---- Stent Table tab --------------------------------------------------
        table_tab = QtWidgets.QWidget()
        table_lay = QtWidgets.QVBoxLayout(table_tab)
        self.lbl_stent_summary = QtWidgets.QLabel("No stent segmented yet.")
        self.lbl_stent_summary.setWordWrap(True)
        table_lay.addWidget(self.lbl_stent_summary)

        self.tbl_stent = QtWidgets.QTableWidget(0, 5)
        self.tbl_stent.setHorizontalHeaderLabels(
            ["Point", "Slice", "X (mm)", "Y (mm)", "Z (mm)"])
        self.tbl_stent.horizontalHeader().setStretchLastSection(True)
        self.tbl_stent.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table_lay.addWidget(self.tbl_stent, 1)
        tabs.addTab(table_tab, "Stent Table")

        # ---- Enhancement Metrics tab -----------------------------------------
        enh_tab = QtWidgets.QWidget()
        enh_lay = QtWidgets.QVBoxLayout(enh_tab)
        self.lbl_enh_metrics = QtWidgets.QLabel(
            "Select an enhancement filter to compare it against the raw slice.")
        self.lbl_enh_metrics.setWordWrap(True)
        enh_lay.addWidget(self.lbl_enh_metrics)
        self.tbl_enh_metrics = QtWidgets.QTableWidget(0, 2)
        self.tbl_enh_metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.tbl_enh_metrics.horizontalHeader().setStretchLastSection(True)
        self.tbl_enh_metrics.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        enh_lay.addWidget(self.tbl_enh_metrics, 1)
        tabs.addTab(enh_tab, "Enhancement Metrics")

        # ---- Segmentation Validation tab -------------------------------------
        seg_tab = QtWidgets.QWidget()
        seg_lay = QtWidgets.QVBoxLayout(seg_tab)
        self.lbl_seg_validation = QtWidgets.QLabel(
            "Load a ground-truth mask to compare against the segmented stent.")
        self.lbl_seg_validation.setWordWrap(True)
        seg_lay.addWidget(self.lbl_seg_validation)
        self.tbl_seg_metrics = QtWidgets.QTableWidget(0, 2)
        self.tbl_seg_metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.tbl_seg_metrics.horizontalHeader().setStretchLastSection(True)
        self.tbl_seg_metrics.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        seg_lay.addWidget(self.tbl_seg_metrics, 1)
        tabs.addTab(seg_tab, "Segmentation Validation")

        return tabs

    # --------------------------------------------------------------- load -- #
    def open_data(self):
        """Ask the user whether to open a DICOM series or a NIfTI file."""
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Open", "Data source:",
            ["DICOM folder", "NIfTI file (.nii / .nii.gz)"], 0, False)
        if not ok:
            return
        if choice.startswith("DICOM"):
            self.open_folder()
        else:
            self.open_nifti()

    def open_nifti(self):
        if nib is None:
            QtWidgets.QMessageBox.critical(
                self, "Missing dependency",
                "NIfTI support requires the 'nibabel' package.\n\n"
                "Install it with:\n    pip install nibabel")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select NIfTI file", "",
            "NIfTI files (*.nii *.nii.gz);;All files (*)")
        if not path:
            return
        if self.loader and self.loader.isRunning():
            return

        # A NIfTI file is a single volume, not a multi-series folder -- and
        # not a DICOM series either, so the HPC centerline/stent pipeline
        # (which needs real DICOM headers) can't be run against it.
        self.series_catalogue = {}
        self.current_dicom_dir = None
        self.cmb_series.blockSignals(True)
        self.cmb_series.clear()
        self.cmb_series.addItem(os.path.basename(path))
        self.cmb_series.blockSignals(False)

        self.progress = QtWidgets.QProgressDialog(
            "Loading NIfTI volume...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(200)

        memmap_path = os.path.join(self._scratch_dir, f"volume_{uuid.uuid4().hex}.dat")
        self.loader = NiftiLoader(path, memmap_path)

        def _on_progress(c, t):
            if t:
                self.progress.setMaximum(t)
                self.progress.setValue(c)

        self.loader.progress.connect(_on_progress)
        self.loader.finished_ok.connect(self._on_loaded)
        self.loader.failed.connect(self._on_load_failed)
        self.loader.start()

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select DICOM folder")
        if not folder:
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.series_catalogue = scan_folder(folder)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if not self.series_catalogue:
            QtWidgets.QMessageBox.warning(
                self, "Nothing found",
                "No DICOM image series were found in that folder.")
            return
        self.current_dicom_dir = folder

        # populate series combo (sorted by slice count, biggest first)
        self.cmb_series.blockSignals(True)
        self.cmb_series.clear()
        ordered = sorted(self.series_catalogue.values(),
                         key=lambda e: e.n_slices, reverse=True)
        for entry in ordered:
            self.cmb_series.addItem(entry.label(), entry.uid)
        self.cmb_series.blockSignals(False)
        self.cmb_series.setCurrentIndex(0)
        self._load_series(ordered[0])

    def _on_series_changed(self, index):
        if index < 0:
            return
        uid = self.cmb_series.itemData(index)
        entry = self.series_catalogue.get(uid)
        if entry:
            self._load_series(entry)

    def _load_series(self, entry):
        if self.loader and self.loader.isRunning():
            return
        self.progress = QtWidgets.QProgressDialog(
            "Loading series...", None, 0, len(entry.files), self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(200)

        memmap_path = os.path.join(self._scratch_dir, f"volume_{uuid.uuid4().hex}.dat")
        self.loader = SeriesLoader(entry, memmap_path)
        self.loader.progress.connect(lambda c, t: self.progress.setValue(c))
        self.loader.finished_ok.connect(self._on_loaded)
        self.loader.failed.connect(self._on_load_failed)
        self.loader.start()

    def _on_load_failed(self, msg):
        self.progress.close()
        QtWidgets.QMessageBox.critical(self, "Load error", msg)

    def _on_loaded(self, result):
        self.progress.close()
        self.volume = result["volume"]
        self.positions = result["positions"]
        self.iop = result.get("iop", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.photometric = result["photometric"]
        self.wc = result["wc"]
        self.ww = result["ww"]
        self.row_spacing, self.col_spacing = result["row_spacing"], result["col_spacing"]
        self._meta_dict = result["meta"]

        if len(self.positions) > 1:
            diffs = [
                np.linalg.norm(np.array(self.positions[i], dtype=float)
                                - np.array(self.positions[i - 1], dtype=float))
                for i in range(1, len(self.positions))
            ]
            diffs = [d for d in diffs if d > 0]
            self.dz = float(np.median(diffs)) if diffs else 1.0
        else:
            self.dz = 1.0

        # Reset any segmentation/enhancement state from a previous series
        self.aorta_mask = None
        self.stent_mask = None
        self.stent_report = None
        self.gt_mask = None
        self.tbl_stent.setRowCount(0)
        self.lbl_stent_summary.setText("No stent segmented yet.")
        self.tbl_seg_metrics.setRowCount(0)
        self.lbl_seg_validation.setText(
            "Load a ground-truth mask to compare against the segmented stent.")
        self.cmb_filter.blockSignals(True)
        self.cmb_filter.setCurrentIndex(0)
        self.cmb_filter.blockSignals(False)
        self.active_filter = "None"
        self._rebuild_filter_params("None")

        nz, ny, nx = self.volume.shape
        self.cursor_z, self.cursor_y, self.cursor_x = nz // 2, ny // 2, nx // 2

        for slider, n in ((self.sld_axial, nz), (self.sld_sagittal, nx),
                          (self.sld_coronal, ny)):
            slider.blockSignals(True)
            slider.setRange(0, n - 1)
            slider.blockSignals(False)
        self._sync_slice_sliders()

        for view in self.planes.values():
            view._has_image = False  # force fit on new series

        self.view3d.set_volume(self.volume, self.spacing_zyx, self.wc, self.ww)

        self._sync_wl_spins()
        self._redraw_all(fit=True)
        self._update_metadata()
        self._update_histogram()

    def _sync_slice_sliders(self):
        for slider, val in ((self.sld_axial, self.cursor_z),
                            (self.sld_sagittal, self.cursor_x),
                            (self.sld_coronal, self.cursor_y)):
            slider.blockSignals(True)
            slider.setValue(val)
            slider.blockSignals(False)

    # ---------------------------------------------------------- rendering -- #
    def _redraw_all(self, fit=False):
        """Refresh all 4 synchronized viewports from the current cursor/W-L/
        filter/overlay state - the single entry point every interaction
        (crosshair click, slider, scroll, W/L drag, filter change, ...)
        funnels through so the views never drift out of sync.

        The window/level-mapped slice is always cheap and drawn immediately.
        If a filter is active, it's applied asynchronously (see FilterWorker)
        so a slow filter (NLM/Wavelet/EADTV, ...) never blocks scrubbing or
        slider drags - the raw slice is shown right away and gets swapped
        for the filtered one once it's ready."""
        if self.volume is None:
            return
        cursor = (self.cursor_z, self.cursor_y, self.cursor_x)
        raws = {plane: self._compute_raw_slice_u8(plane, cursor) for plane in self.planes}
        for plane, u8_raw in raws.items():
            self._finish_plane_render(plane, cursor, u8_raw, u8_raw, fit=fit, is_filtered=False)

        if self.active_filter and self.active_filter != "None":
            self._filter_request_id += 1
            self._filter_pending_cursor = cursor
            self._filter_pending_raws = raws
            self._filter_worker.submit(
                self._filter_request_id, self.active_filter,
                self._current_filter_kwargs(), dict(raws))

        self.view3d.set_window_level(self.wc, self.ww)
        self.view3d.set_cursor((self.cursor_z, self.cursor_y, self.cursor_x))

        self.lbl_slice.setText(
            f"Z: {self.cursor_z + 1}/{self.volume.shape[0]}   "
            f"Y: {self.cursor_y + 1}/{self.volume.shape[1]}   "
            f"X: {self.cursor_x + 1}/{self.volume.shape[2]}")
        pos = self.positions[self.cursor_z] if self.cursor_z < len(self.positions) else None
        if pos is not None:
            try:
                self.lbl_pos.setText(
                    f"Pos: {float(pos[0]):.1f}, {float(pos[1]):.1f}, "
                    f"{float(pos[2]):.1f}")
            except Exception:
                self.lbl_pos.setText("Pos: -")

    def _compute_raw_slice_u8(self, plane, cursor):
        """Extract + window/level a single 2D slice. Always fast (plain numpy
        indexing/arithmetic), so this stays on the UI thread."""
        raw_slice = mpr.plane_slice(self.volume, plane, cursor)
        u8_raw = window_slice(raw_slice, self.wc, self.ww)
        if self.photometric == "MONOCHROME1":
            u8_raw = 255 - u8_raw
        return u8_raw

    def _on_filtered_ready(self, request_id, results):
        """A background FilterWorker result arrived. Drop it if the user has
        since moved on (new slice/param/filter change already superseded
        this request) rather than painting a stale slice."""
        if self.volume is None or request_id != self._filter_request_id:
            return
        cursor = self._filter_pending_cursor
        raws = self._filter_pending_raws
        for plane, u8_display in results.items():
            self._finish_plane_render(
                plane, cursor, raws.get(plane, u8_display), u8_display,
                fit=False, is_filtered=True)

    def _on_filter_failed(self, request_id, msg):
        if request_id != self._filter_request_id:
            return
        self.status.showMessage(f"Filter error: {msg}", 5000)

    def _finish_plane_render(self, plane, cursor, u8_raw, u8_display, fit=False, is_filtered=False):
        """Composite mask overlays onto an already-computed 2D slice
        (u8_display - either the raw slice or a filtered result) and push it
        to that plane's viewport."""
        view = self.planes[plane]

        # The Enhancement Metrics tab / histogram "raw vs filtered" compare
        # against a single reference slice - the axial plane, matching the
        # original single-view behaviour.
        if plane == "axial":
            has_filter = bool(self.active_filter and self.active_filter != "None")
            if is_filtered:
                self._update_enhancement_metrics_table(u8_raw, u8_display)
                self._last_raw_u8 = u8_raw
                self._last_filtered_u8 = u8_display
                self._update_histogram()
            elif not has_filter:
                self.tbl_enh_metrics.setRowCount(0)
                self.lbl_enh_metrics.setText(
                    "Select an enhancement filter to compare it against the raw slice.")
                self._last_raw_u8 = u8_raw
                self._last_filtered_u8 = None
                self._update_histogram()
            else:
                # Filter active but its result isn't back yet - just track
                # the new raw slice; keep showing the previous filtered
                # metrics/histogram until the background result arrives.
                self._last_raw_u8 = u8_raw

        layers = []
        if self.aorta_mask is not None and self.chk_aorta_overlay.isChecked():
            mask_slice = mpr.plane_slice(self.aorta_mask, plane, cursor)
            layers.append((mask_slice.astype(bool), (220, 50, 50), 0.35))
        if self.stent_mask is not None and self.chk_stent_overlay.isChecked():
            mask_slice = mpr.plane_slice(self.stent_mask, plane, cursor)
            layers.append((mask_slice.astype(bool), (0, 220, 220), 0.55))

        if layers:
            rgb = np.ascontiguousarray(seg.blend_overlay(u8_display, layers))
            h, w, _ = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            pm = QtGui.QPixmap.fromImage(qimg)
        else:
            u8c = np.ascontiguousarray(u8_display)
            h, w = u8c.shape
            qimg = QtGui.QImage(u8c.data, w, h, w, QtGui.QImage.Format_Grayscale8)
            pm = QtGui.QPixmap.fromImage(qimg)

        view.set_pixmap(pm, mpr.plane_aspect(plane, self.spacing_zyx))
        if fit:
            view.fit()
        view.set_crosshair(*mpr.plane_crosshair(plane, cursor))

    # ------------------------------------------------------- cursor updates #
    def _set_cursor(self, z, y, x):
        """Single choke point for moving the shared 3D cursor - clamps to the
        volume bounds, keeps the 3 sliders in sync, and refreshes all views."""
        if self.volume is None:
            return
        self.cursor_z, self.cursor_y, self.cursor_x = mpr.clamp_cursor(
            (z, y, x), self.volume.shape)
        self._sync_slice_sliders()
        self._redraw_all()

    def _on_axial_slider(self, value):
        self._set_cursor(value, self.cursor_y, self.cursor_x)

    def _on_sagittal_slider(self, value):
        self._set_cursor(self.cursor_z, self.cursor_y, value)

    def _on_coronal_slider(self, value):
        self._set_cursor(self.cursor_z, value, self.cursor_x)

    def _on_scroll(self, plane, delta):
        """Mouse wheel over a 2D viewport scrolls through *that plane's*
        stack of slices - i.e. it moves the cursor component the plane is
        normal to (see mpr.scroll_axis)."""
        if self.volume is None:
            return
        axis = mpr.scroll_axis(plane)
        z, y, x = self.cursor_z, self.cursor_y, self.cursor_x
        cursor = [z, y, x]
        cursor[axis] += delta
        self._set_cursor(*cursor)

    def _on_picked(self, plane, row, col):
        """Left click/drag inside a 2D viewport: (row, col) is in that
        plane's own pixel space, so map it back to the shared (z, y, x)
        cursor and re-sync every other viewport off of it."""
        cursor = (self.cursor_z, self.cursor_y, self.cursor_x)
        self._set_cursor(*mpr.cursor_from_pick(plane, cursor, row, col))

    def _fit_all_2d(self):
        for view in self.planes.values():
            view.fit()

    def _on_aorta_overlay_toggled(self, checked):
        self.view3d.set_mask_visible("aorta", checked)
        self._redraw_all()

    def _on_stent_overlay_toggled(self, checked):
        self.view3d.set_mask_visible("stent", checked)
        self._redraw_all()

    # ---------------------------------------------------------- window/level #
    def set_wl(self, wc, ww):
        self.wc = float(wc)
        self.ww = float(max(1.0, ww))
        self._sync_wl_spins()
        self._redraw_all()

    def _sync_wl_spins(self):
        for spn, val in ((self.spn_wc, self.wc), (self.spn_ww, self.ww)):
            spn.blockSignals(True)
            spn.setValue(int(round(val)))
            spn.blockSignals(False)

    def _on_wc_spin(self, v):
        self.wc = float(v)
        self._redraw_all()

    def _on_ww_spin(self, v):
        self.ww = float(max(1, v))
        self._redraw_all()

    def _on_wl_drag(self, dx, dy):
        if self.volume is None:
            return
        # horizontal -> width, vertical -> center (up = brighter)
        self.set_wl(self.wc - dy * 2, self.ww + dx * 2)

    def apply_preset(self, name):
        ww, wc = PRESETS[name]
        self.set_wl(wc, ww)

    # ------------------------------------------------------------- metadata #
    def _update_metadata(self):
        m = self._meta_dict
        lines = [
            f"Patient ID     : {m['PatientID']}",
            f"Patient Name   : {m['PatientName']}",
            f"Modality       : {m['Modality']}",
            f"Study          : {m['StudyDescription']}",
            f"Series         : {m['SeriesDescription']}",
            f"Manufacturer   : {m['Manufacturer']}",
            "",
            f"Matrix         : {m['Rows']} x {m['Columns']}",
            f"Slices         : {m['NumSlices']}",
            f"Slice Thickness: {m['SliceThickness']} mm",
            f"Pixel Spacing  : {m['PixelSpacing']} mm",
        ]
        self.meta.setPlainText("\n".join(lines))

    # --------------------------------------------------------------- hover -- #
    def _on_hover(self, plane, rc):
        if rc is None or self.volume is None:
            self.lbl_hu.setText("HU: -")
            return
        row, col = rc
        cursor = (self.cursor_z, self.cursor_y, self.cursor_x)
        z, y, x = mpr.cursor_from_pick(plane, cursor, row, col)
        hu = int(self.volume[z, y, x])
        self.lbl_hu.setText(f"HU: {hu}   (x={x}, y={y}, z={z})")

    # ---------------------------------------------------- background jobs -- #
    def _run_worker(self, fn, on_done, message="Processing...", determinate_max=0,
                     cancel_event=None):
        """Run `fn(progress_cb)` on a background QThread with a progress dialog.

        If `cancel_event` (a threading.Event) is given, the dialog gets a
        Cancel button that sets it; `fn` is expected to check it between
        stages and bail out early (a running scipy/numpy call itself can't
        be interrupted, but this stops it from starting further stages)."""
        dlg = QtWidgets.QProgressDialog(
            message, "Cancel" if cancel_event is not None else None,
            0, determinate_max, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(200)
        if cancel_event is not None:
            dlg.canceled.connect(cancel_event.set)

        worker = Worker(fn)

        def _progress(c, t):
            if t:
                dlg.setMaximum(t)
                dlg.setValue(c)

        def _done(result):
            dlg.close()
            self._active_worker = None
            on_done(result)

        def _failed(msg):
            dlg.close()
            self._active_worker = None
            QtWidgets.QMessageBox.critical(self, "Error", msg)

        worker.progress.connect(_progress)
        worker.finished_ok.connect(_done)
        worker.failed.connect(_failed)
        self._active_worker = worker  # keep a reference so Qt doesn't GC the thread
        worker.start()

    # ------------------------------------------------------ enhancement -- #
    def _on_filter_changed(self, name):
        self.active_filter = name
        self._rebuild_filter_params(name)
        self._redraw_all()
        self._update_histogram()

    def _rebuild_filter_params(self, name):
        while self.filter_param_form.rowCount():
            self.filter_param_form.removeRow(0)
        self._filter_spinboxes = {}
        if name not in enh.FILTER_PARAMS:
            return
        for pname, lo, hi, step, default in enh.FILTER_PARAMS[name]:
            spn = QtWidgets.QDoubleSpinBox()
            spn.setRange(lo, hi)
            spn.setSingleStep(step)
            spn.setValue(default)
            spn.valueChanged.connect(self._on_filter_param_changed)
            self.filter_param_form.addRow(pname, spn)
            self._filter_spinboxes[pname] = spn

    def _on_filter_param_changed(self, _value=None):
        self._redraw_all()
        self._update_histogram()

    def _current_filter_kwargs(self):
        return {name: spn.value() for name, spn in self._filter_spinboxes.items()}

    def _update_enhancement_metrics_table(self, raw_u8, processed_u8):
        try:
            scores = qm.enhancement_metrics(raw_u8, processed_u8)
        except Exception as exc:
            self.tbl_enh_metrics.setRowCount(0)
            self.lbl_enh_metrics.setText(f"Metric computation failed: {exc}")
            return
        self.lbl_enh_metrics.setText(
            f"'{self.active_filter}' vs. raw slice {self.cursor_z + 1}:")
        self.tbl_enh_metrics.setRowCount(len(scores))
        for i, (k, v) in enumerate(scores.items()):
            text = "inf" if v == float("inf") else f"{v:.4f}"
            self.tbl_enh_metrics.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self.tbl_enh_metrics.setItem(i, 1, QtWidgets.QTableWidgetItem(text))

    # ---------------------------------------------------- auto-segmentation #
    def _run_auto_segmentation(self):
        if self.volume is None:
            return
        tolerance = self.spn_aorta_tol.value()
        hu_threshold = self.spn_stent_hu.value()
        min_vol = self.spn_stent_minvol.value()
        max_vol = self.spn_stent_maxvol.value()
        min_aspect = self.spn_stent_aspect.value()
        volume = self.volume
        spacing_zyx = self.spacing_zyx
        aorta_memmap_path = os.path.join(self._scratch_dir, f"aorta_{uuid.uuid4().hex}.dat")
        stent_memmap_path = os.path.join(self._scratch_dir, f"stent_{uuid.uuid4().hex}.dat")
        cancel_event = threading.Event()
        CANCELLED = {"error": "Segmentation cancelled."}

        def job(progress_cb):
            # Every stage below works on a small cropped sub-volume (region
            # of interest around the seed / aorta bounding box) rather than
            # the full (Z, 512, 512) frame - a full-size int32 labeled array
            # alone is a ~1GB allocation, which is exactly what MemoryError'd
            # before on this memory-constrained machine. Only the final masks
            # are expanded back to full-volume shape, and that expansion is
            # memmap-backed (see `seg.embed_mask`) so it doesn't need a large
            # contiguous RAM block either.
            #
            # progress_cb(stage, total_stages) is called between stages so the
            # dialog shows real movement (rather than a bare busy spinner)
            # and so Cancel can be honored between stages instead of only at
            # the very end - a single stage itself still can't be interrupted
            # mid-call, but this bounds how much extra work runs after Cancel.
            total_stages = 5
            progress_cb(0, total_stages)
            sub_aorta, bbox_aorta, seed = seg.auto_segment_aorta(volume, tolerance=tolerance)
            if sub_aorta.size == 0 or sub_aorta.sum() == 0:
                return {"error": (
                    "Automated spine/aorta detection could not locate a plausible "
                    "descending aorta in this series. Try adjusting the HU tolerance.")}
            if cancel_event.is_set():
                return CANCELLED
            aorta_mask = seg.embed_mask(volume.shape, sub_aorta, bbox_aorta, aorta_memmap_path)
            progress_cb(1, total_stages)

            if cancel_event.is_set():
                return CANCELLED
            sub_stent_raw, bbox_stent = seg.segment_stent(
                volume, aorta_mask, hu_threshold=hu_threshold)
            progress_cb(2, total_stages)

            if cancel_event.is_set():
                return CANCELLED
            sub_stent, bbox_stent = seg.shape_filter_stent(
                sub_stent_raw, bbox_stent, spacing_zyx=spacing_zyx,
                min_vol_mm3=min_vol, max_vol_mm3=max_vol, min_aspect=min_aspect)
            progress_cb(3, total_stages)

            if cancel_event.is_set():
                return CANCELLED
            stent_mask = seg.embed_mask(volume.shape, sub_stent, bbox_stent, stent_memmap_path)
            progress_cb(4, total_stages)

            summary = seg.stent_endpoints(sub_stent, bbox_stent, spacing_zyx)
            z_endpoints = seg.stent_z_endpoints(sub_stent, bbox_stent, spacing_zyx)
            centroids = seg.per_slice_centroids(sub_stent, bbox_stent, spacing_zyx)
            progress_cb(5, total_stages)
            return {
                "aorta_mask": aorta_mask, "seed": seed, "stent_mask": stent_mask,
                "summary": summary, "z_endpoints": z_endpoints, "centroids": centroids,
            }

        self._run_worker(job, self._on_auto_segmentation_done,
                          message="Auto-segmenting aorta + stent...",
                          cancel_event=cancel_event)

    def _on_auto_segmentation_done(self, result):
        if "error" in result:
            if result["error"] == "Segmentation cancelled.":
                self.status.showMessage("Segmentation cancelled.", 4000)
            else:
                QtWidgets.QMessageBox.warning(self, "Segmentation failed", result["error"])
            return

        self.aorta_mask = result["aorta_mask"]
        self.stent_mask = result["stent_mask"]
        self.stent_report = result["summary"]
        self._populate_stent_table(result["summary"], result["z_endpoints"], result["centroids"])
        self._write_stent_csv(result["z_endpoints"])
        self.view3d.set_mask("aorta", self.aorta_mask, visible=self.chk_aorta_overlay.isChecked())
        self.view3d.set_mask("stent", self.stent_mask, visible=self.chk_stent_overlay.isChecked())
        self._redraw_all()
        self._update_histogram()
        self._update_segmentation_validation()

        if not result["z_endpoints"]:
            QtWidgets.QMessageBox.information(
                self, "No stent found",
                "No structure matching the stent shape/HU criteria was found "
                "inside the aorta mask. Try lowering the HU threshold or volume filters.")

    # ------------------------------------------------- HPC centerline pipeline #
    def _run_hpc_centerline_pipeline(self):
        if self.volume is None:
            QtWidgets.QMessageBox.information(
                self, "No series", "Load a DICOM series first.")
            return
        if not self.current_dicom_dir:
            QtWidgets.QMessageBox.warning(
                self, "DICOM folder required",
                "This pipeline reads DICOM headers directly (for frame "
                "geometry verification) and can't run against a NIfTI "
                "volume. Use Open... to load a DICOM series folder first.")
            return

        dicom_dir = self.current_dicom_dir
        iso_spacing_mm = self.spn_hpc_iso_spacing.value()

        def job(progress_cb):
            import centerline_pipeline
            return centerline_pipeline.run_local_case(
                dicom_dir, app_dir(), iso_spacing_mm=iso_spacing_mm)

        self._run_worker(
            job, self._on_hpc_pipeline_done,
            message="Running aortic centerline + stent pipeline "
                    "(TotalSegmentator + centerline + crowns)...\n"
                    "This can take several minutes.")

    def _on_hpc_pipeline_done(self, result):
        rec = result["rec"]
        status = rec.get("status")

        if result["hu_volume"] is not None:
            self._load_hpc_volume_into_viewports(result)

        if status == "OK":
            length = rec.get("stent_length_mm")
            verified = "yes" if rec.get("sop_verified") else "no (see reasons below)"
            msg = (f"Case {rec['CaseID']}: stent-graft length "
                   f"{length:.1f} mm.\nSOP-verified: {verified}\n")
            if rec.get("sop_reject_reasons"):
                msg += f"Reasons: {rec['sop_reject_reasons']}\n"
        elif status in ("partial_proximal_only", "partial_distal_only"):
            msg = (f"Case {rec['CaseID']}: only one endpoint (crown) was "
                   f"found ({status}). Length could not be computed.\n")
        else:
            msg = (f"Case {rec['CaseID']}: pipeline did not reach a stent "
                   f"measurement (status = {status}).\n")
            if rec.get("sop_reject_reasons"):
                msg += f"Details: {rec['sop_reject_reasons']}\n"

        msg += f"\nOutput folder:\n{result['case_dir']}"

        html_path = result.get("html_path")
        if html_path and os.path.isfile(html_path):
            msg += f"\n\nInteractive 3D view:\n{html_path}"
            self.status.showMessage(f"Opening {html_path}", 8000)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(html_path))
            QtWidgets.QMessageBox.information(self, "Pipeline finished", msg)
        else:
            reason = rec.get("html_export_error")
            if reason:
                msg += f"\n\nNo interactive HTML was produced: {reason}"
            elif status != "OK" and status not in (
                    "partial_proximal_only", "partial_distal_only"):
                msg += ("\n\nNo interactive HTML was produced: the pipeline "
                        "stopped before a stent surface could be meshed.")
            QtWidgets.QMessageBox.warning(self, "Pipeline finished", msg)

    def _load_hpc_volume_into_viewports(self, result):
        """Swap the 4 viewports to the pipeline's canonical (isotropic-
        resampled) CT volume, with the TotalSegmentator aorta mask overlaid.
        The stent/centerline/crown detail this pipeline also produces is not
        re-projected back onto these voxel grids -- it's in the interactive
        HTML instead, which is the deliverable for that part."""
        hu = result["hu_volume"]
        dz, dy, dx = result["spacing_zyx"]
        ox, oy, oz = result["origin_xyz"]

        memmap_path = os.path.join(self._scratch_dir, f"volume_{uuid.uuid4().hex}.dat")
        mm = np.memmap(memmap_path, dtype=np.int16, mode="w+", shape=hu.shape)
        mm[:] = hu
        mm.flush()
        self.volume = mm

        nz = hu.shape[0]
        self.positions = [[ox, oy, oz + i * dz] for i in range(nz)]
        self.iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.photometric = "MONOCHROME2"
        self.row_spacing, self.col_spacing, self.dz = dy, dx, dz
        self.wc, self.ww = 300.0, 600.0  # CT angio preset -- this is a CTA pipeline
        rec = result["rec"]
        self._meta_dict = {
            "PatientID": rec.get("PatientID") or "-",
            "PatientName": "-",
            "Modality": "CT",
            "StudyDescription": rec.get("StudyDate") or "-",
            "SeriesDescription": f"HPC pipeline canonical volume "
                                 f"({dz:.2f}mm isotropic)",
            "Manufacturer": "-",
            "Rows": hu.shape[2],
            "Columns": hu.shape[1],
            "NumSlices": nz,
            "SliceThickness": f"{dz:.3f}",
            "PixelSpacing": f"{dy:.3f} \\ {dx:.3f}",
        }

        self.aorta_mask = result["aorta_mask"]
        self.stent_mask = None
        self.stent_report = None
        self.gt_mask = None
        self.tbl_stent.setRowCount(0)
        self.lbl_stent_summary.setText(
            "Stent geometry from this run is in the interactive HTML view "
            "(not re-projected onto these voxel grids).")
        self.tbl_seg_metrics.setRowCount(0)

        self.cmb_filter.blockSignals(True)
        self.cmb_filter.setCurrentIndex(0)
        self.cmb_filter.blockSignals(False)
        self.active_filter = "None"
        self._rebuild_filter_params("None")

        ny, nx = hu.shape[1], hu.shape[2]
        self.cursor_z, self.cursor_y, self.cursor_x = nz // 2, ny // 2, nx // 2
        for slider, n in ((self.sld_axial, nz), (self.sld_sagittal, nx),
                          (self.sld_coronal, ny)):
            slider.blockSignals(True)
            slider.setRange(0, n - 1)
            slider.blockSignals(False)
        self._sync_slice_sliders()

        for view in self.planes.values():
            view._has_image = False

        self.view3d.set_volume(self.volume, self.spacing_zyx, self.wc, self.ww)
        if self.aorta_mask is not None:
            self.view3d.set_mask("aorta", self.aorta_mask,
                                 visible=self.chk_aorta_overlay.isChecked())
        self.view3d.set_mask("stent", None, visible=False)

        self._sync_wl_spins()
        self._redraw_all(fit=True)
        self._update_metadata()
        self._update_histogram()

    def _voxel_row_to_mm(self, z_index, row, col):
        position = self.positions[z_index] if z_index < len(self.positions) else [0, 0, 0]
        return seg.voxel_to_patient(
            position, row, col, self.iop[0:3], self.iop[3:6],
            self.row_spacing, self.col_spacing)

    def _write_stent_csv(self, z_endpoints):
        if not z_endpoints:
            return
        start_xyz = self._voxel_row_to_mm(
            z_endpoints["z1"], z_endpoints["row1"], z_endpoints["col1"])
        end_xyz = self._voxel_row_to_mm(
            z_endpoints["z2"], z_endpoints["row2"], z_endpoints["col2"])
        row = {
            "z1_slice": z_endpoints["z1"] + 1,
            "x1_mm": start_xyz[0], "y1_mm": start_xyz[1], "z1_mm": start_xyz[2],
            "z2_slice": z_endpoints["z2"] + 1,
            "x2_mm": end_xyz[0], "y2_mm": end_xyz[1], "z2_mm": end_xyz[2],
        }
        path = os.path.join(output_dir(), "stent_endpoints.csv")
        try:
            seg.write_stent_endpoints_csv(row, path)
            self.status.showMessage(f"Stent endpoints written to {path}", 6000)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "CSV export failed", str(exc))

    def _populate_stent_table(self, summary, z_endpoints, centroids):
        self.tbl_stent.setRowCount(0)
        if not z_endpoints:
            self.lbl_stent_summary.setText("No stent segmented yet.")
            return

        self.lbl_stent_summary.setText(
            f"Length: {summary.get('length_mm', 0.0):.1f} mm   |   "
            f"Volume: {summary.get('volume_mm3', 0.0):.0f} mm³   |   "
            f"Voxels: {summary.get('voxel_count', 0)}")

        rows = []
        for point_label, z, r, c in (
            ("Start", z_endpoints["z1"], z_endpoints["row1"], z_endpoints["col1"]),
            ("End", z_endpoints["z2"], z_endpoints["row2"], z_endpoints["col2"]),
        ):
            xyz = self._voxel_row_to_mm(z, r, c)
            rows.append((point_label, z + 1, xyz))
        for c in centroids:
            xyz = self._voxel_row_to_mm(c["z_index"], c["row"], c["col"])
            rows.append(("Slice centroid", c["z_index"] + 1, xyz))

        self.tbl_stent.setRowCount(len(rows))
        for i, (point_label, slice_no, xyz) in enumerate(rows):
            values = [point_label, str(slice_no), f"{xyz[0]:.1f}", f"{xyz[1]:.1f}", f"{xyz[2]:.1f}"]
            highlight = point_label in ("Start", "End")
            for col_idx, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if highlight:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(QtGui.QColor(45, 65, 90))
                self.tbl_stent.setItem(i, col_idx, item)

    # ------------------------------------------------ segmentation validation #
    def _load_ground_truth(self):
        if self.volume is None:
            QtWidgets.QMessageBox.information(self, "No series", "Load a DICOM series first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ground-truth mask", "", "NumPy array (*.npy)")
        if not path:
            return
        try:
            gt = np.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))
            return
        if gt.shape != self.volume.shape:
            QtWidgets.QMessageBox.critical(
                self, "Shape mismatch",
                f"Ground-truth mask shape {gt.shape} does not match the loaded "
                f"volume shape {self.volume.shape}.")
            return
        self.gt_mask = gt.astype(np.uint8)
        self._update_segmentation_validation()

    def _update_segmentation_validation(self):
        if self.gt_mask is None or self.stent_mask is None:
            self.tbl_seg_metrics.setRowCount(0)
            self.lbl_seg_validation.setText(
                "Load a ground-truth mask to compare against the segmented stent.")
            return
        scores = qm.segmentation_metrics(self.stent_mask, self.gt_mask, self.spacing_zyx)
        self.lbl_seg_validation.setText("Segmented stent mask vs. loaded ground truth:")
        self.tbl_seg_metrics.setRowCount(len(scores))
        for i, (k, v) in enumerate(scores.items()):
            text = "inf" if v == float("inf") else f"{v:.4f}"
            self.tbl_seg_metrics.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self.tbl_seg_metrics.setItem(i, 1, QtWidgets.QTableWidgetItem(text))

    # ----------------------------------------------------------- histogram #
    def _update_histogram(self, *_args):
        self.hist_ax.clear()
        if self.volume is not None:
            source = self.cmb_hist_source.currentText()
            if source == "Current Slice (raw vs filtered)" and self._last_raw_u8 is not None:
                self.hist_ax.hist(self._last_raw_u8.ravel(), bins=64,
                                   color="#888888", alpha=0.6, label="Raw")
                if self._last_filtered_u8 is not None:
                    self.hist_ax.hist(self._last_filtered_u8.ravel(), bins=64,
                                       color="#3498db", alpha=0.6, label="Filtered")
                self.hist_ax.legend(fontsize=7)
                self.hist_ax.set_title(
                    f"Intensity histogram - slice {self.cursor_z + 1} "
                    f"('{self.active_filter}')", fontsize=9)
                self.hist_ax.set_xlabel("Intensity (0-255)")
                self.hist_ax.set_ylabel("Count")
                self.hist_figure.tight_layout()
                self.hist_canvas.draw()
                return
            if source == "Aorta" and self.aorta_mask is not None:
                data = self.volume[self.aorta_mask.astype(bool)]
                title = "HU histogram - Aorta"
            elif source == "Stent" and self.stent_mask is not None:
                data = self.volume[self.stent_mask.astype(bool)]
                title = "HU histogram - Stent"
            else:
                data = self.volume[::4, ::2, ::2]
                title = "HU histogram - Whole volume (subsampled)"
            if data.size:
                self.hist_ax.hist(data.ravel(), bins=100, color="#3498db")
            self.hist_ax.set_title(title, fontsize=9)
            self.hist_ax.set_xlabel("HU")
            self.hist_ax.set_ylabel("Count")
        self.hist_figure.tight_layout()
        self.hist_canvas.draw()

    # -------------------------------------------------------------- export #
    def _export_images(self):
        if self.volume is None:
            QtWidgets.QMessageBox.warning(self, "No series", "Load a DICOM series first.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder")
        if not folder:
            return
        fmt, ok = QtWidgets.QInputDialog.getItem(
            self, "Export format", "Image format:", ["PNG", "JPG"], 0, False)
        if not ok:
            return
        ext = fmt.lower()

        volume = self.volume
        wc, ww, photometric = self.wc, self.ww, self.photometric
        n = volume.shape[0]

        def job(progress_cb):
            for i in range(n):
                u8 = window_slice(volume[i], wc, ww)
                pm = to_pixmap(u8, photometric)
                pm.save(os.path.join(folder, f"slice_{i + 1:04d}.{ext}"))
                progress_cb(i + 1, n)
            return n

        def done(count):
            QtWidgets.QMessageBox.information(
                self, "Export complete", f"Exported {count} slices to:\n{folder}")

        self._run_worker(job, done, message="Exporting images...", determinate_max=n)

    # ----------------------------------------------------------- keyboard -- #
    def keyPressEvent(self, e):
        if self.volume is None:
            return super().keyPressEvent(e)
        # Arrow/Page/Home/End navigate the axial (Z) plane, matching the
        # original single-view behaviour.
        k = e.key()
        if k in (Qt.Key_Down, Qt.Key_Right):
            self._set_cursor(self.cursor_z + 1, self.cursor_y, self.cursor_x)
        elif k in (Qt.Key_Up, Qt.Key_Left):
            self._set_cursor(self.cursor_z - 1, self.cursor_y, self.cursor_x)
        elif k == Qt.Key_PageDown:
            self._set_cursor(self.cursor_z + 10, self.cursor_y, self.cursor_x)
        elif k == Qt.Key_PageUp:
            self._set_cursor(self.cursor_z - 10, self.cursor_y, self.cursor_x)
        elif k == Qt.Key_Home:
            self._set_cursor(0, self.cursor_y, self.cursor_x)
        elif k == Qt.Key_End:
            self._set_cursor(self.volume.shape[0] - 1, self.cursor_y, self.cursor_x)
        else:
            super().keyPressEvent(e)


def main():
    if "--selftest" in sys.argv:
        _run_selftest()
        return
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(app_icon())
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


# --------------------------------------------------------------------------- #
#  Headless self-test - exercises the same code paths a user would hit
#  (DICOM read, NIfTI read, windowing/enhancement, auto-segmentation, 3D
#  render) without showing a window, so a build script can catch missing
#  hidden-imports/DLLs in a frozen exe automatically.
#
#  Usage:  MiniCTViewer.exe --selftest [--data-dir PATH_TO_DICOM_FOLDER]
#
#  Exits non-zero only for hard failures (missing modules, load errors).
#  The VTK 3D-render check is reported but doesn't fail the run, since a
#  CI/build machine may have no GPU or software-OpenGL driver - that's an
#  environment limitation, not a packaging bug.
# --------------------------------------------------------------------------- #
def _run_selftest():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    hard_failures = []
    tmp_root = tempfile.mkdtemp(prefix="ctviewer_selftest_")

    def step(name, fn, critical=True):
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            if critical:
                hard_failures.append(name)

    app = QtWidgets.QApplication(sys.argv)
    loaded = {}

    def _load_via(loader):
        result, errors = {}, []
        loader.finished_ok.connect(result.update)
        loader.failed.connect(errors.append)
        loader.run()  # synchronous: exercise the same code without a thread
        if errors:
            raise RuntimeError(errors[0])
        if "volume" not in result:
            raise RuntimeError("loader produced no result")
        return result

    def _dicom():
        data_dir = None
        if "--data-dir" in sys.argv:
            data_dir = sys.argv[sys.argv.index("--data-dir") + 1]
        else:
            default_sample = resource_path("5119278-1")
            if os.path.isdir(default_sample):
                data_dir = default_sample
        if not data_dir:
            raise RuntimeError("no sample DICOM folder found (pass --data-dir)")
        catalogue = scan_folder(data_dir)
        if not catalogue:
            raise RuntimeError("no DICOM series found in " + data_dir)
        entry = max(catalogue.values(), key=lambda e: e.n_slices)
        memmap_path = os.path.join(tmp_root, "dicom_volume.dat")
        loaded["dicom"] = _load_via(SeriesLoader(entry, memmap_path))

    def _nifti():
        if nib is None:
            raise RuntimeError("nibabel not installed")
        arr = (np.random.rand(12, 24, 24) * 500 - 100).astype(np.float32)
        nii_path = os.path.join(tmp_root, "synthetic.nii.gz")
        nib.save(nib.Nifti1Image(arr, np.eye(4)), nii_path)
        memmap_path = os.path.join(tmp_root, "nifti_volume.dat")
        loaded["nifti"] = _load_via(NiftiLoader(nii_path, memmap_path))

    def _windowing_and_enhancement():
        src = loaded.get("dicom") or loaded.get("nifti")
        if not src:
            raise RuntimeError("no volume loaded to test windowing on")
        vol = src["volume"]
        u8 = window_slice(vol[vol.shape[0] // 2], src["wc"], src["ww"])
        for name, fn in enh.FILTERS.items():
            kwargs = {p: d for p, _lo, _hi, _step, d in enh.FILTER_PARAMS.get(name, [])}
            fn(u8, **kwargs)

    def _segmentation():
        src = loaded.get("dicom")
        if not src:
            raise RuntimeError("segmentation self-test needs the DICOM sample")
        sub_aorta, _bbox, _seed = seg.auto_segment_aorta(src["volume"], tolerance=120)
        if sub_aorta.sum() == 0:
            raise RuntimeError("no aorta found in sample series")

    def _volume3d():
        src = loaded.get("dicom") or loaded.get("nifti")
        if not src:
            raise RuntimeError("no volume loaded to test 3D render on")
        widget = Volume3DWidget()
        widget.resize(64, 64)
        widget.show()
        widget.set_volume(src["volume"], (1.0, 1.0, 1.0), src["wc"], src["ww"])
        widget.render()
        widget.close()

    step("DICOM series read", _dicom)
    step("NIfTI volume read", _nifti)
    step("Windowing + enhancement filters", _windowing_and_enhancement)
    step("Automated aorta/stent segmentation", _segmentation)
    step("VTK 3D volume render (needs GPU/software OpenGL)", _volume3d, critical=False)

    shutil.rmtree(tmp_root, ignore_errors=True)
    print("\nSELFTEST " + ("FAILED: " + ", ".join(hard_failures) if hard_failures else "PASSED"))
    sys.exit(1 if hard_failures else 0)


if __name__ == "__main__":
    main()

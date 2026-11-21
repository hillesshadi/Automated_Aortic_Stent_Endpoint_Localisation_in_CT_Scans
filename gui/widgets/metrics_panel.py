#!/usr/bin/env python3
"""
Evaluation panel: enhancement quality, segmentation morphometry, HU histograms.

Three tabs rather than one long table, because the three answer different
questions and are computed at different times -- enhancement metrics only mean
something once a filter is selected, segmentation metrics only once a mask
exists.

The histogram tab is a plot rather than a table on purpose: the shape of the
HU distribution is what tells you whether a threshold sits in a valley or
halfway up a peak, and that is invisible in 64 rows of counts. The numbers are
still written to CSV on export.
"""

import numpy as np

from ..qt_compat import QtWidgets, Qt, Signal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure                                             # noqa: E402

STRUCTURE_COLORS = {
    "aorta": "#e04447",
    "stent": "#4ddcff",
    "whole_volume": "#93a2c0",
}
# Rows of the morphometry table, in the order they are shown.
MORPHOMETRY_ROWS = (
    ("voxels", "Voxels", "%d"),
    ("volume_ml", "Volume (mL)", "%.2f"),
    ("volume_mm3", "Volume (mm³)", "%.0f"),
    ("hu_mean", "HU mean", "%.1f"),
    ("hu_sd", "HU sd", "%.1f"),
    ("hu_min", "HU min", "%.0f"),
    ("hu_max", "HU max", "%.0f"),
    ("bbox_mm", "Bounding box (mm)", None),
)


class MetricsPanel(QtWidgets.QWidget):
    enhancement_requested = Signal()
    segmentation_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Compute enhancement metrics")
        b.setToolTip("MSE / PSNR / SSIM of the selected filter against the "
                     "raw slices.")
        b.clicked.connect(self.enhancement_requested.emit)
        bar.addWidget(b, 1)
        b = QtWidgets.QPushButton("Compute segmentation metrics")
        b.setToolTip("Morphometry and HU histograms for aorta, stent and the "
                     "whole volume.")
        b.clicked.connect(self.segmentation_requested.emit)
        bar.addWidget(b, 1)
        self.btn_export = QtWidgets.QPushButton("Export metrics…")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_requested.emit)
        bar.addWidget(self.btn_export)
        layout.addLayout(bar)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        # ---- enhancement ------------------------------------------------- #
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        self.enh_summary = QtWidgets.QLabel(
            "<i>Select a filter in the Filtering group, then compute.</i>")
        self.enh_summary.setWordWrap(True)
        self.enh_summary.setStyleSheet("color:#93a2c0;")
        v.addWidget(self.enh_summary)
        self.enh_table = _table(("Metric", "Mean", "SD", "Min", "Max"))
        v.addWidget(self.enh_table, 1)
        self.enh_slices = _table(("Slice", "MSE", "PSNR (dB)", "SSIM", "FSIM"))
        v.addWidget(QtWidgets.QLabel("Per slice"))
        v.addWidget(self.enh_slices, 2)
        self.tabs.addTab(page, "Enhancement")

        # ---- segmentation ------------------------------------------------ #
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)
        self.seg_table = _table(("Measure", "Aorta", "Stent", "Whole volume"))
        v.addWidget(QtWidgets.QLabel("Morphometry"))
        v.addWidget(self.seg_table, 1)
        v.addWidget(QtWidgets.QLabel("Agreement (Dice / IoU / Hausdorff)"))
        self.agree_table = _table(("Structure", "DSC", "IoU", "HD (mm)"))
        v.addWidget(self.agree_table, 1)
        self.agree_note = QtWidgets.QLabel("—")
        self.agree_note.setWordWrap(True)
        self.agree_note.setStyleSheet("color:#93a2c0;")
        v.addWidget(self.agree_note)
        self.tabs.addTab(page, "Segmentation")

        # ---- histograms -------------------------------------------------- #
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 3.2), facecolor="#0e1219")
        self.canvas = FigureCanvas(self.figure)
        v.addWidget(self.canvas, 1)
        self.tabs.addTab(page, "HU histograms")
        self._draw_histograms(None)

    # ------------------------------------------------------------------ API
    def set_enhancement(self, report):
        if not report:
            return
        params = ", ".join("%s=%s" % kv
                           for kv in (report.get("filter_params") or {}).items())
        self.enh_summary.setText(
            "<b>%s</b>%s · %d slices · window %d/%d HU"
            % (report.get("filter", "?"), " (%s)" % params if params else "",
               report.get("n_slices", 0),
               report.get("window", {}).get("wc", 0),
               report.get("window", {}).get("ww", 0)))

        summary = report.get("summary", {})
        self.enh_table.setRowCount(len(summary))
        for r, (key, stats) in enumerate(summary.items()):
            _put(self.enh_table, r, 0, key, left=True)
            for c, stat in enumerate(("mean", "sd", "min", "max"), start=1):
                _put(self.enh_table, r, c, _fmt(key, stats.get(stat)))
        self.enh_table.resizeColumnsToContents()

        rows = report.get("per_slice", [])
        self.enh_slices.setRowCount(len(rows))
        for r, row in enumerate(rows):
            _put(self.enh_slices, r, 0, str(row.get("slice", "")))
            for c, key in enumerate(("MSE", "PSNR", "SSIM", "FSIM"), start=1):
                _put(self.enh_slices, r, c, _fmt(key, row.get(key)))
        self.enh_slices.resizeColumnsToContents()
        self.btn_export.setEnabled(True)
        self.tabs.setCurrentIndex(0)

    def set_segmentation(self, report):
        if not report:
            return
        structures = report.get("structures", {})
        names = ("aorta", "stent", "whole_volume")
        self.seg_table.setRowCount(len(MORPHOMETRY_ROWS))
        for r, (key, label, fmt) in enumerate(MORPHOMETRY_ROWS):
            _put(self.seg_table, r, 0, label, left=True)
            for c, name in enumerate(names, start=1):
                entry = structures.get(name)
                if entry is None or key not in entry:
                    _put(self.seg_table, r, c, "—")
                    continue
                value = entry[key]
                if key == "bbox_mm":
                    text = ("—" if not value else
                            " × ".join("%.0f" % v for v in value))
                elif value is None or (isinstance(value, float)
                                       and value != value):
                    text = "—"
                else:
                    text = fmt % value
                _put(self.seg_table, r, c, text)
        self.seg_table.resizeColumnsToContents()

        agreement = report.get("agreement", {})
        self.agree_table.setRowCount(len(agreement))
        for r, (name, vals) in enumerate(agreement.items()):
            _put(self.agree_table, r, 0, name, left=True)
            for c, key in enumerate(("DSC", "IoU", "HD"), start=1):
                value = vals.get(key, vals.get(key.lower()))
                _put(self.agree_table, r, c,
                     "—" if value is None else "%.4f" % value)
        self.agree_table.resizeColumnsToContents()
        self.agree_note.setText(report.get("agreement_note", ""))

        self._draw_histograms(report)
        self.btn_export.setEnabled(True)
        self.tabs.setCurrentIndex(1)

    # ------------------------------------------------------------- plotting
    def _draw_histograms(self, report):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#0b0e14")
        for spine in ax.spines.values():
            spine.set_color("#2a3347")
        ax.tick_params(colors="#93a2c0", labelsize=8)
        ax.set_xlabel("Hounsfield units", color="#93a2c0", fontsize=9)
        ax.set_ylabel("voxels (log)", color="#93a2c0", fontsize=9)

        structures = (report or {}).get("structures", {})
        drawn = False
        for name, entry in structures.items():
            h = entry.get("histogram")
            if not h or not h.get("n"):
                continue
            edges = np.asarray(h["edges"], float)
            centres = 0.5 * (edges[:-1] + edges[1:])
            counts = np.asarray(h["counts"], float)
            ax.step(centres, np.maximum(counts, 0.5), where="mid",
                    color=STRUCTURE_COLORS.get(name, "#93a2c0"), lw=1.2,
                    label="%s (n=%s)" % (name.replace("_", " "), f"{h['n']:,}"))
            drawn = True

        if drawn:
            ax.set_yscale("log")
            legend = ax.legend(fontsize=8, facecolor="#12161f",
                               edgecolor="#2a3347", labelcolor="#c8d3e6")
            legend.get_frame().set_alpha(0.9)
        else:
            ax.text(0.5, 0.5, "compute segmentation metrics",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#3a465e", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        self.figure.tight_layout()
        self.canvas.draw_idle()


# --------------------------------------------------------------------------- #
def _table(headers):
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.setSelectionBehavior(
        QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _put(table, row, col, text, left=False):
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(
        (Qt.AlignmentFlag.AlignLeft if left else Qt.AlignmentFlag.AlignRight)
        | Qt.AlignmentFlag.AlignVCenter)
    table.setItem(row, col, item)


def _fmt(key, value):
    """Per-metric precision: PSNR in dB needs 2 dp, SSIM needs 4."""
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if key == "MSE":
        return "%.3f" % value
    if key == "PSNR":
        return "%.2f" % value
    return "%.4f" % value

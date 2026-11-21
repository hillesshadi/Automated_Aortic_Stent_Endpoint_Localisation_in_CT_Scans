#!/usr/bin/env python3
"""
Stent specification loader and the landmark table it produces.

The device parameters come from a JSON file rather than from GUI widgets --
see pipeline/stent_specs.py for why. This panel is the view over that: it shows
what was loaded, lets the operator override the two anchors by picking stations
in the MPR panel, and renders the resulting landing-zone landmarks with their
patient coordinates.

Selecting a row emits `station_requested`, so clicking a landmark drives the
2-D and 3-D views to it; a table of coordinates you cannot navigate to is a
report, not a tool.
"""

import os

from ..qt_compat import QtWidgets, Qt, Signal

COLUMNS = ("Landmark", "Arc (mm)", "X (mm)", "Y (mm)", "Z (mm)",
           "Voxel (i, j, k)", "Lumen ⌀ (mm)")


class StentPanel(QtWidgets.QWidget):
    specs_requested = Signal()           # "Upload Stent Specs JSON…"
    template_requested = Signal()
    compute_requested = Signal()
    export_requested = Signal()
    pick_requested = Signal(str)         # "proximal" | "distal" | "clear"
    station_requested = Signal(int)      # navigate to a landmark's station

    def __init__(self, parent=None):
        super().__init__(parent)
        self._table = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ---- device ------------------------------------------------------ #
        box = QtWidgets.QGroupBox("Device specification")
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        btn = QtWidgets.QPushButton("Upload Stent Specs JSON…")
        btn.clicked.connect(self.specs_requested.emit)
        btn.setMinimumHeight(28)
        row.addWidget(btn, 1)
        tpl = QtWidgets.QPushButton("Save template…")
        tpl.setToolTip("Write a starter JSON file with every expected field.")
        tpl.clicked.connect(self.template_requested.emit)
        row.addWidget(tpl)
        v.addLayout(row)

        self.specs_label = QtWidgets.QLabel(
            "<i>No specification loaded. Landing-zone landmarks need "
            "proximal_land_mm, distal_land_mm, target_length_mm and "
            "diameter_mm.</i>")
        self.specs_label.setWordWrap(True)
        self.specs_label.setStyleSheet("color:#93a2c0;")
        v.addWidget(self.specs_label)
        layout.addWidget(box)

        # ---- anchors ----------------------------------------------------- #
        box = QtWidgets.QGroupBox("Anchors")
        v = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel(
            "Landmarks anchor to the fitted stent crowns. Override either end "
            "by scrubbing the station slider and pressing the matching button.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7f8ba3; font-size:11px;")
        v.addWidget(hint)
        row = QtWidgets.QHBoxLayout()
        for label, key in (("Set proximal", "proximal"),
                           ("Set distal", "distal"), ("Clear", "clear")):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _c=False, k=key: self.pick_requested.emit(k))
            row.addWidget(b)
        v.addLayout(row)
        self.anchor_label = QtWidgets.QLabel("Anchors: fitted stent crowns")
        self.anchor_label.setStyleSheet("color:#93a2c0;")
        self.anchor_label.setWordWrap(True)
        v.addWidget(self.anchor_label)
        layout.addWidget(box)

        # ---- landmarks --------------------------------------------------- #
        box = QtWidgets.QGroupBox("Stent landmarks")
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Compute landmarks")
        b.setMinimumHeight(28)
        b.clicked.connect(self.compute_requested.emit)
        row.addWidget(b, 1)
        self.btn_export = QtWidgets.QPushButton("Export CSV / JSON…")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_requested.emit)
        row.addWidget(self.btn_export)
        v.addLayout(row)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.setToolTip("Select a row to drive the 2-D and 3-D views to "
                              "that landmark.")
        v.addWidget(self.table, 1)

        self.summary = QtWidgets.QLabel("—")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setStyleSheet("color:#93a2c0;")
        v.addWidget(self.summary)
        layout.addWidget(box, 1)

    # ------------------------------------------------------------------ API
    def set_specs(self, specs):
        if not specs:
            self.specs_label.setText("<i>No specification loaded.</i>")
            return
        name = specs.get("device") or os.path.basename(
            specs.get("source_file", "specification"))
        parts = ["<b>%s</b>" % name]
        if specs.get("manufacturer"):
            parts.append(specs["manufacturer"])
        parts.append("⌀ %.1f mm · length %.0f mm"
                     % (specs.get("diameter_mm", 0.0),
                        specs.get("target_length_mm", 0.0)))
        parts.append("landing zones %.0f / %.0f mm (prox / dist)"
                     % (specs.get("proximal_land_mm", 0.0),
                        specs.get("distal_land_mm", 0.0)))
        if specs.get("oversizing_pct") is not None:
            parts.append("planned oversizing %.0f%%" % specs["oversizing_pct"])
        self.specs_label.setText(" · ".join(parts))

    def set_anchor_note(self, text):
        self.anchor_label.setText("Anchors: %s" % text)

    def set_table(self, table):
        """Render a landmark table from pipeline.stent_specs.landmark_table."""
        self._table = table
        rows = (table or {}).get("rows", [])
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            ijk = ("%s, %s, %s" % (row.get("i"), row.get("j"), row.get("k"))
                   if row.get("i") is not None else "—")
            diameter = row.get("lumen_diameter_mm")
            values = [
                row.get("label", ""),
                "%.1f" % row.get("arc_mm", 0.0),
                "%.2f" % row.get("x_mm", 0.0),
                "%.2f" % row.get("y_mm", 0.0),
                "%.2f" % row.get("z_mm", 0.0),
                ijk,
                "—" if diameter is None or diameter != diameter
                else "%.1f" % diameter,
            ]
            for c, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if row.get("clamped"):
                    # The landing zone ran past the end of the centerline, so
                    # this row is the clamped position, not the requested one.
                    item.setForeground(Qt.GlobalColor.yellow)
                    item.setToolTip("Clamped to the end of the centerline — "
                                    "the requested position is outside the "
                                    "scanned volume.")
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.btn_export.setEnabled(bool(rows))
        self._render_summary(table)

    def _render_summary(self, table):
        s = (table or {}).get("summary", {})
        if not s:
            self.summary.setText("—")
            return
        parts = ["anchored on <b>%s</b>" % s.get("anchor", "?"),
                 "covered <b>%.1f mm</b>" % s.get("covered_length_mm", 0.0)]
        if "target_length_mm" in s:
            parts.append("device %.0f mm (%+.1f mm)"
                         % (s["target_length_mm"],
                            s.get("length_difference_mm", 0.0)))
        if "min_landing_diameter_mm" in s:
            parts.append("narrowest landing ⌀ %.1f mm"
                         % s["min_landing_diameter_mm"])
        if "oversizing_pct" in s:
            parts.append("oversizing %.0f%%" % s["oversizing_pct"])
        text = " · ".join(parts)
        for w in s.get("warnings", []):
            text += "<br><span style='color:#ffcc66;'>⚠ %s</span>" % w
        self.summary.setText(text)

    def _on_select(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self._table:
            return
        r = rows[0].row()
        data = self._table.get("rows", [])
        if 0 <= r < len(data):
            self.station_requested.emit(int(data[r].get("index", 0)))

#!/usr/bin/env python3
"""
2-D panel: the three axis-aligned MPR planes plus the centerline-orthogonal
cross-section, in a gapless 2x2 grid.

The fourth view is the point of this panel. Axial/sagittal/coronal are plain
array slices (ct_viewer/mpr.py), but a plane perpendicular to a curved vessel
has to be resampled by interpolation -- src/oblique_mpr.py does that, and the
station slider drives both it and the crosshair in the other three views, so
all four always describe the same point in the vessel.

Layout note: the axes are placed with explicit figure rectangles rather than
subplots(), because the subplot machinery reserves room for titles and tick
labels even when both are empty, which is what produced the gaps between
viewports. Titles are drawn inside the axes instead.

Navigation is implemented here rather than with matplotlib's NavigationToolbar
because the toolbar is modal (you pick pan *or* zoom) and occupies a strip of
the panel; radiology viewers expect scroll-to-zoom and middle-drag-to-pan to be
live at all times.
"""

import numpy as np

from ..qt_compat import QtWidgets, Qt, Signal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure                                             # noqa: E402
from matplotlib.patches import Circle                                            # noqa: E402

try:
    import mpr as mpr_mod
    MPR_ERROR = None
except Exception as exc:                                       # noqa: BLE001
    mpr_mod = None
    MPR_ERROR = "%s: %s" % (type(exc).__name__, exc)

VIEWS = ("axial", "sagittal", "coronal", "orthogonal")

# Figure rectangles [left, bottom, width, height] -- a true 2x2 tiling with no
# gutters. Order on screen: top-left, top-right, bottom-left, bottom-right.
RECTS = {
    "axial":      [0.0, 0.5, 0.5, 0.5],
    "sagittal":   [0.5, 0.5, 0.5, 0.5],
    "coronal":    [0.0, 0.0, 0.5, 0.5],
    "orthogonal": [0.5, 0.0, 0.5, 0.5],
}

CROSSHAIR_COLOR = {
    "axial": "#00e5ff", "sagittal": "#00e5ff", "coronal": "#00e5ff",
    # The orthogonal view is centred on the centerline point, not on the voxel
    # cursor, so it gets a different colour to keep that distinction visible.
    "orthogonal": "#ffd24a",
}
ZOOM_STEP = 1.25


class MPRPanel(QtWidgets.QWidget):
    """Four synchronized 2-D views driven by one centerline station."""

    station_changed = Signal(int)
    cursor_info = Signal(str)                # live coordinate / HU readout
    cursor_picked = Signal(int, int, int)    # (z, y, x) after a left-click

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = None
        self._params = {}
        self._station = 0
        self._resampler_id = None
        self._profile = None
        self._cursor_override = None     # set by clicking, cleared on reload
        self._view_limits = {}           # name -> (xlim, ylim), survives redraw
        self._nav = None                 # in-flight pan/zoom drag

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.figure = Figure(figsize=(7, 6), facecolor="#0b0e14")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Expanding)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.canvas, 1)

        self.axes = {}
        for name in VIEWS:
            ax = self.figure.add_axes(RECTS[name])
            ax.set_facecolor("#0b0e14")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#2a3347")
                spine.set_linewidth(0.8)
            self.axes[name] = ax

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(6, 2, 6, 0)
        controls.addWidget(QtWidgets.QLabel("Station along centerline"))
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        controls.addWidget(self.slider, 1)
        self.station_label = QtWidgets.QLabel("—")
        self.station_label.setMinimumWidth(190)
        controls.addWidget(self.station_label)
        reset = QtWidgets.QToolButton()
        reset.setText("Reset zoom")
        reset.clicked.connect(self.reset_zoom)
        controls.addWidget(reset)
        layout.addLayout(controls)

        # Dedicated live-coordinate strip, kept separate from the measurement
        # readout so a moving cursor never overwrites the lumen numbers.
        self.coords = QtWidgets.QLabel("—")
        self.coords.setStyleSheet(
            "color:#7fd1ff; background:#12161f; border-top:1px solid #2a3347;"
            "padding:3px 6px; font-family:Consolas,monospace;")
        layout.addWidget(self.coords)

        self.readout = QtWidgets.QLabel("Load a volume to begin.")
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet("color:#93a2c0; padding:2px 6px;")
        layout.addWidget(self.readout)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("figure_leave_event",
                                lambda _e: self._set_coords("—"))
        self._draw_placeholder()

    # ------------------------------------------------------------------ data
    def set_context(self, ctx, params):
        new_case = ctx is not self.ctx
        self.ctx = ctx
        self._params = dict(params or {})
        if new_case:
            self._cursor_override = None
            self._view_limits = {}
        if ctx is not None and ctx.resampler is not None:
            n = len(ctx.resampler)
            self.slider.setEnabled(True)
            self.slider.setRange(0, max(0, n - 1))
            # Start mid-vessel on a newly extracted centerline, but leave the
            # station alone when this is only a parameter/overlay refresh --
            # otherwise scrubbing would jump back on every redraw.
            if id(ctx.resampler) != self._resampler_id or self._station >= n:
                self._resampler_id = id(ctx.resampler)
                self._station = n // 2
                self.slider.setValue(self._station)
        else:
            self.slider.setEnabled(False)
            self._resampler_id = None
        self.refresh()

    def set_params(self, params):
        self._params.update(params or {})
        self.refresh()

    def set_profile(self, profile):
        self._profile = profile
        self.refresh()

    def station(self):
        return self._station

    def _on_slider(self, value):
        self._station = int(value)
        # A new station defines a new intersection point; drop any click-set
        # cursor so the three axis views follow the vessel again.
        self._cursor_override = None
        self.station_changed.emit(self._station)
        self.refresh()

    def reset_zoom(self):
        self._view_limits = {}
        self.refresh()

    # --------------------------------------------------------------- drawing
    def _draw_placeholder(self):
        for name, ax in self.axes.items():
            ax.clear()
            ax.set_facecolor("#0b0e14")
            ax.set_xticks([])
            ax.set_yticks([])
            self._label(ax, name.capitalize())
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    color="#3a465e", transform=ax.transAxes, fontsize=9)
        self.canvas.draw_idle()

    @staticmethod
    def _label(ax, text, color="#93a2c0"):
        """Title drawn inside the axes -- there is no margin to put one in."""
        ax.text(0.012, 0.978, text, transform=ax.transAxes, ha="left",
                va="top", color=color, fontsize=8,
                bbox=dict(facecolor="#0b0e14", edgecolor="none",
                          alpha=0.65, pad=1.8))

    def refresh(self):
        ctx = self.ctx
        if ctx is None or not ctx.has_volume:
            self._draw_placeholder()
            return
        if mpr_mod is None:
            self.readout.setText("mpr.py unavailable: %s" % MPR_ERROR)
            return

        wc = int(self._params.get("wc", 300))
        ww = int(self._params.get("ww", 600))
        cursor = self.cursor()

        for plane in ("axial", "sagittal", "coronal"):
            self._draw_plane(plane, cursor, wc, ww)
        self._draw_orthogonal(wc, ww)
        self._restore_limits()
        self.canvas.draw_idle()

    def cursor(self):
        """Current (z, y, x) voxel cursor: an explicit click wins, else station."""
        if self._cursor_override is not None:
            return self._cursor_override
        return self._cursor_for_station()

    def _cursor_for_station(self):
        """Voxel cursor for the current centerline station.

        The three axis-aligned views are indexed in voxels, the centerline is
        in world millimetres, so the station has to be mapped back through the
        volume's own affine rather than assumed to be an index.
        """
        ctx = self.ctx
        nz, ny, nx = ctx.volume.shape
        if ctx.resampler is None:
            return (nz // 2, ny // 2, nx // 2)
        from stent_endpoint_sop import world_to_idx
        i = int(np.clip(self._station, 0, len(ctx.resampler) - 1))
        world = ctx.resampler.points[i]
        idx_xyz = world_to_idx(world, ctx.geom)
        z = int(np.clip(round(idx_xyz[2]), 0, nz - 1))
        y = int(np.clip(round(idx_xyz[1]), 0, ny - 1))
        x = int(np.clip(round(idx_xyz[0]), 0, nx - 1))
        return (z, y, x)

    def _draw_plane(self, plane, cursor, wc, ww):
        ctx = self.ctx
        ax = self.axes[plane]
        ax.clear()
        ax.set_facecolor("#0b0e14")
        ax.set_xticks([])
        ax.set_yticks([])

        raw = mpr_mod.plane_slice(ctx.volume, plane, cursor)
        img = _window(raw, wc, ww)
        aspect = mpr_mod.plane_aspect(plane, ctx.spacing_zyx)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect=aspect,
                  interpolation="bilinear", origin="upper")
        # imshow defaults to adjustable="box", which shrinks the axes to honour
        # the pixel aspect and so reintroduces the gaps this layout exists to
        # remove. Adjusting the data limits instead keeps the viewport
        # edge-to-edge and letterboxes the slice inside it, still undistorted.
        ax.set_adjustable("datalim")

        if self._params.get("show_aorta", True) and ctx.aorta_mask is not None:
            m = mpr_mod.plane_slice(ctx.aorta_mask, plane, cursor)
            ax.imshow(_mask_rgba(m, (0.86, 0.26, 0.28), 0.30), aspect=aspect,
                      interpolation="nearest", origin="upper")
        if self._params.get("show_stent", True) and ctx.stent_mask is not None:
            m = mpr_mod.plane_slice(ctx.stent_mask, plane, cursor)
            ax.imshow(_mask_rgba(m, (0.30, 0.95, 1.00), 0.75), aspect=aspect,
                      interpolation="nearest", origin="upper")

        row, col = mpr_mod.plane_crosshair(plane, cursor)
        color = CROSSHAIR_COLOR[plane]
        ax.axhline(row, color=color, lw=0.7, alpha=0.8)
        ax.axvline(col, color=color, lw=0.7, alpha=0.8)
        self._label(ax, "%s  z=%d y=%d x=%d" % (plane.capitalize(), *cursor))

    def _draw_orthogonal(self, wc, ww):
        ctx = self.ctx
        ax = self.axes["orthogonal"]
        ax.clear()
        ax.set_facecolor("#0b0e14")
        ax.set_xticks([])
        ax.set_yticks([])

        if ctx.resampler is None:
            self._label(ax, "Orthogonal to centerline", "#5d6b85")
            ax.text(0.5, 0.5, "run Extract Centerline", ha="center",
                    va="center", color="#3a465e", transform=ax.transAxes,
                    fontsize=9)
            self.station_label.setText("—")
            return

        from pipeline.centerline import cross_section_at
        try:
            patch, extent, metrics, info = cross_section_at(
                ctx, self._station,
                fov_mm=self._params.get("cl_fov", 60),
                res_mm=self._params.get("cl_res", 0.4),
                lumen_hu=self._params.get("lumen_hu", 200))
        except Exception as exc:                               # noqa: BLE001
            self._label(ax, "Orthogonal to centerline", "#5d6b85")
            ax.text(0.5, 0.5, str(exc)[:80], ha="center", va="center",
                    color="#ff6b6b", transform=ax.transAxes, fontsize=7,
                    wrap=True)
            return

        lo, hi = extent
        ax.imshow(_window(patch, wc, ww), cmap="gray", vmin=0, vmax=255,
                  extent=[lo, hi, hi, lo], interpolation="bilinear")
        ax.set_adjustable("datalim")
        color = CROSSHAIR_COLOR["orthogonal"]
        ax.axhline(0, color=color, lw=0.7, alpha=0.85)
        ax.axvline(0, color=color, lw=0.7, alpha=0.85)

        d = metrics.get("diameter_mm", 0.0)
        if d > 0:
            ax.add_patch(Circle((0, 0), d / 2.0, fill=False, ec="#6ee7a0",
                                lw=1.0, alpha=0.9))
        self._label(ax, "Orthogonal to centerline  (%.1f mm along)"
                    % info["arc_mm"])

        self.station_label.setText(
            "%.1f / %.1f mm   (%d/%d)"
            % (info["arc_mm"], ctx.resampler.total_length_mm,
               info["index"], len(ctx.resampler) - 1))

        parts = [
            "Lumen ⌀ <b>%.1f mm</b> (%.0f mm²)"
            % (d, metrics.get("area_mm2", 0.0)),
            "obliquity %.0f°" % info["obliquity_deg"],
        ]
        if info.get("curvature_radius_mm"):
            parts.append("curvature R %.0f mm" % info["curvature_radius_mm"])
        if metrics.get("n_components", 1) > 1:
            # Not a defect: a flat plane re-crosses a bent vessel. Say so,
            # because the extra lumen is visible in the image and would
            # otherwise look like a measurement error.
            parts.append("<span style='color:#ffcc66;'>plane crosses the "
                         "vessel %d× — measuring the one at the "
                         "crosshair</span>" % metrics["n_components"])
        self.readout.setText(" · ".join(parts))

    # ---------------------------------------------------------- zoom and pan
    def _restore_limits(self):
        for name, (xlim, ylim) in self._view_limits.items():
            ax = self.axes.get(name)
            if ax is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

    def _remember(self, name, ax):
        self._view_limits[name] = (ax.get_xlim(), ax.get_ylim())

    def _axis_name(self, ax):
        for name, candidate in self.axes.items():
            if candidate is ax:
                return name
        return None

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        name = self._axis_name(event.inaxes)
        if name is None:
            return
        factor = 1.0 / ZOOM_STEP if event.button == "up" else ZOOM_STEP
        self._zoom_about(event.inaxes, event.xdata, event.ydata, factor)
        self._remember(name, event.inaxes)
        self.canvas.draw_idle()

    @staticmethod
    def _zoom_about(ax, cx, cy, factor):
        """Scale both limits about (cx, cy) so the point under the cursor stays put."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if cx is None or cy is None:
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        ax.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
        ax.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)

    def _on_press(self, event):
        if event.inaxes is None:
            return
        name = self._axis_name(event.inaxes)
        if name is None:
            return
        if event.button == 1:
            self._pick(name, event)
        elif event.button in (2, 3):
            self._nav = {
                "mode": "pan" if event.button == 2 else "zoom",
                "name": name, "ax": event.inaxes,
                "x": event.x, "y": event.y,
                "xdata": event.xdata, "ydata": event.ydata,
                "xlim": event.inaxes.get_xlim(),
                "ylim": event.inaxes.get_ylim(),
            }

    def _on_release(self, _event):
        if self._nav is not None:
            self._remember(self._nav["name"], self._nav["ax"])
            self._nav = None

    def _on_motion(self, event):
        if self._nav is not None:
            self._drag(event)
            return
        self._report_cursor(event)

    def _drag(self, event):
        nav = self._nav
        ax = nav["ax"]
        if event.x is None or event.y is None:
            return
        if nav["mode"] == "pan":
            # Convert the pixel delta into data units through the current
            # transform, so panning tracks the cursor at any zoom level.
            inv = ax.transData.inverted()
            x0, y0 = inv.transform((nav["x"], nav["y"]))
            x1, y1 = inv.transform((event.x, event.y))
            dx, dy = x1 - x0, y1 - y0
            ax.set_xlim(nav["xlim"][0] - dx, nav["xlim"][1] - dx)
            ax.set_ylim(nav["ylim"][0] - dy, nav["ylim"][1] - dy)
        else:
            # Right-drag: vertical travel zooms, anchored where the drag began.
            dy = event.y - nav["y"]
            factor = float(np.clip(ZOOM_STEP ** (-dy / 60.0), 0.02, 50.0))
            ax.set_xlim(nav["xlim"])
            ax.set_ylim(nav["ylim"])
            self._zoom_about(ax, nav["xdata"], nav["ydata"], factor)
        self.canvas.draw_idle()

    # ------------------------------------------------------- cursor tracking
    def _pick(self, name, event):
        """Left-click sets the slice intersection the crosshairs follow."""
        if name == "orthogonal" or self.ctx is None or not self.ctx.has_volume:
            return
        if event.xdata is None or event.ydata is None:
            return
        cursor = self.cursor()
        z, y, x = mpr_mod.cursor_from_pick(
            name, cursor, int(round(event.ydata)), int(round(event.xdata)))
        nz, ny, nx = self.ctx.volume.shape
        z = int(np.clip(z, 0, nz - 1))
        y = int(np.clip(y, 0, ny - 1))
        x = int(np.clip(x, 0, nx - 1))
        self._cursor_override = (z, y, x)
        self.cursor_picked.emit(z, y, x)
        self.refresh()

    def _report_cursor(self, event):
        """Live voxel/world/HU readout under the pointer."""
        if event.inaxes is None or self.ctx is None or not self.ctx.has_volume:
            self._set_coords("—")
            return
        name = self._axis_name(event.inaxes)
        if name is None or event.xdata is None or event.ydata is None:
            self._set_coords("—")
            return

        if name == "orthogonal":
            self._set_coords(self._orthogonal_readout(event))
            return

        cursor = self.cursor()
        z, y, x = mpr_mod.cursor_from_pick(
            name, cursor, int(round(event.ydata)), int(round(event.xdata)))
        nz, ny, nx = self.ctx.volume.shape
        if not (0 <= z < nz and 0 <= y < ny and 0 <= x < nx):
            self._set_coords("—  (outside the volume)")
            return
        hu = int(self.ctx.volume[z, y, x])
        wx, wy, wz = self._world(x, y, z)
        self._set_coords(
            "%-10s voxel [%3d, %3d, %3d]   X %7.1f  Y %7.1f  Z %7.1f mm   "
            "HU %5d" % (name, x, y, z, wx, wy, wz, hu))

    def _orthogonal_readout(self, event):
        """The oblique view is in plane millimetres, not voxels."""
        ctx = self.ctx
        u_mm, v_mm = float(event.xdata), float(event.ydata)
        try:
            i = int(np.clip(self._station, 0, len(ctx.resampler) - 1))
            origin, u, v = ctx.resampler.frame(i)
            world = (np.asarray(origin, float)
                     + u_mm * np.asarray(u, float)
                     + v_mm * np.asarray(v, float))
            from stent_endpoint_sop import world_to_idx
            ix, iy, iz = world_to_idx(world, ctx.geom)
            nz, ny, nx = ctx.volume.shape
            zi, yi, xi = int(round(iz)), int(round(iy)), int(round(ix))
            if 0 <= zi < nz and 0 <= yi < ny and 0 <= xi < nx:
                hu = "%5d" % int(ctx.volume[zi, yi, xi])
            else:
                hu = "   --"
            return ("orthogonal plane [%+6.1f, %+6.1f] mm   X %7.1f  Y %7.1f  "
                    "Z %7.1f mm   HU %s"
                    % (u_mm, v_mm, world[0], world[1], world[2], hu))
        except Exception:                                      # noqa: BLE001
            return "orthogonal plane [%+6.1f, %+6.1f] mm" % (u_mm, v_mm)

    def _world(self, x, y, z):
        try:
            from stent_endpoint_sop import idx_to_world
            w = idx_to_world((x, y, z), self.ctx.geom)
            return float(w[0]), float(w[1]), float(w[2])
        except Exception:                                      # noqa: BLE001
            dz, dy, dx = self.ctx.spacing_zyx
            return x * dx, y * dy, z * dz

    def _set_coords(self, text):
        self.coords.setText(text)
        self.cursor_info.emit(text)


# --------------------------------------------------------------------------- #
def _window(hu, wc, ww):
    lo, hi = wc - ww / 2.0, wc + ww / 2.0
    if hi <= lo:
        hi = lo + 1.0
    out = (np.asarray(hu, np.float32) - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return (out * 255.0).astype(np.uint8)


def _mask_rgba(mask, rgb, alpha):
    """RGBA overlay that is fully transparent outside the mask."""
    m = np.asarray(mask, bool)
    out = np.zeros(m.shape + (4,), np.float32)
    out[m, 0], out[m, 1], out[m, 2] = rgb
    out[m, 3] = alpha
    return out

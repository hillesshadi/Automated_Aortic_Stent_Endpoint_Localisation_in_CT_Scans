#!/usr/bin/env python3
"""
Embedded 3-D viewport: aorta surface, stent surface, centerline tube, and a
marker tracking the current cross-section station.

Built directly on VTK rather than pyvistaqt. Two reasons: ct_viewer/volume3d.py
already proves this exact embedding works in this codebase, and pyvistaqt is an
extra dependency that is not installed here — VTK's own QVTKRenderWindowInteractor
ships with VTK itself. PyVista is still used where it is genuinely convenient
(nothing in this file needs it).

Surfaces come from marching cubes on the segmentation masks, decimated before
display: a full-resolution aortic surface from a 512-slice CTA is millions of
triangles, which is slow to orbit and buys no visible detail at these
framings.
"""

import numpy as np

from ..qt_compat import QtCore, QtWidgets, Signal, QT_API  # noqa: F401 (pins VTK binding)

# Surface construction lives in the pipeline package, which imports no Qt,
# so the headless batch runner builds exactly the same meshes this viewer
# displays. A screenshot is only evidence about an exported file if the two
# came from the same code.
from pipeline.surfaces import (                     # noqa: E402
    mask_to_polydata as _mask_to_polydata,
    arrays_to_polydata as _arrays_to_polydata,
    finish_surface as _finish_surface,
    decimate as _decimate,
    TARGET_TRIANGLES,
)

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_ERROR = None
except Exception as exc:                                       # noqa: BLE001
    vtk = None
    VTK_ERROR = "%s: %s" % (type(exc).__name__, exc)

COLORS = {
    "aorta": (0.86, 0.26, 0.28),
    "stent": (0.85, 0.87, 0.90),
    "centerline": (1.00, 0.85, 0.10),
    "marker": (0.10, 0.90, 1.00),
}


class Viewer3D(QtWidgets.QWidget):
    """VTK render view with named actors that can be replaced independently."""

    cursor_info = Signal(str)      # live world-coordinate / HU readout

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.actors = {}
        self._initialized = False
        self._probe = None          # (volume, geom) for the hover readout
        self._picker = None
        self._last_pick_ms = 0

        if vtk is None:
            msg = QtWidgets.QLabel(
                "3-D viewport unavailable.\n\n%s\n\npip install vtk" % VTK_ERROR)
            msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            msg.setWordWrap(True)
            layout.addWidget(msg)
            self.vtk_widget = None
            self.renderer = None
            return

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.07, 0.08, 0.11)
        self.renderer.SetBackground2(0.15, 0.17, 0.23)
        self.renderer.GradientBackgroundOn()
        window = self.vtk_widget.GetRenderWindow()
        window.AddRenderer(self.renderer)

        # Order-independent transparency. The aorta surface is drawn
        # translucent so the stent and centerline inside it stay visible;
        # without depth peeling a translucent surface does not write depth,
        # so whichever actor happens to be drawn last wins and the stent can
        # appear in front of the vessel wall enclosing it.
        window.SetAlphaBitPlanes(1)
        window.SetMultiSamples(0)          # MSAA and peeling are exclusive
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(8)
        self.renderer.SetOcclusionRatio(0.05)

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # Hover readout. A cell picker (not a prop picker) is needed because
        # only it reports the point *on the surface* under the cursor, which is
        # what the HU probe has to sample.
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.005)
        self.interactor.AddObserver("MouseMoveEvent", self._on_mouse_move)

        axes = vtk.vtkAxesActor()
        self.orientation = vtk.vtkOrientationMarkerWidget()
        self.orientation.SetOrientationMarker(axes)
        self.orientation.SetInteractor(self.interactor)
        self.orientation.SetViewport(0.0, 0.0, 0.18, 0.24)

    # ------------------------------------------------------------------ life
    def showEvent(self, event):
        """Initialize the interactor on first show.

        Calling Initialize() before the widget has a native window handle
        gives a render window with no OpenGL context on some platforms, so it
        is deferred to here rather than done in __init__.
        """
        super().showEvent(event)
        if self.vtk_widget is not None and not self._initialized:
            self.interactor.Initialize()
            try:
                self.orientation.EnabledOn()
                self.orientation.InteractiveOff()
            except Exception:                                  # noqa: BLE001
                pass
            self._initialized = True
            self.reset_camera()

    def close_viewer(self):
        """Release every GL resource, in the order that does not crash.

        Actors first, then the renderer, then the window. Each mapper holds
        GL objects that must be released while a context still exists; leaving
        them for the garbage collector is what produces the post-execution
        SIGSEGV, which happens after the user has closed the window and
        believes the session ended cleanly.

        Every step is guarded independently -- a failure part-way through must
        not skip the remaining releases, because the ones after it are the
        ones that own the context.
        """
        if self.vtk_widget is None:
            return

        try:
            if self.orientation is not None:
                self.orientation.EnabledOff()
                self.orientation.SetInteractor(None)
        except Exception:                                      # noqa: BLE001
            pass

        for name in list(self.actors):
            try:
                self._remove(name)
            except Exception:                                  # noqa: BLE001
                pass
        self.actors = {}

        window = None
        try:
            window = self.vtk_widget.GetRenderWindow()
        except Exception:                                      # noqa: BLE001
            pass

        try:
            if self.renderer is not None:
                self.renderer.RemoveAllViewProps()
                if window is not None:
                    window.RemoveRenderer(self.renderer)
        except Exception:                                      # noqa: BLE001
            pass

        for step in ("Finalize", "TerminateApp"):
            try:
                if step == "Finalize" and window is not None:
                    window.Finalize()
                elif step == "TerminateApp" and self.interactor is not None:
                    self.interactor.TerminateApp()
            except Exception:                                  # noqa: BLE001
                pass

        self.renderer = None
        self._initialized = False

    # ---------------------------------------------------------- hover probe
    def set_probe_volume(self, volume, geom):
        """Volume to sample HU from when the pointer hovers over a surface."""
        self._probe = (volume, geom) if volume is not None else None

    def _on_mouse_move(self, _caller, _event):
        """Report world coordinates and HU under the cursor.

        Throttled: MouseMoveEvent fires far faster than a cell pick over a
        150k-triangle surface can complete, and an unthrottled picker makes
        orbiting feel sticky.
        """
        if not self._initialized or self.renderer is None:
            return
        now = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        if now - self._last_pick_ms < 60:
            return
        self._last_pick_ms = now

        try:
            x, y = self.interactor.GetEventPosition()
            if not self._picker.Pick(x, y, 0, self.renderer):
                self.cursor_info.emit("3-D  (no surface under cursor)")
                return
            px, py, pz = self._picker.GetPickPosition()
        except Exception:                                      # noqa: BLE001
            return

        text = "3-D  X %7.1f  Y %7.1f  Z %7.1f mm" % (px, py, pz)
        if self._probe is not None:
            volume, geom = self._probe
            try:
                from stent_endpoint_sop import world_to_idx
                ix, iy, iz = world_to_idx(np.asarray([px, py, pz], float), geom)
                nz, ny, nx = volume.shape
                zi, yi, xi = int(round(iz)), int(round(iy)), int(round(ix))
                if 0 <= zi < nz and 0 <= yi < ny and 0 <= xi < nx:
                    text += ("   voxel [%3d, %3d, %3d]   HU %5d"
                             % (xi, yi, zi, int(volume[zi, yi, xi])))
            except Exception:                                  # noqa: BLE001
                pass
        self.cursor_info.emit(text)

    # --------------------------------------------------------------- actors
    def _remove(self, name):
        actor = self.actors.pop(name, None)
        if actor is not None and self.renderer is not None:
            self.renderer.RemoveActor(actor)

    def clear(self):
        for name in list(self.actors):
            self._remove(name)
        self.render()

    def set_visible(self, name, visible):
        actor = self.actors.get(name)
        if actor is not None:
            actor.SetVisibility(bool(visible))
            self.render()

    def render(self):
        if self.vtk_widget is not None and self._initialized:
            self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self):
        """Frame all actors from an anterior-oblique viewpoint.

        ResetCamera alone looks straight down +Z and leaves a wide margin,
        which wastes most of the panel on a tall vessel.
        """
        if self.renderer is None:
            return
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(0.0, -1.0, 0.0)
        cam.SetFocalPoint(0.0, 0.0, 0.0)
        cam.SetViewUp(0.0, 0.0, 1.0)       # patient superior is up
        cam.Azimuth(30.0)
        cam.Elevation(15.0)
        cam.OrthogonalizeViewUp()
        self.renderer.ResetCamera()
        cam.Zoom(1.35)
        self.render()

    # ------------------------------------------------------------- geometry
    def set_mask_surface(self, name, mask, geom, opacity=1.0, color=None):
        """Marching-cubes isosurface of a boolean mask, in world coordinates."""
        if vtk is None or mask is None or not mask.any():
            self._remove(name)
            self.render()
            return

        poly = _mask_to_polydata(mask, geom)
        if poly is None or poly.GetNumberOfPoints() == 0:
            self._remove(name)
            return

        poly = _finish_surface(_decimate(poly, TARGET_TRIANGLES))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        rgb = color or COLORS.get(name, (0.8, 0.8, 0.8))
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(float(opacity))
        actor.GetProperty().SetSpecular(0.25)
        actor.GetProperty().SetSpecularPower(30)

        self._remove(name)
        self.renderer.AddActor(actor)
        self.actors[name] = actor
        self.render()

    def set_centerline(self, points, radius_mm=1.2, color=None):
        """Centerline as a tube so it stays visible against the surfaces."""
        if vtk is None or points is None or len(points) < 2:
            self._remove("centerline")
            self.render()
            return

        pts = np.asarray(points, float).reshape(-1, 3)
        vpoints = vtk.vtkPoints()
        vpoints.SetNumberOfPoints(len(pts))
        for i, p in enumerate(pts):
            vpoints.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))

        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(pts))
        for i in range(len(pts)):
            line.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)

        poly = vtk.vtkPolyData()
        poly.SetPoints(vpoints)
        poly.SetLines(cells)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(poly)
        tube.SetRadius(float(radius_mm))
        tube.SetNumberOfSides(16)
        tube.CappingOn()
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*(color or COLORS["centerline"]))
        # High ambient so the tube keeps its colour where it is shadowed by
        # the surrounding surface rather than going black.
        actor.GetProperty().SetAmbient(0.5)
        actor.GetProperty().SetDiffuse(0.6)

        self._remove("centerline")
        self.renderer.AddActor(actor)
        self.actors["centerline"] = actor
        self.render()

    def set_surface_mesh(self, name, vertices, faces, opacity=1.0, color=None):
        """A loaded STL/VTK surface, already in world coordinates."""
        if vtk is None or vertices is None or faces is None or len(faces) == 0:
            self._remove(name)
            self.render()
            return

        poly = _arrays_to_polydata(np.asarray(vertices, float),
                                   np.asarray(faces, np.int64))
        poly = _finish_surface(_decimate(poly, TARGET_TRIANGLES))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*(color or COLORS.get(name, (0.7, 0.7, 0.8))))
        actor.GetProperty().SetOpacity(float(opacity))

        self._remove(name)
        self.renderer.AddActor(actor)
        self.actors[name] = actor
        self.render()

    def set_marker(self, point, radius_mm=3.0):
        """Sphere marking the current cross-section station."""
        if vtk is None or point is None:
            self._remove("marker")
            self.render()
            return
        src = vtk.vtkSphereSource()
        src.SetCenter(float(point[0]), float(point[1]), float(point[2]))
        src.SetRadius(float(radius_mm))
        src.SetThetaResolution(20)
        src.SetPhiResolution(20)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*COLORS["marker"])
        actor.GetProperty().SetAmbient(0.6)
        self._remove("marker")
        self.renderer.AddActor(actor)
        self.actors["marker"] = actor
        self.render()

    def screenshot(self, path):
        """Save the current view (used by tests and by Export)."""
        if self.vtk_widget is None:
            return None
        rw = self.vtk_widget.GetRenderWindow()
        rw.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.ReadFrontBufferOff()
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        return path


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

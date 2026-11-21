#!/usr/bin/env python3
"""
3D volume-rendering viewport (bottom-right quadrant of the MPR grid).

Built directly on VTK (no PyVista/pyvistaqt dependency): a vtkSmartVolumeMapper
ray-casts the CT volume, and segmentation masks (aorta / stent) are shown as
smoothed isosurfaces extracted with vtkDiscreteMarchingCubes. A small sphere
tracks the shared 3D cursor so the user can see where the two orthogonal
crosshairs currently intersect in space.

Memory note: VTK needs its own contiguous buffer, i.e. a full deep copy of
whatever numpy array it's given. For a large series (500MB+ int16 volume)
that copy alone can exhaust memory on a constrained machine, so `set_volume`
always renders a *downsampled* copy (see mpr.downsample_strides) - the 2D
MPR views stay full-resolution, only this 3D preview is decimated.
"""

import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

import mpr

MASK_COLORS = {
    "aorta": (220, 50, 50),
    "stent": (0, 220, 220),
}


def _numpy_volume_to_vtk_image(volume, spacing_xyz, dtype_vtk):
    nz, ny, nx = volume.shape
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*spacing_xyz)
    img.SetOrigin(0.0, 0.0, 0.0)
    flat = np.ascontiguousarray(volume)
    vtk_arr = numpy_to_vtk(flat.ravel(), deep=True, array_type=dtype_vtk)
    img.GetPointData().SetScalars(vtk_arr)
    return img


class Volume3DWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.06, 0.06, 0.08)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.volume_actor = None
        self.outline_actor = None
        self.cursor_actor = None
        self.cursor_source = None
        self.mask_actors = {}          # name -> vtkActor
        self.mask_visibility = {}      # name -> bool (checkbox state)

        self._strides = (1, 1, 1)          # downsample strides applied for 3D
        self._spacing_zyx = (1.0, 1.0, 1.0)  # full-resolution spacing
        self._initialized = False

        # Right-click -> "Save image as..." (PNG/JPG). vtkInteractorStyle-
        # TrackballCamera also uses the right button for dolly-zoom, so a
        # plain contextMenuEvent/CustomContextMenu policy on the VTK child
        # widget never fires - it's intercepted first. Watching press/release
        # here (without consuming the event) lets both behaviours coexist:
        # a real right-drag still zooms, a right click-without-drag pops the
        # save menu.
        self._rmb_press_pos = None
        self.vtk_widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.vtk_widget:
            if (event.type() == QtCore.QEvent.MouseButtonPress
                    and event.button() == Qt.RightButton):
                self._rmb_press_pos = event.pos()
            elif (event.type() == QtCore.QEvent.MouseButtonRelease
                    and event.button() == Qt.RightButton):
                press = self._rmb_press_pos
                self._rmb_press_pos = None
                if press is not None:
                    moved = (event.pos() - press).manhattanLength()
                    if moved <= QtWidgets.QApplication.startDragDistance():
                        self._show_save_menu(event.globalPos())
        return super().eventFilter(obj, event)

    def _show_save_menu(self, global_pos):
        menu = QtWidgets.QMenu(self)
        act_png = menu.addAction("Save 3D View as PNG...")
        act_jpg = menu.addAction("Save 3D View as JPG...")
        chosen = menu.exec_(global_pos)
        if chosen is act_png:
            self.save_screenshot("png")
        elif chosen is act_jpg:
            self.save_screenshot("jpg")

    def save_screenshot(self, ext):
        """Capture the current 3D render as a PNG or JPG file."""
        ext = ext.lower()
        filters = {"png": "PNG Image (*.png)", "jpg": "JPEG Image (*.jpg *.jpeg)"}
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 3D View", f"3d_view.{ext}",
            filters.get(ext, "Image (*)"))
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += f".{ext}"

        self.render()  # make sure the buffer being captured is up to date
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.vtk_widget.GetRenderWindow())
        if ext == "jpg" or ext == "jpeg":
            w2i.SetInputBufferTypeToRGB()   # JPEG has no alpha channel
        else:
            w2i.SetInputBufferTypeToRGBA()
        w2i.ReadFrontBufferOff()
        w2i.Update()

        writer = vtk.vtkJPEGWriter() if ext in ("jpg", "jpeg") else vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(w2i.GetOutputPort())
        try:
            writer.Write()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "Saved", f"3D view saved to:\n{path}")

    def showEvent(self, e):
        super().showEvent(e)
        if not self._initialized:
            self.interactor.Initialize()
            self._initialized = True

    # ------------------------------------------------------------- volume -- #
    def set_volume(self, volume, spacing_zyx, wc, ww):
        self._spacing_zyx = spacing_zyx
        self._strides = mpr.downsample_strides(volume.shape, spacing_zyx)
        sz, sy, sx = self._strides
        small = volume[::sz, ::sy, ::sx]
        ds_spacing_zyx = mpr.downsampled_spacing(spacing_zyx, self._strides)
        dz, dy, dx = ds_spacing_zyx

        img = _numpy_volume_to_vtk_image(small, (dx, dy, dz), vtk.VTK_SHORT)

        if self.volume_actor is not None:
            self.renderer.RemoveVolume(self.volume_actor)

        vtk_mapper = vtk.vtkSmartVolumeMapper()
        vtk_mapper.SetInputData(img)

        prop = vtk.vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()
        self._apply_transfer_functions(prop, wc, ww)

        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(vtk_mapper)
        self.volume_actor.SetProperty(prop)
        self.renderer.AddVolume(self.volume_actor)

        if self.outline_actor is not None:
            self.renderer.RemoveActor(self.outline_actor)
        outline = vtk.vtkOutlineFilter()
        outline.SetInputData(img)
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(outline_mapper)
        self.outline_actor.GetProperty().SetColor(0.5, 0.5, 0.55)
        self.renderer.AddActor(self.outline_actor)

        # Existing masks were built at the old strides/geometry - drop them,
        # the caller re-adds current masks right after loading a new volume.
        for actor in self.mask_actors.values():
            self.renderer.RemoveActor(actor)
        self.mask_actors.clear()

        self.renderer.ResetCamera()
        self.render()

    def _apply_transfer_functions(self, prop, wc, ww):
        lo, hi = wc - ww / 2.0, wc + ww / 2.0
        if hi <= lo:
            hi = lo + 1.0
        color_tf = vtk.vtkColorTransferFunction()
        color_tf.AddRGBPoint(lo, 0.0, 0.0, 0.0)
        color_tf.AddRGBPoint(hi, 1.0, 1.0, 1.0)
        opacity_tf = vtk.vtkPiecewiseFunction()
        opacity_tf.AddPoint(lo, 0.0)
        opacity_tf.AddPoint(lo + (hi - lo) * 0.3, 0.06)
        opacity_tf.AddPoint(hi, 0.85)
        prop.SetColor(color_tf)
        prop.SetScalarOpacity(opacity_tf)

    def set_window_level(self, wc, ww):
        if self.volume_actor is None:
            return
        self._apply_transfer_functions(self.volume_actor.GetProperty(), wc, ww)
        self.render()

    def set_volume_visible(self, visible):
        if self.volume_actor is not None:
            self.volume_actor.SetVisibility(bool(visible))
            self.render()

    # --------------------------------------------------------------- masks -- #
    def set_mask(self, name, mask, visible=True):
        if name in self.mask_actors:
            self.renderer.RemoveActor(self.mask_actors[name])
            del self.mask_actors[name]
        self.mask_visibility[name] = visible

        if mask is None or self.volume_actor is None:
            self.render()
            return
        sz, sy, sx = self._strides
        small = mask[::sz, ::sy, ::sx]
        if not small.any():
            self.render()
            return
        dz, dy, dx = mpr.downsampled_spacing(self._spacing_zyx, self._strides)

        img = _numpy_volume_to_vtk_image(
            np.ascontiguousarray(small.astype(np.uint8)), (dx, dy, dz),
            vtk.VTK_UNSIGNED_CHAR)

        mc = vtk.vtkDiscreteMarchingCubes()
        mc.SetInputData(img)
        mc.SetValue(0, 1)
        mc.Update()
        if mc.GetOutput().GetNumberOfPoints() == 0:
            self.render()
            return

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(mc.GetOutputPort())
        smoother.SetNumberOfIterations(15)
        smoother.BoundarySmoothingOff()
        smoother.Update()

        poly_mapper = vtk.vtkPolyDataMapper()
        poly_mapper.SetInputConnection(smoother.GetOutputPort())
        poly_mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(poly_mapper)
        r, g, b = MASK_COLORS.get(name, (200, 200, 200))
        actor.GetProperty().SetColor(r / 255.0, g / 255.0, b / 255.0)
        actor.GetProperty().SetOpacity(0.85)
        actor.SetVisibility(bool(visible))
        self.renderer.AddActor(actor)
        self.mask_actors[name] = actor
        self.render()

    def set_mask_visible(self, name, visible):
        self.mask_visibility[name] = visible
        actor = self.mask_actors.get(name)
        if actor is not None:
            actor.SetVisibility(bool(visible))
            self.render()

    # -------------------------------------------------------------- cursor -- #
    def set_cursor(self, cursor_zyx):
        z, y, x = cursor_zyx
        dz, dy, dx = self._spacing_zyx
        x_mm, y_mm, z_mm = x * dx, y * dy, z * dz

        if self.cursor_source is None:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(max(self._spacing_zyx) * 3.0)
            sphere.SetPhiResolution(14)
            sphere.SetThetaResolution(14)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            self.cursor_actor = vtk.vtkActor()
            self.cursor_actor.SetMapper(mapper)
            self.cursor_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
            self.cursor_actor.GetProperty().SetOpacity(0.9)
            self.cursor_source = sphere
            self.renderer.AddActor(self.cursor_actor)

        self.cursor_source.SetCenter(x_mm, y_mm, z_mm)
        self.render()

    # ------------------------------------------------------------- misc -- #
    def clear(self):
        self.renderer.RemoveAllViewProps()
        self.volume_actor = None
        self.outline_actor = None
        self.cursor_actor = None
        self.cursor_source = None
        self.mask_actors.clear()
        self.render()

    def render(self):
        self.vtk_widget.GetRenderWindow().Render()

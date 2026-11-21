#!/usr/bin/env python3
"""
Surface generation and mesh I/O -- pure geometry, no GUI and no OpenGL.

This module was extracted from gui/widgets/viewer_3d.py so that the headless
pipeline and the desktop viewer build surfaces the same way. That matters more
than avoiding duplication: a mesh exported on the cluster and a mesh shown on
screen must be the same object, or a screenshot stops being evidence about the
file.

Nothing here needs a render window. Marching cubes, decimation, normals and the
STL/PLY/VTK writers are all CPU-side filters, which is why the headless
pipeline can still produce every mesh artifact on a node where no OpenGL
context can be created at all.
"""

import os

import numpy as np

from . import paths  # noqa: F401  (sys.path bootstrap)

TARGET_TRIANGLES = 150000

# Writer per extension. vtkPolyDataWriter is the legacy .vtk format;
# vtkXMLPolyDataWriter is the modern .vtp.
_WRITERS = {
    ".stl": "vtkSTLWriter",
    ".ply": "vtkPLYWriter",
    ".vtk": "vtkPolyDataWriter",
    ".vtp": "vtkXMLPolyDataWriter",
    ".obj": "vtkOBJWriter",
}


class SurfaceError(RuntimeError):
    pass


def _require_vtk():
    try:
        import vtk
        return vtk
    except ImportError as exc:                                 # noqa: BLE001
        raise SurfaceError(
            "VTK is required to build surfaces.\n\n    pip install vtk\n\n(%s)"
            % exc) from exc


# --------------------------------------------------------------------------- #
#  Mask -> surface
# --------------------------------------------------------------------------- #
def mask_to_polydata(mask, geom):
    """Marching cubes on a boolean mask, output in world/LPS millimetres.

    The mask is handed to VTK on a unit grid and the origin/spacing/direction
    are applied afterwards as an explicit transform. Baking spacing into
    vtkImageData and the direction into the transform separately is what keeps
    this correct for an oblique direction matrix, where simply setting
    SetSpacing/SetOrigin would silently drop the rotation.
    """
    vtk = _require_vtk()
    from vtk.util.numpy_support import numpy_to_vtk

    m = np.ascontiguousarray(np.asarray(mask).astype(np.uint8))
    nz, ny, nx = m.shape

    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    arr = numpy_to_vtk(m.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    img.GetPointData().SetScalars(arr)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(img)
    mc.SetValue(0, 0.5)
    # Normals here would be computed on the unit grid and invalidated by the
    # transform, smoothing and decimation below; finish_surface() recomputes
    # them once the geometry is final.
    mc.ComputeNormalsOff()
    mc.Update()

    poly = mc.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        return None

    origin, spacing, direction = geom
    D = np.asarray(direction, float).reshape(3, 3)
    S = np.asarray(spacing, float)

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    for r in range(3):
        for c in range(3):
            matrix.SetElement(r, c, float(D[r, c] * S[c]))
        matrix.SetElement(r, 3, float(origin[r]))

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(poly)
    tf.SetTransform(transform)
    tf.Update()

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(tf.GetOutputPort())
    smooth.SetNumberOfIterations(30)
    smooth.SetPassBand(0.05)          # low band = removes slice stepping
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    return smooth.GetOutput()


def arrays_to_polydata(vertices, faces):
    """(N,3) vertices + (M,3) triangle indices -> vtkPolyData."""
    vtk = _require_vtk()
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(vertices))
    for i, v in enumerate(vertices):
        points.SetPoint(i, float(v[0]), float(v[1]), float(v[2]))
    cells = vtk.vtkCellArray()
    for f in faces:
        cells.InsertNextCell(3)
        cells.InsertCellPoint(int(f[0]))
        cells.InsertCellPoint(int(f[1]))
        cells.InsertCellPoint(int(f[2]))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly


def finish_surface(poly):
    """Recompute vertex normals once the geometry is final.

    Marching cubes, the world transform, smoothing and decimation each
    invalidate the previous stage's normals, so shading is only correct if
    they are generated last.
    """
    vtk = _require_vtk()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SetFeatureAngle(60.0)
    normals.SplittingOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


def decimate(poly, target_triangles=TARGET_TRIANGLES):
    """Reduce to roughly `target_triangles`, or pass through if already under."""
    vtk = _require_vtk()
    n = poly.GetNumberOfPolys()
    if n <= target_triangles:
        return poly
    deci = vtk.vtkQuadricDecimation()
    deci.SetInputData(poly)
    deci.SetTargetReduction(1.0 - float(target_triangles) / float(n))
    deci.Update()
    return deci.GetOutput()


def build_surface(mask, geom, target_triangles=TARGET_TRIANGLES):
    """Mask -> display/export-ready surface, or None if the mask is empty."""
    if mask is None or not np.asarray(mask).any():
        return None
    poly = mask_to_polydata(mask, geom)
    if poly is None or poly.GetNumberOfPoints() == 0:
        return None
    return finish_surface(decimate(poly, target_triangles))


def centerline_polydata(points):
    """(N,3) world-mm points -> a single vtkPolyData polyline."""
    vtk = _require_vtk()
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) < 2:
        return None
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
    return poly


def tube_polydata(points, radius_mm=1.2, sides=16):
    """Centerline as a tube, so it survives export to a surface-only format."""
    vtk = _require_vtk()
    line = centerline_polydata(points)
    if line is None:
        return None
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(line)
    tube.SetRadius(float(radius_mm))
    tube.SetNumberOfSides(int(sides))
    tube.CappingOn()
    tube.Update()
    # vtkTubeFilter emits triangle *strips*, not polygons. The STL and PLY
    # writers only serialise polygons, so an untriangulated tube writes an
    # empty file. Triangulating here keeps every consumer's assumption true.
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(tube.GetOutput())
    tri.Update()
    return tri.GetOutput()


# --------------------------------------------------------------------------- #
#  Mesh writing
# --------------------------------------------------------------------------- #
def write_mesh(poly, path, binary=True):
    """Write a vtkPolyData to .stl / .ply / .vtk / .vtp / .obj.

    STL and PLY carry triangles only, so a polyline (a centerline) written to
    either would produce an empty file. Callers exporting a centerline should
    pass it through tube_polydata() first; this raises rather than writing that
    empty file silently.
    """
    vtk = _require_vtk()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise SurfaceError("Nothing to write to %s -- the surface is empty."
                           % os.path.basename(path))

    ext = os.path.splitext(path)[1].lower()
    writer_name = _WRITERS.get(ext)
    if writer_name is None:
        raise SurfaceError(
            "Unsupported mesh format '%s'.\nExpected one of: %s."
            % (ext, ", ".join(sorted(_WRITERS))))
    if ext in (".stl", ".ply") and poly.GetNumberOfPolys() == 0             and poly.GetNumberOfStrips() == 0:
        raise SurfaceError(
            "%s stores surface geometry, but this has none (%d lines, 0 "
            "triangles). Export it as .vtk/.vtp, or convert it with "
            "tube_polydata()." % (ext, poly.GetNumberOfLines()))

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    writer = getattr(vtk, writer_name)()
    writer.SetFileName(path)
    writer.SetInputData(poly)
    # The XML and OBJ writers have no SetFileTypeToBinary; only ask for it
    # where it exists.
    if binary and hasattr(writer, "SetFileTypeToBinary"):
        writer.SetFileTypeToBinary()
    if not writer.Write():
        raise SurfaceError("VTK failed to write %s." % path)
    return path


def surface_stats(poly):
    """Points, triangles and bounding box, for the run summary."""
    if poly is None:
        return None
    b = poly.GetBounds()
    return {
        "points": int(poly.GetNumberOfPoints()),
        "triangles": int(poly.GetNumberOfPolys()),
        "lines": int(poly.GetNumberOfLines()),
        "bounds_mm": [float(v) for v in b],
        "extent_mm": [float(b[1] - b[0]), float(b[3] - b[2]),
                      float(b[5] - b[4])],
    }

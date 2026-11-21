#!/usr/bin/env python3
"""
Loading: DICOM series, NIfTI/NRRD volumes, surface meshes, and centerlines.

Volumes go through SimpleITK for both DICOM and NIfTI, because it yields the
origin/spacing/direction triple the rest of the pipeline needs for world-space
work; reading a NIfTI with nibabel and a DICOM series with pydicom would mean
two different conventions to reconcile.

Meshes and polylines go through VTK's own readers where VTK is importable,
because they cover the XML formats (.vtp) including the base64 and
zlib-compressed payload variants that the regex-based reader in
src/stent_html_export.py cannot parse. That reader stays as the fallback.
"""

import os

import numpy as np

from . import paths  # noqa: F401  (sys.path bootstrap)


class LoadError(RuntimeError):
    """Anything the user can fix by choosing a different file."""


# --------------------------------------------------------------------------- #
#  Volumes
# --------------------------------------------------------------------------- #
def _require_sitk():
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError as exc:
        raise LoadError(
            "SimpleITK is required to read images.\n\n"
            "    pip install SimpleITK\n\n(%s)" % exc) from exc


def load_dicom_series(directory, progress=None):
    """Read the largest DICOM series in `directory`.

    Largest = most slices, which is the diagnostic acquisition; a folder
    routinely also holds a scout/localizer of a handful of slices that would
    otherwise be picked arbitrarily.
    """
    sitk = _require_sitk()
    if not os.path.isdir(directory):
        raise LoadError("Not a directory:\n%s" % directory)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    if not series_ids:
        raise LoadError(
            "No DICOM series found in:\n%s\n\n"
            "Expected a folder of .dcm slices." % directory)

    best_id, best_files = None, []
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(directory, sid)
        if len(files) > len(best_files):
            best_id, best_files = sid, files
    if progress:
        progress("Reading %d slices (series %s of %d)…"
                 % (len(best_files), best_id[:16], len(series_ids)))

    reader.SetFileNames(best_files)
    image = reader.Execute()
    return _unpack_image(image, os.path.basename(os.path.normpath(directory)))


def load_image_file(path, progress=None):
    """Read a NIfTI / NRRD / MetaImage volume."""
    sitk = _require_sitk()
    if not os.path.isfile(path):
        raise LoadError("File not found:\n%s" % path)
    if progress:
        progress("Reading %s…" % os.path.basename(path))
    try:
        image = sitk.ReadImage(path)
    except Exception as exc:                                   # noqa: BLE001
        raise LoadError("SimpleITK could not read:\n%s\n\n%s"
                        % (path, exc)) from exc
    return _unpack_image(image, case_id_from_path(path))


def _unpack_image(image, case_id):
    sitk = _require_sitk()
    volume = sitk.GetArrayFromImage(image)
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise LoadError("Expected a 3-D volume, got shape %s." % (volume.shape,))

    volume = volume.astype(np.int16, copy=False)
    sx, sy, sz = image.GetSpacing()
    geom = (tuple(image.GetOrigin()), (sx, sy, sz), tuple(image.GetDirection()))
    return {"volume": volume, "spacing_zyx": (sz, sy, sx),
            "geom": geom, "case_id": case_id}


def case_id_from_path(path):
    """Pipeline case ID for a path.

    A DICOM directory is already named after the case; a volume file is not --
    the pipeline writes "<case>_canonical.nii.gz", so the bare basename would
    never match the "<case>_centerline.vtk" beside it.
    """
    if not path:
        return ""
    base = os.path.basename(os.path.normpath(path))
    if os.path.isdir(path):
        return base
    for ext in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".stl", ".vtk",
                ".vtp", ".ply", ".obj", ".csv", ".json"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    for suffix in ("_canonical", "_ct", "_volume", "_aorta", "_stent",
                   "_centerline"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


# --------------------------------------------------------------------------- #
#  Meshes and polylines
# --------------------------------------------------------------------------- #
_VTK_READERS = {
    ".vtp": "vtkXMLPolyDataReader",
    ".vtk": "vtkPolyDataReader",
    ".stl": "vtkSTLReader",
    ".ply": "vtkPLYReader",
    ".obj": "vtkOBJReader",
}


def _read_polydata(path):
    """Read a surface/polyline file into a vtkPolyData.

    Returns None only when VTK cannot be used at all -- it is unimportable, or
    has no reader for this extension. A file VTK *did* read but which came back
    empty returns the empty polydata, so the caller reports that honestly
    instead of falling through to the legacy reader, which cannot parse the XML
    formats and would blame a missing POINTS section.
    """
    try:
        import vtk
    except ImportError:
        return None
    reader_name = _VTK_READERS.get(os.path.splitext(path)[1].lower())
    if reader_name is None or not hasattr(vtk, reader_name):
        return None
    reader = getattr(vtk, reader_name)()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def _polydata_points(poly):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    return np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), float)


def _polydata_cells(cell_array):
    """vtkCellArray -> list of index lists, without assuming a fixed cell size."""
    import vtk
    cells, ids = [], vtk.vtkIdList()
    cell_array.InitTraversal()
    while cell_array.GetNextCell(ids):
        cells.append([ids.GetId(i) for i in range(ids.GetNumberOfIds())])
    return cells


def _polydata_triangles(poly):
    """Triangulated faces as an (M, 3) array, or None if there are no polygons."""
    if poly.GetNumberOfPolys() == 0 and poly.GetNumberOfStrips() == 0:
        return None
    import vtk
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    out = tri.GetOutput()
    faces = [c for c in _polydata_cells(out.GetPolys()) if len(c) == 3]
    if not faces:
        return None
    return np.asarray(faces, np.int64)


def load_surface_mesh(path, progress=None):
    """Read an STL/VTK/VTP/PLY/OBJ surface into (vertices, faces).

    A file holding only polylines (a centerline exported as .vtk) returns
    faces=None so the caller can route it to the centerline handler instead of
    failing -- the two share an extension and are easy to mix up.
    """
    if not os.path.isfile(path):
        raise LoadError("File not found:\n%s" % path)
    ext = os.path.splitext(path)[1].lower()
    if progress:
        progress("Reading mesh %s…" % os.path.basename(path))

    try:
        poly = _read_polydata(path)
        if poly is not None:
            if poly.GetNumberOfPoints() == 0:
                raise LoadError(
                    "%s holds no points.\n\nIt may be empty or written "
                    "in an unexpected format." % os.path.basename(path))
            verts = _polydata_points(poly)
            faces = _polydata_triangles(poly)
            lines = (_polydata_cells(poly.GetLines())
                     if poly.GetNumberOfLines() else [])
            return {"vertices": verts, "faces": faces, "lines": lines,
                    "case_id": case_id_from_path(path)}

        # ---- fallback: VTK not importable -------------------------------- #
        if ext == ".stl":
            from stent_html_export import read_stl
            verts, faces = read_stl(path)
            return {"vertices": np.asarray(verts, float),
                    "faces": np.asarray(faces, np.int64),
                    "case_id": case_id_from_path(path)}
        if ext in (".vtk", ".vtp"):
            from stent_html_export import read_vtk_polydata
            points, lines = read_vtk_polydata(path)
            return {"vertices": np.asarray(points, float),
                    "faces": None, "lines": lines,
                    "case_id": case_id_from_path(path)}
    except LoadError:
        raise
    except Exception as exc:                                   # noqa: BLE001
        raise LoadError("Could not read mesh:\n%s\n\n%s" % (path, exc)) from exc

    raise LoadError(
        "Unsupported mesh format '%s'.\nExpected one of: %s."
        % (ext, ", ".join(sorted(_VTK_READERS))))


# --------------------------------------------------------------------------- #
#  Centerlines
# --------------------------------------------------------------------------- #
#  Column-name aliases seen across the tools that produce these files (VMTK,
#  3D Slicer markups exports, this pipeline's own CSV writer, and hand-made
#  spreadsheets). Matching runs on a normalised name, so "Pos X", "pos_x" and
#  "POS-X" all resolve to the same axis.
_AXIS_ALIASES = {
    "x": ("x", "xmm", "worldx", "posx", "positionx", "coordx", "cx", "px",
          "pointx"),
    "y": ("y", "ymm", "worldy", "posy", "positiony", "coordy", "cy", "py",
          "pointy"),
    "z": ("z", "zmm", "worldz", "posz", "positionz", "coordz", "cz", "pz",
          "pointz"),
}


def _normalise(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _is_number(text):
    try:
        float(str(text).strip())
        return True
    except (TypeError, ValueError):
        return False


def read_centerline_csv(path):
    """Read (N, 3) world-mm points from a CSV, header-driven where possible.

    Handles the delimiters these files actually arrive with (comma, semicolon,
    tab, pipe, or runs of spaces), '#' comment lines, a UTF-8 BOM, and either a
    recognised header or none at all. Anything unparseable raises LoadError
    naming the reason: a silently empty centerline otherwise surfaces much
    later as a confusing failure inside the resampler.
    """
    import csv as _csv
    import re as _re

    with open(path, "r", newline="", encoding="utf-8-sig",
              errors="replace") as handle:
        raw = [ln for ln in handle.read().splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    if not raw:
        raise LoadError("Centerline CSV is empty:\n%s" % path)

    sample = "\n".join(raw[:2])
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
        rows = list(_csv.reader(raw, dialect))
    except Exception:                                          # noqa: BLE001
        # Sniffing fails on space-aligned or single-column files; splitting on
        # any delimiter run covers what it misses.
        rows = [_re.split(r"[,;\t|]+|\s+", ln.strip()) for ln in raw]
    rows = [[c.strip() for c in r if str(c).strip() != ""] for r in rows]
    rows = [r for r in rows if r]
    if not rows:
        raise LoadError("Centerline CSV has no data rows:\n%s" % path)

    header = [_normalise(c) for c in rows[0]]
    has_header = not _is_number(rows[0][0])

    idx = {}
    if has_header:
        for axis, aliases in _AXIS_ALIASES.items():
            for i, name in enumerate(header):
                if name in aliases:
                    idx[axis] = i
                    break

    if len(idx) == 3:
        ix, iy, iz = idx["x"], idx["y"], idx["z"]
        body = rows[1:]
    else:
        # No usable header: take the first three numeric columns in order.
        body = rows[1:] if has_header else rows
        if not any(len(r) >= 3 and _is_number(r[0]) for r in body):
            raise LoadError(
                "Could not find X/Y/Z columns in:\n%s\n\nHeader was: %s\n\n"
                "Expected columns named x,y,z (or pos_x/pos_y/pos_z, "
                "world_x…), or three numeric columns."
                % (path, ", ".join(rows[0]) or "(none)"))
        ix, iy, iz = 0, 1, 2

    pts, skipped = [], 0
    for r in body:
        if len(r) <= max(ix, iy, iz):
            skipped += 1
            continue
        try:
            pts.append([float(r[ix]), float(r[iy]), float(r[iz])])
        except ValueError:
            skipped += 1

    if len(pts) < 2:
        raise LoadError(
            "Centerline CSV yielded %d usable point(s), skipping %d row(s):\n%s"
            % (len(pts), skipped, path))
    return np.asarray(pts, float)


def read_centerline_json(path):
    """Read (N, 3) world-mm points from a JSON centerline.

    Handles the shapes these files actually arrive in:

      * 3D Slicer markups   {"markups":[{"controlPoints":[{"position":[x,y,z]}]}]}
      * a plain point list  {"points": [[x,y,z], ...]}  (also "centerline",
                            "centerline_points", "control_points", "curve")
      * records             [{"x":..,"y":..,"z":..}, ...]
      * a bare array        [[x,y,z], ...]

    A JSON file that is not a centerline at all -- a migration report, say --
    is rejected with the keys it actually contains, because "unsupported
    format" gives no clue which of several JSON files in a case folder was the
    wrong one.
    """
    import json

    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise LoadError("%s is not valid JSON.\n\nLine %d, column %d: %s"
                        % (os.path.basename(path), exc.lineno, exc.colno,
                           exc.msg)) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise LoadError("Could not read:\n%s\n\n%s" % (path, exc)) from exc

    pts = _points_from_json(raw)
    if pts is not None and len(pts) >= 2:
        return pts

    if isinstance(raw, dict):
        keys = ", ".join(sorted(raw.keys())[:15]) or "(no keys)"
        detail = "Its top-level keys are: %s" % keys
    elif isinstance(raw, list):
        detail = ("It is a list of %d %s." %
                  (len(raw), type(raw[0]).__name__ if raw else "items"))
    else:
        detail = "It holds a bare %s." % type(raw).__name__
    raise LoadError(
        "No centerline points found in:\n%s\n\n%s\n\nExpected a 3D Slicer "
        "markups file, {\"points\": [[x,y,z], ...]}, or a list of "
        "{\"x\":..,\"y\":..,\"z\":..} records. If this file is a report "
        "rather than a centerline, look for a _centerline.vtk or "
        "_centerline.csv beside it." % (path, detail))


def _points_from_json(node, _depth=0):
    """Best-effort (N,3) extraction from a decoded JSON structure."""
    if _depth > 6 or node is None:
        return None

    if isinstance(node, list):
        if len(node) >= 2 and all(
                isinstance(v, (list, tuple)) and len(v) >= 3
                and all(isinstance(c, (int, float)) for c in v[:3])
                for v in node):
            return np.asarray([[float(v[0]), float(v[1]), float(v[2])]
                               for v in node], float)
        if node and all(isinstance(v, dict) for v in node):
            # Records: either {"position":[...]} or {"x":..,"y":..,"z":..}
            out = []
            for item in node:
                pos = item.get("position") or item.get("Position")
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    out.append([float(pos[0]), float(pos[1]), float(pos[2])])
                    continue
                keys = {_normalise(k): k for k in item}
                got = {}
                for axis, aliases in _AXIS_ALIASES.items():
                    for alias in aliases:
                        if alias in keys:
                            got[axis] = item[keys[alias]]
                            break
                if len(got) == 3:
                    try:
                        out.append([float(got["x"]), float(got["y"]),
                                    float(got["z"])])
                    except (TypeError, ValueError):
                        pass
            if len(out) >= 2:
                return np.asarray(out, float)
            # Not records themselves -- recurse (e.g. Slicer's "markups" list).
            for item in node:
                found = _points_from_json(item, _depth + 1)
                if found is not None and len(found) >= 2:
                    return found
        return None

    if isinstance(node, dict):
        for key in ("controlPoints", "control_points", "points",
                    "centerline_points", "centerline", "curve", "markups",
                    "vertices", "coordinates"):
            if key in node:
                found = _points_from_json(node[key], _depth + 1)
                if found is not None and len(found) >= 2:
                    return found
        # Separate x/y/z arrays, as a plotting export would write them.
        keys = {_normalise(k): k for k in node}
        cols = {}
        for axis, aliases in _AXIS_ALIASES.items():
            for alias in aliases:
                if alias in keys and isinstance(node[keys[alias]], list):
                    cols[axis] = node[keys[alias]]
                    break
        if len(cols) == 3 and len(cols["x"]) == len(cols["y"]) == len(cols["z"]):
            try:
                return np.asarray(
                    [[float(a), float(b), float(c)]
                     for a, b, c in zip(cols["x"], cols["y"], cols["z"])], float)
            except (TypeError, ValueError):
                pass
        for value in node.values():
            found = _points_from_json(value, _depth + 1)
            if found is not None and len(found) >= 2:
                return found
    return None


def load_centerline_file(path, progress=None):
    """Read a centerline polyline as (N, 3) world mm from .vtk/.vtp/.csv/.json.

    Reads the file the user actually chose. An earlier version inferred a case
    ID and searched the directory for a matching artifact, which silently
    loaded a *different* file whenever the naming did not line up.
    """
    if not os.path.isfile(path):
        raise LoadError("File not found:\n%s" % path)
    ext = os.path.splitext(path)[1].lower()
    if progress:
        progress("Reading centerline %s…" % os.path.basename(path))

    if ext == ".csv":
        return read_centerline_csv(path)

    if ext == ".json":
        return read_centerline_json(path)

    if ext in (".vtk", ".vtp"):
        try:
            poly = _read_polydata(path)
            if poly is not None:
                if poly.GetNumberOfPoints() == 0:
                    raise LoadError(
                        "%s holds no points.\n\nIt may be empty or "
                        "written in an unexpected format."
                        % os.path.basename(path))
                points = _polydata_points(poly)
                lines = _polydata_cells(poly.GetLines())
            else:
                from stent_html_export import read_vtk_polydata
                points, lines = read_vtk_polydata(path)
                points = np.asarray(points, float)
        except LoadError:
            # Already a message aimed at the user; wrapping it would bury it.
            raise
        except Exception as exc:                               # noqa: BLE001
            raise LoadError("Could not read centerline:\n%s\n\n%s"
                            % (path, exc)) from exc
        if lines:
            # Longest polyline = the vessel itself, not a stray fragment. The
            # cell gives point *order*, which the raw POINTS array does not
            # guarantee.
            return points[np.asarray(max(lines, key=len), np.int64)]
        if len(points) >= 2:
            return points
        raise LoadError("No polyline found in:\n%s" % path)

    raise LoadError(
        "Unsupported centerline format '%s'.\n"
        "Expected .vtk, .vtp, .csv or .json." % ext)

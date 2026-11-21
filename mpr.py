#!/usr/bin/env python3
"""
Multi-Planar Reconstruction (MPR) helpers.

The viewer stores its volume as a slice-major numpy array with shape
(Z, Y, X) - Z is the stack of acquired slices (axial), Y is the row axis
within a slice, X is the column axis. Voxel spacing is carried alongside
as (dz, dy, dx) in mm (dz = slice spacing, dy = row spacing, dx = column
spacing).

A single 3D cursor index (z, y, x) identifies the voxel every viewport is
centred on. Each orthogonal plane is just a 2D slice of the volume taken
by holding one of the three cursor components fixed:

    axial    (XY plane) : volume[z, :, :]   -> holds Z fixed
    sagittal (YZ plane) : volume[:, :, x]   -> holds X fixed
    coronal  (XZ plane) : volume[:, y, :]   -> holds Y fixed

This module only deals with plain numpy indexing / arithmetic so it has
no Qt or VTK dependency and can be unit-tested in isolation.
"""

import numpy as np

PLANES = ("axial", "sagittal", "coronal")


def plane_slice(volume, plane, cursor):
    """Extract the 2D slice of `volume` for `plane` at the given 3D cursor.

    `cursor` is (z, y, x). Returns a 2D array; for axial that's (Y, X),
    for sagittal it's (Z, Y), for coronal it's (Z, X).
    """
    z, y, x = cursor
    if plane == "axial":
        return volume[z, :, :]
    if plane == "sagittal":
        return volume[:, :, x]
    if plane == "coronal":
        return volume[:, y, :]
    raise ValueError(f"Unknown plane: {plane}")


def plane_aspect(plane, spacing_zyx):
    """Pixel aspect ratio (row_spacing / col_spacing) to draw `plane` undistorted."""
    dz, dy, dx = spacing_zyx
    if plane == "axial":
        return (dy / dx) if dx else 1.0
    if plane == "sagittal":
        return (dz / dy) if dy else 1.0
    if plane == "coronal":
        return (dz / dx) if dx else 1.0
    raise ValueError(f"Unknown plane: {plane}")


def plane_crosshair(plane, cursor):
    """(row, col) within `plane`'s 2D image where the other two axes cross it."""
    z, y, x = cursor
    if plane == "axial":
        return (y, x)
    if plane == "sagittal":
        return (z, y)
    if plane == "coronal":
        return (z, x)
    raise ValueError(f"Unknown plane: {plane}")


def cursor_from_pick(plane, cursor, row, col):
    """New (z, y, x) cursor after a click/drag at (row, col) inside `plane`."""
    z, y, x = cursor
    if plane == "axial":
        return (z, row, col)
    if plane == "sagittal":
        return (row, col, x)
    if plane == "coronal":
        return (row, y, col)
    raise ValueError(f"Unknown plane: {plane}")


def scroll_axis(plane):
    """Which cursor component (0=z, 1=y, 2=x) the mouse wheel moves in `plane`."""
    if plane == "axial":
        return 0
    if plane == "sagittal":
        return 2
    if plane == "coronal":
        return 1
    raise ValueError(f"Unknown plane: {plane}")


def clamp_cursor(cursor, shape):
    z, y, x = cursor
    nz, ny, nx = shape
    return (
        int(np.clip(z, 0, nz - 1)),
        int(np.clip(y, 0, ny - 1)),
        int(np.clip(x, 0, nx - 1)),
    )


def downsample_strides(shape, spacing_zyx, max_dim=220):
    """Integer strides (sz, sy, sx) that shrink `shape` to roughly `max_dim`
    voxels per axis, so a full volume can be safely copied into VTK for 3D
    rendering without the multi-hundred-MB spike a full-resolution deep
    copy would cause on a memory-constrained machine.
    """
    return tuple(max(1, dim // max_dim) for dim in shape)


def downsampled_spacing(spacing_zyx, strides):
    return tuple(sp * st for sp, st in zip(spacing_zyx, strides))

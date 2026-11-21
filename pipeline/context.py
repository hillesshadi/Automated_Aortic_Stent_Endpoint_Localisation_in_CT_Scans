#!/usr/bin/env python3
"""
CaseContext -- the single mutable object every pipeline stage reads from and
writes into, and the only thing the GUI widgets are given.

Each stage (preprocess -> centerline -> stent fit -> export) fills in its own
fields and leaves the rest alone, so a stage can be re-run after a parameter
change without discarding the stages before it. Keeping this in one place is
also what lets the worker threads hand a finished result back to the GUI as a
single object rather than a bag of loose signals.

Coordinates follow the project convention throughout:
    volume      numpy (z, y, x), int16 Hounsfield units
    spacing_zyx (dz, dy, dx) mm
    geom        (origin_xyz, spacing_xyz, direction_3x3) -- SimpleITK order,
                which is what stent_endpoint_sop.sample_plane and
                stentlib.centerline both expect
    centerline  (N, 3) world/LPS millimetres
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CaseContext:
    # ---- identity / provenance ---------------------------------------------
    case_id: str = ""
    source_path: str = ""
    source_kind: str = ""            # "dicom" | "image" | "mesh"

    # ---- volume ------------------------------------------------------------
    volume: Optional[np.ndarray] = None
    spacing_zyx: Optional[tuple] = None
    geom: Optional[tuple] = None

    # ---- masks -------------------------------------------------------------
    aorta_mask: Optional[np.ndarray] = None
    stent_mask: Optional[np.ndarray] = None
    aorta_seed: Optional[tuple] = None

    # An externally supplied segmentation (--reference, or loaded in the GUI).
    # Kept separate from aorta_mask so the two can be compared: once
    # aorta_mask IS the reference, Dice against it is 1.0 by construction and
    # means nothing, and `aorta_mask_source` is what lets the metrics report
    # tell the two situations apart.
    reference_mask: Optional[np.ndarray] = None
    reference_name: str = ""
    aorta_mask_source: str = ""        # "segmented" | "reference" | ""

    # ---- geometry ----------------------------------------------------------
    centerline: Optional[np.ndarray] = None      # (N,3) world mm
    resampler: Any = None                        # oblique_mpr.CenterlineResampler
    surface_mesh: Any = None                     # (verts, faces) or None
    stent_mesh: Any = None

    # ---- results -----------------------------------------------------------
    stent_result: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # ---- convenience -------------------------------------------------------
    @property
    def has_volume(self) -> bool:
        return self.volume is not None and self.volume.ndim == 3

    @property
    def has_reference(self) -> bool:
        return self.reference_mask is not None

    def use_reference_as_aorta(self) -> bool:
        """Adopt the reference mask as the working aorta segmentation.

        Returns True if it was adopted. Records the provenance so downstream
        metrics never present a self-comparison as an accuracy figure.
        """
        if self.reference_mask is None:
            return False
        self.aorta_mask = self.reference_mask
        self.aorta_mask_source = "reference"
        return True

    @property
    def has_centerline(self) -> bool:
        return self.centerline is not None and len(self.centerline) >= 2

    @property
    def shape(self):
        return self.volume.shape if self.has_volume else None

    def describe(self) -> str:
        if not self.has_volume:
            return "no volume loaded"
        nz, ny, nx = self.volume.shape
        dz, dy, dx = self.spacing_zyx
        return ("%s — %d×%d×%d voxels, %.2f×%.2f×%.2f mm, HU %d…%d"
                % (self.case_id or "case", nz, ny, nx, dz, dy, dx,
                   int(self.volume.min()), int(self.volume.max())))

    def note(self, text: str):
        self.notes.append(text)
        return text

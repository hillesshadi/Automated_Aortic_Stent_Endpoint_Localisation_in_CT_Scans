#!/usr/bin/env python3
"""
Stent device specifications and the landmark table derived from them.

A device is described by a small JSON file rather than by GUI checkboxes,
because the numbers that matter (landing-zone lengths, target length, labelled
diameter) come from the manufacturer's IFU and belong with the case record --
retyping them into spin boxes for every run is how they end up wrong.

The landmark table answers the question the operator actually asks: given this
device and this centerline, where do the two landing zones start and end in
patient coordinates, and does the vessel there match what the device expects.
Everything is reported in world/LPS millimetres and, alongside it, the voxel
index -- the first is what a report needs, the second is what you type into a
viewer to go and look.
"""

import json
import os

import numpy as np

from . import paths  # noqa: F401
from .context import CaseContext


class SpecError(RuntimeError):
    """A spec file the user can fix by editing it."""


# Field -> (aliases, unit, required). Aliases exist because these files are
# hand-written as often as they are exported.
SPEC_FIELDS = {
    "proximal_land_mm": (("proximal_land_mm", "proximal_land", "prox_land_mm",
                          "proximal_landing_mm", "proximal_landing_zone_mm"),
                         "mm", True),
    "distal_land_mm":   (("distal_land_mm", "distal_land", "dist_land_mm",
                          "distal_landing_mm", "distal_landing_zone_mm"),
                         "mm", True),
    "target_length_mm": (("target_length_mm", "target_length", "length_mm",
                          "device_length_mm", "covered_length_mm"),
                         "mm", True),
    "diameter_mm":      (("diameter_mm", "diameter", "device_diameter_mm",
                          "nominal_diameter_mm", "labelled_diameter_mm"),
                         "mm", True),
    "oversizing_pct":   (("oversizing_pct", "oversizing", "oversize_pct"),
                         "%", False),
    "device":           (("device", "device_name", "name", "model"),
                         "", False),
    "manufacturer":     (("manufacturer", "vendor"), "", False),
    "notes":            (("notes", "comment", "comments"), "", False),
}

TEMPLATE = {
    "device": "Example thoracic endograft",
    "manufacturer": "",
    "proximal_land_mm": 20.0,
    "distal_land_mm": 20.0,
    "target_length_mm": 150.0,
    "diameter_mm": 30.0,
    "oversizing_pct": 15.0,
    "notes": "Lengths are landing-zone requirements from the IFU.",
}


def _normalise(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum() or ch == "_")


def load_specs(path):
    """Parse and validate a stent specification JSON file."""
    if not os.path.isfile(path):
        raise SpecError("File not found:\n%s" % path)
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SpecError(
            "%s is not valid JSON.\n\nLine %d, column %d: %s"
            % (os.path.basename(path), exc.lineno, exc.colno, exc.msg)) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise SpecError("Could not read:\n%s\n\n%s" % (path, exc)) from exc

    if isinstance(raw, list):
        raise SpecError(
            "%s holds a list of devices.\n\nThis loader expects one device per "
            "file -- a single JSON object." % os.path.basename(path))
    if not isinstance(raw, dict):
        raise SpecError("Expected a JSON object, got %s."
                        % type(raw).__name__)

    # Accept a nested {"stent": {...}} or {"device": {...}} wrapper.
    for wrapper in ("stent", "specs", "specification"):
        if wrapper in raw and isinstance(raw[wrapper], dict):
            raw = {**{k: v for k, v in raw.items() if k != wrapper},
                   **raw[wrapper]}
            break

    lookup = {_normalise(k): v for k, v in raw.items()}
    specs, missing = {"source_file": path}, []
    for field, (aliases, unit, required) in SPEC_FIELDS.items():
        value = None
        for alias in aliases:
            if _normalise(alias) in lookup:
                value = lookup[_normalise(alias)]
                break
        if value is None:
            if required:
                missing.append("%s (%s)" % (field, unit) if unit else field)
            continue
        if unit in ("mm", "%"):
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise SpecError(
                    "'%s' must be a number, got %r." % (field, value)) from None
            if value <= 0 and field != "oversizing_pct":
                raise SpecError("'%s' must be positive, got %g." % (field, value))
        specs[field] = value

    if missing:
        raise SpecError(
            "%s is missing required field(s):\n\n    %s\n\nA minimal file looks "
            "like:\n\n%s" % (os.path.basename(path), "\n    ".join(missing),
                             json.dumps(TEMPLATE, indent=2)))
    return specs


def write_template(path):
    """Write a starter spec file, so the format never has to be guessed."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(TEMPLATE, handle, indent=2)
    return path


# --------------------------------------------------------------------------- #
#  Landmarks
# --------------------------------------------------------------------------- #
def _station_row(ctx: CaseContext, arc_mm, label, role, lumen_hu=200.0):
    """One landmark row: arc position, world mm, voxel index, lumen diameter."""
    from stent_endpoint_sop import world_to_idx

    rs = ctx.resampler
    total = rs.total_length_mm
    arc = float(np.clip(arc_mm, 0.0, total))
    index = int(rs.index_at_mm(arc))
    world = np.asarray(rs.points[index], float)

    row = {
        "label": label,
        "role": role,
        "arc_mm": arc,
        "index": index,
        "x_mm": float(world[0]), "y_mm": float(world[1]),
        "z_mm": float(world[2]),
        "clamped": abs(arc - float(arc_mm)) > 1e-6,
    }
    try:
        ix, iy, iz = world_to_idx(world, ctx.geom)
        row.update({"i": int(round(ix)), "j": int(round(iy)),
                    "k": int(round(iz))})
    except Exception:                                          # noqa: BLE001
        row.update({"i": None, "j": None, "k": None})

    try:
        from .centerline import cross_section_at
        _patch, _extent, metrics, _info = cross_section_at(
            ctx, index, lumen_hu=lumen_hu)
        row["lumen_diameter_mm"] = float(metrics.get("diameter_mm", 0.0))
        row["lumen_area_mm2"] = float(metrics.get("area_mm2", 0.0))
    except Exception:                                          # noqa: BLE001
        row["lumen_diameter_mm"] = float("nan")
        row["lumen_area_mm2"] = float("nan")
    return row


def landmark_table(ctx: CaseContext, specs=None, lumen_hu=200.0,
                   manual=None, progress=None):
    """Proximal and distal landing-zone landmarks along the centerline.

    Anchors, in order of preference:
      1. `manual` -- {"proximal_arc_mm": .., "distal_arc_mm": ..} set by the
         operator picking stations in the GUI;
      2. the fitted stent crowns from ctx.stent_result;
      3. the two ends of the centerline, which is only a fallback for
         previewing a device against an unstented vessel.

    The landing zones extend *away* from the covered segment: proximal
    upstream, distal downstream. That is the direction the seal is measured in.
    """
    if ctx.resampler is None:
        raise SpecError(
            "No centerline.\n\nRun Extract Centerline first -- landmarks are "
            "positions along it.")

    rs = ctx.resampler
    total = rs.total_length_mm
    result = ctx.stent_result or {}

    if manual and manual.get("proximal_arc_mm") is not None \
            and manual.get("distal_arc_mm") is not None:
        prox_arc = float(manual["proximal_arc_mm"])
        dist_arc = float(manual["distal_arc_mm"])
        anchor = "manual picks"
    elif "proximal_arc_mm" in result and "distal_arc_mm" in result:
        prox_arc = float(result["proximal_arc_mm"])
        dist_arc = float(result["distal_arc_mm"])
        anchor = "fitted stent crowns"
    else:
        prox_arc, dist_arc = 0.0, total
        anchor = "centerline ends (no stent fitted)"

    # Orient so proximal is the smaller arc position; the fit does not
    # guarantee which crown came out first.
    if prox_arc > dist_arc:
        prox_arc, dist_arc = dist_arc, prox_arc

    prox_land = float((specs or {}).get("proximal_land_mm", 0.0) or 0.0)
    dist_land = float((specs or {}).get("distal_land_mm", 0.0) or 0.0)

    rows = [
        _station_row(ctx, prox_arc - prox_land,
                     "Proximal landing — start", "proximal", lumen_hu),
        _station_row(ctx, prox_arc,
                     "Proximal landing — end (stent start)", "proximal",
                     lumen_hu),
        _station_row(ctx, dist_arc,
                     "Distal landing — start (stent end)", "distal", lumen_hu),
        _station_row(ctx, dist_arc + dist_land,
                     "Distal landing — end", "distal", lumen_hu),
    ]

    covered = dist_arc - prox_arc
    summary = {
        "anchor": anchor,
        "covered_length_mm": covered,
        "centerline_length_mm": total,
        "proximal_arc_mm": prox_arc,
        "distal_arc_mm": dist_arc,
    }
    if specs:
        summary.update(_compliance(specs, covered, rows, total,
                                   prox_arc, dist_arc))

    table = {"rows": rows, "summary": summary, "specs": dict(specs or {})}
    ctx.metrics["stent_landmarks"] = table
    if progress:
        progress(100, "Landmarks ready (%s)." % anchor)
    return table


def _compliance(specs, covered, rows, total, prox_arc, dist_arc):
    """Compare the measured vessel against what the device expects."""
    out, warnings = {}, []

    target = float(specs.get("target_length_mm", 0.0) or 0.0)
    if target > 0:
        out["target_length_mm"] = target
        out["length_difference_mm"] = covered - target
        if abs(covered - target) > 0.1 * target:
            warnings.append(
                "Covered length %.1f mm differs from the %.0f mm device by "
                "%+.1f mm (>10%%)." % (covered, target, covered - target))

    # Landing zones must fit inside the imaged vessel, or the seal cannot be
    # assessed from this scan at all.
    if prox_arc - float(specs.get("proximal_land_mm", 0) or 0) < -1e-6:
        warnings.append(
            "Proximal landing zone extends %.1f mm beyond the start of the "
            "centerline — it is outside the scanned volume."
            % (float(specs.get("proximal_land_mm", 0)) - prox_arc))
    if dist_arc + float(specs.get("distal_land_mm", 0) or 0) > total + 1e-6:
        warnings.append(
            "Distal landing zone extends %.1f mm beyond the end of the "
            "centerline — it is outside the scanned volume."
            % (dist_arc + float(specs.get("distal_land_mm", 0)) - total))

    device_d = float(specs.get("diameter_mm", 0.0) or 0.0)
    if device_d > 0:
        out["device_diameter_mm"] = device_d
        seal = [r["lumen_diameter_mm"] for r in rows
                if np.isfinite(r.get("lumen_diameter_mm", float("nan")))]
        if seal:
            neck = float(min(seal))
            out["min_landing_diameter_mm"] = neck
            if neck > 0:
                # Oversizing is quoted against the vessel, which is what the
                # IFU windows are specified against.
                out["oversizing_pct"] = 100.0 * (device_d - neck) / neck
                planned = specs.get("oversizing_pct")
                if planned is not None and \
                        abs(out["oversizing_pct"] - float(planned)) > 10.0:
                    warnings.append(
                        "Measured oversizing %.0f%% vs %.0f%% planned."
                        % (out["oversizing_pct"], float(planned)))
    out["warnings"] = warnings
    return out


def export_landmarks(table, out_dir, case_id):
    """Write the landmark table as CSV plus JSON."""
    import csv
    os.makedirs(out_dir, exist_ok=True)
    stem = case_id or "case"
    written = []

    cpath = os.path.join(out_dir, "%s_stent_landmarks.csv" % stem)
    cols = ["label", "role", "arc_mm", "index", "x_mm", "y_mm", "z_mm",
            "i", "j", "k", "lumen_diameter_mm", "lumen_area_mm2", "clamped"]
    with open(cpath, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in table["rows"]:
            w.writerow(row)
    written.append(cpath)

    jpath = os.path.join(out_dir, "%s_stent_landmarks.json" % stem)
    with open(jpath, "w", encoding="utf-8") as handle:
        json.dump({"case_id": case_id, **table}, handle, indent=2,
                  default=str)
    written.append(jpath)
    return written

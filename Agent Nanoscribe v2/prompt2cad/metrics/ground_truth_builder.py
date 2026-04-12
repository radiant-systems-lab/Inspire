"""
Ground-truth CadQuery shape builder from dataset CSV parameters.

All parameters in the CSV are stored in **nanometres (nm)**.
This module converts them to **micrometres (µm)** by dividing by 1000, so
that the resulting shapes are on the same scale as what the generator LLM
produces (which operates in µm).

Usage::

    from prompt2cad.metrics.ground_truth_builder import build_gt_shape_um

    # params_nm: dict from parameters_ground_truth column (values in nm)
    shape_wp = build_gt_shape_um(shape="cylinder",
                                  params_nm={"radius": 3903.69, "height": 6863.74})
    # shape_wp is a cadquery.Workplane with the geometry ready to measure
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gt_shape_um(
    shape: str,
    params_nm: Dict[str, float],
) -> Any:
    """Build a CadQuery Workplane from nm-scale dataset parameters.

    Args:
        shape:     Shape type key, e.g. ``"cylinder"``, ``"cone_on_cylinder"``.
        params_nm: Parameter dict with values in **nanometres** as stored in
                   ``parameters_ground_truth`` in the dataset CSV.

    Returns:
        ``cadquery.Workplane`` containing the assembled geometry.

    Raises:
        ImportError: if cadquery is not installed.
        ValueError:  if *shape* is unrecognised.
    """
    import cadquery as cq

    # Convert all nm values → µm
    p: Dict[str, float] = {k: v / 1000.0 for k, v in params_nm.items()}

    builders = {
        # ── Single primitives ─────────────────────────────────────────────
        "cylinder":          _build_cylinder,
        "box":               _build_box,
        "sphere":            _build_sphere,
        "cone":              _build_cone,
        "torus":             _build_torus,
        # ── Stacked composites ────────────────────────────────────────────
        "cone_on_cylinder":  _build_cone_on_cylinder,
        "sphere_on_cylinder":_build_sphere_on_cylinder,
        "box_on_cylinder":   _build_box_on_cylinder,
    }

    builder = builders.get(shape)
    if builder is None:
        raise ValueError(
            f"Unknown shape {shape!r}. "
            f"Available: {sorted(builders)}"
        )

    return builder(cq, p)


# ---------------------------------------------------------------------------
# Single primitive builders (all return cq.Workplane)
# ---------------------------------------------------------------------------

def _build_cylinder(cq: Any, p: Dict[str, float]) -> Any:
    r = p["radius"]
    h = p["height"]
    wp = cq.Workplane("XY").cylinder(h, r)
    return _apply_array(cq, wp, p, r * 2, r * 2)


def _build_box(cq: Any, p: Dict[str, float]) -> Any:
    w = p["width"]
    d = p["depth"]
    h = p["height"]
    wp = cq.Workplane("XY").box(w, d, h)
    return _apply_array(cq, wp, p, w, d)


def _build_sphere(cq: Any, p: Dict[str, float]) -> Any:
    r = p["radius"]
    wp = cq.Workplane("XY").sphere(r)
    return _apply_array(cq, wp, p, r * 2, r * 2)


def _build_cone(cq: Any, p: Dict[str, float]) -> Any:
    """Cone: solid truncated to a point (top_radius=0)."""
    r = p["radius"]
    h = p["height"]
    # CadQuery: Solid.makeCone(r1, r2, height)
    solid = cq.Solid.makeCone(r, 0.0, h)
    wp = cq.Workplane("XY").add(solid)
    return _apply_array(cq, wp, p, r * 2, r * 2)


def _build_torus(cq: Any, p: Dict[str, float]) -> Any:
    R = p["major_radius"]
    r = p["minor_radius"]
    solid = cq.Solid.makeTorus(R, r)
    wp = cq.Workplane("XY").add(solid)
    return _apply_array(cq, wp, p, (R + r) * 2, (R + r) * 2)


# ---------------------------------------------------------------------------
# Stacked composite builders
# ---------------------------------------------------------------------------

def _build_cone_on_cylinder(cq: Any, p: Dict[str, float]) -> Any:
    """Cylinder base with cone on top, union, Z-centred at bottom."""
    base_r  = p["base_radius"]
    base_h  = p["base_height"]
    top_r   = p["top_radius"]
    top_h   = p["top_height"]

    # Base cylinder: centred at z = base_h/2
    cyl_solid = cq.Solid.makeCylinder(base_r, base_h)   # bottom at origin
    # Top cone: sits on top of cylinder
    cone_solid = cq.Solid.makeCone(top_r, 0.0, top_h)
    cone_solid = cone_solid.translate(cq.Vector(0, 0, base_h))

    result_solid = cyl_solid.fuse(cone_solid)
    wp = cq.Workplane("XY").add(result_solid)
    return _apply_array(cq, wp, p, base_r * 2, base_r * 2)


def _build_sphere_on_cylinder(cq: Any, p: Dict[str, float]) -> Any:
    base_r    = p["base_radius"]
    base_h    = p["base_height"]
    sph_r     = p["sphere_radius"]

    cyl_solid = cq.Solid.makeCylinder(base_r, base_h)
    sph_solid = cq.Solid.makeSphere(sph_r)
    sph_solid = sph_solid.translate(cq.Vector(0, 0, base_h + sph_r))

    result_solid = cyl_solid.fuse(sph_solid)
    wp = cq.Workplane("XY").add(result_solid)
    return _apply_array(cq, wp, p, base_r * 2, base_r * 2)


def _build_box_on_cylinder(cq: Any, p: Dict[str, float]) -> Any:
    base_r  = p["base_radius"]
    base_h  = p["base_height"]
    box_w   = p["box_width"]
    box_d   = p["box_depth"]
    box_h   = p["box_height"]

    cyl_solid = cq.Solid.makeCylinder(base_r, base_h)
    # Box centred at XY, translated to sit on top of cylinder
    box_solid = cq.Solid.makeBox(box_w, box_d, box_h,
                                  cq.Vector(-box_w / 2, -box_d / 2, base_h))
    result_solid = cyl_solid.fuse(box_solid)
    wp = cq.Workplane("XY").add(result_solid)
    return _apply_array(cq, wp, p, base_r * 2, base_r * 2)


# ---------------------------------------------------------------------------
# Absolute-position builder (delegates to base shape + translate)
# ---------------------------------------------------------------------------

def build_gt_abs_shape_um(
    shape: str,
    params_nm: Dict[str, float],
) -> Any:
    """Build an absolute-positioned shape (pos_x, pos_y, pos_z in nm)."""
    import cadquery as cq

    p = {k: v / 1000.0 for k, v in params_nm.items()}
    px = p.get("pos_x", 0.0)
    py = p.get("pos_y", 0.0)
    pz = p.get("pos_z", 0.0)

    # Build base shape ignoring pos parameters
    base_params_nm = {k: v for k, v in params_nm.items()
                      if k not in ("pos_x", "pos_y", "pos_z")}
    base_shape = build_gt_shape_um(shape, base_params_nm)

    # Translate
    solid = base_shape.val()
    translated = solid.translate(cq.Vector(px, py, pz))
    return cq.Workplane("XY").add(translated)


# ---------------------------------------------------------------------------
# Array helper
# ---------------------------------------------------------------------------

def _apply_array(
    cq: Any,
    wp: Any,
    p: Dict[str, float],
    unit_bx: float,
    unit_by: float,
) -> Any:
    """If period_x/period_y and count_x/count_y are present, tile the unit cell.

    Uses a manual translate approach so it works on any shape type without
    requiring a 2D profile (rarray requires a Workplane sketch context).
    """
    cx = int(round(p.get("count_x", 1)))
    cy = int(round(p.get("count_y", 1)))
    px = p.get("period_x", unit_bx)
    py = p.get("period_y", unit_by)

    if cx <= 1 and cy <= 1:
        return wp

    # Get base solid
    base_solid = wp.val()

    # Build compound of all instances
    solids = []
    for ix in range(cx):
        for iy in range(cy):
            dx = ix * px
            dy = iy * py
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                solids.append(base_solid)
            else:
                solids.append(
                    base_solid.translate(cq.Vector(dx, dy, 0))
                )

    # Fuse into compound
    compound = solids[0]
    for s in solids[1:]:
        try:
            compound = compound.fuse(s)
        except Exception:
            # If fuse fails (touching/overlapping), just union via Compound
            pass

    return cq.Workplane("XY").add(compound)

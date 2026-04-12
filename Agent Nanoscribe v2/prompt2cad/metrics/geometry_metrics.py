"""
Geometric metrics extractor for Prompt2CAD verification.

Works on any CadQuery Workplane or Shape object and optionally on the source
code string for code-complexity features.

Usage::

    import cadquery as cq
    from prompt2cad.metrics.geometry_metrics import compute_metrics

    shape = cq.Workplane("XY").box(10, 5, 2)
    m = compute_metrics(shape)
    print(m["volume"], m["surface_area"])
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    shape_or_wp: Any,
    code_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute all geometric metrics from a CadQuery shape or Workplane.

    Args:
        shape_or_wp: A ``cadquery.Workplane`` **or** a ``cadquery.Shape``
                     (``cq.Solid``, ``cq.Compound``, etc.).
        code_str:    Optional Python source string; used for code-complexity
                     metrics only (Category 8).

    Returns:
        Flat dict with ~55 scalar entries.  Missing values are ``None``.
    """
    # Normalise to a Shape
    if hasattr(shape_or_wp, "val"):          # Workplane
        shape = shape_or_wp.val()
    elif hasattr(shape_or_wp, "Volume"):     # already a Shape/Solid
        shape = shape_or_wp
    else:
        raise TypeError(f"Expected Workplane or Shape, got {type(shape_or_wp)}")

    m: Dict[str, Any] = {}

    # ── 1. Core geometry ────────────────────────────────────────────────────
    try:
        vol  = shape.Volume()
        area = shape.Area()
    except Exception:
        vol, area = None, None

    m["volume"]       = vol
    m["surface_area"] = area
    m["va_ratio"]     = (vol / area) if (vol and area and area > 0) else None

    bb = None
    bx = by = bz = None
    try:
        bb = shape.BoundingBox()
        bx = bb.xmax - bb.xmin
        by = bb.ymax - bb.ymin
        bz = bb.zmax - bb.zmin
    except Exception:
        pass

    m["bbox_x"] = bx
    m["bbox_y"] = by
    m["bbox_z"] = bz
    m["aspect_xy"] = (bx / by) if (bx and by and by > 0) else None
    m["aspect_xz"] = (bx / bz) if (bx and bz and bz > 0) else None
    m["aspect_yz"] = (by / bz) if (by and bz and bz > 0) else None

    # ── 2. Fill metrics ─────────────────────────────────────────────────────
    bbox_vol = (bx * by * bz) if (bx and by and bz) else None
    m["bbox_volume"]   = bbox_vol
    m["fill_fraction"] = (vol / bbox_vol) if (vol and bbox_vol and bbox_vol > 0) else None

    # ── 3. Base contact ─────────────────────────────────────────────────────
    base = _compute_base_contact(shape)
    m["base_contact_area"]    = base["contact_area"]
    m["base_stability_ratio"] = base["stability_ratio"]
    m["base_coverage"]        = base["coverage"]

    # ── 4. Overhang metrics (fixed: angle(normal, +Z), no abs()) ────────────
    oh = _compute_overhang(shape)
    m["overhang_max_angle_deg"]    = oh["max_angle_deg"]     # 0–180°; 180 = fully downward
    m["overhang_mean_angle_deg"]   = oh["mean_angle_deg"]    # area-weighted mean of OH faces
    m["overhang_area_fraction"]    = oh["overhang_area_fraction"]  # fraction with angle > 45°
    m["overhang_max_excess_deg"]   = oh["max_excess_deg"]    # max(0, angle - 45°)

    # ── 5. Slenderness ──────────────────────────────────────────────────────
    m["slenderness"] = (bz / min(bx, by)) if (bx and by and bz and min(bx, by) > 0) else None

    # ── 6. Curvature proxy ───────────────────────────────────────────────────
    curv = _compute_curvature_proxy(shape)
    m["curvature_mean"] = curv["mean"]
    m["curvature_max"]  = curv["max"]
    m["curvature_std"]  = curv["std"]

    # ── 7. Feature spacing ───────────────────────────────────────────────────
    fsp = _compute_feature_spacing(shape)
    m["feature_spacing_min"]     = fsp["min_spacing"]
    m["feature_spacing_mean"]    = fsp["mean_spacing"]
    m["num_disconnected_solids"] = fsp["n_solids"]

    # ── 8. Code complexity (requires code_str) ───────────────────────────────
    cx = _analyze_code_complexity(code_str)
    m["code_n_primitives"]  = cx["n_primitives"]
    m["code_n_boolean_ops"] = cx["n_boolean_ops"]
    m["code_n_patterns"]    = cx["n_patterns"]
    m["code_complexity"]    = cx["complexity"]

    # ── 9. Surface topology ──────────────────────────────────────────────────
    try:
        nf = len(shape.Faces())
        ne = len(shape.Edges())
        nv = len(shape.Vertices())
        m["num_faces"]             = nf
        m["num_edges"]             = ne
        m["num_vertices"]          = nv
        m["topological_complexity"] = nf + ne   # more stable than face count alone
    except Exception:
        m["num_faces"] = m["num_edges"] = m["num_vertices"] = None
        m["topological_complexity"] = None

    # ── 10. Printability heuristics ──────────────────────────────────────────
    m["fragility_score"] = _fragility_score(
        m.get("slenderness"),
        m.get("overhang_area_fraction"),
        m.get("fill_fraction"),
    )
    m["anchor_score"] = _anchor_score(
        m.get("base_contact_area"),
        m.get("volume"),
    )

    # ── 11. Feature density ──────────────────────────────────────────────────
    m["feature_density"] = (
        m["num_faces"] / bbox_vol
        if (m["num_faces"] and bbox_vol and bbox_vol > 0)
        else None
    )

    # ── 12. Overhang directionality ──────────────────────────────────────────
    od = _compute_overhang_directionality(shape)
    m["overhang_dir_x"]    = od["dir_x"]
    m["overhang_dir_y"]    = od["dir_y"]
    m["overhang_isotropy"] = od["isotropy"]

    # ── 13. Array metrics ────────────────────────────────────────────────────
    arr = _compute_array_metrics(shape)
    m["array_num_instances"] = arr["num_instances"]
    m["array_pitch_x"]       = arr["pitch_x"]
    m["array_pitch_y"]       = arr["pitch_y"]
    m["array_density"]       = arr["array_density"]

    # ── 14. Centre of mass offset (normalised by bbox diagonal) ─────────────
    com = _compute_com_offset(shape, bb, bx, by, bz)
    m["com_offset_x"]    = com["offset_x"]
    m["com_offset_y"]    = com["offset_y"]
    m["com_offset_z"]    = com["offset_z"]
    m["com_offset_norm"] = com["offset_norm"]   # 0–1; 0 = perfectly centred

    # ── 15. Shape entropy ────────────────────────────────────────────────────
    m["shape_entropy"] = _shape_entropy(m.get("num_faces"), m.get("curvature_mean"))

    # ── 16. Feature thickness (min wall thickness proxy) ─────────────────────
    m["feature_thickness_min"] = _compute_feature_thickness(shape)

    # ── 17. Unsupported height ────────────────────────────────────────────────
    m["unsupported_height"] = _compute_unsupported_height(shape, bb)

    # ── 18. ML forward-model features ────────────────────────────────────────
    # These match the feature names expected by StepForwardModel / nanoscribe_ml.ipynb.
    # Computed here so predict_printability() can call the trained model without
    # any approximation fallback.

    # area_volume_ratio  (inverse of va_ratio)
    m["area_volume_ratio"] = (area / vol) if (vol and vol > 0 and area) else None

    # slenderness_ratio: max_dim^3 / volume  (scale-invariant elongation proxy)
    if bx and by and bz and vol and vol > 0:
        m["slenderness_ratio"] = (max(bx, by, bz) ** 3) / vol
    else:
        m["slenderness_ratio"] = None

    # compactness: 36π V² / A³  (isoperimetric quotient — 1 for sphere, < 1 otherwise)
    if vol and area and area > 0:
        m["compactness"] = (36.0 * math.pi * vol ** 2) / (area ** 3)
    else:
        m["compactness"] = None

    # Face-type area ratios — match STEP-pipeline feature names
    plane_a = cyl_a = bsp_a = typed_total = 0.0
    try:
        for _f in shape.Faces():
            try:
                gt = _f.geomType().upper()
                fa = _f.Area()
                typed_total += fa
                if gt == "PLANE":
                    plane_a += fa
                elif gt == "CYLINDER":
                    cyl_a += fa
                elif gt in ("BSPLINE", "BEZIER", "NURBS", "BSPLINE_SURFACE"):
                    bsp_a += fa
            except Exception:
                pass
    except Exception:
        pass

    if typed_total > 0:
        m["plane_face_ratio"]       = round(plane_a / typed_total, 6)
        m["cylindrical_face_ratio"] = round(cyl_a   / typed_total, 6)
        m["bspline_face_ratio"]     = round(bsp_a   / typed_total, 6)
    else:
        m["plane_face_ratio"]       = None
        m["cylindrical_face_ratio"] = None
        m["bspline_face_ratio"]     = None

    return m


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------

def _compute_base_contact(shape: Any, z_tol: float = 0.05) -> Dict[str, Any]:
    """Faces with outward normal pointing mostly downward (n.z < -0.85) near z_min."""
    try:
        faces = shape.Faces()
        if not faces:
            return {"contact_area": None, "stability_ratio": None, "coverage": None}

        bb      = shape.BoundingBox()
        z_min   = bb.zmin
        z_range = max(bb.zmax - bb.zmin, 1e-9)

        total_area = shape.Area()
        base_area  = 0.0
        for face in faces:
            try:
                n = face.normalAt()
                if n.z < -0.85:
                    fbb = face.BoundingBox()
                    if (fbb.zmin - z_min) / z_range < z_tol:
                        base_area += face.Area()
            except Exception:
                continue

        bb_footprint = max((bb.xmax - bb.xmin) * (bb.ymax - bb.ymin), 1e-12)
        return {
            "contact_area":    base_area,
            "stability_ratio": base_area / total_area if total_area > 0 else None,
            "coverage":        base_area / bb_footprint,
        }
    except Exception:
        return {"contact_area": None, "stability_ratio": None, "coverage": None}


def _compute_overhang(shape: Any, threshold_deg: float = 45.0) -> Dict[str, Any]:
    """
    Overhang angles measured as angle(normal, +Z).

    - angle = 0°   → face points straight up (no support needed)
    - angle = 90°  → vertical face (side wall)
    - angle = 180° → face points straight down (needs support)

    Bug fix vs v1: removed abs() so top/bottom faces are no longer conflated.
    All faces are included; overhang = angle > threshold_deg.
    """
    try:
        faces = shape.Faces()
        if not faces:
            return {
                "max_angle_deg": None, "mean_angle_deg": None,
                "overhang_area_fraction": None, "max_excess_deg": None,
            }

        total_area    = 0.0
        overhang_area = 0.0
        oh_angles: List[float] = []   # angles of overhanging faces only
        all_angles: List[float] = []

        for face in faces:
            try:
                n  = face.normalAt()
                fa = face.Area()
                # angle(normal, +Z) ∈ [0°, 180°]
                nz          = max(-1.0, min(1.0, n.z))
                angle_deg   = math.degrees(math.acos(nz))
                excess_deg  = max(0.0, angle_deg - threshold_deg)

                all_angles.append(angle_deg)
                total_area += fa

                if angle_deg > threshold_deg:
                    overhang_area += fa
                    oh_angles.append(angle_deg)
            except Exception:
                continue

        if not all_angles:
            return {
                "max_angle_deg": None, "mean_angle_deg": None,
                "overhang_area_fraction": None, "max_excess_deg": None,
            }

        max_angle   = max(all_angles)
        max_excess  = max(0.0, max_angle - threshold_deg)
        mean_oh_ang = (sum(oh_angles) / len(oh_angles)) if oh_angles else 0.0
        oh_frac     = overhang_area / total_area if total_area > 0 else 0.0

        return {
            "max_angle_deg":          round(max_angle, 3),
            "mean_angle_deg":         round(mean_oh_ang, 3),
            "overhang_area_fraction": round(oh_frac, 4),
            "max_excess_deg":         round(max_excess, 3),
        }
    except Exception:
        return {
            "max_angle_deg": None, "mean_angle_deg": None,
            "overhang_area_fraction": None, "max_excess_deg": None,
        }


def _GEOM_TYPE_CURVATURE(gt: str) -> float:
    """Map OCC face geometry type string to a proxy curvature value."""
    gt = gt.upper()
    if gt == "PLANE":    return 0.0
    if gt == "CYLINDER": return 0.1
    if gt in ("SPHERE", "CONE", "TORUS"): return 0.3
    if gt in ("BSPLINE", "BEZIER", "NURBS"): return 0.5
    return 0.2


def _compute_curvature_proxy(shape: Any) -> Dict[str, Any]:
    """Proxy curvature via face.geomType()."""
    try:
        faces = shape.Faces()
        if not faces:
            return {"mean": None, "max": None, "std": None}

        values = []
        for face in faces:
            try:
                values.append(_GEOM_TYPE_CURVATURE(face.geomType()))
            except Exception:
                values.append(0.2)

        if not values:
            return {"mean": None, "max": None, "std": None}

        n    = len(values)
        mean = sum(values) / n
        std  = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
        return {"mean": mean, "max": max(values), "std": std}
    except Exception:
        return {"mean": None, "max": None, "std": None}


def _compute_feature_spacing(shape: Any) -> Dict[str, Any]:
    """Inter-solid centroid distances (for disconnected arrays)."""
    try:
        solids = shape.Solids()
        n = len(solids)
        if n <= 1:
            return {"min_spacing": None, "mean_spacing": None, "n_solids": n}

        centres: List = []
        for s in solids:
            try:
                c = s.Center()
                centres.append((c.x, c.y, c.z))
            except Exception:
                pass

        if len(centres) < 2:
            return {"min_spacing": None, "mean_spacing": None, "n_solids": n}

        dists = []
        for i in range(len(centres)):
            for j in range(i + 1, len(centres)):
                dx = centres[i][0] - centres[j][0]
                dy = centres[i][1] - centres[j][1]
                dz = centres[i][2] - centres[j][2]
                dists.append(math.sqrt(dx*dx + dy*dy + dz*dz))

        return {
            "min_spacing":  min(dists),
            "mean_spacing": sum(dists) / len(dists),
            "n_solids":     n,
        }
    except Exception:
        return {"min_spacing": None, "mean_spacing": None, "n_solids": None}


def _analyze_code_complexity(code_str: Optional[str]) -> Dict[str, Any]:
    """Complexity = n_primitives + 2·n_boolean_ops + 0.5·n_patterns."""
    if not code_str:
        return {"n_primitives": None, "n_boolean_ops": None,
                "n_patterns": None, "complexity": None}

    primitive_pats = [
        r'\.box\s*\(', r'\.cylinder\s*\(', r'\.sphere\s*\(', r'\.cone\s*\(',
        r'makeTorus\s*\(', r'\.extrude\s*\(', r'\.revolve\s*\(',
        r'\.loft\s*\(', r'\.sweep\s*\(',
    ]
    n_prim = sum(len(re.findall(p, code_str)) for p in primitive_pats)
    n_bool = sum(len(re.findall(p, code_str))
                 for p in [r'\.cut\s*\(', r'\.union\s*\(', r'\.intersect\s*\('])
    n_pat  = sum(len(re.findall(p, code_str))
                 for p in [r'\.rarray\s*\(', r'\.polarArray\s*\(', r'\.array\s*\('])

    return {
        "n_primitives":  n_prim,
        "n_boolean_ops": n_bool,
        "n_patterns":    n_pat,
        "complexity":    n_prim + 2 * n_bool + 0.5 * n_pat,
    }


def _fragility_score(
    slenderness: Optional[float],
    overhang_frac: Optional[float],
    fill_frac: Optional[float],
) -> Optional[float]:
    """Heuristic fragility ∈ [0, 1] — higher = more fragile.

    0.4 × slenderness_norm  +  0.4 × overhang_frac  +  0.2 × (1 - fill_frac)
    """
    if slenderness is None and overhang_frac is None and fill_frac is None:
        return None
    s  = min((slenderness or 0) / 20.0, 1.0)
    oh = float(overhang_frac or 0)
    ff = 1.0 - min(float(fill_frac or 1), 1.0)
    return round(s * 0.4 + oh * 0.4 + ff * 0.2, 4)


def _anchor_score(
    base_contact: Optional[float],
    volume: Optional[float],
) -> Optional[float]:
    """A_base / V^(2/3) — higher = better substrate adhesion."""
    if base_contact is None or volume is None or volume <= 0:
        return None
    return round(base_contact / (volume ** (2.0 / 3.0)), 6)


def _compute_overhang_directionality(shape: Any) -> Dict[str, Any]:
    """X/Y decomposition of overhang normals (faces with n.z < -0.5)."""
    try:
        faces = shape.Faces()
        sum_nx = sum_ny = total_oh_area = 0.0
        for face in faces:
            try:
                n = face.normalAt()
                if n.z < -0.5:
                    fa = face.Area()
                    sum_nx += abs(n.x) * fa
                    sum_ny += abs(n.y) * fa
                    total_oh_area += fa
            except Exception:
                continue

        if total_oh_area <= 0:
            return {"dir_x": 0.0, "dir_y": 0.0, "isotropy": 1.0}

        dx  = sum_nx / total_oh_area
        dy  = sum_ny / total_oh_area
        mag = math.sqrt(dx*dx + dy*dy)
        isotropy = 1.0 - min(abs(dx - dy) / (mag + 1e-9), 1.0) if mag > 1e-9 else 1.0
        return {"dir_x": round(dx, 4), "dir_y": round(dy, 4),
                "isotropy": round(isotropy, 4)}
    except Exception:
        return {"dir_x": None, "dir_y": None, "isotropy": None}


def _compute_array_metrics(shape: Any) -> Dict[str, Any]:
    """Pitch / density from disconnected solid centroid positions."""
    try:
        solids = shape.Solids()
        n = len(solids)
        if n < 2:
            return {"num_instances": n, "pitch_x": None, "pitch_y": None,
                    "array_density": None}

        xs = sorted({round(s.Center().x, 3) for s in solids})
        ys = sorted({round(s.Center().y, 3) for s in solids})

        bb       = shape.BoundingBox()
        footprint = max((bb.xmax - bb.xmin) * (bb.ymax - bb.ymin), 1e-12)
        return {
            "num_instances": n,
            "pitch_x":       _median_gap(xs) if len(xs) > 1 else None,
            "pitch_y":       _median_gap(ys) if len(ys) > 1 else None,
            "array_density": n / footprint,
        }
    except Exception:
        return {"num_instances": None, "pitch_x": None, "pitch_y": None,
                "array_density": None}


def _median_gap(sorted_vals: List[float]) -> Optional[float]:
    gaps = sorted([sorted_vals[i+1] - sorted_vals[i]
                   for i in range(len(sorted_vals) - 1)
                   if sorted_vals[i+1] - sorted_vals[i] > 1e-6])
    return gaps[len(gaps) // 2] if gaps else None


def _compute_com_offset(
    shape: Any,
    bb: Any,
    bx: Optional[float],
    by: Optional[float],
    bz: Optional[float],
) -> Dict[str, Any]:
    """COM offset from bbox centre, normalised by half the bbox diagonal.

    Result is dimensionless ∈ [0, 1]:
      0 = COM exactly at bbox centre
      1 = COM at a corner of the bbox  (theoretical maximum)
    """
    try:
        com   = shape.Center()
        bb_cx = (bb.xmin + bb.xmax) / 2
        bb_cy = (bb.ymin + bb.ymax) / 2
        bb_cz = (bb.zmin + bb.zmax) / 2
        ox = com.x - bb_cx
        oy = com.y - bb_cy
        oz = com.z - bb_cz
        raw_norm = math.sqrt(ox*ox + oy*oy + oz*oz)

        # Normalise by half the bbox diagonal (= max possible offset)
        if bx and by and bz:
            half_diag = math.sqrt(bx**2 + by**2 + bz**2) / 2.0
            norm = raw_norm / half_diag if half_diag > 1e-12 else 0.0
        else:
            norm = None

        return {
            "offset_x":    round(ox, 4),
            "offset_y":    round(oy, 4),
            "offset_z":    round(oz, 4),
            "offset_norm": round(norm, 4) if norm is not None else None,
        }
    except Exception:
        return {"offset_x": None, "offset_y": None,
                "offset_z": None, "offset_norm": None}


def _shape_entropy(num_faces: Optional[int], curv_mean: Optional[float]) -> Optional[float]:
    """log(1 + n_faces) × (1 + curvature_mean) — higher = more complex."""
    if num_faces is None:
        return None
    return round(math.log1p(float(num_faces)) * (1.0 + float(curv_mean or 0)), 4)


def _compute_feature_thickness(shape: Any) -> Optional[float]:
    """Minimum wall thickness proxy.

    For every pair of faces whose outward normals are roughly antiparallel
    (dot product < –0.5), the face-to-face distance is estimated by
    projecting the vector between centroids onto the shared normal direction.
    The minimum over all such pairs is returned.

    This gives the correct answer for prismatic shapes (boxes, cylinders)
    and a reasonable lower bound for more complex geometry.
    """
    try:
        faces = shape.Faces()
        if len(faces) < 2:
            return None

        # Collect (normal_xyz, centroid_xyz) per face
        face_data: List[tuple] = []
        for face in faces:
            try:
                n = face.normalAt()
                c = face.Center()
                face_data.append((n.x, n.y, n.z, c.x, c.y, c.z))
            except Exception:
                continue

        min_t = float("inf")
        for i in range(len(face_data)):
            nx1, ny1, nz1, cx1, cy1, cz1 = face_data[i]
            for j in range(i + 1, len(face_data)):
                nx2, ny2, nz2, cx2, cy2, cz2 = face_data[j]
                dot = nx1 * nx2 + ny1 * ny2 + nz1 * nz2
                if dot >= -0.5:          # not antiparallel → skip
                    continue
                # Distance along face-1 normal between the two centroids
                dx, dy, dz = cx2 - cx1, cy2 - cy1, cz2 - cz1
                dist = abs(dx * nx1 + dy * ny1 + dz * nz1)
                if dist > 1e-6:
                    min_t = min(min_t, dist)

        return round(min_t, 4) if min_t != float("inf") else None
    except Exception:
        return None


def _compute_unsupported_height(shape: Any, bb: Any) -> Optional[float]:
    """Vertical span of overhanging material above the base support level.

    Algorithm:
      1. Find z_base = z_min (build plate level).
      2. Collect all face centroids whose face has angle(normal, +Z) > 45°
         (i.e. needs support).
      3. unsupported_height = max_z_overhang - z_base.

    For a fully supported solid (cylinder, box) this will be ~0.
    For a cone-on-cylinder the cone faces all overhang and the metric
    returns the vertical span of the cone above the cylinder top.
    """
    try:
        if bb is None:
            return None
        z_base  = bb.zmin
        THRESHOLD = 45.0

        max_z_oh = z_base   # initialise to base; grows if overhangs are found
        faces = shape.Faces()
        for face in faces:
            try:
                n  = face.normalAt()
                nz = max(-1.0, min(1.0, n.z))
                if math.degrees(math.acos(nz)) > THRESHOLD:
                    cz = face.Center().z
                    if cz > max_z_oh:
                        max_z_oh = cz
            except Exception:
                continue

        span = max_z_oh - z_base
        return round(span, 4) if span > 1e-6 else 0.0
    except Exception:
        return None

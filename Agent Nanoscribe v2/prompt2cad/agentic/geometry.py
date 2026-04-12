"""Parameterized geometry generation and analytic feature extraction."""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, Mapping

from .families import get_family, has_family


def generate_parametric_geometry(
    family_name: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    source: str = "planner",
    source_path: str | None = None,
) -> Dict[str, Any]:
    if has_family(family_name):
        family = get_family(family_name)
        normalized_family = family.name
        clamped = family.clamp_parameters(parameters)
    else:
        normalized_family = str(family_name or "custom_geometry").strip().lower() or "custom_geometry"
        clamped = _coerce_numeric_parameters(parameters)
    return {
        "geometry_id": f"geo_{uuid.uuid4().hex[:10]}",
        "family": normalized_family,
        "parameters": clamped,
        "prompt2cad_prompt": build_prompt2cad_prompt(normalized_family, clamped),
        "representation": "parameterized_template",
        "source": source,
        "source_path": source_path,
    }


def extract_geometry_parameters(geometry: Mapping[str, Any]) -> Dict[str, float]:
    return dict(geometry.get("parameters") or {})


def compute_geometric_features(geometry: Mapping[str, Any]) -> Dict[str, Any]:
    family = str(geometry.get("family") or "pillar_array")
    params = extract_geometry_parameters(geometry)

    if family == "cylinder":
        r = params["radius_um"]
        h = params["height_um"]
        volume = math.pi * r * r * h
        surface = 2.0 * math.pi * r * (h + r)
        bbox_x = bbox_y = 2.0 * r
        bbox_z = h
        feature_spacing = 2.0 * r
        overhang = 0.0
    elif family == "box":
        w = params["width_um"]
        d = params["depth_um"]
        h = params["height_um"]
        volume = w * d * h
        surface = 2.0 * ((w * d) + (w * h) + (d * h))
        bbox_x, bbox_y, bbox_z = w, d, h
        feature_spacing = min(w, d)
        overhang = 0.0
    elif family == "cone":
        rb = params["base_radius_um"]
        rt = params["tip_radius_um"]
        h = params["height_um"]
        volume = (math.pi * h / 3.0) * ((rb * rb) + (rb * rt) + (rt * rt))
        slant = math.sqrt((rb - rt) ** 2 + h**2)
        surface = math.pi * (rb + rt) * slant + math.pi * rb * rb + math.pi * rt * rt
        bbox_x = bbox_y = 2.0 * rb
        bbox_z = h
        feature_spacing = max(2.0 * rt, 0.05)
        overhang = math.degrees(math.atan2(max(rb - rt, 0.0), max(h, 1e-6)))
    elif family == "microlens_array":
        r = params["lens_radius_um"]
        sag = params["sag_um"]
        pitch = params["pitch_um"]
        count_x = max(int(params["count_x"]), 1)
        count_y = max(int(params["count_y"]), 1)
        base = params["base_thickness_um"]
        lens_count = count_x * count_y
        lens_volume = lens_count * (0.5 * math.pi * r * r * max(sag, 0.05))
        base_volume = max(pitch * count_x, 2.0 * r) * max(pitch * count_y, 2.0 * r) * base
        volume = lens_volume + base_volume
        surface = (lens_count * (2.0 * math.pi * r * max(sag, 0.05))) + (2.0 * base_volume / max(base, 0.05))
        bbox_x = max(pitch * count_x, 2.0 * r)
        bbox_y = max(pitch * count_y, 2.0 * r)
        bbox_z = base + sag
        feature_spacing = max(pitch - (2.0 * r), 0.0)
        overhang = 25.0 + (20.0 * sag / max(r, 0.1))
    elif family == "pillar_array":
        r = params["radius_um"]
        h = params["height_um"]
        pitch = params["pitch_um"]
        count_x = max(int(params["count_x"]), 1)
        count_y = max(int(params["count_y"]), 1)
        pillar_count = count_x * count_y
        volume = pillar_count * math.pi * r * r * h
        surface = pillar_count * (2.0 * math.pi * r * (h + r))
        bbox_x = max(pitch * max(count_x - 1, 0) + (2.0 * r), 2.0 * r)
        bbox_y = max(pitch * max(count_y - 1, 0) + (2.0 * r), 2.0 * r)
        bbox_z = h
        feature_spacing = max(pitch - (2.0 * r), 0.0)
        overhang = 0.0
    else:
        bbox_x, bbox_y, bbox_z = _infer_bbox_from_parameters(params)
        volume = max(bbox_x * bbox_y * bbox_z * 0.35, 1e-6)
        surface = max(2.0 * ((bbox_x * bbox_y) + (bbox_x * bbox_z) + (bbox_y * bbox_z)), 1e-6)
        feature_spacing = _infer_feature_spacing(params, bbox_x, bbox_y)
        overhang = 0.0

    bbox_volume = max(bbox_x * bbox_y * bbox_z, 1e-6)
    min_feature = _min_feature_size(family, params)
    if family == "pillar_array":
        # For arrays, the overall bbox can be much wider than an individual pillar.
        # Use per-pillar slenderness so buckling risk scales with pillar height vs diameter.
        slenderness = bbox_z / max(2.0 * float(params.get("radius_um") or 1e-6), 1e-6)
    else:
        slenderness = bbox_z / max(min(bbox_x, bbox_y), 1e-6)
    voxel_size_um = 0.2
    voxel_count = volume / max(voxel_size_um**3, 1e-6)

    return {
        "family": family,
        "volume_um3": round(volume, 6),
        "surface_area_um2": round(surface, 6),
        "bbox_x_um": round(bbox_x, 6),
        "bbox_y_um": round(bbox_y, 6),
        "bbox_z_um": round(bbox_z, 6),
        "bounding_box_um": {
            "x": round(bbox_x, 6),
            "y": round(bbox_y, 6),
            "z": round(bbox_z, 6),
        },
        "fill_fraction": round(min(volume / bbox_volume, 1.0), 6),
        "aspect_ratio": round(bbox_z / max(min_feature, 1e-6), 6),
        "slenderness": round(slenderness, 6),
        "minimum_feature_size_um": round(min_feature, 6),
        "feature_spacing_um": round(feature_spacing, 6),
        "overhang_deg": round(overhang, 6),
        "voxel_count": round(voxel_count, 3),
    }


def build_prompt2cad_prompt(family_name: str, parameters: Mapping[str, Any]) -> str:
    """Create a deterministic Prompt2CAD text prompt from a family + parameters."""
    family = str(family_name or "").strip().lower()
    params = dict(parameters or {})

    def _f(name: str) -> str:
        value = float(params.get(name) or 0.0)
        return f"{value:.4f}".rstrip("0").rstrip(".")

    if family == "cylinder":
        return (
            "generate cylinder with "
            f"radius {_f('radius_um')} um and height {_f('height_um')} um"
        )
    if family == "box":
        return (
            "generate box with "
            f"width {_f('width_um')} um, depth {_f('depth_um')} um and height {_f('height_um')} um"
        )
    if family == "cone":
        return (
            "generate cone with "
            f"base_radius {_f('base_radius_um')} um, tip_radius {_f('tip_radius_um')} um "
            f"and height {_f('height_um')} um"
        )
    if family == "microlens_array":
        return (
            "generate microlens array with "
            f"lens_radius {_f('lens_radius_um')} um, sag {_f('sag_um')} um, pitch {_f('pitch_um')} um, "
            f"base_thickness {_f('base_thickness_um')} um, count_x {int(round(float(params.get('count_x') or 1.0)))} "
            f"and count_y {int(round(float(params.get('count_y') or 1.0)))}"
        )
    if family == "pillar_array":
        return (
            "generate pillar array with "
            f"radius {_f('radius_um')} um, height {_f('height_um')} um, pitch {_f('pitch_um')} um, "
            f"count_x {int(round(float(params.get('count_x') or 1.0)))} and "
            f"count_y {int(round(float(params.get('count_y') or 1.0)))}"
        )
    if params:
        ordered = ", ".join(f"{name}={float(value):.6g}" for name, value in sorted(params.items()))
        return f"generate {family or 'custom geometry'} with parameters: {ordered}"
    return f"generate {family or 'custom geometry'}"


def _min_feature_size(family: str, params: Dict[str, float]) -> float:
    if family == "box":
        return min(params["width_um"], params["depth_um"], params["height_um"])
    if family == "cone":
        return max(2.0 * params["tip_radius_um"], 0.05)
    if family == "microlens_array":
        return min(2.0 * params["lens_radius_um"], params["sag_um"], params["base_thickness_um"])
    if family == "pillar_array":
        return 2.0 * params["radius_um"]
    numeric_values = [abs(float(value)) for value in params.values() if float(value) > 0.0]
    if not numeric_values:
        return 0.5
    return max(min(numeric_values), 0.05)


def _coerce_numeric_parameters(parameters: Mapping[str, Any] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in dict(parameters or {}).items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _infer_bbox_from_parameters(params: Dict[str, float]) -> tuple[float, float, float]:
    width = _first_matching_value(params, ("width", "diameter", "base_diameter"), default=1.0)
    depth = _first_matching_value(params, ("depth", "diameter", "base_diameter"), default=width)
    height = _first_matching_value(params, ("height", "thickness", "length"), default=max(width, depth, 1.0))

    count_x = max(int(round(params.get("count_x", 1.0))), 1)
    count_y = max(int(round(params.get("count_y", 1.0))), 1)
    pitch = _first_matching_value(params, ("pitch", "period", "spacing"), default=max(width, depth))
    pitch_x = float(params.get("pitch_x", pitch))
    pitch_y = float(params.get("pitch_y", pitch))

    bbox_x = max((count_x - 1) * pitch_x + width, width)
    bbox_y = max((count_y - 1) * pitch_y + depth, depth)
    bbox_z = max(height, 0.1)
    return max(bbox_x, 0.1), max(bbox_y, 0.1), max(bbox_z, 0.1)


def _infer_feature_spacing(params: Dict[str, float], bbox_x: float, bbox_y: float) -> float:
    pitch = _first_matching_value(params, ("pitch", "period", "spacing"), default=min(bbox_x, bbox_y))
    diameter = _first_matching_value(params, ("diameter", "width", "radius"), default=0.5)
    if "radius" in " ".join(params.keys()):
        diameter = max(2.0 * diameter, 0.05)
    return max(pitch - diameter, 0.0)


def _first_matching_value(params: Dict[str, float], needles: tuple[str, ...], default: float) -> float:
    for key, value in params.items():
        lower = key.lower()
        if any(token in lower for token in needles):
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
    return float(default)

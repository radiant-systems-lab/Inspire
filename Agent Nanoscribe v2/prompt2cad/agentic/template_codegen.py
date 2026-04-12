"""Deterministic CadQuery template code generation for structured sweeps."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .families import get_family, has_family


def build_template_cadquery_code(
    family_name: str,
    parameters: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Generate deterministic CadQuery code for a geometry family.

    This path is used for controlled experiments where we want one stable
    geometric template and only parameter values to vary between candidates.
    """
    raw_family = str(family_name or "custom_geometry").strip().lower() or "custom_geometry"
    if has_family(raw_family):
        family = get_family(raw_family)
        normalized = family.clamp_parameters(parameters)
        canonical_family = family.name
    else:
        canonical_family = raw_family
        normalized = _coerce_numeric_parameters(parameters)

    code: str
    template_name: str
    if canonical_family == "cylinder":
        code, template_name = _code_cylinder(normalized), "cylinder"
    elif canonical_family == "box":
        code, template_name = _code_box(normalized), "box"
    elif canonical_family == "cone":
        code, template_name = _code_cone(normalized), "cone"
    elif canonical_family == "pillar_array":
        code, template_name = _code_pillar_array(normalized), "pillar_array"
    elif canonical_family == "microlens_array":
        code, template_name = _code_microlens_array(normalized), "microlens_array"
    else:
        code, template_name, normalized = _code_generic(canonical_family, normalized)

    return {
        "family": canonical_family,
        "template_name": template_name,
        "parameters": normalized,
        "code": code,
    }


def _code_cylinder(params: Mapping[str, Any]) -> str:
    radius = _fmt(_safe_float(params.get("radius_um"), 0.8, minimum=0.01))
    height = _fmt(_safe_float(params.get("height_um"), 1.0, minimum=0.01))
    return "\n".join(
        [
            "import cadquery as cq",
            "",
            f"radius_um = {radius}",
            f"height_um = {height}",
            "",
            "result = cq.Workplane('XY').circle(radius_um).extrude(height_um)",
            "",
        ]
    )


def _code_box(params: Mapping[str, Any]) -> str:
    width = _fmt(_safe_float(params.get("width_um"), 2.0, minimum=0.01))
    depth = _fmt(_safe_float(params.get("depth_um"), 2.0, minimum=0.01))
    height = _fmt(_safe_float(params.get("height_um"), 1.0, minimum=0.01))
    return "\n".join(
        [
            "import cadquery as cq",
            "",
            f"width_um = {width}",
            f"depth_um = {depth}",
            f"height_um = {height}",
            "",
            "result = cq.Workplane('XY').box(width_um, depth_um, height_um, centered=(True, True, False))",
            "",
        ]
    )


def _code_cone(params: Mapping[str, Any]) -> str:
    base_radius = _fmt(_safe_float(params.get("base_radius_um"), 1.0, minimum=0.01))
    tip_radius = _fmt(_safe_float(params.get("tip_radius_um"), 0.1, minimum=0.0))
    height = _fmt(_safe_float(params.get("height_um"), 1.0, minimum=0.01))
    return "\n".join(
        [
            "import cadquery as cq",
            "",
            f"base_radius_um = {base_radius}",
            f"tip_radius_um = {tip_radius}",
            f"height_um = {height}",
            "",
            # Use loft for broad CadQuery compatibility (avoids Workplane.cone API drift).
            "result = (",
            "    cq.Workplane('XY')",
            "    .circle(base_radius_um)",
            "    .workplane(offset=height_um)",
            "    .circle(max(tip_radius_um, 0.001))",
            "    .loft(combine=True)",
            ")",
            "",
        ]
    )


def _code_pillar_array(params: Mapping[str, Any]) -> str:
    radius = _fmt(_safe_float(params.get("radius_um"), 0.5, minimum=0.01))
    height = _fmt(_safe_float(params.get("height_um"), 1.0, minimum=0.01))
    pitch = _fmt(_safe_float(params.get("pitch_um"), 1.5, minimum=0.01))
    count_x = _safe_int(params.get("count_x"), 4, minimum=1)
    count_y = _safe_int(params.get("count_y"), 4, minimum=1)
    return "\n".join(
        [
            "import cadquery as cq",
            "",
            f"radius_um = {radius}",
            f"height_um = {height}",
            f"pitch_um = {pitch}",
            f"count_x = {count_x}",
            f"count_y = {count_y}",
            "",
            "points = [",
            "    (",
            "        (ix - (count_x - 1) / 2.0) * pitch_um,",
            "        (iy - (count_y - 1) / 2.0) * pitch_um,",
            "    )",
            "    for ix in range(count_x)",
            "    for iy in range(count_y)",
            "]",
            "result = cq.Workplane('XY').pushPoints(points).circle(radius_um).extrude(height_um)",
            "",
        ]
    )


def _code_microlens_array(params: Mapping[str, Any]) -> str:
    lens_radius = _fmt(_safe_float(params.get("lens_radius_um"), 2.0, minimum=0.05))
    sag = _fmt(_safe_float(params.get("sag_um"), 0.6, minimum=0.02))
    pitch = _fmt(_safe_float(params.get("pitch_um"), 5.0, minimum=0.05))
    base = _fmt(_safe_float(params.get("base_thickness_um"), 0.5, minimum=0.02))
    count_x = _safe_int(params.get("count_x"), 3, minimum=1)
    count_y = _safe_int(params.get("count_y"), 3, minimum=1)
    return "\n".join(
        [
            "import cadquery as cq",
            "",
            f"lens_radius_um = {lens_radius}",
            f"sag_um = {sag}",
            f"pitch_um = {pitch}",
            f"base_thickness_um = {base}",
            f"count_x = {count_x}",
            f"count_y = {count_y}",
            "",
            "span_x = max((count_x - 1) * pitch_um + 2.0 * lens_radius_um, 2.0 * lens_radius_um)",
            "span_y = max((count_y - 1) * pitch_um + 2.0 * lens_radius_um, 2.0 * lens_radius_um)",
            "result = cq.Workplane('XY').box(span_x, span_y, base_thickness_um, centered=(True, True, False))",
            "for ix in range(count_x):",
            "    for iy in range(count_y):",
            "        x = (ix - (count_x - 1) / 2.0) * pitch_um",
            "        y = (iy - (count_y - 1) / 2.0) * pitch_um",
            "        lens = (",
            "            cq.Workplane('XY')",
            "            .workplane(offset=base_thickness_um)",
            "            .center(x, y)",
            "            .circle(lens_radius_um)",
            "            .workplane(offset=sag_um)",
            "            .circle(max(lens_radius_um * 0.08, 0.01))",
            "            .loft(combine=True)",
            "        )",
            "        result = result.union(lens)",
            "",
        ]
    )


def _code_generic(family_name: str, params: Mapping[str, Any]) -> Tuple[str, str, Dict[str, float]]:
    normalized = dict(params)
    if not normalized:
        normalized = {"height_um": 1.0, "radius_um": 0.5}

    is_array_hint = ("array" in family_name) or family_name.endswith("s")

    height = _pick(
        normalized,
        ("height_um", "height", "length_um", "length", "thickness_um", "thickness"),
        default=1.0,
    )
    pitch = _pick(
        normalized,
        ("pitch_um", "period_um", "pitch", "period", "spacing_um", "spacing"),
        default=0.5,
    )

    base_diameter = _pick(
        normalized,
        ("base_diameter_um", "diameter_um", "diameter", "base_diameter"),
        default=None,
    )
    radius = _pick(normalized, ("radius_um", "radius", "r"), default=None)
    if base_diameter is not None:
        base_radius = max(base_diameter / 2.0, 0.01)
    elif radius is not None:
        base_radius = max(radius, 0.01)
    else:
        base_radius = 0.5

    tip_diameter = _pick(
        normalized,
        ("tip_diameter_um", "tip_diameter"),
        default=None,
    )
    tip_radius = _pick(
        normalized,
        ("tip_radius_um", "tip_radius"),
        default=None,
    )
    if tip_diameter is not None:
        tip_radius = max(tip_diameter / 2.0, 0.0)
    elif tip_radius is None:
        tip_radius = max(base_radius * 0.08, 0.0)

    count_x = _safe_int(
        _pick(normalized, ("count_x", "nx", "columns"), default=3 if is_array_hint else 1),
        3 if is_array_hint else 1,
        minimum=1,
    )
    count_y = _safe_int(
        _pick(normalized, ("count_y", "ny", "rows"), default=count_x),
        count_x,
        minimum=1,
    )

    normalized["height_um"] = float(max(height, 0.01))
    normalized["pitch_um"] = float(max(pitch, 0.01))
    normalized["base_radius_um"] = float(max(base_radius, 0.01))
    normalized["tip_radius_um"] = float(max(tip_radius, 0.0))
    normalized["count_x"] = float(count_x)
    normalized["count_y"] = float(count_y)

    # Generic fallback: cone array template works well for nanocones and
    # similar tapered microfeatures.
    code = "\n".join(
        [
            "import cadquery as cq",
            "",
            f"height_um = {_fmt(normalized['height_um'])}",
            f"pitch_um = {_fmt(normalized['pitch_um'])}",
            f"base_radius_um = {_fmt(normalized['base_radius_um'])}",
            f"tip_radius_um = {_fmt(normalized['tip_radius_um'])}",
            f"count_x = {count_x}",
            f"count_y = {count_y}",
            "",
            "result = None",
            "for ix in range(count_x):",
            "    for iy in range(count_y):",
            "        x = (ix - (count_x - 1) / 2.0) * pitch_um",
            "        y = (iy - (count_y - 1) / 2.0) * pitch_um",
            "        cone = (",
            "            cq.Workplane('XY')",
            "            .center(x, y)",
            "            .circle(base_radius_um)",
            "            .workplane(offset=height_um)",
            "            .circle(max(tip_radius_um, 0.001))",
            "            .loft(combine=True)",
            "        )",
            "        result = cone if result is None else result.union(cone)",
            "",
            "if result is None:",
            "    result = (",
            "        cq.Workplane('XY')",
            "        .circle(base_radius_um)",
            "        .workplane(offset=height_um)",
            "        .circle(max(tip_radius_um, 0.001))",
            "        .loft(combine=True)",
            "    )",
            "",
        ]
    )
    return code, "generic_cone_array", normalized


def _coerce_numeric_parameters(parameters: Mapping[str, Any] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in dict(parameters or {}).items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _pick(params: Mapping[str, Any], keys: tuple[str, ...], default: float | None) -> float | None:
    lowered = {str(name).lower(): value for name, value in params.items()}
    for key in keys:
        if key in lowered:
            try:
                return float(lowered[key])
            except (TypeError, ValueError):
                continue
    return default


def _safe_float(value: Any, default: float, *, minimum: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    return max(out, minimum)


def _safe_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        out = int(default)
    return max(out, minimum)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"

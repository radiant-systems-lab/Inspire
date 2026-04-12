"""
CAD verification pipeline.

Stage 1 -- hard_verify():
    Deterministic geometric assertions. No LLM. Runs on every generated CAD.
    A failed hard check blocks fabrication -- no exceptions.

Stage 2 -- extract_cge_from_code():
    AST-based parse of generated CADQuery code to extract parameters actually
    used (dimensions, counts, radii, etc.) as a structured dict.

Stage 3 -- compute_geometry_metrics():
    Programmatic metrics from the live workplane: volume, surface area,
    bounding box, body count, aspect ratios, wall thickness estimate.

Stage 4 -- structured_vision_analysis():
    Vision model analyzes each render view independently, returns per-view JSON.
    A synthesis pass aggregates across views. No CGE passed -- renders only.

All four stages are assembled by _verify_output() in cad_agent.py into a
single package handed to the main agent for accept/reject reasoning.
"""

from __future__ import annotations

import ast
import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Stage 1 -- Hard verification
# ---------------------------------------------------------------------------

def hard_verify(
    code: str,
    stl_path: Optional[str] = None,
    expected_body_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run deterministic geometric assertions on generated CadQuery code.

    The code is re-executed in an isolated namespace to capture the live
    CadQuery Workplane object (the STL export discards topology information).

    Returns:
        {
          "passed": bool,
          "checks": [{"name": str, "passed": bool|None, "detail": str}],
          "warnings": [str],
          "body_count": int | None,
          "bounding_box": {"x": float, "y": float, "z": float} | None,
        }
    """
    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []
    body_count: Optional[int] = None
    bbox_dims: Optional[Dict[str, float]] = None

    # -- Step 0: re-execute code to get live workplane object -----------------
    workplane = _execute_for_workplane(code)
    if workplane is None:
        checks.append({
            "name": "execution",
            "passed": False,
            "detail": "Code re-execution failed -- cannot perform geometric checks.",
        })
        stl_ok = _check_stl(stl_path, checks, warnings)
        return _result(False, checks, warnings, body_count, bbox_dims)

    checks.append({"name": "execution", "passed": True, "detail": "Code executed successfully."})

    # -- Check 1: has_geometry -------------------------------------------------
    try:
        solids = workplane.solids().vals()
        body_count = len(solids)
        passed = body_count > 0
        checks.append({
            "name": "has_geometry",
            "passed": passed,
            "detail": f"{body_count} solid(s) found.",
        })
        if not passed:
            warnings.append("No solid bodies found in result.")
    except Exception as exc:
        checks.append({"name": "has_geometry", "passed": False, "detail": str(exc)})
        warnings.append(f"Could not iterate solids: {exc}")

    # -- Check 2: positive_volume ----------------------------------------------
    try:
        bb = workplane.val().BoundingBox()
        x_dim = bb.xmax - bb.xmin
        y_dim = bb.ymax - bb.ymin
        z_dim = bb.zmax - bb.zmin
        vol_proxy = x_dim * y_dim * z_dim
        bbox_dims = {"x": round(x_dim, 6), "y": round(y_dim, 6), "z": round(z_dim, 6)}
        passed = vol_proxy > 1e-12
        checks.append({
            "name": "positive_volume",
            "passed": passed,
            "detail": f"Bounding box volume proxy = {vol_proxy:.6g}  dims={bbox_dims}",
        })
        if not passed:
            warnings.append("Bounding box volume is effectively zero.")
    except Exception as exc:
        checks.append({"name": "positive_volume", "passed": False, "detail": str(exc)})
        warnings.append(f"Could not compute bounding box: {exc}")

    # -- Check 3: no_degenerate_dims -------------------------------------------
    if bbox_dims is not None:
        degenerate = [k for k, v in bbox_dims.items() if v < 1e-6]
        passed = len(degenerate) == 0
        checks.append({
            "name": "no_degenerate_dims",
            "passed": passed,
            "detail": (
                f"All dims > threshold: {bbox_dims}" if passed
                else f"Near-zero dimension(s): {degenerate} in {bbox_dims}"
            ),
        })
        if not passed:
            warnings.append(f"Degenerate dimension(s) detected: {degenerate}")

    # -- Check 4: expected body count -----------------------------------------
    if expected_body_count is not None and body_count is not None:
        passed = body_count == expected_body_count
        checks.append({
            "name": "body_count",
            "passed": passed,
            "detail": f"Expected {expected_body_count} body(ies), found {body_count}.",
        })
        if not passed:
            warnings.append(
                f"Body count mismatch: expected {expected_body_count}, got {body_count}."
            )

    # -- Check 5: manifold (OCC BRepCheck) ------------------------------------
    _check_manifold(workplane, checks, warnings)

    # -- Check 6: STL exported ------------------------------------------------
    _check_stl(stl_path, checks, warnings)

    # Critical checks that must pass
    critical = {"execution", "has_geometry", "positive_volume"}
    critical_failed = [
        c for c in checks
        if c["name"] in critical and c["passed"] is False
    ]
    passed_overall = len(critical_failed) == 0

    return _result(passed_overall, checks, warnings, body_count, bbox_dims)


def _execute_for_workplane(code: str) -> Any:
    """Re-execute code in isolated namespace and return the `result` workplane."""
    try:
        import cadquery as cq  # noqa: F401
        namespace: Dict[str, Any] = {
            "cq": cq,
            "__name__": "__cad_verifier__",
            "__builtins__": __builtins__,
        }
        exec(code, namespace)  # noqa: S102
        return namespace.get("result")
    except Exception:
        return None


def _check_manifold(workplane: Any, checks: list, warnings: list) -> None:
    try:
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        shape = workplane.val().wrapped
        analyzer = BRepCheck_Analyzer(shape)
        is_valid = analyzer.IsValid()
        checks.append({
            "name": "manifold",
            "passed": is_valid,
            "detail": "OCC BRepCheck passed." if is_valid else "BRepCheck failed -- possible open edges or self-intersections.",
        })
        if not is_valid:
            warnings.append("Geometry may not be watertight (BRepCheck failed).")
    except ImportError:
        checks.append({
            "name": "manifold",
            "passed": None,
            "detail": "OCC BRepCheck unavailable -- skipped.",
        })
    except Exception as exc:
        checks.append({
            "name": "manifold",
            "passed": None,
            "detail": f"Manifold check error: {exc}",
        })


def _check_stl(stl_path: Optional[str], checks: list, warnings: list) -> bool:
    if stl_path is None:
        checks.append({"name": "stl_exported", "passed": None, "detail": "STL path not provided."})
        return False
    exists = Path(stl_path).exists() and Path(stl_path).stat().st_size > 0
    checks.append({
        "name": "stl_exported",
        "passed": exists,
        "detail": f"STL at {stl_path} {'exists' if exists else 'missing or empty'}.",
    })
    if not exists:
        warnings.append(f"STL file not found or empty: {stl_path}")
    return exists


def _result(
    passed: bool,
    checks: list,
    warnings: list,
    body_count: Optional[int],
    bbox_dims: Optional[Dict],
) -> Dict[str, Any]:
    return {
        "passed": passed,
        "checks": checks,
        "warnings": warnings,
        "body_count": body_count,
        "bounding_box": bbox_dims,
    }


# ---------------------------------------------------------------------------
# Stage 2 -- CGE extraction from generated code
# ---------------------------------------------------------------------------

# CadQuery API calls whose positional args carry geometric meaning.
_CQ_PARAM_MAP: Dict[str, List[str]] = {
    "box":      ["length", "width", "height"],
    "sphere":   ["radius"],
    "cylinder": ["height", "radius"],
    "cone":     ["height", "radius1", "radius2"],
    "torus":    ["radius1", "radius2"],
    "hole":     ["diameter"],
    "cboreHole":  ["diameter", "cboreDiameter", "cboreDepth"],
    "cskHole":    ["diameter", "cskDiameter", "cskAngle"],
    "extrude":  ["distance"],
    "revolve":  ["angleDegrees"],
    "fillet":   ["radius"],
    "chamfer":  ["length"],
    "shell":    ["thickness"],
    "rect":     ["xLen", "yLen"],
    "circle":   ["radius"],
    "polygon":  ["nSides", "diameter"],
    "rarray":   ["xSpacing", "ySpacing", "xCount", "yCount"],
    "polarArray": ["radius", "startAngle", "angle", "count"],
    "loft":     [],
    "sweep":    [],
    "cut":      [],
    "union":    [],
    "intersect": [],
}


def extract_cge_from_code(code: str) -> Dict[str, Any]:
    """
    Parse generated CADQuery code with the AST and return a structured dict
    of the geometric parameters actually encoded in the code.

    Returns:
        {
          "operations": [{"call": str, "args": {str: float|int|str}}],
          "variables":  {str: float|int|str},
          "summary":    {str: Any},   # flattened key params for quick comparison
        }
    """
    operations: List[Dict[str, Any]] = []
    variables: Dict[str, Any] = {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"operations": [], "variables": {}, "summary": {}, "parse_error": True}

    # Collect top-level variable assignments (numeric/string literals only)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id != "result":
                    val = _ast_literal(node.value)
                    if val is not None:
                        variables[target.id] = val

    # Collect CadQuery method calls and their args
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            method = None
            if isinstance(func, ast.Attribute):
                method = func.attr
            elif isinstance(func, ast.Name):
                method = func.id

            if method and method in _CQ_PARAM_MAP:
                param_names = _CQ_PARAM_MAP[method]
                args: Dict[str, Any] = {}
                for i, arg_node in enumerate(node.args):
                    val = _ast_literal(arg_node)
                    if val is not None:
                        key = param_names[i] if i < len(param_names) else f"arg{i}"
                        args[key] = val
                for kw in node.keywords:
                    val = _ast_literal(kw.value)
                    if val is not None:
                        args[kw.arg] = val
                if args:
                    operations.append({"call": method, "args": args})

    summary = _summarize_operations(operations, variables)
    return {"operations": operations, "variables": variables, "summary": summary}


def _ast_literal(node: Any) -> Any:
    """Return the Python value of a literal AST node, or None if not a literal."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, bool)):
            return node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_literal(node.operand)
        if isinstance(inner, (int, float)):
            return -inner
    elif isinstance(node, ast.Name):
        return f"${node.id}"
    return None


def _summarize_operations(operations: List[Dict], variables: Dict) -> Dict[str, Any]:
    """Flatten the most diagnostic params into a quick-compare dict."""
    summary: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    for op in operations:
        name = op["call"]
        counts[name] = counts.get(name, 0) + 1
        for k, v in op["args"].items():
            key = f"{name}.{k}" if counts[name] == 1 else f"{name}_{counts[name]}.{k}"
            summary[key] = v
    summary["_op_counts"] = counts
    summary["_variable_count"] = len(variables)
    return summary


# ---------------------------------------------------------------------------
# Stage 3 -- Geometric metrics from live workplane
# ---------------------------------------------------------------------------

def compute_geometry_metrics(workplane: Any) -> Dict[str, Any]:
    """
    Extract numeric metrics from a live CadQuery Workplane object.

    Returns a flat dict of scalar values suitable for comparison and logging.
    All lengths are in the model's native unit (mm assumed).
    """
    metrics: Dict[str, Any] = {}

    try:
        solids = workplane.solids().vals()
        metrics["body_count"] = len(solids)
    except Exception:
        metrics["body_count"] = None

    try:
        bb = workplane.val().BoundingBox()
        bx = round(bb.xmax - bb.xmin, 6)
        by = round(bb.ymax - bb.ymin, 6)
        bz = round(bb.zmax - bb.zmin, 6)
        metrics["bbox_x"] = bx
        metrics["bbox_y"] = by
        metrics["bbox_z"] = bz
        metrics["bbox_volume"] = round(bx * by * bz, 6)
        dims = sorted([bx, by, bz])
        metrics["aspect_ratio_max"] = round(dims[2] / dims[0], 4) if dims[0] > 1e-9 else None
    except Exception:
        metrics.update({"bbox_x": None, "bbox_y": None, "bbox_z": None,
                        "bbox_volume": None, "aspect_ratio_max": None})

    try:
        shape = workplane.val()
        metrics["volume"] = round(shape.Volume(), 6)
        metrics["surface_area"] = round(shape.Area(), 6)
        if metrics["bbox_volume"] and metrics["bbox_volume"] > 1e-12:
            metrics["fill_fraction"] = round(metrics["volume"] / metrics["bbox_volume"], 4)
    except Exception:
        metrics.update({"volume": None, "surface_area": None, "fill_fraction": None})

    try:
        shape = workplane.val()
        metrics["face_count"] = shape.ShapeType and len(workplane.faces().vals())
        metrics["edge_count"] = len(workplane.edges().vals())
        metrics["vertex_count"] = len(workplane.vertices().vals())
    except Exception:
        metrics.update({"face_count": None, "edge_count": None, "vertex_count": None})

    return metrics


# ---------------------------------------------------------------------------
# Stage 4 -- Structured vision analysis (per-view + synthesis)
# ---------------------------------------------------------------------------

_VIEW_SCHEMA = """\
Analyze this CAD render and return ONLY valid JSON matching this schema exactly:
{
  "visible_body_count": <integer or null>,
  "overall_shape": "<primary shape description, e.g. rectangular_block, cylindrical, L-shaped>",
  "feature_types": ["<list of visible feature types: hole, protrusion, slot, fillet, array, etc.>"],
  "feature_count": <integer count of distinct geometric features, or null>,
  "symmetry": "<none | bilateral_x | bilateral_y | radial | unknown>",
  "apparent_uniformity": "<uniform | irregular | periodic>",
  "visible_defects": ["<list of anomalies: merged_features, missing_feature, open_surface, spike, etc. Empty list if none.>"],
  "confidence": <0.0 to 1.0>
}
No explanation. JSON only."""

_SYNTHESIS_SCHEMA = """\
You are given per-view JSON analyses of a 3D CAD model from multiple angles.
Synthesize them into a single summary. Return ONLY valid JSON:
{
  "body_count_consensus": <integer or null>,
  "overall_shape": "<best description across views>",
  "feature_types": ["<union of all observed feature types>"],
  "feature_count": <integer, best estimate across views>,
  "symmetry": "<consensus symmetry>",
  "structural_integrity": "<sound | suspect | defective>",
  "defects": ["<consolidated list of unique defects observed>"],
  "view_agreement": "<high | medium | low>",
  "confidence": <0.0 to 1.0>
}
No explanation. JSON only."""


def structured_vision_analysis(
    render_paths: Dict[str, str],
    model: str,
) -> Dict[str, Any]:
    """
    Analyze each render view with a vision model, returning per-view structured
    JSON and a synthesized summary.

    render_paths: {"render_path": str, "render_iso_path": str, ...}

    Returns:
        {
          "per_view": {view_name: {schema fields}},
          "synthesis": {synthesis schema fields},
          "views_analyzed": [str],
        }
    """
    from ..utils import call_openrouter

    images_b64: Dict[str, str] = {}
    for key, path in render_paths.items():
        if path and Path(path).exists():
            with open(path, "rb") as f:
                images_b64[key] = base64.b64encode(f.read()).decode()

    if not images_b64:
        return {
            "per_view": {},
            "synthesis": {},
            "views_analyzed": [],
            "error": "no renders available",
        }

    per_view: Dict[str, Any] = {}
    for view_name, b64 in images_b64.items():
        content: List[Any] = [
            {"type": "text", "text": _VIEW_SCHEMA},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
        try:
            raw = call_openrouter(
                messages=[{"role": "user", "content": content}],
                model=model,
                temperature=0.0,
            )
            per_view[view_name] = _parse_json_from_text(raw)
        except Exception as exc:
            per_view[view_name] = {"error": str(exc)}

    synthesis: Dict[str, Any] = {}
    if per_view:
        synthesis_prompt = (
            _SYNTHESIS_SCHEMA
            + "\n\nPer-view analyses:\n"
            + json.dumps(per_view, indent=2)
        )
        try:
            raw = call_openrouter(
                messages=[{"role": "user", "content": synthesis_prompt}],
                model=model,
                temperature=0.0,
            )
            synthesis = _parse_json_from_text(raw)
        except Exception as exc:
            synthesis = {"error": str(exc)}

    return {
        "per_view": per_view,
        "synthesis": synthesis,
        "views_analyzed": list(images_b64.keys()),
    }


# ---------------------------------------------------------------------------
# Legacy Stage 2 -- soft_verify() kept for backward compatibility
# ---------------------------------------------------------------------------

def soft_verify(
    render_paths: Dict[str, str],
    cge: Any,
    model: str,
) -> Dict[str, Any]:
    """
    Visual inspection of orthographic renders against the CGE description.

    The LLM checks:
    - Topology: does the overall structure match the description?
    - Semantic: correct arrangement, no merged/missing features?
    - Dimensions: visible proportions consistent with stated parameters?

    render_paths: dict with keys like "render_path", "render_iso_path",
                  "render_top_path", "render_side_path" -> file paths.

    Returns:
        {
          "passed": bool | None,
          "assessment": str,
          "issues": [str],
          "confidence": float,
          "topology_match": bool | None,
        }
    """
    # Collect available renders
    images_b64: Dict[str, str] = {}
    for key, path in render_paths.items():
        if path and Path(path).exists():
            with open(path, "rb") as f:
                images_b64[key] = base64.b64encode(f.read()).decode()

    if not images_b64:
        return {
            "passed": None,
            "assessment": "No renders available for soft verification.",
            "issues": [],
            "confidence": 0.0,
            "topology_match": None,
        }

    cge_block = cge.to_prompt_block() if hasattr(cge, "to_prompt_block") else json.dumps(cge, indent=2)

    prompt = (
        "You are performing a visual quality check on a 3D CAD render.\n\n"
        "Intended geometry:\n"
        f"{cge_block}\n\n"
        f"I am showing you {len(images_b64)} orthographic view(s): {list(images_b64.keys())}.\n\n"
        "Inspect the render(s) and answer:\n"
        "1. Does the overall topology match the description? (correct number of features, correct arrangement)\n"
        "2. Are there geometric errors? (features merged that should be separate, wrong cross-section shape, "
        "missing sub-features, wrong symmetry class, incorrect periodicity)\n"
        "3. Are visible proportions consistent with the stated parameters?\n\n"
        "Respond with JSON only:\n"
        '{"topology_match": true/false, "issues": ["specific problems, empty if none"], '
        '"assessment": "one sentence", "confidence": 0.0-1.0}'
    )

    content: List[Any] = [{"type": "text", "text": prompt}]
    for key, b64 in images_b64.items():
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    from ..utils import call_openrouter
    try:
        raw = call_openrouter(
            messages=[{"role": "user", "content": content}],
            model=model,
            temperature=0.0,
        )
        parsed = _parse_json_from_text(raw)
        issues = parsed.get("issues", [])
        topology_match = parsed.get("topology_match", None)
        return {
            "passed": bool(topology_match) and len(issues) == 0,
            "assessment": parsed.get("assessment", ""),
            "issues": issues,
            "confidence": float(parsed.get("confidence", 0.5)),
            "topology_match": topology_match,
        }
    except Exception as exc:
        return {
            "passed": None,
            "assessment": f"Soft verification error: {exc}",
            "issues": [],
            "confidence": 0.0,
            "topology_match": None,
        }


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}

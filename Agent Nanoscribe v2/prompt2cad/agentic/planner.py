"""Experiment planner for the autonomous microfabrication loop."""

from __future__ import annotations

import itertools
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping

from .canonical_geometry import normalize_agent_input
from .families import DEFAULT_FAMILY_NAME, FAMILY_REGISTRY, get_family, has_family, infer_family_from_text


class ExperimentPlanner:
    def plan(self, raw_request: Any, dataset_records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        request = normalize_agent_input(raw_request)
        records = [dict(record) for record in dataset_records]
        family_name = self._select_family(request, records)
        known_family = has_family(family_name)
        family = get_family(family_name)
        canonical_family_name = family.name if known_family else family_name
        family_records = [record for record in records if str(record.get("family")) == canonical_family_name]

        structured = request.get("canonical_geometry") or {}
        structured_geometry = structured.get("geometry") if isinstance(structured, dict) else {}
        structured_constraints = (
            structured_geometry.get("constraints")
            if isinstance(structured_geometry, Mapping)
            and isinstance(structured_geometry.get("constraints"), Mapping)
            else {}
        )
        fixed_parameters = _constraint_parameter_names(structured_constraints, "fixed_parameters")
        vary_parameters = _constraint_parameter_names(structured_constraints, "vary_parameters")

        if structured_geometry and structured_geometry.get("parameters"):
            if known_family:
                base_parameters = family.clamp_parameters(structured_geometry["parameters"])
            else:
                base_parameters = _coerce_numeric_parameters(structured_geometry["parameters"])
        else:
            best_record = _best_family_record(family_records)
            if known_family:
                base_parameters = family.clamp_parameters(
                    (best_record or {}).get("geometry_parameters") or family.default_parameters()
                )
            else:
                base_parameters = _coerce_numeric_parameters((best_record or {}).get("geometry_parameters") or {})

        goal_constraints = _infer_goal_constraints(
            goal=str(request.get("goal") or ""),
            family_name=canonical_family_name,
            known_family=known_family,
            base_parameters=base_parameters,
        )
        if goal_constraints["parameter_overrides"]:
            merged = dict(base_parameters)
            merged.update(goal_constraints["parameter_overrides"])
            if known_family:
                base_parameters = family.clamp_parameters(merged)
            else:
                base_parameters = _coerce_numeric_parameters(merged)
        fixed_parameters = _merge_unique_names(fixed_parameters, list(goal_constraints["fixed_parameters"]))
        vary_parameters = _merge_unique_names(vary_parameters, list(goal_constraints["vary_parameters"]))

        if structured_geometry and (structured_geometry.get("sweeps") or structured_geometry.get("parameters")):
            strategy = "execute_structured_sweep" if structured_geometry.get("sweeps") else "execute_structured_point"
            sweep_specs = list(structured_geometry.get("sweeps") or [])
            explore_mode = False
        elif goal_constraints["sweep_specs"]:
            strategy = "execute_structured_sweep"
            sweep_specs = list(goal_constraints["sweep_specs"])
            explore_mode = False
        elif not family_records:
            strategy = "explore_new_geometry" if request["input_type"] == "unspecified_goal" else "bootstrap_parameter_sweep"
            sweep_specs = _initial_sweeps(family.name, base_parameters, sample_count=5)
            explore_mode = True
        else:
            uncertain_record = _most_uncertain_record(family_records)
            boundary_record = _boundary_record(family_records)
            if uncertain_record and float((uncertain_record.get("uncertainty") or {}).get("uncertainty_score") or 0.0) >= 0.45:
                strategy = "increase_sample_density_in_uncertain_region"
                base_parameters = (
                    family.clamp_parameters(uncertain_record.get("geometry_parameters") or base_parameters)
                    if known_family
                    else _coerce_numeric_parameters(uncertain_record.get("geometry_parameters") or base_parameters)
                )
                sweep_specs = _local_refinement_sweeps(family.name, base_parameters, sample_count=4)
                explore_mode = True
            elif boundary_record is not None:
                strategy = "refine_near_success_boundary"
                base_parameters = (
                    family.clamp_parameters(boundary_record.get("geometry_parameters") or base_parameters)
                    if known_family
                    else _coerce_numeric_parameters(boundary_record.get("geometry_parameters") or base_parameters)
                )
                sweep_specs = _local_refinement_sweeps(family.name, base_parameters, sample_count=4)
                explore_mode = False
            else:
                strategy = "refine_existing_geometry"
                sweep_specs = _local_refinement_sweeps(family.name, base_parameters, sample_count=4)
                explore_mode = False

        sweep_specs = _apply_parameter_constraints_to_sweeps(
            sweep_specs=sweep_specs,
            fixed_parameters=fixed_parameters,
            vary_parameters=vary_parameters,
        )

        if known_family:
            candidates = design_parameter_sweep(
                family_name=family.name,
                base_parameters=base_parameters,
                sweep_specs=sweep_specs,
                max_candidates=6,
                fixed_parameters=fixed_parameters,
            )
        else:
            candidates = design_parameter_sweep_generic(
                family_name=canonical_family_name or "custom_geometry",
                base_parameters=base_parameters,
                sweep_specs=sweep_specs,
                max_candidates=6,
                fixed_parameters=fixed_parameters,
            )

        return {
            "plan_id": f"plan_{uuid.uuid4().hex[:10]}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": request,
            "workflow": _workflow_for_request(request["input_type"]),
            "strategy": strategy,
            "geometry_decision": "explore_new_geometry" if explore_mode else "refine_existing_geometry",
            "geometry_action": {
                "mode": "parameterized_template" if request["input_type"] != "step" else "reference_geometry_bootstrap",
                "family": canonical_family_name,
                "source_path": request.get("source_path"),
                "hard_constraints_applied": bool(structured_geometry and structured_geometry.get("hard_constraints_applied")),
                "fixed_parameters": fixed_parameters,
                "vary_parameters": vary_parameters,
            },
            "parameters_of_interest": [sweep["parameter"] for sweep in sweep_specs],
            "parameter_sweeps": sweep_specs,
            "candidate_experiments": candidates,
            "sample_count": len(candidates),
            "success_metrics": [
                {"name": "actual_success", "target": True},
                {"name": "risk_score", "max": 0.35},
                {"name": "uncertainty_score", "max": 0.60},
            ],
            "decision_policy": {
                "mode": "exploration" if explore_mode else "refinement",
                "explore_fraction": 0.70 if explore_mode else 0.30,
                "exploit_fraction": 0.30 if explore_mode else 0.70,
            },
            "rationale": _build_rationale(
                request,
                canonical_family_name,
                strategy,
                family_records,
                fixed_parameters=fixed_parameters,
                vary_parameters=vary_parameters,
            ),
        }

    def _select_family(self, request: Mapping[str, Any], records: List[Mapping[str, Any]]) -> str:
        canonical = request.get("canonical_geometry")
        if isinstance(canonical, Mapping):
            geometry = canonical.get("geometry")
            if isinstance(geometry, Mapping) and geometry.get("family"):
                return str(geometry["family"])
        family_hint = str(request.get("geometry_family_hint") or "").strip()
        if family_hint:
            return family_hint
        if request.get("goal"):
            return infer_family_from_text(str(request["goal"]))
        if not records:
            return DEFAULT_FAMILY_NAME
        family_counts = {
            family_name: sum(1 for record in records if str(record.get("family")) == family_name)
            for family_name in FAMILY_REGISTRY
        }
        return min(family_counts, key=family_counts.get)


def plan_next_experiments(raw_request: Any, dataset_records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    return ExperimentPlanner().plan(raw_request, dataset_records)


def design_parameter_sweep(
    *,
    family_name: str,
    base_parameters: Mapping[str, Any],
    sweep_specs: Iterable[Mapping[str, Any]],
    max_candidates: int = 6,
    fixed_parameters: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    family = get_family(family_name)
    base = family.clamp_parameters(base_parameters)
    fixed = {
        name
        for name in (fixed_parameters or [])
        if name in family.parameters
    }
    ordered_values: List[tuple[str, List[float]]] = []

    for sweep in sweep_specs:
        parameter = str(sweep.get("parameter") or "").strip()
        if parameter not in family.parameters or parameter in fixed:
            continue
        values = sweep.get("values")
        if not isinstance(values, list) or not values:
            continue
        ordered_values.append(
            (
                parameter,
                [family.parameters[parameter].clamp(value) for value in values],
            )
        )

    if not ordered_values:
        return [{"family": family.name, "parameters": base, "tags": ["baseline"]}]

    candidates: List[Dict[str, Any]] = []
    for combo in itertools.product(*[values for _, values in ordered_values]):
        parameters = dict(base)
        tags = []
        for (parameter, _), value in zip(ordered_values, combo):
            parameters[parameter] = value
            tags.append(f"{parameter}={value}")
        for parameter in fixed:
            parameters[parameter] = base[parameter]
        clamped = family.clamp_parameters(parameters)
        if clamped not in [candidate["parameters"] for candidate in candidates]:
            candidates.append({"family": family.name, "parameters": clamped, "tags": tags})
        if len(candidates) >= max_candidates:
            break
    return candidates


def design_parameter_sweep_generic(
    *,
    family_name: str,
    base_parameters: Mapping[str, Any],
    sweep_specs: Iterable[Mapping[str, Any]],
    max_candidates: int = 6,
    fixed_parameters: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    base = _coerce_numeric_parameters(base_parameters)
    fixed = set(fixed_parameters or [])
    ordered_values: List[tuple[str, List[float]]] = []

    for sweep in sweep_specs:
        parameter = str(sweep.get("parameter") or "").strip()
        if not parameter or parameter in fixed:
            continue
        values = sweep.get("values")
        if not isinstance(values, list) or not values:
            continue
        numeric_values = []
        for value in values:
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
        if numeric_values:
            ordered_values.append((parameter, numeric_values))

    if not ordered_values:
        return [{"family": family_name, "parameters": dict(base), "tags": ["baseline"]}]

    candidates: List[Dict[str, Any]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for combo in itertools.product(*[values for _, values in ordered_values]):
        parameters = dict(base)
        tags = []
        for (parameter, _), value in zip(ordered_values, combo):
            parameters[parameter] = float(value)
            tags.append(f"{parameter}={value}")
        for parameter in fixed:
            if parameter in base:
                parameters[parameter] = base[parameter]
        key = tuple(sorted((name, float(value)) for name, value in parameters.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"family": family_name, "parameters": parameters, "tags": tags})
        if len(candidates) >= max_candidates:
            break
    return candidates


def _initial_sweeps(family_name: str, base_parameters: Mapping[str, Any], sample_count: int) -> List[Dict[str, Any]]:
    family = get_family(family_name)
    primary = family.sweep_priority[0]
    secondary = family.sweep_priority[1] if len(family.sweep_priority) > 1 else primary
    return [
        {
            "parameter": primary,
            "values": _global_values(family_name, primary, count=min(sample_count, 4)),
        },
        {
            "parameter": secondary,
            "values": _local_values(family_name, secondary, base_parameters.get(secondary), count=2),
        },
    ]


def _local_refinement_sweeps(
    family_name: str,
    base_parameters: Mapping[str, Any],
    sample_count: int,
) -> List[Dict[str, Any]]:
    family = get_family(family_name)
    primary = family.sweep_priority[0]
    secondary = family.sweep_priority[1] if len(family.sweep_priority) > 1 else primary
    return [
        {
            "parameter": primary,
            "values": _local_values(family_name, primary, base_parameters.get(primary), count=min(sample_count, 3)),
        },
        {
            "parameter": secondary,
            "values": _local_values(family_name, secondary, base_parameters.get(secondary), count=2),
        },
    ]


def _workflow_for_request(input_type: str) -> str:
    return {
        "canonical_geometry": "parameterized_geometry_refinement",
        "pdf": "literature_guided_experiments",
        "specified_geometry": "goal_conditioned_experimentation",
        "step": "reference_geometry_refinement",
        "unspecified_goal": "autonomous_goal_exploration",
    }.get(input_type, "autonomous_goal_exploration")


def _global_values(family_name: str, parameter: str, count: int) -> List[float]:
    family = get_family(family_name)
    spec = family.parameters[parameter]
    if count <= 1:
        return [spec.default]
    if spec.kind == "int":
        step = max(1, int((spec.max_value - spec.min_value) / max(count - 1, 1)))
        return [
            float(min(int(spec.max_value), int(spec.min_value) + (idx * step)))
            for idx in range(count)
        ]
    span = spec.max_value - spec.min_value
    return [
        round(spec.min_value + (span * idx / max(count - 1, 1)), 6)
        for idx in range(count)
    ]


def _local_values(family_name: str, parameter: str, center: Any, count: int) -> List[float]:
    family = get_family(family_name)
    spec = family.parameters[parameter]
    center_value = spec.clamp(center)
    if count <= 1:
        return [center_value]
    if spec.kind == "int":
        values = {spec.clamp(center_value), spec.clamp(center_value - 1), spec.clamp(center_value + 1)}
        return sorted(values)
    span = spec.max_value - spec.min_value
    delta = max(span * 0.08, 0.05)
    values = [
        spec.clamp(center_value - delta),
        spec.clamp(center_value),
        spec.clamp(center_value + delta),
    ]
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped[: max(count, 1)]


def _best_family_record(records: List[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not records:
        return None
    return max(
        records,
        key=lambda record: float((record.get("evaluation") or {}).get("actual_success_probability") or 0.0),
    )


def _most_uncertain_record(records: List[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not records:
        return None
    return max(
        records,
        key=lambda record: float((record.get("uncertainty") or {}).get("uncertainty_score") or 0.0),
    )


def _boundary_record(records: List[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not records:
        return None
    return min(
        records,
        key=lambda record: abs(float((record.get("prediction") or {}).get("success_probability") or 0.5) - 0.5),
    )


def _build_rationale(
    request: Mapping[str, Any],
    family_name: str,
    strategy: str,
    family_records: List[Mapping[str, Any]],
    *,
    fixed_parameters: List[str],
    vary_parameters: List[str],
) -> List[str]:
    rationale = [f"selected geometry family '{family_name}'"]
    if request.get("input_type") == "canonical_geometry":
        rationale.append("canonical geometry input accepted with hard parameter constraints")
    if request.get("input_type") == "pdf":
        rationale.append("using literature-guided workflow for PDF input")
    if request.get("input_type") == "step":
        rationale.append("using reference-geometry workflow for STEP input")
    if not family_records:
        rationale.append("no prior experiments for this family, so the planner is bootstrapping a sweep")
    else:
        rationale.append(f"found {len(family_records)} prior experiments for this family")
    if fixed_parameters:
        rationale.append(f"fixed parameters enforced: {', '.join(fixed_parameters)}")
    if vary_parameters:
        rationale.append(f"vary-only parameters requested: {', '.join(vary_parameters)}")
    rationale.append(f"strategy set to '{strategy}'")
    return rationale


def _constraint_parameter_names(constraints: Mapping[str, Any], key: str) -> List[str]:
    value = constraints.get(key)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _apply_parameter_constraints_to_sweeps(
    *,
    sweep_specs: List[Mapping[str, Any]],
    fixed_parameters: List[str],
    vary_parameters: List[str],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    fixed_set = set(fixed_parameters)
    vary_set = set(vary_parameters)

    for sweep in sweep_specs:
        parameter = str(sweep.get("parameter") or "").strip()
        values = sweep.get("values")
        if parameter in fixed_set:
            continue
        if vary_set and parameter not in vary_set:
            continue
        if not isinstance(values, list) or not values:
            continue
        filtered.append({"parameter": parameter, "values": list(values)})
    return filtered


def _coerce_numeric_parameters(payload: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _merge_unique_names(left: List[str], right: List[str]) -> List[str]:
    out: List[str] = []
    for name in list(left) + list(right):
        key = str(name).strip()
        if key and key not in out:
            out.append(key)
    return out


def _infer_goal_constraints(
    *,
    goal: str,
    family_name: str,
    known_family: bool,
    base_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    text = str(goal or "").strip().lower()
    if not text:
        return {
            "parameter_overrides": {},
            "sweep_specs": [],
            "fixed_parameters": [],
            "vary_parameters": [],
        }

    overrides: Dict[str, float] = {}
    fixed: List[str] = []
    vary: List[str] = []
    sweep_specs: List[Dict[str, Any]] = []

    family = get_family(family_name)
    valid_param_names = set(family.parameters.keys()) if known_family else set(str(k) for k in base_parameters.keys())

    spacing_match = re.search(
        r"spacing(?:\s+of)?\s*([0-9]*\.?[0-9]+)\s*(nm|um|μm|mm|cm)",
        text,
    )
    if spacing_match:
        spacing_um = _to_um(float(spacing_match.group(1)), spacing_match.group(2))
        spacing_param = "pitch_um" if known_family and "pitch_um" in valid_param_names else "period_um"
        overrides[spacing_param] = spacing_um
        fixed.append(spacing_param)

    radius_match = re.search(
        r"(?:radius|r)\s*(?:of)?\s*([0-9]*\.?[0-9]+)\s*(nm|um|μm|mm|cm)",
        text,
    )
    if radius_match:
        radius_um = _to_um(float(radius_match.group(1)), radius_match.group(2))
        if known_family and "radius_um" in valid_param_names:
            overrides["radius_um"] = radius_um
            fixed.append("radius_um")
        elif "lens_radius_um" in valid_param_names:
            overrides["lens_radius_um"] = radius_um
            fixed.append("lens_radius_um")

    # Parse array size like "20x20", "3×3", optionally near "array"/"pillars".
    # For known families with count_x/count_y parameters (pillar_array, microlens_array).
    count_match = re.search(r"(\d+)\s*[x×]\s*(\d+)", text)
    if count_match and known_family and {"count_x", "count_y"} <= valid_param_names:
        try:
            cx = int(count_match.group(1))
            cy = int(count_match.group(2))
            overrides["count_x"] = float(max(1, cx))
            overrides["count_y"] = float(max(1, cy))
            fixed.extend(["count_x", "count_y"])
        except Exception:
            pass

    domain_match = re.search(
        r"array\s+over\s+(?:a\s+)?([0-9]*\.?[0-9]+)\s*(nm|um|μm|mm|cm)(?:\s*[x×]\s*([0-9]*\.?[0-9]+)\s*(nm|um|μm|mm|cm)?)?\s+domain",
        text,
    )
    if domain_match:
        x_val = _to_um(float(domain_match.group(1)), domain_match.group(2))
        if domain_match.group(3):
            y_unit = domain_match.group(4) or domain_match.group(2)
            y_val = _to_um(float(domain_match.group(3)), y_unit)
        else:
            y_val = x_val
        overrides["domain_x_um"] = x_val
        overrides["domain_y_um"] = y_val
        fixed.extend(["domain_x_um", "domain_y_um"])

    range_match = re.search(
        r"vary(?:ing)?\s+(?:the\s+)?([a-z0-9_ ]+?)\s+from\s+([0-9]*\.?[0-9]+)\s*(nm|um|μm|mm|cm)?\s*(?:to|-)\s*([0-9]*\.?[0-9]+)\s*(nm|um|μm|mm|cm)?",
        text,
    )
    if range_match:
        param_phrase = str(range_match.group(1) or "").strip()
        lo = float(range_match.group(2))
        hi = float(range_match.group(4))
        unit_lo = range_match.group(3) or range_match.group(5) or "um"
        unit_hi = range_match.group(5) or range_match.group(3) or "um"
        lo_um = _to_um(lo, unit_lo)
        hi_um = _to_um(hi, unit_hi)
        if hi_um < lo_um:
            lo_um, hi_um = hi_um, lo_um
        parameter = _map_phrase_to_parameter(param_phrase, known_family=known_family, family=family, base_parameters=base_parameters)
        if parameter:
            sweep_specs.append({"parameter": parameter, "values": _linspace(lo_um, hi_um, count=5)})
            vary.append(parameter)
            overrides.setdefault(parameter, lo_um)

    return {
        "parameter_overrides": overrides,
        "sweep_specs": sweep_specs,
        "fixed_parameters": fixed,
        "vary_parameters": vary,
    }


def _map_phrase_to_parameter(
    phrase: str,
    *,
    known_family: bool,
    family: Any,
    base_parameters: Mapping[str, Any],
) -> str:
    text = str(phrase or "").strip().lower()
    if not text:
        return "height_um"
    if "height" in text:
        return "height_um"
    if "radius" in text:
        if known_family and "radius_um" in family.parameters:
            return "radius_um"
        return "radius_um"
    if "pitch" in text or "spacing" in text or "period" in text:
        if known_family and "pitch_um" in family.parameters:
            return "pitch_um"
        return "period_um"
    key = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if known_family and key in family.parameters:
        return key
    if key in base_parameters:
        return key
    return "height_um"


def _to_um(value: float, unit: str) -> float:
    u = str(unit or "").strip().lower()
    if u in {"um", "μm"}:
        return float(value)
    if u == "nm":
        return float(value) / 1000.0
    if u == "mm":
        return float(value) * 1000.0
    if u == "cm":
        return float(value) * 10000.0
    return float(value)


def _linspace(lo: float, hi: float, *, count: int = 5) -> List[float]:
    count = max(2, int(count))
    span = float(hi) - float(lo)
    return [round(float(lo) + (span * idx / float(count - 1)), 6) for idx in range(count)]

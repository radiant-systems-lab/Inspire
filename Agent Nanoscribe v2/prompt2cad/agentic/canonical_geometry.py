"""Canonical geometry parsing and request normalization."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping

from .families import DEFAULT_FAMILY_NAME, FAMILY_REGISTRY, get_family, has_family, infer_family_from_text


def parse_canonical_geo_language(raw: Any) -> Dict[str, Any]:
    """Parse JSON-like or line-based canonical geometry input."""
    payload = _coerce_to_mapping(raw)
    if payload:
        canonical_keys = {"family", "geometry", "geometry_family", "parameters", "sweeps", "sweep", "constraints"}
        has_structured_key = any(key in payload for key in canonical_keys)
        has_prefixed_key = any(str(key).startswith(("sweep.", "constraint.")) for key in payload)
        requested_family = str(
            payload.get("family")
            or payload.get("geometry")
            or payload.get("geometry_family")
            or DEFAULT_FAMILY_NAME
        ).strip().lower()
        known_family = has_family(requested_family)
        family = get_family(requested_family)
        extracted = payload.get("parameters") if isinstance(payload.get("parameters"), Mapping) else _extract_parameters(payload)
        extracted_mapping = dict(extracted) if isinstance(extracted, Mapping) else {}
        if known_family:
            valid_parameters = {
                str(name): value
                for name, value in extracted_mapping.items()
                if str(name) in family.parameters
            }
        else:
            valid_parameters = _coerce_numeric_parameters(extracted_mapping)
        has_parameter_signal = bool(valid_parameters)
        if not (has_structured_key or has_prefixed_key or has_parameter_signal):
            return _empty_canonical_result()

        if known_family:
            parameters = family.clamp_parameters(valid_parameters) if valid_parameters else {}
            sweeps = _normalize_sweeps(payload.get("sweeps") or payload.get("sweep"), family.name)
            family_name = family.name
        else:
            parameters = dict(valid_parameters)
            sweeps = _normalize_sweeps_generic(payload.get("sweeps") or payload.get("sweep"))
            family_name = requested_family or "custom_geometry"

        raw_constraints = payload.get("constraints") if isinstance(payload.get("constraints"), Mapping) else {}
        allowed_parameter_names = set(parameters.keys()) | {str(item.get("parameter") or "").strip() for item in sweeps}
        constraints = _normalize_constraints(
            raw_constraints,
            family_name=family_name,
            allowed_parameter_names=allowed_parameter_names,
        )
        goal = str(payload.get("goal") or "").strip()
        accepted = bool(parameters) or bool(sweeps)
        confidence = 0.98 if known_family and accepted else 0.75 if accepted else 0.0
        return {
            "accepted": accepted,
            "confidence": confidence,
            "reason": "structured canonical geometry input" if accepted else "unable to validate structured geometry",
            "geometry": {
                "family": family_name,
                "parameters": parameters,
                "sweeps": sweeps,
                "constraints": dict(constraints),
                "goal": goal,
                "hard_constraints_applied": accepted,
            },
        }

    if not isinstance(raw, str):
        return _empty_canonical_result()

    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if not lines or all(":" not in line for line in lines):
        return _empty_canonical_result()

    parsed: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}
    sweeps: Dict[str, Any] = {}
    for line in lines:
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in {"family", "geometry", "geometry_family"}:
            parsed["family"] = value
        elif key == "goal":
            parsed["goal"] = value
        elif key.startswith("sweep."):
            sweeps[key[6:]] = _parse_scalar_or_list(value)
        elif key.startswith("constraint."):
            constraints = parsed.setdefault("constraints", {})
            constraints[key[11:]] = _parse_scalar_or_list(value)
        else:
            parameters[key] = _parse_scalar_or_list(value)
    if parameters:
        parsed["parameters"] = parameters
    if sweeps:
        parsed["sweeps"] = sweeps
    return parse_canonical_geo_language(parsed)


def normalize_agent_input(raw_input: Any) -> Dict[str, Any]:
    """Normalize arbitrary user input into a planner-friendly request record."""
    request_id = f"req_{uuid.uuid4().hex[:10]}"
    canonical = parse_canonical_geo_language(raw_input)

    goal = ""
    source_path = None
    input_type = "unspecified_goal"
    workflow_hint = "autonomous_exploration"

    if isinstance(raw_input, Mapping):
        goal = str(raw_input.get("goal") or raw_input.get("prompt") or "").strip()
        source_path = raw_input.get("source_path")
        mapping_family_hint = str(
            raw_input.get("family")
            or raw_input.get("geometry_family")
            or ""
        ).strip()
    else:
        mapping_family_hint = ""
    if isinstance(raw_input, str):
        goal = raw_input.strip()
        possible_path = Path(goal)
        suffix = possible_path.suffix.lower()
        if suffix == ".pdf":
            input_type = "pdf"
            workflow_hint = "literature_guided_experiments"
            source_path = str(possible_path)
        elif suffix in {".step", ".stp"}:
            input_type = "step"
            workflow_hint = "reference_geometry_refinement"
            source_path = str(possible_path)

    if canonical["accepted"]:
        input_type = "canonical_geometry"
        workflow_hint = "parameterized_geometry_refinement"
        if not goal:
            goal = canonical["geometry"].get("goal") or ""
    elif goal and input_type == "unspecified_goal":
        input_type = "specified_geometry"
        workflow_hint = "goal_conditioned_experimentation"

    family_hint = (
        canonical["geometry"]["family"]
        if canonical["accepted"]
        else (mapping_family_hint or infer_family_from_text(goal))
    )

    return {
        "request_id": request_id,
        "goal": goal,
        "input_type": input_type,
        "workflow_hint": workflow_hint,
        "source_path": source_path,
        "canonical_geometry": canonical if canonical["accepted"] else None,
        "geometry_family_hint": family_hint,
        "structured_input_confidence": canonical.get("confidence", 0.0),
        "raw_input": raw_input,
    }


def _coerce_to_mapping(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{") and text.endswith("}"):
        try:
            value = json.loads(text)
            return dict(value) if isinstance(value, Mapping) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_parameters(payload: Mapping[str, Any]) -> Dict[str, Any]:
    reserved = {
        "family",
        "geometry",
        "geometry_family",
        "goal",
        "constraints",
        "sweeps",
        "sweep",
        "source_path",
        "prompt",
    }
    parameters = {
        str(key): value
        for key, value in payload.items()
        if key not in reserved and not str(key).startswith("constraint.")
    }
    return parameters


def _normalize_sweeps(value: Any, family_name: str) -> list[Dict[str, Any]]:
    family = get_family(family_name)
    if isinstance(value, list):
        sweeps = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            parameter = str(item.get("parameter") or "").strip()
            if parameter not in family.parameters:
                continue
            values = item.get("values")
            if isinstance(values, list) and values:
                sweeps.append(
                    {
                        "parameter": parameter,
                        "values": [family.parameters[parameter].clamp(v) for v in values],
                    }
                )
        return sweeps

    if not isinstance(value, Mapping):
        return []

    sweeps = []
    for parameter, values in value.items():
        if parameter not in family.parameters:
            continue
        parsed_values = values if isinstance(values, list) else _parse_scalar_or_list(str(values))
        if isinstance(parsed_values, list) and parsed_values:
            sweeps.append(
                {
                    "parameter": parameter,
                    "values": [family.parameters[parameter].clamp(v) for v in parsed_values],
                }
            )
    return sweeps


def _normalize_sweeps_generic(value: Any) -> list[Dict[str, Any]]:
    if isinstance(value, list):
        sweeps = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            parameter = str(item.get("parameter") or "").strip()
            if not parameter:
                continue
            values = item.get("values")
            if isinstance(values, list) and values:
                numeric_values = _coerce_numeric_list(values)
                if numeric_values:
                    sweeps.append({"parameter": parameter, "values": numeric_values})
        return sweeps

    if not isinstance(value, Mapping):
        return []

    sweeps = []
    for parameter, values in value.items():
        parameter_name = str(parameter).strip()
        if not parameter_name:
            continue
        parsed_values = values if isinstance(values, list) else _parse_scalar_or_list(str(values))
        if isinstance(parsed_values, list) and parsed_values:
            numeric_values = _coerce_numeric_list(parsed_values)
            if numeric_values:
                sweeps.append({"parameter": parameter_name, "values": numeric_values})
    return sweeps


def _normalize_constraints(
    value: Mapping[str, Any],
    *,
    family_name: str,
    allowed_parameter_names: set[str] | None = None,
) -> Dict[str, Any]:
    constraints = dict(value or {})

    fixed_parameters = _coerce_parameter_list(
        constraints.get("fixed_parameters")
        or constraints.get("keep_fixed")
        or constraints.get("locked_parameters"),
        family_name=family_name,
        allowed_parameter_names=allowed_parameter_names,
    )
    vary_parameters = _coerce_parameter_list(
        constraints.get("vary_parameters")
        or constraints.get("sweep_only"),
        family_name=family_name,
        allowed_parameter_names=allowed_parameter_names,
    )

    # Fixed wins on overlap: do not allow "vary and keep fixed" simultaneously.
    if fixed_parameters and vary_parameters:
        vary_parameters = [name for name in vary_parameters if name not in fixed_parameters]

    normalized: Dict[str, Any] = {
        key: item
        for key, item in constraints.items()
        if key not in {"keep_fixed", "locked_parameters", "sweep_only"}
    }
    if fixed_parameters:
        normalized["fixed_parameters"] = fixed_parameters
    if vary_parameters:
        normalized["vary_parameters"] = vary_parameters
    return normalized


def _coerce_parameter_list(
    value: Any,
    *,
    family_name: str,
    allowed_parameter_names: set[str] | None = None,
) -> list[str]:
    if allowed_parameter_names:
        names = set(allowed_parameter_names)
    elif has_family(family_name):
        family = get_family(family_name)
        names = set(family.parameters.keys())
    else:
        names = set()

    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        raw = [str(item).strip() for item in value if str(item).strip()]
    else:
        raw = []

    ordered: list[str] = []
    for name in raw:
        if (not names or name in names) and name not in ordered:
            ordered.append(name)
    return ordered


def _coerce_numeric_parameters(payload: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _coerce_numeric_list(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _parse_scalar_or_list(value: str) -> Any:
    text = str(value).strip()
    if not text:
        return text
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else parsed
        except json.JSONDecodeError:
            pass
    if "," in text:
        return [_parse_scalar_or_list(part) for part in text.split(",") if part.strip()]
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def _empty_canonical_result() -> Dict[str, Any]:
    return {
        "accepted": False,
        "confidence": 0.0,
        "reason": "input is not canonical geometry",
        "geometry": {
            "family": DEFAULT_FAMILY_NAME,
            "parameters": {},
            "sweeps": [],
            "constraints": {},
            "goal": "",
            "hard_constraints_applied": False,
        },
    }

"""Parameterized geometry families for the agentic experiment loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    min_value: float
    max_value: float
    default: float
    kind: str = "float"
    importance: float = 1.0

    def clamp(self, value: Any) -> float:
        if value is None:
            value = self.default
        if self.kind == "int":
            number = int(round(float(value)))
            return float(max(int(self.min_value), min(int(self.max_value), number)))
        number = float(value)
        return max(self.min_value, min(self.max_value, number))


@dataclass(frozen=True)
class GeometryFamily:
    name: str
    description: str
    keywords: tuple[str, ...]
    parameters: Dict[str, ParameterSpec]
    sweep_priority: tuple[str, ...]
    default_success_metrics: tuple[str, ...]

    def default_parameters(self) -> Dict[str, float]:
        return {name: spec.default for name, spec in self.parameters.items()}

    def clamp_parameters(self, values: Mapping[str, Any] | None = None) -> Dict[str, float]:
        payload = dict(values or {})
        return {
            name: spec.clamp(payload.get(name))
            for name, spec in self.parameters.items()
        }

    def parameter_names(self) -> Iterable[str]:
        return self.parameters.keys()


FAMILY_REGISTRY: Dict[str, GeometryFamily] = {
    "cylinder": GeometryFamily(
        name="cylinder",
        description="Single upright cylindrical test structure.",
        keywords=("cylinder", "pillar", "post", "rod"),
        parameters={
            "radius_um": ParameterSpec("radius_um", 0.20, 4.00, 0.80, importance=1.0),
            "height_um": ParameterSpec("height_um", 0.50, 20.00, 6.00, importance=1.0),
        },
        sweep_priority=("radius_um", "height_um"),
        default_success_metrics=("actual_success", "risk_score", "uncertainty_score"),
    ),
    "box": GeometryFamily(
        name="box",
        description="Rectangular calibration block.",
        keywords=("box", "block", "brick", "rectangular"),
        parameters={
            "width_um": ParameterSpec("width_um", 0.40, 10.00, 2.00, importance=0.9),
            "depth_um": ParameterSpec("depth_um", 0.40, 10.00, 2.00, importance=0.9),
            "height_um": ParameterSpec("height_um", 0.50, 15.00, 4.00, importance=0.8),
        },
        sweep_priority=("height_um", "width_um"),
        default_success_metrics=("actual_success", "risk_score", "surface_area_um2"),
    ),
    "cone": GeometryFamily(
        name="cone",
        description="Tapered cone or microneedle-like feature.",
        keywords=("cone", "needle", "tip", "spike"),
        parameters={
            "base_radius_um": ParameterSpec("base_radius_um", 0.25, 4.50, 1.10, importance=1.0),
            "tip_radius_um": ParameterSpec("tip_radius_um", 0.05, 1.00, 0.20, importance=0.8),
            "height_um": ParameterSpec("height_um", 0.80, 20.00, 7.00, importance=1.0),
        },
        sweep_priority=("height_um", "base_radius_um"),
        default_success_metrics=("actual_success", "risk_score", "slenderness"),
    ),
    "pillar_array": GeometryFamily(
        name="pillar_array",
        description="Array of vertical pillars for process-window sweeps.",
        keywords=("pillar array", "post array", "array", "pillar", "pillars", "post", "posts", "lattice", "grid"),
        parameters={
            "radius_um": ParameterSpec("radius_um", 0.20, 2.50, 0.70, importance=1.0),
            "height_um": ParameterSpec("height_um", 0.80, 20.00, 6.50, importance=1.0),
            "pitch_um": ParameterSpec("pitch_um", 0.60, 12.00, 2.40, importance=1.0),
            "count_x": ParameterSpec("count_x", 1.0, 25.0, 4.0, kind="int", importance=0.4),
            "count_y": ParameterSpec("count_y", 1.0, 25.0, 4.0, kind="int", importance=0.4),
        },
        sweep_priority=("radius_um", "pitch_um", "height_um"),
        default_success_metrics=("actual_success", "risk_score", "feature_spacing_um"),
    ),
    "microlens_array": GeometryFamily(
        name="microlens_array",
        description="Array of microlenses on a common base.",
        keywords=("lens", "microlens", "optical", "array"),
        parameters={
            "lens_radius_um": ParameterSpec("lens_radius_um", 0.50, 8.00, 2.50, importance=1.0),
            "sag_um": ParameterSpec("sag_um", 0.10, 4.00, 0.80, importance=1.0),
            "pitch_um": ParameterSpec("pitch_um", 1.00, 15.00, 5.50, importance=0.9),
            "count_x": ParameterSpec("count_x", 1.0, 8.0, 3.0, kind="int", importance=0.4),
            "count_y": ParameterSpec("count_y", 1.0, 8.0, 3.0, kind="int", importance=0.4),
            "base_thickness_um": ParameterSpec("base_thickness_um", 0.10, 3.00, 0.60, importance=0.5),
        },
        sweep_priority=("sag_um", "pitch_um", "lens_radius_um"),
        default_success_metrics=("actual_success", "risk_score", "fill_fraction"),
    ),
}

DEFAULT_FAMILY_NAME = "pillar_array"


def get_family(name: str | None) -> GeometryFamily:
    key = str(name or DEFAULT_FAMILY_NAME).strip().lower()
    return FAMILY_REGISTRY.get(key, FAMILY_REGISTRY[DEFAULT_FAMILY_NAME])


def has_family(name: str | None) -> bool:
    key = str(name or "").strip().lower()
    return key in FAMILY_REGISTRY


def infer_family_from_text(text: str | None) -> str:
    payload = str(text or "").strip().lower()
    if not payload:
        return DEFAULT_FAMILY_NAME
    best_name = DEFAULT_FAMILY_NAME
    best_score = (-1, -1)
    for family in FAMILY_REGISTRY.values():
        match_count = sum(1 for keyword in family.keywords if keyword in payload)
        specificity = sum(len(keyword) for keyword in family.keywords if keyword in payload)
        score = (match_count, specificity)
        if score > best_score:
            best_name = family.name
            best_score = score
    return best_name


def family_bounds(name: str | None) -> Dict[str, tuple[float, float]]:
    family = get_family(name)
    return {
        param_name: (spec.min_value, spec.max_value)
        for param_name, spec in family.parameters.items()
    }

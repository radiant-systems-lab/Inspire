"""Structured evaluation for simulated microfabrication runs."""

from __future__ import annotations

import hashlib
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping


class SimulatedEvaluationModule:
    def evaluate(self, execution_batch: Mapping[str, Any]) -> Dict[str, Any]:
        evaluated_records: List[Dict[str, Any]] = []
        for experiment in execution_batch.get("experiments", []):
            record = dict(experiment)
            hidden_probability = _simulated_ground_truth_probability(record)
            actual_success = hidden_probability >= 0.55
            failure_modes = _failure_modes(record, hidden_probability)
            evaluation = {
                "actual_success": actual_success,
                "actual_success_probability": round(hidden_probability, 6),
                "quality_score": round((0.6 * hidden_probability) + (0.4 * (1.0 - float(record["prediction"]["risk_score"]))), 6),
                "failure_modes": failure_modes,
                "evaluation_mode": "simulated",
            }
            record["evaluation"] = evaluation
            record["execution_status"] = "evaluated"
            evaluated_records.append(record)

        success_rates = [float(bool(record["evaluation"]["actual_success"])) for record in evaluated_records]
        return {
            "batch_id": execution_batch.get("batch_id"),
            "plan_id": execution_batch.get("plan_id"),
            "records": evaluated_records,
            "summary": {
                "batch_size": len(evaluated_records),
                "actual_success_rate": round(mean(success_rates), 6) if success_rates else None,
                "best_actual_success_probability": round(
                    max((record["evaluation"]["actual_success_probability"] for record in evaluated_records), default=0.0),
                    6,
                ),
            },
        }


def evaluate_results(execution_batch: Mapping[str, Any]) -> Dict[str, Any]:
    return SimulatedEvaluationModule().evaluate(execution_batch)


def _simulated_ground_truth_probability(record: Mapping[str, Any]) -> float:
    family = str(record.get("family") or "")
    params = record.get("geometry_parameters") or {}
    features = record.get("features") or {}
    process = record.get("process_params") or record.get("process_parameters") or {}

    min_feature = float(features.get("minimum_feature_size_um") or 0.0)
    slenderness = float(features.get("slenderness") or 0.0)
    spacing = float(features.get("feature_spacing_um") or 0.0)
    overhang = float(features.get("overhang_deg") or 0.0)

    probability = 0.88
    probability -= max(0.0, 0.42 * (0.45 - min_feature)) if min_feature < 0.45 else 0.0
    probability -= max(0.0, 0.03 * (slenderness - 5.5)) if slenderness > 5.5 else 0.0
    probability -= max(0.0, 0.30 * (0.60 - spacing)) if spacing < 0.60 else 0.0
    probability -= max(0.0, 0.007 * (overhang - 35.0)) if overhang > 35.0 else 0.0

    if family == "pillar_array":
        # Extra buckling penalty for tall, thin pillars (encourages a meaningful failure height in demos).
        # Tuned so r=1um pillars exhibit a failure around h≈18–20um.
        probability -= max(0.0, 0.16 * (slenderness - 7.0)) if slenderness > 7.0 else 0.0
        probability += _window_bonus(float(params.get("radius_um", 0.0)), 0.55, 1.10, 0.08)
        probability += _window_bonus(float(params.get("height_um", 0.0)), 4.0, 8.5, 0.08)
        probability += _window_bonus(float(params.get("pitch_um", 0.0)), 1.8, 4.5, 0.06)
    elif family == "microlens_array":
        probability += _window_bonus(float(params.get("sag_um", 0.0)), 0.35, 1.40, 0.09)
        probability += _window_bonus(float(params.get("pitch_um", 0.0)), 4.0, 7.0, 0.06)
    elif family == "cone":
        probability += _window_bonus(float(params.get("base_radius_um", 0.0)), 0.50, 1.60, 0.06)
        probability += _window_bonus(float(params.get("height_um", 0.0)), 3.0, 8.0, 0.08)
    elif family == "cylinder":
        probability += _window_bonus(float(params.get("radius_um", 0.0)), 0.50, 1.20, 0.05)
        probability += _window_bonus(float(params.get("height_um", 0.0)), 2.0, 7.0, 0.05)

    # Optional process-window effect (used by process-sweep dry runs).
    # If hatch/slice are present, penalize values beyond a geometry-dependent threshold.
    hatch = _coerce_float(process.get("hatch_um"))
    if hatch is None:
        hatch = _coerce_float(process.get("hatch_distance_um"))
    slice_um = _coerce_float(process.get("slice_um"))
    if slice_um is None:
        slice_um = _coerce_float(process.get("slice_thickness_um"))

    if hatch is not None or slice_um is not None:
        # Geometry-dependent threshold: smaller minimum features tolerate smaller hatch/slice.
        # Tuned so typical sweeps in ~[0.10, 0.55] yield both pass and fail for "borderline" geometries.
        critical = max(float(hatch or 0.0), float(slice_um or 0.0))

        # Threshold in roughly [0.28, 0.50] as min_feature increases.
        scale = max(0.0, min(1.0, (min_feature - 0.35) / 0.35))
        thr = 0.28 + 0.22 * scale
        thr = max(0.18, min(0.50, thr))

        if critical > thr:
            probability -= min(0.80, (critical - thr) * 3.5)
        elif critical < (thr - 0.08):
            probability += min(0.08, (thr - 0.08 - critical) * 0.7)

    probability += _stable_noise(family, params)
    return max(0.01, min(0.99, probability))


def _window_bonus(value: float, low: float, high: float, amplitude: float) -> float:
    if low <= value <= high:
        return amplitude
    if value < low:
        return -min(amplitude, (low - value) * amplitude)
    return -min(amplitude, (value - high) * amplitude)


def _stable_noise(family: str, params: Mapping[str, Any]) -> float:
    key = family + "|" + "|".join(f"{name}={params[name]}" for name in sorted(params))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    scaled = int(digest[:8], 16) / 0xFFFFFFFF
    return (scaled - 0.5) * 0.06


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _failure_modes(record: Mapping[str, Any], probability: float) -> List[str]:
    if probability >= 0.55:
        return []
    features = record.get("features") or {}
    failures = []
    if float(features.get("minimum_feature_size_um") or 0.0) < 0.40:
        failures.append("minimum feature collapsed")
    if float(features.get("slenderness") or 0.0) > 6.0:
        failures.append("pillar buckling")
    if float(features.get("feature_spacing_um") or 0.0) < 0.50:
        failures.append("feature fusion")
    if float(features.get("overhang_deg") or 0.0) > 40.0:
        failures.append("unsupported overhang")
    return failures or ["process window miss"]

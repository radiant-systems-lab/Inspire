"""Distance-based uncertainty model for experiment design."""

from __future__ import annotations

import math
from statistics import mean, pvariance
from typing import Any, Dict, Iterable, List, Mapping

from .classifier import ensemble_classifier_probabilities, predict_print_success
from .families import get_family, has_family
from .geometry import compute_geometric_features, generate_parametric_geometry


class DistanceBasedUncertaintyModel:
    def predict(
        self,
        *,
        family_name: str,
        parameters: Mapping[str, Any],
        dataset_records: Iterable[Mapping[str, Any]],
        classifier_prediction: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        known_family = has_family(family_name)
        family = get_family(family_name)
        normalized_family = family.name if known_family else str(family_name or "custom_geometry")
        records = [record for record in dataset_records if str(record.get("family")) == normalized_family]
        classifier_prediction = classifier_prediction or predict_print_success({})
        classifier_prob = float(classifier_prediction.get("success_probability") or 0.5)

        if not records:
            return {
                "predicted_success_probability": round(classifier_prob, 6),
                "uncertainty_score": 1.0,
                "distance_to_nearest_test": None,
                "reason": "no prior experiments for this family",
            }

        candidate = family.clamp_parameters(parameters) if known_family else _coerce_numeric_parameters(parameters)
        distances = []
        for record in records:
            hist_params = (record.get("geometry_parameters") or {})
            if known_family:
                distances.append(_normalized_distance(family.name, candidate, hist_params))
            else:
                distances.append(_normalized_distance_generic(candidate, hist_params))
        nearest = min(distances)
        coverage = max(0.0, 1.0 - min(nearest, 1.0))

        local_scores = [
            float(bool((record.get("evaluation") or {}).get("actual_success")))
            for record in records
            if isinstance(record.get("evaluation"), dict)
        ]
        local_mean = mean(local_scores) if local_scores else classifier_prob

        candidate_geometry = generate_parametric_geometry(normalized_family, candidate, source="uncertainty")
        candidate_features = compute_geometric_features(candidate_geometry)
        ensemble_probs = list(ensemble_classifier_probabilities(candidate_features))
        ensemble_var = pvariance(ensemble_probs) if len(ensemble_probs) > 1 else 0.0

        predicted = max(0.01, min(0.99, (0.55 * classifier_prob) + (0.45 * local_mean)))
        uncertainty = max(0.01, min(1.0, (0.70 * (1.0 - coverage)) + min(0.30, ensemble_var * 20.0)))

        return {
            "predicted_success_probability": round(predicted, 6),
            "uncertainty_score": round(uncertainty, 6),
            "distance_to_nearest_test": round(nearest, 6),
            "reason": "distance from tested points blended with ensemble variance",
        }


def predict_uncertainty(
    family_name: str,
    parameters: Mapping[str, Any],
    dataset_records: Iterable[Mapping[str, Any]],
    classifier_prediction: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return DistanceBasedUncertaintyModel().predict(
        family_name=family_name,
        parameters=parameters,
        dataset_records=dataset_records,
        classifier_prediction=classifier_prediction,
    )


def _normalized_distance(
    family_name: str,
    candidate: Mapping[str, Any],
    history: Mapping[str, Any],
) -> float:
    family = get_family(family_name)
    squares: List[float] = []
    for param_name, spec in family.parameters.items():
        low = spec.min_value
        high = spec.max_value
        span = max(high - low, 1e-6)
        cand = float(candidate.get(param_name, spec.default))
        prev = float(history.get(param_name, spec.default))
        squares.append(((cand - prev) / span) ** 2)
    return math.sqrt(sum(squares) / max(len(squares), 1))


def _normalized_distance_generic(
    candidate: Mapping[str, Any],
    history: Mapping[str, Any],
) -> float:
    keys = sorted(set(candidate.keys()) | set(history.keys()))
    if not keys:
        return 1.0
    squares: List[float] = []
    for key in keys:
        cand = _to_float(candidate.get(key))
        prev = _to_float(history.get(key))
        scale = max(abs(cand), abs(prev), 1.0)
        squares.append(((cand - prev) / scale) ** 2)
    return math.sqrt(sum(squares) / len(squares))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_numeric_parameters(payload: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out

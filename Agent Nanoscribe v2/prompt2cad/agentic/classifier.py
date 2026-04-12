"""Placeholder printability classifier for agentic planning."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class HeuristicPrintabilityClassifier:
    """Synthetic geometry-to-printability classifier."""

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        risk = 0.08
        reasons: List[Dict[str, Any]] = []

        min_feature = float(features.get("minimum_feature_size_um") or 0.0)
        if min_feature < 0.40:
            penalty = min(0.45, (0.40 - min_feature) * 1.1)
            risk += penalty
            reasons.append({"feature": "minimum_feature_size_um", "penalty": penalty, "reason": "minimum feature too small"})

        slenderness = float(features.get("slenderness") or 0.0)
        if slenderness > 6.0:
            penalty = min(0.30, (slenderness - 6.0) * 0.025)
            risk += penalty
            reasons.append({"feature": "slenderness", "penalty": penalty, "reason": "feature is slender and likely fragile"})

        spacing = float(features.get("feature_spacing_um") or 0.0)
        if spacing < 0.50:
            penalty = min(0.25, (0.50 - spacing) * 0.5)
            risk += penalty
            reasons.append({"feature": "feature_spacing_um", "penalty": penalty, "reason": "neighboring features are too close"})

        overhang = float(features.get("overhang_deg") or 0.0)
        if overhang > 40.0:
            penalty = min(0.20, (overhang - 40.0) * 0.01)
            risk += penalty
            reasons.append({"feature": "overhang_deg", "penalty": penalty, "reason": "unsupported overhang risk"})

        height = float(features.get("bbox_z_um") or 0.0)
        if height > 12.0:
            penalty = min(0.18, (height - 12.0) * 0.015)
            risk += penalty
            reasons.append({"feature": "bbox_z_um", "penalty": penalty, "reason": "print is tall for the current support assumptions"})

        fill_fraction = float(features.get("fill_fraction") or 0.0)
        if fill_fraction < 0.10:
            penalty = min(0.10, (0.10 - fill_fraction) * 0.8)
            risk += penalty
            reasons.append({"feature": "fill_fraction", "penalty": penalty, "reason": "geometry is sparse and may under-anchor"})

        risk = max(0.01, min(0.99, risk))
        success_probability = round(1.0 - risk, 6)
        ranked_reasons = sorted(reasons, key=lambda item: item["penalty"], reverse=True)

        return {
            "pass_fail_prediction": success_probability >= 0.55,
            "success_probability": success_probability,
            "risk_score": round(risk, 6),
            "reason": "; ".join(item["reason"] for item in ranked_reasons[:3]) or "low predicted risk",
            "feature_risk_breakdown": [
                {
                    "feature": item["feature"],
                    "penalty": round(item["penalty"], 6),
                    "reason": item["reason"],
                }
                for item in ranked_reasons
            ],
        }


def predict_print_success(features: Dict[str, Any]) -> Dict[str, Any]:
    return HeuristicPrintabilityClassifier().predict(features)


def ensemble_classifier_probabilities(features: Dict[str, Any]) -> Iterable[float]:
    base = predict_print_success(features)["success_probability"]
    min_feature = float(features.get("minimum_feature_size_um") or 0.0)
    spacing = float(features.get("feature_spacing_um") or 0.0)
    yield max(0.01, min(0.99, base))
    yield max(0.01, min(0.99, base + (0.04 if min_feature > 0.55 else -0.04)))
    yield max(0.01, min(0.99, base + (0.03 if spacing > 0.9 else -0.03)))

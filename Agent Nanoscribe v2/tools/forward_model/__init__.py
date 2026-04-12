"""
tools/forward_model
-------------------
Forward model facade: printability prediction + uncertainty.

Two backends are available:

StepForwardModel  (NEW — use this for the ML pipeline)
    Trained on STEP-derived geometry features + print recipe (slice_um, hatch_um).
    Input:  geometry_metrics dict (from compute_geometry_metrics / STEP pipeline)
            + slice_um, hatch_um floats
    Output: {p_pass, uncertainty, model_type, features_approximated, recipe, warning}
    Train:  StepForwardModel.fit_from_data()   — fits XGBoost on all labeled CSVs
    Load:   StepForwardModel.load()            — loads from data/models/forward_model.pkl

Convenience functions:
    design_print_experiment(candidates, budget, strategy)  — see tools/experiments
    get_best_recipe(geometry_metrics)                      — single-geometry inverse

Legacy (geometry-family-based, no recipe awareness):
    HeuristicPrintabilityClassifier, TrainedSurrogate, get_surrogate, etc.
    These are kept for backward compatibility with the existing agent_core loop.
"""

# ── New ML-backed forward model ────────────────────────────────────────────────
from .step_predictor import (
    StepForwardModel,
    ENG_FEATURE_NAMES,
)

# ── Legacy family-based models (backward compat) ───────────────────────────────
from prompt2cad.agentic.classifier import HeuristicPrintabilityClassifier, predict_print_success
from prompt2cad.agentic.uncertainty import DistanceBasedUncertaintyModel, predict_uncertainty
from prompt2cad.agentic.surrogate import (
    HeuristicSurrogate,
    TrainedSurrogate,
    PrintOutcomeSurrogate,
    get_surrogate,
)
from prompt2cad.agentic.geometry import compute_geometric_features, generate_parametric_geometry

__all__ = [
    # New
    "StepForwardModel",
    "ENG_FEATURE_NAMES",
    # (CANDIDATE_RECIPES lives in tools.experiments.recipe_designer)
    # Legacy
    "HeuristicPrintabilityClassifier",
    "predict_print_success",
    "DistanceBasedUncertaintyModel",
    "predict_uncertainty",
    "HeuristicSurrogate",
    "TrainedSurrogate",
    "PrintOutcomeSurrogate",
    "get_surrogate",
    "compute_geometric_features",
    "generate_parametric_geometry",
]

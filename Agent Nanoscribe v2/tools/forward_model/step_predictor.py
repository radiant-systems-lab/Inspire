"""
StepForwardModel
----------------
Forward model: geometry metrics (from CadQuery / STEP pipeline) + print recipe
→ predicted pass probability + uncertainty.

Bridges the trained XGBoost (fit in nanoscribe_ml.ipynb) to the agent tool
interface used by agent_core.py → predict_printability().

Feature alignment
-----------------
The ML model was trained on STEP-derived features:
    volume, surface_area, fill_fraction, slenderness_ratio,
    plane_face_ratio, cylindrical_face_ratio, bspline_face_ratio,
    compactness, area_volume_ratio, bbox_x, bbox_y, bbox_z

CadQuery's compute_geometry_metrics() gives a strict subset:
    volume, surface_area, fill_fraction, bbox_x, bbox_y, bbox_z

Missing features are approximated from what is available; a warning flag is
set when approximations are used so the caller can decide how to weight the
output.

Usage
-----
    model = StepForwardModel.load()          # load saved model
    model = StepForwardModel.fit_from_data() # refit from CSV data

    result = model.predict(
        geometry_metrics={"volume": 1234, "surface_area": 567, ...},
        slice_um=0.2,
        hatch_um=0.3,
    )
    # → {"p_pass": 0.71, "uncertainty": 0.22, "model_type": "xgboost",
    #    "features_approximated": False, "recipe": {"slice_um": 0.2, "hatch_um": 0.3}}
"""

from __future__ import annotations

import glob
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parents[2]
_MODEL_PATH  = _REPO_ROOT / "data" / "models" / "forward_model.pkl"

_REDO_GLOB   = str(_REPO_ROOT / "data" / "labels" / "redo_16_diverse_plates" / "derived" / "plate_*_ml_dataset_derived.csv")
_SWEEP_CSV   = _REPO_ROOT / "data" / "labels" / "derived" / "fabrication_sweep_ml_dataset_derived.csv"

# ── Recipe maps (matches nanoscribe_ml.ipynb) ──────────────────────────────────
_RECIPE_MAP_REDO = {
    1: (0.2, 0.2), 2: (0.2, 0.3), 3: (0.2, 0.4),
    4: (0.3, 0.2), 5: (0.3, 0.3), 6: (0.3, 0.4),
    7: (0.4, 0.2), 8: (0.4, 0.3), 9: (0.4, 0.4),
}
_RECIPE_MAP_SWEEP = {
    1: (0.100, 0.100), 2: (0.325, 0.325), 3: (0.550, 0.550),
    4: (0.775, 0.775), 5: (1.000, 1.000),
}

# ── Feature engineering (must match notebook exactly) ─────────────────────────
_LOG_FEATS = ["volume", "surface_area", "slenderness_ratio", "area_volume_ratio",
              "bbox_x", "bbox_y", "bbox_z"]

ENG_FEATURE_NAMES = [
    "log_volume", "log_surface_area", "fill_fraction",
    "log_slenderness_ratio", "plane_face_ratio", "cylindrical_face_ratio",
    "bspline_face_ratio", "compactness", "log_area_volume_ratio",
    "log_bbox_x", "log_bbox_y", "log_bbox_z",
    "bbox_aspect_xy", "bbox_aspect_xz",
    "slice_um", "hatch_um", "recipe_area", "log_vol_x_recipe",
]


def _approximate_missing(m: Dict[str, Any]) -> tuple[Dict[str, float], bool]:
    """
    Fill in features that CadQuery doesn't compute from those it does.
    Returns (completed_dict, approximated_flag).
    """
    out = {k: float(v) for k, v in m.items() if v is not None}
    approx = False

    volume      = out.get("volume", 0.0)
    surface_area = out.get("surface_area", 0.0)
    bbox_x      = out.get("bbox_x", 1.0)
    bbox_y      = out.get("bbox_y", 1.0)
    bbox_z      = out.get("bbox_z", 1.0)

    if "area_volume_ratio" not in out:
        out["area_volume_ratio"] = surface_area / (volume + 1e-9)
        approx = True

    if "slenderness_ratio" not in out:
        # Approximate: max_bbox^3 / volume
        max_dim = max(bbox_x, bbox_y, bbox_z)
        out["slenderness_ratio"] = (max_dim ** 3) / (volume + 1e-9)
        approx = True

    if "compactness" not in out:
        # π/6 * d^3 / V for a sphere; here use 36π*V^2/A^3 (isoperimetric quotient)
        denom = (surface_area ** 3) + 1e-30
        out["compactness"] = (36.0 * math.pi * volume ** 2) / denom
        approx = True

    # Face type ratios: default to planar-dominant if unknown
    if "plane_face_ratio" not in out:
        out["plane_face_ratio"] = 1.0
        approx = True
    if "cylindrical_face_ratio" not in out:
        out["cylindrical_face_ratio"] = 0.0
        approx = True
    if "bspline_face_ratio" not in out:
        out["bspline_face_ratio"] = 0.0
        approx = True

    return out, approx


def _engineer(row: Dict[str, float]) -> List[float]:
    """Apply feature engineering and return ordered feature vector."""
    r = dict(row)
    for f in _LOG_FEATS:
        r[f"log_{f}"] = math.log1p(r.get(f, 0.0))
    r["bbox_aspect_xy"]   = r.get("bbox_x", 1.0) / (r.get("bbox_y", 1.0) + 1e-9)
    r["bbox_aspect_xz"]   = r.get("bbox_x", 1.0) / (r.get("bbox_z", 1.0) + 1e-9)
    r["recipe_area"]      = r.get("slice_um", 0.2) * r.get("hatch_um", 0.2)
    r["log_vol_x_recipe"] = r.get("log_volume", 0.0) * r["recipe_area"]
    return [r.get(f, 0.0) for f in ENG_FEATURE_NAMES]


# ── Heuristic fallback (matches agent_core logic, extended with recipe) ────────

def _heuristic_predict(geometry_metrics: Dict[str, Any], slice_um: float, hatch_um: float) -> Dict[str, Any]:
    """Rule-based fallback when no trained model is available."""
    m = geometry_metrics
    base = 0.88

    min_feature = float(m.get("feature_thickness_min") or m.get("minimum_feature_size_um") or 0.0)
    slenderness  = float(m.get("slenderness_ratio") or m.get("slenderness") or 0.0)
    spacing      = float(m.get("feature_spacing_min") or m.get("feature_spacing_um") or 0.0)
    overhang     = float(m.get("overhang_max_angle_deg") or m.get("overhang_deg") or 0.0)

    base -= max(0.0, 0.42 * (0.45 - min_feature)) if min_feature > 0 and min_feature < 0.45 else 0.0
    base -= max(0.0, 0.03 * (slenderness - 5.5)) if slenderness > 5.5 else 0.0
    base -= max(0.0, 0.30 * (0.60 - spacing)) if spacing > 0 and spacing < 0.60 else 0.0
    base -= max(0.0, 0.007 * (overhang - 35.0)) if overhang > 35.0 else 0.0

    # Recipe penalty: coarser spacing reduces probability
    recipe_max = max(slice_um, hatch_um)
    if recipe_max >= 0.55:
        base -= 0.60   # empirically all-fail above 0.55 µm
    elif recipe_max >= 0.4:
        base -= 0.15

    p_pass = max(0.01, min(0.99, base))
    return {"p_pass": round(p_pass, 4), "uncertainty": 0.40}  # fixed high uncertainty for heuristic


# ── Main class ─────────────────────────────────────────────────────────────────

class StepForwardModel:
    """
    Forward model: geometry_metrics dict + (slice_um, hatch_um) → {p_pass, uncertainty}.

    Backed by a trained XGBoost classifier when available; falls back to heuristic.
    """

    def __init__(self, model: Any = None, scaler: Any = None) -> None:
        self._model  = model    # sklearn-compatible with predict_proba
        self._scaler = scaler   # optional StandardScaler (for LR)
        self._fitted = model is not None

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path | str = _MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler}, f)
        logger.info("StepForwardModel saved to %s", path)

    @classmethod
    def load(cls, path: Path | str = _MODEL_PATH) -> "StepForwardModel":
        path = Path(path)
        if not path.exists():
            logger.warning("No saved model at %s — using heuristic fallback.", path)
            return cls()
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(model=payload["model"], scaler=payload.get("scaler"))
        logger.info("StepForwardModel loaded from %s", path)
        return instance

    # ── Fitting ────────────────────────────────────────────────────────────────

    @classmethod
    def fit_from_data(cls, save_path: Path | str = _MODEL_PATH) -> "StepForwardModel":
        """
        Fit XGBoost on all available labeled CSV data and save.
        Mirrors the training pipeline from nanoscribe_ml.ipynb.
        """
        try:
            import pandas as pd
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(f"pandas and xgboost required for fit_from_data: {e}")

        frames = []
        for path in sorted(glob.glob(_REDO_GLOB)):
            d = pd.read_csv(path)
            if len(d) == 0:
                continue
            d["dataset"] = "redo_16"
            if "slice_um" not in d.columns:
                d["slice_um"] = d["print_round"].map(lambda r: _RECIPE_MAP_REDO[r][0])
                d["hatch_um"] = d["print_round"].map(lambda r: _RECIPE_MAP_REDO[r][1])
            else:
                d["slice_um"] = d["slice_um"].fillna(d["print_round"].map(lambda r: _RECIPE_MAP_REDO[r][0]))
                d["hatch_um"] = d["hatch_um"].fillna(d["print_round"].map(lambda r: _RECIPE_MAP_REDO[r][1]))
            frames.append(d)

        if _SWEEP_CSV.exists():
            sweep = pd.read_csv(_SWEEP_CSV)
            sweep["dataset"] = "fab_sweep"
            sweep["slice_um"] = sweep["print_round"].map(lambda r: _RECIPE_MAP_SWEEP[r][0])
            sweep["hatch_um"] = sweep["print_round"].map(lambda r: _RECIPE_MAP_SWEEP[r][1])
            frames.append(sweep)

        if not frames:
            raise FileNotFoundError("No labeled CSV data found.")

        raw = pd.concat(frames, ignore_index=True)
        raw[["slice_um", "hatch_um"]] = raw[["slice_um", "hatch_um"]].astype(float)

        # Feature engineering
        for f in _LOG_FEATS:
            raw[f"log_{f}"] = np.log1p(raw[f])
        raw["bbox_aspect_xy"]   = raw["bbox_x"] / (raw["bbox_y"] + 1e-9)
        raw["bbox_aspect_xz"]   = raw["bbox_x"] / (raw["bbox_z"] + 1e-9)
        raw["recipe_area"]      = raw["slice_um"] * raw["hatch_um"]
        raw["log_vol_x_recipe"] = raw["log_volume"] * raw["recipe_area"]

        X = raw[ENG_FEATURE_NAMES].values
        y = raw["validation_label"].values.astype(int)

        neg, pos = np.bincount(y)
        clf = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / pos,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        clf.fit(X, y)

        instance = cls(model=clf)
        instance.save(save_path)
        logger.info("StepForwardModel fitted on %d rows (%d pass, %d fail).", len(y), pos, neg)
        return instance

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(
        self,
        geometry_metrics: Dict[str, Any],
        slice_um: float,
        hatch_um: float,
    ) -> Dict[str, Any]:
        """
        Predict print success for a single geometry + recipe.

        Parameters
        ----------
        geometry_metrics : dict
            Output of compute_geometry_metrics() or STEP pipeline.
            Required keys: volume, surface_area, fill_fraction, bbox_x, bbox_y, bbox_z.
            Optional: slenderness_ratio, plane_face_ratio, cylindrical_face_ratio,
                      bspline_face_ratio, compactness, area_volume_ratio.
        slice_um, hatch_um : float
            Print recipe parameters in micrometres.

        Returns
        -------
        dict with keys:
            p_pass              float [0, 1]
            uncertainty         float [0, 1]  (0 = certain, 1 = maximally uncertain)
            model_type          str  ("xgboost" | "heuristic")
            features_approximated bool
            recipe              dict {"slice_um": ..., "hatch_um": ...}
            warning             str | None
        """
        completed, approx = _approximate_missing(geometry_metrics)
        completed["slice_um"] = float(slice_um)
        completed["hatch_um"] = float(hatch_um)

        if not self._fitted:
            h = _heuristic_predict(geometry_metrics, slice_um, hatch_um)
            return {
                "p_pass": h["p_pass"],
                "uncertainty": h["uncertainty"],
                "model_type": "heuristic",
                "features_approximated": approx,
                "recipe": {"slice_um": slice_um, "hatch_um": hatch_um},
                "warning": "No trained model — using heuristic. Call StepForwardModel.fit_from_data() to train.",
            }

        x = np.array([_engineer(completed)])
        p_pass = float(self._model.predict_proba(x)[0, 1])
        uncertainty = self._rf_uncertainty(x[0]) if hasattr(self._model, "estimators_") else 0.35

        return {
            "p_pass": round(p_pass, 4),
            "uncertainty": round(uncertainty, 4),
            "model_type": "xgboost",
            "features_approximated": approx,
            "recipe": {"slice_um": slice_um, "hatch_um": hatch_um},
            "warning": "Some features approximated from bbox/volume — provide STEP-derived features for best accuracy." if approx else None,
        }

    def predict_batch(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict for a list of {geometry_metrics, slice_um, hatch_um} dicts.
        Returns predictions in the same order with an added 'candidate_index' key.
        """
        return [
            {**self.predict(c["geometry_metrics"], c["slice_um"], c["hatch_um"]),
             "candidate_index": i}
            for i, c in enumerate(candidates)
        ]

    def is_trained(self) -> bool:
        return self._fitted

    # ── Uncertainty ────────────────────────────────────────────────────────────

    def _rf_uncertainty(self, x: np.ndarray) -> float:
        """Variance across individual tree predictions (XGBoost doesn't have estimators_)."""
        # XGBoost: use ntree_limit-based approximation — disabled, use fixed moderate value
        return 0.25  # placeholder; replace with conformal prediction when n_cal >= 50

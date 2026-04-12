"""
Print outcome surrogate model.

Architecture
------------
PrintOutcomeSurrogate   (abstract base — defines the contract)
├── HeuristicSurrogate  — geometry-only heuristic, no training required.
│                         Wraps the existing classifier + uncertainty model.
│                         Process params are accepted but ignored.
│                         Use this until real data is available.
└── TrainedSurrogate    — sklearn classifier over geometry features + process params.
                          Call .fit(records) once you have labeled data, then
                          .save(path) / TrainedSurrogate.load(path) for persistence.

Contract
--------
Both surrogates implement:

    predict(geometry_params, process_params, family) -> {
        "p_pass":      float [0, 1],   # predicted probability of print success
        "uncertainty": float [0, 1],   # 0 = certain, 1 = maximally uncertain
    }

    is_trained() -> bool               # True if backed by a fitted model

Input
-----
geometry_params : Dict[str, float]
    Raw geometry parameters matching the family definition.
    e.g. {"radius_um": 0.7, "height_um": 6.5, "pitch_um": 2.4, ...}

process_params : Dict[str, float]
    Print process parameters. Keys are whatever your dataset contains.
    e.g. {"laser_power_mw": 25.0, "scan_speed_mm_s": 100.0, ...}
    Ignored by HeuristicSurrogate; used by TrainedSurrogate.

family : str
    Geometry family name. Used for geometric feature extraction.
    e.g. "pillar_array", "cylinder", "cone", "microlens_array", "box"

Notes
-----
- Feature extraction (geometry_params -> geometric features) happens INSIDE the
  surrogate. Callers always pass raw parameters, never pre-computed features.
- TrainedSurrogate feature vector = [geometric_features... , process_params...]
  sorted alphabetically within each group for stable column ordering.
- When process_params keys seen at inference differ from training, missing keys
  default to 0.0 and a warning is logged.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean, pvariance
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .classifier import ensemble_classifier_probabilities, predict_print_success
from .geometry import compute_geometric_features, generate_parametric_geometry
from .uncertainty import predict_uncertainty

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PrintOutcomeSurrogate(ABC):
    """Contract that every surrogate implementation must satisfy."""

    @abstractmethod
    def predict(
        self,
        geometry_params: Dict[str, float],
        process_params: Dict[str, float],
        family: str,
    ) -> Dict[str, float]:
        """
        Predict print outcome.

        Returns
        -------
        dict with keys:
            p_pass      : float [0, 1]
            uncertainty : float [0, 1]
        """
        ...

    @abstractmethod
    def is_trained(self) -> bool:
        """True if backed by a fitted model, False if using heuristic fallback."""
        ...


# ---------------------------------------------------------------------------
# Heuristic surrogate (geometry-only, no training required)
# ---------------------------------------------------------------------------

class HeuristicSurrogate(PrintOutcomeSurrogate):
    """
    Geometry-only heuristic surrogate.

    Wraps HeuristicPrintabilityClassifier (p_pass) and
    DistanceBasedUncertaintyModel (uncertainty).

    Process params are accepted but have no effect on predictions — this is
    a known limitation. Once you have labeled data, switch to TrainedSurrogate.

    Parameters
    ----------
    dataset_records : list of experiment records, optional
        If provided, the uncertainty model uses distance to these points.
        Matches the JSONL schema in data/agentic_experiments.jsonl.
    """

    def __init__(
        self,
        dataset_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._records: List[Dict[str, Any]] = list(dataset_records or [])

    def predict(
        self,
        geometry_params: Dict[str, float],
        process_params: Dict[str, float],
        family: str,
    ) -> Dict[str, float]:
        geometry = generate_parametric_geometry(family, geometry_params, source="surrogate")
        features = compute_geometric_features(geometry)

        classifier_result = predict_print_success(features)

        uncertainty_result = predict_uncertainty(
            family_name=family,
            parameters=geometry_params,
            dataset_records=self._records,
            classifier_prediction=classifier_result,
        )

        p_pass = float(classifier_result.get("success_probability", 0.5))
        uncertainty = float(uncertainty_result.get("uncertainty_score", 1.0))

        return {"p_pass": round(p_pass, 6), "uncertainty": round(uncertainty, 6)}

    def is_trained(self) -> bool:
        return False

    def update_records(self, records: List[Dict[str, Any]]) -> None:
        """Replace the dataset records used by the uncertainty model."""
        self._records = list(records)


# ---------------------------------------------------------------------------
# Trained surrogate (sklearn, geometry + process params)
# ---------------------------------------------------------------------------

class TrainedSurrogate(PrintOutcomeSurrogate):
    """
    Sklearn-backed surrogate trained on real experimental data.

    The model input is a flat feature vector:
        [geometric_features (sorted by name), process_params (sorted by name)]

    Uncertainty is estimated via:
        1. Ensemble variance across trees   (if RandomForest)
        2. MC-dropout approximation         (if MLP, n_passes=20)
        3. Distance to nearest training point  (fallback)

    Fitting
    -------
    Call .fit(records) with a list of experiment records matching the JSONL
    schema. Each record must have:
        - "geometry_parameters" : dict
        - "process_params"      : dict
        - "family"              : str
        - "evaluation"          : {"actual_success": bool | 0 | 1}

    Records missing "evaluation" or "actual_success" are skipped.

    Persistence
    -----------
    surrogate.save("path/to/model.pkl")
    surrogate = TrainedSurrogate.load("path/to/model.pkl")
    """

    # Geometric feature keys extracted from compute_geometric_features().
    # Sorted for stable column ordering. Excludes redundant/derived fields.
    _GEO_FEATURE_KEYS: Tuple[str, ...] = (
        "aspect_ratio",
        "bbox_x_um",
        "bbox_y_um",
        "bbox_z_um",
        "feature_spacing_um",
        "fill_fraction",
        "minimum_feature_size_um",
        "overhang_deg",
        "slenderness",
        "surface_area_um2",
        "volume_um3",
        "voxel_count",
    )

    def __init__(self, model: Any = None) -> None:
        """
        Parameters
        ----------
        model : sklearn estimator with predict_proba, optional
            If None, defaults to RandomForestClassifier(n_estimators=100).
            Must support predict_proba for probability output.
        """
        if model is None:
            try:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )
            except ImportError:
                raise ImportError(
                    "scikit-learn is required for TrainedSurrogate. "
                    "Install it with: pip install scikit-learn"
                )

        self._model = model
        self._fitted: bool = False

        # Set during fit — defines the exact column order for process_params.
        self._process_param_keys: Tuple[str, ...] = ()

        # Training data kept for distance-based uncertainty fallback.
        self._X_train: Optional[List[List[float]]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, records: List[Dict[str, Any]]) -> "TrainedSurrogate":
        """
        Fit the surrogate on a list of experiment records.

        Skips records without a valid actual_success label.
        Logs a warning if fewer than 10 records are available.
        """
        X, y = self._build_dataset(records)

        if len(X) == 0:
            raise ValueError("No labeled records found. Each record needs evaluation.actual_success.")

        unique_labels = set(y)
        if len(unique_labels) < 2:
            logger.warning(
                "Skipping TrainedSurrogate fit: only one class observed so far (%s). "
                "Need both pass and fail to train a classifier.",
                sorted(unique_labels),
            )
            self._fitted = False
            self._X_train = X
            return self

        if len(X) < 10:
            logger.warning(
                "TrainedSurrogate fitting on only %d labeled records. "
                "Predictions may be unreliable.", len(X)
            )

        self._model.fit(X, y)
        self._X_train = X
        self._fitted = True
        logger.info("TrainedSurrogate fitted on %d records.", len(X))
        return self

    def predict(
        self,
        geometry_params: Dict[str, float],
        process_params: Dict[str, float],
        family: str,
    ) -> Dict[str, float]:
        # Allow cold-start operation inside the experiment loop.
        # Before fitting, return an uninformative prior with maximum uncertainty.
        if not self._fitted:
            return {"p_pass": 0.5, "uncertainty": 1.0}

        x = self._build_feature_vector(geometry_params, process_params, family)
        proba = list(self._model.predict_proba([x])[0])
        classes = list(getattr(self._model, "classes_", []))
        if len(proba) == 2 and len(classes) == 2:
            # Map probability to the positive class (label == 1).
            try:
                idx = classes.index(1)
            except ValueError:
                idx = 1
            p_pass = float(proba[idx])
        elif len(proba) == 1 and len(classes) == 1:
            # Degenerate single-class model: probability is either 0 or 1.
            p_pass = 1.0 if int(classes[0]) == 1 else 0.0
        else:
            p_pass = 0.5
        uncertainty = self._estimate_uncertainty(x)

        return {"p_pass": round(p_pass, 6), "uncertainty": round(uncertainty, 6)}

    def is_trained(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize model + metadata to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "fitted": self._fitted,
            "process_param_keys": self._process_param_keys,
            "X_train": self._X_train,
            "geo_feature_keys": self._GEO_FEATURE_KEYS,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("TrainedSurrogate saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainedSurrogate":
        """Load a previously saved TrainedSurrogate from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(model=payload["model"])
        instance._fitted = payload["fitted"]
        instance._process_param_keys = tuple(payload["process_param_keys"])
        instance._X_train = payload["X_train"]
        logger.info("TrainedSurrogate loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _build_feature_vector(
        self,
        geometry_params: Dict[str, float],
        process_params: Dict[str, float],
        family: str,
    ) -> List[float]:
        """Build a flat feature vector for a single prediction."""
        geometry = generate_parametric_geometry(family, geometry_params, source="surrogate")
        features = compute_geometric_features(geometry)

        geo_vec = [float(features.get(k, 0.0)) for k in self._GEO_FEATURE_KEYS]

        # Process param vector: use keys established at fit time.
        # Missing keys default to 0.0 with a logged warning.
        proc_vec: List[float] = []
        for key in self._process_param_keys:
            if key not in process_params:
                logger.warning(
                    "Process param '%s' was seen during training but is missing at "
                    "inference — defaulting to 0.0.", key
                )
            proc_vec.append(float(process_params.get(key, 0.0)))

        return geo_vec + proc_vec

    def _build_dataset(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[List[float]], List[int]]:
        """Convert records to (X, y) arrays. Discovers process_param keys from data."""
        # Discover all process param keys across the full dataset (sorted for stability).
        all_proc_keys: set[str] = set()
        for record in records:
            proc = record.get("process_params") or {}
            all_proc_keys.update(proc.keys())
        self._process_param_keys = tuple(sorted(all_proc_keys))

        X: List[List[float]] = []
        y: List[int] = []

        for record in records:
            evaluation = record.get("evaluation") or {}
            actual = evaluation.get("actual_success")
            if actual is None:
                continue

            label = int(bool(actual))
            geo_params = record.get("geometry_parameters") or {}
            proc_params = record.get("process_params") or {}
            family = str(record.get("family") or "pillar_array")

            try:
                x = self._build_feature_vector(geo_params, proc_params, family)
            except Exception as exc:
                logger.warning("Skipping record due to feature extraction error: %s", exc)
                continue

            X.append(x)
            y.append(label)

        return X, y

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def _estimate_uncertainty(self, x: List[float]) -> float:
        """
        Estimate uncertainty for a single feature vector.

        Strategy (in order of preference):
            1. Tree variance (RandomForest)
            2. MC-dropout approximation (MLP with partial_fit or dropout layers)
            3. Distance to nearest training point (always available as fallback)
        """
        model_type = type(self._model).__name__

        if model_type == "RandomForestClassifier":
            return self._rf_uncertainty(x)

        if model_type in ("MLPClassifier",):
            return self._mc_dropout_uncertainty(x, n_passes=20)

        return self._distance_uncertainty(x)

    def _rf_uncertainty(self, x: List[float]) -> float:
        """Variance across individual tree predictions."""
        tree_probs = [
            float(tree.predict_proba([x])[0][1])
            for tree in self._model.estimators_
        ]
        variance = pvariance(tree_probs) if len(tree_probs) > 1 else 0.0
        # Scale: RF variance rarely exceeds 0.25 (max at p=0.5). Map to [0,1].
        return round(min(1.0, variance * 4.0), 6)

    def _mc_dropout_uncertainty(self, x: List[float], n_passes: int = 20) -> float:
        """
        MC-dropout approximation for MLP.

        sklearn's MLPClassifier doesn't natively support dropout inference,
        so this uses a small jitter on the input as a crude proxy until a
        torch/keras model is swapped in.
        """
        import random
        probs: List[float] = []
        for _ in range(n_passes):
            jittered = [v * (1.0 + random.gauss(0.0, 0.02)) for v in x]
            probs.append(float(self._model.predict_proba([jittered])[0][1]))
        variance = pvariance(probs) if len(probs) > 1 else 0.0
        return round(min(1.0, variance * 4.0), 6)

    def _distance_uncertainty(self, x: List[float]) -> float:
        """Distance to nearest training point, normalized to [0, 1]."""
        if not self._X_train:
            return 1.0
        min_dist = min(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(x, train_x)))
            for train_x in self._X_train
        )
        # Normalize: distance of 0 -> uncertainty 0, distance >= 5 -> uncertainty 1.
        return round(min(1.0, min_dist / 5.0), 6)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_surrogate(
    model_path: Optional[str | Path] = None,
    dataset_records: Optional[List[Dict[str, Any]]] = None,
    mode: str = "auto",
) -> PrintOutcomeSurrogate:
    """
    Return the best available surrogate.

    Modes
    -----
    mode="auto"
        If model_path points to an existing .pkl file, loads and returns a
        TrainedSurrogate. Otherwise returns a HeuristicSurrogate seeded with
        any provided dataset_records.

    mode="trained"
        Returns a fresh, unfitted TrainedSurrogate (requires scikit-learn).
        This is useful for online fitting inside the experiment loop.

    mode="heuristic"
        Always returns a HeuristicSurrogate.

    Parameters
    ----------
    model_path : path to a saved TrainedSurrogate (.pkl), optional
    dataset_records : experiment records for the heuristic uncertainty model
    """
    mode = str(mode or "auto").strip().lower()
    if mode not in ("auto", "trained", "heuristic"):
        mode = "auto"

    if mode == "heuristic":
        logger.info("Using HeuristicSurrogate (mode=heuristic).")
        return HeuristicSurrogate(dataset_records=dataset_records)

    if mode == "trained":
        try:
            logger.info("Using TrainedSurrogate (mode=trained).")
            return TrainedSurrogate()
        except Exception as exc:
            logger.warning(
                "Could not construct TrainedSurrogate (%s). Falling back to HeuristicSurrogate.",
                exc,
            )
            return HeuristicSurrogate(dataset_records=dataset_records)

    if model_path is not None:
        path = Path(model_path)
        if path.exists():
            try:
                surrogate = TrainedSurrogate.load(path)
                logger.info("Loaded TrainedSurrogate from %s", path)
                return surrogate
            except Exception as exc:
                logger.warning(
                    "Could not load TrainedSurrogate from %s (%s). "
                    "Falling back to HeuristicSurrogate.", path, exc
                )

    logger.info("Using HeuristicSurrogate (no trained model available).")
    return HeuristicSurrogate(dataset_records=dataset_records)

"""
Experiment planning tools for confidence-guided local exploration.

Four tools
----------
predict_print_outcome      Score a single (geometry, process) point.
sample_local_neighborhood  Generate candidate experiments near a base point.
select_experiment_batch    Pick a budget-constrained batch from scored candidates.
log_experiment_outcome     Record an actual print result; optionally retrain surrogate.

TOOL_SCHEMAS follows the OpenAI function-calling format so it can be passed
directly to call_openrouter_agent() when driving the loop via LLM.

Sampling details
----------------
strategy="gaussian"
    Sigma (in normalized [0,1] parameter space) = uncertainty × 0.30.
    High uncertainty → wide exploration; low uncertainty → tight refinement.
    Integer params are sampled continuously then rounded and clamped.

strategy="axis_aligned"
    For each free parameter, generates two variants: center ± δ,
    where δ = uncertainty × 0.15 × (max - min).
    Useful for systematic sensitivity checks.

Batch selection annealing
-------------------------
The exploit / explore / boundary split adapts with iteration count:
    iter  1– 3 : 20% exploit, 70% explore, 10% boundary  (cold start)
    iter  4–10 : 30% exploit, 50% explore, 20% boundary
    iter 11+   : 40% exploit, 40% explore, 20% boundary  (steady state)
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .families import ParameterSpec, get_family, has_family
from .surrogate import HeuristicSurrogate, PrintOutcomeSurrogate

logger = logging.getLogger(__name__)

# Minimum labeled records before attempting surrogate retrain
MIN_RETRAIN_SAMPLES = 10


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "predict_print_outcome",
            "description": (
                "Score a single geometry + process parameter point using the surrogate model. "
                "Returns predicted probability of print success (p_pass) and uncertainty. "
                "Call this before selecting experiments to build a scored candidate list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "geometry_params": {
                        "type": "object",
                        "description": "Geometry parameters for the family (e.g. radius_um, height_um, pitch_um).",
                        "additionalProperties": {"type": "number"},
                    },
                    "process_params": {
                        "type": "object",
                        "description": "Print process parameters (e.g. laser_power_mw, scan_speed_mm_s). Pass {} if unknown.",
                        "additionalProperties": {"type": "number"},
                    },
                    "family": {
                        "type": "string",
                        "description": "Geometry family name: cylinder, box, cone, pillar_array, or microlens_array.",
                    },
                },
                "required": ["geometry_params", "process_params", "family"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_local_neighborhood",
            "description": (
                "Generate N candidate experiments near a base point. "
                "Uncertainty controls sampling width: high uncertainty → wide spread, "
                "low uncertainty → tight refinement. "
                "Returns a list of candidate parameter sets, unscored."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "base_geometry_params": {
                        "type": "object",
                        "description": "Center of the neighborhood in geometry parameter space.",
                        "additionalProperties": {"type": "number"},
                    },
                    "base_process_params": {
                        "type": "object",
                        "description": "Center of the neighborhood in process parameter space.",
                        "additionalProperties": {"type": "number"},
                    },
                    "family": {
                        "type": "string",
                        "description": "Geometry family name.",
                    },
                    "uncertainty": {
                        "type": "number",
                        "description": "Surrogate uncertainty at the base point [0, 1]. Controls sampling width.",
                    },
                    "N": {
                        "type": "integer",
                        "description": "Number of candidate points to generate. Default 12.",
                        "default": 12,
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["gaussian", "axis_aligned"],
                        "description": "Sampling strategy. gaussian=random, axis_aligned=±δ per dimension.",
                        "default": "gaussian",
                    },
                    "vary_geometry": {
                        "type": "boolean",
                        "description": "If true, sample geometry parameters. If false, keep geometry fixed at base_geometry_params.",
                        "default": True,
                    },
                    "vary_process": {
                        "type": "boolean",
                        "description": "If true, sample process parameters. If false, keep process fixed at base_process_params.",
                        "default": False,
                    },
                    "fixed_geometry_params": {
                        "type": "array",
                        "description": "Optional list of geometry parameter names to keep fixed when vary_geometry=true.",
                        "items": {"type": "string"},
                    },
                    "process_param_specs": {
                        "type": "object",
                        "description": (
                            "Optional sampling specs for process parameters. "
                            "Format: {param_name: {min_value, max_value, default, kind}}. "
                            "kind is 'float' or 'int'. Only used when vary_process=true."
                        ),
                        "additionalProperties": {"type": "object"},
                    },
                },
                "required": ["base_geometry_params", "base_process_params", "family", "uncertainty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_experiment_batch",
            "description": (
                "Select a budget-constrained batch from a list of scored candidates. "
                "Splits the budget between exploitation (high p_pass), exploration (high uncertainty), "
                "and boundary probing (p_pass ≈ 0.5), with the split annealing over iterations "
                "(explore-heavy early, exploit-heavy late). Returns the selected subset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "description": "Scored candidates. Each must have geometry_params, process_params, family, p_pass, uncertainty.",
                        "items": {"type": "object"},
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Number of experiments to select.",
                    },
                    "iteration": {
                        "type": "integer",
                        "description": "Current iteration index (1-based). Controls explore/exploit annealing.",
                        "default": 1,
                    },
                },
                "required": ["candidates", "budget"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_experiment_outcome",
            "description": (
                "Record an actual print outcome (0=fail, 1=pass) for an experiment. "
                "Appends the record to the dataset. "
                "If the surrogate is a TrainedSurrogate and enough labeled records are available, "
                "it will be retrained on the updated dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "geometry_params": {
                        "type": "object",
                        "description": "Geometry parameters of the experiment that was fabricated.",
                        "additionalProperties": {"type": "number"},
                    },
                    "process_params": {
                        "type": "object",
                        "description": "Process parameters of the fabricated experiment.",
                        "additionalProperties": {"type": "number"},
                    },
                    "family": {"type": "string"},
                    "predicted_p_pass": {
                        "type": "number",
                        "description": "Surrogate p_pass prediction made before fabrication.",
                    },
                    "predicted_uncertainty": {
                        "type": "number",
                        "description": "Surrogate uncertainty prediction made before fabrication.",
                    },
                    "actual_outcome": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "Observed print outcome: 1=pass, 0=fail.",
                    },
                    "iteration_id": {
                        "type": "string",
                        "description": "Iteration identifier for traceability.",
                    },
                },
                "required": [
                    "geometry_params", "process_params", "family",
                    "predicted_p_pass", "predicted_uncertainty", "actual_outcome",
                ],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def predict_print_outcome(
    *,
    geometry_params: Dict[str, float],
    process_params: Dict[str, float],
    family: str,
    surrogate: Optional[PrintOutcomeSurrogate] = None,
    dataset_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Score a single point using the surrogate.

    Returns
    -------
    {
        "geometry_params": ...,
        "process_params": ...,
        "family": ...,
        "p_pass": float,
        "uncertainty": float,
        "surrogate_type": "heuristic" | "trained",
    }
    """
    if surrogate is None:
        surrogate = HeuristicSurrogate(dataset_records=dataset_records or [])

    result = surrogate.predict(
        geometry_params=geometry_params,
        process_params=process_params,
        family=family,
    )
    return {
        "geometry_params": geometry_params,
        "process_params": process_params,
        "family": family,
        "p_pass": result["p_pass"],
        "uncertainty": result["uncertainty"],
        "surrogate_type": "trained" if surrogate.is_trained() else "heuristic",
    }


def sample_local_neighborhood(
    *,
    base_geometry_params: Dict[str, float],
    base_process_params: Dict[str, float],
    family: str,
    uncertainty: float,
    N: int = 12,
    strategy: str = "gaussian",
    vary_geometry: bool = True,
    vary_process: bool = False,
    fixed_geometry_params: Optional[List[str]] = None,
    process_param_specs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate N candidate experiments near a base point.

    Geometry parameters are sampled within family bounds (unless vary_geometry=False).
    Process parameters are sampled within provided bounds (process_param_specs)
    if vary_process=True; otherwise they are carried through unchanged.

    Returns
    -------
    List of dicts: {geometry_params, process_params, family}
    """
    uncertainty = float(max(0.0, min(1.0, uncertainty)))
    N = max(1, int(N))
    rng = random.Random(seed)

    known = has_family(family)
    fam = get_family(family)

    fixed_geometry = {str(k).strip() for k in (fixed_geometry_params or []) if str(k).strip()}

    if vary_geometry:
        if known:
            geometry_specs = fam.parameters
        else:
            # Build synthetic specs from the base point values
            geometry_specs = {
                k: _synthetic_spec(k, float(v))
                for k, v in base_geometry_params.items()
            }
    else:
        geometry_specs = {}

    if fixed_geometry and geometry_specs:
        geometry_specs = {k: v for k, v in geometry_specs.items() if k not in fixed_geometry}

    process_specs = _build_process_specs(
        base_process_params=base_process_params,
        process_param_specs=process_param_specs,
    ) if vary_process else {}

    # Ensure base_process_params is populated with defaults for sampled specs.
    base_process_params = dict(base_process_params)
    for name, spec in process_specs.items():
        if name not in base_process_params:
            base_process_params[name] = float(getattr(spec, "default", 0.0))

    if strategy == "axis_aligned":
        return _axis_aligned_sample(
            base_geometry_params=base_geometry_params,
            base_process_params=base_process_params,
            family=family,
            geometry_specs=geometry_specs,
            process_specs=process_specs,
            vary_geometry=vary_geometry,
            vary_process=vary_process,
            uncertainty=uncertainty,
            N=N,
            rng=rng,
        )

    return _gaussian_sample(
        base_geometry_params=base_geometry_params,
        base_process_params=base_process_params,
        family=family,
        geometry_specs=geometry_specs,
        process_specs=process_specs,
        vary_geometry=vary_geometry,
        vary_process=vary_process,
        uncertainty=uncertainty,
        N=N,
        rng=rng,
    )


def select_experiment_batch(
    *,
    candidates: List[Dict[str, Any]],
    budget: int,
    iteration: int = 1,
) -> List[Dict[str, Any]]:
    """
    Select a budget-constrained batch from scored candidates.

    Allocation annealing:
        iter  1– 3 : 20% exploit / 70% explore / 10% boundary
        iter  4–10 : 30% exploit / 50% explore / 20% boundary
        iter 11+   : 40% exploit / 40% explore / 20% boundary

    Returns
    -------
    List of selected candidate dicts (with added "selection_reason" key).
    """
    if not candidates:
        return []
    budget = max(1, int(budget))
    n_exploit, n_explore, n_boundary = _annealed_allocation(iteration, budget)

    # Sort into three ranked lists
    by_p_pass = sorted(candidates, key=lambda c: float(c.get("p_pass", 0.0)), reverse=True)
    by_uncertainty = sorted(candidates, key=lambda c: float(c.get("uncertainty", 0.0)), reverse=True)
    by_boundary = sorted(candidates, key=lambda c: abs(float(c.get("p_pass", 0.5)) - 0.5))

    selected: List[Dict[str, Any]] = []
    seen_keys: set = set()

    def _add(candidate: Dict[str, Any], reason: str) -> bool:
        key = _candidate_key(candidate)
        if key in seen_keys:
            return False
        seen_keys.add(key)
        entry = dict(candidate)
        entry["selection_reason"] = reason
        selected.append(entry)
        return True

    for c in by_p_pass:
        if len([s for s in selected if s.get("selection_reason") == "exploit"]) >= n_exploit:
            break
        _add(c, "exploit")

    for c in by_uncertainty:
        if len([s for s in selected if s.get("selection_reason") == "explore"]) >= n_explore:
            break
        _add(c, "explore")

    for c in by_boundary:
        if len([s for s in selected if s.get("selection_reason") == "boundary"]) >= n_boundary:
            break
        _add(c, "boundary")

    # Fill remaining budget if allocation left gaps
    for c in by_p_pass:
        if len(selected) >= budget:
            break
        _add(c, "fill")

    return selected[:budget]


def log_experiment_outcome(
    *,
    geometry_params: Dict[str, float],
    process_params: Dict[str, float],
    family: str,
    predicted_p_pass: float,
    predicted_uncertainty: float,
    actual_outcome: int,
    iteration_id: str = "",
    dataset: Any = None,
    surrogate: Optional[PrintOutcomeSurrogate] = None,
) -> Dict[str, Any]:
    """
    Append an actual print outcome to the dataset.

    Retrains surrogate if it is a TrainedSurrogate and
    the dataset has >= MIN_RETRAIN_SAMPLES labeled records.

    Returns
    -------
    {
        "record_id": str,
        "records_in_dataset": int,
        "surrogate_retrained": bool,
    }
    """
    from .geometry import compute_geometric_features, generate_parametric_geometry

    geometry = generate_parametric_geometry(family, geometry_params, source="experiment")
    features = compute_geometric_features(geometry)

    record_id = _make_record_id(geometry_params, process_params, family)

    record: Dict[str, Any] = {
        "record_id": record_id,
        "iteration_id": iteration_id or "",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "geometry_parameters": geometry_params,
        "process_params": process_params,
        "features": features,
        "prediction": {
            "predicted_p_pass": round(float(predicted_p_pass), 6),
            "predicted_uncertainty": round(float(predicted_uncertainty), 6),
            "risk_score": round(1.0 - float(predicted_p_pass), 6),
            "success_probability": round(float(predicted_p_pass), 6),
        },
        "uncertainty": {
            "uncertainty_score": round(float(predicted_uncertainty), 6),
        },
        "evaluation": {
            "actual_success": bool(actual_outcome),
            "actual_success_probability": float(actual_outcome),
            "evaluation_mode": "real",
        },
    }

    retrained = False
    if dataset is not None:
        dataset.append_batch([record])
        retrained = _maybe_retrain(surrogate, dataset)

    return {
        "record_id": record_id,
        "records_in_dataset": len(dataset.records) if dataset else 1,
        "surrogate_retrained": retrained,
    }


# ---------------------------------------------------------------------------
# Internal helpers — sampling
# ---------------------------------------------------------------------------

def _gaussian_sample(
    *,
    base_geometry_params: Dict[str, float],
    base_process_params: Dict[str, float],
    family: str,
    geometry_specs: Dict[str, Any],
    process_specs: Dict[str, Any],
    vary_geometry: bool,
    vary_process: bool,
    uncertainty: float,
    N: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Sample N points from a Gaussian centered at the provided base params."""
    sigma_norm = uncertainty * 0.30  # sigma in normalized [0,1] space
    candidates = []
    seen: set = set()

    attempts = 0
    while len(candidates) < N and attempts < N * 10:
        attempts += 1
        sampled_geometry = (
            _sample_param_dict_gaussian(
                base=base_geometry_params,
                specs=geometry_specs,
                sigma_norm=sigma_norm,
                rng=rng,
            ) if vary_geometry else dict(base_geometry_params)
        )
        sampled_process = (
            _sample_param_dict_gaussian(
                base=base_process_params,
                specs=process_specs,
                sigma_norm=sigma_norm,
                rng=rng,
            ) if vary_process else dict(base_process_params)
        )

        key = _candidate_key({"geometry_params": sampled_geometry, "process_params": sampled_process})
        if key not in seen:
            seen.add(key)
            candidates.append({
                "geometry_params": sampled_geometry,
                "process_params": sampled_process,
                "family": family,
            })

    return candidates


def _axis_aligned_sample(
    *,
    base_geometry_params: Dict[str, float],
    base_process_params: Dict[str, float],
    family: str,
    geometry_specs: Dict[str, Any],
    process_specs: Dict[str, Any],
    vary_geometry: bool,
    vary_process: bool,
    uncertainty: float,
    N: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Generate ±δ variants per parameter dimension."""
    delta_frac = uncertainty * 0.15  # fraction of param range
    candidates = []
    seen: set = set()

    if vary_geometry:
        for name, spec in geometry_specs.items():
            lo, hi = spec.min_value, spec.max_value
            span = max(hi - lo, 1e-6)
            delta = span * delta_frac
            center = float(base_geometry_params.get(name, spec.default))

            for offset in (-delta, +delta):
                geo = dict(base_geometry_params)
                geo[name] = _clamp_value(spec, center + offset)

                proc = dict(base_process_params)
                key = _candidate_key({"geometry_params": geo, "process_params": proc})
                if key not in seen:
                    seen.add(key)
                    candidates.append({"geometry_params": geo, "process_params": proc, "family": family})
                    if len(candidates) >= N:
                        return candidates

    if vary_process:
        for name, spec in process_specs.items():
            lo, hi = spec.min_value, spec.max_value
            span = max(hi - lo, 1e-6)
            delta = span * delta_frac
            center = float(base_process_params.get(name, spec.default))

            for offset in (-delta, +delta):
                geo = dict(base_geometry_params)
                proc = dict(base_process_params)
                proc[name] = _clamp_value(spec, center + offset)

                key = _candidate_key({"geometry_params": geo, "process_params": proc})
                if key not in seen:
                    seen.add(key)
                    candidates.append({"geometry_params": geo, "process_params": proc, "family": family})
                    if len(candidates) >= N:
                        return candidates

    # Pad with Gaussian if axis_aligned didn't fill budget
    if len(candidates) < N:
        extras = _gaussian_sample(
            base_geometry_params=base_geometry_params,
            base_process_params=base_process_params,
            family=family,
            geometry_specs=geometry_specs,
            process_specs=process_specs,
            vary_geometry=vary_geometry,
            vary_process=vary_process,
            uncertainty=uncertainty,
            N=N - len(candidates),
            rng=rng,
        )
        seen_keys = {_candidate_key(c) for c in candidates}
        for extra in extras:
            if _candidate_key(extra) not in seen_keys:
                candidates.append(extra)

    return candidates[:N]


# ---------------------------------------------------------------------------
# Internal helpers — batch selection
# ---------------------------------------------------------------------------

def _annealed_allocation(iteration: int, budget: int) -> Tuple[int, int, int]:
    """Return (n_exploit, n_explore, n_boundary) for the given iteration."""
    if iteration <= 3:
        fracs = (0.20, 0.70, 0.10)
    elif iteration <= 10:
        fracs = (0.30, 0.50, 0.20)
    else:
        fracs = (0.40, 0.40, 0.20)

    n_exploit = max(0, round(budget * fracs[0]))
    n_explore = max(0, round(budget * fracs[1]))
    n_boundary = max(0, budget - n_exploit - n_explore)
    return n_exploit, n_explore, n_boundary


def _candidate_key(candidate: Dict[str, Any]) -> str:
    geo = candidate.get("geometry_params") or {}
    proc = candidate.get("process_params") or {}
    return _geo_key(geo) + "||" + _geo_key(proc)


def _geo_key(params: Dict[str, Any]) -> str:
    return "|".join(f"{k}={round(float(v), 4)}" for k, v in sorted(params.items()))


# ---------------------------------------------------------------------------
# Internal helpers — surrogate retrain
# ---------------------------------------------------------------------------

def _maybe_retrain(
    surrogate: Optional[PrintOutcomeSurrogate],
    dataset: Any,
) -> bool:
    """Retrain TrainedSurrogate if it exists and dataset has enough labeled data."""
    from .surrogate import TrainedSurrogate

    if surrogate is None:
        return False

    # HeuristicSurrogate: update its records for uncertainty estimates
    if not isinstance(surrogate, TrainedSurrogate):
        if hasattr(surrogate, "update_records"):
            surrogate.update_records(dataset.records)
        return False

    labeled = [
        r for r in dataset.records
        if isinstance(r.get("evaluation"), dict)
        and r["evaluation"].get("actual_success") is not None
    ]
    if len(labeled) < MIN_RETRAIN_SAMPLES:
        logger.info(
            "Skipping surrogate retrain: only %d labeled records (need %d).",
            len(labeled), MIN_RETRAIN_SAMPLES,
        )
        return False

    try:
        surrogate.fit(dataset.records)
        logger.info("Surrogate retrained on %d labeled records.", len(labeled))
        return True
    except Exception as exc:
        logger.warning("Surrogate retrain failed: %s", exc)
        return False


def _build_process_specs(
    *,
    base_process_params: Dict[str, float],
    process_param_specs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a {name: spec}-mapping for process parameter sampling.

    Spec payload format:
        {
          "hatch_um": {"min_value": 0.1, "max_value": 0.55, "default": 0.325, "kind": "float"},
          ...
        }
    """
    if isinstance(process_param_specs, dict) and process_param_specs:
        specs: Dict[str, Any] = {}
        for name, raw in process_param_specs.items():
            if not isinstance(raw, dict):
                continue
            try:
                min_value = float(raw.get("min_value"))
                max_value = float(raw.get("max_value"))
            except (TypeError, ValueError):
                continue
            if max_value < min_value:
                min_value, max_value = max_value, min_value
            default = raw.get("default")
            try:
                default_value = float(default) if default is not None else float(min_value + (max_value - min_value) / 2.0)
            except (TypeError, ValueError):
                default_value = float(min_value + (max_value - min_value) / 2.0)
            kind = str(raw.get("kind") or "float").strip().lower()
            if kind not in ("float", "int"):
                kind = "float"
            specs[str(name)] = ParameterSpec(
                name=str(name),
                min_value=float(min_value),
                max_value=float(max_value),
                default=float(default_value),
                kind=kind,
                importance=1.0,
            )
        if specs:
            return specs

    # Fallback: synthetic specs from the base point values
    return {k: _synthetic_spec(k, float(v)) for k, v in (base_process_params or {}).items()}


def _clamp_value(spec: Any, value: float) -> float:
    lo, hi = float(spec.min_value), float(spec.max_value)
    if getattr(spec, "kind", "float") == "int":
        value = float(int(round(float(value))))
        value = max(float(int(lo)), min(float(int(hi)), value))
    else:
        value = max(lo, min(hi, float(value)))
    return round(value, 6)


def _sample_param_dict_gaussian(
    *,
    base: Dict[str, float],
    specs: Dict[str, Any],
    sigma_norm: float,
    rng: random.Random,
) -> Dict[str, float]:
    sampled: Dict[str, float] = {}
    for name, spec in specs.items():
        lo, hi = float(spec.min_value), float(spec.max_value)
        span = max(hi - lo, 1e-6)
        center = float(base.get(name, getattr(spec, "default", 0.0)))
        center_norm = (center - lo) / span

        noise = rng.gauss(0.0, sigma_norm)
        sampled_norm = max(0.0, min(1.0, center_norm + noise))
        value = lo + sampled_norm * span
        sampled[name] = _clamp_value(spec, value)
    # Preserve any fixed dims not listed in specs
    for k, v in base.items():
        if k not in sampled:
            sampled[k] = round(float(v), 6)
    return sampled


# ---------------------------------------------------------------------------
# Internal helpers — misc
# ---------------------------------------------------------------------------

def _make_record_id(
    geometry_params: Dict[str, float],
    process_params: Dict[str, float],
    family: str,
) -> str:
    payload = family + _geo_key(geometry_params) + _geo_key(process_params)
    return "exp_" + hashlib.sha256(payload.encode()).hexdigest()[:10]


class _SyntheticSpec:
    """Minimal stand-in for ParameterSpec when family is unknown."""
    def __init__(self, name: str, center: float) -> None:
        self.name = name
        self.min_value = max(center * 0.1, 1e-3)
        self.max_value = center * 2.0
        self.default = center
        self.kind = "float"


def _synthetic_spec(name: str, center: float) -> _SyntheticSpec:
    return _SyntheticSpec(name, max(abs(center), 0.01))

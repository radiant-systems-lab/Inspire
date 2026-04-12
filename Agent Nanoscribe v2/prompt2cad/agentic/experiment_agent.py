"""
Confidence-guided experiment planning agent.

Implements the agent loop from the proposal:

    FOR each iteration:
        1. Seed: find best known point (argmax p_pass from dataset)
        2. Predict: score the seed with the surrogate
        3. Sample: generate N candidates near the seed
        4. Score: predict_print_outcome for each candidate
        5. Select: select_experiment_batch (budget K, annealed allocation)
        6. Fabricate: call evaluate_fn (real or simulated)
        7. Log: log_experiment_outcome for each result
        8. Retrain: if surrogate is TrainedSurrogate and enough data
        9. Repeat until confidence_threshold is reached or max_iterations hit

Stopping criterion
------------------
The loop stops when the best p_pass in the dataset exceeds
`confidence_threshold` (default 0.85) with uncertainty < 0.30.
This avoids running forever on a solved problem.

Usage (simulated, no real printer)
-----------------------------------
    from prompt2cad.agentic.experiment_agent import ExperimentPlanningAgent
    from prompt2cad.agentic.surrogate import get_surrogate
    from prompt2cad.agentic.memory import ExperimentDataset

    dataset = ExperimentDataset("data/my_run.jsonl")
    surrogate = get_surrogate()  # HeuristicSurrogate until model.pkl exists

    agent = ExperimentPlanningAgent(
        surrogate=surrogate,
        dataset=dataset,
        evaluate_fn=simulated_evaluate,   # swap for real hardware callback
    )
    result = agent.run(iterations=5, budget=4, family="pillar_array")

Usage (human-in-the-loop)
--------------------------
    agent = ExperimentPlanningAgent(
        surrogate=surrogate,
        dataset=dataset,
        evaluate_fn=None,   # None triggers pause-and-log mode
    )
    agent.run(iterations=5, budget=4, family="pillar_array")
    # → agent prints the selected batch, then waits for human input
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .experiment_tools import (
    log_experiment_outcome,
    predict_print_outcome,
    sample_local_neighborhood,
    select_experiment_batch,
)
from .families import get_family, has_family
from .memory import ExperimentDataset
from .surrogate import HeuristicSurrogate, PrintOutcomeSurrogate, get_surrogate

logger = logging.getLogger(__name__)

# Stopping criterion: stop when best p_pass exceeds this AND uncertainty is low
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_MAX_UNCERTAINTY_AT_STOP = 0.30

EvaluateFn = Callable[[List[Dict[str, Any]]], List[int]]
"""
Type of the evaluation callback.

    evaluate_fn(batch: List[dict]) -> List[int]

Each dict has keys: geometry_params, process_params, family, p_pass, uncertainty.
Returns a parallel list of outcomes: 1 = pass, 0 = fail.
"""


class ExperimentPlanningAgent:
    """
    Confidence-guided experiment planning agent.

    Parameters
    ----------
    surrogate : PrintOutcomeSurrogate
        The prediction model. Pass get_surrogate() for automatic selection.
    dataset : ExperimentDataset
        Persistent storage for experiment records.
    evaluate_fn : callable or None
        Called with the selected batch; returns a list of 0/1 outcomes.
        If None, the agent pauses and prompts for human input.
    neighborhood_size : int
        Number of candidates to sample per iteration (default 16).
    confidence_threshold : float
        Stop when best p_pass exceeds this with low uncertainty.
    sampling_strategy : str
        "gaussian" or "axis_aligned".
    trace_path : str or Path, optional
        If provided, each iteration's trace is appended as JSONL.
    """

    def __init__(
        self,
        surrogate: Optional[PrintOutcomeSurrogate] = None,
        dataset: Optional[ExperimentDataset] = None,
        evaluate_fn: Optional[EvaluateFn] = None,
        *,
        neighborhood_size: int = 16,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        sampling_strategy: str = "gaussian",
        vary_geometry: bool = True,
        vary_process: bool = False,
        fixed_geometry_params: Optional[List[str]] = None,
        process_param_specs: Optional[Dict[str, Any]] = None,
        trace_path: Optional[str | Path] = None,
    ) -> None:
        self.surrogate = surrogate or get_surrogate()
        self.dataset = dataset or ExperimentDataset()
        self.evaluate_fn = evaluate_fn
        self.neighborhood_size = neighborhood_size
        self.confidence_threshold = confidence_threshold
        self.sampling_strategy = sampling_strategy
        self.vary_geometry = bool(vary_geometry)
        self.vary_process = bool(vary_process)
        self.fixed_geometry_params = [str(x).strip() for x in (fixed_geometry_params or []) if str(x).strip()]
        self.process_param_specs = dict(process_param_specs or {}) or None
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        iterations: int = 5,
        budget: int = 4,
        family: str = "pillar_array",
        base_process_params: Optional[Dict[str, float]] = None,
        seed_geometry_params: Optional[Dict[str, float]] = None,
        fabricate: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full confidence-guided experiment loop.

        Parameters
        ----------
        iterations : int
            Maximum number of iterations to run.
        budget : int
            Number of experiments to select per iteration.
        family : str
            Geometry family to explore.
        base_process_params : dict, optional
            Fixed process parameters passed through to all experiments.
            Ignored by HeuristicSurrogate; used by TrainedSurrogate.

        Returns
        -------
        Summary dict with per-iteration traces and final stats.
        """
        base_process_params = base_process_params or {}
        history: List[Dict[str, Any]] = []
        converged = False

        for idx in range(1, iterations + 1):
            trace = self.step(
                iteration=idx,
                budget=budget,
                family=family,
                base_process_params=base_process_params,
                seed_geometry_params=seed_geometry_params,
                fabricate=fabricate,
            )
            history.append(trace)

            if self._should_stop(trace):
                logger.info(
                    "Stopping at iteration %d: confidence threshold reached.", idx
                )
                converged = True
                break

        return {
            "iterations": history,
            "converged": converged,
            "final_stats": self._compute_final_stats(family),
        }

    def step(
        self,
        *,
        iteration: int,
        budget: int,
        family: str,
        base_process_params: Dict[str, float],
        seed_geometry_params: Optional[Dict[str, float]] = None,
        fabricate: bool = True,
    ) -> Dict[str, Any]:
        """Run a single iteration of the experiment planning loop."""
        t0 = time.perf_counter()
        iter_id = f"iter_{iteration:03d}"

        # 1. Seed — best known point or family defaults
        seed_params = dict(seed_geometry_params) if seed_geometry_params else self._best_known_params(family)

        # 2. Score the seed
        seed_score = predict_print_outcome(
            geometry_params=seed_params,
            process_params=base_process_params,
            family=family,
            surrogate=self.surrogate,
            dataset_records=self.dataset.records,
        )

        # 3. Sample neighborhood
        candidates_unscored = sample_local_neighborhood(
            base_geometry_params=seed_params,
            base_process_params=base_process_params,
            family=family,
            uncertainty=seed_score["uncertainty"],
            N=self.neighborhood_size,
            strategy=self.sampling_strategy,
            vary_geometry=self.vary_geometry,
            vary_process=self.vary_process,
            fixed_geometry_params=self.fixed_geometry_params,
            process_param_specs=self.process_param_specs,
        )

        # 4. Score all candidates
        candidates_scored = [
            predict_print_outcome(
                geometry_params=c["geometry_params"],
                process_params=c["process_params"],
                family=family,
                surrogate=self.surrogate,
                dataset_records=self.dataset.records,
            )
            for c in candidates_unscored
        ]

        # 5. Select batch
        batch = select_experiment_batch(
            candidates=candidates_scored,
            budget=budget,
            iteration=iteration,
        )

        # 6. Fabricate (simulated or human-in-the-loop) — optionally skipped
        if not fabricate:
            elapsed = round(time.perf_counter() - t0, 4)
            trace: Dict[str, Any] = {
                "iteration": iteration,
                "iteration_id": iter_id,
                "elapsed_s": elapsed,
                "seed_p_pass": round(seed_score["p_pass"], 4),
                "seed_uncertainty": round(seed_score["uncertainty"], 4),
                "candidates_sampled": len(candidates_unscored),
                "candidates_scored": len(candidates_scored),
                "budget": budget,
                "batch_selected": len(batch),
                "actual_pass_rate": None,
                "mean_predicted_p_pass": _mean([c["p_pass"] for c in batch]),
                "mean_predicted_uncertainty": _mean([c["uncertainty"] for c in batch]),
                "allocation": _describe_allocation(batch),
                "surrogate_type": "trained" if self.surrogate.is_trained() else "heuristic",
                "dataset_size": len(self.dataset.records),
                "experiments": batch,
                "candidates": candidates_scored,
                "fabricated": False,
            }
            if self.trace_path:
                self._append_trace(trace)
            return trace

        outcomes = self._get_outcomes(batch, iteration)

        # 7. Log all outcomes
        log_results = []
        for experiment, outcome in zip(batch, outcomes):
            result = log_experiment_outcome(
                geometry_params=experiment["geometry_params"],
                process_params=experiment["process_params"],
                family=family,
                predicted_p_pass=experiment["p_pass"],
                predicted_uncertainty=experiment["uncertainty"],
                actual_outcome=outcome,
                iteration_id=iter_id,
                dataset=self.dataset,
                surrogate=self.surrogate,
            )
            log_results.append({**experiment, "actual_outcome": outcome, **result})

        elapsed = round(time.perf_counter() - t0, 4)
        actual_pass_rate = (
            sum(outcomes) / len(outcomes) if outcomes else 0.0
        )

        trace: Dict[str, Any] = {
            "iteration": iteration,
            "iteration_id": iter_id,
            "elapsed_s": elapsed,
            "seed_p_pass": round(seed_score["p_pass"], 4),
            "seed_uncertainty": round(seed_score["uncertainty"], 4),
            "candidates_sampled": len(candidates_unscored),
            "candidates_scored": len(candidates_scored),
            "budget": budget,
            "batch_selected": len(batch),
            "actual_pass_rate": round(actual_pass_rate, 4),
            "mean_predicted_p_pass": _mean([c["p_pass"] for c in batch]),
            "mean_predicted_uncertainty": _mean([c["uncertainty"] for c in batch]),
            "allocation": _describe_allocation(batch),
            "surrogate_type": "trained" if self.surrogate.is_trained() else "heuristic",
            "dataset_size": len(self.dataset.records),
            "experiments": log_results,
            "fabricated": True,
        }

        if self.trace_path:
            self._append_trace(trace)

        return trace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_known_params(self, family: str) -> Dict[str, float]:
        """Return geometry params of the best known record, or family defaults."""
        family_records = [
            r for r in self.dataset.records
            if str(r.get("family")) == family
        ]
        if not family_records:
            fam = get_family(family)
            return fam.default_parameters()

        best = max(
            family_records,
            key=lambda r: float(
                (r.get("evaluation") or {}).get("actual_success_probability")
                or (r.get("prediction") or {}).get("predicted_p_pass")
                or 0.0
            ),
        )
        return dict(best.get("geometry_parameters") or best.get("geometry_params") or {})

    def _get_outcomes(
        self,
        batch: List[Dict[str, Any]],
        iteration: int,
    ) -> List[int]:
        """Obtain actual outcomes: simulated, LLM-evaluate, or human input."""
        if self.evaluate_fn is not None:
            return list(self.evaluate_fn(batch))
        return self._human_input(batch, iteration)

    def _human_input(
        self,
        batch: List[Dict[str, Any]],
        iteration: int,
    ) -> List[int]:
        """Pause and collect actual outcomes from the user."""
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration} — BATCH FOR FABRICATION")
        print(f"{'='*60}")
        for i, exp in enumerate(batch, 1):
            print(f"\n  [{i}] {exp['family']}: {exp['geometry_params']}")
            print(f"       predicted p_pass={exp['p_pass']:.3f}  "
                  f"uncertainty={exp['uncertainty']:.3f}  "
                  f"reason={exp.get('selection_reason', '?')}")

        print(f"\nEnter outcomes for {len(batch)} experiments (space-separated 0/1):")
        while True:
            raw = input("  > ").strip()
            parts = raw.split()
            if len(parts) == len(batch) and all(p in ("0", "1") for p in parts):
                return [int(p) for p in parts]
            print(f"  Expected {len(batch)} values, each 0 or 1. Try again.")

    def _should_stop(self, trace: Dict[str, Any]) -> bool:
        """Check stopping criterion: high p_pass AND low uncertainty."""
        family_records = [
            r for r in self.dataset.records
            if isinstance(r.get("evaluation"), dict)
            and r["evaluation"].get("actual_success") is not None
        ]
        if not family_records:
            return False

        # Find best actual-pass record and check surrogate score there
        passed = [r for r in family_records if r["evaluation"]["actual_success"]]
        if not passed:
            return False

        best = max(
            passed,
            key=lambda r: float(
                (r.get("prediction") or {}).get("predicted_p_pass", 0.0)
            ),
        )
        pred = (best.get("prediction") or {})
        p_pass = float(pred.get("predicted_p_pass", 0.0))
        uncertainty = float(
            (best.get("uncertainty") or {}).get("uncertainty_score", 1.0)
        )
        return p_pass >= self.confidence_threshold and uncertainty <= DEFAULT_MAX_UNCERTAINTY_AT_STOP

    def _compute_final_stats(self, family: str) -> Dict[str, Any]:
        """Aggregate stats across the full dataset for this family."""
        records = [r for r in self.dataset.records if str(r.get("family")) == family]
        labeled = [
            r for r in records
            if isinstance(r.get("evaluation"), dict)
            and r["evaluation"].get("actual_success") is not None
        ]
        if not labeled:
            return {"family": family, "labeled_count": 0}

        pass_rate = sum(
            int(bool(r["evaluation"]["actual_success"])) for r in labeled
        ) / len(labeled)

        predicted = [
            float((r.get("prediction") or {}).get("predicted_p_pass", 0.5))
            for r in labeled
        ]
        uncertainties = [
            float((r.get("uncertainty") or {}).get("uncertainty_score", 1.0))
            for r in labeled
        ]

        return {
            "family": family,
            "labeled_count": len(labeled),
            "actual_pass_rate": round(pass_rate, 4),
            "mean_predicted_p_pass": _mean(predicted),
            "mean_uncertainty": _mean(uncertainties),
            "surrogate_type": "trained" if self.surrogate.is_trained() else "heuristic",
        }

    def _append_trace(self, trace: Dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=True, default=str) + "\n")


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def build_agent(
    *,
    dataset_path: Optional[str] = None,
    model_path: Optional[str] = None,
    surrogate_mode: str = "auto",
    evaluate_fn: Optional[EvaluateFn] = None,
    neighborhood_size: int = 16,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    sampling_strategy: str = "gaussian",
    vary_geometry: bool = True,
    vary_process: bool = False,
    fixed_geometry_params: Optional[List[str]] = None,
    process_param_specs: Optional[Dict[str, Any]] = None,
    trace_path: Optional[str] = None,
) -> ExperimentPlanningAgent:
    """
    Construct an ExperimentPlanningAgent with sensible defaults.

    Parameters
    ----------
    dataset_path : path to the JSONL experiment dataset (created if absent)
    model_path   : path to a saved TrainedSurrogate .pkl (uses heuristic if absent)
    evaluate_fn  : outcome callback; None = human-in-the-loop
    """
    dataset = ExperimentDataset(dataset_path)
    surrogate = get_surrogate(
        model_path=model_path,
        dataset_records=dataset.records,
        mode=surrogate_mode,
    )
    return ExperimentPlanningAgent(
        surrogate=surrogate,
        dataset=dataset,
        evaluate_fn=evaluate_fn,
        neighborhood_size=neighborhood_size,
        confidence_threshold=confidence_threshold,
        sampling_strategy=sampling_strategy,
        vary_geometry=vary_geometry,
        vary_process=vary_process,
        fixed_geometry_params=fixed_geometry_params,
        process_param_specs=process_param_specs,
        trace_path=trace_path,
    )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _describe_allocation(batch: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for exp in batch:
        reason = str(exp.get("selection_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts

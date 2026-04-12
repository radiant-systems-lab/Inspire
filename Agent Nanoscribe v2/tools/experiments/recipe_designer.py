"""
RecipeExperimentDesigner
------------------------
Inverse model: given a batch of candidate geometries, design the next
print experiment by selecting (geometry, recipe) pairs that maximize
expected information gain.

Scoring
-------
Each candidate (geometry_id, slice_um, hatch_um) is scored by:

    score = α * uncertainty  +  β * boundary_proximity  +  γ * p_pass

where:
    uncertainty         → how much the model doesn't know (exploration)
    boundary_proximity  → |p_pass - 0.5|  inverted  (near decision boundary)
    p_pass              → predicted success probability (exploitation)

The weights α, β, γ are controlled by the `strategy` parameter:
    "uncertainty"   α=0.6, β=0.3, γ=0.1  — maximise learning
    "exploit"       α=0.1, β=0.2, γ=0.7  — maximise pass rate
    "balanced"      α=0.4, β=0.3, γ=0.3  — default

The designer then selects the top-`budget` candidates, enforcing geometric
diversity (no two slots go to the same geometry unless budget > n_geometries).

Usage
-----
    from tools.forward_model.step_predictor import StepForwardModel
    from tools.experiments.recipe_designer import RecipeExperimentDesigner

    model    = StepForwardModel.load()
    designer = RecipeExperimentDesigner(model)

    candidates = [
        {"geometry_id": "abc123", "geometry_metrics": {...}},
        {"geometry_id": "def456", "geometry_metrics": {...}},
    ]
    plan = designer.design_experiment(candidates, budget=9)
    # → list of dicts, each with: geometry_id, slice_um, hatch_um,
    #   p_pass, uncertainty, score, rationale, priority
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Recipe grid ────────────────────────────────────────────────────────────────
# Candidate recipes to consider for each geometry.
# Extend this as more recipe data becomes available.
CANDIDATE_RECIPES: List[tuple[float, float]] = [
    # (slice_um, hatch_um)
    (0.10, 0.10),
    (0.20, 0.20),
    (0.20, 0.30),
    (0.30, 0.20),
    (0.30, 0.30),
    (0.20, 0.40),
    (0.40, 0.20),
    (0.30, 0.40),
    (0.40, 0.30),
    (0.40, 0.40),
]

# Strategy weight presets
_STRATEGIES: Dict[str, tuple[float, float, float]] = {
    "uncertainty": (0.6, 0.3, 0.1),   # α, β, γ
    "exploit":     (0.1, 0.2, 0.7),
    "balanced":    (0.4, 0.3, 0.3),
}


class RecipeExperimentDesigner:
    """
    Designs the next print experiment batch for a set of candidate geometries.
    Uses the forward model to score each (geometry, recipe) pair.
    """

    def __init__(self, forward_model: Any) -> None:
        """
        Parameters
        ----------
        forward_model : StepForwardModel (or any object with .predict())
        """
        self._model = forward_model

    def design_experiment(
        self,
        candidates: List[Dict[str, Any]],
        budget: int = 9,
        strategy: str = "balanced",
        recipes: Optional[List[tuple[float, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select the best (geometry, recipe) pairs to print next.

        Parameters
        ----------
        candidates : list of dicts, each with:
            - geometry_id    : str
            - geometry_metrics : dict  (from compute_geometry_metrics / STEP pipeline)
        budget : int
            Number of print slots available (e.g. one plate = 9 spots).
        strategy : "uncertainty" | "exploit" | "balanced"
        recipes : optional override of CANDIDATE_RECIPES

        Returns
        -------
        List of dicts (sorted by priority, highest first):
            geometry_id, slice_um, hatch_um, p_pass, uncertainty,
            score, priority, rationale
        """
        if not candidates:
            return []

        recipe_grid = recipes or CANDIDATE_RECIPES
        alpha, beta, gamma = _STRATEGIES.get(strategy, _STRATEGIES["balanced"])

        # Score every (geometry, recipe) pair
        all_scored: List[Dict[str, Any]] = []
        for cand in candidates:
            gid     = str(cand.get("geometry_id") or cand.get("model_id") or "unknown")
            metrics = dict(cand.get("geometry_metrics") or {})

            for slice_um, hatch_um in recipe_grid:
                pred = self._model.predict(metrics, slice_um, hatch_um)
                p    = pred["p_pass"]
                u    = pred["uncertainty"]
                bp   = 1.0 - abs(p - 0.5) * 2.0   # boundary proximity: 1 at p=0.5, 0 at p=0 or 1
                score = alpha * u + beta * bp + gamma * p

                all_scored.append({
                    "geometry_id": gid,
                    "slice_um":    slice_um,
                    "hatch_um":    hatch_um,
                    "p_pass":      round(p, 4),
                    "uncertainty": round(u, 4),
                    "boundary_proximity": round(bp, 4),
                    "score":       round(score, 4),
                    "model_type":  pred.get("model_type", "unknown"),
                    "features_approximated": pred.get("features_approximated", False),
                    "rationale":   _build_rationale(p, u, bp, strategy, slice_um, hatch_um),
                })

        # Sort by score descending
        all_scored.sort(key=lambda x: x["score"], reverse=True)

        # Select top-budget, enforcing geometry diversity:
        # Fill each slot with highest-scoring candidate, preferring unrepresented geometries first
        selected: List[Dict[str, Any]] = []
        seen_geoms: set[str] = set()

        # First pass: one best per geometry (up to budget)
        for row in all_scored:
            if row["geometry_id"] not in seen_geoms:
                seen_geoms.add(row["geometry_id"])
                selected.append({**row, "priority": len(selected) + 1})
            if len(selected) >= budget:
                break

        # Second pass: fill remaining slots with next-best overall
        if len(selected) < budget:
            selected_keys = {(r["geometry_id"], r["slice_um"], r["hatch_um"]) for r in selected}
            for row in all_scored:
                key = (row["geometry_id"], row["slice_um"], row["hatch_um"])
                if key not in selected_keys:
                    selected.append({**row, "priority": len(selected) + 1})
                    selected_keys.add(key)
                if len(selected) >= budget:
                    break

        return selected

    def recipe_surface(
        self,
        geometry_id: str,
        geometry_metrics: Dict[str, Any],
        recipes: Optional[List[tuple[float, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return predicted pass probability for all recipes for a single geometry.
        Useful for visualisation (the heatmap in the notebook).
        """
        recipe_grid = recipes or CANDIDATE_RECIPES
        rows = []
        for slice_um, hatch_um in recipe_grid:
            pred = self._model.predict(geometry_metrics, slice_um, hatch_um)
            rows.append({
                "geometry_id": geometry_id,
                "slice_um":    slice_um,
                "hatch_um":    hatch_um,
                "p_pass":      round(pred["p_pass"], 4),
                "uncertainty": round(pred["uncertainty"], 4),
            })
        return sorted(rows, key=lambda x: x["p_pass"], reverse=True)

    def summarise_plan(self, plan: List[Dict[str, Any]]) -> str:
        """Return a human-readable summary of a design_experiment() result."""
        if not plan:
            return "No experiments planned."
        lines = [f"Experiment plan ({len(plan)} slots, strategy inferred from scores):"]
        for row in plan:
            approx_flag = " [approx features]" if row.get("features_approximated") else ""
            lines.append(
                f"  [{row['priority']:2d}] {row['geometry_id'][:16]:16s} "
                f"slice={row['slice_um']:.2f} hatch={row['hatch_um']:.2f}  "
                f"p_pass={row['p_pass']:.0%}  uncertainty={row['uncertainty']:.2f}  "
                f"score={row['score']:.3f}{approx_flag}"
            )
        return "\n".join(lines)


# ── Top-level convenience functions ───────────────────────────────────────────

def design_print_experiment(
    candidate_geometries: List[Dict[str, Any]],
    budget: int = 9,
    strategy: str = "balanced",
    model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function: load the model and design an experiment in one call.

    Parameters
    ----------
    candidate_geometries : list of {geometry_id, geometry_metrics} dicts
    budget : number of print slots
    strategy : "uncertainty" | "exploit" | "balanced"
    model_path : override path to saved .pkl model

    Returns
    -------
    Ranked experiment list (same as RecipeExperimentDesigner.design_experiment)
    """
    from tools.forward_model.step_predictor import StepForwardModel
    model_path_arg = model_path or None
    model = StepForwardModel.load(model_path_arg) if model_path_arg else StepForwardModel.load()
    designer = RecipeExperimentDesigner(model)
    return designer.design_experiment(candidate_geometries, budget=budget, strategy=strategy)


def get_best_recipe(
    geometry_metrics: Dict[str, Any],
    geometry_id: str = "query",
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function: return the single best recipe for one geometry.

    Returns
    -------
    dict: geometry_id, slice_um, hatch_um, p_pass, uncertainty, rationale
    """
    from tools.forward_model.step_predictor import StepForwardModel
    model = StepForwardModel.load(model_path) if model_path else StepForwardModel.load()
    designer = RecipeExperimentDesigner(model)
    surface = designer.recipe_surface(geometry_id, geometry_metrics)
    return surface[0] if surface else {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_rationale(
    p_pass: float,
    uncertainty: float,
    boundary_proximity: float,
    strategy: str,
    slice_um: float,
    hatch_um: float,
) -> str:
    parts: List[str] = []
    if uncertainty >= 0.5:
        parts.append(f"high uncertainty ({uncertainty:.2f}) — good for learning")
    if boundary_proximity >= 0.6:
        parts.append("near pass/fail boundary — high information value")
    if p_pass >= 0.65:
        parts.append(f"strong pass signal ({p_pass:.0%})")
    elif p_pass <= 0.25:
        parts.append(f"likely fail ({p_pass:.0%}) — useful as negative example")
    if slice_um != hatch_um:
        parts.append("off-diagonal recipe — expands independent hatch/slice coverage")
    if not parts:
        parts.append(f"p_pass={p_pass:.0%}, uncertainty={uncertainty:.2f}")
    return "; ".join(parts)

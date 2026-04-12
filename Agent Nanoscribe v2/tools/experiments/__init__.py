"""
tools/experiments
-----------------
Experiment design facade: recipe selection, planning, and active learning.

New (recipe-aware, use these):
    RecipeExperimentDesigner   — scores (geometry, recipe) pairs by expected info gain
    design_print_experiment()  — convenience: load model + design in one call
    get_best_recipe()          — convenience: best recipe for a single geometry

Legacy (geometry parameter sweeps, no recipe awareness):
    ExperimentPlanner, ExperimentExecutor, AutonomousExperimentAgent, etc.
"""

# ── New recipe-aware experiment design ────────────────────────────────────────
from .recipe_designer import (
    RecipeExperimentDesigner,
    design_print_experiment,
    get_best_recipe,
    CANDIDATE_RECIPES,
)

# ── Legacy geometry-sweep planner (backward compat) ──────────────────────────
from prompt2cad.agentic.planner import ExperimentPlanner, plan_next_experiments, design_parameter_sweep
from prompt2cad.agentic.executor import ExperimentExecutor
from prompt2cad.agentic.evaluation import SimulatedEvaluationModule, evaluate_results
from prompt2cad.agentic.memory import ExperimentDataset, update_dataset
from prompt2cad.agentic.loop import AutonomousExperimentAgent, run_autonomous_experiment_loop
from prompt2cad.agentic.experiment_agent import ExperimentPlanningAgent, build_agent

__all__ = [
    # New
    "RecipeExperimentDesigner",
    "design_print_experiment",
    "get_best_recipe",
    "CANDIDATE_RECIPES",
    # Legacy
    "ExperimentPlanner",
    "plan_next_experiments",
    "design_parameter_sweep",
    "ExperimentExecutor",
    "SimulatedEvaluationModule",
    "evaluate_results",
    "ExperimentDataset",
    "update_dataset",
    "AutonomousExperimentAgent",
    "run_autonomous_experiment_loop",
    "ExperimentPlanningAgent",
    "build_agent",
]

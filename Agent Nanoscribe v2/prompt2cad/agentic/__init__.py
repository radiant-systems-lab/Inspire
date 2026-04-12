"""Agentic experimentation modules for Prompt2CAD."""

from .canonical_geometry import normalize_agent_input, parse_canonical_geo_language
from .classifier import HeuristicPrintabilityClassifier, predict_print_success
from .evaluation import SimulatedEvaluationModule, evaluate_results
from .executor import ExperimentExecutor, run_experiment_batch
from .geometry import (
    compute_geometric_features,
    extract_geometry_parameters,
    generate_parametric_geometry,
)
from .loop import AutonomousExperimentAgent, run_autonomous_experiment_loop
from .memory import ExperimentDataset, update_dataset
from .planner import ExperimentPlanner, design_parameter_sweep, plan_next_experiments
from .template_codegen import build_template_cadquery_code
from .uncertainty import DistanceBasedUncertaintyModel, predict_uncertainty

__all__ = [
    "AutonomousExperimentAgent",
    "DistanceBasedUncertaintyModel",
    "ExperimentDataset",
    "ExperimentExecutor",
    "ExperimentPlanner",
    "HeuristicPrintabilityClassifier",
    "SimulatedEvaluationModule",
    "compute_geometric_features",
    "design_parameter_sweep",
    "evaluate_results",
    "extract_geometry_parameters",
    "generate_parametric_geometry",
    "build_template_cadquery_code",
    "normalize_agent_input",
    "parse_canonical_geo_language",
    "plan_next_experiments",
    "predict_print_success",
    "predict_uncertainty",
    "run_autonomous_experiment_loop",
    "run_experiment_batch",
    "update_dataset",
]

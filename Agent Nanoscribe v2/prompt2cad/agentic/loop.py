"""Closed-loop orchestration for autonomous experimentation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .evaluation import SimulatedEvaluationModule
from .executor import ExperimentExecutor
from .memory import ExperimentDataset, update_dataset
from .planner import ExperimentPlanner


class AutonomousExperimentAgent:
    def __init__(
        self,
        *,
        dataset_path: str | None = None,
        trace_path: str | None = None,
        planner: ExperimentPlanner | None = None,
        executor: ExperimentExecutor | None = None,
        evaluator: SimulatedEvaluationModule | None = None,
    ) -> None:
        self.dataset = ExperimentDataset(dataset_path)
        default_trace_path = self.dataset.path.with_suffix(".trace.jsonl")
        self.trace_path = Path(trace_path) if trace_path else default_trace_path
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.planner = planner or ExperimentPlanner()
        self.executor = executor or ExperimentExecutor()
        self.evaluator = evaluator or SimulatedEvaluationModule()

    def step(self, raw_request: Any = None, *, iteration_index: int | None = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        t_plan = time.perf_counter()
        plan = self.planner.plan(raw_request, self.dataset.records)
        plan_seconds = round(time.perf_counter() - t_plan, 4)

        t_exec = time.perf_counter()
        execution_batch = self.executor.run(plan, self.dataset.records)
        execute_seconds = round(time.perf_counter() - t_exec, 4)

        t_eval = time.perf_counter()
        evaluation = self.evaluator.evaluate(execution_batch)
        evaluate_seconds = round(time.perf_counter() - t_eval, 4)

        t_mem = time.perf_counter()
        dataset_summary = update_dataset(self.dataset, evaluation["records"])
        memory_seconds = round(time.perf_counter() - t_mem, 4)

        trace = {
            "iteration_index": iteration_index,
            "request_type": (plan.get("request") or {}).get("input_type"),
            "strategy": plan.get("strategy"),
            "family": (plan.get("geometry_action") or {}).get("family"),
            "candidate_count": len(plan.get("candidate_experiments") or []),
            "rationale": list(plan.get("rationale") or []),
            "timings": {
                "plan_seconds": plan_seconds,
                "execute_seconds": execute_seconds,
                "evaluate_seconds": evaluate_seconds,
                "memory_seconds": memory_seconds,
                "iteration_seconds": round(time.perf_counter() - t0, 4),
            },
            "llm_usage": _aggregate_llm_usage(execution_batch),
            "success_rate_batch": (evaluation.get("summary") or {}).get("actual_success_rate"),
            "dataset_record_count": dataset_summary.get("record_count"),
        }
        self._append_trace(trace)
        return {
            "plan": plan,
            "execution_batch": execution_batch,
            "evaluation": evaluation,
            "dataset_summary": dataset_summary,
            "trace": trace,
        }

    def run(self, raw_request: Any = None, *, iterations: int = 3) -> Dict[str, Any]:
        iterations = max(1, int(iterations))
        history = []
        for idx in range(1, iterations + 1):
            history.append(self.step(raw_request, iteration_index=idx))
        return {
            "iterations": history,
            "final_dataset_summary": self.dataset.summary(),
        }

    def _append_trace(self, payload: Dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_autonomous_experiment_loop(
    raw_request: Any = None,
    *,
    iterations: int = 3,
    dataset_path: Optional[str] = None,
    trace_path: Optional[str] = None,
) -> Dict[str, Any]:
    agent = AutonomousExperimentAgent(dataset_path=dataset_path, trace_path=trace_path)
    return agent.run(raw_request, iterations=iterations)


def _aggregate_llm_usage(execution_batch: Dict[str, Any]) -> Dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for experiment in execution_batch.get("experiments", []):
        backend = experiment.get("backend_trace") if isinstance(experiment, dict) else None
        usage = (backend or {}).get("llm_usage")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
    return totals

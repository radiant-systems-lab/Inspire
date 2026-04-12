"""Experiment execution for the agentic loop."""

from __future__ import annotations

import json
import importlib.util
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from ..config import (
    DEFAULT_GENERATOR_MODEL,
    LOGS_DIR,
    MAX_REPAIR_ATTEMPTS,
    OPENROUTER_API_KEY,
    OUTPUT_DIR,
)
from ..design_store import create_design
from ..execution.cad_executor import execute as execute_cad
from ..execution.render_utils import render_stl
from ..repair.repair_loop import repair
from ..utils import call_openrouter, get_last_openrouter_trace
from .classifier import HeuristicPrintabilityClassifier
from .geometry import compute_geometric_features, generate_parametric_geometry
from .template_codegen import build_template_cadquery_code
from .uncertainty import DistanceBasedUncertaintyModel


class ExperimentExecutor:
    def __init__(
        self,
        *,
        classifier: HeuristicPrintabilityClassifier | None = None,
        uncertainty_model: DistanceBasedUncertaintyModel | None = None,
        enable_prompt2cad_backend: bool = True,
        cad_model: str = DEFAULT_GENERATOR_MODEL,
        backend_output_dir: str | Path | None = None,
        prompt2cad_verbose: bool = False,
        use_prompt2cad_base_edit_for_structured: bool = True,
        use_template_backend_for_structured: bool = False,
        force_template_backend: bool = False,
        force_prompt2cad_base_edit: bool = False,
    ) -> None:
        self.classifier = classifier or HeuristicPrintabilityClassifier()
        self.uncertainty_model = uncertainty_model or DistanceBasedUncertaintyModel()
        self.enable_prompt2cad_backend = bool(enable_prompt2cad_backend)
        self.cad_model = cad_model
        self.prompt2cad_verbose = bool(prompt2cad_verbose)
        self.use_prompt2cad_base_edit_for_structured = bool(use_prompt2cad_base_edit_for_structured)
        self.use_template_backend_for_structured = bool(use_template_backend_for_structured)
        self.force_template_backend = bool(force_template_backend)
        self.force_prompt2cad_base_edit = bool(force_prompt2cad_base_edit)
        self.backend_output_dir = Path(backend_output_dir) if backend_output_dir else (OUTPUT_DIR / "agentic_loop")
        self.backend_output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        plan: Mapping[str, Any],
        dataset_records: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        records = [dict(record) for record in dataset_records]
        backend_mode = self._select_backend_mode(plan)
        base_prompt2cad_bundle: Dict[str, Any] | None = None
        experiments = []
        for candidate in plan.get("candidate_experiments", []):
            t_candidate = time.perf_counter()
            family = str(candidate.get("family") or plan["geometry_action"]["family"])
            geometry = generate_parametric_geometry(
                family,
                candidate.get("parameters"),
                source=str(plan["geometry_action"].get("mode") or "planner"),
                source_path=plan["geometry_action"].get("source_path"),
            )
            features = compute_geometric_features(geometry)
            prediction = self.classifier.predict(features)
            uncertainty = self.uncertainty_model.predict(
                family_name=family,
                parameters=geometry["parameters"],
                dataset_records=records,
                classifier_prediction=prediction,
            )
            experiment_id = f"exp_{uuid.uuid4().hex[:10]}"
            if backend_mode == "prompt2cad_base_edit":
                if base_prompt2cad_bundle is None:
                    base_prompt2cad_bundle = self._create_prompt2cad_base_candidate(
                        experiment_id=f"{experiment_id}_base",
                        plan_id=str(plan["plan_id"]),
                        family=family,
                        geometry=geometry,
                    )
                backend_result = self._run_prompt2cad_param_edit_candidate(
                    experiment_id=experiment_id,
                    plan_id=str(plan["plan_id"]),
                    family=family,
                    geometry=geometry,
                    base_bundle=base_prompt2cad_bundle,
                    plan=plan,
                )
            else:
                backend_result = self._run_geometry_candidate(
                    experiment_id=experiment_id,
                    plan_id=str(plan["plan_id"]),
                    family=family,
                    geometry=geometry,
                    backend_mode=backend_mode,
                )

            pipeline_artifacts = backend_result.get("artifacts") or {}
            backend_success = bool(backend_result.get("success"))
            hardware_dispatch = backend_mode if backend_success else "simulated"
            execution_status = "prepared_with_artifacts" if backend_success else "prepared"

            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "plan_id": plan["plan_id"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "family": family,
                    "geometry": geometry,
                    "geometry_parameters": dict(geometry["parameters"]),
                    "print_parameters": {
                        "scan_speed_mm_s": 20.0,
                        "laser_power_mw": 25.0,
                        "slice_thickness_um": 0.20,
                        "hardware_dispatch": hardware_dispatch,
                    },
                    "artifacts": {
                        "generation_prompt": geometry.get("prompt2cad_prompt"),
                        "stl_path": pipeline_artifacts.get("stl_path"),
                        "cad_code_path": pipeline_artifacts.get("cad_code_path"),
                        "render_path": pipeline_artifacts.get("render_path"),
                        "render_iso_path": pipeline_artifacts.get("render_iso_path"),
                        "render_top_path": pipeline_artifacts.get("render_top_path"),
                        "render_side_path": pipeline_artifacts.get("render_side_path"),
                        "slice_path": None,
                        "print_file_path": None,
                    },
                    "features": features,
                    "prediction": prediction,
                    "uncertainty": uncertainty,
                    "candidate_tags": list(candidate.get("tags") or []),
                    "execution_status": execution_status,
                    "backend_trace": backend_result,
                    "execution_metrics": {
                        "candidate_seconds": round(time.perf_counter() - t_candidate, 4),
                    },
                }
            )
        return {
            "batch_id": f"batch_{uuid.uuid4().hex[:10]}",
            "plan_id": plan["plan_id"],
            "experiments": experiments,
        }

    def _select_backend_mode(self, plan: Mapping[str, Any]) -> str:
        if self.force_template_backend:
            return "template"
        if self.force_prompt2cad_base_edit:
            return "prompt2cad_base_edit"
        strategy = str(plan.get("strategy") or "").strip().lower()
        request_type = str((plan.get("request") or {}).get("input_type") or "").strip().lower()
        is_structured = strategy in {"execute_structured_sweep", "execute_structured_point"} or request_type == "canonical_geometry"
        if is_structured:
            if self.use_prompt2cad_base_edit_for_structured:
                return "prompt2cad_base_edit"
            if self.use_template_backend_for_structured:
                return "template"
        return "prompt2cad"

    def _run_geometry_candidate(
        self,
        *,
        experiment_id: str,
        plan_id: str,
        family: str,
        geometry: Mapping[str, Any],
        backend_mode: str,
    ) -> Dict[str, Any]:
        mode = str(backend_mode or "").strip().lower()
        if mode == "template":
            return self._run_template_candidate(
                experiment_id=experiment_id,
                plan_id=plan_id,
                family=family,
                geometry=geometry,
            )
        result = self._run_prompt2cad_candidate(
            experiment_id=experiment_id,
            plan_id=plan_id,
            family=family,
            geometry=geometry,
        )
        result["backend_mode"] = "prompt2cad"
        return result

    def _run_prompt2cad_candidate(
        self,
        *,
        experiment_id: str,
        plan_id: str,
        family: str,
        geometry: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not self.enable_prompt2cad_backend:
            return {
                "attempted": False,
                "success": False,
                "reason": "prompt2cad backend disabled",
                "artifacts": {},
                "trace": {},
            }
        if not OPENROUTER_API_KEY:
            return {
                "attempted": False,
                "success": False,
                "reason": "OPENROUTER_API_KEY not configured",
                "artifacts": {},
                "trace": {},
            }
        if importlib.util.find_spec("numpy") is None:
            return {
                "attempted": False,
                "success": False,
                "reason": "missing optional dependency: numpy",
                "artifacts": {},
                "trace": {},
            }

        prompt = str(geometry.get("prompt2cad_prompt") or "").strip()
        if not prompt:
            return {
                "attempted": False,
                "success": False,
                "reason": "missing prompt2cad prompt",
                "artifacts": {},
                "trace": {},
            }

        log_before = _load_prompt_run_log()
        before_len = len(log_before)
        run_dir = self.backend_output_dir / plan_id / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        try:
            from ..pipeline.run_pipeline import run_pipeline

            pipeline_result = run_pipeline(
                prompt=prompt,
                model=self.cad_model,
                verbose=self.prompt2cad_verbose,
                output_dir=run_dir / "pipeline_temp",
                stl_filename=f"{family}_{experiment_id}.stl",
                return_trace=True,
                agent_mode="legacy",
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return {
                "attempted": True,
                "success": False,
                "reason": f"prompt2cad execution exception: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "artifacts": {},
                "trace": {},
            }

        log_after = _load_prompt_run_log()
        log_entry = log_after[before_len] if len(log_after) > before_len else {}
        trace = pipeline_result.get("trace") if isinstance(pipeline_result, dict) else {}
        if not trace and isinstance(log_entry, dict):
            trace = {
                "total_seconds": log_entry.get("total_seconds"),
                "planner_output": log_entry.get("planner_output"),
                "retrieved_examples": log_entry.get("retrieved_examples"),
            }

        return {
            "attempted": True,
            "backend_mode": "prompt2cad",
            "success": bool(pipeline_result.get("success")),
            "reason": None if pipeline_result.get("success") else str(pipeline_result.get("error") or "pipeline failed"),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "artifacts": {
                "design_id": pipeline_result.get("design_id"),
                "stl_path": pipeline_result.get("stl_path"),
                "cad_code_path": pipeline_result.get("cad_code_path"),
                "render_path": pipeline_result.get("render_path"),
                "render_iso_path": pipeline_result.get("render_iso_path"),
                "render_top_path": pipeline_result.get("render_top_path"),
                "render_side_path": pipeline_result.get("render_side_path"),
            },
            "trace": trace or {},
            "llm_usage": (trace or {}).get("llm_usage_totals") if isinstance(trace, dict) else {},
        }

    def _create_prompt2cad_base_candidate(
        self,
        *,
        experiment_id: str,
        plan_id: str,
        family: str,
        geometry: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not self.enable_prompt2cad_backend:
            return {
                "attempted": False,
                "success": False,
                "reason": "prompt2cad backend disabled",
                "base_code": "",
                "base_parameters": dict(geometry.get("parameters") or {}),
                "trace": {},
                "artifacts": {},
            }
        if not OPENROUTER_API_KEY:
            return {
                "attempted": False,
                "success": False,
                "reason": "OPENROUTER_API_KEY not configured",
                "base_code": "",
                "base_parameters": dict(geometry.get("parameters") or {}),
                "trace": {},
                "artifacts": {},
            }

        prompt = str(geometry.get("prompt2cad_prompt") or "").strip()
        if not prompt:
            return {
                "attempted": False,
                "success": False,
                "reason": "missing prompt2cad prompt",
                "base_code": "",
                "base_parameters": dict(geometry.get("parameters") or {}),
                "trace": {},
                "artifacts": {},
            }

        param_names = sorted(str(name) for name in dict(geometry.get("parameters") or {}).keys())
        prompt_augmented = self._base_prompt_with_parameter_contract(prompt, param_names)

        run_dir = self.backend_output_dir / plan_id / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            from ..pipeline.run_pipeline import run_pipeline

            pipeline_result = run_pipeline(
                prompt=prompt_augmented,
                model=self.cad_model,
                verbose=self.prompt2cad_verbose,
                output_dir=run_dir / "pipeline_temp",
                stl_filename=f"{family}_{experiment_id}.stl",
                return_trace=True,
                agent_mode="legacy",
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return {
                "attempted": True,
                "success": False,
                "reason": f"base prompt2cad execution exception: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "base_code": "",
                "base_parameters": dict(geometry.get("parameters") or {}),
                "trace": {},
                "artifacts": {},
            }

        trace = pipeline_result.get("trace") if isinstance(pipeline_result, dict) else {}
        base_code = ""
        if isinstance(trace, dict):
            base_code = str(trace.get("final_code") or trace.get("generated_code") or "")
        cad_code_path = pipeline_result.get("cad_code_path")
        if not base_code and cad_code_path and Path(str(cad_code_path)).exists():
            try:
                base_code = Path(str(cad_code_path)).read_text(encoding="utf-8")
            except OSError:
                base_code = ""

        return {
            "attempted": True,
            "success": bool(pipeline_result.get("success")) and bool(base_code.strip()),
            "reason": None if pipeline_result.get("success") and base_code.strip() else str(pipeline_result.get("error") or "missing base cad code"),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "base_code": base_code,
            "base_parameters": dict(geometry.get("parameters") or {}),
            "trace": trace or {},
            "artifacts": {
                "design_id": pipeline_result.get("design_id"),
                "stl_path": pipeline_result.get("stl_path"),
                "cad_code_path": pipeline_result.get("cad_code_path"),
                "render_path": pipeline_result.get("render_path"),
                "render_iso_path": pipeline_result.get("render_iso_path"),
                "render_top_path": pipeline_result.get("render_top_path"),
                "render_side_path": pipeline_result.get("render_side_path"),
            },
        }

    def _run_prompt2cad_param_edit_candidate(
        self,
        *,
        experiment_id: str,
        plan_id: str,
        family: str,
        geometry: Mapping[str, Any],
        base_bundle: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not bool(base_bundle.get("success")):
            return {
                "attempted": True,
                "backend_mode": "prompt2cad_base_edit",
                "success": False,
                "reason": f"base prompt2cad generation failed: {base_bundle.get('reason')}",
                "artifacts": {},
                "trace": {"base_trace": base_bundle.get("trace") or {}},
                "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        base_code = str(base_bundle.get("base_code") or "")
        if not base_code.strip():
            return {
                "attempted": True,
                "backend_mode": "prompt2cad_base_edit",
                "success": False,
                "reason": "base prompt2cad code missing",
                "artifacts": {},
                "trace": {"base_trace": base_bundle.get("trace") or {}},
                "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        run_dir = self.backend_output_dir / plan_id / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = run_dir / "edited_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        target_parameters = dict(geometry.get("parameters") or {})
        fixed_parameters = list((plan.get("geometry_action") or {}).get("fixed_parameters") or [])
        vary_parameters = list((plan.get("geometry_action") or {}).get("vary_parameters") or [])
        edit_prompt = str(geometry.get("prompt2cad_prompt") or "")
        retrieved: List[dict] = []
        retrieved_summary: List[Dict[str, Any]] = []
        try:
            from ..retrieval.retriever import get_retriever  # lazy import (numpy optional)

            retriever = get_retriever()
            retrieved = retriever.retrieve(
                edit_prompt,
                plan={"intent": "parameter_edit", "operations": ["update_parameters"]},
            )
            retrieved_summary = [
                {
                    "title": chunk.get("title", ""),
                    "source": chunk.get("source", ""),
                    "score": float(chunk.get("score", 0.0) or 0.0),
                    "tags": list(((chunk.get("metadata") or {}).get("tags") or [])),
                }
                for chunk in retrieved
            ]
        except Exception:
            retrieved = []
            retrieved_summary = []

        retrieved_context = self._format_retrieved_context(retrieved)

        edit_messages = [
            {
                "role": "system",
                "content": (
                    "You are a Prompt2CAD parameter editor.\n"
                    "Update ONLY numeric parameter definitions/uses to match target parameters.\n"
                    "Preserve geometry topology and modeling sequence.\n"
                    "Return ONLY valid CadQuery Python code.\n"
                    "Must include `import cadquery as cq` and assign final geometry to `result`.\n"
                    "CadQuery 2.x API only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Base parameters: {json.dumps(base_bundle.get('base_parameters') or {}, ensure_ascii=True)}\n"
                    f"Target parameters: {json.dumps(target_parameters, ensure_ascii=True)}\n"
                    f"Fixed parameters: {json.dumps(fixed_parameters, ensure_ascii=True)}\n"
                    f"Vary parameters: {json.dumps(vary_parameters, ensure_ascii=True)}\n\n"
                    "Retrieved examples/context:\n"
                    f"{retrieved_context}\n\n"
                    "Base code:\n"
                    f"{base_code}\n"
                ),
            },
        ]

        llm_traces: list[Dict[str, Any]] = []
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            edited_code = call_openrouter(edit_messages, model=self.cad_model, temperature=0.0)
            edit_trace = get_last_openrouter_trace()
            if isinstance(edit_trace, dict) and edit_trace:
                edit_trace["stage"] = "parameter_edit"
                llm_traces.append(edit_trace)
                self._accumulate_usage(usage_totals, edit_trace.get("usage"))
        except Exception as exc:
            return {
                "attempted": True,
                "backend_mode": "prompt2cad_base_edit",
                "success": False,
                "reason": f"parameter edit call failed: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "artifacts": {},
                "trace": {
                    "base_trace": base_bundle.get("trace") or {},
                    "execution_result": {"success": False, "error": str(exc), "stl_path": None},
                },
                "llm_usage": usage_totals,
            }

        t_execute = time.perf_counter()
        exec_res = execute_cad(
            edited_code,
            output_dir=temp_dir,
            stl_filename=f"{family}_{experiment_id}.stl",
        )
        execute_seconds = round(time.perf_counter() - t_execute, 4)

        final_code = edited_code
        stl_path = exec_res.get("stl_path")
        repair_attempts = 0
        repair_errors: list[str] = []
        if not exec_res.get("success"):
            repair_res = repair(
                prompt=edit_prompt,
                code=edited_code,
                error=str(exec_res.get("error") or "execution failed"),
                retrieved=retrieved,
                model=self.cad_model,
                max_retries=MAX_REPAIR_ATTEMPTS,
                output_dir=str(temp_dir),
            )
            final_code = str(repair_res.get("code") or edited_code)
            stl_path = repair_res.get("stl_path")
            repair_attempts = int(repair_res.get("repair_attempts") or 0)
            repair_errors = [str(item) for item in list(repair_res.get("errors") or [])]
            for item in list(repair_res.get("llm_traces") or []):
                trace = dict(item)
                trace["stage"] = "repair"
                llm_traces.append(trace)
                self._accumulate_usage(usage_totals, trace.get("usage"))

        success = bool(stl_path)
        render_paths: Dict[str, Any] = {
            "render_path": None,
            "render_iso_path": None,
            "render_top_path": None,
            "render_side_path": None,
        }
        render_seconds = 0.0
        if success:
            t_render = time.perf_counter()
            render_paths = render_stl(str(stl_path), str(temp_dir))
            render_seconds = round(time.perf_counter() - t_render, 4)

        execution_error = None
        if not success:
            execution_error = str(exec_res.get("error") or "")
            if repair_errors:
                execution_error = repair_errors[-1]

        run_trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": edit_prompt,
            "model": self.cad_model,
            "agent_mode": "prompt2cad_base_edit",
            "tool_mode": "off",
            "planner_output": {
                "intent": "parameterized_code_edit",
                "operations": ["base_prompt2cad_generation", "parameter_edit", "execute", "repair"],
                "geometry_type": "solid",
                "tags": ["base_model", "parameter_sweep"],
                "difficulty": "controlled",
            },
            "retrieved_examples": retrieved_summary,
            "decision_trace": [
                {
                    "step": 1,
                    "intent": "base_prompt2cad_generation",
                    "selected_tools": ["run_pipeline_legacy"],
                    "why": "build canonical base CAD model from prompt2cad",
                    "confidence": 0.95,
                    "result": "success" if bool(base_bundle.get("success")) else "failure",
                    "metadata": {
                        "base_design_id": (base_bundle.get("artifacts") or {}).get("design_id"),
                        "base_reason": base_bundle.get("reason"),
                    },
                },
                {
                    "step": 2,
                    "intent": "parameter_edit",
                    "selected_tools": ["llm_code_edit"],
                    "why": "edit only parameters while preserving base geometry topology",
                    "confidence": 0.9,
                    "result": "success" if bool(edited_code.strip()) else "failure",
                    "metadata": {
                        "target_parameters": target_parameters,
                        "fixed_parameters": fixed_parameters,
                        "vary_parameters": vary_parameters,
                    },
                },
                {
                    "step": 3,
                    "intent": "execute_and_repair",
                    "selected_tools": ["cad_executor", "repair_loop"],
                    "why": "always include repair cycle on execution failure",
                    "confidence": 0.9,
                    "result": "success" if success else "failure",
                    "metadata": {
                        "repair_attempts": repair_attempts,
                        "execution_error": execution_error,
                    },
                },
            ],
            "generated_code": edited_code,
            "execution_result": {
                "success": success,
                "error": execution_error,
                "stl_path": stl_path,
            },
            "repair_attempts": repair_attempts,
            "final_code": final_code,
            "stl_path": stl_path,
            "success": success,
            "total_seconds": round(time.perf_counter() - started, 2),
            "stage_timings": {
                "execute_seconds": execute_seconds,
                "render_seconds": render_seconds,
            },
            "llm_traces": llm_traces,
            "llm_usage_totals": usage_totals,
            "base_trace": base_bundle.get("trace") or {},
        }

        t_store = time.perf_counter()
        meta = create_design(
            prompt=edit_prompt,
            cad_code=final_code,
            stl_path=stl_path,
            render_path=render_paths.get("render_path"),
            render_iso_path=render_paths.get("render_iso_path"),
            render_top_path=render_paths.get("render_top_path"),
            render_side_path=render_paths.get("render_side_path"),
            parameters_detected={
                "family": family,
                "base_design_id": (base_bundle.get("artifacts") or {}).get("design_id"),
                "target_parameters": target_parameters,
                "fixed_parameters": fixed_parameters,
                "vary_parameters": vary_parameters,
            },
            status="success" if success else "failed",
            run_trace=run_trace,
        )
        run_trace["stage_timings"]["store_seconds"] = round(time.perf_counter() - t_store, 4)

        return {
            "attempted": True,
            "backend_mode": "prompt2cad_base_edit",
            "success": success,
            "reason": None if success else execution_error or "parameter edit execution failed",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "artifacts": {
                "design_id": meta.get("design_id"),
                "stl_path": meta.get("stl_path"),
                "cad_code_path": meta.get("cad_code_path"),
                "render_path": meta.get("render_path"),
                "render_iso_path": meta.get("render_iso_path"),
                "render_top_path": meta.get("render_top_path"),
                "render_side_path": meta.get("render_side_path"),
                "base_design_id": (base_bundle.get("artifacts") or {}).get("design_id"),
            },
            "trace": run_trace,
            "llm_usage": usage_totals,
        }

    def _base_prompt_with_parameter_contract(self, prompt: str, parameter_names: list[str]) -> str:
        names = ", ".join(parameter_names) if parameter_names else "(none provided)"
        return (
            f"{prompt}\n\n"
            "Important code contract:\n"
            "- This is a base model for later parameter sweeps.\n"
            "- Define top-level numeric variables for these parameters exactly by name.\n"
            f"- Parameter variable names: {names}\n"
            "- Use those variables in modeling operations.\n"
            "- Assign final geometry to `result`.\n"
        )

    def _format_retrieved_context(self, retrieved: List[dict]) -> str:
        if not retrieved:
            return "(no retrieval context available)"

        blocks: List[str] = []
        for chunk in retrieved[:4]:
            title = str(chunk.get("title") or "").strip()
            source = str(chunk.get("source") or "").strip()
            score = float(chunk.get("score", 0.0) or 0.0)
            text = str(chunk.get("text") or "").strip()
            snippet = text[:900]
            blocks.append(
                f"[{source}] {title} (score={score:.3f})\n{snippet}"
            )
        return "\n\n".join(blocks)

    def _accumulate_usage(self, target: Dict[str, int], usage: Any) -> None:
        if not isinstance(target, dict) or not isinstance(usage, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            target[key] = int(target.get(key) or 0) + int(usage.get(key) or 0)

    def _run_template_candidate(
        self,
        *,
        experiment_id: str,
        plan_id: str,
        family: str,
        geometry: Mapping[str, Any],
    ) -> Dict[str, Any]:
        run_dir = self.backend_output_dir / plan_id / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = run_dir / "template_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        codegen = build_template_cadquery_code(family, geometry.get("parameters"))
        code = str(codegen.get("code") or "")
        template_name = str(codegen.get("template_name") or "template")
        normalized_parameters = dict(codegen.get("parameters") or {})
        prompt = str(geometry.get("prompt2cad_prompt") or "")

        t_execute = time.perf_counter()
        exec_res = execute_cad(
            code,
            output_dir=temp_dir,
            stl_filename=f"{family}_{experiment_id}.stl",
        )
        execute_seconds = round(time.perf_counter() - t_execute, 4)
        success = bool(exec_res.get("success") and exec_res.get("stl_path"))
        render_paths: Dict[str, Any] = {
            "render_path": None,
            "render_iso_path": None,
            "render_top_path": None,
            "render_side_path": None,
        }

        render_seconds = 0.0
        if success:
            t_render = time.perf_counter()
            render_paths = render_stl(str(exec_res["stl_path"]), str(temp_dir))
            render_seconds = round(time.perf_counter() - t_render, 4)

        total_seconds = round(time.perf_counter() - started, 2)
        run_trace = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "model": None,
            "agent_mode": "template",
            "tool_mode": "off",
            "planner_output": {
                "intent": "deterministic_template_generation",
                "operations": [template_name],
                "geometry_type": "solid",
                "tags": ["parameterized_template"],
                "difficulty": "deterministic",
            },
            "retrieved_examples": [],
            "decision_trace": [
                {
                    "step": 1,
                    "intent": "deterministic_template_generation",
                    "selected_tools": ["template_codegen", "cad_executor", "render_stl"],
                    "why": "structured sweep uses a fixed base geometry template",
                    "confidence": 0.99,
                    "result": "success" if success else "failure",
                    "metadata": {
                        "template_name": template_name,
                        "family": family,
                        "parameters": normalized_parameters,
                    },
                }
            ],
            "generated_code": code,
            "execution_result": {
                "success": bool(exec_res.get("success")),
                "error": exec_res.get("error"),
                "stl_path": exec_res.get("stl_path"),
            },
            "repair_attempts": 0,
            "final_code": code,
            "stl_path": exec_res.get("stl_path"),
            "success": success,
            "total_seconds": total_seconds,
            "stage_timings": {
                "template_codegen_seconds": round(max(total_seconds - execute_seconds - render_seconds, 0.0), 4),
                "execute_seconds": execute_seconds,
                "render_seconds": render_seconds,
            },
            "llm_traces": [],
            "llm_usage_totals": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        t_store = time.perf_counter()
        try:
            meta = create_design(
                prompt=prompt,
                cad_code=code,
                stl_path=exec_res.get("stl_path"),
                render_path=render_paths.get("render_path"),
                render_iso_path=render_paths.get("render_iso_path"),
                render_top_path=render_paths.get("render_top_path"),
                render_side_path=render_paths.get("render_side_path"),
                parameters_detected={
                    "family": family,
                    "template_name": template_name,
                    "parameters": normalized_parameters,
                },
                status="success" if success else "failed",
                run_trace=run_trace,
            )
            run_trace["stage_timings"]["store_seconds"] = round(time.perf_counter() - t_store, 4)
        except Exception as exc:  # pragma: no cover - filesystem/runtime guard
            run_trace["stage_timings"]["store_seconds"] = round(time.perf_counter() - t_store, 4)
            run_trace["execution_result"]["error"] = str(exc)
            return {
                "attempted": True,
                "backend_mode": "template",
                "success": False,
                "reason": f"template design storage failed: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "artifacts": {
                    "design_id": None,
                    "stl_path": exec_res.get("stl_path"),
                    "cad_code_path": None,
                    "render_path": render_paths.get("render_path"),
                    "render_iso_path": render_paths.get("render_iso_path"),
                    "render_top_path": render_paths.get("render_top_path"),
                    "render_side_path": render_paths.get("render_side_path"),
                },
                "trace": run_trace,
                "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        return {
            "attempted": True,
            "backend_mode": "template",
            "success": success,
            "reason": None if success else str(exec_res.get("error") or "template execution failed"),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "artifacts": {
                "design_id": meta.get("design_id"),
                "stl_path": meta.get("stl_path"),
                "cad_code_path": meta.get("cad_code_path"),
                "render_path": meta.get("render_path"),
                "render_iso_path": meta.get("render_iso_path"),
                "render_top_path": meta.get("render_top_path"),
                "render_side_path": meta.get("render_side_path"),
            },
            "trace": run_trace,
            "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def run_experiment_batch(plan: Mapping[str, Any], dataset_records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    return ExperimentExecutor().run(plan, dataset_records)


def _load_prompt_run_log() -> list[dict]:
    path = LOGS_DIR / "prompt_runs.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [row for row in rows if isinstance(row, dict)]

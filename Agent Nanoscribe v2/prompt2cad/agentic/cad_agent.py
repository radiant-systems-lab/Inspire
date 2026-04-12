"""
CADAgent -- standalone iterative tool-calling agent for CadQuery geometry construction.

The LLM drives geometry construction step-by-step via tool calls.
It never writes raw Python -- it calls tools like create_primitive(),
array_wrap_xy(), boolean_operation(), etc. and inspects intermediate
results via get_workspace() before deciding the next step.

This eliminates entire classes of bugs (z-staircase, wrong API calls,
syntax errors) because the tool implementations handle all CadQuery
API details internally.

Entry point:
    agent = CADAgent(model="deepseek/deepseek-chat", output_dir="output/")
    result = agent.run(cge_prompt_block, cge=cge_obj)
    # result: {
    #   success, stl_path, code, steps, tokens, tool_log, path,
    #   retrieved_examples, api_context, verification, error
    # }
"""

from __future__ import annotations

import json
import re
import time
from statistics import mean
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys as _sys
from pathlib import Path as _Path

# forks/ sits alongside prompt2cad/ inside subsystems/prompt2cad/
# e.g.  subsystems/prompt2cad/forks/cadquery_agent/
_FORKS_ROOT = _Path(__file__).resolve().parents[2] / "forks"
if str(_FORKS_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_FORKS_ROOT))

from ..utils import (
    assistant_tool_call_message,
    call_openrouter_agent,
    get_last_openrouter_trace,
    tool_result_message,
)
from ..execution.cad_executor import execute as cad_execute
from ..execution.cad_verifier import (
    hard_verify,
    soft_verify,
    extract_cge_from_code,
    compute_geometry_metrics,
    structured_vision_analysis,
    _execute_for_workplane,
)
from ..execution.render_utils import render_stl
from ..generator.cad_generator import generate as generate_code
from ..repair.repair_loop import repair as repair_code

# Re-use the tool implementations and session from the fork
from cadquery_agent.session import CadSessionState
from cadquery_agent.tool_registry import ToolRegistry
from cadquery_agent.tools import create_default_registry
from cadquery_agent.types import ToolResult


_SYSTEM = """You are a CadQuery geometry construction agent.

You build 3D geometry step-by-step using tool calls.
You do NOT write Python code -- you call tools that execute directly on a CadQuery workspace.

== TOOLS OVERVIEW ==
- create_primitive : box, cylinder, sphere, cone, torus -- with optional x/y/z position
- transform_object : translate and/or rotate a named object
- array_wrap_xy    : create a 2D rectangular array (ALWAYS use this for grids -- never loop)
- pattern_linear   : 1D linear repeat along any axis
- pattern_polar    : polar/circular repeat
- boolean_operation: union (fuse), subtract (cut), intersect (common)
- fillet_edges     : round all edges of an object
- chamfer_edges    : chamfer all edges
- mirror_object    : mirror across XY / XZ / YZ
- measure          : get volume and bounding box of any object
- get_workspace    : list all named objects currently in the workspace
- export_model     : export final result to STL (MUST call this when done)
- execute_code     : escape hatch for operations not covered by the above tools

== RULES ==
- Call get_workspace() at any time to inspect what objects exist and their names
- Use measure() after creating objects to verify dimensions are correct
- For 2D pillar / hole arrays: ALWAYS use array_wrap_xy(nx, ny, spacing_x, spacing_y)
  This anchors every instance at z=0 -- do not use pattern_linear in a loop
- Name objects descriptively: "pillar_unit", "array_5x5", "substrate", etc.
- When the geometry is complete, call export_model() to produce the STL
- If a tool fails, read the error and try a corrected call -- do not give up immediately
- Use execute_code() only for truly complex operations (sweeps, lofts, custom profiles)
  that cannot be expressed with the standard tools
"""


# Extra tool schema for get_workspace (not in the fork registry)
_GET_WORKSPACE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_workspace",
        "description": "List all named objects currently in the CAD workspace with their bounding box dimensions. Call this to inspect what has been built so far.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class CADAgent:
    """
    Standalone iterative tool-calling CAD construction agent.

    The LLM receives the geometry description (CGE prompt block or plain text),
    then makes tool calls to build it step by step, inspecting intermediate
    results after each call before deciding the next step.
    """

    def __init__(
        self,
        model: str = "deepseek/deepseek-chat",
        vision_model: str = "openai/gpt-4o",
        output_dir: str = "output/cad_agent",
        max_steps: int = 30,
        verbose: bool = False,
        stl_filename: str = "result.stl",
        max_retrieved_examples: int = 4,
    ) -> None:
        self.model = model
        self.vision_model = vision_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.verbose = verbose
        self.stl_filename = stl_filename
        self.max_retrieved_examples = max(1, int(max_retrieved_examples))

    # ------------------------------------------------------------------
    def run(self, description: str, cge: Optional[Any] = None) -> Dict[str, Any]:
        """
        Build geometry from a text description or CGE prompt block.
        This method is self-contained:
          1) Retrieve context/examples + API snippets
          2) Attempt iterative tool-based construction
          3) Fallback to codegen+repair when needed
          4) Run hard + medium + soft verification and return structured outputs

        Returns:
            {
                success    : bool,
                stl_path   : str | None,
                code       : str,       # command log (tool path) or python (fallback path)
                steps      : int,
                tokens     : {prompt, completion, total},
                tool_log   : [{step, tool, args, result}],
                path       : "cad_agent" | "fallback_codegen",
                retrieved_examples: [{title, source, score, tags}],
                api_context: str,
                verification: {hard, medium, soft, renders, intended_cge, extracted_cge, geometry_metrics},
                error      : str | None,
            }
        """
        retrieved, retrieved_summary, api_context, examples_context = self._retrieve_context(description)

        state = CadSessionState()
        registry = create_default_registry(
            state,
            output_dir=str(self.output_dir),
            stl_filename=self.stl_filename,
        )
        tool_schemas = registry.to_openai_schema() + [_GET_WORKSPACE_SCHEMA]

        agent_prompt = self._build_agent_prompt(
            description=description,
            examples_context=examples_context,
            api_context=api_context,
        )
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": agent_prompt},
        ]
        tokens = {"prompt": 0, "completion": 0, "total": 0}
        tool_log: List[Dict[str, Any]] = []
        started = time.perf_counter()
        step = 0
        error: Optional[str] = None

        for step in range(1, self.max_steps + 1):
            response = call_openrouter_agent(
                messages=messages,
                model=self.model,
                tools=tool_schemas,
                temperature=0.0,
            )

            # Accumulate token usage
            trace = get_last_openrouter_trace()
            if trace:
                usage = trace.get("usage") or {}
                tokens["prompt"]     += int(usage.get("prompt_tokens", 0))
                tokens["completion"] += int(usage.get("completion_tokens", 0))
                tokens["total"]      += int(usage.get("total_tokens", 0))

            finish = response.get("finish_reason", "")
            tool_calls = response.get("tool_calls") or []

            if not tool_calls:
                # LLM finished without more tool calls
                if self.verbose:
                    print(f"[CADAgent] Step {step}: LLM stopped (finish={finish})")
                break

            # Add assistant turn to message history
            messages.append(
                assistant_tool_call_message(tool_calls, response.get("content"))
            )

            # Execute each tool call and collect results
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                if self.verbose:
                    print(f"[CADAgent] Step {step}: {fn_name}({_fmt_args(fn_args)})")

                if fn_name == "get_workspace":
                    result_payload = self._workspace_snapshot(state, registry)
                    result_str = json.dumps(result_payload)
                    tool_log.append({"step": step, "tool": fn_name, "args": {}, "result": result_payload})
                else:
                    tr: ToolResult = registry.execute(fn_name, fn_args)
                    result_payload = {
                        "success": tr.success,
                        "output":  tr.output,
                        "error":   tr.error or None,
                        "data":    tr.data or {},
                    }
                    result_str = json.dumps(result_payload)
                    tool_log.append({"step": step, "tool": fn_name, "args": fn_args, "result": result_payload})
                    if not tr.success:
                        error = tr.error or f"{fn_name} failed"

                messages.append(tool_result_message(tc["id"], result_str))

            # Stop as soon as STL is produced
            if state.artifacts.get("stl_path"):
                break

        elapsed = time.perf_counter() - started

        # Auto-export if LLM never called export_model
        stl_path = state.artifacts.get("stl_path")
        if not stl_path and state.active_object:
            if self.verbose:
                print(f"[CADAgent] Auto-exporting '{state.active_object}'")
            # If multiple objects remain, union them first
            self._auto_union(registry, state)
            export_r = registry.execute(
                "export_model",
                {
                    "object_name": state.active_object or "",
                    "file_path":   str(self.output_dir / self.stl_filename),
                    "format":      "stl",
                },
            )
            if export_r.success:
                stl_path = (export_r.data or {}).get("path")
            else:
                error = error or export_r.error or "export failed"

        stl_path = self._normalize_stl_path(stl_path)

        final_code = state.render_command_log()
        path = "cad_agent"
        fallback_hard: Optional[Dict[str, Any]] = None

        if not stl_path:
            path = "fallback_codegen"
            fb = self._fallback_codegen(
                description=description,
                retrieved=retrieved,
                tokens=tokens,
            )
            if fb.get("success"):
                stl_path = str(fb.get("stl_path") or "")
                final_code = str(fb.get("code") or final_code)
                fallback_hard = fb.get("hard")
                error = None
            else:
                error = str(fb.get("error") or error or "CADAgent produced no STL")

        stl_path = self._normalize_stl_path(stl_path)

        expected_bodies = self._expected_body_count(cge)
        verification = self._verify_output(
            path=path,
            stl_path=stl_path,
            code=final_code,
            state=state,
            registry=registry,
            cge=cge,
            expected_body_count=expected_bodies,
            precomputed_hard=fallback_hard,
        )

        success = bool(stl_path) and bool((verification.get("hard") or {}).get("passed", False))

        return {
            "success":  success,
            "stl_path": stl_path,
            "code":     final_code,
            "steps":    step,
            "elapsed_s": round(elapsed, 2),
            "tokens":   tokens,
            "tool_log": tool_log,
            "path": path,
            "retrieved_examples": retrieved_summary,
            "api_context": api_context,
            "verification": verification,
            "error": None if success else (error or self._verification_error(verification)),
        }

    # ------------------------------------------------------------------
    def _workspace_snapshot(
        self, state: CadSessionState, registry: ToolRegistry
    ) -> Dict[str, Any]:
        """Return a compact view of the current workspace for the LLM."""
        objects_info = []
        for name in state.objects:
            tr = registry.execute("measure", {"object_name": name})
            info: Dict[str, Any] = {"name": name}
            if tr.success and tr.data:
                bb = tr.data.get("bounding_box") or {}
                info["bbox"] = {
                    "x": round(bb.get("x", 0), 4),
                    "y": round(bb.get("y", 0), 4),
                    "z": round(bb.get("z", 0), 4),
                }
                info["volume"] = round(tr.data.get("volume", 0), 4)
            objects_info.append(info)

        return {
            "objects": objects_info,
            "active":  state.active_object,
            "count":   len(objects_info),
            "stl_ready": bool(state.artifacts.get("stl_path")),
        }

    def _auto_union(self, registry: ToolRegistry, state: CadSessionState) -> None:
        """Union all remaining objects into one before export."""
        names = list(state.objects.keys())
        if len(names) <= 1:
            return
        base = state.active_object if state.active_object in state.objects else names[0]
        for name in names:
            if name == base:
                continue
            r = registry.execute(
                "boolean_operation",
                {"operation": "fuse", "base_object": base, "tool_object": name, "result_name": "final"},
            )
            if r.success:
                base = "final"
        if base in state.objects:
            state.active_object = base

    # ------------------------------------------------------------------
    def _build_agent_prompt(self, description: str, examples_context: str, api_context: str) -> str:
        return (
            f"{description}\n\n"
            "=== Retrieved CadQuery Examples (high-similarity) ===\n"
            f"{examples_context}\n\n"
            "=== Retrieved CadQuery API Context ===\n"
            f"{api_context}\n\n"
            "Execution contract:\n"
            "- Prefer native geometry tools and inspect workspace before export.\n"
            "- For arrays, use array_wrap_xy and avoid looped workplane accumulation.\n"
            "- Export STL when complete.\n"
            "- If blocked by missing operation, use execute_code as last resort.\n"
        )

    def _retrieve_context(self, description: str) -> tuple[List[dict], List[Dict[str, Any]], str, str]:
        try:
            from ..retrieval.retriever import get_retriever

            retrieved = get_retriever().retrieve(
                description,
                plan=None,
                n_examples=self.max_retrieved_examples,
                n_cheatsheet=1,
            )
        except Exception:
            retrieved = []

        retrieved_summary = [
            {
                "title": c.get("title", ""),
                "source": c.get("source", ""),
                "score": float(c.get("score", 0.0) or 0.0),
                "tags": list(((c.get("metadata") or {}).get("tags") or [])),
            }
            for c in retrieved
        ]

        examples = [c for c in retrieved if c.get("source") == "examples"][: self.max_retrieved_examples]
        ref = [c for c in retrieved if c.get("source") != "examples"]
        best_ref = max(ref, key=lambda c: float(c.get("score", 0.0) or 0.0), default=None)

        if examples:
            ex_blocks: List[str] = []
            for idx, ex in enumerate(examples, start=1):
                code = self._extract_python_block(str(ex.get("text", "")))
                code = code[:900].strip() if code else str(ex.get("text", ""))[:900].strip()
                tags = ", ".join(((ex.get("metadata") or {}).get("tags") or []))
                title = str(ex.get("title") or f"example_{idx}")
                ex_blocks.append(
                    f"[{idx}] {title} score={float(ex.get('score', 0.0) or 0.0):.3f}\n"
                    + (f"tags: {tags}\n" if tags else "")
                    + code
                )
            examples_context = "\n\n".join(ex_blocks)
        else:
            examples_context = "(no example snippets retrieved)"

        if best_ref:
            api_context = str(best_ref.get("text", ""))[:1600]
        else:
            api_context = "(no API context retrieved)"

        return retrieved, retrieved_summary, api_context, examples_context

    def _extract_python_block(self, text: str) -> str:
        m = re.search(r"```python\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*([\s\S]*?)\s*```", text)
        if m:
            return m.group(1).strip()
        return ""

    def _fallback_codegen(
        self,
        description: str,
        retrieved: List[dict],
        tokens: Dict[str, int],
    ) -> Dict[str, Any]:
        code = generate_code(description, retrieved=retrieved, rag_examples=[], model=self.model)
        gen_trace = get_last_openrouter_trace()
        if gen_trace:
            usage = gen_trace.get("usage") or {}
            tokens["prompt"] += int(usage.get("prompt_tokens", 0))
            tokens["completion"] += int(usage.get("completion_tokens", 0))
            tokens["total"] += int(usage.get("total_tokens", 0))

        exec_result = cad_execute(code, output_dir=str(self.output_dir), stl_filename=self.stl_filename)
        if exec_result.get("success"):
            hard = hard_verify(code=code, stl_path=exec_result.get("stl_path"))
            return {
                "success": True,
                "code": code,
                "stl_path": exec_result.get("stl_path"),
                "hard": hard,
            }

        repair_res = repair_code(
            prompt=description,
            code=code,
            error=str(exec_result.get("error") or "execution failed"),
            retrieved=retrieved,
            model=self.model,
            output_dir=str(self.output_dir),
        )
        for item in list(repair_res.get("llm_traces") or []):
            usage = (item or {}).get("usage") or {}
            tokens["prompt"] += int(usage.get("prompt_tokens", 0))
            tokens["completion"] += int(usage.get("completion_tokens", 0))
            tokens["total"] += int(usage.get("total_tokens", 0))

        fixed_code = str(repair_res.get("code") or code)
        stl_path = repair_res.get("stl_path")
        if repair_res.get("success"):
            hard = hard_verify(code=fixed_code, stl_path=stl_path)
            return {
                "success": True,
                "code": fixed_code,
                "stl_path": stl_path,
                "hard": hard,
            }

        err = str((repair_res.get("errors") or [exec_result.get("error")])[-1] or "fallback failed")
        return {"success": False, "code": fixed_code, "error": err}

    def _verify_output(
        self,
        *,
        path: str,
        stl_path: Optional[str],
        code: str,
        state: CadSessionState,
        registry: ToolRegistry,
        cge: Optional[Any],
        expected_body_count: Optional[int],
        precomputed_hard: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        intended_cge: Dict[str, Any] = {}
        if cge is not None:
            try:
                if hasattr(cge, "to_dict"):
                    intended_cge = cge.to_dict()
                else:
                    intended_cge = {"raw": str(cge)}
            except Exception:
                intended_cge = {"raw": str(cge)}

        # Stage 1: hard assertions
        hard: Dict[str, Any]
        if precomputed_hard is not None:
            hard = dict(precomputed_hard)
        elif path == "cad_agent":
            hard = self._hard_verify_tool_state(
                state=state,
                registry=registry,
                stl_path=stl_path,
                expected_body_count=expected_body_count,
            )
        else:
            hard = hard_verify(
                code=code,
                stl_path=stl_path,
                expected_body_count=expected_body_count,
            )

        if not hard.get("passed", False):
            return {
                "hard": hard,
                "medium": {
                    "passed": False,
                    "score": 0.0,
                    "checks": [],
                    "warnings": ["Skipped medium checks because hard assertions failed."],
                    "param_drift": [],
                    "missing_parameters": [],
                },
                "soft": {},
                "vision": {},
                "intended_cge": intended_cge,
                "extracted_cge": {},
                "geometry_metrics": {},
                "renders": {},
            }

        # Stage 2: extract CGE from code (only meaningful for codegen path)
        extracted_cge: Dict[str, Any] = {}
        try:
            extracted_cge = extract_cge_from_code(code)
        except Exception as exc:
            extracted_cge = {"error": str(exc)}

        # Stage 3: geometry metrics
        # For tool path use the live workplane; for codegen path re-execute code.
        geometry_metrics: Dict[str, Any] = {}
        try:
            if path == "cad_agent" and state is not None:
                wp = state.get_object(state.active_object) if state.active_object else None
            else:
                wp = _execute_for_workplane(code)
            if wp is not None:
                geometry_metrics = compute_geometry_metrics(wp)
        except Exception as exc:
            geometry_metrics = {"error": str(exc)}

        # Stage 4: medium assertions (deterministic but non-blocking)
        medium = self._build_medium_assertions(
            cge=cge,
            extracted_cge=extracted_cge,
            geometry_metrics=geometry_metrics,
            hard=hard,
            expected_body_count=expected_body_count,
        )

        # Stage 5: renders + structured vision analysis (soft, non-blocking)
        renders: Dict[str, str] = {}
        soft: Dict[str, Any] = {}
        if stl_path and Path(stl_path).exists():
            r = render_stl(stl_path=stl_path, output_dir=str(self.output_dir))
            renders = {k: v for k, v in r.items() if v}
        if renders:
            try:
                soft = structured_vision_analysis(render_paths=renders, model=self.vision_model)
            except Exception as exc:
                soft = {"error": str(exc)}

        return {
            "hard": hard,
            "medium": medium,
            "soft": soft,
            "vision": soft,  # backward-compatible alias
            "intended_cge": intended_cge,
            "extracted_cge": extracted_cge,
            "geometry_metrics": geometry_metrics,
            "renders": renders,
        }

    def _build_medium_assertions(
        self,
        *,
        cge: Optional[Any],
        extracted_cge: Dict[str, Any],
        geometry_metrics: Dict[str, Any],
        hard: Dict[str, Any],
        expected_body_count: Optional[int],
    ) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        warnings: List[str] = []

        expected_params = self._extract_expected_numeric_parameters(cge)
        summary = (extracted_cge or {}).get("summary") or {}
        variables = (extracted_cge or {}).get("variables") or {}

        drift: List[Dict[str, Any]] = []
        missing_parameters: List[str] = []

        matched = 0
        rel_errors: List[float] = []
        for param_name, expected_value in expected_params.items():
            matched_key, observed_value = self._find_observed_param_value(
                parameter_name=param_name,
                observed_summary=summary,
                observed_variables=variables,
            )
            if observed_value is None:
                missing_parameters.append(param_name)
                drift.append(
                    {
                        "parameter": param_name,
                        "expected": expected_value,
                        "observed": None,
                        "matched_key": None,
                        "rel_error": None,
                    }
                )
                continue

            rel_error = abs(observed_value - expected_value) / max(abs(expected_value), 1e-6)
            matched += 1
            rel_errors.append(float(rel_error))
            drift.append(
                {
                    "parameter": param_name,
                    "expected": expected_value,
                    "observed": observed_value,
                    "matched_key": matched_key,
                    "rel_error": round(float(rel_error), 6),
                }
            )

        coverage: Optional[float] = None
        if expected_params:
            coverage = matched / max(len(expected_params), 1)
            checks.append(
                {
                    "name": "cge_parameter_coverage",
                    "passed": coverage >= 0.5,
                    "severity": "medium",
                    "detail": f"Matched {matched}/{len(expected_params)} expected numeric CGE parameters.",
                    "value": round(float(coverage), 6),
                }
            )
            if coverage < 0.5:
                warnings.append("Low CGE parameter coverage in extracted code summary.")

        mean_rel: Optional[float] = None
        if rel_errors:
            mean_rel = float(mean(rel_errors))
            checks.append(
                {
                    "name": "cge_parameter_alignment",
                    "passed": mean_rel <= 0.25,
                    "severity": "medium",
                    "detail": f"Mean relative error across matched parameters = {mean_rel:.4f}",
                    "value": round(mean_rel, 6),
                }
            )
            if mean_rel > 0.25:
                warnings.append("High parameter drift between intended CGE and extracted code parameters.")

        body_count = hard.get("body_count")
        body_ok: Optional[bool] = None
        if expected_body_count is not None and body_count is not None:
            try:
                body_ok = int(body_count) == int(expected_body_count)
            except Exception:
                body_ok = None
            checks.append(
                {
                    "name": "body_count_consistency",
                    "passed": body_ok,
                    "severity": "high",
                    "detail": f"Expected {expected_body_count}, observed {body_count}.",
                    "expected": expected_body_count,
                    "observed": body_count,
                }
            )
            if body_ok is False:
                warnings.append(f"Body count mismatch: expected {expected_body_count}, observed {body_count}.")

        bbox = geometry_metrics or {}
        bx = self._as_float(bbox.get("bbox_x"))
        by = self._as_float(bbox.get("bbox_y"))
        bz = self._as_float(bbox.get("bbox_z"))
        bbox_positive = bx is not None and by is not None and bz is not None and bx > 0 and by > 0 and bz > 0
        checks.append(
            {
                "name": "bbox_metric_sanity",
                "passed": bool(bbox_positive),
                "severity": "medium",
                "detail": f"bbox=({bx}, {by}, {bz})",
            }
        )
        if not bbox_positive:
            warnings.append("Bounding-box metrics are missing or non-positive.")

        bbox_expected = {
            "bbox_x": expected_params.get("bbox_x"),
            "bbox_y": expected_params.get("bbox_y"),
            "bbox_z": expected_params.get("bbox_z"),
        }
        bbox_rel: Optional[float] = None
        if all(v is not None for v in bbox_expected.values()) and bbox_positive:
            errs: List[float] = []
            for dim_key, observed in (("bbox_x", bx), ("bbox_y", by), ("bbox_z", bz)):
                expected = bbox_expected.get(dim_key)
                if expected is None or observed is None:
                    continue
                errs.append(abs(observed - expected) / max(abs(expected), 1e-6))
            if errs:
                bbox_rel = float(mean(errs))
                checks.append(
                    {
                        "name": "bbox_against_cge",
                        "passed": bbox_rel <= 0.30,
                        "severity": "medium",
                        "detail": f"Mean bbox relative error vs CGE = {bbox_rel:.4f}",
                        "value": round(bbox_rel, 6),
                    }
                )
                if bbox_rel > 0.30:
                    warnings.append("Bounding-box dimensions drift from CGE intent.")

        score_terms: List[float] = []
        if coverage is not None:
            score_terms.append(max(0.0, min(1.0, coverage)))
        if mean_rel is not None:
            score_terms.append(max(0.0, 1.0 - min(mean_rel, 1.0)))
        if bbox_rel is not None:
            score_terms.append(max(0.0, 1.0 - min(bbox_rel, 1.0)))
        if body_ok is not None:
            score_terms.append(1.0 if body_ok else 0.0)
        score_terms.append(1.0 if bbox_positive else 0.0)
        medium_score = float(mean(score_terms)) if score_terms else 0.0

        high_fail = any(c.get("severity") == "high" and c.get("passed") is False for c in checks)
        medium_pass = (not high_fail) and medium_score >= 0.55

        return {
            "passed": bool(medium_pass),
            "score": round(medium_score, 6),
            "checks": checks,
            "warnings": warnings,
            "param_drift": drift,
            "missing_parameters": missing_parameters,
            "expected_parameter_count": len(expected_params),
            "matched_parameter_count": matched,
        }

    def _extract_expected_numeric_parameters(self, cge: Optional[Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if cge is None or not hasattr(cge, "parameters"):
            return out
        params = getattr(cge, "parameters", {}) or {}
        for name, param in params.items():
            value = getattr(param, "value", None) if param is not None else None
            f = self._as_float(value)
            if f is not None:
                out[str(name)] = f
        return out

    def _find_observed_param_value(
        self,
        *,
        parameter_name: str,
        observed_summary: Dict[str, Any],
        observed_variables: Dict[str, Any],
    ) -> tuple[Optional[str], Optional[float]]:
        pname = str(parameter_name).strip().lower()
        if not pname:
            return None, None

        keys = list(observed_summary.keys())
        # strongest: exact-key endings like "box.length" or "box_2.length"
        ranked: List[str] = []
        for k in keys:
            kl = str(k).lower()
            if kl == pname or kl.endswith(f".{pname}"):
                ranked.append(k)
        for k in keys:
            kl = str(k).lower()
            if k not in ranked and pname in kl:
                ranked.append(k)

        for key in ranked:
            raw = observed_summary.get(key)
            value = raw
            if isinstance(raw, str) and raw.startswith("$"):
                value = observed_variables.get(raw[1:])
            f = self._as_float(value)
            if f is not None:
                return key, f
        return None, None

    def _as_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _hard_verify_tool_state(
        self,
        *,
        state: CadSessionState,
        registry: ToolRegistry,
        stl_path: Optional[str],
        expected_body_count: Optional[int],
    ) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []
        warnings: List[str] = []
        bbox_dims: Optional[Dict[str, float]] = None
        body_count: Optional[int] = None

        obj = state.get_object(state.active_object)
        if obj is None:
            checks.append({"name": "has_geometry", "passed": False, "detail": "No active object in CAD session."})
            checks.append({"name": "execution", "passed": False, "detail": "No geometry to inspect."})
            self._append_stl_check(stl_path, checks, warnings)
            return self._hard_result(False, checks, warnings, body_count, bbox_dims)

        checks.append({"name": "execution", "passed": True, "detail": "Tool construction completed."})

        try:
            solids = obj.solids().vals() if hasattr(obj, "solids") else []
            body_count = len(solids) if solids is not None else 0
            has_geom = bool(body_count and body_count > 0)
            checks.append(
                {"name": "has_geometry", "passed": has_geom, "detail": f"{body_count} solid(s) found."}
            )
            if not has_geom:
                warnings.append("No solid bodies found in active object.")
        except Exception as exc:
            checks.append({"name": "has_geometry", "passed": False, "detail": str(exc)})
            warnings.append(f"Could not inspect solids: {exc}")

        m = registry.execute("measure", {"object_name": state.active_object or ""})
        if m.success and m.data:
            bb = m.data.get("bounding_box") or {}
            bbox_dims = {
                "x": float(bb.get("x", 0.0) or 0.0),
                "y": float(bb.get("y", 0.0) or 0.0),
                "z": float(bb.get("z", 0.0) or 0.0),
            }
            vol_proxy = bbox_dims["x"] * bbox_dims["y"] * bbox_dims["z"]
            checks.append(
                {
                    "name": "positive_volume",
                    "passed": vol_proxy > 1e-12,
                    "detail": f"Bounding box volume proxy = {vol_proxy:.6g}  dims={bbox_dims}",
                }
            )
            degenerate = [k for k, v in bbox_dims.items() if float(v) < 1e-6]
            checks.append(
                {
                    "name": "no_degenerate_dims",
                    "passed": len(degenerate) == 0,
                    "detail": (
                        f"All dims > threshold: {bbox_dims}"
                        if len(degenerate) == 0
                        else f"Near-zero dimension(s): {degenerate} in {bbox_dims}"
                    ),
                }
            )
        else:
            checks.append(
                {
                    "name": "positive_volume",
                    "passed": False,
                    "detail": str(m.error or "measure failed"),
                }
            )
            warnings.append(str(m.error or "Could not compute bounding box from active object."))

        if expected_body_count is not None and body_count is not None:
            checks.append(
                {
                    "name": "body_count",
                    "passed": body_count == expected_body_count,
                    "detail": f"Expected {expected_body_count} body(ies), found {body_count}.",
                }
            )
            if body_count != expected_body_count:
                warnings.append(
                    f"Body count mismatch: expected {expected_body_count}, got {body_count}."
                )

        try:
            from OCC.Core.BRepCheck import BRepCheck_Analyzer

            shape = obj.val().wrapped
            analyzer = BRepCheck_Analyzer(shape)
            is_valid = bool(analyzer.IsValid())
            checks.append(
                {
                    "name": "manifold",
                    "passed": is_valid,
                    "detail": "OCC BRepCheck passed." if is_valid else "BRepCheck failed.",
                }
            )
            if not is_valid:
                warnings.append("Geometry may not be watertight (BRepCheck failed).")
        except Exception as exc:
            checks.append(
                {
                    "name": "manifold",
                    "passed": None,
                    "detail": f"Manifold check unavailable: {exc}",
                }
            )

        self._append_stl_check(stl_path, checks, warnings)
        critical = {"execution", "has_geometry", "positive_volume"}
        passed = not any(c.get("name") in critical and c.get("passed") is False for c in checks)
        return self._hard_result(passed, checks, warnings, body_count, bbox_dims)

    def _append_stl_check(self, stl_path: Optional[str], checks: List[Dict[str, Any]], warnings: List[str]) -> None:
        if not stl_path:
            checks.append({"name": "stl_exported", "passed": False, "detail": "STL path not provided."})
            warnings.append("No STL path produced by CAD run.")
            return
        p = Path(stl_path)
        exists = p.exists() and p.stat().st_size > 0
        checks.append(
            {
                "name": "stl_exported",
                "passed": exists,
                "detail": f"STL at {stl_path} {'exists' if exists else 'missing or empty'}.",
            }
        )
        if not exists:
            warnings.append(f"STL file not found or empty: {stl_path}")

    def _normalize_stl_path(self, stl_path: Optional[str]) -> Optional[str]:
        if not stl_path:
            return stl_path
        p = Path(str(stl_path)).expanduser()
        if p.is_absolute():
            return str(p)

        project_root = Path(__file__).resolve().parents[5]
        candidates = [
            self.output_dir / p,
            Path.cwd() / p,
            project_root / p,
            project_root / "notebooks" / p,
        ]
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    return str(candidate.resolve())
            except Exception:
                continue

        return str((self.output_dir / p).resolve())

    def _hard_result(
        self,
        passed: bool,
        checks: List[Dict[str, Any]],
        warnings: List[str],
        body_count: Optional[int],
        bbox_dims: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        return {
            "passed": passed,
            "checks": checks,
            "warnings": warnings,
            "body_count": body_count,
            "bounding_box": bbox_dims,
        }

    def _verification_error(self, verification: Dict[str, Any]) -> str:
        hard = verification.get("hard") or {}
        checks = hard.get("checks") or []
        failed = [c.get("name") for c in checks if c.get("passed") is False]
        if failed:
            return f"Hard verification failed: {', '.join(str(x) for x in failed)}"
        return "CAD output did not pass verification"

    def _expected_body_count(self, cge: Optional[Any]) -> Optional[int]:
        if cge is None or not hasattr(cge, "parameters"):
            return None
        params = getattr(cge, "parameters", {}) or {}
        nx = ny = None
        for name in ("array_nx", "nx", "array_x", "columns", "n_columns", "n_x"):
            p = params.get(name)
            value = getattr(p, "value", None) if p is not None else None
            if value is not None:
                try:
                    nx = int(round(float(value)))
                    break
                except (TypeError, ValueError):
                    pass
        for name in ("array_ny", "ny", "array_y", "rows", "n_rows", "n_y"):
            p = params.get(name)
            value = getattr(p, "value", None) if p is not None else None
            if value is not None:
                try:
                    ny = int(round(float(value)))
                    break
                except (TypeError, ValueError):
                    pass
        if nx is not None and ny is not None:
            return nx * ny
        for name in ("array_count", "n_pillars", "n_features", "feature_count"):
            p = params.get(name)
            value = getattr(p, "value", None) if p is not None else None
            if value is not None:
                try:
                    return int(round(float(value)))
                except (TypeError, ValueError):
                    pass
        return None


# ------------------------------------------------------------------
def _fmt_args(args: Dict[str, Any]) -> str:
    """Compact one-line summary of tool arguments for logging."""
    parts = [f"{k}={repr(v)[:40]}" for k, v in args.items()]
    s = ", ".join(parts)
    return s[:120] + ("..." if len(s) > 120 else "")

"""
Main pipeline orchestrator.

Stages:
  1  Plan     -- cheap LLM extracts intent / operations / tags
  2  Retrieve -- semantic search over RAG corpus
  3  Generate -- configurable LLM writes CadQuery code
  4  Execute  -- exec() in isolated namespace, export STL
  5  Repair   -- if execution fails, loop up to MAX_REPAIR_ATTEMPTS times
  6  Render   -- generate front/iso/top/side renders from the final STL
  7  Store    -- save all artifacts to a persistent design folder
  8  Log      -- append full run record to logs/prompt_runs.json

Entry point:

    run_pipeline(prompt=None, design_id=None, ...)

    If design_id is provided, load the existing design and return its metadata
    without regenerating anything.

    If prompt is provided, run the full pipeline, save the design, and return
    the design metadata dict.
"""

from __future__ import annotations

import json
import time
import traceback as _tb
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import (
    DEFAULT_GENERATOR_MODEL,
    LOGS_DIR,
    OUTPUT_DIR,
    MAX_REPAIR_ATTEMPTS,
)
from ..planner.planner_llm import plan as _plan
from ..generator.cad_generator import generate as _generate
from ..execution.cad_executor import execute as _execute
from ..execution.render_utils import render_stl as _render_stl
from ..repair.repair_loop import repair as _repair
from ..design_store import create_design, load_design
from ..utils import get_last_openrouter_trace


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    prompt: Optional[str] = None,
    design_id: Optional[str] = None,
    model: str = DEFAULT_GENERATOR_MODEL,
    verbose: bool = True,
    output_dir: Optional[str | Path] = None,
    stl_filename: str = "result.stl",
    return_trace: bool = False,
    agent_mode: str = "act",
    tool_mode: str = "auto",
    tools_enabled: bool = True,
    mcp_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Run the geometry generation pipeline or load an existing design.

    Args:
        prompt:       Natural-language geometry description. Required when
                      design_id is None.
        design_id:    ID of a previously generated design. When provided,
                      the pipeline is skipped and the stored metadata is
                      returned immediately.
        model:        OpenRouter model slug for the generator / repair LLM.
        verbose:      Print progress to stdout.
        output_dir:   Temporary directory for intermediate artifacts.
        stl_filename: Filename for the intermediate STL (before archiving).
        agent_mode:   "act" routes through the CadQuery agent runtime.
        tool_mode:    "auto" enables tools + code fallback; "off" disables tools.
        tools_enabled: Master switch for structured tool calling.
        mcp_enabled:  Enable optional MCP manager wiring in agent runtime.

    Returns:
        A dict with keys:
            design_id      -- str
            stl_path       -- path to part.stl inside the design folder
            render_path    -- path to render.png (or None)
            render_iso_path-- path to render_iso.png (or None)
            render_top_path-- path to render_top.png (or None)
            render_side_path-- path to render_side.png (or None)
            cad_code_path  -- path to cad_code.py inside the design folder
            prompt         -- the original prompt
            success        -- bool
            error          -- error message string, or None
    """
    # ── Load existing design ──────────────────────────────────────────────────
    if design_id is not None:
        meta = load_design(design_id)
        if meta is None:
            return _failure_result(f"Design '{design_id}' not found.")
        return _meta_to_result(meta)

    if not prompt:
        return _failure_result("Either prompt or design_id must be provided.")

    # ── Run full pipeline ─────────────────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    _log = _vprint if verbose else _noop

    run_log: Dict[str, Any] = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "prompt":             prompt,
        "model":              model,
        "agent_mode":         agent_mode,
        "tool_mode":          tool_mode,
        "planner_output":     None,
        "retrieved_examples": [],
        "decision_trace":     [],
        "generated_code":     None,
        "execution_result":   None,
        "repair_attempts":    0,
        "final_code":         None,
        "stl_path":           None,
        "success":            False,
        "total_seconds":      0.0,
        "stage_timings":      {},
        "llm_traces":         [],
        "llm_usage_totals":   {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    t0 = time.perf_counter()
    final_code: Optional[str] = None
    stl_path: Optional[str] = None

    try:
        if str(agent_mode or "").lower() == "act":
            _log("\n[1/4] Agent runtime ...")
            t_agent = time.perf_counter()
            from forks.cadquery_agent import run_agent as _run_agent

            agent_res = _run_agent(
                request=prompt,
                mode="act",
                stream=False,
                tools_enabled=bool(tools_enabled and str(tool_mode).lower() != "off"),
                mcp_enabled=bool(mcp_enabled),
                model=model,
                output_dir=str(out_dir),
                stl_filename=stl_filename,
                verbose=verbose,
            )
            run_log["stage_timings"]["agent_seconds"] = round(time.perf_counter() - t_agent, 4)
            run_log["planner_output"] = dict(agent_res.get("planner_output") or {})
            run_log["retrieved_examples"] = list(agent_res.get("retrieved_examples") or [])
            run_log["decision_trace"] = list(agent_res.get("decision_trace") or [])
            run_log["generated_code"] = agent_res.get("code")
            run_log["final_code"] = agent_res.get("code")
            run_log["stl_path"] = agent_res.get("stl_path")
            run_log["success"] = bool(agent_res.get("success"))
            run_log["llm_traces"] = [dict(item) for item in list(agent_res.get("llm_traces") or [])]
            agent_usage = agent_res.get("llm_usage_totals")
            if isinstance(agent_usage, dict):
                run_log["llm_usage_totals"] = {
                    "prompt_tokens": int(agent_usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(agent_usage.get("completion_tokens") or 0),
                    "total_tokens": int(agent_usage.get("total_tokens") or 0),
                }

            final_code = str(agent_res.get("code") or "")
            stl_path = agent_res.get("stl_path")
            run_log["execution_result"] = {
                "success": bool(agent_res.get("success")),
                "error": agent_res.get("error"),
                "stl_path": stl_path,
            }
            if run_log["success"]:
                _log(f"      OK  ->  {stl_path}")
            else:
                _short = str(agent_res.get("error") or "agent execution failed")[:180].replace("\n", " ")
                _log(f"      FAIL: {_short}")
        else:
            # 1. Plan
            _log("\n[1/6] Planning ...")
            t_plan = time.perf_counter()
            planner_out = _plan(prompt)
            run_log["stage_timings"]["plan_seconds"] = round(time.perf_counter() - t_plan, 4)
            plan_trace = get_last_openrouter_trace()
            if plan_trace:
                plan_trace["stage"] = "plan"
                run_log["llm_traces"].append(plan_trace)
                _accumulate_usage(run_log["llm_usage_totals"], plan_trace.get("usage"))
            run_log["planner_output"] = planner_out
            _log(
                f"      intent={planner_out.get('intent')!r}  "
                f"ops={planner_out.get('operations')}  "
                f"difficulty={planner_out.get('difficulty')!r}"
            )

            # 2. Retrieve
            _log("[2/6] Retrieving examples ...")
            t_retrieve = time.perf_counter()
            from ..retrieval.retriever import get_retriever  # lazy import (numpy optional)

            retriever = get_retriever()
            retrieved: List[dict] = retriever.retrieve(prompt, plan=planner_out)
            run_log["stage_timings"]["retrieve_seconds"] = round(time.perf_counter() - t_retrieve, 4)
            run_log["retrieved_examples"] = [
                {
                    "title":  c.get("title", ""),
                    "source": c.get("source", ""),
                    "score":  round(c.get("score", 0.0), 4),
                    "tags":   c.get("metadata", {}).get("tags", []),
                }
                for c in retrieved
            ]

            # 3. Generate
            _log(f"[3/6] Generating code ({model}) ...")
            t_generate = time.perf_counter()
            code = _generate(prompt, retrieved, model=model)
            run_log["stage_timings"]["generate_seconds"] = round(time.perf_counter() - t_generate, 4)
            gen_trace = get_last_openrouter_trace()
            if gen_trace:
                gen_trace["stage"] = "generate"
                run_log["llm_traces"].append(gen_trace)
                _accumulate_usage(run_log["llm_usage_totals"], gen_trace.get("usage"))
            run_log["generated_code"] = code
            _log(f"      {len(code.splitlines())} lines generated")

            # 4. Execute
            _log("[4/6] Executing ...")
            t_execute = time.perf_counter()
            exec_res = _execute(code, output_dir=out_dir, stl_filename=stl_filename)
            run_log["stage_timings"]["execute_seconds"] = round(time.perf_counter() - t_execute, 4)
            run_log["execution_result"] = {
                "success":  exec_res["success"],
                "error":    exec_res.get("error"),
                "stl_path": exec_res.get("stl_path"),
            }

            if exec_res["success"]:
                final_code = code
                stl_path = exec_res["stl_path"]
                run_log.update(final_code=code, stl_path=stl_path, success=True)
                _log(f"      OK  ->  {stl_path}")
            else:
                _short = (exec_res.get("error") or "")[:160].replace("\n", " ")
                _log(f"      FAIL: {_short}")

                # 5. Repair
                _log(f"[5/6] Repair loop (max {MAX_REPAIR_ATTEMPTS} attempts) ...")
                t_repair = time.perf_counter()
                repair_res = _repair(
                    prompt=prompt,
                    code=code,
                    error=exec_res["error"] or "",
                    retrieved=retrieved,
                    model=model,
                    max_retries=MAX_REPAIR_ATTEMPTS,
                    output_dir=str(out_dir),
                )
                run_log.update(
                    repair_attempts=repair_res["repair_attempts"],
                    final_code=repair_res["code"],
                    stl_path=repair_res["stl_path"],
                    success=repair_res["success"],
                )
                run_log["stage_timings"]["repair_seconds"] = round(time.perf_counter() - t_repair, 4)
                repair_traces = list(repair_res.get("llm_traces") or [])
                for item in repair_traces:
                    trace = dict(item)
                    trace["stage"] = "repair"
                    run_log["llm_traces"].append(trace)
                    _accumulate_usage(run_log["llm_usage_totals"], trace.get("usage"))
                final_code = repair_res["code"]
                stl_path = repair_res["stl_path"]
                status = "repaired" if repair_res["success"] else "repair failed"
                _log(f"      {status} after {repair_res['repair_attempts']} attempt(s)")

    except Exception:
        run_log["execution_result"] = {
            "success": False,
            "error":   _tb.format_exc(),
            "stl_path": None,
        }
        if verbose:
            _tb.print_exc()

    run_log["total_seconds"] = round(time.perf_counter() - t0, 2)
    _append_log(run_log)

    # 6. Render
    render_path: Optional[str] = None
    render_iso_path: Optional[str] = None
    render_top_path: Optional[str] = None
    render_side_path: Optional[str] = None
    if stl_path:
        _log("[6/6] Rendering ...")
        t_render = time.perf_counter()
        renders = _render_stl(stl_path, str(out_dir))
        render_path = renders.get("render_path")
        render_iso_path = renders.get("render_iso_path")
        render_top_path = renders.get("render_top_path")
        render_side_path = renders.get("render_side_path")
        run_log["stage_timings"]["render_seconds"] = round(time.perf_counter() - t_render, 4)
        _log(
            f"      render={render_path}  iso={render_iso_path}  "
            f"top={render_top_path}  side={render_side_path}"
        )

    # 7. Store design
    t_store = time.perf_counter()
    meta = create_design(
        prompt=prompt,
        cad_code=final_code or "",
        stl_path=stl_path,
        render_path=render_path,
        render_iso_path=render_iso_path,
        render_top_path=render_top_path,
        render_side_path=render_side_path,
        parameters_detected=run_log.get("planner_output") or {},
        status="success" if run_log["success"] else "failed",
        run_trace=run_log,
    )
    run_log["stage_timings"]["store_seconds"] = round(time.perf_counter() - t_store, 4)

    _log(
        f"\n{'Done' if run_log['success'] else 'Failed'}  "
        f"in {run_log['total_seconds']:.1f}s  "
        f"design={meta['design_id']}\n"
    )

    return _meta_to_result(
        meta,
        error=None if run_log["success"] else "Pipeline failed",
        trace=run_log if return_trace else None,
    )


# ── Result helpers ────────────────────────────────────────────────────────────

def _meta_to_result(
    meta: Dict[str, Any],
    error: Optional[str] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "design_id":       meta["design_id"],
        "prompt":          meta.get("prompt", ""),
        "stl_path":        meta.get("stl_path"),
        "render_path":     meta.get("render_path"),
        "render_iso_path": meta.get("render_iso_path"),
        "render_top_path": meta.get("render_top_path"),
        "render_side_path": meta.get("render_side_path"),
        "cad_code_path":   meta.get("cad_code_path"),
        "success":         meta.get("status") == "success",
        "error":           error,
    }
    if trace is not None:
        result["trace"] = trace
    return result


def _failure_result(error: str) -> Dict[str, Any]:
    return {
        "design_id":       None,
        "prompt":          None,
        "stl_path":        None,
        "render_path":     None,
        "render_iso_path": None,
        "render_top_path": None,
        "render_side_path": None,
        "cad_code_path":   None,
        "success":         False,
        "error":           error,
    }


def _accumulate_usage(target: Dict[str, int], usage: Any) -> None:
    if not isinstance(target, dict) or not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] = int(target.get(key) or 0) + int(usage.get(key) or 0)


# ── Logging ───────────────────────────────────────────────────────────────────

def _append_log(entry: Dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "prompt_runs.json"

    existing: List[dict] = []
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(entry)
    log_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vprint(msg: str) -> None:
    print(msg)


def _noop(*_: Any) -> None:
    pass

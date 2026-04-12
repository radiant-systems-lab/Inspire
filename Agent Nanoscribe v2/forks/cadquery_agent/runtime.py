from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt2cad.config import DEFAULT_GENERATOR_MODEL, MAX_REPAIR_ATTEMPTS
from prompt2cad.generator.cad_generator import generate as generate_code
from prompt2cad.planner.planner_llm import plan as plan_prompt
from prompt2cad.repair.repair_loop import repair as repair_code
from prompt2cad.utils import get_last_openrouter_trace

from .conversation import Conversation
from .executor import execute_code_safely
from .llm_client import LLMClient
from .session import CadSessionState
from .tools import create_default_registry
from .types import AgentRunResult, DecisionTraceEvent, ToolCall


TOOL_PLANNER_SYSTEM = """You are a CadQuery tool planner.
Return tool calls that can satisfy the user request.
If no tools apply, return an empty tool_calls list.
"""

NON_ADDITIVE_BOOLEAN_OPS = {"cut", "subtract", "difference", "intersect", "intersection", "common"}



def run_agent(
    request: Any,
    *,
    mode: str = "act",
    stream: bool = True,
    tools_enabled: bool = True,
    mcp_enabled: bool = True,
    model: str = DEFAULT_GENERATOR_MODEL,
    output_dir: str | Path = "output",
    stl_filename: str = "result.stl",
    verbose: bool = False,
    max_tool_turns: int = 8,
) -> Dict[str, Any]:
    prompt = _extract_prompt(request)
    out_dir = str(output_dir)

    conversation = Conversation()
    conversation.add_user(prompt)
    state = CadSessionState()
    registry = create_default_registry(state, output_dir=out_dir, stl_filename=stl_filename)

    llm_traces: List[Dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    decision_trace: List[DecisionTraceEvent] = []

    try:
        planner_output = plan_prompt(prompt)
        plan_trace = get_last_openrouter_trace()
        if plan_trace:
            plan_trace = dict(plan_trace)
            plan_trace["stage"] = "plan"
            llm_traces.append(plan_trace)
            _accumulate_usage(usage, plan_trace.get("usage"))
    except Exception:
        planner_output = {
            "intent": "profile_extrude",
            "operations": ["Workplane"],
            "geometry_type": "solid",
            "tags": [],
            "difficulty": "moderate",
        }
    decision_trace.append(
        DecisionTraceEvent(
            step=1,
            intent=str(planner_output.get("intent") or "unknown"),
            selected_tools=[],
            why="planner intent extraction",
            confidence=0.7,
            result="planned",
            metadata={"operations": list(planner_output.get("operations") or [])},
        )
    )

    try:
        from prompt2cad.retrieval.retriever import get_retriever  # lazy import (numpy optional)

        retriever = get_retriever()
        retrieved = retriever.retrieve(prompt, plan=planner_output)
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

    code = ""
    stl_path: Optional[str] = None
    error: Optional[str] = None

    if mode.lower() == "act" and tools_enabled:
        client = LLMClient(model)
        planned_calls: List[ToolCall] = []
        successful_tool_calls: List[ToolCall] = []
        try:
            response = client.send_with_tools(
                [{"role": "user", "content": _tool_planner_user_prompt(prompt, planner_output)}],
                system=TOOL_PLANNER_SYSTEM,
                tools=registry.to_openai_schema(),
            )
            if response.trace:
                trace = dict(response.trace)
                trace["stage"] = "tool_plan"
                llm_traces.append(trace)
                _accumulate_usage(usage, trace.get("usage"))
            planned_calls = list(response.tool_calls)
        except Exception as exc:
            error = f"tool planning failed: {exc}"

        if not planned_calls:
            planned_calls = _heuristic_tool_calls(prompt)

        for idx, call in enumerate(planned_calls[: max(1, int(max_tool_turns))], start=1):
            result = registry.execute(call.name, call.arguments)
            result_msg = result.output if result.success else result.error
            conversation.add_tool(call.name, result_msg, call_id=call.id)
            decision_trace.append(
                DecisionTraceEvent(
                    step=idx,
                    intent=str(planner_output.get("intent") or "tool_act"),
                    selected_tools=[call.name],
                    why="tool plan execution",
                    confidence=0.72 if result.success else 0.35,
                    result="success" if result.success else "failed",
                    metadata={"tool_result": result.data, "tool_error": result.error},
                )
            )
            if not result.success:
                error = result.error or error
                continue
            successful_tool_calls.append(call)
            stl_path = str((result.data or {}).get("path") or "") or stl_path

        # Completeness gate for additive assemblies:
        # if multiple objects remain in session, merge them before final export.
        if _is_additive_only(successful_tool_calls) and len(state.objects) > 1:
            merged_name, merge_count, merge_error = _auto_union_all_objects(registry, state)
            if merge_error:
                error = merge_error
                decision_trace.append(
                    DecisionTraceEvent(
                        step=len(decision_trace) + 1,
                        intent=str(planner_output.get("intent") or "tool_act"),
                        selected_tools=["boolean_operation"],
                        why="auto-finalize assembly before export",
                        confidence=0.35,
                        result="failed",
                        metadata={
                            "merge_count": merge_count,
                            "object_count": len(state.objects),
                            "error": merge_error,
                        },
                    )
                )
            else:
                decision_trace.append(
                    DecisionTraceEvent(
                        step=len(decision_trace) + 1,
                        intent=str(planner_output.get("intent") or "tool_act"),
                        selected_tools=["boolean_operation"],
                        why="auto-finalize assembly before export",
                        confidence=0.78,
                        result="success",
                        metadata={
                            "merge_count": merge_count,
                            "object_count": len(state.objects),
                            "final_object": merged_name,
                        },
                    )
                )

                export_result = registry.execute(
                    "export_model",
                    {
                        "object_name": merged_name or state.active_object or "",
                        "file_path": str(Path(out_dir) / stl_filename),
                        "format": "stl",
                    },
                )
                if export_result.success:
                    stl_path = str((export_result.data or {}).get("path") or "") or stl_path
                    error = None
                    decision_trace.append(
                        DecisionTraceEvent(
                            step=len(decision_trace) + 1,
                            intent=str(planner_output.get("intent") or "tool_act"),
                            selected_tools=["export_model"],
                            why="export finalized assembly",
                            confidence=0.8,
                            result="success",
                            metadata={"path": stl_path},
                        )
                    )
                else:
                    error = export_result.error or error
                    decision_trace.append(
                        DecisionTraceEvent(
                            step=len(decision_trace) + 1,
                            intent=str(planner_output.get("intent") or "tool_act"),
                            selected_tools=["export_model"],
                            why="export finalized assembly",
                            confidence=0.3,
                            result="failed",
                            metadata={"error": export_result.error},
                        )
                    )

        if not stl_path and state.active_object:
            export_result = registry.execute(
                "export_model",
                {
                    "object_name": state.active_object,
                    "file_path": str(Path(out_dir) / stl_filename),
                    "format": "stl",
                },
            )
            if export_result.success:
                stl_path = str((export_result.data or {}).get("path") or "") or stl_path
            else:
                error = export_result.error or error
            decision_trace.append(
                DecisionTraceEvent(
                    step=len(decision_trace) + 1,
                    intent=str(planner_output.get("intent") or "tool_act"),
                    selected_tools=["export_model"],
                    why="final export attempt from active object",
                    confidence=0.72 if export_result.success else 0.28,
                    result="success" if export_result.success else "failed",
                    metadata={"path": stl_path, "error": export_result.error},
                )
            )

    if not stl_path:
        decision_trace.append(
            DecisionTraceEvent(
                step=len(decision_trace) + 1,
                intent=str(planner_output.get("intent") or "codegen"),
                selected_tools=["execute_code"],
                why="tool path did not produce STL; using code generation fallback",
                confidence=0.62,
                result="fallback_started",
                metadata={},
            )
        )
        code = generate_code(prompt, retrieved, model=model)
        gen_trace = get_last_openrouter_trace()
        if gen_trace:
            gen_trace = dict(gen_trace)
            gen_trace["stage"] = "generate"
            llm_traces.append(gen_trace)
            _accumulate_usage(usage, gen_trace.get("usage"))

        exec_result = execute_code_safely(
            code,
            output_dir=out_dir,
            stl_filename=stl_filename,
            timeout=90,
            run_preflight=True,
        )

        if exec_result.get("success"):
            stl_path = exec_result.get("stl_path")
            error = None
            decision_trace.append(
                DecisionTraceEvent(
                    step=len(decision_trace) + 1,
                    intent=str(planner_output.get("intent") or "codegen"),
                    selected_tools=["execute_code"],
                    why="code execution succeeded",
                    confidence=0.75,
                    result="success",
                    metadata={"stl_path": stl_path},
                )
            )
        else:
            repair_res = repair_code(
                prompt=prompt,
                code=code,
                error=str(exec_result.get("error") or "execution failure"),
                retrieved=retrieved,
                model=model,
                max_retries=MAX_REPAIR_ATTEMPTS,
                output_dir=out_dir,
            )
            for item in list(repair_res.get("llm_traces") or []):
                trace = dict(item)
                trace["stage"] = "repair"
                llm_traces.append(trace)
                _accumulate_usage(usage, trace.get("usage"))
            code = str(repair_res.get("code") or code)
            stl_path = repair_res.get("stl_path")
            if not repair_res.get("success"):
                error = str((repair_res.get("errors") or [exec_result.get("error")])[-1])
                decision_trace.append(
                    DecisionTraceEvent(
                        step=len(decision_trace) + 1,
                        intent=str(planner_output.get("intent") or "repair"),
                        selected_tools=["repair_loop"],
                        why="code execution failed and repair attempts exhausted",
                        confidence=0.25,
                        result="failed",
                        metadata={"error": error},
                    )
                )
            else:
                error = None
                decision_trace.append(
                    DecisionTraceEvent(
                        step=len(decision_trace) + 1,
                        intent=str(planner_output.get("intent") or "repair"),
                        selected_tools=["repair_loop"],
                        why="repair loop recovered execution",
                        confidence=0.68,
                        result="success",
                        metadata={"stl_path": stl_path},
                    )
                )

    if not code:
        code = state.render_command_log()

    success = bool(stl_path)
    if not success and not error:
        error = "agent failed to produce STL"

    result = AgentRunResult(
        success=success,
        code=code,
        stl_path=stl_path,
        error=error,
        decision_trace=decision_trace,
        planner_output=planner_output,
        retrieved_examples=retrieved_summary,
        llm_traces=llm_traces,
        llm_usage_totals=usage,
    )
    return result.to_dict()



def _extract_prompt(request: Any) -> str:
    if isinstance(request, str):
        return request.strip()
    if isinstance(request, dict):
        for key in ("prompt", "request", "goal", "text"):
            value = request.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(request or "").strip()



def _tool_planner_user_prompt(prompt: str, planner_output: Dict[str, Any]) -> str:
    return (
        f"User prompt: {prompt}\n"
        f"Planner output: {planner_output}\n"
        "Choose tool calls that can construct geometry and export STL."
    )



def _heuristic_tool_calls(prompt: str) -> List[ToolCall]:
    low = prompt.lower()
    number_values = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", low)]

    calls: List[ToolCall] = []
    if "box" in low or "cube" in low:
        length = number_values[0] if len(number_values) > 0 else 10.0
        width = number_values[1] if len(number_values) > 1 else length
        height = number_values[2] if len(number_values) > 2 else length
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name="create_primitive",
                arguments={
                    "shape_type": "box",
                    "length": length,
                    "width": width,
                    "height": height,
                    "label": "box_1",
                },
            )
        )
    elif "cylinder" in low or "pillar" in low or "post" in low:
        radius = number_values[0] if len(number_values) > 0 else 1.0
        height = number_values[1] if len(number_values) > 1 else 5.0
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name="create_primitive",
                arguments={
                    "shape_type": "cylinder",
                    "radius": radius,
                    "height": height,
                    "label": "cylinder_1",
                },
            )
        )
    elif "sphere" in low:
        radius = number_values[0] if number_values else 2.0
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name="create_primitive",
                arguments={"shape_type": "sphere", "radius": radius, "label": "sphere_1"},
            )
        )

    if calls:
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name="export_model",
                arguments={"format": "stl"},
            )
        )

    return calls



def _accumulate_usage(target: Dict[str, int], usage: Any) -> None:
    if not isinstance(target, dict) or not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] = int(target.get(key) or 0) + int(usage.get(key) or 0)


def _is_additive_only(calls: List[ToolCall]) -> bool:
    for call in calls:
        if str(call.name or "").strip() != "boolean_operation":
            continue
        op = str((call.arguments or {}).get("operation") or "").strip().lower()
        if op in NON_ADDITIVE_BOOLEAN_OPS:
            return False
    return True


def _auto_union_all_objects(registry: Any, state: CadSessionState) -> tuple[str, int, Optional[str]]:
    names = [str(name) for name in state.objects.keys() if str(name).strip()]
    if not names:
        return "", 0, "No objects available for auto-union"
    if len(names) == 1:
        return names[0], 0, None

    base = state.active_object if state.active_object in state.objects else names[0]
    remaining = [name for name in names if name != base]
    merge_count = 0

    for idx, name in enumerate(remaining, start=1):
        result_name = f"auto_union_{idx}"
        merged = registry.execute(
            "boolean_operation",
            {
                "operation": "fuse",
                "base_object": base,
                "tool_object": name,
                "result_name": result_name,
            },
        )
        if not merged.success:
            return base, merge_count, str(merged.error or "auto-union failed")
        merged_name = str((merged.data or {}).get("name") or "").strip()
        if not merged_name:
            return base, merge_count, "auto-union produced unnamed result"
        base = merged_name
        merge_count += 1

    return base, merge_count, None

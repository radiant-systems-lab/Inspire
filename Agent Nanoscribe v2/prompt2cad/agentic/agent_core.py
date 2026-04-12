"""
NanoscribeAgent -- true tool-calling agent for geometry design.

The LLM drives every decision. Tools are atomic, deterministic functions.
No fixed pipeline: the agent reasons about what to call next at each step
based on current CGE state, prior tool results, and confidence.

Entry point:
    agent = NanoscribeAgent(model="anthropic/claude-sonnet-4-6")
    result = agent.run("Design a woodpile photonic crystal from this paper: path/to/paper.pdf")
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..utils import (
    assistant_tool_call_message,
    call_openrouter_agent,
    get_last_openrouter_trace,
    tool_result_message,
)
from .cge import CanonicalGeometry

# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    success: bool
    cge: Optional[CanonicalGeometry]
    cad_code: Optional[str]
    verification: Optional[Dict[str, Any]]
    message: str
    decision_trace: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    total_tokens: Dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    elapsed_s: float = 0.0
    sweep_results: Optional[List[Dict[str, Any]]] = None  # populated by execute_sweep


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an autonomous nanoscale geometry design agent. Your goal is to produce:
  1. A fully-grounded Canonical Geometry Encoding (CGE) for the requested design
  2. Verified CadQuery CAD code that implements the CGE
  3. Optionally: printability assessment and a GWL job file for the Nanoscribe printer

You have tools. You decide what to call next at every step -- there is no fixed sequence.

== RULES ==

EXTRACTION
- Never set a CGE field as "inferred" without calling flag_issue() to document the inference.
- If a required parameter is missing from the source, call request_clarification() -- do not guess.
- When you find conflicting values for the same parameter, call flag_issue() with both values; do not silently pick one.
- Always call resolve_units() when a parameter has a unit string -- never trust raw unit text.

CGE COMPLETENESS
- Before generating CAD, call get_cge_status(). If ready_for_cad is False, you must resolve all unset fields first.
- "Default" source is only acceptable for non-critical fields (e.g., array count when the paper focuses on unit cell).
- A CGE with any required field "unset" must not proceed to CAD.

CAD GENERATION & VERIFICATION
- generate_cad() runs the full pipeline: hard assertions, medium deterministic checks, and
  soft vision analysis per render view. All results are bundled into the returned verification dict.
- The verification dict contains:
  hard (blocking checks),
  medium (bbox/feature/body consistency + extracted-vs-intended parameter drift),
  extracted_cge (params the code actually used),
  geometry_metrics (volume, surface area, dims, face/edge counts),
  soft/vision (per_view JSON + synthesis),
  renders (raw render paths), and intended_cge.
- Compare extracted_cge against intended_cge to catch parameter drift or missing features.
- Use medium + soft together to assess structural correctness.
- If verification exposes a problem, send a specific corrective instruction back to generate_cad().
- Do not retry with the same notes if the same failure repeats -- change your diagnosis.

POST-CAD ANALYSIS
- After a successful generate_cad(), call analyze_geometry() to compute ~55 printability-relevant metrics
  (slenderness, overhang, feature_spacing, fragility_score, anchor_score, etc.).
- Call predict_printability() after analyze_geometry() to get a success probability and failure modes.
  If risk is high (success_probability < 0.55), explain the failure modes and suggest geometry changes.
- Call generate_gwl_job() when the user wants a print-ready GWL job file for the Nanoscribe device.
  This requires DescribeX to be installed; pass the path to DescribeX.exe. The tool returns the gwl_path
  and files_dir that together form the complete job package.

ESCALATION
- When confidence is below 0.6 on a key parameter, flag it and ask the user.
- You may call request_clarification() mid-loop at any time -- it does not end the session.

== WORKFLOW GUIDANCE (not a script -- use judgment) ==

For PDF input:       parse_pdf -> extract_parameters -> build CGE -> fill gaps -> check completeness -> generate CAD -> verify_cad_hard
For NL/text input:   interpret description -> build CGE -> fill gaps -> check completeness -> generate CAD -> verify_cad_hard
For structured input: parse fields into CGE -> check completeness -> generate CAD -> verify_cad_hard
For sweep request:   build CGE as normal -> plan_sweep(parameter, values) -> execute_sweep()
                     execute_sweep() calls prompt2cad ONCE for parameterized base code, then substitutes.
                     Do NOT call generate_cad() separately when doing a sweep.
For full print pipeline: generate CAD -> analyze_geometry -> predict_printability -> generate_gwl_job

The CGE is your primary artifact. Every parameter should be traceable to its source.
When you are done, present: final CGE summary, verification results, and the CAD code.
"""


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "parse_pdf",
            "description": "Parse a PDF file and extract its text, tables, and figure list. Returns raw content -- does not interpret or extract parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the PDF file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_parameters",
            "description": "Extract geometry parameters from a text chunk. Returns each found parameter with its value, unit, source location, and confidence. Lists unfound parameters explicitly in 'unresolved'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text chunk to extract from (e.g. methods section, table content)."},
                    "target_parameters": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of parameter names to look for, based on what the geometry needs.",
                    },
                    "context_hint": {"type": "string", "description": "Short description of the geometry to guide extraction (e.g. 'woodpile photonic crystal')."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_reference_parameter",
            "description": "Search the full document for all mentions of a specific parameter. Useful when initial extraction missed it. Returns all occurrences and any conflicts between sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_text": {"type": "string", "description": "Full document text to search."},
                    "parameter_name": {"type": "string", "description": "Parameter to search for (e.g. 'pillar_height', 'lattice_constant')."},
                },
                "required": ["document_text", "parameter_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_units",
            "description": "Normalize a value+unit string to a standard unit. Catches nm/umm confusion, period vs half-period, and other common ambiguities. Always call this before setting a CGE field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value to normalize."},
                    "unit_string": {"type": "string", "description": "Raw unit string from source (e.g. 'nm', 'umm', 'um', 'micron')."},
                    "target_unit": {"type": "string", "description": "Desired output unit (e.g. 'um')."},
                },
                "required": ["value", "unit_string", "target_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_cge",
            "description": "Initialize a new empty Canonical Geometry Encoding with a free-form geometry description. Call this once you understand what the geometry is.",
            "parameters": {
                "type": "object",
                "properties": {
                    "geometry_description": {
                        "type": "string",
                        "description": "Rich natural language description of the geometry. This is the ground truth -- be specific about topology, symmetry, arrangement, and any relational facts.",
                    },
                },
                "required": ["geometry_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_cge_field",
            "description": "Set or update a parameter in the CGE. Every field must have provenance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Parameter name (e.g. 'pillar_radius', 'lattice_constant_a')."},
                    "value": {"description": "Parameter value (number, string, or null for unset)."},
                    "unit": {"type": "string", "description": "Unit string (e.g. 'um', 'nm', 'deg'). Null if dimensionless."},
                    "source": {
                        "type": "string",
                        "enum": ["extracted", "inferred", "default", "human", "unset"],
                        "description": "Where this value came from.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0-1.0. Use <0.6 for uncertain values.",
                    },
                    "provenance_location": {"type": "string", "description": "Where in the source this was found (e.g. 'Table 1, row 3' or 'Figure 2 caption')."},
                    "provenance_raw_text": {"type": "string", "description": "The exact raw text from the source."},
                    "notes": {"type": "string", "description": "Any notes about this field (ambiguities, inference reasoning)."},
                },
                "required": ["name", "value", "source", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_cge_constraint",
            "description": "Add a relational constraint to the CGE -- a fact that cannot be captured by a scalar parameter. e.g. 'alternating layers are orthogonal', 'holes span full slab thickness'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Plain language description of the constraint."},
                    "source": {"type": "string", "enum": ["extracted", "inferred", "human"], "description": "Where this constraint came from."},
                    "confidence": {"type": "number", "description": "Confidence 0.0-1.0."},
                    "provenance_raw_text": {"type": "string", "description": "Supporting evidence from the source."},
                },
                "required": ["description", "source", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cge_status",
            "description": "Get the current CGE completeness report. Check this before generating CAD. Returns: completeness score, unset fields, low-confidence fields, and whether ready_for_cad is True.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_cad",
            "description": (
                "Invoke the standalone CAD tool on the current CGE. "
                "The CAD tool handles retrieval context, tool-based generation, fallback repair, "
                "and integrated hard+soft verification internally. "
                "Only call when get_cge_status() shows ready_for_cad=True."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {"type": "string", "description": "Optional notes to guide code generation (e.g. specific topology hints, known CadQuery patterns to use or avoid)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_cad_hard",
            "description": (
                "Run the full verification pipeline on the generated CAD. "
                "Returns: hard assertions (volume, manifold, body count), "
                "extracted_cge (parameters parsed from code), "
                "geometry_metrics (volume, surface area, aspect ratio, face/edge/vertex counts), "
                "and vision (per-view structured JSON from renders + synthesis). "
                "If generate_cad() already ran, this returns the cached result from that run."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_clarification",
            "description": "Ask the user a specific question when you cannot resolve a parameter or conflict from the source. Blocks until answered. Use this instead of guessing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The specific question to ask."},
                    "context": {"type": "string", "description": "Brief context explaining why you need this information."},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_issue",
            "description": "Flag an anomaly, conflict, or low-confidence inference. Does not stop execution -- records the issue for the user to review. Always call this before using an inferred or uncertain value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "description": "The parameter or aspect this issue concerns."},
                    "issue": {"type": "string", "description": "Description of the issue."},
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"], "description": "How serious is this issue?"},
                },
                "required": ["field", "issue", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_sweep",
            "description": (
                "Set up a parameter sweep over the current CGE. Validates that the parameter exists, "
                "records the sweep plan. Call execute_sweep() afterwards to generate the CAD variants. "
                "Use this when the user wants to explore a range of values for one parameter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter_name": {
                        "type": "string",
                        "description": "Name of the CGE parameter to sweep (must already be set in the CGE).",
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numeric values to sweep over.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional human-readable label for the sweep (e.g. 'pillar height sweep').",
                    },
                },
                "required": ["parameter_name", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sweep",
            "description": (
                "Execute a planned parameter sweep. Generates parameterized base CAD code ONCE via prompt2cad, "
                "then derives all variants by substituting the swept parameter value -- no further LLM CAD calls. "
                "Each variant is executed, hard-verified, and rendered. Call plan_sweep() first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": "Optional guidance for base code generation (e.g. topology hints).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_geometry",
            "description": (
                "Compute ~55 printability-relevant geometry metrics from the current STL. "
                "Metrics include: volume, surface_area, aspect ratios, slenderness, overhang_max_angle_deg, "
                "overhang_area_fraction, feature_spacing_min, fragility_score, anchor_score, base_stability_ratio, "
                "num_disconnected_solids, array_pitch, com_offset_norm, shape_entropy, feature_thickness_min, "
                "unsupported_height. Call after generate_cad() succeeds. Required before predict_printability()."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_printability",
            "description": (
                "Predict Nanoscribe two-photon lithography fabrication success for a given "
                "geometry and print recipe. Uses a trained ML model (XGBoost) when available, "
                "otherwise falls back to heuristic rules. "
                "Returns: success_probability (0-1), uncertainty (0-1), risk_score, "
                "failure_modes, and recommendations. "
                "Must call analyze_geometry() first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slicing_distance_um": {
                        "type": "number",
                        "description": "Slice thickness in micrometres. Default 0.2.",
                    },
                    "hatching_distance_um": {
                        "type": "number",
                        "description": "Hatch line spacing in micrometres. Default 0.2.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_gwl_job",
            "description": (
                "Convert the current STL into a Nanoscribe GWL job file using DescribeX. "
                "Writes {stem}_data.gwl + {stem}_files/ to the output directory. "
                "Both the .gwl and the _files/ folder must stay together -- they form the complete job package. "
                "Requires DescribeX.exe installed on the system."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slicing_distance_um": {
                        "type": "number",
                        "description": "Distance between slicing planes in micrometres. Default 0.1.",
                    },
                    "hatching_distance_um": {
                        "type": "number",
                        "description": "Hatch line spacing inside each slice in micrometres. Default 0.1.",
                    },
                    "power_scaling": {
                        "type": "number",
                        "description": "Global laser power scale factor. Default 1.0.",
                    },
                    "describe_exe_path": {
                        "type": "string",
                        "description": "Absolute path to DescribeX.exe. Required.",
                    },
                },
                "required": ["describe_exe_path"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def _substitute_parameter(code: str, param_name: str, new_value: float) -> str:
    """
    Replace the top-level variable assignment for `param_name` in the code.

    Matches lines of the form:
        param_name = <number>          # optional comment
        param_name = <number>  # ...

    If no assignment line is found, prepends one -- the code will still run
    but the substitution should be treated as approximate.
    """
    pattern = rf'^(\s*{re.escape(param_name)}\s*=\s*)[\d.eE+\-]+(\b.*)?$'
    replacement = rf'\g<1>{new_value}\g<2>'
    new_code, n_subs = re.subn(pattern, replacement, code, flags=re.MULTILINE)
    if n_subs == 0:
        # Variable not found as a top-level assignment -- prepend it
        new_code = f"{param_name} = {new_value}  # injected by sweep\n" + code
    return new_code


# ---------------------------------------------------------------------------
# NanoscribeAgent
# ---------------------------------------------------------------------------

class NanoscribeAgent:
    """
    True tool-calling agent for nanoscale geometry design.

    The LLM decides what to call at each step. Tools are executed deterministically.
    The agent loops until it reaches a final answer or hits max_steps.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-6",
        max_steps: int = 40,
        verbose: bool = True,
        output_dir: Optional[str] = None,
        vision_model: Optional[str] = None,
        on_step: Optional[Callable[[int, str, Any], None]] = None,
    ):
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.output_dir = Path(output_dir) if output_dir else Path("output/agent_runs")
        self.vision_model = vision_model or model
        self.on_step = on_step  # optional callback(step, tool_name, tool_result)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mutable agent state -- reset on each run()
        self._cge: Optional[CanonicalGeometry] = None
        self._cad_code: Optional[str] = None
        self._render_paths: Dict[str, str] = {}
        self._document_text: str = ""
        self._hard_result: Optional[Dict] = None
        self._medium_result: Dict = {}
        self._extracted_cge: Dict = {}
        self._geometry_metrics: Dict = {}
        self._vision_analysis: Dict = {}
        self._geometry_analysis: Dict = {}
        self._printability: Dict = {}
        self._gwl_result: Dict = {}
        self._messages: List[dict] = []
        self._flags: List[Dict] = []
        self._decision_trace: List[Dict] = []
        self._step_count: int = 0
        self._pending_clarification: Optional[str] = None
        self._sweep_plan: Optional[Dict] = None
        self._sweep_results: Optional[List[Dict]] = None
        self._token_totals: Dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, goal: str) -> AgentResult:
        """Run the agent toward the goal. Returns AgentResult when done."""
        self._reset()
        self._start_time = time.time()
        self._messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]
        self._log(f"Agent started. Goal: {goal[:120]}")

        for step in range(self.max_steps):
            self._step_count = step + 1
            step_t0 = time.time()

            response = call_openrouter_agent(
                messages=self._messages,
                model=self.model,
                tools=TOOL_SCHEMAS,
                temperature=0.1,
            )

            # Capture token usage from the LLM call just made
            llm_trace = get_last_openrouter_trace()
            usage = llm_trace.get("usage", {})
            step_tokens = {
                "prompt": int(usage.get("prompt_tokens", 0)),
                "completion": int(usage.get("completion_tokens", 0)),
                "total": int(usage.get("total_tokens", 0)),
            }
            self._token_totals["prompt"] += step_tokens["prompt"]
            self._token_totals["completion"] += step_tokens["completion"]
            self._token_totals["total"] += step_tokens["total"]

            tool_calls = response.get("tool_calls", [])
            content = response.get("content", "")
            step_elapsed = round(time.time() - step_t0, 2)

            if tool_calls:
                self._messages.append(assistant_tool_call_message(tool_calls, content))
                for tc in tool_calls:
                    tool_t0 = time.time()
                    result = self._dispatch(tc)
                    # Annotate the trace entry that _dispatch already created
                    if self._decision_trace:
                        self._decision_trace[-1]["llm_tokens"] = step_tokens
                        self._decision_trace[-1]["llm_time_s"] = step_elapsed
                        self._decision_trace[-1]["tool_time_s"] = round(time.time() - tool_t0, 2)
                    self._messages.append(tool_result_message(tc["id"], result))
            else:
                self._messages.append({"role": "assistant", "content": content})
                elapsed = round(time.time() - self._start_time, 2)
                self._log(
                    f"Agent finished at step {self._step_count} "
                    f"| tokens={self._token_totals['total']} "
                    f"| elapsed={elapsed}s"
                )
                result = self._make_result(success=self._cad_code is not None, message=content)
                self._save_trace(goal, result)
                return result

        elapsed = round(time.time() - self._start_time, 2)
        result = self._make_result(
            success=False,
            message=f"Agent reached max_steps ({self.max_steps}) without completing.",
        )
        self._save_trace(goal, result)
        return result

    def _make_result(self, success: bool, message: str) -> AgentResult:
        return AgentResult(
            success=success,
            cge=self._cge,
            cad_code=self._cad_code,
            verification=self._build_verification_summary(),
            message=message,
            decision_trace=self._decision_trace,
            flags=self._flags,
            iterations=self._step_count,
            total_tokens=dict(self._token_totals),
            elapsed_s=round(time.time() - self._start_time, 2),
            sweep_results=self._sweep_results,
        )

    def _save_trace(self, goal: str, result: AgentResult) -> None:
        """Persist full trace to output_dir/trace.json for notebook inspection."""
        trace = {
            "goal": goal,
            "model": self.model,
            "success": result.success,
            "iterations": result.iterations,
            "elapsed_s": result.elapsed_s,
            "total_tokens": result.total_tokens,
            "message": result.message,
            "decision_trace": result.decision_trace,
            "flags": result.flags,
            "cge": result.cge.to_dict() if result.cge else None,
            "cad_code": result.cad_code,
            "verification": result.verification,
            "sweep_results": result.sweep_results,
        }
        path = self.output_dir / "trace.json"
        path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
        self._log(f"Trace saved: {path}")

    def provide_clarification(self, response_text: str) -> None:
        """
        Inject a user clarification response.
        Call this after run() returns with a request_clarification message,
        then call run() again (it will resume from the queued response).
        """
        self._pending_clarification = response_text

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tool_call: dict) -> Any:
        name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"].get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}

        self._record_step(name, args)
        handler = self._tool_handlers().get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = handler(**args)
            if self.on_step:
                try:
                    self.on_step(self._step_count, name, result)
                except Exception:
                    pass
            return result
        except Exception as exc:
            err = {"error": str(exc), "traceback": traceback.format_exc(limit=5)}
            self._log(f"  Tool {name} raised: {exc}")
            if self.on_step:
                try:
                    self.on_step(self._step_count, name, {"error": str(exc)})
                except Exception:
                    pass
            return err

    def _tool_handlers(self) -> Dict[str, Callable]:
        return {
            "parse_pdf": self._tool_parse_pdf,
            "extract_parameters": self._tool_extract_parameters,
            "cross_reference_parameter": self._tool_cross_reference,
            "resolve_units": self._tool_resolve_units,
            "create_cge": self._tool_create_cge,
            "set_cge_field": self._tool_set_cge_field,
            "add_cge_constraint": self._tool_add_cge_constraint,
            "get_cge_status": self._tool_get_cge_status,
            "generate_cad": self._tool_generate_cad,
            "verify_cad_hard": self._tool_verify_cad_hard,
            "request_clarification": self._tool_request_clarification,
            "flag_issue": self._tool_flag_issue,
            "plan_sweep": self._tool_plan_sweep,
            "execute_sweep": self._tool_execute_sweep,
            "analyze_geometry": self._tool_analyze_geometry,
            "predict_printability": self._tool_predict_printability,
            "generate_gwl_job": self._tool_generate_gwl_job,
        }

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_parse_pdf(self, path: str) -> Dict[str, Any]:
        from ..pdf.pdf_processor import process_pdf_to_agent_doc
        parsed = process_pdf_to_agent_doc(pdf_path=path, extract_images=False)
        text = str(parsed.get("combined_text", "")).strip()
        self._document_text = text
        # Return a summary -- not the full text (too large for context)
        tables = parsed.get("tables", [])
        word_count = len(text.split())
        return {
            "status": "ok",
            "word_count": word_count,
            "section_count": text.count("\n\n"),
            "table_count": len(tables) if tables else 0,
            "text_preview": text[:800],
            "full_text_stored": True,
            "note": "Full text stored internally. Use extract_parameters() with relevant sections.",
        }

    def _tool_extract_parameters(
        self,
        text: str,
        target_parameters: Optional[List[str]] = None,
        context_hint: str = "",
    ) -> Dict[str, Any]:
        from ..utils import call_openrouter
        from ..config import PLANNER_MODEL

        target_str = ", ".join(target_parameters) if target_parameters else "all geometric parameters"
        system = (
            "You extract geometry parameters from scientific text.\n"
            "Return JSON: {\"found\": [{\"name\": str, \"value\": number_or_str, \"unit\": str, "
            "\"confidence\": 0-1, \"location\": str, \"raw_text\": str}], "
            "\"unresolved\": [str]}\n"
            "found: parameters you found. unresolved: parameters from target list NOT found.\n"
            "Never invent values. If uncertain, set confidence < 0.6.\n"
            "Output JSON only."
        )
        user = (
            f"Context: {context_hint or 'nanoscale geometry'}\n"
            f"Target parameters: {target_str}\n\n"
            f"Text:\n{text[:6000]}"
        )
        raw = call_openrouter(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=PLANNER_MODEL,
            temperature=0.0,
            json_mode=True,
        )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"found": [], "unresolved": target_parameters or [], "parse_error": raw[:200]}
        return result

    def _tool_cross_reference(self, document_text: str, parameter_name: str) -> Dict[str, Any]:
        from ..utils import call_openrouter
        from ..config import PLANNER_MODEL

        # Use stored doc text if placeholder passed
        text = document_text if len(document_text) > 100 else self._document_text

        system = (
            "Find all mentions of a parameter in a document.\n"
            "Return JSON: {\"occurrences\": [{\"value\": any, \"unit\": str, \"location\": str, \"raw_text\": str}], "
            "\"conflicts\": [{\"value_a\": any, \"location_a\": str, \"value_b\": any, \"location_b\": str, \"note\": str}]}\n"
            "Output JSON only."
        )
        user = f"Parameter to find: {parameter_name}\n\nDocument:\n{text[:8000]}"
        raw = call_openrouter(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=PLANNER_MODEL,
            temperature=0.0,
            json_mode=True,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"occurrences": [], "conflicts": [], "parse_error": raw[:200]}

    def _tool_resolve_units(self, value: float, unit_string: str, target_unit: str) -> Dict[str, Any]:
        """Deterministic unit resolver -- no LLM call needed."""
        unit_str = str(unit_string).strip().lower()
        target = str(target_unit).strip().lower()

        # Normalise common variants first
        aliases = {
            "umm": "um", "micron": "um", "microns": "um", "micrometer": "um", "micrometers": "um",
            "nanometer": "nm", "nanometers": "nm",
            "millimeter": "mm", "millimeters": "mm",
            "degree": "deg", "degrees": "deg", "deg": "deg",
        }
        unit_norm = aliases.get(unit_str, unit_str)

        conversions = {
            ("nm", "um"): 1e-3,
            ("um", "nm"): 1e3,
            ("mm", "um"): 1e3,
            ("um", "mm"): 1e-3,
            ("nm", "mm"): 1e-6,
            ("mm", "nm"): 1e6,
            ("cm", "um"): 1e4,
            ("um", "cm"): 1e-4,
        }

        if unit_norm == target:
            return {"normalized_value": value, "unit": target, "conversion_applied": "none", "warning": None}

        factor = conversions.get((unit_norm, target))
        if factor is not None:
            return {
                "normalized_value": round(value * factor, 6),
                "unit": target,
                "conversion_applied": f"{unit_norm} -> {target} (x{factor})",
                "warning": None,
            }

        return {
            "normalized_value": value,
            "unit": unit_norm,
            "conversion_applied": "none -- unknown conversion",
            "warning": f"Could not convert {unit_norm} -> {target}. Value left as-is in {unit_norm}.",
        }

    def _tool_create_cge(self, geometry_description: str) -> Dict[str, Any]:
        self._cge = CanonicalGeometry(geometry_description=geometry_description)
        return {
            "status": "created",
            "geometry_description": geometry_description,
            "note": "CGE initialized. Use set_cge_field() to populate parameters.",
        }

    def _tool_set_cge_field(
        self,
        name: str,
        value: Any,
        source: str,
        confidence: float,
        unit: Optional[str] = None,
        provenance_location: Optional[str] = None,
        provenance_raw_text: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._cge is None:
            return {"error": "No CGE exists. Call create_cge() first."}
        prov = None
        if provenance_location or provenance_raw_text:
            prov = {"location": provenance_location, "raw_text": provenance_raw_text}
        self._cge.set_field(
            name=name, value=value, unit=unit,
            source=source, confidence=confidence,  # type: ignore[arg-type]
            provenance=prov, notes=notes,
        )
        return {
            "status": "set",
            "field": name,
            "value": value,
            "unit": unit,
            "source": source,
            "confidence": confidence,
        }

    def _tool_add_cge_constraint(
        self,
        description: str,
        source: str = "extracted",
        confidence: float = 1.0,
        provenance_raw_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._cge is None:
            return {"error": "No CGE exists. Call create_cge() first."}
        prov = {"raw_text": provenance_raw_text} if provenance_raw_text else None
        self._cge.add_constraint(description, source=source, confidence=confidence, provenance=prov)  # type: ignore[arg-type]
        return {"status": "added", "constraint": description}

    def _tool_get_cge_status(self) -> Dict[str, Any]:
        if self._cge is None:
            return {"error": "No CGE exists yet.", "ready_for_cad": False}
        report = self._cge.completeness_report()
        report["summary"] = self._cge.summary()
        return report

    def _rag_retrieve(self, prompt: str) -> List[Dict[str, Any]]:
        """Return retrieved RAG chunks for prompt, or [] on failure."""
        try:
            from ..retrieval.retriever import get_retriever
            retrieved = get_retriever().retrieve(prompt, n_examples=10, n_cheatsheet=1)
            n_ex = len([c for c in retrieved if c.get("source") == "examples"])
            self._log(f"  RAG: retrieved {n_ex} examples + cheatsheet for prompt")
            return retrieved
        except Exception as exc:
            self._log(f"  RAG retrieval failed ({exc}) -- proceeding without examples")
            return []

    def _generate_and_repair(
        self,
        prompt: str,
        retrieved: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate CadQuery code, execute it, and repair up to MAX_REPAIR_ATTEMPTS times
        if execution fails. Returns {success, code, repair_attempts, errors, stl_path}.
        """
        from ..generator.cad_generator import generate as generate_cad
        from ..execution.cad_executor import execute as cad_execute
        from ..repair.repair_loop import repair
        from ..config import DEFAULT_GENERATOR_MODEL, MAX_REPAIR_ATTEMPTS

        out_dir = output_dir or str(self.output_dir)

        # 1. Initial generation
        code = generate_cad(prompt=prompt, retrieved=retrieved, rag_examples=[], model=DEFAULT_GENERATOR_MODEL)

        # 2. First execution attempt
        exec_result = cad_execute(code, output_dir=out_dir)
        if exec_result.get("success"):
            return {"success": True, "code": code, "repair_attempts": 0,
                    "errors": [], "stl_path": exec_result.get("stl_path")}

        # 3. Repair loop
        self._log(f"  Initial execution failed -- starting repair loop (max {MAX_REPAIR_ATTEMPTS} attempts)")
        repair_result = repair(
            prompt=prompt,
            code=code,
            error=exec_result.get("error", "unknown error"),
            retrieved=retrieved,
            model=DEFAULT_GENERATOR_MODEL,
            max_retries=MAX_REPAIR_ATTEMPTS,
            output_dir=out_dir,
        )
        return {
            "success":         repair_result["success"],
            "code":            repair_result["code"],
            "repair_attempts": repair_result["repair_attempts"],
            "errors":          repair_result["errors"],
            "stl_path":        repair_result.get("stl_path"),
        }

    def _tool_generate_cad(self, notes: str = "") -> Dict[str, Any]:
        if self._cge is None:
            return {"error": "No CGE exists."}
        status = self._cge.completeness_report()
        if not status["ready_for_cad"]:
            return {
                "error": "CGE not ready for CAD. Unset fields must be resolved first.",
                "unset_fields": status["unset"],
            }

        from ..config import DEFAULT_GENERATOR_MODEL
        from .cad_agent import CADAgent

        cge_block = self._cge.to_prompt_block()
        description = (
            f"Build CadQuery geometry for the following specification.\n\n"
            f"{cge_block}"
        )
        if notes:
            description += f"\n\nAdditional notes: {notes}"

        # CAD tool owns retrieval/context + generation + fallback + validation.
        cad_agent = CADAgent(
            model=DEFAULT_GENERATOR_MODEL,
            vision_model=self.vision_model,
            output_dir=str(self.output_dir),
            max_steps=25,
            verbose=self.verbose,
        )
        self._log("  CADTool: starting autonomous CGE -> CAD run...")
        cad_result = cad_agent.run(description, cge=self._cge)

        verification = cad_result.get("verification") or {}
        self._hard_result = verification.get("hard")
        self._medium_result = verification.get("medium") or {}
        self._render_paths = verification.get("renders") or {}
        self._extracted_cge = verification.get("extracted_cge") or {}
        self._geometry_metrics = verification.get("geometry_metrics") or {}
        self._vision_analysis = verification.get("soft") or verification.get("vision") or {}

        if cad_result.get("success"):
            self._cad_code = cad_result.get("code", "")
            self._log(
                f"  CADTool: done in {cad_result['steps']} steps, "
                f"{cad_result['tokens'].get('total', 0)} tokens"
            )
            return {
                "status":      "generated",
                "path":        cad_result.get("path", "cad_agent"),
                "stl_path":    cad_result["stl_path"],
                "code_length": len(self._cad_code),
                "code_preview": self._cad_code[:400],
                "steps":       cad_result["steps"],
                "tokens":      cad_result["tokens"],
                "retrieved_examples": cad_result.get("retrieved_examples", []),
                "verification": {
                    "hard":             self._hard_result,
                    "medium":           self._medium_result,
                    "extracted_cge":    self._extracted_cge,
                    "geometry_metrics": self._geometry_metrics,
                    "soft":             self._vision_analysis,
                    "vision":           self._vision_analysis,
                    "renders":          self._render_paths,
                    "intended_cge":     verification.get("intended_cge") or (self._cge.to_dict() if self._cge else {}),
                },
            }

        self._cad_code = cad_result.get("code", "")
        return {
            "error": cad_result.get("error", "CAD tool failed"),
            "path": cad_result.get("path", "cad_agent"),
            "code_preview": self._cad_code[:300],
            "retrieved_examples": cad_result.get("retrieved_examples", []),
            "verification": {
                "hard":             self._hard_result,
                "medium":           self._medium_result,
                "extracted_cge":    self._extracted_cge,
                "geometry_metrics": self._geometry_metrics,
                "soft":             self._vision_analysis,
                "vision":           self._vision_analysis,
                "renders":          self._render_paths,
                "intended_cge":     verification.get("intended_cge") or (self._cge.to_dict() if self._cge else {}),
            },
        }

    def _expected_body_count(self) -> Optional[int]:
        """Derive expected solid body count from CGE array parameters, if available."""
        if self._cge is None:
            return None
        params = self._cge.parameters
        # Look for common array dimension naming conventions
        nx = ny = None
        for name in ("array_nx", "nx", "array_x", "columns", "n_columns", "n_x"):
            if name in params and params[name].value is not None:
                try:
                    nx = int(round(float(params[name].value)))
                    break
                except (TypeError, ValueError):
                    pass
        for name in ("array_ny", "ny", "array_y", "rows", "n_rows", "n_y"):
            if name in params and params[name].value is not None:
                try:
                    ny = int(round(float(params[name].value)))
                    break
                except (TypeError, ValueError):
                    pass
        if nx is not None and ny is not None:
            return nx * ny
        # Also handle a single "array_count" or "n_pillars" field
        for name in ("array_count", "n_pillars", "n_features", "feature_count"):
            if name in params and params[name].value is not None:
                try:
                    return int(round(float(params[name].value)))
                except (TypeError, ValueError):
                    pass
        return None

    def _tool_verify_cad_hard(self) -> Dict[str, Any]:
        # Return cached result from generate_cad() if available.
        if self._hard_result is not None:
            return {
                "hard":             self._hard_result,
                "medium":           self._medium_result,
                "extracted_cge":    self._extracted_cge,
                "geometry_metrics": self._geometry_metrics,
                "soft":             self._vision_analysis,
                "vision":           self._vision_analysis,
                "renders":          self._render_paths,
                "intended_cge":     self._cge.to_dict() if self._cge else {},
                "note": "Returning cached verification from latest CAD tool run.",
            }

        if not self._cad_code:
            return {"error": "No CAD code to verify. Call generate_cad() first."}

        from ..execution.cad_executor import execute as cad_execute
        from ..execution.cad_verifier import (
            hard_verify, extract_cge_from_code,
            compute_geometry_metrics, structured_vision_analysis,
            _execute_for_workplane,
        )
        from ..execution.render_utils import render_stl

        exec_result = cad_execute(self._cad_code, output_dir=str(self.output_dir))
        if not exec_result.get("success"):
            return {"passed": False, "error": exec_result.get("error", "execution failed"), "checks": []}

        expected = self._expected_body_count()
        hard = hard_verify(
            code=self._cad_code,
            stl_path=exec_result.get("stl_path"),
            expected_body_count=expected,
        )
        self._hard_result = hard

        if not hard.get("passed", False):
            self._medium_result = {
                "passed": False,
                "score": 0.0,
                "checks": [],
                "warnings": ["Skipped medium checks because hard assertions failed."],
                "param_drift": [],
                "missing_parameters": [],
            }
            return {"hard": hard, "medium": self._medium_result, "extracted_cge": {}, "geometry_metrics": {}, "soft": {}, "vision": {}, "renders": self._render_paths, "intended_cge": self._cge.to_dict() if self._cge else {}}

        self._extracted_cge = extract_cge_from_code(self._cad_code)

        wp = _execute_for_workplane(self._cad_code)
        self._geometry_metrics = compute_geometry_metrics(wp) if wp is not None else {}

        # Reuse CADAgent medium assertions for deterministic non-blocking checks.
        try:
            from .cad_agent import CADAgent
            evaluator = CADAgent(
                model=self.model,
                vision_model=self.vision_model,
                output_dir=str(self.output_dir),
                max_steps=1,
                verbose=False,
            )
            self._medium_result = evaluator._build_medium_assertions(  # noqa: SLF001
                cge=self._cge,
                extracted_cge=self._extracted_cge,
                geometry_metrics=self._geometry_metrics,
                hard=self._hard_result or {},
                expected_body_count=expected,
            )
        except Exception:
            self._medium_result = {}

        if exec_result.get("stl_path"):
            try:
                paths = render_stl(stl_path=exec_result["stl_path"], output_dir=str(self.output_dir))
                self._render_paths = {k: v for k, v in paths.items() if v}
            except Exception:
                self._render_paths = {}

        self._vision_analysis = {}
        if self._render_paths:
            try:
                self._vision_analysis = structured_vision_analysis(
                    render_paths=self._render_paths, model=self.vision_model
                )
            except Exception:
                pass

        return {
            "hard":             self._hard_result,
            "medium":           self._medium_result,
            "extracted_cge":    self._extracted_cge,
            "geometry_metrics": self._geometry_metrics,
            "soft":             self._vision_analysis,
            "vision":           self._vision_analysis,
            "renders":          self._render_paths,
            "intended_cge":     self._cge.to_dict() if self._cge else {},
        }

    def _tool_plan_sweep(
        self,
        parameter_name: str,
        values: List[float],
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._cge is None:
            return {"error": "No CGE. Build a CGE first."}
        param = self._cge.parameters.get(parameter_name)
        if param is None:
            available = list(self._cge.parameters.keys())
            return {"error": f"'{parameter_name}' not in CGE. Available: {available}"}
        if not values:
            return {"error": "values list is empty."}
        self._sweep_plan = {
            "parameter": parameter_name,
            "base_value": param.value,
            "unit": param.unit,
            "values": list(values),
            "label": label or f"{parameter_name} sweep",
            "n_variants": len(values),
        }
        self._log(f"  Sweep planned: {parameter_name} over {values}")
        return {
            "status": "planned",
            "parameter": parameter_name,
            "base_value": param.value,
            "unit": param.unit,
            "values": values,
            "n_variants": len(values),
            "note": "Call execute_sweep() to generate base CAD (once) and all variants.",
        }

    def _tool_execute_sweep(self, notes: str = "") -> Dict[str, Any]:
        if self._sweep_plan is None:
            return {"error": "No sweep plan. Call plan_sweep() first."}
        if self._cge is None:
            return {"error": "No CGE."}

        status = self._cge.completeness_report()
        if not status["ready_for_cad"]:
            return {"error": "CGE not ready for CAD.", "unset_fields": status["unset"]}

        param_name = self._sweep_plan["parameter"]
        values = self._sweep_plan["values"]
        unit = self._sweep_plan["unit"] or ""

        from ..execution.cad_executor import execute as cad_execute
        from ..execution.render_utils import render_stl
        from ..execution.cad_verifier import hard_verify

        # -- Step 1: Generate parameterized base code (RAG + repair loop) ------
        self._log(f"  Sweep: generating parameterized base code for {param_name} ...")
        base_prompt = (
            f"Generate CadQuery Python code for the following geometry.\n\n"
            f"CRITICAL REQUIREMENT: The parameter '{param_name}' MUST be defined as a "
            f"named variable at the very top of the code, like:\n"
            f"    {param_name} = {self._cge.parameters[param_name].value}  # {unit}\n"
            f"All geometry calls must reference this variable -- never hardcode the value inline.\n"
            f"All other sweep-relevant parameters should also be top-level variables.\n\n"
            f"ARRAY PATTERN REQUIREMENT: If this geometry uses repeated features (arrays, grids), "
            f"each feature MUST be built from a fresh cq.Workplane('XY') and positioned with "
            f".transformed(offset=cq.Vector(x, y, 0)), then added with result.add(). "
            f"Do NOT call .workplane() on an accumulated result -- this causes a z-staircase bug "
            f"where each feature starts at the bounding box centre of all previous ones.\n\n"
            f"CGE:\n{self._cge.to_prompt_block()}"
        )
        if notes:
            base_prompt += f"\n\nNotes: {notes}"

        base_dir = self.output_dir / "sweep_base"
        base_dir.mkdir(parents=True, exist_ok=True)

        retrieved = self._rag_retrieve(base_prompt)
        gen = self._generate_and_repair(base_prompt, retrieved, output_dir=str(base_dir))

        if not gen["success"]:
            return {
                "status": "error",
                "error": f"Base code generation failed after {gen['repair_attempts']} repair attempt(s): {gen['errors'][-1][:300] if gen['errors'] else ''}",
                "note": "Fix the base code issue before running the sweep.",
            }

        base_code = gen["code"]
        self._cad_code = base_code  # store base as the canonical code
        (base_dir / "base_code.py").write_text(base_code, encoding="utf-8")
        self._log(f"  Base code written to {base_dir / 'base_code.py'} (repair_attempts={gen['repair_attempts']}, rag_examples={len([c for c in retrieved if c.get('source')=='examples'])})")

        # -- Step 2: Hard verify base geometry (reuse STL from generate_and_repair)
        expected_bodies = self._expected_body_count()
        base_hard = hard_verify(
            code=base_code,
            stl_path=gen.get("stl_path"),
            expected_body_count=expected_bodies,
        )
        self._hard_result = base_hard

        base_renders: Dict[str, str] = {}
        if gen.get("stl_path"):
            r = render_stl(stl_path=gen["stl_path"], output_dir=str(base_dir))
            base_renders = {k: v for k, v in r.items() if v}

        # -- Step 3: Substitute and execute each variant -- no more LLM calls --
        results: List[Dict[str, Any]] = []
        for i, val in enumerate(values):
            variant_code = _substitute_parameter(base_code, param_name, val)
            variant_dir = self.output_dir / f"sweep_{param_name}_{val}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "variant_code.py").write_text(variant_code, encoding="utf-8")

            exec_r = cad_execute(variant_code, output_dir=str(variant_dir))
            hard_r = hard_verify(
                code=variant_code,
                stl_path=exec_r.get("stl_path"),
                expected_body_count=expected_bodies,
            ) if exec_r.get("success") else {"passed": False, "error": exec_r.get("error")}

            renders: Dict[str, str] = {}
            if exec_r.get("success") and hard_r.get("passed") and exec_r.get("stl_path"):
                r = render_stl(stl_path=exec_r["stl_path"], output_dir=str(variant_dir))
                renders = {k: v for k, v in r.items() if v}

            entry = {
                "index": i,
                "parameter": param_name,
                "value": val,
                "unit": unit,
                "success": exec_r.get("success", False) and hard_r.get("passed", False),
                "hard_verification": hard_r,
                "renders": renders,
                "variant_dir": str(variant_dir),
                "substitution_applied": variant_code != base_code,
            }
            results.append(entry)
            status_str = "OK" if entry["success"] else "FAIL"
            self._log(f"  [{status_str}] variant {i+1}/{len(values)}: {param_name}={val}{unit}")

        self._sweep_results = results
        self._render_paths = base_renders  # show base renders in notebook

        passed = sum(1 for r in results if r["success"])
        return {
            "status": "complete",
            "parameter": param_name,
            "n_variants": len(values),
            "n_passed": passed,
            "n_failed": len(values) - passed,
            "base_hard_verification_passed": base_hard.get("passed"),
            "note": f"Base code generated once. {len(values)} variants by substitution only.",
            "results_summary": [
                {"value": r["value"], "unit": unit, "success": r["success"]} for r in results
            ],
        }

    def _tool_analyze_geometry(self) -> Dict[str, Any]:
        from ..metrics.geometry_metrics import compute_metrics

        if not self._hard_result:
            return {"error": "No verified CAD. Call generate_cad() first."}

        # Prefer the STL path stored during the last CAD run.
        stl_path: Optional[str] = None
        for cad_out_dir in [self.output_dir]:
            candidates = sorted(cad_out_dir.glob("**/*.stl"))
            if candidates:
                stl_path = str(candidates[-1])
                break

        if stl_path is None:
            return {"error": "No STL file found in output directory. Run generate_cad() first."}

        try:
            import cadquery as cq
            result_wp = cq.importers.importStep(stl_path) if stl_path.endswith(".step") else None
        except Exception:
            result_wp = None

        # Fall back to executing the code if we have it and no workplane
        if result_wp is None and self._cad_code:
            try:
                from ..execution.cad_verifier import _execute_for_workplane
                result_wp = _execute_for_workplane(self._cad_code)
            except Exception:
                pass

        if result_wp is None:
            return {"error": "Could not obtain a live workplane for geometry analysis."}

        try:
            metrics = compute_metrics(result_wp, code_str=self._cad_code or "")
        except Exception as exc:
            return {"error": f"compute_metrics failed: {exc}"}

        self._geometry_analysis = metrics
        return {"status": "ok", "metrics": metrics}

    def _tool_predict_printability(
        self,
        slicing_distance_um: float = 0.2,
        hatching_distance_um: float = 0.2,
    ) -> Dict[str, Any]:
        if not self._geometry_analysis:
            return {"error": "No geometry analysis available. Call analyze_geometry() first."}

        m = self._geometry_analysis

        # ── Try ML model first ────────────────────────────────────────────────
        ml_result: Dict[str, Any] = {}
        model_type = "heuristic"
        try:
            import sys as _sys
            import os as _os
            _repo = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))
            if _repo not in _sys.path:
                _sys.path.insert(0, _repo)
            from tools.forward_model.step_predictor import StepForwardModel
            _model = StepForwardModel.load()
            if _model.is_trained():
                ml_result = _model.predict(m, slicing_distance_um, hatching_distance_um)
                model_type = ml_result.get("model_type", "xgboost")
        except Exception:
            pass  # fall through to heuristic

        # ── Heuristic (used as fallback and for failure-mode labels) ──────────
        min_feature = float(m.get("feature_thickness_min") or m.get("minimum_feature_size_um") or 0.0)
        slenderness = float(m.get("slenderness") or m.get("slenderness_ratio") or 0.0)
        spacing     = float(m.get("feature_spacing_min") or m.get("feature_spacing_um") or 0.0)
        overhang    = float(m.get("overhang_max_angle_deg") or m.get("overhang_deg") or 0.0)

        if ml_result:
            probability = float(ml_result["p_pass"])
            uncertainty = float(ml_result["uncertainty"])
        else:
            probability = 0.88
            recipe_max  = max(slicing_distance_um, hatching_distance_um)
            probability -= max(0.0, 0.42 * (0.45 - min_feature)) if min_feature < 0.45 else 0.0
            probability -= max(0.0, 0.03 * (slenderness - 5.5)) if slenderness > 5.5 else 0.0
            probability -= max(0.0, 0.30 * (0.60 - spacing)) if spacing < 0.60 else 0.0
            probability -= max(0.0, 0.007 * (overhang - 35.0)) if overhang > 35.0 else 0.0
            if recipe_max >= 0.55:
                probability -= 0.60
            elif recipe_max >= 0.40:
                probability -= 0.15
            probability = max(0.01, min(0.99, probability))
            uncertainty = 0.40  # high uncertainty for heuristic

        failure_modes: List[str] = []
        if min_feature > 0 and min_feature < 0.40:
            failure_modes.append("minimum feature collapse (feature_thickness_min < 0.40 um)")
        if slenderness > 6.0:
            failure_modes.append(f"pillar buckling (slenderness={slenderness:.2f} > 6.0)")
        if spacing > 0 and spacing < 0.50:
            failure_modes.append(f"feature fusion (feature_spacing_min={spacing:.3f} um < 0.50 um)")
        if overhang > 40.0:
            failure_modes.append(f"unsupported overhang ({overhang:.1f} deg > 40 deg)")
        if not failure_modes and probability < 0.55:
            failure_modes.append("process window miss — parameters near limits without single dominant failure mode")

        recommendations: List[str] = []
        if min_feature > 0 and min_feature < 0.45:
            recommendations.append(f"Increase minimum feature size from {min_feature:.3f} um to >= 0.45 um")
        if slenderness > 5.5:
            recommendations.append(f"Reduce slenderness from {slenderness:.2f} to <= 5.5 (widen or shorten features)")
        if spacing > 0 and spacing < 0.60:
            recommendations.append(f"Increase feature spacing from {spacing:.3f} um to >= 0.60 um")
        if overhang > 35.0:
            recommendations.append(f"Reduce overhang from {overhang:.1f} deg to <= 35 deg or add support")
        if ml_result and ml_result.get("features_approximated"):
            recommendations.append(
                "Predictions used approximated geometry features — "
                "provide full STEP-derived metrics for best accuracy."
            )

        result = {
            "success_probability": round(probability, 4),
            "uncertainty":         round(uncertainty, 4),
            "risk_score":          round(1.0 - probability, 4),
            "predicted_success":   probability >= 0.55,
            "model_type":          model_type,
            "recipe": {
                "slicing_distance_um":  slicing_distance_um,
                "hatching_distance_um": hatching_distance_um,
            },
            "failure_modes":    failure_modes,
            "recommendations":  recommendations,
            "inputs": {
                "min_feature_um":       min_feature,
                "slenderness":          slenderness,
                "feature_spacing_min":  spacing,
                "overhang_max_deg":     overhang,
            },
        }
        self._printability = result
        return result

    def _tool_generate_gwl_job(
        self,
        describe_exe_path: str,
        slicing_distance_um: float = 0.1,
        hatching_distance_um: float = 0.1,
        power_scaling: float = 1.0,
    ) -> Dict[str, Any]:
        import sys
        import tempfile
        import shutil

        dsa0_path = "/Users/simonfernandez/Desktop/Agent_Nanoscribe/DescribeXSliceAnyting-DSA0"
        if dsa0_path not in sys.path:
            sys.path.insert(0, dsa0_path)

        try:
            from describe_slice_anything.recipe_params import RecipeParams
            from describe_slice_anything.slicer import slice_folder
        except ImportError as exc:
            return {"error": f"Could not import DSA0 slicer: {exc}. Ensure the repo is at {dsa0_path}"}

        # Locate the current STL
        stl_path: Optional[str] = None
        for candidate in sorted(self.output_dir.glob("**/*.stl")):
            stl_path = str(candidate)
            break

        if stl_path is None:
            return {"error": "No STL found. Call generate_cad() first."}

        params = RecipeParams(
            slicing_distance_um=slicing_distance_um,
            hatching_distance_um=hatching_distance_um,
            power_scaling=power_scaling,
        )

        # Create a temp input folder containing just the target STL
        tmp_in = Path(tempfile.mkdtemp(prefix="gwl_in_"))
        tmp_out = self.output_dir / "gwl_output"
        tmp_out.mkdir(parents=True, exist_ok=True)

        try:
            stl_dest = tmp_in / Path(stl_path).name
            shutil.copy2(stl_path, stl_dest)

            gwl_paths = slice_folder(
                input_folder=tmp_in,
                output_folder=tmp_out,
                describe_exe=Path(describe_exe_path),
                params=params,
            )
        except FileNotFoundError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"DescribeX slicing failed: {exc}"}
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)

        if not gwl_paths:
            return {"error": "DescribeX ran but produced no output GWL files."}

        data_gwl = str(gwl_paths[0])
        stem = Path(data_gwl).stem.replace("_data", "")
        files_dir = str(tmp_out / f"{stem}_files")

        result = {
            "status": "ok",
            "gwl_path": data_gwl,
            "files_dir": files_dir,
            "recipe": {
                "slicing_distance_um": slicing_distance_um,
                "hatching_distance_um": hatching_distance_um,
                "power_scaling": power_scaling,
            },
            "note": "gwl_path and files_dir must stay in the same folder to open in NanoWrite.",
        }
        self._gwl_result = result
        return result

    def _tool_request_clarification(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        In interactive mode, this surfaces the question to the user.
        In notebook usage, the agent will pause and the user can call
        agent.provide_clarification() before resuming.
        """
        print(f"\n[Agent needs clarification]\n  Question: {question}")
        if context:
            print(f"  Context: {context}")

        if self._pending_clarification is not None:
            response = self._pending_clarification
            self._pending_clarification = None
            return {"response": response}

        # In non-interactive environments, record the question and continue
        return {
            "response": None,
            "status": "pending",
            "note": "No clarification provided. Call agent.provide_clarification() and re-run.",
        }

    def _tool_flag_issue(self, field: str, issue: str, severity: str = "warning") -> Dict[str, Any]:
        entry = {"field": field, "issue": issue, "severity": severity, "step": self._step_count}
        self._flags.append(entry)
        self._log(f"  [{severity.upper()}] {field}: {issue}")
        return {"status": "flagged", **entry}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._cge = None
        self._cad_code = None
        self._render_paths = {}
        self._document_text = ""
        self._hard_result = None
        self._medium_result = {}
        self._extracted_cge = {}
        self._geometry_metrics = {}
        self._vision_analysis = {}
        self._geometry_analysis = {}
        self._printability = {}
        self._gwl_result = {}
        self._messages = []
        self._flags = []
        self._decision_trace = []
        self._step_count = 0
        self._sweep_plan = None
        self._sweep_results = None
        self._token_totals = {"prompt": 0, "completion": 0, "total": 0}
        self._start_time = 0.0

    def _record_step(self, tool_name: str, args: dict) -> None:
        entry = {
            "step": self._step_count,
            "tool": tool_name,
            "args_preview": str(args)[:200],
            "llm_tokens": {},    # filled in after the LLM call returns
            "llm_time_s": 0.0,
            "tool_time_s": 0.0,
        }
        self._decision_trace.append(entry)
        self._log(f"  -> {tool_name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items())})")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[step {self._step_count:02d}] {msg}")

    def _build_verification_summary(self) -> Optional[Dict[str, Any]]:
        if self._hard_result is None:
            return None
        return {
            "hard":               self._hard_result,
            "medium":             self._medium_result,
            "renders":            self._render_paths,
            "intended_cge":       self._cge.to_dict() if self._cge else None,
            "extracted_cge":      self._extracted_cge,
            "geometry_metrics":   self._geometry_metrics,
            "soft":               self._vision_analysis,
            "vision":             self._vision_analysis,
            "geometry_analysis":  self._geometry_analysis or None,
            "printability":       self._printability or None,
            "gwl":                self._gwl_result or None,
        }

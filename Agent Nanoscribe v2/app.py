"""
Nanoscribe Agent -- Chat UI

A conversational interface for the NanoscribeAgent. Supports:
  - Natural language geometry design
  - PDF-driven geometry extraction
  - Parameter sweeps with inline render grids
  - Human-in-the-loop clarification
  - Live step tracking during agent execution

Run from the Prompt2CAD/subsystems/prompt2cad directory:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# -- Path setup ----------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prompt2cad.agentic.agent_core import NanoscribeAgent, AgentResult
from prompt2cad.agentic.cge import CanonicalGeometry

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Nanoscribe Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 0.5rem; max-width: 1100px; }
.stChatMessage { border-radius: 10px; }
.cge-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; font-family: monospace; }
.cge-table th { background: #1e1e2e; color: #cdd6f4; padding: 5px 10px; text-align: left; }
.cge-table td { padding: 4px 10px; border-bottom: 1px solid #2a2a3e; }
.cge-row-ok    { background: #1e3a2e; }
.cge-row-warn  { background: #3a3a1e; }
.cge-row-bad   { background: #3a1e1e; }
.badge-pass { background:#1a7a3a; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.78rem; }
.badge-fail { background:#7a1a1a; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.78rem; }
.badge-skip { background:#444;    color:#ccc; padding:2px 8px; border-radius:4px; font-size:0.78rem; }
.sweep-label { font-size:0.78rem; text-align:center; font-family:monospace; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# -- Constants -----------------------------------------------------------------
_DEFAULT_OUTPUT = _HERE / "output" / "agent_chat"
_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "deepseek/deepseek-chat",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-opus-4-6",
]
_VISION_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-4o",
    "anthropic/claude-opus-4-6",
]

# -- Session state init --------------------------------------------------------
def _init_state():
    defaults = {
        "messages": [],
        "agent": None,
        "awaiting_clarification": False,
        "design_complete": False,       # True only after CAD verified or sweep done
        "pending_goal": "",
        "last_tokens": {},
        "last_elapsed": 0.0,
        "run_count": 0,
        "pending_input": None,          # holds submitted input across rerun
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Nanoscribe Agent")
    st.divider()

    model = st.selectbox("Reasoning model", _MODELS, index=0,
                         help="Model used for agent reasoning and tool calls.")
    vision_model = st.selectbox("Vision model (soft verify)", _VISION_MODELS, index=0,
                                help="Model used for render inspection. Must support vision.")
    max_steps = st.slider("Max steps", 10, 80, 40, step=5)
    output_dir = st.text_input("Output directory", value=str(_DEFAULT_OUTPUT))
    verbose = st.checkbox("Verbose (console log)", value=False)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = None
        st.session_state.awaiting_clarification = False
        st.session_state.design_complete = False
        st.session_state.pending_goal = ""
        st.session_state.pending_input = None
        st.rerun()

    if st.session_state.last_tokens:
        st.divider()
        st.markdown("**Last run stats**")
        tok = st.session_state.last_tokens
        st.markdown(
            f"```\n"
            f"Prompt:     {tok.get('prompt',0):>8,}\n"
            f"Completion: {tok.get('completion',0):>8,}\n"
            f"Total:      {tok.get('total',0):>8,}\n"
            f"Elapsed:    {st.session_state.last_elapsed:>7.1f}s\n"
            f"```"
        )
    if st.session_state.agent and not st.session_state.design_complete:
        st.divider()
        st.markdown("**Session active** -- agent memory preserved until design is complete or cleared.")

# -- Header --------------------------------------------------------------------
st.markdown("## Nanoscribe Agent")
st.caption("Describe a geometry, reference a paper, or request a parameter sweep.")

# -- Result renderer -----------------------------------------------------------

def _render_result(result: AgentResult, run_dir: Path) -> None:

    # CGE table
    if result.cge and result.cge.parameters:
        with st.expander("Canonical Geometry Encoding", expanded=False):
            rows_html = ""
            for name, p in result.cge.parameters.items():
                if p.source == "unset":
                    cls = "cge-row-bad"
                elif p.source in ("inferred", "default") or p.confidence < 0.6:
                    cls = "cge-row-warn"
                else:
                    cls = "cge-row-ok"
                val = f"{p.value} {p.unit or ''}".strip() if p.value is not None else "UNSET"
                rows_html += (
                    f"<tr class='{cls}'>"
                    f"<td><b>{name}</b></td>"
                    f"<td>{val}</td>"
                    f"<td>{p.source}</td>"
                    f"<td>{p.confidence:.0%}</td>"
                    f"</tr>"
                )
            if result.cge.constraints:
                rows_html += (
                    "<tr><td colspan='4' style='padding-top:8px;color:#888;'>"
                    "<i>Constraints: "
                    + " | ".join(c.description[:60] for c in result.cge.constraints)
                    + "</i></td></tr>"
                )
            st.markdown(
                "<table class='cge-table'>"
                "<thead><tr><th>Parameter</th><th>Value</th><th>Source</th><th>Conf.</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )
            report = result.cge.completeness_report()
            st.caption(
                f"Completeness: {report['completeness']:.0%}  |  "
                f"ready_for_cad: {report['ready_for_cad']}"
            )

    # Flags
    if result.flags:
        criticals = [f for f in result.flags if f["severity"] == "critical"]
        warnings  = [f for f in result.flags if f["severity"] == "warning"]
        if criticals:
            label = f"[!] {len(criticals)} critical, {len(warnings)} warning(s)"
        else:
            label = f"[!] {len(result.flags)} flag(s)"
        with st.expander(label, expanded=bool(criticals)):
            for f in result.flags:
                prefix = {"critical": "[CRITICAL]", "warning": "[WARNING]", "info": "[INFO]"}.get(f["severity"], "[FLAG]")
                st.markdown(f"{prefix} **{f['field']}** -- {f['issue']}")

    # Verification
    v = result.verification
    if v:
        hard = v.get("hard") or {}
        soft = v.get("soft") or {}
        hp = hard.get("passed")
        sp = soft.get("passed")
        h_label = "PASS" if hp else ("FAIL" if hp is False else "--")
        s_label = "PASS" if sp else ("FAIL" if sp is False else "--")
        h_cls = "pass" if hp else ("fail" if hp is False else "skip")
        s_cls = "pass" if sp else ("fail" if sp is False else "skip")
        h_badge = f"<span class='badge-{h_cls}'>Hard {h_label}</span>"
        s_badge = f"<span class='badge-{s_cls}'>Soft {s_label}</span>"
        st.markdown(f"**Verification** &nbsp; {h_badge} &nbsp; {s_badge}", unsafe_allow_html=True)
        if hp is False and hard.get("checks"):
            for c in hard["checks"]:
                if c.get("passed") is False:
                    st.caption(f"  Hard fail -- {c['name']}: {c['detail']}")
        if sp is False and soft.get("issues"):
            for issue in soft["issues"]:
                st.caption(f"  Soft fail -- {issue}")
        if soft.get("assessment"):
            st.caption(f"  Soft assessment: {soft['assessment']}")
        bb = hard.get("bounding_box")
        if bb:
            st.caption(
                f"  Bounding box: {bb['x']:.2f} x {bb['y']:.2f} x {bb['z']:.2f} um  |  "
                f"bodies: {hard.get('body_count', '?')}"
            )

    # Renders (non-sweep)
    renders = (v or {}).get("renders") or {}
    if not result.sweep_results and renders:
        _show_renders_row(renders)
    elif not result.sweep_results:
        _show_renders_fallback(run_dir)

    # Sweep grid
    if result.sweep_results:
        _show_sweep_grid(result.sweep_results)

    # Footer
    tok = result.total_tokens
    st.caption(
        f"{result.elapsed_s:.1f}s  |  "
        f"{tok.get('total',0):,} tokens "
        f"({tok.get('prompt',0):,} in + {tok.get('completion',0):,} out)  |  "
        f"{result.iterations} steps"
    )


def _show_renders_row(renders: Dict[str, str]) -> None:
    order = ["render_iso_path", "render_top_path", "render_side_path", "render_path"]
    labels = {"render_iso_path": "Iso", "render_top_path": "Top",
              "render_side_path": "Side", "render_path": "Front"}
    available = [(labels[k], renders[k]) for k in order if renders.get(k) and Path(renders[k]).exists()]
    if available:
        cols = st.columns(len(available))
        for col, (label, path) in zip(cols, available):
            with col:
                st.image(path, caption=label, use_container_width=True)


def _show_renders_fallback(run_dir: Path) -> None:
    candidates = [
        ("Iso",   run_dir / "render_iso.png"),
        ("Top",   run_dir / "render_top.png"),
        ("Side",  run_dir / "render_side.png"),
        ("Front", run_dir / "render.png"),
    ]
    available = [(label, str(p)) for label, p in candidates if p.exists()]
    if available:
        cols = st.columns(len(available))
        for col, (label, path) in zip(cols, available):
            with col:
                st.image(path, caption=label, use_container_width=True)


def _show_sweep_grid(sweep_results: List[Dict[str, Any]]) -> None:
    if not sweep_results:
        return
    param = sweep_results[0].get("parameter", "param")
    unit  = sweep_results[0].get("unit", "")
    n = len(sweep_results)
    st.markdown(f"**Sweep: `{param}` -- {n} variants**")
    cols_per_row = min(4, n)
    rows = [sweep_results[i:i+cols_per_row] for i in range(0, n, cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for col, variant in zip(cols, row):
            val = variant["value"]
            ok  = variant["success"]
            renders = variant.get("renders") or {}
            iso = renders.get("render_iso_path") or renders.get("render_path")
            with col:
                status = "PASS" if ok else "FAIL"
                st.markdown(
                    f"<div class='sweep-label'>[{status}] {param}={val}{unit}</div>",
                    unsafe_allow_html=True,
                )
                if iso and Path(iso).exists():
                    st.image(iso, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='background:#222;text-align:center;"
                        "padding:20px;font-size:0.75rem;color:#666;border-radius:4px;'>"
                        "no render</div>",
                        unsafe_allow_html=True,
                    )
                hv = variant.get("hard_verification") or {}
                bodies = hv.get("body_count")
                bb = hv.get("bounding_box") or {}
                z = bb.get("z")
                if bodies is not None:
                    st.caption(f"bodies={bodies}" + (f"  z={z:.1f}" if z else ""))
                if not ok:
                    for c in (hv.get("checks") or []):
                        if c.get("passed") is False:
                            st.caption(f"  fail: {c['name']}")


# -- Replay conversation history -----------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("has_result"):
            result_data = msg.get("result_data")
            run_dir = Path(msg.get("run_dir", str(_DEFAULT_OUTPUT)))
            if result_data:
                cge = CanonicalGeometry.from_dict(result_data["cge"]) if result_data.get("cge") else None
                _mini = AgentResult(
                    success=result_data.get("success", False),
                    cge=cge,
                    cad_code=result_data.get("cad_code"),
                    verification=result_data.get("verification"),
                    message=result_data.get("message", ""),
                    decision_trace=result_data.get("decision_trace", []),
                    flags=result_data.get("flags", []),
                    iterations=result_data.get("iterations", 0),
                    total_tokens=result_data.get("total_tokens", {}),
                    elapsed_s=result_data.get("elapsed_s", 0.0),
                    sweep_results=result_data.get("sweep_results"),
                )
                _render_result(_mini, run_dir)

# -- Clarification banner ------------------------------------------------------
if st.session_state.awaiting_clarification:
    st.info("The agent asked a question above. Type your answer in the chat box.")

# -- Chat input ----------------------------------------------------------------
# Use a key-based approach to avoid the double-submit glitch:
# We store submitted input in session_state.pending_input and process it
# after the rerun, rather than inline in the walrus-operator branch.

placeholder = (
    "Type your answer..." if st.session_state.awaiting_clarification
    else "Describe a geometry, reference a PDF, or request a sweep..."
)

user_input = st.chat_input(placeholder, key="chat_input")

if user_input:
    st.session_state.pending_input = user_input
    st.rerun()

# -- Process pending input (runs on the rerun after submission) ----------------
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None  # clear immediately to prevent re-processing

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Determine run mode
    if st.session_state.awaiting_clarification and st.session_state.agent is not None:
        agent = st.session_state.agent
        agent.provide_clarification(user_input)
        goal = st.session_state.pending_goal
        st.session_state.awaiting_clarification = False
    elif not st.session_state.design_complete and st.session_state.agent is not None:
        # Continue existing agent session (e.g. user refining the design)
        agent = st.session_state.agent
        goal = user_input
        st.session_state.pending_goal = goal
    else:
        # Fresh run
        goal = user_input
        run_out = Path(output_dir) / f"run_{st.session_state.run_count:03d}"
        run_out.mkdir(parents=True, exist_ok=True)
        st.session_state.run_count += 1
        st.session_state.design_complete = False

        agent = NanoscribeAgent(
            model=model,
            vision_model=vision_model,
            max_steps=max_steps,
            verbose=verbose,
            output_dir=str(run_out),
        )
        st.session_state.agent = agent
        st.session_state.pending_goal = goal

    run_dir = Path(agent.output_dir)

    # Run agent with live step display
    with st.chat_message("assistant"):
        status_box = st.status("Agent working...", expanded=True)

        def _on_step(step: int, tool_name: str, result: Any) -> None:
            extra = ""
            if isinstance(result, dict):
                if result.get("error"):
                    extra = f" -- ERROR: {str(result['error'])[:80]}"
                elif tool_name == "get_cge_status":
                    extra = f" -- completeness={result.get('completeness', 0):.0%}"
                elif tool_name == "verify_cad_hard":
                    hard_ok = result.get("passed")
                    soft_ok = (result.get("soft") or {}).get("passed")
                    extra = f" -- hard={hard_ok} soft={soft_ok}"
                elif tool_name == "plan_sweep":
                    extra = f" -- {result.get('n_variants', 0)} variants"
                elif tool_name == "execute_sweep":
                    extra = f" -- {result.get('n_passed', 0)}/{result.get('n_variants', 0)} passed"
                elif tool_name == "flag_issue":
                    extra = f" -- [{result.get('severity', '')}] {result.get('field', '')}"
            status_box.update(label=f"Step {step}: {tool_name}{extra}")
            status_box.write(f"**{tool_name}**{extra}")

        agent.on_step = _on_step

        result = agent.run(goal)

        if result.success:
            status_box.update(
                label=f"Done -- {result.iterations} steps, {result.total_tokens.get('total', 0):,} tokens",
                state="complete",
                expanded=False,
            )
        else:
            status_box.update(
                label=f"Finished with issues -- {result.iterations} steps",
                state="error",
                expanded=True,
            )

        st.markdown(result.message)

        # Detect if agent needs clarification
        needs_clarification = (
            result.message.strip().endswith("?")
            and not result.cad_code
            and not result.sweep_results
        )
        if needs_clarification:
            st.session_state.awaiting_clarification = True
            st.info("Please type your response in the chat box below.")

        # Mark design complete when CAD or sweep is produced and verified
        if result.cad_code or result.sweep_results:
            v = result.verification or {}
            hard_ok = (v.get("hard") or {}).get("passed")
            if hard_ok or result.sweep_results:
                st.session_state.design_complete = True

        _render_result(result, run_dir)

    # Persist to history
    st.session_state.last_tokens  = result.total_tokens
    st.session_state.last_elapsed = result.elapsed_s

    result_data = {
        "success":        result.success,
        "message":        result.message,
        "cge":            result.cge.to_dict() if result.cge else None,
        "flags":          result.flags,
        "verification":   result.verification,
        "decision_trace": result.decision_trace,
        "iterations":     result.iterations,
        "total_tokens":   result.total_tokens,
        "elapsed_s":      result.elapsed_s,
        "sweep_results":  result.sweep_results,
        "cad_code":       result.cad_code,
    }
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     result.message,
        "has_result":  True,
        "result_data": result_data,
        "run_dir":     str(run_dir),
    })

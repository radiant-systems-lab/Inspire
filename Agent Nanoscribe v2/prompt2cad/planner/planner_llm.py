"""
Planner LLM — cheap, fast intent extraction.

Converts a natural-language CAD prompt into a structured JSON plan that
guides retrieval and code generation.  No code is generated here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from ..config import OPENROUTER_API_KEY, PLANNER_MODEL
from ..utils import call_openrouter

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM = """\
You are a CAD planning system for CadQuery (a Python 3D-modelling library).
Your only job is to analyse the user's geometry request and output a JSON plan.

Output EXACTLY one JSON object — no markdown, no explanation, no code.

JSON fields (all required):
  "intent"        – modelling strategy, one of:
                    profile_extrude | boolean_ops | revolve | sweep | loft |
                    pattern_array   | stacked_primitive | shell | free_form
  "operations"    – list of CadQuery method names the code will likely need
                    (e.g. ["Workplane", "box", "faces", "hole", "fillet"])
  "geometry_type" – one of: solid | sketch | surface | assembly
  "tags"          – 2–5 short descriptive labels (e.g. ["hollow","cylinder","thin-wall"])
  "difficulty"    – one of: simple | moderate | complex

Do NOT generate Python code. Output the JSON plan only.\
"""

_USER_TEMPLATE = "User request: {prompt}"

# ── Valid intent categories (used in fallback) ────────────────────────────────
_INTENT_MAP: Dict[str, str] = {
    "extrude":    "profile_extrude",
    "revolve":    "revolve",
    "sweep":      "sweep",
    "loft":       "loft",
    "array":      "pattern_array",
    "pattern":    "pattern_array",
    "boolean":    "boolean_ops",
    "union":      "boolean_ops",
    "cut":        "boolean_ops",
    "shell":      "shell",
    "hollow":     "shell",
    "stacked":    "stacked_primitive",
    "on top":     "stacked_primitive",
}


# ── Public API ────────────────────────────────────────────────────────────────

def plan(
    prompt: str,
    model: str = PLANNER_MODEL,
) -> Dict[str, Any]:
    """
    Call the planner LLM and return a structured modelling plan.

    Returns a dict with keys: intent, operations, geometry_type, tags, difficulty.
    Falls back gracefully if the LLM returns malformed JSON.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": _USER_TEMPLATE.format(prompt=prompt)},
    ]

    raw = call_openrouter(
        messages,
        model=model,
        temperature=0.1,
        json_mode=True,   # request JSON output; not all free models honour this
    )

    return _parse_plan(raw, prompt)


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_plan(text: str, original_prompt: str) -> Dict[str, Any]:
    """Extract the JSON plan from the LLM response, with a heuristic fallback."""

    # 1. Strip markdown code fences if present
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        text = m.group(1)

    # 2. Try direct JSON parse
    try:
        parsed = json.loads(text.strip())
        return _validate_plan(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Try extracting the first {...} block
    m2 = re.search(r'\{[\s\S]*\}', text)
    if m2:
        try:
            parsed = json.loads(m2.group(0))
            return _validate_plan(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Heuristic fallback from the raw text
    return _heuristic_plan(original_prompt)


def _validate_plan(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all required keys exist with sensible types."""
    valid_intents = {
        "profile_extrude", "boolean_ops", "revolve", "sweep", "loft",
        "pattern_array", "stacked_primitive", "shell", "free_form",
    }
    valid_geo = {"solid", "sketch", "surface", "assembly"}
    valid_diff = {"simple", "moderate", "complex"}

    intent = str(d.get("intent", "profile_extrude"))
    if intent not in valid_intents:
        intent = "profile_extrude"

    geo = str(d.get("geometry_type", "solid"))
    if geo not in valid_geo:
        geo = "solid"

    diff = str(d.get("difficulty", "simple"))
    if diff not in valid_diff:
        diff = "moderate"

    ops = d.get("operations", [])
    if not isinstance(ops, list):
        ops = [str(ops)]

    tags = d.get("tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]

    return {
        "intent":        intent,
        "operations":    [str(o) for o in ops],
        "geometry_type": geo,
        "tags":          [str(t) for t in tags],
        "difficulty":    diff,
    }


def _heuristic_plan(prompt: str) -> Dict[str, Any]:
    """Keyword-based fallback when LLM output cannot be parsed as JSON."""
    lower = prompt.lower()

    intent = "profile_extrude"
    for kw, val in _INTENT_MAP.items():
        if kw in lower:
            intent = val
            break

    # Guess operations from keywords
    ops = ["Workplane"]
    for kw, op in [
        ("box", "box"), ("cylinder", "cylinder"), ("sphere", "sphere"),
        ("hole", "hole"), ("fillet", "fillet"), ("chamfer", "chamfer"),
        ("shell", "shell"), ("loft", "loft"), ("sweep", "sweep"),
        ("revolve", "revolve"), ("extrude", "extrude"),
        ("hollow", "shell"), ("array", "rarray"),
    ]:
        if kw in lower:
            ops.append(op)

    return {
        "intent":        intent,
        "operations":    ops,
        "geometry_type": "solid",
        "tags":          [w for w in lower.split() if len(w) > 3][:5],
        "difficulty":    "simple",
    }

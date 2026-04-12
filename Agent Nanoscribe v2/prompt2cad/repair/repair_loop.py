"""
Repair loop -- ask the generator LLM to fix broken CadQuery code.

On each attempt the LLM receives:
  - the original user prompt
  - the failing code
  - the full error traceback
  - a snippet from the most relevant retrieved example
  - targeted common-pitfall reminders

A maximum of MAX_REPAIR_ATTEMPTS rounds are tried; the loop stops early
on the first successful execution.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_GENERATOR_MODEL, MAX_REPAIR_ATTEMPTS
from ..utils import call_openrouter, get_last_openrouter_trace
from ..execution.cad_executor import execute

# -- Prompt templates ----------------------------------------------------------

_SYSTEM = """\
You are a CadQuery debugging assistant.
Fix the broken Python code below so it runs without errors.

Rules (MUST follow):
1. Return ONLY the corrected Python code -- no markdown, no explanation
2. The final geometry MUST be assigned to `result`
3. Never use show_object(), display(), or any GUI function
4. CadQuery 2.x (OCP-based) API

Frequent fixes needed:
- .torus() does not exist -> use cq.Workplane("XY").add(cq.Solid.makeTorus(R, r))
- circle() takes RADIUS; hole() takes DIAMETER
- Open wires must be closed with .close() before .extrude()
- revolve() profile must have one edge on the axis (use centered=False for rect)
- For absolute positioning use: .translate(cq.Vector(x, y, z))\
"""

_USER_TEMPLATE = """\
Original user request:
{prompt}

Broken code:
```python
{code}
```

Error traceback:
```
{error}
```

Relevant example for reference:
{context_snippet}

Fix the code. Return ONLY the corrected Python.\
"""


# -- Public API ----------------------------------------------------------------

def repair(
    prompt: str,
    code: str,
    error: str,
    retrieved: List[dict],
    model: str = DEFAULT_GENERATOR_MODEL,
    max_retries: int = MAX_REPAIR_ATTEMPTS,
    output_dir: str = "output",
) -> Dict[str, Any]:
    """
    Attempt to repair broken CadQuery code up to ``max_retries`` times.

    Args:
        prompt:      Original user request (for context).
        code:        The failing CadQuery source code.
        error:       The exception traceback from the executor.
        retrieved:   Retriever output (used to include a code snippet as hint).
        model:       OpenRouter model slug (same as generator for consistency).
        max_retries: Maximum repair attempts before giving up.
        output_dir:  Directory for STL export on success.

    Returns:
        Dict with keys: success, code, stl_path, repair_attempts, errors.
    """
    context_snippet = _best_example_snippet(retrieved)
    errors_seen: List[str] = [error]
    current_code = code
    llm_traces: List[Dict[str, Any]] = []

    for attempt in range(1, max_retries + 1):
        print(f"    [repair {attempt}/{max_retries}] asking {model} to fix ...")

        user_msg = _USER_TEMPLATE.format(
            prompt=prompt,
            code=current_code,
            error=errors_seen[-1][:2000],   # cap traceback length
            context_snippet=context_snippet,
        )

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_msg},
        ]

        raw = call_openrouter(messages, model=model, temperature=0.0)
        trace = get_last_openrouter_trace()
        if trace:
            trace["attempt"] = attempt
            llm_traces.append(trace)
        fixed_code = _extract_code(raw)

        exec_result = _execute_in_subprocess(fixed_code, output_dir=output_dir)

        if exec_result["success"]:
            print(f"    [repair {attempt}/{max_retries}] OK succeeded")
            return {
                "success":         True,
                "code":            fixed_code,
                "stl_path":        exec_result["stl_path"],
                "repair_attempts": attempt,
                "errors":          errors_seen,
                "llm_traces":      llm_traces,
            }

        errors_seen.append(exec_result["error"] or "unknown error")
        current_code = fixed_code
        print(f"    [repair {attempt}/{max_retries}] FAIL still failing")

    return {
        "success":         False,
        "code":            current_code,
        "stl_path":        None,
        "repair_attempts": max_retries,
        "errors":          errors_seen,
        "llm_traces":      llm_traces,
    }


# -- Internal helpers ----------------------------------------------------------

def _best_example_snippet(retrieved: List[dict]) -> str:
    """Return a short Python snippet from the highest-scoring example chunk."""
    examples = sorted(
        [c for c in retrieved if c.get("source") == "examples"],
        key=lambda c: c.get("score", 0.0),
        reverse=True,
    )
    if not examples:
        return "(no example available)"

    chunk = examples[0]
    title = chunk.get("title", "")
    code_m = re.search(r'```python\s*([\s\S]*?)\s*```', chunk["text"])
    snippet = code_m.group(1).strip()[:700] if code_m else chunk["text"][:500]
    return f"# Similar example: {title}\n{snippet}"


def _execute_in_subprocess(code: str, output_dir: str) -> Dict[str, Any]:
    """Execute repaired code out-of-process so toxic geometry cannot crash Jupyter."""
    project_root = Path(__file__).resolve().parents[2]
    payload = {"code": code, "output_dir": output_dir}
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {project_root.as_posix()!r})\n"
        "from prompt2cad.execution.cad_executor import execute\n"
        "payload = json.loads(sys.stdin.read())\n"
        "result = execute(payload['code'], output_dir=payload['output_dir'])\n"
        "sys.stdout.write(json.dumps(result))\n"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "repair execution subprocess timed out after 180s",
            "stl_path": None,
            "code": code,
        }

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"subprocess exited with code {proc.returncode}"
        return {
            "success": False,
            "error": f"repair execution subprocess failed: {detail[:1000]}",
            "stl_path": None,
            "code": code,
        }

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": f"repair execution subprocess returned invalid JSON: {(proc.stdout or '')[:1000]}",
            "stl_path": None,
            "code": code,
        }

    if not isinstance(result, dict):
        return {
            "success": False,
            "error": "repair execution subprocess returned a non-dict result",
            "stl_path": None,
            "code": code,
        }

    return result


def _extract_code(text: str) -> str:
    """Extract Python from the LLM response (handles fenced blocks or raw code)."""
    m = re.search(r'```python\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if m:
        return m.group(1).strip()
    return text.strip()

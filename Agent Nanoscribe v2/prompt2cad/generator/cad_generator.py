"""
Generator LLM -- produces executable CadQuery Python code.

The generator receives the user prompt, retrieved example snippets,
and a cheatsheet chunk as context.  The model is configurable per call
so multiple generators can be benchmarked against the same prompt.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

from ..config import DEFAULT_GENERATOR_MODEL, RAG_DIR
from ..utils import call_openrouter

# -- Prompt templates ----------------------------------------------------------

_SYSTEM = """You are a CadQuery Python code generator.

Rules (MUST follow):
1. Always begin with: import cadquery as cq
2. The final geometry MUST be assigned to a variable named `result`
3. Never call show_object(), display(), or any GUI/notebook function
4. Study the provided examples carefully and match their modeling patterns
5. Return ONLY valid Python code -- no markdown fences, no prose explanation
6. CadQuery 2.x (OCP-based) API only

CRITICAL -- Arrays and repeated geometry:
Calling .workplane() on a CadQuery object that already has accumulated geometry
places the new workplane at the BOUNDING BOX CENTER of existing solids -- NOT at z=0.
This causes each repeated feature to be offset higher than the last (staircase effect).

CORRECT pattern for arrays (each solid anchored at z=0):

    result = cq.Workplane("XY")
    for i in range(rows):
        for j in range(cols):
            x = j * spacing
            y = i * spacing
            cyl = cq.Workplane("XY").transformed(offset=cq.Vector(x, y, 0)).circle(r).extrude(h)
            result = result.add(cyl)
    result = result.combine()

WRONG pattern (DO NOT USE -- causes z-staircase bug):

    for i in range(rows):
        for j in range(cols):
            result = result.workplane(...).center(x, y).circle(r).extrude(h)  # WRONG

Always build each repeated solid from a fresh cq.Workplane("XY"), then add/combine.

Other pitfalls to avoid:
- Workplane has NO .torus() method -- use cq.Solid.makeTorus(major_r, minor_r)
  then wrap: cq.Workplane("XY").add(cq.Solid.makeTorus(...))
- circle() takes RADIUS, not diameter
- hole() takes DIAMETER, not radius
- revolve() profile must not be centred on the axis of revolution
  (use rect(..., centered=False) for a solid of revolution)
- Always call .close() before .extrude() on open wires"""

_USER_TEMPLATE = """\
User request:
{prompt}

------------------------------------------------------------
Retrieved examples (use these modeling patterns as guidance):
{examples_block}

------------------------------------------------------------
CadQuery cheatsheet reference:
{cheatsheet_block}
------------------------------------------------------------

Generate the CadQuery Python code. Assign the result to `result`.\
"""


# -- Public API ----------------------------------------------------------------

def generate(
    prompt: str,
    retrieved: Optional[List[dict]] = None,
    rag_examples: Optional[List[dict]] = None,
    model: str = DEFAULT_GENERATOR_MODEL,
) -> str:
    """
    Generate CadQuery code for `prompt` using the retrieved context.

    Args:
        prompt:    Original user request.
        retrieved: Chunks from the retriever (mix of examples + cheatsheet).
        rag_examples: Structured perfect-example entries to inject as examples.
        model:     OpenRouter model slug for the generator.

    Returns:
        Python source code as a string (ready to exec()).
    """
    merged = _merge_context(retrieved or [], rag_examples or [])
    examples_block   = _format_examples(merged)
    cheatsheet_block = _format_cheatsheet(merged)

    user_msg = _USER_TEMPLATE.format(
        prompt=prompt,
        examples_block=examples_block,
        cheatsheet_block=cheatsheet_block,
    )

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    raw = call_openrouter(messages, model=model, temperature=0.0)
    return _extract_code(raw)


# -- Formatting helpers --------------------------------------------------------

def _format_examples(retrieved: List[dict]) -> str:
    """Extract clean Python code from each retrieved example chunk."""
    examples = [c for c in retrieved if c.get("source") == "examples"]
    if not examples:
        return "(no examples retrieved)"

    blocks = []
    for chunk in examples:
        title = chunk.get("title", "")
        score = chunk.get("score", 0.0)

        # Extract only the Python code block from the full markdown chunk
        code_m = re.search(r'```python\s*([\s\S]*?)\s*```', chunk["text"])
        code_snippet = code_m.group(1).strip() if code_m else chunk["text"][:600]

        # Extract tags for semantic context
        meta = chunk.get("metadata", {})
        tags = meta.get("tags", [])
        tag_line = f"# tags: {', '.join(tags)}" if tags else ""

        blocks.append(
            f"# -- Example: {title}  (score={score:.3f}) --\n"
            + (tag_line + "\n" if tag_line else "")
            + code_snippet
        )

    return "\n\n".join(blocks)


def _format_cheatsheet(retrieved: List[dict]) -> str:
    """Return the highest-scoring non-example chunk (cheatsheet or index)."""
    ref_chunks = [c for c in retrieved if c.get("source") != "examples"]
    if not ref_chunks:
        return "(no cheatsheet chunk retrieved)"

    # Sort by score descending, take the best one
    best = max(ref_chunks, key=lambda c: c.get("score", 0.0))
    return best["text"][:1400]  # cap to ~350 tokens


def _merge_context(retrieved: List[dict], rag_examples: List[dict]) -> List[dict]:
    """Merge legacy retrieval chunks with structured experiment RAG examples."""
    merged = list(retrieved)

    for idx, entry in enumerate(rag_examples, start=1):
        merged.append(_rag_entry_to_chunk(entry, idx))

    if not any(chunk.get("source") != "examples" for chunk in merged):
        merged.append(_default_cheatsheet_chunk())

    return merged


def _rag_entry_to_chunk(entry: dict, idx: int) -> dict:
    prompt = str(entry.get("prompt", "")).strip()
    geometry_type = str(entry.get("geometry_type", "unknown")).strip()
    accuracy = float(entry.get("accuracy", 0.0) or 0.0)
    code = str(entry.get("cad_code", "")).strip()

    expected_params = entry.get("expected_parameters", {}) or {}
    complexity = entry.get("complexity_metrics", {}) or {}

    param_line = ", ".join(
        f"{key}={value}" for key, value in expected_params.items()
    ) or "none"
    complexity_line = ", ".join(
        f"{key}={value}" for key, value in complexity.items()
    ) or "none"

    text = (
        f"Prompt: {prompt}\n"
        f"Geometry: {geometry_type}\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Expected parameters: {param_line}\n"
        f"Complexity metrics: {complexity_line}\n\n"
        f"```python\n{code}\n```"
    )

    return {
        "source": "examples",
        "title": f"Perfect example {idx}: {geometry_type}",
        "score": accuracy,
        "text": text,
        "metadata": {
            "tags": [geometry_type, "perfect-example"],
        },
    }


def _default_cheatsheet_chunk() -> dict:
    path = Path(RAG_DIR) / "cadquery_cheatsheet.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = "(CadQuery cheatsheet unavailable)"

    return {
        "source": "cheatsheet",
        "title": "CadQuery cheatsheet",
        "score": 1.0,
        "text": text,
        "metadata": {"tags": ["cadquery", "reference"]},
    }


def _extract_code(text: str) -> str:
    """
    Extract Python code from the LLM response.

    Handles:
    - Fenced code blocks (```python ... ```)
    - Generic fenced blocks (``` ... ```)
    - Raw code with no fences
    """
    # Try fenced python block first
    m = re.search(r'```python\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Any fenced block
    m = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if m:
        code = m.group(1).strip()
        # Only return if it looks like Python (has 'import' or 'cq.')
        if "import" in code or "cq." in code:
            return code

    # No fences - return the whole response (might already be code)
    return text.strip()

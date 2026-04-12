"""
Convert PDFs or parsed paper text into grounded Prompt2CAD input prompts.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import PLANNER_MODEL
from ..utils import call_openrouter
from .pdf_processor import process_pdf_to_agent_doc

_SYSTEM = """\
You convert parsed academic paper content into a single CAD generation prompt.

Return EXACTLY one JSON object with these required fields:
- "design_prompt": a concise but specific prompt for a CAD generation pipeline
- "paper_summary": 1-3 sentences on the structure or device described
- "geometry_keywords": list of short geometry terms
- "dimensions": list of grounded dimension strings copied from the paper when available
- "fabrication_constraints": list of fabrication or manufacturability constraints
- "missing_information": list of important details that were not explicitly specified

Rules:
1. Prefer explicit geometry, dimensions, and arrangement described in the paper
2. Do not invent unsupported numeric values
3. If a detail is missing, say it is unspecified
4. The "design_prompt" must be plain text suitable for a CAD model generator
5. Keep the prompt focused on geometry and fabrication constraints, not scientific background
"""

_USER_TEMPLATE = """\
User design goal:
{design_goal}

Paper name:
{paper_name}

Selected paper excerpts:
{paper_excerpt}
"""

_GEOMETRY_HINTS = (
    "diameter",
    "radius",
    "height",
    "width",
    "length",
    "pitch",
    "spacing",
    "thickness",
    "array",
    "pillar",
    "hole",
    "cylinder",
    "cone",
    "channel",
    "lattice",
    "grid",
    "shell",
    "cap",
    "post",
)

_UNIT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:nm|um|μm|mm|cm|deg|degrees|x)\b",
    re.IGNORECASE,
)

_PDF_SYSTEM = """\
You are reading a research PDF directly and turning it into a CAD-generation prompt.

Return EXACTLY one JSON object with these required fields:
- "design_prompt": a concrete prompt to feed into a CAD generation pipeline
- "paper_summary": 1-3 sentences describing the most relevant geometry from the paper
- "geometry_keywords": list of short geometry terms
- "dimensions": list of explicit dimensions or counts found in the PDF
- "fabrication_constraints": list of fabrication constraints or process notes relevant to CAD
- "missing_information": list of key geometry details not explicitly specified

Rules:
1. Base the output only on the PDF contents and the user goal
2. Do not invent unsupported numeric values
3. Prefer the most concrete, manufacturable geometry in the paper
4. Keep the design_prompt focused on shape, arrangement, scale, and fabrication constraints
5. If the paper is ambiguous, say what is missing instead of guessing
"""

_PDF_USER_TEMPLATE = """\
User design goal:
{design_goal}

Read the attached PDF and produce a Prompt2CAD-ready prompt.
"""


def build_design_prompt_from_pdf(
    pdf_path: str | Path,
    design_goal: Optional[str] = None,
    model: str = PLANNER_MODEL,
    artifact_dir: str | Path | None = None,
    pdf_engine: str = "mistral-ocr",
    use_local_parser: bool = False,
) -> Dict[str, Any]:
    """
    Build a Prompt2CAD-ready prompt from a PDF.

    Default path sends the raw PDF to OpenRouter with the file-parser plugin.
    Set ``use_local_parser=True`` to parse locally with PyMuPDF first, then
    build the prompt from extracted text/images.
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    goal = design_goal or "Create the most concrete manufacturable geometry described by this paper."
    if use_local_parser:
        parsed_artifact_dir = None
        if artifact_dir is not None:
            parsed_artifact_dir = Path(artifact_dir) / "parsed_pdf"
        parsed_paper = process_pdf_to_agent_doc(
            path,
            output_dir=parsed_artifact_dir,
            extract_images=True,
        )
        result = build_design_prompt_from_parsed_paper(
            parsed_paper=parsed_paper,
            design_goal=goal,
            model=model,
            artifact_dir=artifact_dir,
        )
        result["source_path"] = str(path)
        result["parsing_mode"] = "local-pymupdf"
        result["pdf_engine"] = "local-pymupdf"
        result["parsed_artifacts"] = parsed_paper.get("artifacts", {})
        return result

    file_data = base64.b64encode(path.read_bytes()).decode("ascii")
    data_url = f"data:application/pdf;base64,{file_data}"

    raw = call_openrouter(
        messages=[
            {"role": "system", "content": _PDF_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PDF_USER_TEMPLATE.format(design_goal=goal)},
                    {
                        "type": "file",
                        "file": {
                            "filename": path.name,
                            "file_data": data_url,
                        },
                    },
                ],
            },
        ],
        model=model,
        temperature=0.0,
        json_mode=True,
        extra_body={
            "plugins": [
                {
                    "id": "file-parser",
                    "pdf": {"engine": pdf_engine},
                }
            ]
        },
        timeout=240,
    )

    payload = _parse_json_payload(raw)
    design_prompt = str(payload.get("design_prompt", "")).strip()
    if not design_prompt:
        raise ValueError("The PDF-to-prompt LLM response did not include a design_prompt.")

    result: Dict[str, Any] = {
        "source_name": path.name,
        "source_path": str(path),
        "design_goal": goal,
        "design_prompt": design_prompt,
        "paper_summary": str(payload.get("paper_summary", "")).strip(),
        "geometry_keywords": _coerce_str_list(payload.get("geometry_keywords")),
        "dimensions": _coerce_str_list(payload.get("dimensions")),
        "fabrication_constraints": _coerce_str_list(payload.get("fabrication_constraints")),
        "missing_information": _coerce_str_list(payload.get("missing_information")),
        "model": model,
        "pdf_engine": pdf_engine,
        "parsing_mode": "openrouter-file-parser",
        "raw_response": raw,
    }

    if artifact_dir is not None:
        _write_pdf_artifacts(result, Path(artifact_dir))

    return result


def build_design_prompt_from_parsed_paper(
    parsed_paper: Dict[str, Any],
    design_goal: Optional[str] = None,
    model: str = PLANNER_MODEL,
    artifact_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Build a grounded CAD prompt from parsed paper text.
    """
    source_name = str(parsed_paper.get("source_name", "paper.pdf"))
    combined_text = str(parsed_paper.get("combined_text", "")).strip()
    excerpt = select_relevant_excerpt(combined_text)
    goal = design_goal or "Create the most concrete manufacturable geometry described by this paper."

    raw = call_openrouter(
        [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    design_goal=goal,
                    paper_name=source_name,
                    paper_excerpt=excerpt or "(no parsed text available)",
                ),
            },
        ],
        model=model,
        temperature=0.0,
        json_mode=True,
    )

    payload = _parse_json_payload(raw)
    design_prompt = str(payload.get("design_prompt", "")).strip()
    if not design_prompt:
        design_prompt = _fallback_prompt(goal, excerpt)

    result: Dict[str, Any] = {
        "source_name": source_name,
        "design_goal": goal,
        "paper_excerpt": excerpt,
        "design_prompt": design_prompt,
        "paper_summary": str(payload.get("paper_summary", "")).strip(),
        "geometry_keywords": _coerce_str_list(payload.get("geometry_keywords")),
        "dimensions": _coerce_str_list(payload.get("dimensions")),
        "fabrication_constraints": _coerce_str_list(payload.get("fabrication_constraints")),
        "missing_information": _coerce_str_list(payload.get("missing_information")),
        "model": model,
        "raw_response": raw,
    }

    if artifact_dir is not None:
        _write_artifacts(result, Path(artifact_dir))

    return result


def select_relevant_excerpt(text: str, max_blocks: int = 18, max_chars: int = 12000) -> str:
    """
    Deterministically select geometry-heavy paragraphs from parsed paper text.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return ""

    scored: List[tuple[int, int, str]] = []
    for idx, block in enumerate(blocks):
        lower = block.lower()
        score = 0
        score += 3 * len(_UNIT_PATTERN.findall(block))
        score += sum(2 for hint in _GEOMETRY_HINTS if hint in lower)
        if any(word in lower for word in ("method", "fabrication", "structure", "device", "geometry")):
            score += 2
        if "abstract" in lower or "reference" in lower:
            score -= 2
        scored.append((score, idx, block))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    selected = sorted(ranked[:max_blocks], key=lambda item: item[1])

    parts: List[str] = []
    total_chars = 0
    for _, idx, block in selected:
        tagged = f"[Block {idx + 1}]\n{block}"
        if total_chars + len(tagged) > max_chars:
            break
        parts.append(tagged)
        total_chars += len(tagged) + 2

    return "\n\n".join(parts).strip()


def _fallback_prompt(design_goal: str, excerpt: str) -> str:
    excerpt = excerpt[:1800].strip()
    return (
        f"{design_goal}\n\n"
        "Use only explicit geometric and fabrication details from the parsed paper below. "
        "If dimensions are missing, keep them unspecified rather than inventing them.\n\n"
        f"{excerpt}"
    ).strip()


def _parse_json_payload(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    item = str(value).strip()
    return [item] if item else []


def _write_artifacts(result: Dict[str, Any], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "prompt_package.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "generated_prompt.txt").write_text(
        result.get("design_prompt", ""),
        encoding="utf-8",
    )
    (artifact_dir / "paper_excerpt.md").write_text(
        result.get("paper_excerpt", ""),
        encoding="utf-8",
    )


def _write_pdf_artifacts(result: Dict[str, Any], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "prompt_package.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "generated_prompt.txt").write_text(
        result.get("design_prompt", ""),
        encoding="utf-8",
    )

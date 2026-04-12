"""
Build Canonical Geometry Experiment (CGE) records from parsed papers.

Two entry points:
  build_cge_from_pdf()     -- legacy v1, returns family-based dict (preserved for compatibility)
  build_cge_v2_from_pdf()  -- v2, returns CanonicalGeometry with open provenance-tagged schema
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import PDF_CGE_OUTPUT_DIR, PLANNER_MODEL
from ..utils import call_openrouter
from .pdf_ingest import list_pdf_uploads
from .pdf_processor import process_pdf_to_agent_doc

_CGE_SYSTEM = """\
You convert parsed research-paper excerpts into a Canonical Geometry Experiment (CGE).

Return EXACTLY one JSON object with this shape:
{
  "family": "short family label",
  "components": [
    {
      "name": "short name",
      "type": "shape/component type",
      "count": 1,
      "role": "primary|support|array_element|feature|unknown",
      "evidence": [{"page": 1, "snippet": "short quote"}]
    }
  ],
  "arrangement": {
    "type": "single|array|grid|line|ring|cluster|unknown",
    "count_x": 1,
    "count_y": 1,
    "pitch_x_um": null,
    "pitch_y_um": null,
    "notes": "short note"
  },
  "known_parameters": [
    {
      "name": "parameter name",
      "value": 0.0,
      "unit": "um|nm|mm|cm|deg|ratio|count|unknown",
      "evidence": [{"page": 1, "snippet": "short quote"}]
    }
  ],
  "unknown_parameters": [
    {
      "name": "parameter name",
      "reason": "why unknown",
      "suggested_bounds": {"min": null, "max": null, "unit": "um|nm|mm|cm|deg|unknown"}
    }
  ],
  "hard_assertions": ["must-hold geometric/fabrication assertions from explicit paper statements"],
  "soft_targets": ["optimization objectives"],
  "fabrication_constraints": ["process constraints relevant to manufacturability"]
}

Rules:
- Use only evidence from the provided excerpts.
- Do not invent unsupported numeric values.
- Put uncertain values in unknown_parameters rather than known_parameters.
- Keep hard_assertions strict and testable.
- Output JSON only.
"""

_CGE_USER_TEMPLATE = """\
User design goal:
{design_goal}

Paper name:
{paper_name}

Selected paper excerpts:
{paper_excerpt}
"""

_DIMENSION_PATTERN = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>nm|um|umm|mm|cm|deg|degrees)\b",
    re.IGNORECASE,
)

_ARRAY_PATTERN = re.compile(r"\b(?P<x>\d+)\s*[xx]\s*(?P<y>\d+)\b", re.IGNORECASE)


def build_cge_from_pdf(
    pdf_path: str | Path,
    *,
    design_goal: Optional[str] = None,
    model: str = PLANNER_MODEL,
    use_llm: bool = True,
    parse_output_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Parse one PDF locally, then build a CGE package (no CAD generation)."""
    parsed = process_pdf_to_agent_doc(
        pdf_path=pdf_path,
        output_dir=parse_output_dir,
        extract_images=True,
    )
    result = build_cge_from_parsed_paper(
        parsed_paper=parsed,
        design_goal=design_goal,
        model=model,
        use_llm=use_llm,
        artifact_dir=artifact_dir,
    )
    result["source_path"] = str(Path(pdf_path).expanduser().resolve())
    result["parsed_artifacts"] = parsed.get("artifacts", {})
    return result


def build_cge_from_parsed_paper(
    parsed_paper: Dict[str, Any],
    *,
    design_goal: Optional[str] = None,
    model: str = PLANNER_MODEL,
    use_llm: bool = True,
    artifact_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Build a Canonical Geometry Experiment (CGE) from parsed paper text."""
    source_name = str(parsed_paper.get("source_name", "paper.pdf"))
    combined_text = str(parsed_paper.get("combined_text", "")).strip()
    excerpt = _select_relevant_excerpt(combined_text)
    goal = design_goal or "Extract a canonical geometry experiment representation from this paper."

    raw = ""
    payload: Dict[str, Any] = {}
    mode = "heuristic"

    if use_llm:
        try:
            raw = call_openrouter(
                [
                    {"role": "system", "content": _CGE_SYSTEM},
                    {
                        "role": "user",
                        "content": _CGE_USER_TEMPLATE.format(
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
            if payload:
                mode = "llm"
        except Exception:
            payload = {}

    cge = _normalize_cge_payload(payload)
    if not _has_min_cge_signal(cge):
        cge = _heuristic_cge_from_text(combined_text)
        mode = "heuristic"

    result: Dict[str, Any] = {
        "source_name": source_name,
        "design_goal": goal,
        "paper_excerpt": excerpt,
        "cge": cge,
        "builder_mode": mode,
        "model": model if use_llm else None,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw,
    }

    if artifact_dir is not None:
        _write_cge_artifacts(result, Path(artifact_dir))

    return result


def process_pdf_dir_to_cge(
    *,
    pdf_dir: str | Path,
    output_root: str | Path | None = None,
    design_goal: Optional[str] = None,
    model: str = PLANNER_MODEL,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory into CGE records."""
    root = Path(output_root).expanduser().resolve() if output_root else PDF_CGE_OUTPUT_DIR
    root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for pdf_path in list_pdf_uploads(pdf_dir):
        stem = _slugify(pdf_path.stem)
        out_dir = root / stem
        parsed_dir = out_dir / "parsed_pdf"
        result = build_cge_from_pdf(
            pdf_path,
            design_goal=design_goal,
            model=model,
            use_llm=use_llm,
            parse_output_dir=parsed_dir,
            artifact_dir=out_dir,
        )
        records.append(result)
    return records


def _has_min_cge_signal(cge: Dict[str, Any]) -> bool:
    return bool(cge.get("family") not in {"", "unknown"} or cge.get("known_parameters") or cge.get("hard_assertions"))


def _normalize_cge_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}

    components: List[Dict[str, Any]] = []
    for item in data.get("components") if isinstance(data.get("components"), list) else []:
        if not isinstance(item, dict):
            continue
        comp = {
            "name": str(item.get("name") or "component").strip() or "component",
            "type": str(item.get("type") or "unknown").strip() or "unknown",
            "count": _to_int(item.get("count"), default=1),
            "role": str(item.get("role") or "unknown").strip() or "unknown",
            "evidence": _normalize_evidence_list(item.get("evidence")),
        }
        components.append(comp)

    arrangement_raw = data.get("arrangement") if isinstance(data.get("arrangement"), dict) else {}
    arrangement = {
        "type": str(arrangement_raw.get("type") or "unknown").strip() or "unknown",
        "count_x": _to_int(arrangement_raw.get("count_x"), default=1),
        "count_y": _to_int(arrangement_raw.get("count_y"), default=1),
        "pitch_x_um": _to_float_or_none(arrangement_raw.get("pitch_x_um")),
        "pitch_y_um": _to_float_or_none(arrangement_raw.get("pitch_y_um")),
        "notes": str(arrangement_raw.get("notes") or "").strip(),
    }

    known_parameters: List[Dict[str, Any]] = []
    for item in data.get("known_parameters") if isinstance(data.get("known_parameters"), list) else []:
        if not isinstance(item, dict):
            continue
        value = _to_float_or_none(item.get("value"))
        if value is None:
            continue
        known_parameters.append(
            {
                "name": str(item.get("name") or "parameter").strip() or "parameter",
                "value": value,
                "unit": _normalize_unit(item.get("unit")),
                "evidence": _normalize_evidence_list(item.get("evidence")),
            }
        )

    unknown_parameters: List[Dict[str, Any]] = []
    for item in data.get("unknown_parameters") if isinstance(data.get("unknown_parameters"), list) else []:
        if not isinstance(item, dict):
            continue
        bounds = item.get("suggested_bounds") if isinstance(item.get("suggested_bounds"), dict) else {}
        unknown_parameters.append(
            {
                "name": str(item.get("name") or "parameter").strip() or "parameter",
                "reason": str(item.get("reason") or "unspecified in source").strip() or "unspecified in source",
                "suggested_bounds": {
                    "min": _to_float_or_none(bounds.get("min")),
                    "max": _to_float_or_none(bounds.get("max")),
                    "unit": _normalize_unit(bounds.get("unit")),
                },
            }
        )

    return {
        "family": str(data.get("family") or "unknown").strip() or "unknown",
        "components": components,
        "arrangement": arrangement,
        "known_parameters": known_parameters,
        "unknown_parameters": unknown_parameters,
        "hard_assertions": _coerce_str_list(data.get("hard_assertions")),
        "soft_targets": _coerce_str_list(data.get("soft_targets")),
        "fabrication_constraints": _coerce_str_list(data.get("fabrication_constraints")),
    }


def _heuristic_cge_from_text(text: str) -> Dict[str, Any]:
    lower = text.lower()
    family = "unknown"
    if "lens" in lower:
        family = "micro_lens_array"
    elif "cylinder" in lower:
        family = "cylinder"
    elif "sphere" in lower:
        family = "sphere"
    elif "cone" in lower:
        family = "cone"

    known_parameters: List[Dict[str, Any]] = []
    for idx, match in enumerate(_DIMENSION_PATTERN.finditer(text)):
        if idx >= 10:
            break
        value = float(match.group("value"))
        unit = _normalize_unit(match.group("unit"))
        snippet = _extract_snippet(text, match.start(), match.end())
        known_parameters.append(
            {
                "name": f"dimension_{idx + 1}",
                "value": value,
                "unit": unit,
                "evidence": [{"page": None, "snippet": snippet}],
            }
        )

    count_x, count_y = 1, 1
    arrangement_type = "single"
    arr = _ARRAY_PATTERN.search(text)
    if arr:
        count_x = _to_int(arr.group("x"), default=1)
        count_y = _to_int(arr.group("y"), default=1)
        arrangement_type = "array"

    hard_assertions = []
    if known_parameters:
        hard_assertions.append("Use only dimensions explicitly reported in the source paper.")
    if count_x > 1 or count_y > 1:
        hard_assertions.append(f"Array cardinality must remain {count_x} x {count_y}.")

    unknown_parameters: List[Dict[str, Any]] = []
    if not any("focal" in k.get("name", "") for k in known_parameters) and "focal" in lower:
        unknown_parameters.append(
            {
                "name": "focal_length",
                "reason": "mentioned conceptually but not captured as a reliable numeric field",
                "suggested_bounds": {"min": None, "max": None, "unit": "um"},
            }
        )

    return {
        "family": family,
        "components": [
            {
                "name": family if family != "unknown" else "primary_structure",
                "type": family if family != "unknown" else "unknown",
                "count": max(1, count_x * count_y),
                "role": "primary",
                "evidence": [],
            }
        ],
        "arrangement": {
            "type": arrangement_type,
            "count_x": count_x,
            "count_y": count_y,
            "pitch_x_um": None,
            "pitch_y_um": None,
            "notes": "heuristic extraction",
        },
        "known_parameters": known_parameters,
        "unknown_parameters": unknown_parameters,
        "hard_assertions": hard_assertions,
        "soft_targets": ["maximize printability margin"],
        "fabrication_constraints": [],
    }


def _write_cge_artifacts(result: Dict[str, Any], artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    package_path = artifact_dir / "cge_package.json"
    cge_path = artifact_dir / "cge.json"
    excerpt_path = artifact_dir / "paper_excerpt.md"

    package_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    cge_path.write_text(json.dumps(result.get("cge", {}), indent=2, ensure_ascii=False), encoding="utf-8")
    excerpt_path.write_text(str(result.get("paper_excerpt") or ""), encoding="utf-8")


def _select_relevant_excerpt(text: str, *, max_blocks: int = 18, max_chars: int = 12000) -> str:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return ""

    scored: List[tuple[int, int, str]] = []
    for idx, block in enumerate(blocks):
        lower = block.lower()
        score = 0
        score += 3 * len(_DIMENSION_PATTERN.findall(block))
        if any(word in lower for word in ("diameter", "radius", "pitch", "thickness", "array", "focal")):
            score += 3
        if "figure" in lower or "table" in lower:
            score += 1
        if "reference" in lower:
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


def _parse_json_payload(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _normalize_evidence_list(value: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "page": _to_int_or_none(item.get("page")),
                "snippet": str(item.get("snippet") or "").strip(),
            }
        )
    return rows


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    item = str(value).strip()
    return [item] if item else []


def _to_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_unit(value: Any) -> str:
    unit = str(value or "unknown").strip().lower()
    if unit == "umm":
        return "um"
    if unit == "degrees":
        return "deg"
    allowed = {"um", "nm", "mm", "cm", "deg", "ratio", "count", "unknown"}
    return unit if unit in allowed else "unknown"


def _extract_snippet(text: str, start: int, end: int, *, window: int = 70) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return " ".join(text[lo:hi].split())


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_").lower()
    return slug or "paper"


# ---------------------------------------------------------------------------
# v2 entry point -- returns CanonicalGeometry
# ---------------------------------------------------------------------------

_CGE_V2_SYSTEM = """\
You convert parsed research-paper excerpts into a structured geometry representation.

Return EXACTLY one JSON object:
{
  "geometry_description": "Rich, specific natural language description of the geometry. Include: topology, symmetry class, arrangement type, cross-section shapes, any relational facts (e.g. 'alternating layers are orthogonal'). This is the ground truth.",
  "parameters": [
    {
      "name": "parameter_name",
      "value": <number or string>,
      "unit": "um|nm|mm|deg|ratio|count",
      "confidence": 0.0-1.0,
      "location": "where in the paper (e.g. Table 1, Section 2.3)",
      "raw_text": "exact quote from source"
    }
  ],
  "constraints": [
    {
      "description": "relational fact that cannot be captured by a scalar",
      "confidence": 0.0-1.0,
      "raw_text": "supporting quote"
    }
  ],
  "unresolved": [
    {
      "name": "parameter_name",
      "reason": "why it could not be found"
    }
  ]
}

Rules:
- geometry_description must be specific, not generic. Describe THIS geometry, not geometry in general.
- Only put parameters in "parameters" if they are explicitly stated or clearly implied with high confidence.
- Low-confidence values go in "unresolved" with a reason, NOT in "parameters".
- Do not invent values. Do not use defaults.
- Output JSON only.
"""

_CGE_V2_USER_TEMPLATE = """\
Paper: {paper_name}
Design goal: {design_goal}

Paper excerpts:
{paper_excerpt}
"""


def build_cge_v2_from_pdf(
    pdf_path: str | Path,
    *,
    design_goal: Optional[str] = None,
    model: str = PLANNER_MODEL,
    artifact_dir: str | Path | None = None,
) -> "CanonicalGeometry":
    """
    Parse a PDF and return a v2 CanonicalGeometry with provenance-tagged fields.
    No family taxonomy -- geometry_description is free-form.
    """
    from ..agentic.cge import CanonicalGeometry

    parsed = process_pdf_to_agent_doc(pdf_path=pdf_path, extract_images=False)
    source_name = str(parsed.get("source_name", Path(pdf_path).name))
    combined_text = str(parsed.get("combined_text", "")).strip()
    excerpt = _select_relevant_excerpt(combined_text)
    goal = design_goal or "Extract the geometry described in this paper."

    raw = call_openrouter(
        [
            {"role": "system", "content": _CGE_V2_SYSTEM},
            {
                "role": "user",
                "content": _CGE_V2_USER_TEMPLATE.format(
                    paper_name=source_name,
                    design_goal=goal,
                    paper_excerpt=excerpt or "(no parsed text available)",
                ),
            },
        ],
        model=model,
        temperature=0.0,
        json_mode=True,
    )

    payload = _parse_json_payload(raw)
    cge = CanonicalGeometry(
        geometry_description=payload.get("geometry_description", f"Geometry from {source_name}")
    )

    for p in payload.get("parameters", []):
        if not isinstance(p, dict):
            continue
        value = p.get("value")
        if value is None:
            continue
        prov = {
            "document": source_name,
            "location": p.get("location", ""),
            "raw_text": p.get("raw_text", ""),
        }
        cge.set_field(
            name=str(p.get("name", "unknown")),
            value=value,
            unit=p.get("unit"),
            source="extracted",
            confidence=float(p.get("confidence", 0.75)),
            provenance=prov,
        )

    for c in payload.get("constraints", []):
        if not isinstance(c, dict):
            continue
        cge.add_constraint(
            description=str(c.get("description", "")),
            source="extracted",
            confidence=float(c.get("confidence", 0.8)),
            provenance={"raw_text": c.get("raw_text", ""), "document": source_name},
        )

    for u in payload.get("unresolved", []):
        if not isinstance(u, dict):
            continue
        cge.set_field(
            name=str(u.get("name", "unknown")),
            value=None,
            unit=None,
            source="unset",
            confidence=0.0,
            notes=str(u.get("reason", "not found in source")),
        )

    if artifact_dir is not None:
        out = Path(artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "cge_v2.json").write_text(cge.to_json(), encoding="utf-8")

    return cge

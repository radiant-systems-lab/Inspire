"""
Design storage for the geometry generation tool.

Handles creating, saving, and loading persistent design records so the
closed-loop agent can revisit designs for parameter exploration, fabrication
settings, and SEM validation.

Folder layout:
    agent_workspace/
        designs/
            design_001/
                prompt.txt
                cad_code.py
                render.png
                render_iso.png
                part.stl
                metadata.json
            design_002/
                ...
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DESIGNS_DIR


# ── Public API ────────────────────────────────────────────────────────────────

def create_design(
    prompt: str,
    cad_code: str,
    stl_path: Optional[str] = None,
    render_path: Optional[str] = None,
    render_iso_path: Optional[str] = None,
    render_top_path: Optional[str] = None,
    render_side_path: Optional[str] = None,
    parameters_detected: Optional[Dict[str, Any]] = None,
    status: str = "success",
    run_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a new design record, copy all artifacts into the design folder,
    and write metadata.json.

    Returns the metadata dict.
    """
    design_id = _next_design_id()
    design_dir = DESIGNS_DIR / design_id
    design_dir.mkdir(parents=True, exist_ok=True)

    (design_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (design_dir / "cad_code.py").write_text(cad_code, encoding="utf-8")

    stl_dest = _copy_file(stl_path, design_dir / "part.stl")
    render_dest = _copy_file(render_path, design_dir / "render.png")
    render_iso_dest = _copy_file(render_iso_path, design_dir / "render_iso.png")
    render_top_dest = _copy_file(render_top_path, design_dir / "render_top.png")
    render_side_dest = _copy_file(render_side_path, design_dir / "render_side.png")

    run_trace_path: Optional[str] = None
    decision_trace_path: Optional[str] = None
    if isinstance(run_trace, dict) and run_trace:
        trace_file = design_dir / "run_trace.json"
        trace_file.write_text(
            json.dumps(run_trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        run_trace_path = str(trace_file)

        decision_trace = run_trace.get("decision_trace")
        if isinstance(decision_trace, list):
            decision_file = design_dir / "decision_trace.json"
            decision_file.write_text(
                json.dumps(decision_trace, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            decision_trace_path = str(decision_file)

    metadata: Dict[str, Any] = {
        "design_id":           design_id,
        "prompt":              prompt,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "stl_path":            stl_dest,
        "render_path":         render_dest,
        "render_iso_path":     render_iso_dest,
        "render_top_path":     render_top_dest,
        "render_side_path":    render_side_dest,
        "cad_code_path":       str(design_dir / "cad_code.py"),
        "parameters_detected": parameters_detected or {},
        "status":              status,
        "run_trace_path":      run_trace_path,
        "decision_trace_path": decision_trace_path,
    }

    (design_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return metadata


def load_design(design_id: str) -> Optional[Dict[str, Any]]:
    """Load an existing design by ID. Returns None if not found."""
    meta_path = DESIGNS_DIR / design_id / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_designs() -> List[Dict[str, Any]]:
    """Return a list of all design metadata dicts, sorted by design ID."""
    if not DESIGNS_DIR.exists():
        return []

    designs = []
    for d in sorted(DESIGNS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("design_"):
            meta = load_design(d.name)
            if meta is not None:
                designs.append(meta)
    return designs


# ── Internal helpers ──────────────────────────────────────────────────────────

def _next_design_id() -> str:
    DESIGNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        d.name for d in DESIGNS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("design_")
    )
    if not existing:
        return "design_001"
    last_num = int(existing[-1].split("_")[1])
    return f"design_{last_num + 1:03d}"


def _copy_file(src: Optional[str], dest: Path) -> Optional[str]:
    if src and Path(src).exists():
        shutil.copy2(src, dest)
        return str(dest)
    return None

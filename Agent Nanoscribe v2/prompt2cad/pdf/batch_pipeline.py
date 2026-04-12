"""
Batch PDF -> prompt -> Prompt2CAD orchestration.
"""

from __future__ import annotations

import importlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import (
    DEFAULT_GENERATOR_MODEL,
    LOGS_DIR,
    PDF_PIPELINE_OUTPUT_DIR,
    PDF_UPLOAD_DIR,
    PLANNER_MODEL,
)
from ..pipeline.run_pipeline import run_pipeline
from .pdf_ingest import list_pdf_uploads
from .prompt_builder import build_design_prompt_from_pdf


def run_pdf_to_cad(
    pdf_path: str | Path,
    design_goal: Optional[str] = None,
    prompt_model: str = PLANNER_MODEL,
    cad_model: str = DEFAULT_GENERATOR_MODEL,
    output_root: str | Path | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process a single PDF through raw-PDF prompting and Prompt2CAD.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    run_dir = _make_run_dir(pdf_path, output_root)

    prompt_package = build_design_prompt_from_pdf(
        pdf_path=pdf_path,
        design_goal=design_goal,
        model=prompt_model,
        artifact_dir=run_dir / "prompt",
    )
    pipeline_result = run_pipeline(
        prompt=prompt_package["design_prompt"],
        model=cad_model,
        verbose=verbose,
        output_dir=run_dir / "pipeline_temp",
        stl_filename=f"{_slugify(pdf_path.stem)}.stl",
    )

    record: Dict[str, Any] = {
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "run_dir": str(run_dir),
        "prompt_package_path": str(run_dir / "prompt" / "prompt_package.json"),
        "generated_prompt_path": str(run_dir / "prompt" / "generated_prompt.txt"),
        "generated_prompt": prompt_package.get("design_prompt", ""),
        "paper_summary": prompt_package.get("paper_summary", ""),
        "geometry_keywords": prompt_package.get("geometry_keywords", []),
        "dimensions": prompt_package.get("dimensions", []),
        "fabrication_constraints": prompt_package.get("fabrication_constraints", []),
        "missing_information": prompt_package.get("missing_information", []),
        "design_goal": prompt_package.get("design_goal", ""),
        "prompt_model": prompt_model,
        "cad_model": cad_model,
        "design_id": pipeline_result.get("design_id"),
        "success": pipeline_result.get("success", False),
        "error": pipeline_result.get("error"),
        "stl_path": pipeline_result.get("stl_path"),
        "render_path": pipeline_result.get("render_path"),
        "render_iso_path": pipeline_result.get("render_iso_path"),
        "render_top_path": pipeline_result.get("render_top_path"),
        "render_side_path": pipeline_result.get("render_side_path"),
        "cad_code_path": pipeline_result.get("cad_code_path"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


def run_pdf_model_sweep(
    pdf_dir: str | Path | None = None,
    design_goal: Optional[str] = None,
    prompt_model: str = PLANNER_MODEL,
    cad_models: Optional[List[str]] = None,
    output_root: str | Path | None = None,
    max_repair_attempts: int = 2,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build one prompt per PDF, then run multiple CAD models against that same prompt.
    """
    pdfs = list_pdf_uploads(pdf_dir or PDF_UPLOAD_DIR)
    batch_root = Path(output_root) if output_root else PDF_PIPELINE_OUTPUT_DIR
    batch_root.mkdir(parents=True, exist_ok=True)

    models = cad_models or [DEFAULT_GENERATOR_MODEL]
    records: List[Dict[str, Any]] = []

    for pdf_path in pdfs:
        pdf_path = Path(pdf_path).expanduser().resolve()
        pdf_run_dir = _make_run_dir(pdf_path, batch_root)

        try:
            prompt_package = build_design_prompt_from_pdf(
                pdf_path=pdf_path,
                design_goal=design_goal,
                model=prompt_model,
                artifact_dir=pdf_run_dir / "prompt",
            )
        except Exception as exc:
            record = {
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
                "run_dir": str(pdf_run_dir),
                "prompt_package_path": None,
                "generated_prompt_path": None,
                "generated_prompt": "",
                "paper_summary": "",
                "geometry_keywords": [],
                "dimensions": [],
                "fabrication_constraints": [],
                "missing_information": [],
                "design_goal": design_goal or "",
                "prompt_model": prompt_model,
                "cad_model": None,
                "design_id": None,
                "success": False,
                "error": f"prompt_generation_failed: {exc}",
                "stl_path": None,
                "render_path": None,
                "render_iso_path": None,
                "render_top_path": None,
                "render_side_path": None,
                "cad_code_path": None,
                "repair_attempts": None,
                "total_seconds": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            (pdf_run_dir / "summary.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            records.append(record)
            continue

        for cad_model in models:
            model_dir = pdf_run_dir / "models" / _slugify(cad_model)
            model_dir.mkdir(parents=True, exist_ok=True)
            run_record = _run_prompt_with_model(
                prompt=prompt_package["design_prompt"],
                pdf_path=pdf_path,
                prompt_package=prompt_package,
                prompt_model=prompt_model,
                cad_model=cad_model,
                run_dir=model_dir,
                max_repair_attempts=max_repair_attempts,
                verbose=verbose,
            )
            records.append(run_record)

    (batch_root / "batch_model_sweep_summary.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def run_pdf_batch(
    pdf_dir: str | Path | None = None,
    design_goal: Optional[str] = None,
    prompt_model: str = PLANNER_MODEL,
    cad_model: str = DEFAULT_GENERATOR_MODEL,
    output_root: str | Path | None = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process every PDF in the upload folder and return one record per file.
    """
    pdfs = list_pdf_uploads(pdf_dir or PDF_UPLOAD_DIR)
    records: List[Dict[str, Any]] = []

    batch_root = Path(output_root) if output_root else PDF_PIPELINE_OUTPUT_DIR
    batch_root.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdfs:
        try:
            record = run_pdf_to_cad(
                pdf_path=pdf_path,
                design_goal=design_goal,
                prompt_model=prompt_model,
                cad_model=cad_model,
                output_root=batch_root,
                verbose=verbose,
            )
        except Exception as exc:
            run_dir = _make_run_dir(pdf_path, batch_root, create=True)
            record = {
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
                "run_dir": str(run_dir),
                "generated_prompt": "",
                "paper_summary": "",
                "geometry_keywords": [],
                "dimensions": [],
                "fabrication_constraints": [],
                "missing_information": [],
                "design_goal": design_goal or "",
                "prompt_model": prompt_model,
                "cad_model": cad_model,
                "design_id": None,
                "success": False,
                "error": str(exc),
                "stl_path": None,
                "render_path": None,
                "render_iso_path": None,
                "render_top_path": None,
                "render_side_path": None,
                "cad_code_path": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            (run_dir / "summary.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        records.append(record)

    (batch_root / "batch_summary.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten batch records for notebook display.
    """
    return pd.DataFrame(
        [
            {
                "pdf_name": rec.get("pdf_name"),
                "cad_model": rec.get("cad_model"),
                "success": rec.get("success"),
                "design_id": rec.get("design_id"),
                "repair_attempts": rec.get("repair_attempts"),
                "total_seconds": rec.get("total_seconds"),
                "prompt_preview": str(rec.get("generated_prompt", ""))[:140],
                "render_path": rec.get("render_path"),
                "render_iso_path": rec.get("render_iso_path"),
                "render_top_path": rec.get("render_top_path"),
                "render_side_path": rec.get("render_side_path"),
                "cad_code_path": rec.get("cad_code_path"),
                "error": rec.get("error"),
            }
            for rec in records
        ]
    )


def _make_run_dir(
    pdf_path: Path,
    output_root: str | Path | None,
    create: bool = True,
) -> Path:
    root = Path(output_root) if output_root else PDF_PIPELINE_OUTPUT_DIR
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{_slugify(pdf_path.stem)}_{stamp}"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower() or "paper"


def _run_prompt_with_model(
    prompt: str,
    pdf_path: Path,
    prompt_package: Dict[str, Any],
    prompt_model: str,
    cad_model: str,
    run_dir: Path,
    max_repair_attempts: int,
    verbose: bool,
) -> Dict[str, Any]:
    pipeline_module = importlib.import_module("prompt2cad.pipeline.run_pipeline")
    log_path = LOGS_DIR / "prompt_runs.json"
    before = []
    if log_path.exists():
        try:
            before = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            before = []
    before_len = len(before)

    original_max_repairs = getattr(pipeline_module, "MAX_REPAIR_ATTEMPTS", None)
    try:
        pipeline_module.MAX_REPAIR_ATTEMPTS = max_repair_attempts
        pipeline_result = pipeline_module.run_pipeline(
            prompt=prompt,
            model=cad_model,
            verbose=verbose,
            output_dir=run_dir / "pipeline_temp",
            stl_filename=f"{_slugify(pdf_path.stem)}.stl",
        )
    finally:
        if original_max_repairs is not None:
            pipeline_module.MAX_REPAIR_ATTEMPTS = original_max_repairs

    after = []
    if log_path.exists():
        try:
            after = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            after = []
    run_log = after[before_len] if len(after) > before_len else {}

    record: Dict[str, Any] = {
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "run_dir": str(run_dir),
        "prompt_package_path": str(run_dir.parent.parent / "prompt" / "prompt_package.json"),
        "generated_prompt_path": str(run_dir.parent.parent / "prompt" / "generated_prompt.txt"),
        "generated_prompt": prompt_package.get("design_prompt", ""),
        "paper_summary": prompt_package.get("paper_summary", ""),
        "geometry_keywords": prompt_package.get("geometry_keywords", []),
        "dimensions": prompt_package.get("dimensions", []),
        "fabrication_constraints": prompt_package.get("fabrication_constraints", []),
        "missing_information": prompt_package.get("missing_information", []),
        "design_goal": prompt_package.get("design_goal", ""),
        "prompt_model": prompt_model,
        "cad_model": cad_model,
        "design_id": pipeline_result.get("design_id"),
        "success": pipeline_result.get("success", False),
        "error": pipeline_result.get("error"),
        "stl_path": pipeline_result.get("stl_path"),
        "render_path": pipeline_result.get("render_path"),
        "render_iso_path": pipeline_result.get("render_iso_path"),
        "render_top_path": pipeline_result.get("render_top_path"),
        "render_side_path": pipeline_result.get("render_side_path"),
        "cad_code_path": pipeline_result.get("cad_code_path"),
        "repair_attempts": run_log.get("repair_attempts"),
        "total_seconds": run_log.get("total_seconds"),
        "execution_error": (run_log.get("execution_result") or {}).get("error"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record

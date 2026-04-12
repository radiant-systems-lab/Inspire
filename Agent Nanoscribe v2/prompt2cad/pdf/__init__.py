"""
PDF-driven Prompt2CAD helpers.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_design_prompt_from_pdf",
    "build_design_prompt_from_parsed_paper",
    "build_cge_from_pdf",
    "build_cge_from_parsed_paper",
    "list_pdf_uploads",
    "process_pdf_to_agent_doc",
    "process_pdf_uploads_to_agent_docs",
    "process_pdf_dir_to_cge",
    "run_pdf_batch",
    "run_pdf_model_sweep",
    "run_pdf_to_cad",
]


def list_pdf_uploads(*args: Any, **kwargs: Any):
    from .pdf_ingest import list_pdf_uploads as _list_pdf_uploads

    return _list_pdf_uploads(*args, **kwargs)


def build_design_prompt_from_pdf(*args: Any, **kwargs: Any):
    from .prompt_builder import build_design_prompt_from_pdf as _build_design_prompt_from_pdf

    return _build_design_prompt_from_pdf(*args, **kwargs)


def build_design_prompt_from_parsed_paper(*args: Any, **kwargs: Any):
    from .prompt_builder import (
        build_design_prompt_from_parsed_paper as _build_design_prompt_from_parsed_paper,
    )

    return _build_design_prompt_from_parsed_paper(*args, **kwargs)


def build_cge_from_pdf(*args: Any, **kwargs: Any):
    from .cge_builder import build_cge_from_pdf as _build_cge_from_pdf

    return _build_cge_from_pdf(*args, **kwargs)


def build_cge_from_parsed_paper(*args: Any, **kwargs: Any):
    from .cge_builder import build_cge_from_parsed_paper as _build_cge_from_parsed_paper

    return _build_cge_from_parsed_paper(*args, **kwargs)


def process_pdf_to_agent_doc(*args: Any, **kwargs: Any):
    from .pdf_processor import process_pdf_to_agent_doc as _process_pdf_to_agent_doc

    return _process_pdf_to_agent_doc(*args, **kwargs)


def process_pdf_uploads_to_agent_docs(*args: Any, **kwargs: Any):
    from .pdf_processor import (
        process_pdf_uploads_to_agent_docs as _process_pdf_uploads_to_agent_docs,
    )

    return _process_pdf_uploads_to_agent_docs(*args, **kwargs)


def process_pdf_dir_to_cge(*args: Any, **kwargs: Any):
    from .cge_builder import process_pdf_dir_to_cge as _process_pdf_dir_to_cge

    return _process_pdf_dir_to_cge(*args, **kwargs)


def run_pdf_to_cad(*args: Any, **kwargs: Any):
    from .batch_pipeline import run_pdf_to_cad as _run_pdf_to_cad

    return _run_pdf_to_cad(*args, **kwargs)


def run_pdf_batch(*args: Any, **kwargs: Any):
    from .batch_pipeline import run_pdf_batch as _run_pdf_batch

    return _run_pdf_batch(*args, **kwargs)


def run_pdf_model_sweep(*args: Any, **kwargs: Any):
    from .batch_pipeline import run_pdf_model_sweep as _run_pdf_model_sweep

    return _run_pdf_model_sweep(*args, **kwargs)

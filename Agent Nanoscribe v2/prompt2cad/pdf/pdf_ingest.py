"""
Local PDF file discovery helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..config import PDF_UPLOAD_DIR


def list_pdf_uploads(pdf_dir: str | Path | None = None) -> List[Path]:
    root = Path(pdf_dir) if pdf_dir else PDF_UPLOAD_DIR
    root.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")

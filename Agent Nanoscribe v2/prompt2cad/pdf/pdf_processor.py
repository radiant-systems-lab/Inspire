"""
Local PDF -> agent-friendly document processing (PyMuPDF).

This module extracts page text and embedded images from a PDF, then emits:
- a Markdown document suitable for LLM ingestion
- structured JSON metadata
- extracted image assets
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import PDF_PARSED_OUTPUT_DIR, PDF_UPLOAD_DIR
from .pdf_ingest import list_pdf_uploads


def process_pdf_to_agent_doc(
    pdf_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    extract_images: bool = True,
    max_caption_distance: float = 72.0,
) -> Dict[str, Any]:
    """
    Parse one PDF into an agent-friendly Markdown package.

    Returns a dict that is intentionally compatible with downstream prompt
    builders (`source_name`, `combined_text`) and includes artifact paths.
    """
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "PyMuPDF is required for local PDF parsing. Install with: "
            "python -m pip install pymupdf"
        ) from exc

    src = Path(pdf_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"PDF not found: {src}")

    run_dir = _resolve_output_dir(src, output_dir=output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir = run_dir / "images"
    if extract_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(src)
    try:
        page_count = doc.page_count
        pages: List[Dict[str, Any]] = []
        markdown_lines: List[str] = []
        combined_text_parts: List[str] = []
        image_hash_to_relpath: Dict[str, str] = {}

        markdown_lines.append(f"# {src.stem}")
        markdown_lines.append("")
        markdown_lines.append(f"- source: `{src.name}`")
        markdown_lines.append(f"- pages: {page_count}")
        markdown_lines.append("")

        for page_idx in range(page_count):
            page = doc[page_idx]
            page_no = page_idx + 1
            page_dict = page.get_text("dict")
            raw_blocks = list(page_dict.get("blocks") or [])
            blocks = sorted(raw_blocks, key=lambda b: _block_sort_key(b))

            text_blocks: List[Dict[str, Any]] = []
            image_blocks: List[Dict[str, Any]] = []

            for block in blocks:
                b_type = int(block.get("type", -1))
                if b_type == 0:
                    text = _extract_text_block(block)
                    if not text:
                        continue
                    text_blocks.append(
                        {
                            "bbox": _bbox_list(block.get("bbox")),
                            "text": text,
                        }
                    )
                elif b_type == 1 and extract_images:
                    image_info = _save_image_block(
                        block=block,
                        page_no=page_no,
                        image_index=len(image_blocks) + 1,
                        images_dir=images_dir,
                        image_hash_to_relpath=image_hash_to_relpath,
                    )
                    if image_info is not None:
                        image_blocks.append(image_info)

            for img in image_blocks:
                caption = _guess_caption_for_image(
                    image_bbox=img.get("bbox") or [0.0, 0.0, 0.0, 0.0],
                    text_blocks=text_blocks,
                    max_caption_distance=max_caption_distance,
                )
                img["caption_guess"] = caption

            page_text = "\n\n".join(block["text"] for block in text_blocks).strip()
            page_record: Dict[str, Any] = {
                "page_number": page_no,
                "text": page_text,
                "text_blocks": text_blocks,
                "images": image_blocks,
            }
            pages.append(page_record)

            combined_text_parts.append(f"[Page {page_no}]\n{page_text}".strip())

            markdown_lines.append(f"## Page {page_no}")
            markdown_lines.append("")
            markdown_lines.append("### Text")
            markdown_lines.append("")
            markdown_lines.append(page_text if page_text else "_No extractable text found._")
            markdown_lines.append("")

            if image_blocks:
                markdown_lines.append("### Images")
                markdown_lines.append("")
                for img in image_blocks:
                    rel = img.get("relative_path", "")
                    label = f"p{page_no:03d}_img{int(img.get('image_index', 0)):02d}"
                    markdown_lines.append(f"![{label}]({rel})")
                    if img.get("caption_guess"):
                        markdown_lines.append(f"- caption_guess: {img['caption_guess']}")
                    markdown_lines.append(
                        f"- bbox: {img.get('bbox')}  size: {img.get('width')}x{img.get('height')}  "
                        f"digest: {img.get('sha1', '')[:12]}"
                    )
                    markdown_lines.append("")

        combined_text = "\n\n".join(part for part in combined_text_parts if part.strip()).strip()
        markdown_text = "\n".join(markdown_lines).strip() + "\n"
    finally:
        doc.close()

    markdown_path = run_dir / "paper_agent.md"
    metadata_path = run_dir / "paper_agent.json"
    combined_text_path = run_dir / "combined_text.txt"

    markdown_path.write_text(markdown_text, encoding="utf-8")
    combined_text_path.write_text(combined_text, encoding="utf-8")

    result: Dict[str, Any] = {
        "source_name": src.name,
        "source_path": str(src),
        "page_count": page_count,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "combined_text": combined_text,
        "pages": pages,
        "artifacts": {
            "output_dir": str(run_dir),
            "markdown_path": str(markdown_path),
            "metadata_path": str(metadata_path),
            "combined_text_path": str(combined_text_path),
            "images_dir": str(images_dir) if extract_images else None,
        },
    }
    metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


def process_pdf_uploads_to_agent_docs(
    *,
    pdf_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    extract_images: bool = True,
    max_caption_distance: float = 72.0,
) -> List[Dict[str, Any]]:
    """
    Process all PDFs in an upload folder into agent docs.
    """
    rows: List[Dict[str, Any]] = []
    root = Path(output_root) if output_root else PDF_PARSED_OUTPUT_DIR
    root.mkdir(parents=True, exist_ok=True)
    for pdf in list_pdf_uploads(pdf_dir or PDF_UPLOAD_DIR):
        out_dir = root / _slugify(pdf.stem)
        rows.append(
            process_pdf_to_agent_doc(
                pdf,
                output_dir=out_dir,
                extract_images=extract_images,
                max_caption_distance=max_caption_distance,
            )
        )
    return rows


def _resolve_output_dir(src: Path, *, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return (PDF_PARSED_OUTPUT_DIR / _slugify(src.stem)).resolve()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_").lower()
    return slug or "paper"


def _bbox_list(value: Any) -> List[float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(v) for v in value]
    return [0.0, 0.0, 0.0, 0.0]


def _block_sort_key(block: Dict[str, Any]) -> Tuple[float, float]:
    bbox = _bbox_list(block.get("bbox"))
    return (bbox[1], bbox[0])


def _extract_text_block(block: Dict[str, Any]) -> str:
    lines = list(block.get("lines") or [])
    out_lines: List[str] = []
    for line in lines:
        spans = list(line.get("spans") or [])
        merged = "".join(str(span.get("text") or "") for span in spans).strip()
        if merged:
            out_lines.append(merged)
    return "\n".join(out_lines).strip()


def _save_image_block(
    *,
    block: Dict[str, Any],
    page_no: int,
    image_index: int,
    images_dir: Path,
    image_hash_to_relpath: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    raw = block.get("image")
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return None

    digest = hashlib.sha1(raw).hexdigest()
    ext = str(block.get("ext") or "png").strip().lower() or "png"
    if ext in {"jpeg"}:
        ext = "jpg"
    if not re.match(r"^[a-z0-9]+$", ext):
        ext = "png"

    if digest in image_hash_to_relpath:
        rel = image_hash_to_relpath[digest]
        was_deduped = True
    else:
        filename = f"page_{page_no:03d}_img_{image_index:02d}.{ext}"
        abs_path = images_dir / filename
        abs_path.write_bytes(bytes(raw))
        rel = str(Path("images") / filename)
        image_hash_to_relpath[digest] = rel
        was_deduped = False

    return {
        "page_number": page_no,
        "image_index": image_index,
        "relative_path": rel,
        "bbox": _bbox_list(block.get("bbox")),
        "width": int(block.get("width") or 0),
        "height": int(block.get("height") or 0),
        "sha1": digest,
        "deduplicated": was_deduped,
    }


def _guess_caption_for_image(
    *,
    image_bbox: List[float],
    text_blocks: List[Dict[str, Any]],
    max_caption_distance: float,
) -> str:
    ix0, iy0, ix1, iy1 = image_bbox
    best_text = ""
    best_score = float("inf")

    for block in text_blocks:
        bbox = _bbox_list(block.get("bbox"))
        tx0, ty0, tx1, _ = bbox
        text = str(block.get("text") or "").strip()
        if not text:
            continue

        # Prefer text immediately below image with horizontal overlap.
        if ty0 < iy1 - 2.0:
            continue
        y_delta = ty0 - iy1
        if y_delta > max_caption_distance:
            continue

        overlap = max(0.0, min(ix1, tx1) - max(ix0, tx0))
        min_w = max(1.0, min(ix1 - ix0, tx1 - tx0))
        overlap_ratio = overlap / min_w
        if overlap_ratio < 0.15:
            continue

        score = y_delta + (1.0 - overlap_ratio) * 25.0
        if score < best_score:
            best_score = score
            best_text = text.replace("\n", " ").strip()

    return best_text[:300]

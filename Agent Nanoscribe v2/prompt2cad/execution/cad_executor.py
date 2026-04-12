"""
CadQuery execution engine.

Runs generated code in an isolated namespace, captures exceptions,
and exports the result to STL if execution succeeds.
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


def execute(
    code: str,
    output_dir: str | Path = "output",
    stl_filename: str = "result.stl",
) -> Dict[str, Any]:
    """
    Execute CadQuery Python source code and export the result to STL.

    The code is executed inside a namespace that contains ``cq`` (the cadquery
    module) but is otherwise isolated from the calling process.  The code must
    assign its final geometry to a variable named ``result``.

    Args:
        code:         Python source string to execute.
        output_dir:   Directory where the STL file is written.
        stl_filename: Name of the exported STL file.

    Returns:
        A dict with keys:
          • ``"success"``   – bool
          • ``"error"``     – error/traceback string, or None on success
          • ``"stl_path"``  – path to exported STL, or None
          • ``"code"``      – the code that was executed (echoed back)
    """
    # ── 0. Check cadquery is importable ───────────────────────────────────────
    try:
        import cadquery as cq  # noqa: F401 (checked here, used in namespace)
    except ImportError:
        return _err(
            "cadquery is not installed.\n"
            "Install it with:  pip install cadquery",
            code,
        )

    # ── 1. Build a clean execution namespace ──────────────────────────────────
    namespace: Dict[str, Any] = {
        "cq": cq,
        "__name__": "__generated_cad__",
        "__builtins__": __builtins__,
    }

    # ── 2. Execute ────────────────────────────────────────────────────────────
    stdout_buf = io.StringIO()
    try:
        compiled = compile(code, "<generated_cad>", "exec")
        with _suppress_stdout(stdout_buf):
            exec(compiled, namespace)
    except Exception:
        return _err(traceback.format_exc(), code)

    # ── 3. Locate `result` variable ───────────────────────────────────────────
    result_obj = namespace.get("result")
    if result_obj is None:
        return _err(
            "Generated code did not define a variable named 'result'.\n"
            "The final geometry must be assigned like:\n"
            "    result = cq.Workplane('XY').box(10, 10, 5)",
            code,
        )

    # ── 4. Export STL ─────────────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stl_path = str(out_path / stl_filename)

    try:
        cq.exporters.export(result_obj, stl_path)
    except Exception as exc:
        # Code ran fine but STL export failed (uncommon; usually means
        # the result is empty or degenerate geometry)
        return {
            "success":  True,  # execution OK
            "error":    f"STL export failed: {exc}",
            "stl_path": None,
            "code":     code,
        }

    return {
        "success":  True,
        "error":    None,
        "stl_path": stl_path,
        "code":     code,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _err(message: str, code: str) -> Dict[str, Any]:
    return {"success": False, "error": message, "stl_path": None, "code": code}


class _suppress_stdout:
    """Context manager: redirect stdout to a buffer (suppresses CadQuery logging)."""

    def __init__(self, buf: io.StringIO) -> None:
        self._buf = buf
        self._saved: Optional[Any] = None

    def __enter__(self) -> "_suppress_stdout":
        self._saved = sys.stdout
        sys.stdout = self._buf
        return self

    def __exit__(self, *_: Any) -> None:
        if self._saved is not None:
            sys.stdout = self._saved

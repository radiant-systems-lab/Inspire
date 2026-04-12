from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from prompt2cad.execution.cad_executor import execute as execute_cadquery


_DANGEROUS_PATTERNS = [
    (r"\bos\.system\s*\(", "os.system is blocked"),
    (r"\bsubprocess\b", "subprocess module is blocked"),
    (r"\bshutil\.rmtree\b", "shutil.rmtree is blocked"),
    (r"\b__import__\s*\(", "dynamic imports are blocked"),
]


class ExecutionErrorType:
    VALIDATION = "validation_error"
    PREFLIGHT = "preflight_error"
    TIMEOUT = "timeout"
    EXECUTION = "execution_error"
    EXPORT = "export_error"



def validate_code(code: str) -> List[str]:
    warnings: List[str] = []
    for pattern, message in _DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            warnings.append(message)
    if "exec(" in code or "eval(" in code:
        warnings.append("nested exec/eval usage is blocked")
    return warnings



def preflight_subprocess(
    code: str,
    *,
    output_dir: str,
    stl_filename: str,
    timeout: int = 25,
) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    payload = {
        "code": code,
        "output_dir": output_dir,
        "stl_filename": stl_filename,
    }
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {project_root.as_posix()!r})\n"
        "from prompt2cad.execution.cad_executor import execute\n"
        "payload = json.loads(sys.stdin.read())\n"
        "result = execute(payload['code'], output_dir=payload['output_dir'], stl_filename=payload['stl_filename'])\n"
        "sys.stdout.write(json.dumps(result))\n"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"preflight timed out after {timeout}s",
            "error_type": ExecutionErrorType.TIMEOUT,
            "stl_path": None,
        }

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {
            "success": False,
            "error": f"preflight subprocess failed: {detail[:1200]}",
            "error_type": ExecutionErrorType.PREFLIGHT,
            "stl_path": None,
        }

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": f"preflight produced invalid JSON: {(proc.stdout or '')[:500]}",
            "error_type": ExecutionErrorType.PREFLIGHT,
            "stl_path": None,
        }

    if not isinstance(result, dict):
        return {
            "success": False,
            "error": "preflight returned non-dict response",
            "error_type": ExecutionErrorType.PREFLIGHT,
            "stl_path": None,
        }

    if not result.get("success"):
        result["error_type"] = ExecutionErrorType.PREFLIGHT
        return result

    result["error_type"] = None
    return result



def execute_code_safely(
    code: str,
    *,
    output_dir: str,
    stl_filename: str,
    timeout: int = 60,
    run_preflight: bool = True,
) -> Dict[str, Any]:
    warnings = validate_code(code)
    if warnings:
        return {
            "success": False,
            "error": "\n".join(warnings),
            "error_type": ExecutionErrorType.VALIDATION,
            "stl_path": None,
            "code": code,
        }

    if run_preflight:
        preflight = preflight_subprocess(
            code,
            output_dir=output_dir,
            stl_filename=stl_filename,
            timeout=min(timeout, 25),
        )
        if not preflight.get("success"):
            preflight["code"] = code
            return preflight

    result = execute_cadquery(code, output_dir=output_dir, stl_filename=stl_filename)
    if not result.get("success"):
        result["error_type"] = ExecutionErrorType.EXECUTION
        return result
    if result.get("error") and not result.get("stl_path"):
        result["error_type"] = ExecutionErrorType.EXPORT
    else:
        result["error_type"] = None
    return result

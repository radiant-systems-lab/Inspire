"""
CAD verification facade.
"""

from prompt2cad.execution.cad_verifier import (
    hard_verify,
    extract_cge_from_code,
    compute_geometry_metrics,
    structured_vision_analysis,
)

__all__ = [
    "hard_verify",
    "extract_cge_from_code",
    "compute_geometry_metrics",
    "structured_vision_analysis",
]


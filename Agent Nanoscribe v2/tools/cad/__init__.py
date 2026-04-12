"""
CAD tools facade.

Re-exports the core CAD entrypoints while the repo is being reorganized.
"""

from prompt2cad.agentic.cad_agent import CADAgent
from prompt2cad.pipeline.run_pipeline import run_pipeline

__all__ = ["CADAgent", "run_pipeline"]


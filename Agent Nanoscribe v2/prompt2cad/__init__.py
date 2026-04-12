"""
Prompt2CAD
==========
Natural language -> CadQuery code generation pipeline.

Quick start::

    from prompt2cad import run_pipeline

    result = run_pipeline(
        prompt="make a hollow cylinder with 2 mm wall thickness",
        model="openai/gpt-4o-mini",
    )
    print(result["final_code"])
    print(result["stl_path"])
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_pipeline", "run_autonomous_experiment_loop", "run_agent"]


def run_pipeline(*args: Any, **kwargs: Any):
    """
    Lazily import the heavy pipeline entrypoint so lightweight modules like
    ``prompt2cad.config`` can be imported without loading the full runtime.
    """
    from .pipeline.run_pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


def run_autonomous_experiment_loop(*args: Any, **kwargs: Any):
    """Lazily import the agentic closed-loop entrypoint."""
    from .agentic.loop import run_autonomous_experiment_loop as _run_autonomous_experiment_loop

    return _run_autonomous_experiment_loop(*args, **kwargs)


def run_agent(*args: Any, **kwargs: Any):
    """Lazily import the CadQuery agent runtime from the forked architecture package."""
    from forks.cadquery_agent import run_agent as _run_agent

    return _run_agent(*args, **kwargs)

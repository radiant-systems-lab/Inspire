"""
tools/

Stable import surface for the AgentNanoscribeV2 stack.

This package is a lightweight "facade" over the existing code layout
(`prompt2cad/`, `forks/`, etc.) so callers can depend on a clean namespace:

  - tools.cad            CAD generation + verification + CadQuery runtime
  - tools.forward_model  printability prediction + uncertainty + surrogates
  - tools.experiments    planners/executors/evaluators + active-learning loops

During the reorg, these modules primarily re-export implementations from
their current locations to avoid breaking existing code.
"""


"""CadQuery-native agent runtime patterned after freecad-ai architecture."""

from .session import CadSessionState
from .tool_registry import ToolDefinition, ToolRegistry
from .types import (
    AgentRunResult,
    DecisionTraceEvent,
    McpToolCallRequest,
    McpToolCallResponse,
    ToolCall,
    ToolResult,
)

__all__ = [
    "run_agent",
    "CadSessionState",
    "ToolDefinition",
    "ToolRegistry",
    "ToolCall",
    "ToolResult",
    "DecisionTraceEvent",
    "McpToolCallRequest",
    "McpToolCallResponse",
    "AgentRunResult",
]


def run_agent(*args, **kwargs):
    from .runtime import run_agent as _run_agent

    return _run_agent(*args, **kwargs)

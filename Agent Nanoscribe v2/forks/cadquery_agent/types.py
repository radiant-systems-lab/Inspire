from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTraceEvent:
    step: int
    intent: str
    selected_tools: List[str]
    why: str
    confidence: float
    result: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class McpToolCallRequest:
    name: str
    arguments: Dict[str, Any]


@dataclass
class McpToolCallResponse:
    success: bool
    content: List[Dict[str, Any]]
    is_error: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "content": list(self.content),
            "isError": bool(self.is_error),
            "error": self.error,
        }


@dataclass
class AgentRunResult:
    success: bool
    code: str
    stl_path: Optional[str]
    error: Optional[str]
    decision_trace: List[DecisionTraceEvent] = field(default_factory=list)
    planner_output: Dict[str, Any] = field(default_factory=dict)
    retrieved_examples: List[Dict[str, Any]] = field(default_factory=list)
    llm_traces: List[Dict[str, Any]] = field(default_factory=list)
    llm_usage_totals: Dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "code": self.code,
            "stl_path": self.stl_path,
            "error": self.error,
            "decision_trace": [event.to_dict() for event in self.decision_trace],
            "planner_output": dict(self.planner_output),
            "retrieved_examples": [dict(item) for item in self.retrieved_examples],
            "llm_traces": [dict(item) for item in self.llm_traces],
            "llm_usage_totals": dict(self.llm_usage_totals),
        }

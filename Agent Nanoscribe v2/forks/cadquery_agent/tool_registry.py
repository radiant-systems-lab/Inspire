from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .session import CadSessionState
from .types import ToolResult


@dataclass
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None
    items: Optional[Dict[str, Any]] = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: List[ToolParam]
    handler: Callable[..., ToolResult]
    category: str = "cadquery"
    aliases: List[str] = field(default_factory=list)
    mutates_state: bool = True
    lazy_params: Optional[Callable[[], List[ToolParam]]] = None

    def resolve_params(self) -> List[ToolParam]:
        if self.lazy_params is not None and not self.parameters:
            self.parameters = list(self.lazy_params())
            self.lazy_params = None
        return self.parameters


class ToolRegistry:
    def __init__(self, state: Optional[CadSessionState] = None) -> None:
        self.state = state or CadSessionState()
        self._tools: Dict[str, ToolDefinition] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        for alias in tool.aliases:
            self._aliases[alias] = tool.name

    def get(self, name: str) -> Optional[ToolDefinition]:
        canonical = self._aliases.get(name, name)
        return self._tools.get(canonical)

    def list_tools(self) -> List[ToolDefinition]:
        tools = list(self._tools.values())
        for tool in tools:
            tool.resolve_params()
        return tools

    def search_tools(self, query: str) -> List[ToolDefinition]:
        q = str(query or "").lower().strip()
        if not q:
            return self.list_tools()
        out: List[ToolDefinition] = []
        for tool in self._tools.values():
            if q in tool.name.lower() or q in tool.description.lower():
                tool.resolve_params()
                out.append(tool)
        return out

    def execute(self, name: str, params: Optional[Dict[str, Any]] = None) -> ToolResult:
        payload = dict(params or {})
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, output="", error=f"Unknown tool: {name}")

        resolved = tool.resolve_params()
        ok, errors, coerced = _validate_and_coerce(payload, resolved)
        if not ok:
            return ToolResult(success=False, output="", error="; ".join(errors))

        if tool.mutates_state:
            self.state.snapshot(f"tool:{tool.name}")
        try:
            result = tool.handler(**coerced)
        except Exception as exc:  # pragma: no cover - defensive guard
            if tool.mutates_state:
                self.state.rollback()
            return ToolResult(success=False, output="", error=f"Tool {tool.name} failed: {exc}")

        if not result.success and tool.mutates_state:
            self.state.rollback()
        elif result.success and tool.mutates_state:
            self.state.commit()
        return result

    def to_openai_schema(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            params = tool.resolve_params()
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _params_to_json_schema(params),
                    },
                }
            )
        return out

    def to_mcp_schema(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            params = tool.resolve_params()
            out.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": _params_to_json_schema(params),
                }
            )
        return out



def _validate_and_coerce(payload: Dict[str, Any], params: List[ToolParam]) -> tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    out: Dict[str, Any] = {}

    for spec in params:
        value = payload.get(spec.name, spec.default)
        if value is None and spec.required:
            errors.append(f"Missing required parameter: {spec.name}")
            continue
        if value is None:
            out[spec.name] = None
            continue
        coerced, ok = _coerce_value(value, spec)
        if not ok:
            errors.append(f"Invalid value for {spec.name}: expected {spec.type}")
            continue
        if spec.enum and coerced not in spec.enum:
            errors.append(f"Invalid enum value for {spec.name}: {coerced}")
            continue
        out[spec.name] = coerced

    return (len(errors) == 0), errors, out



def _coerce_value(value: Any, spec: ToolParam) -> tuple[Any, bool]:
    try:
        if spec.type == "string":
            return str(value), True
        if spec.type == "number":
            return float(value), True
        if spec.type == "integer":
            return int(float(value)), True
        if spec.type == "boolean":
            if isinstance(value, bool):
                return value, True
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "on"}:
                    return True, True
                if lowered in {"false", "0", "no", "off"}:
                    return False, True
            return bool(value), True
        if spec.type == "array":
            if isinstance(value, list):
                return value, True
            return [value], True
        if spec.type == "object":
            if isinstance(value, dict):
                return value, True
            return {}, False
    except Exception:
        return value, False
    return value, False



def _params_to_json_schema(params: List[ToolParam]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for param in params:
        prop: Dict[str, Any] = {"type": param.type, "description": param.description}
        if param.enum:
            prop["enum"] = list(param.enum)
        if param.default is not None:
            prop["default"] = param.default
        if param.items is not None:
            prop["items"] = dict(param.items)
        properties[param.name] = prop
        if param.required:
            required.append(param.name)
    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema

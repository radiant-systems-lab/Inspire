from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..tool_registry import ToolDefinition, ToolParam, ToolRegistry
from ..types import ToolResult


class ExternalMCPClient(Protocol):
    name: str

    def list_tools(self) -> List[Dict[str, Any]]:
        ...

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        ...

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class MCPToolInfo:
    server: str
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]]


class MCPManager:
    def __init__(self) -> None:
        self._clients: Dict[str, ExternalMCPClient] = {}

    def register_client(self, client: ExternalMCPClient) -> None:
        self._clients[client.name] = client

    def connected_servers(self) -> List[str]:
        return sorted(self._clients.keys())

    def register_tools_into(self, registry: ToolRegistry) -> None:
        for server_name, client in self._clients.items():
            for item in client.list_tools():
                tool_name = str(item.get("name") or "")
                if not tool_name:
                    continue
                description = str(item.get("description") or "")
                schema = item.get("inputSchema")

                if isinstance(schema, dict):
                    params = _json_schema_to_params(schema)
                    lazy = None
                else:
                    params = []
                    lazy = _build_lazy_params(client, tool_name)

                handler = _build_handler(client, tool_name)
                registry.register(
                    ToolDefinition(
                        name=f"{server_name}__{tool_name}",
                        description=f"[{server_name}] {description}",
                        parameters=params,
                        handler=handler,
                        category="mcp",
                        lazy_params=lazy,
                        mutates_state=False,
                    )
                )

    def search_tools(self, query: str) -> List[MCPToolInfo]:
        out: List[MCPToolInfo] = []
        q = str(query or "").lower()
        for server_name, client in self._clients.items():
            for item in client.list_tools():
                name = str(item.get("name") or "")
                desc = str(item.get("description") or "")
                if q in name.lower() or q in desc.lower():
                    out.append(
                        MCPToolInfo(
                            server=server_name,
                            name=name,
                            description=desc,
                            input_schema=item.get("inputSchema") if isinstance(item.get("inputSchema"), dict) else None,
                        )
                    )
        return out



def _build_lazy_params(client: ExternalMCPClient, tool_name: str) -> Callable[[], List[ToolParam]]:
    def loader() -> List[ToolParam]:
        schema = client.get_tool_schema(tool_name)
        return _json_schema_to_params(schema)

    return loader



def _build_handler(client: ExternalMCPClient, tool_name: str):
    def handler(**kwargs: Any) -> ToolResult:
        response = client.call_tool(tool_name, kwargs)
        content = response.get("content")
        is_error = bool(response.get("isError"))
        if is_error:
            if isinstance(content, list) and content:
                first = content[0]
                message = str(first.get("text") if isinstance(first, dict) else first)
            else:
                message = str(response.get("error") or "MCP tool call failed")
            return ToolResult(success=False, output="", error=message)

        text_parts: List[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(str(item.get("text") or ""))
                else:
                    text_parts.append(str(item))
        output = "\n".join(x for x in text_parts if x)
        return ToolResult(success=True, output=output, data={"raw": response})

    return handler



def _json_schema_to_params(schema: Dict[str, Any]) -> List[ToolParam]:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return []
    required = set(schema.get("required") or [])
    params: List[ToolParam] = []
    for name, prop in (schema.get("properties") or {}).items():
        if not isinstance(prop, dict):
            continue
        params.append(
            ToolParam(
                name=str(name),
                type=str(prop.get("type") or "string"),
                description=str(prop.get("description") or ""),
                required=str(name) in required,
                enum=list(prop.get("enum") or []) or None,
                default=prop.get("default"),
                items=prop.get("items") if isinstance(prop.get("items"), dict) else None,
            )
        )
    return params

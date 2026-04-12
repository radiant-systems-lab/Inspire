from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List, Optional


class StdioMCPClient:
    """Minimal stdio JSON-RPC MCP client for external tool servers."""

    def __init__(self, name: str, command: List[str]) -> None:
        self.name = name
        self.command = list(command)
        self.proc: Optional[subprocess.Popen[str]] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    def connect(self) -> None:
        if self.proc is not None:
            return
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._rpc("initialize", {})
        self._notify("notifications/initialized", {})

    def disconnect(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        self.proc = None

    def list_tools(self) -> List[Dict[str, Any]]:
        if self._tools_cache is not None:
            return list(self._tools_cache)
        response = self._rpc("tools/list", {})
        tools = response.get("tools") if isinstance(response, dict) else []
        if not isinstance(tools, list):
            tools = []
        self._tools_cache = [dict(item) for item in tools if isinstance(item, dict)]
        return list(self._tools_cache)

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        for tool in self.list_tools():
            if str(tool.get("name") or "") == tool_name:
                schema = tool.get("inputSchema")
                if isinstance(schema, dict):
                    return schema
        return {}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._rpc("tools/call", {"name": tool_name, "arguments": dict(arguments)})
        if isinstance(payload, dict):
            return payload
        return {"content": [{"type": "text", "text": "invalid MCP response"}], "isError": True}

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=True) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method: str, params: Dict[str, Any]) -> Any:
        if self.proc is None:
            self.connect()
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("MCP process is not available")

        msg_id = 1
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=True) + "\n")
        self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed stdout")
            text = line.strip()
            if not text:
                continue
            try:
                response = json.loads(text)
            except json.JSONDecodeError:
                continue
            if response.get("id") != msg_id:
                continue
            if response.get("error"):
                return {"content": [{"type": "text", "text": str(response.get('error'))}], "isError": True}
            return response.get("result")

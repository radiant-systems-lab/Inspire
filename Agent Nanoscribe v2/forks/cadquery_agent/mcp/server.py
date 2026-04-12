from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

from ..tool_registry import ToolRegistry
from . import protocol

SERVER_INFO = {"name": "Prompt2CAD CadQuery Agent", "version": "0.1.0"}
PROTOCOL_VERSION = "2025-03-26"


class MCPServer:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = str(msg.get("method") or "")
        msg_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            return protocol.make_response(
                msg_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            )

        if method == "notifications/initialized":
            return None

        if method == "tools/list":
            return protocol.make_response(msg_id, {"tools": self.registry.to_mcp_schema()})

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            result = self.registry.execute(str(name or ""), dict(arguments))
            if result.success:
                payload = [{"type": "text", "text": result.output}]
                if result.data:
                    payload.append({"type": "text", "text": json.dumps(result.data, ensure_ascii=True)})
                return protocol.make_response(msg_id, {"content": payload, "isError": False})
            return protocol.make_response(
                msg_id,
                {"content": [{"type": "text", "text": result.error or "unknown error"}], "isError": True},
            )

        if method == "ping":
            return protocol.make_response(msg_id, {})

        if msg_id is not None:
            return protocol.make_error(msg_id, protocol.METHOD_NOT_FOUND, f"Method not found: {method}")
        return None

    def run_stdio(self) -> None:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                err = protocol.make_error(None, protocol.PARSE_ERROR, "invalid JSON")
                sys.stdout.write(json.dumps(err, ensure_ascii=True) + "\n")
                sys.stdout.flush()
                continue

            response = self.handle(msg)
            if response is None:
                continue
            sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
            sys.stdout.flush()

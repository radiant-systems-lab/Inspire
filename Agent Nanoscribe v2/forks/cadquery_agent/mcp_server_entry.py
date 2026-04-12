#!/usr/bin/env python3
from __future__ import annotations

from .mcp.server import MCPServer
from .session import CadSessionState
from .tools import create_default_registry


def main() -> int:
    state = CadSessionState()
    registry = create_default_registry(state)
    server = MCPServer(registry)
    server.run_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

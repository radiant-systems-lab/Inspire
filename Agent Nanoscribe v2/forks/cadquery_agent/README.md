# CadQuery Agent Fork

This folder contains a forked architecture replacement inspired by `freecad-ai`, adapted for CadQuery-only execution.

## Main Runtime

- `runtime.py`: `run_agent(request, mode='act', ...)`
- `tools.py`: structured CadQuery tool surface (core 16 + `execute_code` fallback)
- `tool_registry.py`: schema-aware tool registry, aliases, coercion, rollback-aware execution
- `session.py`: persistent CAD session state and history snapshots
- `executor.py`: static validation + subprocess preflight + bounded execution wrapper

## MCP

- `mcp/server.py`: MCP server (`initialize`, `tools/list`, `tools/call`, `ping`)
- `mcp/manager.py`: optional external MCP tool manager with namespaced registration and deferred schemas
- `mcp_server_entry.py`: stdio server launcher

## Compatibility

`prompt2cad.pipeline.run_pipeline()` now supports:

- `agent_mode='act'` (default): uses this runtime
- `tool_mode='auto'|'off'`
- `tools_enabled`, `mcp_enabled`

Legacy path remains available by calling with `agent_mode='legacy'`.

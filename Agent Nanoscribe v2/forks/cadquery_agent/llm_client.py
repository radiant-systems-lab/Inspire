from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from prompt2cad.utils import call_openrouter, get_last_openrouter_trace

from .types import ToolCall


@dataclass
class LLMResponse:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    trace: Dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Provider-agnostic wrapper currently backed by OpenRouter chat completion."""

    def __init__(self, model: str, *, temperature: float = 0.1) -> None:
        self.model = model
        self.temperature = float(temperature)

    def send(self, messages: List[Dict[str, Any]], *, system: str = "") -> LLMResponse:
        payload = list(messages)
        if system:
            payload = [{"role": "system", "content": system}] + payload
        text = call_openrouter(payload, model=self.model, temperature=self.temperature)
        trace = get_last_openrouter_trace()
        return LLMResponse(text=text, trace=trace)

    def stream(self, messages: List[Dict[str, Any]], *, system: str = "") -> Generator[str, None, None]:
        # OpenRouter helper in this repo is non-streaming. We preserve the interface.
        response = self.send(messages, system=system)
        yield response.text

    def send_with_tools(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        tool_instructions = _tool_instruction_block(tools or [])
        merged_system = (system.strip() + "\n\n" + tool_instructions).strip()
        payload = list(messages)
        if merged_system:
            payload = [{"role": "system", "content": merged_system}] + payload

        raw = call_openrouter(
            payload,
            model=self.model,
            temperature=0.0,
            json_mode=True,
            extra_body={"top_p": 1},
        )
        trace = get_last_openrouter_trace()
        return _parse_tool_response(raw, trace=trace)



def _tool_instruction_block(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return "Return JSON with {\"text\": \"...\", \"tool_calls\": []}."
    compact_tools = []
    for item in tools:
        fn = ((item or {}).get("function") or {})
        compact_tools.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            }
        )
    return (
        "You may call tools. Return JSON only with this schema:\n"
        "{\"text\": string, \"tool_calls\": [{\"id\": string, \"name\": string, \"arguments\": object}]}\n"
        "Use only tool names from this list:\n"
        + json.dumps(compact_tools, ensure_ascii=True)
    )



def _parse_tool_response(raw: str, *, trace: Optional[Dict[str, Any]] = None) -> LLMResponse:
    text = str(raw or "").strip()
    trace_payload = dict(trace or {})

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn", trace=trace_payload)

    if isinstance(parsed, dict):
        content = str(parsed.get("text") or parsed.get("response") or "")
        calls: List[ToolCall] = []
        for idx, item in enumerate(parsed.get("tool_calls") or [], start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            args = item.get("arguments")
            if not isinstance(args, dict):
                args = {}
            call_id = str(item.get("id") or f"call_{idx}")
            calls.append(ToolCall(id=call_id, name=name, arguments=args))

        stop_reason = "tool_use" if calls else "end_turn"
        return LLMResponse(
            text=content,
            tool_calls=calls,
            stop_reason=stop_reason,
            trace=trace_payload,
        )

    return LLMResponse(text=text, tool_calls=[], stop_reason="end_turn", trace=trace_payload)

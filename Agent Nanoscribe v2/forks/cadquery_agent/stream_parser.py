from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


KNOWN_TYPES = {
    "text_delta",
    "thinking_delta",
    "tool_call_start",
    "tool_call_delta",
    "tool_call_end",
    "done",
}


def parse_stream_events(chunks: Iterable[str]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk or "").strip()
        if not text:
            continue
        if text.startswith("data:"):
            text = text[5:].strip()
        if text == "[DONE]":
            events.append({"type": "done"})
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            events.append({"type": "text_delta", "text": text})
            continue

        event_type = payload.get("type")
        if event_type in KNOWN_TYPES:
            events.append(payload)
            continue

        # OpenAI-like chunk fallback
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            delta = (choices[0] or {}).get("delta") or {}
            if delta.get("content"):
                events.append({"type": "text_delta", "text": str(delta.get("content"))})
            if delta.get("tool_calls"):
                for tc in delta.get("tool_calls") or []:
                    events.append({"type": "tool_call_delta", "tool_call": tc})
            finish = (choices[0] or {}).get("finish_reason")
            if finish:
                events.append({"type": "done", "finish_reason": finish})
            continue

        if payload.get("text"):
            events.append({"type": "text_delta", "text": str(payload.get("text"))})

    if not events or events[-1].get("type") != "done":
        events.append({"type": "done"})
    return events

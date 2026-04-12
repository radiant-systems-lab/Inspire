from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Conversation:
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": str(content)})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": str(content)})

    def add_tool(self, name: str, content: str, *, call_id: str = "") -> None:
        payload = {"role": "tool", "content": str(content), "name": str(name)}
        if call_id:
            payload["tool_call_id"] = str(call_id)
        self.messages.append(payload)

    def compact(self, keep_last: int = 12) -> None:
        if len(self.messages) <= keep_last:
            return
        head = self.messages[:-keep_last]
        tail = self.messages[-keep_last:]
        summary = _summarize(head)
        self.messages = [{"role": "system", "content": summary}] + tail



def _summarize(messages: List[Dict[str, Any]]) -> str:
    bullet_lines: List[str] = []
    for msg in messages[-20:]:
        role = str(msg.get("role") or "unknown")
        content = str(msg.get("content") or "")
        content = " ".join(content.split())
        if len(content) > 120:
            content = content[:117] + "..."
        bullet_lines.append(f"- {role}: {content}")
    if not bullet_lines:
        return "Conversation summary unavailable."
    return "Conversation summary:\n" + "\n".join(bullet_lines)

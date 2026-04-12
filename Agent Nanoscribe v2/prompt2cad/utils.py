"""
Shared utilities: OpenRouter API calls with retry / back-off.
Includes tool-calling support for the agent loop.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Dict, List, Optional

import requests

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

_LAST_OPENROUTER_TRACE: Dict[str, Any] = {}


def get_last_openrouter_trace() -> Dict[str, Any]:
    """Return metadata for the most recent OpenRouter request in this process."""
    return copy.deepcopy(_LAST_OPENROUTER_TRACE)


def call_openrouter(
    messages: List[dict],
    model: str,
    temperature: float = 0.1,
    json_mode: bool = False,
    max_retries: int = 3,
    timeout: int = 90,
    api_key: str | None = None,
    extra_body: dict | None = None,
) -> str:
    """
    POST to the OpenRouter chat completions endpoint.

    Args:
        messages:    List of {"role": ..., "content": ...} dicts.
        model:       OpenRouter model slug, e.g. "openai/gpt-4o-mini".
        temperature: Sampling temperature (lower = more deterministic).
        json_mode:   Request JSON-formatted output (not all models support it).
        max_retries: Number of retry attempts on transient errors.
        timeout:     Per-request timeout in seconds.
        api_key:     Override the key from config (useful for testing).

    Returns:
        The assistant response string.

    Raises:
        ValueError:   If no API key is configured.
        RuntimeError: If all retry attempts fail.
    """
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is not configured.\n"
            "  Option 1: add it to a .env file at the project root\n"
            "  Option 2: export OPENROUTER_API_KEY=<your-key>"
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://prompt2cad.local",
        "X-Title": "Prompt2CAD",
    }

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if extra_body:
        body.update(extra_body)

    global _LAST_OPENROUTER_TRACE

    started = time.perf_counter()
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            _LAST_OPENROUTER_TRACE = {
                "ok": True,
                "model": model,
                "json_mode": bool(json_mode),
                "attempts": attempt,
                "latency_seconds": round(time.perf_counter() - started, 4),
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                },
                "id": data.get("id"),
            }
            return data["choices"][0]["message"]["content"]

        except requests.HTTPError as exc:
            last_err = exc
            status = exc.response.status_code if exc.response is not None else 0
            if status in (429, 503):
                wait = 2 ** attempt
                print(f"  [OpenRouter] {status} - retrying in {wait}s (attempt {attempt}/{max_retries}) ...")
                time.sleep(wait)
            else:
                raise  # non-retriable HTTP error

        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = exc
            wait = 2 ** attempt
            print(f"  [OpenRouter] network error (attempt {attempt}/{max_retries}): {exc} - retrying in {wait}s ...")
            time.sleep(wait)

    _LAST_OPENROUTER_TRACE = {
        "ok": False,
        "model": model,
        "json_mode": bool(json_mode),
        "attempts": max_retries,
        "latency_seconds": round(time.perf_counter() - started, 4),
        "error": str(last_err),
    }
    raise RuntimeError(
        f"OpenRouter call failed after {max_retries} retries. Last error: {last_err}"
    )


def call_openrouter_agent(
    messages: List[dict],
    model: str,
    tools: Optional[List[dict]] = None,
    temperature: float = 0.1,
    max_retries: int = 3,
    timeout: int = 120,
    api_key: str | None = None,
) -> Dict[str, Any]:
    """
    OpenRouter call that supports tool/function calling.

    Returns the full message dict from choices[0]["message"], which may contain:
      - "content": str  (when the model responds in text)
      - "tool_calls": list  (when the model calls a tool)
      - "finish_reason": "tool_calls" | "stop" | "end_turn"

    Args:
        tools: List of tool schemas in OpenAI format:
               [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]
    """
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://prompt2cad.local",
        "X-Title": "Prompt2CAD-Agent",
    }

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    global _LAST_OPENROUTER_TRACE
    started = time.perf_counter()
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason", "")

            _LAST_OPENROUTER_TRACE = {
                "ok": True,
                "model": model,
                "tool_calling": bool(tools),
                "finish_reason": finish_reason,
                "attempts": attempt,
                "latency_seconds": round(time.perf_counter() - started, 4),
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                },
            }
            return {
                "role": message.get("role", "assistant"),
                "content": message.get("content") or "",
                "tool_calls": message.get("tool_calls") or [],
                "finish_reason": finish_reason,
            }

        except requests.HTTPError as exc:
            last_err = exc
            status = exc.response.status_code if exc.response is not None else 0
            if status in (429, 503):
                wait = 2 ** attempt
                print(f"  [OpenRouter] {status} - retrying in {wait}s (attempt {attempt}/{max_retries}) ...")
                time.sleep(wait)
            else:
                raise

        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = exc
            wait = 2 ** attempt
            print(f"  [OpenRouter] network error (attempt {attempt}/{max_retries}): {exc} - retrying in {wait}s ...")
            time.sleep(wait)

    raise RuntimeError(
        f"OpenRouter agent call failed after {max_retries} retries. Last error: {last_err}"
    )


def tool_result_message(tool_call_id: str, content: Any) -> dict:
    """Build an OpenAI-format tool result message."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content,
    }


def assistant_tool_call_message(tool_calls: List[dict], content: str = "") -> dict:
    """Build an assistant message that contains tool calls (for conversation history)."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }

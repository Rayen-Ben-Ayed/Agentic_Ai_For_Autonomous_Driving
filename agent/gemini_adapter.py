"""Convert OpenAI-style chat messages/tools to Gemini generateContent requests."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    type: str
    function: _Function


@dataclass
class GeminiChatMessage:
    """OpenAI-compatible assistant message returned to the agent loop."""

    content: str | None
    tool_calls: list[_ToolCall] | None = None


def openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declarations = []
    for tool in tools:
        fn = tool["function"]
        declarations.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return [{"functionDeclarations": declarations}]


def messages_to_gemini(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_instruction: dict[str, Any] | None = None
    contents: list[dict[str, Any]] = []
    tool_id_to_name: dict[str, str] = {}

    idx = 0
    while idx < len(messages):
        msg = messages[idx]
        role = msg["role"]

        if role == "system":
            system_instruction = {"parts": [{"text": msg["content"]}]}
        elif role == "user":
            contents.append(
                {"role": "user", "parts": [{"text": msg["content"]}]},
            )
        elif role == "assistant":
            parts: list[dict[str, Any]] = []
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                tc_id = tc["id"]
                name = tc["function"]["name"]
                tool_id_to_name[tc_id] = name
                args = json.loads(tc["function"].get("arguments") or "{}")
                parts.append({"functionCall": {"name": name, "args": args}})
            contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            parts = []
            while idx < len(messages) and messages[idx]["role"] == "tool":
                tmsg = messages[idx]
                name = tool_id_to_name.get(tmsg["tool_call_id"], "unknown")
                try:
                    response = json.loads(tmsg["content"])
                except json.JSONDecodeError:
                    response = {"result": tmsg["content"]}
                parts.append(
                    {
                        "functionResponse": {
                            "name": name,
                            "response": response,
                        }
                    }
                )
                idx += 1
            contents.append({"role": "user", "parts": parts})
            continue

        idx += 1

    return system_instruction, contents


def parse_gemini_response(data: dict[str, Any]) -> GeminiChatMessage:
    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {data}")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text_parts: list[str] = []
    tool_calls: list[_ToolCall] = []

    for part in parts:
        # Gemma 4 may return internal reasoning in thought parts; skip those.
        if part.get("thought"):
            continue
        if "text" in part:
            text_parts.append(part["text"])
        if "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append(
                _ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=_Function(
                        name=fc["name"],
                        arguments=json.dumps(fc.get("args") or {}),
                    ),
                )
            )

    return GeminiChatMessage(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls or None,
    )

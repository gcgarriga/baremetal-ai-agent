"""Small SSE parser and OpenAI chat-completion stream assembler."""

import json
from collections.abc import Iterable, Iterator
from typing import Any


class StreamingError(RuntimeError):
    """Raised when a streaming response cannot be parsed."""


def _data_value(line: str) -> str:
    value = line[5:]
    if value.startswith(" "):
        value = value[1:]
    return value


def iter_sse_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield JSON payloads from Server-Sent Events lines.

    Only ``data:`` fields are meaningful for OpenAI-compatible chat streams.
    Blank lines dispatch an event, comments are ignored, and ``data: [DONE]``
    terminates the stream.
    """
    data_lines: list[str] = []

    def parse_event(payload: str) -> dict[str, Any] | None:
        if payload == "[DONE]":
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StreamingError(f"Invalid JSON in streaming response: {exc}") from exc
        if not isinstance(parsed, dict):
            raise StreamingError(f"Invalid streaming response payload: expected object, got {type(parsed).__name__}")
        return parsed

    for raw in lines:
        if raw == "":
            if not data_lines:
                continue
            payload = "\n".join(data_lines)
            data_lines.clear()
            event = parse_event(payload)
            if event is None:
                return
            yield event
            continue

        for line in raw.splitlines():
            line = line.rstrip("\r")
            if not line:
                if not data_lines:
                    continue
                payload = "\n".join(data_lines)
                data_lines.clear()
                event = parse_event(payload)
                if event is None:
                    return
                yield event
                continue

            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                if data_lines:
                    event = parse_event("\n".join(data_lines))
                    data_lines.clear()
                    if event is None:
                        return
                    yield event
                data_lines.append(_data_value(line))

    if data_lines:
        event = parse_event("\n".join(data_lines))
        if event is not None:
            yield event


def parse_sse_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Parse SSE lines into JSON chunks."""
    return list(iter_sse_events(lines))


def _choice_delta(chunk: dict[str, Any]) -> dict[str, Any]:
    choices = chunk.get("choices")
    if not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    delta = first.get("delta", {})
    return delta if isinstance(delta, dict) else {}


def assemble_chat_message(chunks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Assemble streaming chunks into a final assistant message."""
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        delta = _choice_delta(chunk)
        content = delta.get("content")
        if content is not None:
            content_parts.append(str(content))

        for tool_delta in delta.get("tool_calls") or []:
            if not isinstance(tool_delta, dict):
                continue
            index = tool_delta.get("index")
            if not isinstance(index, int):
                raise StreamingError("Invalid streaming tool_call delta: missing integer index")

            assembled = tool_calls_by_index.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if tool_delta.get("id") is not None:
                assembled["id"] = str(tool_delta["id"])
            if tool_delta.get("type") is not None:
                assembled["type"] = str(tool_delta["type"])

            function_delta = tool_delta.get("function")
            if isinstance(function_delta, dict):
                function = assembled["function"]
                if function_delta.get("name") is not None:
                    function["name"] += str(function_delta["name"])
                if function_delta.get("arguments") is not None:
                    function["arguments"] += str(function_delta["arguments"])

    content_text = "".join(content_parts)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content_text if content_text or not tool_calls_by_index else None,
    }
    if tool_calls_by_index:
        message["tool_calls"] = [tool_calls_by_index[index] for index in sorted(tool_calls_by_index)]
    return message


def assemble_response(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap assembled stream chunks in a chat-completion response shape."""
    message = assemble_chat_message(chunks)
    response: dict[str, Any] = {
        "choices": [
            {
                "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
        "usage": {},
    }
    for chunk in chunks:
        if chunk.get("id") is not None:
            response["id"] = chunk["id"]
        if chunk.get("created") is not None:
            response["created"] = chunk["created"]
        if chunk.get("model") is not None:
            response["model"] = chunk["model"]
        if isinstance(chunk.get("usage"), dict):
            response["usage"] = chunk["usage"]
    return response

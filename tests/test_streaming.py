"""Tests for streaming.py — SSE parsing and tool-call assembly."""

import pytest

from baremetal_agent import safety, streaming


def _line_events(*payloads: str) -> list[str]:
    return [f"data: {payload}\n\n" for payload in payloads]


def test_content_deltas_assemble_to_final_content():
    chunks = streaming.parse_sse_events(
        _line_events(
            '{"choices":[{"delta":{"content":"hel"}}]}',
            '{"choices":[{"delta":{"content":"lo"}}]}',
            "[DONE]",
        )
    )

    message = streaming.assemble_chat_message(chunks)

    assert message == {"role": "assistant", "content": "hello"}


def test_blank_sse_lines_are_ignored():
    chunks = streaming.parse_sse_events(
        [
            "\n",
            "",
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
            "\n",
            "data: [DONE]\n",
        ]
    )

    message = streaming.assemble_chat_message(chunks)

    assert message["content"] == "ok"


def test_httpx_style_blank_line_dispatches_without_waiting_for_next_event():
    class StopsIfReadAgain:
        def __init__(self) -> None:
            self._lines = iter(['data: {"choices":[{"delta":{"content":"now"}}]}', ""])

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self._lines)
            except StopIteration as exc:
                raise AssertionError("parser waited past the blank line") from exc

    events = streaming.iter_sse_events(StopsIfReadAgain())

    assert next(events)["choices"][0]["delta"]["content"] == "now"


def test_done_terminates_stream_cleanly():
    chunks = streaming.parse_sse_events(
        _line_events(
            '{"choices":[{"delta":{"content":"before"}}]}',
            "[DONE]",
            '{"choices":[{"delta":{"content":"after"}}]}',
        )
    )

    assert len(chunks) == 1
    assert streaming.assemble_chat_message(chunks)["content"] == "before"


def test_malformed_sse_json_raises_clear_stream_error():
    with pytest.raises(streaming.StreamingError, match="Invalid JSON in streaming response"):
        streaming.parse_sse_events(["data: {not-json}\n\n"])


def test_single_tool_call_fragments_assemble_in_arrival_order():
    chunks = streaming.parse_sse_events(
        _line_events(
            '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function"}]}}]}',
            '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file"}}]}}]}',
            '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"pa"}}]}}]}',
            '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\\":\\"README.md\\"}"}}]}}]}',
            "[DONE]",
        )
    )

    message = streaming.assemble_chat_message(chunks)

    assert message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
            }
        ],
    }


def test_multiple_tool_calls_assemble_independently_by_index():
    chunks = streaming.parse_sse_events(
        _line_events(
            '{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"search_code","arguments":"{\\"pattern\\":"}},{"index":0,"id":"call_a","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":"}}]}}]}',
            '{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"README.md\\"}"}},{"index":1,"function":{"arguments":"\\"agent\\"}"}}]}}]}',
            "[DONE]",
        )
    )

    message = streaming.assemble_chat_message(chunks)

    assert message["tool_calls"] == [
        {
            "id": "call_a",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
        },
        {
            "id": "call_b",
            "type": "function",
            "function": {"name": "search_code", "arguments": '{"pattern":"agent"}'},
        },
    ]


def test_final_message_includes_content_and_tool_calls_when_both_streamed():
    payload = (
        '{"choices":[{"delta":{"content":"I will inspect that.","tool_calls":['
        '{"index":0,"id":"call_1","type":"function",'
        '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
        "]}}]}"
    )
    chunks = streaming.parse_sse_events(
        _line_events(
            payload,
            "[DONE]",
        )
    )

    message = streaming.assemble_chat_message(chunks)

    assert message == {
        "role": "assistant",
        "content": "I will inspect that.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
            }
        ],
    }


def test_stream_errors_can_be_redacted_by_shared_redactor():
    raw_error = streaming.StreamingError("Invalid JSON in streaming response: token=abcdef123456")

    redacted = safety.redact_secrets(str(raw_error))

    assert "abcdef123456" not in redacted
    assert "token=abcdef******" in redacted

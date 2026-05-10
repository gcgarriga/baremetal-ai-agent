"""Raw HTTP client for the GitHub Models API (OpenAI-compatible chat completions)."""

import atexit
import json
import sys
import time
from collections.abc import Callable

import httpx

from baremetal_agent import safety, streaming
from baremetal_agent.config import AgentConfig, require_token

_client = httpx.Client(timeout=120.0)
atexit.register(_client.close)

_BOX_TOP = "╭─ {} ─{}"
_BOX_BOT = "╰" + "─" * 60 + "╯"


def _log_box(title: str, body: str) -> None:
    """Print a payload inside a box-drawing frame."""
    pad = "─" * max(0, 58 - len(title))
    print(_BOX_TOP.format(title, pad + "╮"))
    for line in body.splitlines():
        print(f"│ {line}")
    print(_BOX_BOT)
    print()


def chat_completion(messages: list[dict], tools: list[dict], *, cfg: AgentConfig) -> dict:
    """Send a chat completion request and return the parsed response JSON.

    Logs the full request and response payloads to stdout when `cfg.log_payloads`.
    Retries on 429 (rate limit) and 5xx (server error) up to 3 times.
    """
    body = {
        "model": cfg.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }

    token = require_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    if cfg.log_payloads:
        _log_box("API Request", safety.redact_secrets(f"POST {cfg.api_url}\n{json.dumps(body, indent=2)}"))

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            resp = _client.post(cfg.api_url, headers=headers, json=body)
        except httpx.RequestError as exc:
            if attempt < max_retries:
                wait = 2**attempt
                print(f"│ Connection error: {exc}. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Connection failed after {max_retries} retries: {exc}") from exc

        if resp.status_code == 200:
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid JSON in 200 response: {exc}\nBody: {resp.text[:500]}") from exc
            if cfg.log_payloads:
                _log_box("API Response", safety.redact_secrets(f"{resp.status_code} OK\n{json.dumps(data, indent=2)}"))
            return data

        if resp.status_code == 401:
            raise RuntimeError(f"Authentication failed (401). Check your GITHUB_TOKEN.\nResponse: {resp.text}")

        if resp.status_code == 429:
            try:
                retry_after = int(resp.headers.get("Retry-After", "5"))
            except (ValueError, TypeError):
                retry_after = 5
            if attempt < max_retries:
                print(f"│ Rate limited (429). Waiting {retry_after}s...", file=sys.stderr)
                time.sleep(retry_after)
                continue
            raise RuntimeError(f"Rate limited after {max_retries} retries. Response: {resp.text}")

        if resp.status_code >= 500:
            if attempt < max_retries:
                wait = 2**attempt
                print(f"│ Server error ({resp.status_code}). Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Server error after {max_retries} retries.\nStatus: {resp.status_code}\nResponse: {resp.text}"
            )

        # Other client errors — don't retry
        raise RuntimeError(f"API error {resp.status_code}.\nResponse: {resp.text}")

    # Should not reach here, but just in case
    raise RuntimeError("Exhausted retries without a response.")


def chat_completion_stream(
    messages: list[dict],
    tools: list[dict],
    *,
    cfg: AgentConfig,
    on_stream_delta: Callable[[str], None] | None = None,
) -> dict:
    """Send a streaming chat completion request and return an assembled response.

    Retries only before a successful streaming response begins. Once a 200 stream
    starts, parse/connection errors are surfaced instead of replaying partial output.
    """
    body = {
        "model": cfg.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    token = require_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    if cfg.log_payloads:
        _log_box(
            "API Request",
            safety.redact_secrets(f"POST {cfg.api_url}\n{json.dumps(body, indent=2)}"),
        )

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            with _client.stream("POST", cfg.api_url, headers=headers, json=body) as resp:
                status_code = resp.status_code
                response_headers = resp.headers
                if resp.status_code == 200:
                    chunks: list[dict] = []
                    try:
                        for chunk in streaming.iter_sse_events(resp.iter_lines()):
                            chunks.append(chunk)
                            if on_stream_delta is not None:
                                choices = chunk.get("choices") or []
                                if choices and isinstance(choices[0], dict):
                                    delta = choices[0].get("delta", {})
                                    if isinstance(delta, dict) and delta.get("content") is not None:
                                        on_stream_delta(str(delta["content"]))
                    except (httpx.RequestError, streaming.StreamingError) as exc:
                        raise RuntimeError(f"Streaming response failed: {exc}") from exc

                    data = streaming.assemble_response(chunks)
                    if cfg.log_payloads:
                        _log_box(
                            "API Response",
                            safety.redact_secrets(f"{resp.status_code} OK\n{json.dumps(data, indent=2)}"),
                        )
                    return data

                response_text = resp.read().decode("utf-8", errors="replace")

        except httpx.RequestError as exc:
            if attempt < max_retries:
                wait = 2**attempt
                print(f"│ Connection error: {exc}. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Connection failed after {max_retries} retries: {exc}") from exc

        if status_code == 401:
            raise RuntimeError(f"Authentication failed (401). Check your GITHUB_TOKEN.\nResponse: {response_text}")

        if status_code == 429:
            try:
                retry_after = int(response_headers.get("Retry-After", "5"))
            except (ValueError, TypeError):
                retry_after = 5
            if attempt < max_retries:
                print(f"│ Rate limited (429). Waiting {retry_after}s...", file=sys.stderr)
                time.sleep(retry_after)
                continue
            raise RuntimeError(f"Rate limited after {max_retries} retries. Response: {response_text}")

        if status_code >= 500:
            if attempt < max_retries:
                wait = 2**attempt
                print(f"│ Server error ({status_code}). Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Server error after {max_retries} retries.\nStatus: {status_code}\nResponse: {response_text}"
            )

        raise RuntimeError(f"API error {status_code}.\nResponse: {response_text}")

    raise RuntimeError("Exhausted retries without a response.")

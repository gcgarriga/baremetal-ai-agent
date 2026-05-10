"""Configuration loaded from environment variables (with .env fallback).

Configuration is represented as a frozen `AgentConfig` dataclass produced by
`load_config()`. Treat it as immutable; produce derived configs with
`dataclasses.replace`. Default constants below are exported so callers (CLI,
tests, docs) can reference the same defaults that env-var fallback uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_API_URL: str = "https://models.github.ai/inference/chat/completions"
DEFAULT_MODEL: str = "openai/gpt-4.1"
DEFAULT_MAX_ITERATIONS: int = 10
DEFAULT_RENDER_VERBOSE: bool = False
DEFAULT_LOG_PAYLOADS: bool = False
DEFAULT_STREAM: bool = False

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a developer assistant with access to tools for working with the local "
    "filesystem and git repositories. When the user asks about files, code, or git "
    "history, use the available tools to get real information — never guess or "
    "fabricate file contents, git output, or command results.\n\n"
    "You can chain multiple tool calls when needed. For example, to understand a "
    "codebase you might: list_directory → read_file on interesting files → search_code "
    "for specific patterns.\n\n"
    "Always explain what you found after using tools. Be concise and direct."
)


def _load_dotenv() -> None:
    """Load key=value pairs from .env file into os.environ."""
    env_path = Path(os.environ.get("AGENT_DOTENV", Path.cwd() / ".env"))
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def require_token() -> str:
    """Return the GitHub token required for model calls.

    Reads `GITHUB_TOKEN` from the environment at call time. Token is
    intentionally not part of `AgentConfig` so it is never accidentally logged
    or serialized.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable is required to make model calls.")
    return token


def _truthy(value: str | None) -> bool:
    return (value or "").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class AgentConfig:
    """Immutable runtime configuration for the agent.

    Construct via `load_config()` (env-driven) or directly. Produce derived
    configs with `dataclasses.replace`. The token is intentionally excluded —
    see `require_token()`.
    """

    model: str
    api_url: str
    max_iterations: int
    working_dir: Path
    render_verbose: bool
    log_payloads: bool
    stream: bool
    system_prompt: str


def load_config() -> AgentConfig:
    """Load configuration from environment variables (with .env fallback).

    Raises `ValueError` if `AGENT_MAX_ITERATIONS` is set to a non-integer.
    """
    _load_dotenv()
    raw_max_iter = os.environ.get("AGENT_MAX_ITERATIONS")
    if raw_max_iter is None:
        max_iterations = DEFAULT_MAX_ITERATIONS
    else:
        try:
            max_iterations = int(raw_max_iter)
        except ValueError as exc:
            raise ValueError(f"AGENT_MAX_ITERATIONS must be an integer, got {raw_max_iter!r}.") from exc

    # `AGENT_VERBOSE=1` is the legacy one-knob flag: it sets both render
    # verbosity and payload logging. The granular env vars
    # `AGENT_RENDER_VERBOSE` and `AGENT_LOG_PAYLOADS` override it when set,
    # so callers can opt into raw payload logging without losing the rich UI
    # (or vice versa).
    legacy_verbose = _truthy(os.environ.get("AGENT_VERBOSE"))
    raw_render = os.environ.get("AGENT_RENDER_VERBOSE")
    raw_log = os.environ.get("AGENT_LOG_PAYLOADS")
    render_verbose = _truthy(raw_render) if raw_render is not None else legacy_verbose
    log_payloads = _truthy(raw_log) if raw_log is not None else legacy_verbose

    return AgentConfig(
        model=os.environ.get("AGENT_MODEL", DEFAULT_MODEL),
        api_url=os.environ.get("AGENT_API_URL", DEFAULT_API_URL),
        max_iterations=max_iterations,
        working_dir=Path(os.environ.get("AGENT_WORKING_DIR", ".")).resolve(),
        render_verbose=render_verbose,
        log_payloads=log_payloads,
        stream=_truthy(os.environ.get("AGENT_STREAM")),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

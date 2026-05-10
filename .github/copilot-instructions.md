# Copilot Instructions

## Commands

| Task    | Command                        |
|---------|--------------------------------|
| Run     | `python -m baremetal_agent`    |
| Install | `pip install -e .`             |
| Dev     | `pip install -e .[dev]`        |
| Test    | `pytest`                       |
| Lint    | `ruff check .`                 |
| Format  | `ruff format --check .`        |

## Rules

- **Tiny framework, framework-free internals.** No LangChain, no agent frameworks. Raw API calls + a loop. This is intentional.
- **Tiny dependency set.** `httpx` handles raw API calls and `rich` renders terminal UI. Prefer stdlib for everything else.
- **Tools return strings, never raise.** Every tool function returns a string result (including errors). Wrap all exceptions inside the function.
- **Keyword-only arguments.** Tool functions accept `**kwargs` matching their JSON Schema definition in the `TOOLS` registry.
- **Path safety via `_resolve_safe()`.** All file-access tools resolve paths through `_resolve_safe()` using `ToolContext.working_dir` or the configured working directory.
- **Dangerous tools require confirmation.** Set `requires_confirmation: True` in the registry for any tool that writes files or executes commands.
- **Config via `AgentConfig` dataclass.** Load once with `load_config()` at the entry point and pass `cfg` through the call graph. Treat it as immutable; produce new configs with `dataclasses.replace`. Env vars with `.env` fallback.
- **Log everything.** API payloads are printed in full using box-drawing frames (`╭─╮│╰─╯`). Secrets are redacted via `safety.redact_secrets()`.
- **ATIF-v1.4 for trajectories.** Keep trajectory export compatible with the spec and sanitize persisted strings through `safety.py`.
- **Retry with backoff.** HTTP retries use exponential backoff on 429/5xx, max 3 retries.

## Boundaries

- **Always:** Follow existing patterns (tools return strings, kwargs-only, registry-based). Use `_resolve_safe()` for any path access. Keep the package module structure.
- **Ask first:** Adding new dependencies. Changing the system prompt. Modifying the API protocol or retry logic.
- **Never:** Add agent frameworks (LangChain, CrewAI, etc). Commit `.env` or secrets. Remove user confirmation from `write_file` or `shell_exec`.

## Canonical Examples

Follow these files as templates for new code:
- **Tool implementation:** `baremetal_agent/tools.py` — see `read_file` function + its `TOOLS` registry entry
- **API client:** `baremetal_agent/client.py` — `chat_completion` with retry logic and payload logging
- **Agent loop:** `baremetal_agent/agent.py` — `run_agent_turn` showing the tool-call loop pattern

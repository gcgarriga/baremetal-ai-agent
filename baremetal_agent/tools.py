"""Tool registry and implementations.

Each tool is a plain function that accepts keyword arguments and returns a string.
The @tool decorator registers each function into TOOLS with its confirmation flag
and OpenAI-format schema definition (description comes from the function's docstring).
"""

import fnmatch
import inspect
import os
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from baremetal_agent import safety

# ---------------------------------------------------------------------------
# Working directory (sandbox root for path-aware tools)
# ---------------------------------------------------------------------------
#
# The only way to give tools a sandbox root is to thread a ``ToolContext``
# through ``execute_tool``: tools that need filesystem isolation declare
# ``ctx: ToolContext`` as a keyword argument, and the dispatcher injects the
# caller-supplied context. Production callers (agent loop, eval runner) build
# the context explicitly per call from ``cfg.working_dir``.
#
# When ``ctx`` is ``None`` (test convenience for direct ``tools.read_file(...)``
# calls), tools fall back to ``Path.cwd()``. Production code never relies on
# this — the agent always builds a ctx — so concurrent ``run_task`` calls
# never share mutable global state.


@dataclass(frozen=True)
class ToolContext:
    """Per-call execution context for tools.

    Currently carries only the sandbox root, but is intentionally a struct so
    future per-call concerns (logger, cancellation, tool-call id, etc.) can be
    added without changing every tool signature.
    """

    working_dir: Path


def _ctx_working_dir(ctx: ToolContext | None) -> Path:
    """Return the effective sandbox root for a tool call, fully resolved.

    Uses ``ctx.working_dir`` when supplied, else ``Path.cwd()``. Always
    ``.resolve()`` the result so callers can safely use ``Path.relative_to(...)``
    against paths produced by ``_resolve_safe`` (which itself resolves the
    root). Without this, a symlinked ctx root (e.g. ``/tmp`` on macOS, which
    canonicalizes to ``/private/tmp``) would cause ``relative_to`` to raise.
    """
    return (ctx.working_dir if ctx is not None else Path.cwd()).resolve()


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

_TYPE_MAP = {"string": str, "integer": int, "boolean": bool, "number": (int, float)}


class PathEscape(ValueError):
    """Raised when a requested path resolves outside the sandbox root."""


def _validate_args(name: str, args: dict, schema: dict) -> str | None:
    """Validate arguments against a tool's JSON Schema parameters.

    Returns None if valid, or an error description string if invalid.
    """
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for req in required:
        if req not in args:
            return f"Missing required argument '{req}'"

    for key, value in args.items():
        if key not in props:
            continue  # extra args are ignored, not an error
        expected_type = props[key].get("type")
        if expected_type and expected_type in _TYPE_MAP:
            if expected_type in {"integer", "number"} and isinstance(value, bool):
                return f"Argument '{key}' must be {expected_type}, got {type(value).__name__}"
            if not isinstance(value, _TYPE_MAP[expected_type]):
                return f"Argument '{key}' must be {expected_type}, got {type(value).__name__}"

    return None


# ---------------------------------------------------------------------------
# Tool registry + decorator
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict] = {}


def tool(*, requires_confirmation: bool, parameters: dict) -> Callable:
    """Register a function as a tool.

    Builds the OpenAI-format definition from the function's name, its docstring's
    first paragraph (used as the description, with internal whitespace collapsed),
    and the supplied JSON Schema parameters. Adds an entry to TOOLS on import.

    ``ctx`` is a reserved internal kwarg threaded by ``execute_tool``; it must
    not appear in the tool's JSON Schema (``parameters``). The decorator
    asserts this so it can never accidentally leak into the model-facing API.
    """

    if "ctx" in parameters.get("properties", {}) or "ctx" in parameters.get("required", []):
        raise ValueError("'ctx' is a reserved internal kwarg and must not be declared in the tool schema")

    def decorator(func: Callable) -> Callable:
        first_paragraph = (func.__doc__ or "").strip().split("\n\n", 1)[0]
        description = " ".join(first_paragraph.split())
        # Cache whether this handler accepts ctx so execute_tool doesn't pay
        # the cost of inspect.signature() on every call (hot path: agent loop,
        # eval). Computed once at registration time.
        accepts_ctx = "ctx" in inspect.signature(func).parameters
        TOOLS[func.__name__] = {
            "handler": func,
            "requires_confirmation": requires_confirmation,
            "accepts_ctx": accepts_ctx,
            "definition": {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": description,
                    "parameters": parameters,
                },
            },
        }
        return func

    return decorator


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _resolve_safe(path: str, *, working_dir: Path | None = None) -> Path:
    """Resolve a path and verify it stays within ``working_dir``.

    Returns the resolved Path, or raises PathEscape if it escapes. If
    ``working_dir`` is omitted, falls back to ``Path.cwd()`` (test convenience
    for direct calls; production callers always pass ``working_dir``).
    """
    root = (working_dir if working_dir is not None else Path.cwd()).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise PathEscape(f"Error: Path escapes working directory: {path}")
    return target


@tool(
    requires_confirmation=False,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to working directory",
            }
        },
        "required": ["path"],
    },
)
def read_file(*, path: str, ctx: ToolContext | None = None) -> str:
    """Read the contents of a file at the given path (relative to working directory).
    Output is redacted and capped at 10,000 characters."""
    wd = _ctx_working_dir(ctx)
    try:
        target = _resolve_safe(path, working_dir=wd)
        content = target.read_text(encoding="utf-8", errors="replace")
        return safety.sanitize_text(content, label="read_file")
    except PathEscape as exc:
        return str(exc)
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool(
    requires_confirmation=True,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to working directory",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
)
def write_file(*, path: str, content: str, ctx: ToolContext | None = None) -> str:
    """Write content to a file, creating parent directories if needed"""
    wd = _ctx_working_dir(ctx)
    try:
        target = _resolve_safe(path, working_dir=wd)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"
    except PathEscape as exc:
        return str(exc)
    except Exception as exc:
        return f"Error writing file: {exc}"


@tool(
    requires_confirmation=False,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path (default: current directory)",
            }
        },
        "required": [],
    },
)
def list_directory(*, path: str = ".", ctx: ToolContext | None = None) -> str:
    """List files and directories at the given path. Output is redacted and capped at 10,000 characters."""
    wd = _ctx_working_dir(ctx)
    try:
        target = _resolve_safe(path, working_dir=wd)
        if not target.is_dir():
            return f"Error: Not a directory: {path}"
        entries = sorted(target.iterdir())
        lines = []
        for entry in entries:
            indicator = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{indicator}")
        result = "\n".join(lines) if lines else "(empty directory)"
        return safety.sanitize_text(result, label="list_directory")
    except PathEscape as exc:
        return str(exc)
    except Exception as exc:
        return f"Error listing directory: {exc}"


@tool(
    requires_confirmation=False,
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search in (default: current directory)",
            },
            "file_glob": {
                "type": "string",
                "description": "Glob pattern to filter files, e.g. '*.py' (default: all files)",
            },
        },
        "required": ["pattern"],
    },
)
def search_code(*, pattern: str, path: str = ".", file_glob: str = "*", ctx: ToolContext | None = None) -> str:
    """Search for a regex pattern in files. Returns redacted matching lines with file paths
    and line numbers. Caps at 50 matches and 10,000 characters."""
    wd = _ctx_working_dir(ctx)
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"Error: Invalid regex pattern: {exc}"

    try:
        target = _resolve_safe(path, working_dir=wd)
    except PathEscape as exc:
        return str(exc)
    if not target.exists():
        return f"Error: Path not found: {path}"

    matches = []
    max_matches = 50
    files_to_search = []

    if target.is_file():
        files_to_search = [target]
    else:
        for root, _dirs, filenames in os.walk(target):
            for filename in filenames:
                if fnmatch.fnmatch(filename, file_glob):
                    files_to_search.append(Path(root) / filename)

    max_file_bytes = 1 * 1024 * 1024  # skip files larger than 1 MB
    for filepath in files_to_search:
        if len(matches) >= max_matches:
            break
        try:
            if filepath.stat().st_size > max_file_bytes:
                continue
            with open(filepath, encoding="utf-8", errors="replace") as fh:
                for line_num, line in enumerate(fh, 1):
                    if compiled.search(line):
                        rel_path = filepath.relative_to(wd)
                        matches.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                        if len(matches) >= max_matches:
                            break
        except (PermissionError, OSError):
            continue

    if not matches:
        return safety.sanitize_text(f"No matches found for pattern '{pattern}'", label="search_code")

    result = "\n".join(matches)
    if len(matches) >= max_matches:
        result += f"\n\n(truncated at {max_matches} matches)"
    return safety.sanitize_text(result, label="search_code")


@tool(
    requires_confirmation=True,
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)",
            },
        },
        "required": ["command"],
    },
)
def shell_exec(*, command: str, timeout: int = 30, ctx: ToolContext | None = None) -> str:
    """Execute a shell command and return stdout + stderr. Requires user confirmation.
    Output is redacted and capped at 10,000 characters."""
    wd = _ctx_working_dir(ctx)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=wd,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += f"[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        return safety.sanitize_text(output, label="shell_exec") if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as exc:
        return f"Error executing command: {exc}"


@tool(
    requires_confirmation=False,
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def git_status(*, ctx: ToolContext | None = None) -> str:
    """Show the current git status (short format)"""
    wd = _ctx_working_dir(ctx)
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            cwd=wd,
        )
        if result.returncode != 0 and result.stderr.strip():
            return f"git status error: {result.stderr.strip()}"
        output = result.stdout.strip()
        return output if output else "(working tree clean)"
    except Exception as exc:
        return f"Error running git status: {exc}"


@tool(
    requires_confirmation=False,
    parameters={
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "Specific file to diff (optional — omit for full diff)",
            }
        },
        "required": [],
    },
)
def git_diff(*, file: str | None = None, ctx: ToolContext | None = None) -> str:
    """Show redacted, capped git diff output, optionally for a specific file"""
    wd = _ctx_working_dir(ctx)
    try:
        cmd = ["git", "diff", "--"]
        if file:
            target = _resolve_safe(file, working_dir=wd)
            cmd.append(str(target.relative_to(wd)))
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=wd)
        if result.returncode != 0 and result.stderr.strip():
            return f"git diff error: {result.stderr.strip()}"
        output = result.stdout.strip()
        if not output:
            return "(no changes)" if not file else f"(no changes for {file})"

        return safety.sanitize_text(output, label="git_diff")
    except PathEscape as exc:
        return str(exc)
    except Exception as exc:
        return f"Error running git diff: {exc}"


@tool(
    requires_confirmation=False,
    parameters={
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of commits to show (default: 10)",
            }
        },
        "required": [],
    },
)
def git_log(*, count: int = 10, ctx: ToolContext | None = None) -> str:
    """Show recent git commit history (oneline format)"""
    wd = _ctx_working_dir(ctx)
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{count}"],
            capture_output=True,
            text=True,
            cwd=wd,
        )
        if result.returncode != 0 and result.stderr.strip():
            return f"git log error: {result.stderr.strip()}"
        output = result.stdout.strip()
        return output if output else "(no commits)"
    except Exception as exc:
        return f"Error running git log: {exc}"


# ---------------------------------------------------------------------------
# Public registry helpers
# ---------------------------------------------------------------------------


def get_read_only_tool_names() -> list[str]:
    """Return tool names that do not require confirmation."""
    return [name for name, tool in TOOLS.items() if not tool["requires_confirmation"]]


def get_tool_definitions(names: Iterable[str] | None = None) -> list[dict]:
    """Return the OpenAI-format tools array for the API request."""
    if names is None:
        return [tool["definition"] for tool in TOOLS.values()]

    requested = set(names)
    return [tool["definition"] for name, tool in TOOLS.items() if name in requested]


def get_tool_names() -> list[str]:
    """Return a list of all registered tool names."""
    return list(TOOLS.keys())


def execute_tool(name: str, arguments: dict, *, ctx: ToolContext | None = None) -> str:
    """Execute a tool by name with the given arguments.

    Returns the tool result string. Handles unknown tools, validation errors,
    and execution exceptions — never raises.

    ``ctx`` is a reserved internal kwarg: any caller-supplied ``"ctx"`` key in
    ``arguments`` is silently stripped before validation and never reaches the
    handler. The real ``ctx`` is injected by the dispatcher only into handlers
    whose signature declares a ``ctx`` parameter. Tools that don't need a
    sandbox don't see it. If ``ctx`` is not provided, handlers that need a
    working directory fall back to ``Path.cwd()`` (test convenience).
    """
    if name not in TOOLS:
        available = ", ".join(get_tool_names())
        return f"Error: Unknown tool '{name}'. Available tools: {available}"

    tool = TOOLS[name]
    schema = tool["definition"]["function"]["parameters"]

    # Strip caller-supplied "ctx" — it's a reserved internal kwarg. The real
    # ctx (if any) is passed by keyword below; this prevents a model from
    # smuggling a forged context through the JSON arguments.
    if "ctx" in arguments:
        arguments = {k: v for k, v in arguments.items() if k != "ctx"}

    validation_error = _validate_args(name, arguments, schema)
    if validation_error:
        return f"Error: Invalid arguments for '{name}': {validation_error}"

    handler = tool["handler"]
    try:
        if tool["accepts_ctx"]:
            return handler(ctx=ctx, **arguments)
        return handler(**arguments)
    except TypeError as exc:
        return f"Error: Bad arguments for '{name}': {exc}"
    except Exception as exc:
        return f"Error executing '{name}': {exc}"

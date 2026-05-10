"""Live trajectory visualization using rich.

Rendering is dispatched through a ``Renderer`` protocol so the agent loop
takes its visibility behaviour as an explicit dependency instead of reading
a module-level flag. ``make_renderer(cfg)`` is the canonical factory:

* ``RichRenderer`` — full Rich panel rendering (default REPL behaviour).
* ``PlainRenderer`` — only error messages print as plain text; everything
  else is silent. Used in verbose mode where the user is reading raw API
  payloads via ``client.py`` and Rich panels would just add noise.
* ``NullRenderer`` — total silence. Used by eval and one-shot mode where
  the caller writes its own output.

The module-level ``_fmt_*`` helpers and ``ToolCallResult`` stay as plain
formatting helpers — ``replay.py`` reuses them.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

console = Console(highlight=False)


class ToolCallResult(TypedDict):
    name: str
    args: dict[str, Any]
    result: str
    duration_ms: float
    denied: bool


# ---------------------------------------------------------------------------
# Formatting helpers — pure functions, reused by replay.py.
# ---------------------------------------------------------------------------


def _fmt_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 60:
            v = v[:57] + "..."
        parts.append(f'{k}="{escape(str(v))}"' if isinstance(v, str) else f"{k}={v}")
    return ", ".join(parts)


def _fmt_result_summary(result: str) -> str:
    lines = result.splitlines()
    preview_lines = [escape(line) for line in lines[:3]]
    indented = "\n     ".join(preview_lines)
    if len(lines) > 3:
        indented += f"\n     ... ({len(lines) - 3} more lines)"
    return indented


def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{int(ms)}ms"


def _fmt_tokens(metrics: dict[str, Any]) -> str:
    prompt = metrics.get("prompt_tokens", 0)
    completion = metrics.get("completion_tokens", 0)
    return f"{prompt + completion} tok"


# ---------------------------------------------------------------------------
# Renderer protocol and concrete implementations.
# ---------------------------------------------------------------------------


class Renderer(Protocol):
    """Visibility surface for the agent loop.

    Each method represents one rendering hook the loop may call. Concrete
    renderers decide whether to display, log, or silently drop the event.
    """

    def render_tool_call_step(
        self,
        iteration: int,
        tool_calls_with_results: list[ToolCallResult],
        api_duration_ms: float,
        metrics: dict[str, Any],
    ) -> None: ...

    def render_response(self, text: str, api_duration_ms: float, metrics: dict[str, Any]) -> None: ...

    def render_stream_delta(self, text: str) -> None: ...

    def render_stream_end(self) -> None: ...

    def render_error(self, message: str) -> None: ...

    def render_trajectory_summary(self, iterations: int, total_tokens: int, total_ms: float) -> None: ...


class RichRenderer:
    """Default REPL renderer — full Rich panels for every event."""

    def render_tool_call_step(
        self,
        iteration: int,
        tool_calls_with_results: list[ToolCallResult],
        api_duration_ms: float,
        metrics: dict[str, Any],
    ) -> None:
        header = Text()
        header.append("🧠 Agent → tool_calls", style="bold cyan")
        header.append(f"  {_fmt_ms(api_duration_ms)}  {_fmt_tokens(metrics)}", style="dim")

        lines = []
        for i, tc in enumerate(tool_calls_with_results, 1):
            circled = "①②③④⑤⑥⑦⑧⑨⑩"[i - 1] if i <= 10 else f"({i})"
            lines.append("")
            lines.append(
                f"  [bold]{circled}[/bold] [bold green]{escape(tc['name'])}[/bold green]({_fmt_args(tc['args'])})"
            )

            if tc.get("denied"):
                lines.append("     [yellow]⚠️  denied by user[/yellow]")
            else:
                summary = _fmt_result_summary(tc["result"])
                timing = f"  [dim italic]{_fmt_ms(tc['duration_ms'])}[/dim italic]" if tc["duration_ms"] > 0 else ""
                lines.append(f"     [dim]→ {summary}[/dim]{timing}")
            lines.append("")

        panel = Panel(
            "\n".join(lines),
            title=f"[bold]Step {iteration}[/bold]",
            title_align="left",
            subtitle=header,
            subtitle_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
        console.print(panel)

    def render_response(self, text: str, api_duration_ms: float, metrics: dict[str, Any]) -> None:
        header = Text()
        header.append("💬 Agent response", style="bold green")
        header.append(f"  {_fmt_ms(api_duration_ms)}  {_fmt_tokens(metrics)}", style="dim")

        panel = Panel(
            Text(text),
            title="[bold]Response[/bold]",
            title_align="left",
            subtitle=header,
            subtitle_align="left",
            border_style="green",
            padding=(0, 1),
        )
        console.print(panel)

    def render_stream_delta(self, text: str) -> None:
        console.print(Text(text), end="")

    def render_stream_end(self) -> None:
        console.print()

    def render_error(self, message: str) -> None:
        panel = Panel(
            Text(message, style="red"),
            title="[bold red]❌ Error[/bold red]",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        )
        console.print(panel)

    def render_trajectory_summary(self, iterations: int, total_tokens: int, total_ms: float) -> None:
        line = (
            f"[dim]─── Trajectory: {iterations} step{'s' if iterations != 1 else ''}"
            f" │ {total_tokens} tokens"
            f" │ {_fmt_ms(total_ms)} total " + "─" * 20 + "[/dim]"
        )
        console.print(line)
        console.print()


class PlainRenderer:
    """Verbose-mode renderer — only errors print, as plain text.

    In verbose mode the user is inspecting raw API payloads written by
    ``client.py``; Rich panels would interleave noisily, but losing terminal
    errors entirely would be worse than printing them plainly.
    """

    def render_tool_call_step(
        self,
        iteration: int,
        tool_calls_with_results: list[ToolCallResult],
        api_duration_ms: float,
        metrics: dict[str, Any],
    ) -> None:
        return None

    def render_response(self, text: str, api_duration_ms: float, metrics: dict[str, Any]) -> None:
        return None

    def render_stream_delta(self, text: str) -> None:
        return None

    def render_stream_end(self) -> None:
        return None

    def render_error(self, message: str) -> None:
        print(f"\n❌ {message}\n")

    def render_trajectory_summary(self, iterations: int, total_tokens: int, total_ms: float) -> None:
        return None


class NullRenderer:
    """Silent renderer — every method is a no-op.

    Used by eval (results captured via ATIF) and one-shot mode (caller prints
    the final answer to stdout itself).
    """

    def render_tool_call_step(
        self,
        iteration: int,
        tool_calls_with_results: list[ToolCallResult],
        api_duration_ms: float,
        metrics: dict[str, Any],
    ) -> None:
        return None

    def render_response(self, text: str, api_duration_ms: float, metrics: dict[str, Any]) -> None:
        return None

    def render_stream_delta(self, text: str) -> None:
        return None

    def render_stream_end(self) -> None:
        return None

    def render_error(self, message: str) -> None:
        return None

    def render_trajectory_summary(self, iterations: int, total_tokens: int, total_ms: float) -> None:
        return None


def make_renderer(cfg: Any) -> Renderer:
    """Pick a renderer for ``cfg``.

    Verbose mode opts into ``PlainRenderer`` so raw API payloads from
    ``client.py`` are not interleaved with Rich panels. Otherwise the default
    REPL behaviour is the full Rich UI.
    """
    return PlainRenderer() if cfg.render_verbose else RichRenderer()

"""The agentic tool-use loop — the core of the project.

Takes a user message, sends it to the LLM with tool definitions, parses the
response, executes any tool calls, feeds results back, and loops until the
model produces a final text response or the iteration limit is reached.

Visibility and confirmation behaviour are injected as a ``Renderer`` and a
``Confirmer`` — the loop never reads module-level flags or calls ``input()``
itself. See ``baremetal_agent.visualizer.make_renderer`` and the
``*_confirmer`` callables defined below.
"""

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypedDict

from baremetal_agent import client, tools
from baremetal_agent.config import AgentConfig
from baremetal_agent.visualizer import Renderer, ToolCallResult


class _RequiredMessage(TypedDict):
    role: str


class Message(_RequiredMessage, total=False):
    content: str | None
    tool_calls: list[dict[str, object]]
    tool_call_id: str
    _model: str


STATUS_OK = "ok"
STATUS_API_ERROR = "api_error"
STATUS_MAX_ITERATIONS = "max_iterations"


# ---------------------------------------------------------------------------
# Confirmer protocol — controls whether dangerous tools execute.
# ---------------------------------------------------------------------------

# A Confirmer answers: should this dangerous tool call run? Returning False
# converts the call into a "denied" result. Confirmers handle their own I/O;
# the agent loop never calls ``input()``.
Confirmer = Callable[[str, dict], bool]


def auto_approve_confirmer(_tool_name: str, _arguments: dict) -> bool:
    """Always approve — used by ``--allow-dangerous-tools`` and write-eval."""
    return True


def auto_deny_confirmer(_tool_name: str, _arguments: dict) -> bool:
    """Always deny — used by ``--read-only``, default one-shot, and read-eval."""
    return False


@dataclass(frozen=True)
class AgentTurnResult:
    """Final text and machine-readable status for a single user turn."""

    content: str
    status: str


def _make_tool_result(
    *,
    name: str,
    args: dict,
    call_id: str,
    content: str,
    duration_ms: float = 0.0,
    denied: bool = False,
) -> tuple[ToolCallResult, Message]:
    """Build a (visualizer result, tool history message) pair from a tool outcome."""
    result: ToolCallResult = {
        "name": name,
        "args": args,
        "result": content,
        "duration_ms": duration_ms,
        "denied": denied,
    }
    tool_msg: Message = {"role": "tool", "tool_call_id": call_id, "content": content}
    return result, tool_msg


def _strip_internal_keys(messages: list[Message]) -> list[dict]:
    """Return a shallow-copied message list with any underscore-prefixed
    metadata keys (e.g. ``_model``) removed.

    Internal keys travel with the assistant message through ``history`` so
    that downstream consumers (like trajectory export) can attribute each
    turn to the model that produced it. They must be stripped before the
    history is sent to the API to avoid sending non-OpenAI fields.
    """
    return [{k: v for k, v in msg.items() if not k.startswith("_")} for msg in messages]


def _dispatch_tool_call(
    tool_call: dict,
    *,
    confirmer: Confirmer,
    allowed_tool_names: set[str] | None,
    ctx: tools.ToolContext | None = None,
) -> tuple[ToolCallResult, Message]:
    """Execute a single tool call and return (visualizer result, tool history message).

    Handles argument parsing, the read-only/policy filter, the dangerous-tool
    confirmation gate, and timed execution. Argument-parse errors and denials
    are returned as well-formed results. Tool execution is delegated to
    ``tools.execute_tool()``, which returns a result string and reports tool
    handler failures as error text rather than propagating those exceptions to
    the caller.
    """
    call_id = tool_call["id"]
    func = tool_call["function"]
    tool_name = func["name"]
    raw_args = func["arguments"]

    try:
        arguments = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as exc:
        return _make_tool_result(
            name=tool_name,
            args={},
            call_id=call_id,
            content=f"Error: Could not parse arguments as JSON: {exc}\nRaw: {raw_args}",
        )

    if not isinstance(arguments, dict):
        return _make_tool_result(
            name=tool_name,
            args={},
            call_id=call_id,
            content=f"Error: Tool arguments must be a JSON object, got {type(arguments).__name__}",
        )

    if allowed_tool_names is not None and tool_name not in allowed_tool_names:
        return _make_tool_result(
            name=tool_name,
            args=arguments,
            call_id=call_id,
            content=f"Tool execution denied by policy: {tool_name} is not enabled.",
            denied=True,
        )

    if (
        tool_name in tools.TOOLS
        and tools.TOOLS[tool_name]["requires_confirmation"]
        and not confirmer(tool_name, arguments)
    ):
        return _make_tool_result(
            name=tool_name,
            args=arguments,
            call_id=call_id,
            content="Tool execution denied.",
            denied=True,
        )

    tool_start = time.time()
    content = tools.execute_tool(tool_name, arguments, ctx=ctx)
    duration_ms = (time.time() - tool_start) * 1000
    return _make_tool_result(
        name=tool_name,
        args=arguments,
        call_id=call_id,
        content=content,
        duration_ms=duration_ms,
    )


def run_agent_turn(
    user_message: str,
    history: list[Message],
    api_responses: list[dict],
    *,
    cfg: AgentConfig,
    renderer: Renderer,
    confirmer: Confirmer,
    tool_names: Iterable[str] | None = None,
    on_stream_delta: Callable[[str], None] | None = None,
) -> AgentTurnResult:
    """Run a single user turn through the agentic loop.

    Appends to ``history`` and ``api_responses`` in place. Returns final text
    plus status. Streaming is controlled by ``cfg.stream``; callers wanting
    an override should pass a derived cfg via ``dataclasses.replace(cfg,
    stream=...)``. ``renderer`` controls every visible event (use
    ``NullRenderer`` for silence); ``confirmer`` decides whether
    confirmation-required tools execute.
    """
    # Mark rollback point before any mutations
    history_start = len(history)
    responses_start = len(api_responses)
    history.append({"role": "user", "content": user_message})

    allowed_tool_names = set(tool_names) if tool_names is not None else None
    tool_definitions = tools.get_tool_definitions(allowed_tool_names)
    # Build the per-turn tool context from cfg. Production code paths read the
    # sandbox root from this ctx — there is no module-global fallback in
    # production.
    tool_ctx = tools.ToolContext(working_dir=cfg.working_dir)
    iteration = 0
    turn_start = time.time()
    cumulative_tokens = 0
    stream = cfg.stream

    while iteration < cfg.max_iterations:
        # Call the LLM
        try:
            api_start = time.time()
            api_messages = _strip_internal_keys(history)
            if stream:
                response = client.chat_completion_stream(
                    api_messages,
                    tool_definitions,
                    cfg=cfg,
                    on_stream_delta=(on_stream_delta if on_stream_delta is not None else renderer.render_stream_delta),
                )
            else:
                response = client.chat_completion(api_messages, tool_definitions, cfg=cfg)
            api_duration_ms = (time.time() - api_start) * 1000
        except RuntimeError as exc:
            # Roll back everything added during this turn
            del history[history_start:]
            del api_responses[responses_start:]
            error_msg = f"API error: {exc}"
            renderer.render_error(error_msg)
            return AgentTurnResult(error_msg, STATUS_API_ERROR)

        api_responses.append(response)
        metrics = response.get("usage", {})
        step_tokens = metrics.get("prompt_tokens", 0) + metrics.get("completion_tokens", 0)
        cumulative_tokens += step_tokens

        # Parse the response
        choices = response.get("choices")
        if not choices:
            del history[history_start:]
            del api_responses[responses_start:]
            error_msg = "API error: Response contained no choices"
            renderer.render_error(error_msg)
            return AgentTurnResult(error_msg, STATUS_API_ERROR)
        choice = choices[0]
        message = choice["message"]

        # Case 1: model wants to call tools (check this first — some providers
        # may return tool_calls alongside finish_reason="stop")
        if message.get("tool_calls"):
            assistant_msg: Message = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message["tool_calls"],
                "_model": cfg.model,
            }
            history.append(assistant_msg)

            tool_calls_with_results: list[ToolCallResult] = []
            for tool_call in message["tool_calls"]:
                result, tool_msg = _dispatch_tool_call(
                    tool_call,
                    confirmer=confirmer,
                    allowed_tool_names=allowed_tool_names,
                    ctx=tool_ctx,
                )
                tool_calls_with_results.append(result)
                history.append(tool_msg)

            iteration += 1
            renderer.render_tool_call_step(
                iteration,
                tool_calls_with_results,
                api_duration_ms,
                metrics,
            )
            continue

        # Case 2: model produced a final text response (no tool_calls)
        content = message.get("content", "")
        history.append({"role": "assistant", "content": content, "_model": cfg.model})

        if stream:
            renderer.render_stream_end()
        else:
            renderer.render_response(content, api_duration_ms, metrics)
        total_ms = (time.time() - turn_start) * 1000
        renderer.render_trajectory_summary(iteration + 1, cumulative_tokens, total_ms)

        return AgentTurnResult(content, STATUS_OK)

    # Hit the iteration limit
    limit_msg = (
        f"Reached maximum iteration limit ({cfg.max_iterations}). "
        f"The agent made {cfg.max_iterations} rounds of tool calls without "
        f"producing a final answer. Use 'clear' to reset the conversation."
    )
    renderer.render_error(limit_msg)
    return AgentTurnResult(limit_msg, STATUS_MAX_ITERATIONS)

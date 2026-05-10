"""Offline replay and diff helpers for ATIF-v1.4 trajectories."""

import hashlib
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any

from baremetal_agent import safety, visualizer
from baremetal_agent import trajectory as trajectory_module

SCHEMA_VERSION = trajectory_module.SCHEMA_VERSION
DISPLAY_MAX_CHARS = 1_000
SUMMARY_MAX_CHARS = 160
OBSERVATION_PREVIEW_CHARS = 80
MAX_OBSERVATION_DIFFS = 8


def load_trajectory(path: str) -> dict:
    """Load a readable ATIF-v1.4 trajectory JSON file."""
    try:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise ValueError(f"could not read trajectory {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in trajectory {path!r}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("trajectory JSON must be an object")
    if data.get("schema_version") != SCHEMA_VERSION:
        found = data.get("schema_version", "(missing)")
        raise ValueError(f"unsupported trajectory schema_version {found!r}; expected {SCHEMA_VERSION!r}")
    _steps(data)
    return data


def render(trajectory: dict, step_id: int | None = None) -> None:
    """Print a terminal-friendly replay view of ATIF steps."""
    steps = _steps(trajectory)
    selected_steps = steps
    if step_id is not None:
        selected_steps = [step for step in steps if step.get("step_id") == step_id]
        if not selected_steps:
            raise ValueError(f"no step with step_id={step_id}")

    totals = _token_totals(trajectory)
    session_id = _safe_inline(trajectory.get("session_id", "(unknown)"), max_chars=SUMMARY_MAX_CHARS)
    print(f"Trajectory {session_id} ({trajectory.get('schema_version', '(unknown schema)')})")

    agent_info = trajectory.get("agent", {})
    if isinstance(agent_info, dict):
        agent_name = _safe_inline(agent_info.get("name", "unknown-agent"), max_chars=SUMMARY_MAX_CHARS)
        model = _safe_inline(agent_info.get("model_name", "unknown-model"), max_chars=SUMMARY_MAX_CHARS)
        version = agent_info.get("version")
        version_text = f" {_safe_inline(version, max_chars=SUMMARY_MAX_CHARS)}" if version else ""
        print(f"Agent: {agent_name}{version_text} | model: {model}")

    print(_format_totals("Totals", totals))
    if step_id is not None:
        print(f"Filtered to ATIF step_id={step_id}")
    print()

    if not selected_steps:
        print("(no steps)")
        return

    for index, step in enumerate(selected_steps):
        if index:
            print()
        _render_step(step)


def diff(traj_a: dict, traj_b: dict) -> str:
    """Return a plain-text comparison between two ATIF-v1.4 trajectories."""
    a_totals = _token_totals(traj_a)
    b_totals = _token_totals(traj_b)
    a_final = _final_response_summary(traj_a)
    b_final = _final_response_summary(traj_b)
    a_sequence = _tool_sequence(traj_a)
    b_sequence = _tool_sequence(traj_b)
    a_counts = Counter(item["name"] for item in a_sequence)
    b_counts = Counter(item["name"] for item in b_sequence)

    lines = [
        "Trajectory diff",
        _format_totals("A", a_totals),
        _format_totals("B", b_totals),
        "",
        "Final response:",
        f"  A: {_format_final_response(a_final)}",
        f"  B: {_format_final_response(b_final)}",
        "",
        "Tool-call sequence by index:",
    ]

    if not a_sequence and not b_sequence:
        lines.append("  (no tool calls)")
    else:
        for index, (a_call, b_call) in enumerate(zip_longest(a_sequence, b_sequence), 1):
            marker = "" if _tool_call_signature(a_call) == _tool_call_signature(b_call) else "  != "
            lines.append(f"  {index}. A={_format_tool_call(a_call)} | B={_format_tool_call(b_call)}{marker}")

    lines.extend(["", "Aggregate tool-call counts:"])
    tool_names = sorted(set(a_counts) | set(b_counts))
    if not tool_names:
        lines.append("  (none)")
    else:
        for name in tool_names:
            lines.append(f"  {name}: A={a_counts[name]} B={b_counts[name]}")

    lines.extend(["", "Observation/result differences:"])
    observation_diffs = _observation_differences(traj_a, traj_b)
    if not observation_diffs:
        lines.append("  none")
    else:
        for diff_line in observation_diffs[:MAX_OBSERVATION_DIFFS]:
            lines.append(f"  {diff_line}")
        remaining = len(observation_diffs) - MAX_OBSERVATION_DIFFS
        if remaining > 0:
            lines.append(f"  ... {remaining} more observation difference(s)")

    return "\n".join(lines)


def _steps(trajectory: dict) -> list[dict]:
    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("trajectory steps must be a list")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("trajectory steps must contain objects")
    return steps


def _render_step(step: dict) -> None:
    source = _safe_inline(step.get("source", "unknown"), max_chars=SUMMARY_MAX_CHARS)
    step_id = step.get("step_id", "?")
    header_parts = [f"Step {step_id} [{source}]"]

    model = step.get("model_name")
    if model:
        header_parts.append(f"model={_safe_inline(model, max_chars=SUMMARY_MAX_CHARS)}")

    metrics = step.get("metrics", {})
    if isinstance(metrics, dict) and metrics:
        header_parts.append(visualizer._fmt_tokens(metrics))

    print(" | ".join(header_parts))

    message = step.get("message")
    if message is not None:
        label = "Response" if source == "agent" else "Message"
        _print_block(label, _safe_text(message, max_chars=DISPLAY_MAX_CHARS))

    tool_calls = _tool_calls(step)
    if tool_calls:
        print("  Tool calls:")
        for index, tool_call in enumerate(tool_calls, 1):
            call_id = _safe_inline(tool_call.get("tool_call_id", "?"), max_chars=SUMMARY_MAX_CHARS)
            name = _safe_inline(tool_call.get("function_name", "(unknown)"), max_chars=SUMMARY_MAX_CHARS)
            args = _format_args(tool_call.get("arguments", {}))
            print(f"    {index}. {name}({args}) id={call_id}")

        observations = _observation_results(step)
        if observations:
            print("  Observations:")
            for index, result in enumerate(observations, 1):
                call_id = _safe_inline(result.get("source_call_id", "?"), max_chars=SUMMARY_MAX_CHARS)
                content = _safe_text(result.get("content", ""), max_chars=DISPLAY_MAX_CHARS)
                print(f"    {index}. {call_id}: {visualizer._fmt_result_summary(content)}")
        else:
            print("  Observations: (none)")


def _print_block(label: str, text: str) -> None:
    lines = text.splitlines() or [""]
    print(f"  {label}: {lines[0]}")
    for line in lines[1:]:
        print(f"    {line}")


def _token_totals(trajectory: dict) -> dict[str, int]:
    steps = _steps(trajectory)
    final_metrics = trajectory.get("final_metrics", {})
    if not isinstance(final_metrics, dict):
        final_metrics = {}

    has_final_token_metrics = any(key in final_metrics for key in trajectory_module.FINAL_METRIC_FIELDS)
    if has_final_token_metrics:
        prompt = _as_int(final_metrics.get(trajectory_module.FINAL_PROMPT_TOKENS_FIELD))
        completion = _as_int(final_metrics.get(trajectory_module.FINAL_COMPLETION_TOKENS_FIELD))
        cached = _as_int(final_metrics.get(trajectory_module.FINAL_CACHED_TOKENS_FIELD))
        total_steps = (
            _as_int(final_metrics.get(trajectory_module.FINAL_STEPS_FIELD))
            if trajectory_module.FINAL_STEPS_FIELD in final_metrics
            else len(steps)
        )
    else:
        prompt = 0
        completion = 0
        cached = 0
        for step in steps:
            metrics = step.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            prompt += _as_int(metrics.get("prompt_tokens"))
            completion += _as_int(metrics.get("completion_tokens"))
            cached += _as_int(metrics.get("cached_tokens"))
        total_steps = len(steps)

    return {
        "steps": total_steps,
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
        "total": prompt + completion,
    }


def _format_totals(label: str, totals: dict[str, int]) -> str:
    return (
        f"{label}: steps={totals['steps']}, tokens={totals['total']} "
        f"(prompt={totals['prompt']}, completion={totals['completion']}, cached={totals['cached']})"
    )


def _final_response_summary(trajectory: dict) -> str | None:
    for step in reversed(_steps(trajectory)):
        if step.get("source") != "agent":
            continue
        message = step.get("message")
        if isinstance(message, str) and message.strip():
            return _safe_inline(message, max_chars=SUMMARY_MAX_CHARS)
    return None


def _format_final_response(summary: str | None) -> str:
    if summary is None:
        return "absent"
    return f'present "{summary}"'


def _tool_sequence(trajectory: dict) -> list[dict[str, Any]]:
    sequence: list[dict[str, Any]] = []
    for step in _steps(trajectory):
        for tool_call in _tool_calls(step):
            sequence.append(
                {
                    "step_id": step.get("step_id", "?"),
                    "name": _safe_inline(tool_call.get("function_name", "(unknown)"), max_chars=SUMMARY_MAX_CHARS),
                }
            )
    return sequence


def _tool_call_signature(tool_call: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if tool_call is None:
        return None
    return (tool_call["step_id"], tool_call["name"])


def _format_tool_call(tool_call: dict[str, Any] | None) -> str:
    if tool_call is None:
        return "(missing)"
    return f"step {tool_call['step_id']} {tool_call['name']}"


def _observation_differences(traj_a: dict, traj_b: dict) -> list[str]:
    a_observations = _observations(traj_a)
    b_observations = _observations(traj_b)
    differences = []

    for index, (a_obs, b_obs) in enumerate(zip_longest(a_observations, b_observations), 1):
        if _observation_signature(a_obs) == _observation_signature(b_obs):
            continue
        differences.append(f"{index}. A={_format_observation(a_obs)} | B={_format_observation(b_obs)}")

    return differences


def _observations(trajectory: dict) -> list[dict[str, Any]]:
    observations = []
    for step in _steps(trajectory):
        for result in _observation_results(step):
            observations.append(
                {
                    "step_id": step.get("step_id", "?"),
                    "source_call_id": result.get("source_call_id", "?"),
                    "content": result.get("content", ""),
                }
            )
    return observations


def _observation_signature(observation: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if observation is None:
        return None
    content = _redacted_text(observation.get("content", ""))
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return (observation["step_id"], digest)


def _format_observation(observation: dict[str, Any] | None) -> str:
    if observation is None:
        return "(missing)"

    content = _redacted_text(observation.get("content", ""))
    line_count = len(content.splitlines()) if content else 0
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    preview_source = content.splitlines()[0] if content else ""
    preview = _safe_inline(preview_source, max_chars=OBSERVATION_PREVIEW_CHARS)
    call_id = _safe_inline(observation.get("source_call_id", "?"), max_chars=SUMMARY_MAX_CHARS)
    return (
        f"step {observation['step_id']}/{call_id}: {line_count} line(s), "
        f'{len(content)} chars, sha256={digest}, preview="{preview}"'
    )


def _tool_calls(step: dict) -> list[dict]:
    tool_calls = step.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        return []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]


def _observation_results(step: dict) -> list[dict]:
    observation = step.get("observation", {})
    if not isinstance(observation, dict):
        return []
    results = observation.get("results", [])
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, dict)]


def _format_args(args: object) -> str:
    safe_args = safety.sanitize_json_value(args, max_string_chars=SUMMARY_MAX_CHARS, label="replay")
    if isinstance(safe_args, dict):
        return visualizer._fmt_args(safe_args)
    return _safe_inline(json.dumps(safe_args, sort_keys=True, ensure_ascii=False), max_chars=SUMMARY_MAX_CHARS)


def _safe_text(value: object, *, max_chars: int) -> str:
    return safety.truncate_text(_redacted_text(value), max_chars=max_chars, label="replay")


def _redacted_text(value: object) -> str:
    return safety.redact_secrets(str(value))


def _safe_inline(value: object, *, max_chars: int) -> str:
    text = _safe_text(value, max_chars=max_chars)
    return text.replace("\n", "\\n")


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

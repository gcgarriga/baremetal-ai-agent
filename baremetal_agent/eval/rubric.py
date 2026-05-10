"""Rubric evaluation dispatch table.

Each rubric type maps to a single handler function in ``RUBRICS``. Looking up
a rubric is therefore a one-line dict lookup; adding a new rubric type means
adding one entry to the table.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from baremetal_agent import trajectory

if TYPE_CHECKING:
    from .cache import WorkspaceSnapshot
    from .loader import EvalTask


@dataclass(frozen=True)
class CheckResult:
    type: str
    passed: bool
    message: str


@dataclass(frozen=True)
class _RubricContext:
    """Pre-computed values shared across rubric handlers for one task run."""

    atif: dict
    workspace: WorkspaceSnapshot
    agent_steps: list[dict]
    flattened_calls: list[dict]
    final_response: str


RubricHandler = Callable[[Mapping[str, object], _RubricContext, str], CheckResult]


def _agent_steps(atif: dict) -> list[dict]:
    steps = atif.get("steps") if isinstance(atif, dict) else None
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict) and step.get("source") == "agent"]


def _final_response(atif: dict) -> str:
    for step in reversed(_agent_steps(atif)):
        message = step.get("message")
        if isinstance(message, str):
            return message
    return ""


def _tool_calls(atif: dict) -> list[dict]:
    calls: list[dict] = []
    for step in _agent_steps(atif):
        tool_calls = step.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if isinstance(call, dict):
                calls.append(call)
    return calls


def _json_subset(expected: object, actual: object) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if not _json_subset(expected_value, actual[key]):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if len(expected) != len(actual):
            return False
        return all(
            _json_subset(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual, strict=True)
        )
    return expected == actual


def _check_final_response_contains(
    params: Mapping[str, object], context: _RubricContext, check_type: str
) -> CheckResult:
    contains = params.get("contains")
    pattern = params.get("regex")
    if (contains is None and pattern is None) or (contains is not None and pattern is not None):
        return CheckResult(
            type=check_type,
            passed=False,
            message="final_response_contains requires exactly one of 'contains' or 'regex'",
        )
    if contains is not None:
        if not isinstance(contains, str):
            return CheckResult(type=check_type, passed=False, message="'contains' must be a string")
        passed = contains in context.final_response
        message = (
            "final response contains substring" if passed else f"final response did not contain substring: {contains}"
        )
        return CheckResult(type=check_type, passed=passed, message=message)
    if not isinstance(pattern, str):
        return CheckResult(type=check_type, passed=False, message="'regex' must be a string")
    try:
        passed = re.search(pattern, context.final_response) is not None
    except re.error as exc:
        return CheckResult(type=check_type, passed=False, message=f"invalid regex for final response: {exc}")
    message = "final response matched regex" if passed else f"final response did not match regex: {pattern}"
    return CheckResult(type=check_type, passed=passed, message=message)


def _check_trajectory_atif(params: Mapping[str, object], context: _RubricContext, check_type: str) -> CheckResult:
    expected_schema = params.get("schema_version", trajectory.SCHEMA_VERSION)
    if not isinstance(expected_schema, str):
        return CheckResult(type=check_type, passed=False, message="'schema_version' must be a string")

    require_final_metrics = params.get("require_final_metrics", False)
    if not isinstance(require_final_metrics, bool):
        return CheckResult(type=check_type, passed=False, message="'require_final_metrics' must be a boolean")

    actual_schema = context.atif.get("schema_version")
    if actual_schema != expected_schema:
        actual_display = actual_schema if isinstance(actual_schema, str) else "(missing)"
        return CheckResult(
            type=check_type,
            passed=False,
            message=f"schema_version mismatch: expected {expected_schema}, actual {actual_display}",
        )

    steps = context.atif.get("steps")
    if not isinstance(steps, list):
        return CheckResult(type=check_type, passed=False, message="trajectory steps must be a list")
    if not all(isinstance(step, dict) for step in steps):
        return CheckResult(type=check_type, passed=False, message="trajectory steps must contain objects")

    if require_final_metrics:
        final_metrics = context.atif.get("final_metrics")
        if not isinstance(final_metrics, dict):
            return CheckResult(type=check_type, passed=False, message="final_metrics missing or not an object")
        missing_fields = [field for field in trajectory.FINAL_METRIC_FIELDS if field not in final_metrics]
        if missing_fields:
            return CheckResult(
                type=check_type,
                passed=False,
                message=f"final_metrics missing required field(s): {', '.join(missing_fields)}",
            )

    metrics_message = " and final_metrics fields present" if require_final_metrics else ""
    return CheckResult(
        type=check_type,
        passed=True,
        message=f"trajectory schema_version {expected_schema}{metrics_message}",
    )


def _check_tool_called(params: Mapping[str, object], context: _RubricContext, check_type: str) -> CheckResult:
    name = params.get("name")
    if not isinstance(name, str):
        return CheckResult(type=check_type, passed=False, message="'name' must be a string")
    passed = any(call.get("function_name") == name for call in context.flattened_calls)
    message = f"tool '{name}' was called" if passed else f"tool '{name}' was not called"
    return CheckResult(type=check_type, passed=passed, message=message)


def _check_tool_called_with(params: Mapping[str, object], context: _RubricContext, check_type: str) -> CheckResult:
    name = params.get("name")
    expected_arguments = params.get("arguments")
    if not isinstance(name, str):
        return CheckResult(type=check_type, passed=False, message="'name' must be a string")
    if not isinstance(expected_arguments, dict):
        return CheckResult(type=check_type, passed=False, message="'arguments' must be an object")
    passed = False
    for call in context.flattened_calls:
        if call.get("function_name") != name:
            continue
        if _json_subset(expected_arguments, call.get("arguments")):
            passed = True
            break
    message = (
        f"tool '{name}' was called with expected arguments"
        if passed
        else f"tool '{name}' was not called with expected arguments"
    )
    return CheckResult(type=check_type, passed=passed, message=message)


def _check_tool_not_called(params: Mapping[str, object], context: _RubricContext, check_type: str) -> CheckResult:
    name = params.get("name")
    if not isinstance(name, str):
        return CheckResult(type=check_type, passed=False, message="'name' must be a string")
    passed = not any(call.get("function_name") == name for call in context.flattened_calls)
    message = f"tool '{name}' was not called" if passed else f"tool '{name}' was called"
    return CheckResult(type=check_type, passed=passed, message=message)


def _check_file_exists(params: Mapping[str, object], context: _RubricContext, check_type: str) -> CheckResult:
    path = params.get("path")
    if not isinstance(path, str):
        return CheckResult(type=check_type, passed=False, message="'path' must be a string")
    normalized = Path(path).as_posix()
    snapshot = context.workspace.files.get(normalized)
    passed = snapshot is not None and snapshot.exists is True
    message = f"file exists: {normalized}" if passed else f"file does not exist: {normalized}"
    return CheckResult(type=check_type, passed=passed, message=message)


def _check_file_contains(params: Mapping[str, object], context: _RubricContext, check_type: str) -> CheckResult:
    path = params.get("path")
    contains = params.get("contains")
    pattern = params.get("regex")
    if not isinstance(path, str):
        return CheckResult(type=check_type, passed=False, message="'path' must be a string")
    if (contains is None and pattern is None) or (contains is not None and pattern is not None):
        return CheckResult(
            type=check_type,
            passed=False,
            message="file_contains requires exactly one of 'contains' or 'regex'",
        )
    normalized = Path(path).as_posix()
    snapshot = context.workspace.files.get(normalized)
    if snapshot is None or snapshot.exists is not True or not isinstance(snapshot.content, str):
        return CheckResult(
            type=check_type,
            passed=False,
            message=f"file not available in workspace snapshot: {normalized}",
        )
    content = snapshot.content
    if contains is not None:
        if not isinstance(contains, str):
            return CheckResult(type=check_type, passed=False, message="'contains' must be a string")
        passed = contains in content
        message = (
            f"file contained substring: {normalized}" if passed else f"file did not contain substring: {normalized}"
        )
        return CheckResult(type=check_type, passed=passed, message=message)
    if not isinstance(pattern, str):
        return CheckResult(type=check_type, passed=False, message="'regex' must be a string")
    try:
        passed = re.search(pattern, content) is not None
    except re.error as exc:
        return CheckResult(type=check_type, passed=False, message=f"invalid regex for file: {exc}")
    message = f"file matched regex: {normalized}" if passed else f"file did not match regex: {normalized}"
    return CheckResult(type=check_type, passed=passed, message=message)


def _check_max_agent_steps(params: Mapping[str, object], context: _RubricContext, check_type: str) -> CheckResult:
    value = params.get("value")
    if not isinstance(value, int) or isinstance(value, bool):
        return CheckResult(type=check_type, passed=False, message="'value' must be an integer")
    count = len(context.agent_steps)
    passed = count <= value
    message = f"agent steps {count} <= {value}" if passed else f"agent steps {count} exceeded {value}"
    return CheckResult(type=check_type, passed=passed, message=message)


# Dispatch table — single source of truth for both validation and execution.
RUBRICS: dict[str, RubricHandler] = {
    "final_response_contains": _check_final_response_contains,
    "trajectory_atif": _check_trajectory_atif,
    "tool_called": _check_tool_called,
    "tool_called_with": _check_tool_called_with,
    "tool_not_called": _check_tool_not_called,
    "file_exists": _check_file_exists,
    "file_contains": _check_file_contains,
    "max_agent_steps": _check_max_agent_steps,
}

RUBRIC_TYPES: frozenset[str] = frozenset(RUBRICS)


def evaluate_rubric(task: EvalTask, atif: dict, workspace: WorkspaceSnapshot) -> tuple[CheckResult, ...]:
    context = _RubricContext(
        atif=atif,
        workspace=workspace,
        agent_steps=_agent_steps(atif),
        flattened_calls=_tool_calls(atif),
        final_response=_final_response(atif),
    )
    results: list[CheckResult] = []
    for check in task.rubric:
        handler = RUBRICS.get(check.type)
        if handler is None:
            results.append(CheckResult(type=check.type, passed=False, message=f"unsupported rubric type: {check.type}"))
            continue
        results.append(handler(check.params, context, check.type))
    return tuple(results)

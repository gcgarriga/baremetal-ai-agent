"""Eval harness public API.

This package replaces the previous single ``baremetal_agent/eval.py`` module.
Submodules:

* ``loader``  — task discovery, JSON parsing, dataclass types.
* ``rubric``  — rubric dispatch table and ``evaluate_rubric``.
* ``cache``   — cache key hashing, on-disk read/write, sandbox/repo helpers.
* ``runner``  — ``execute_task``, ``run_task``, ``run_eval_suite``.
* ``report``  — markdown/JSON report rendering.
"""

from __future__ import annotations

# Re-export the live ``baremetal_agent`` modules at the package level so tests
# (and callers) can do ``baremetal_agent.eval.agent`` / ``.trajectory`` and
# monkey-patch attributes on the same module object the runner uses.
from baremetal_agent import agent, replay, tools, trajectory  # noqa: E402

from . import cache, loader, report, rubric, runner

# Cache: snapshot types.
from .cache import FileSnapshot, WorkspaceSnapshot

# Loader: task types and parsing.
from .loader import (
    EvalTask,
    RubricCheck,
    SetupFile,
    TaskSetup,
    load_task,
    load_tasks,
    validate_task,
)

# Report: rendering.
from .report import render_json_report, render_markdown_report

# Rubric: dispatch table and check result type.
from .rubric import RUBRIC_TYPES, RUBRICS, CheckResult, evaluate_rubric

# Runner: orchestration entry points and status constants.
from .runner import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    EvalResult,
    execute_task,
    run_eval_suite,
    run_task,
)

__all__ = [
    # status constants
    "STATUS_ERROR",
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_SKIPPED",
    # rubric registry
    "RUBRICS",
    "RUBRIC_TYPES",
    # types
    "CheckResult",
    "EvalResult",
    "EvalTask",
    "FileSnapshot",
    "RubricCheck",
    "SetupFile",
    "TaskSetup",
    "WorkspaceSnapshot",
    # public functions
    "evaluate_rubric",
    "execute_task",
    "load_task",
    "load_tasks",
    "render_json_report",
    "render_markdown_report",
    "run_eval_suite",
    "run_task",
    "validate_task",
    # re-exported submodule references (used by tests/integrations)
    "agent",
    "cache",
    "loader",
    "replay",
    "report",
    "rubric",
    "runner",
    "tools",
    "trajectory",
]

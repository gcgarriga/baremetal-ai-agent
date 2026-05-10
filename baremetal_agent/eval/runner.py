"""Eval task execution and suite orchestration."""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from baremetal_agent import agent, replay, trajectory
from baremetal_agent.config import AgentConfig, load_config
from baremetal_agent.visualizer import NullRenderer

from .cache import (
    WorkspaceSnapshot,
    _copy_repo_to_sandbox,
    _execution_hash,
    _initialize_sandbox_git_repo,
    _load_cache_metadata,
    _load_workspace_snapshot,
    _resolve_fixture_source,
    _resolve_sandbox_path,
    _save_cache_metadata,
    _save_workspace_snapshot,
    _tool_names_for_task,
    _workspace_snapshot,
)
from .loader import EvalTask, load_tasks
from .report import _write_report_outputs, render_json_report, render_markdown_report
from .rubric import CheckResult, evaluate_rubric

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class EvalResult:
    task_id: str
    description: str
    status: str
    checks: tuple[CheckResult, ...] = ()
    error: str | None = None
    trajectory_path: str | None = None
    cached: bool = False


def _apply_setup(task: EvalTask, sandbox: Path) -> None:
    sandbox.mkdir(parents=True, exist_ok=True)

    if task.setup.copy_repo:
        _copy_repo_to_sandbox(task, sandbox)

    for directory in task.setup.directories:
        target_dir = _resolve_sandbox_path(sandbox, directory)
        target_dir.mkdir(parents=True, exist_ok=True)

    for setup_file in task.setup.files:
        target_file = _resolve_sandbox_path(sandbox, setup_file.path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(setup_file.content, encoding="utf-8")

    if task.setup.fixture_dir is not None:
        fixture_source = _resolve_fixture_source(task)
        shutil.copytree(fixture_source, sandbox, dirs_exist_ok=True)

    if task.setup.copy_repo:
        _initialize_sandbox_git_repo(sandbox)


def execute_task(task: EvalTask, sandbox: Path, *, cfg: AgentConfig) -> tuple[dict, WorkspaceSnapshot]:
    sandbox = sandbox.resolve()
    _apply_setup(task, sandbox)
    history: list[dict[str, object]] = [{"role": "system", "content": cfg.system_prompt}]
    api_responses: list[dict] = []
    tool_names = list(_tool_names_for_task(task))
    confirmer = agent.auto_approve_confirmer if task.requires_writes else agent.auto_deny_confirmer

    result = agent.run_agent_turn(
        task.prompt,
        history,
        api_responses,
        cfg=replace(cfg, working_dir=sandbox),
        renderer=NullRenderer(),
        confirmer=confirmer,
        tool_names=tool_names,
    )

    if result.status != agent.STATUS_OK:
        raise ValueError(f"agent turn failed with status={result.status}: {result.content}")

    atif = trajectory.history_to_atif(history, api_responses, cfg.model)
    return atif, _workspace_snapshot(sandbox)


def run_task(task: EvalTask, cache_dir: Path, *, cfg: AgentConfig) -> EvalResult:
    if task.skip:
        return EvalResult(
            task_id=task.id,
            description=task.description,
            status=STATUS_SKIPPED,
            error=task.skip,
        )

    try:
        execution_hash, hash_inputs = _execution_hash(task, cfg)
        cache_dir.mkdir(parents=True, exist_ok=True)
        task_cache_dir = cache_dir / f"{task.id}-{execution_hash}"
        trajectory_path = task_cache_dir / "trajectory.json"
        workspace_path = task_cache_dir / "workspace.json"
        metadata_path = task_cache_dir / "metadata.json"

        if task_cache_dir.exists():
            if not task_cache_dir.is_dir():
                raise ValueError(f"cache path is not a directory: {task_cache_dir}")
            atif = replay.load_trajectory(str(trajectory_path))
            workspace = _load_workspace_snapshot(workspace_path)
            _load_cache_metadata(metadata_path, expected_hash=execution_hash)
            checks = evaluate_rubric(task, atif, workspace)
            status = STATUS_PASS if all(check.passed for check in checks) else STATUS_FAIL
            return EvalResult(
                task_id=task.id,
                description=task.description,
                status=status,
                checks=checks,
                trajectory_path=str(trajectory_path),
                cached=True,
            )

        sandbox = cache_dir / f".eval-sandbox-{task.id}-{uuid.uuid4().hex}"
        try:
            atif, workspace = execute_task(task, sandbox, cfg=cfg)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

        checks = evaluate_rubric(task, atif, workspace)
        status = STATUS_PASS if all(check.passed for check in checks) else STATUS_FAIL

        temp_cache_dir = cache_dir / f".cache-write-{task.id}-{uuid.uuid4().hex}"
        temp_trajectory_path = temp_cache_dir / "trajectory.json"
        temp_workspace_path = temp_cache_dir / "workspace.json"
        temp_metadata_path = temp_cache_dir / "metadata.json"
        try:
            temp_cache_dir.mkdir(parents=False, exist_ok=False)
            trajectory.save_trajectory(atif, str(temp_trajectory_path))
            _save_workspace_snapshot(workspace, temp_workspace_path)
            _save_cache_metadata(temp_metadata_path, execution_hash=execution_hash, hash_inputs=hash_inputs)
            os.replace(temp_cache_dir, task_cache_dir)
        except Exception:
            shutil.rmtree(temp_cache_dir, ignore_errors=True)
            raise

        return EvalResult(
            task_id=task.id,
            description=task.description,
            status=status,
            checks=checks,
            trajectory_path=str(trajectory_path),
            cached=False,
        )
    except Exception as exc:  # noqa: BLE001
        return EvalResult(task_id=task.id, description=task.description, status=STATUS_ERROR, error=str(exc))


def run_eval_suite(
    tasks_path: str | Path,
    out_path: str | Path,
    json_out_path: str | Path,
    *,
    cfg: AgentConfig | None = None,
    workers: int = 1,
) -> int:
    if cfg is None:
        cfg = load_config()
    if workers < 1:
        print(f"eval: workers must be >= 1 (got {workers})", file=sys.stderr)
        return 1
    try:
        tasks = load_tasks(tasks_path)
    except (ValueError, OSError) as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 1

    cache_dir = Path(".baremetal-eval-cache")
    if workers == 1 or len(tasks) <= 1:
        results = [run_task(task, cache_dir, cfg=cfg) for task in tasks]
    else:
        # Parallel path. ``run_task`` already isolates per-task state via
        # ``ToolContext`` and an exception-free contract (errors are captured
        # into ``EvalResult`` rather than raised), so the orchestrator only
        # needs to guarantee deterministic ordering of the final results list.
        # Futures are collected and iterated in submission order so the report
        # is identical to the serial run modulo execution timing.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_task, task, cache_dir, cfg=cfg) for task in tasks]
            results = [future.result() for future in futures]

    markdown_report = render_markdown_report(results, cfg=cfg)
    json_report = render_json_report(results, cfg=cfg)

    try:
        _write_report_outputs(markdown_report, json_report, out_path, json_out_path)
    except ValueError as exc:
        print(f"eval: {exc}", file=sys.stderr)
        return 1

    has_fail_or_error = any(result.status in {STATUS_FAIL, STATUS_ERROR} for result in results)
    return 1 if has_fail_or_error else 0

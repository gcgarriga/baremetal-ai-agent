"""Cache key hashing, on-disk read/write, and shared sandbox/repo helpers.

Owns the digest computation and persistence layer for the eval cache, plus
sandbox-path resolution and fixture/repo helpers shared with the runner. The
hashing inputs in ``_execution_hash_inputs`` are the cache key contract — any
change here invalidates existing cache entries.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from baremetal_agent import __version__, replay, tools
from baremetal_agent.safety import sanitize_text

if TYPE_CHECKING:
    from baremetal_agent.config import AgentConfig

    from .loader import EvalTask

_EVAL_FILE_TRUNCATION_MARKER = "[truncated: eval_file exceeded "
_EVAL_FILE_TRUNCATION_PREFIX = f"\n\n{_EVAL_FILE_TRUNCATION_MARKER}"
_CACHE_SCHEMA_VERSION = "baremetal-eval-cache-v1"
_COPY_REPO_EXCLUDED_FILENAMES = {".env"}


@dataclass(frozen=True)
class FileSnapshot:
    exists: bool
    content: str | None = None
    truncated: bool = False


@dataclass(frozen=True)
class WorkspaceSnapshot:
    files: dict[str, FileSnapshot]


def _resolve_sandbox_path(base: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"path must be relative: {relative}")
    if any(part == ".." for part in relative_path.parts):
        raise ValueError(f"path must not contain '..': {relative}")

    resolved_base = base.resolve()
    resolved_candidate = (resolved_base / relative_path).resolve()
    try:
        resolved_candidate.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"path escapes sandbox: {relative}") from exc
    return resolved_candidate


def _run_git_command(cwd: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"could not run git {' '.join(args)}: {exc}") from exc

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise ValueError(f"git {' '.join(args)} failed: {message}")
    return result.stdout


def _resolve_repo_root(task: EvalTask) -> Path:
    if task.source_path is None:
        raise ValueError("setup.copy_repo requires source_path")

    source_parent = task.source_path.parent.resolve()
    output = _run_git_command(source_parent, ["rev-parse", "--show-toplevel"])
    return Path(output.strip()).resolve()


def _tracked_repo_files(task: EvalTask) -> tuple[tuple[Path, str], ...]:
    repo_root = _resolve_repo_root(task)
    output = _run_git_command(repo_root, ["ls-files", "-z"])
    tracked_paths = [path for path in output.split("\0") if path]
    files: list[tuple[Path, str]] = []

    for relative in tracked_paths:
        relative_path = Path(relative)
        if relative_path.name in _COPY_REPO_EXCLUDED_FILENAMES:
            continue
        source_file = _resolve_sandbox_path(repo_root, relative)
        if source_file.is_symlink():
            raise ValueError(f"setup.copy_repo tracked file must not be a symlink: {relative}")
        if not source_file.is_file():
            continue
        files.append((source_file, Path(relative).as_posix()))

    if not files:
        raise ValueError("setup.copy_repo found no tracked files to copy")

    return tuple(files)


def _copy_repo_to_sandbox(task: EvalTask, sandbox: Path) -> None:
    for source_file, relative in _tracked_repo_files(task):
        target_file = _resolve_sandbox_path(sandbox, relative)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)


def _initialize_sandbox_git_repo(sandbox: Path) -> None:
    _run_git_command(sandbox, ["init", "--quiet"])
    disabled_hooks_dir = sandbox / ".git" / "hooks-disabled"
    disabled_hooks_dir.mkdir()
    _run_git_command(sandbox, ["config", "core.hooksPath", str(disabled_hooks_dir)])
    _run_git_command(sandbox, ["config", "user.name", "Baremetal Eval"])
    _run_git_command(sandbox, ["config", "user.email", "baremetal-eval@example.invalid"])
    _run_git_command(sandbox, ["add", "-A"])
    _run_git_command(sandbox, ["commit", "--quiet", "--no-verify", "--no-gpg-sign", "-m", "eval sandbox seed"])


def _resolve_fixture_source(task: EvalTask) -> Path:
    fixture_dir = task.setup.fixture_dir
    if fixture_dir is None:
        raise ValueError("task fixture_dir is not configured")
    if task.source_path is None:
        raise ValueError("task fixture_dir requires source_path")

    task_parent = task.source_path.parent
    fixture_source = _resolve_sandbox_path(task_parent, fixture_dir)
    if not fixture_source.exists():
        raise ValueError(f"fixture_dir not found: {fixture_dir}")
    if not fixture_source.is_dir():
        raise ValueError(f"fixture_dir must be a directory: {fixture_dir}")
    if fixture_source.is_symlink():
        raise ValueError(f"fixture_dir must not be a symlink: {fixture_dir}")

    for fixture_path in fixture_source.rglob("*"):
        if fixture_path.is_symlink():
            raise ValueError(f"fixture_dir contains symlink: {fixture_path.relative_to(task_parent)}")

    return fixture_source


def _workspace_snapshot(base: Path) -> WorkspaceSnapshot:
    files: dict[str, FileSnapshot] = {}
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
        if ".git" in path.relative_to(base).parts:
            continue
        if path.is_symlink() or not path.is_file():
            continue

        relative_path = path.relative_to(base).as_posix()
        content = path.read_text(encoding="utf-8", errors="replace")
        sanitized_content = sanitize_text(content, label="eval_file")
        files[relative_path] = FileSnapshot(
            exists=True,
            content=sanitized_content,
            truncated=_EVAL_FILE_TRUNCATION_PREFIX in sanitized_content,
        )

    return WorkspaceSnapshot(files=files)


def _tool_names_for_task(task: EvalTask) -> tuple[str, ...]:
    if task.requires_writes:
        names = [name for name in tools.get_read_only_tool_names() if name != "shell_exec"]
        if "write_file" not in names:
            names.append("write_file")
    else:
        names = list(tools.get_read_only_tool_names())
    return tuple(names)


def _normalize_setup(task: EvalTask) -> dict:
    return {
        "directories": [Path(directory).as_posix() for directory in task.setup.directories],
        "files": [
            {"path": Path(setup_file.path).as_posix(), "content": setup_file.content} for setup_file in task.setup.files
        ],
        "fixture_dir": Path(task.setup.fixture_dir).as_posix() if task.setup.fixture_dir is not None else None,
        "copy_repo": task.setup.copy_repo,
    }


def _compute_fixture_digest(task: EvalTask) -> str | None:
    if task.setup.fixture_dir is None:
        return None

    fixture_source = _resolve_fixture_source(task)
    digest = hashlib.sha256()
    file_paths = sorted(
        (path for path in fixture_source.rglob("*") if not path.is_symlink() and path.is_file()),
        key=lambda path: path.relative_to(fixture_source).as_posix(),
    )
    for file_path in file_paths:
        relative = file_path.relative_to(fixture_source).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        try:
            digest.update(file_path.read_bytes())
        except OSError as exc:
            raise ValueError(f"could not read fixture file {relative}: {exc}") from exc
        digest.update(b"\x00")
    return digest.hexdigest()


def _compute_repo_digest(task: EvalTask) -> str | None:
    if not task.setup.copy_repo:
        return None

    digest = hashlib.sha256()
    for source_file, relative in _tracked_repo_files(task):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        try:
            digest.update(source_file.read_bytes())
        except OSError as exc:
            raise ValueError(f"could not read tracked repo file {relative}: {exc}") from exc
        digest.update(b"\x00")
    return digest.hexdigest()


def _execution_hash_inputs(task: EvalTask, cfg: AgentConfig) -> dict:
    system_prompt_hash = hashlib.sha256(cfg.system_prompt.encode("utf-8")).hexdigest()
    return {
        "cache_schema": _CACHE_SCHEMA_VERSION,
        "task_id": task.id,
        "prompt": task.prompt,
        "setup": _normalize_setup(task),
        "fixture_digest": _compute_fixture_digest(task),
        "repo_digest": _compute_repo_digest(task),
        "requires_writes": task.requires_writes,
        "model": cfg.model,
        "max_iterations": cfg.max_iterations,
        "tool_names": sorted(_tool_names_for_task(task)),
        "system_prompt_sha256": system_prompt_hash,
        "agent_version": __version__,
        "atif_schema_version": replay.SCHEMA_VERSION,
    }


def _execution_hash(task: EvalTask, cfg: AgentConfig) -> tuple[str, dict]:
    hash_inputs = _execution_hash_inputs(task, cfg)
    payload = json.dumps(hash_inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), hash_inputs


def _save_workspace_snapshot(workspace: WorkspaceSnapshot, path: Path) -> None:
    entries = [
        {
            "path": file_path,
            "exists": snapshot.exists,
            "content": snapshot.content,
            "truncated": snapshot.truncated,
        }
        for file_path, snapshot in sorted(workspace.files.items())
    ]
    payload = {"files": entries}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_workspace_snapshot(path: Path) -> WorkspaceSnapshot:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read cached workspace {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in cached workspace {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("cached workspace payload must be an object")
    files_raw = payload.get("files")
    if not isinstance(files_raw, list):
        raise ValueError("cached workspace field 'files' must be a list")

    files: dict[str, FileSnapshot] = {}
    for index, raw_file in enumerate(files_raw):
        if not isinstance(raw_file, dict):
            raise ValueError(f"cached workspace files[{index}] must be an object")
        path_value = raw_file.get("path")
        exists_value = raw_file.get("exists")
        content_value = raw_file.get("content")
        truncated_value = raw_file.get("truncated", False)
        if not isinstance(path_value, str):
            raise ValueError(f"cached workspace files[{index}].path must be a string")
        if not isinstance(exists_value, bool):
            raise ValueError(f"cached workspace files[{index}].exists must be a boolean")
        if content_value is not None and not isinstance(content_value, str):
            raise ValueError(f"cached workspace files[{index}].content must be a string or null")
        if not isinstance(truncated_value, bool):
            raise ValueError(f"cached workspace files[{index}].truncated must be a boolean")
        normalized_path = Path(path_value).as_posix()
        files[normalized_path] = FileSnapshot(exists=exists_value, content=content_value, truncated=truncated_value)

    return WorkspaceSnapshot(files=files)


def _save_cache_metadata(path: Path, *, execution_hash: str, hash_inputs: dict) -> None:
    payload = {
        "cache_schema": _CACHE_SCHEMA_VERSION,
        "execution_hash": execution_hash,
        "execution_hash_inputs": hash_inputs,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _load_cache_metadata(path: Path, *, expected_hash: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read cache metadata {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in cache metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("cache metadata must be an object")
    if payload.get("cache_schema") != _CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported cache schema in metadata: {payload.get('cache_schema')!r}")
    if payload.get("execution_hash") != expected_hash:
        raise ValueError("cache metadata execution_hash mismatch")
    return payload

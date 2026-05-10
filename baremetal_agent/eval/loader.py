"""Eval task discovery, JSON parsing, and dataclass types."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .rubric import RUBRIC_TYPES

_ALLOWED_TOP_LEVEL_KEYS = {"id", "description", "prompt", "setup", "rubric", "requires_writes", "skip"}
_REQUIRED_TOP_LEVEL_KEYS = {"id", "description", "prompt", "rubric"}
_ALLOWED_SETUP_KEYS = {"directories", "files", "fixture_dir", "copy_repo"}
_TASK_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class SetupFile:
    path: str
    content: str


@dataclass(frozen=True)
class TaskSetup:
    directories: tuple[str, ...] = ()
    files: tuple[SetupFile, ...] = ()
    fixture_dir: str | None = None
    copy_repo: bool = False


@dataclass(frozen=True)
class RubricCheck:
    type: str
    params: Mapping[str, object]


@dataclass(frozen=True)
class EvalTask:
    id: str
    description: str
    prompt: str
    rubric: tuple[RubricCheck, ...]
    setup: TaskSetup
    requires_writes: bool = False
    skip: str | None = None
    source_path: Path | None = None


def load_tasks(path: str | Path) -> list[EvalTask]:
    """Load one or more eval task files from a file path or directory.

    Raises ``ValueError`` if two task JSONs share the same ``id``. Duplicate
    IDs are disallowed because they are ambiguous in reporting and, if the
    bodies also produce identical ``execution_hash`` values, they would
    collide on the on-disk cache key (``cache_dir / f"{task.id}-{hash}"``)
    and race when ``run_task`` calls execute concurrently.
    """
    source = Path(path)

    if source.is_file():
        return [load_task(source)]
    if source.is_dir():
        json_files = sorted(source.glob("*.json"), key=lambda file_path: (file_path.name, str(file_path)))
        if not json_files:
            raise ValueError(f"no eval task JSON files found in directory: {source}")
        tasks = [load_task(file_path) for file_path in json_files]
        _reject_duplicate_ids(tasks)
        return tasks

    raise ValueError(f"task path must be a file or directory: {source}")


def _reject_duplicate_ids(tasks: list[EvalTask]) -> None:
    seen: dict[str, Path | None] = {}
    for task in tasks:
        if task.id in seen:
            first = seen[task.id]
            second = task.source_path
            raise ValueError(f"duplicate eval task id {task.id!r}: defined in {first} and {second}")
        seen[task.id] = task.source_path


def load_task(path: str | Path) -> EvalTask:
    """Load and validate a single eval task JSON file."""
    source = Path(path)

    try:
        with source.open(encoding="utf-8") as file_handle:
            raw = json.load(file_handle)
    except OSError as exc:
        raise ValueError(f"could not read task file {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in task file {source}: {exc}") from exc

    return validate_task(raw, source)


def validate_task(raw: object, source_path: Path) -> EvalTask:
    """Validate raw task payload and return an EvalTask dataclass."""
    if not isinstance(raw, dict):
        raise ValueError(f"task {source_path} must be a JSON object")

    unknown_keys = sorted(set(raw) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown_keys:
        raise ValueError(f"task {source_path} has unknown top-level key(s): {', '.join(unknown_keys)}")

    for required in sorted(_REQUIRED_TOP_LEVEL_KEYS):
        if required not in raw:
            raise ValueError(f"task {source_path} is missing required field: {required}")

    task_id = raw["id"]
    if not isinstance(task_id, str):
        raise ValueError(f"task {source_path} field 'id' must be a string")
    if not _TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(f"task {source_path} field 'id' must match [a-z0-9][a-z0-9_-]*")

    description = raw["description"]
    if not isinstance(description, str):
        raise ValueError(f"task {source_path} field 'description' must be a string")

    prompt = raw["prompt"]
    if not isinstance(prompt, str):
        raise ValueError(f"task {source_path} field 'prompt' must be a string")

    requires_writes = raw.get("requires_writes", False)
    if not isinstance(requires_writes, bool):
        raise ValueError(f"task {source_path} field 'requires_writes' must be a boolean")

    skip = raw.get("skip")
    if skip is not None and not isinstance(skip, str):
        raise ValueError(f"task {source_path} field 'skip' must be a string or null")

    setup = _parse_setup(raw.get("setup"), source_path)
    rubric = _parse_rubric(raw["rubric"], source_path)

    return EvalTask(
        id=task_id,
        description=description,
        prompt=prompt,
        rubric=rubric,
        setup=setup,
        requires_writes=requires_writes,
        skip=skip,
        source_path=source_path,
    )


def _parse_setup(raw_setup: object, source_path: Path) -> TaskSetup:
    if raw_setup is None:
        return TaskSetup()
    if not isinstance(raw_setup, dict):
        raise ValueError(f"task {source_path} field 'setup' must be an object")

    unknown_keys = sorted(set(raw_setup) - _ALLOWED_SETUP_KEYS)
    if unknown_keys:
        raise ValueError(f"task {source_path} setup has unknown key(s): {', '.join(unknown_keys)}")

    directories_raw = raw_setup.get("directories", [])
    if not isinstance(directories_raw, list):
        raise ValueError(f"task {source_path} setup.directories must be a list of strings")
    directories: list[str] = []
    for index, value in enumerate(directories_raw):
        if not isinstance(value, str):
            raise ValueError(f"task {source_path} setup.directories[{index}] must be a string")
        _validate_setup_path(value, source_path=source_path, field=f"setup.directories[{index}]")
        directories.append(value)

    files_raw = raw_setup.get("files", [])
    if not isinstance(files_raw, list):
        raise ValueError(f"task {source_path} setup.files must be a list of objects")
    files: list[SetupFile] = []
    for index, file_item in enumerate(files_raw):
        if not isinstance(file_item, dict):
            raise ValueError(f"task {source_path} setup.files[{index}] must be an object")
        unknown_file_keys = sorted(set(file_item) - {"path", "content"})
        if unknown_file_keys:
            joined = ", ".join(unknown_file_keys)
            raise ValueError(f"task {source_path} setup.files[{index}] has unknown key(s): {joined}")
        if "path" not in file_item:
            raise ValueError(f"task {source_path} setup.files[{index}] is missing required field: path")
        if "content" not in file_item:
            raise ValueError(f"task {source_path} setup.files[{index}] is missing required field: content")

        file_path = file_item["path"]
        content = file_item["content"]
        if not isinstance(file_path, str):
            raise ValueError(f"task {source_path} setup.files[{index}].path must be a string")
        if not isinstance(content, str):
            raise ValueError(f"task {source_path} setup.files[{index}].content must be a string")
        _validate_setup_path(file_path, source_path=source_path, field=f"setup.files[{index}].path")
        files.append(SetupFile(path=file_path, content=content))

    fixture_dir = raw_setup.get("fixture_dir")
    if fixture_dir is not None and not isinstance(fixture_dir, str):
        raise ValueError(f"task {source_path} setup.fixture_dir must be a string or null")
    if fixture_dir is not None:
        _validate_setup_path(fixture_dir, source_path=source_path, field="setup.fixture_dir")

    copy_repo = raw_setup.get("copy_repo", False)
    if not isinstance(copy_repo, bool):
        raise ValueError(f"task {source_path} setup.copy_repo must be a boolean")

    return TaskSetup(
        directories=tuple(directories),
        files=tuple(files),
        fixture_dir=fixture_dir,
        copy_repo=copy_repo,
    )


def _parse_rubric(raw_rubric: object, source_path: Path) -> tuple[RubricCheck, ...]:
    if not isinstance(raw_rubric, list) or not raw_rubric:
        raise ValueError(f"task {source_path} field 'rubric' must be a non-empty list")

    checks: list[RubricCheck] = []
    for index, raw_check in enumerate(raw_rubric):
        if not isinstance(raw_check, dict):
            raise ValueError(f"task {source_path} rubric[{index}] must be an object")

        check_type = raw_check.get("type")
        if not isinstance(check_type, str):
            raise ValueError(f"task {source_path} rubric[{index}].type must be a string")
        if check_type not in RUBRIC_TYPES:
            raise ValueError(f"task {source_path} rubric[{index}] has invalid type: {check_type}")

        params = MappingProxyType(dict((key, value) for key, value in raw_check.items() if key != "type"))
        checks.append(RubricCheck(type=check_type, params=params))

    return tuple(checks)


def _validate_setup_path(path_value: str, *, source_path: Path, field: str) -> None:
    candidate = Path(path_value)
    if candidate.is_absolute():
        raise ValueError(f"task {source_path} {field} must be a relative path")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"task {source_path} {field} must not contain '..'")

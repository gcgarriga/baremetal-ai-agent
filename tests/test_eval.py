import json
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from baremetal_agent import agent, tools
from baremetal_agent import eval as eval_module
from baremetal_agent.eval import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED,
    CheckResult,
    EvalResult,
    EvalTask,
    FileSnapshot,
    RubricCheck,
    SetupFile,
    TaskSetup,
    WorkspaceSnapshot,
    evaluate_rubric,
    execute_task,
    load_tasks,
    render_json_report,
    render_markdown_report,
    run_eval_suite,
    run_task,
)
from baremetal_agent.eval.cache import _resolve_sandbox_path, _workspace_snapshot
from baremetal_agent.eval.runner import _apply_setup
from baremetal_agent.safety import DEFAULT_MAX_CHARS
from baremetal_agent.visualizer import NullRenderer


@pytest.fixture
def cfg(make_cfg):
    return make_cfg()


def _task_payload(**overrides):
    payload = {
        "id": "task-1",
        "description": "A test task",
        "prompt": "Do the thing.",
        "rubric": [{"type": "tool_called", "name": "read_file"}],
    }
    payload.update(overrides)
    return payload


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _require_git() -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is required for repository-copy eval tests")
    return git


def _run_git(repo: Path, *args: str) -> str:
    git = _require_git()
    result = subprocess.run(
        [git, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _init_repo_fixture(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _run_git(repo, "init", "--quiet")
    _run_git(repo, "config", "user.name", "Eval Test")
    _run_git(repo, "config", "user.email", "eval-test@example.invalid")
    (repo / "evals" / "tasks").mkdir(parents=True)
    return repo


def _sample_atif(*, message="done") -> dict:
    return {
        "schema_version": "ATIF-v1.4",
        "session_id": "session-1",
        "agent": {"name": "baremetal-agent", "version": "0.1.0", "model_name": "model"},
        "steps": [{"step_id": 1, "timestamp": "2025-01-01T00:00:00+00:00", "source": "agent", "message": message}],
        "final_metrics": {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_steps": 1,
        },
    }


def test_valid_task_json_loads_with_defaults(tmp_path):
    path = tmp_path / "task.json"
    _write_json(path, _task_payload())

    tasks = load_tasks(path)

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, EvalTask)
    assert task.id == "task-1"
    assert task.setup == TaskSetup()
    assert task.requires_writes is False
    assert task.skip is None
    assert task.source_path == path


def test_directory_load_sorts_json_and_ignores_non_json(tmp_path):
    _write_json(tmp_path / "b.json", _task_payload(id="b-task"))
    _write_json(tmp_path / "a.json", _task_payload(id="a-task"))
    (tmp_path / "ignore.txt").write_text("not json", encoding="utf-8")

    tasks = load_tasks(tmp_path)

    assert [task.id for task in tasks] == ["a-task", "b-task"]


def test_empty_task_directory_is_rejected(tmp_path):
    (tmp_path / "ignore.txt").write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="no eval task JSON files"):
        load_tasks(tmp_path)


def test_duplicate_task_ids_are_rejected(tmp_path):
    """Duplicate task IDs are ambiguous in reporting and, when the bodies
    hash identically, can also collide on ``cache_dir / f"{task.id}-{hash}"``
    and race ``os.replace`` under concurrent eval runs. ``load_tasks`` must
    surface duplicates upfront."""
    _write_json(tmp_path / "a.json", _task_payload(id="dup"))
    _write_json(tmp_path / "b.json", _task_payload(id="dup"))

    with pytest.raises(ValueError, match="duplicate eval task id"):
        load_tasks(tmp_path)


def test_loads_all_starter_eval_tasks_from_repository():
    tasks = load_tasks(Path(__file__).resolve().parents[1] / "evals" / "tasks")

    assert len(tasks) == 7
    assert {task.id for task in tasks} == {
        "architecture-summary",
        "tool-registry-location",
        "config-env-vars",
        "one-shot-safety-policy",
        "atif-trajectory-format",
        "write-exact-result-file",
        "git-status-without-shell",
    }


def test_starter_repo_inspection_tasks_copy_controlled_repo():
    tasks = {task.id: task for task in load_tasks(Path(__file__).resolve().parents[1] / "evals" / "tasks")}

    assert tasks["write-exact-result-file"].setup.copy_repo is False
    for task_id in {
        "architecture-summary",
        "atif-trajectory-format",
        "config-env-vars",
        "git-status-without-shell",
        "one-shot-safety-policy",
        "tool-registry-location",
    }:
        assert tasks[task_id].setup.copy_repo is True


def test_starter_repo_inspection_tasks_do_not_require_read_file_tool_choice():
    tasks = {task.id: task for task in load_tasks(Path(__file__).resolve().parents[1] / "evals" / "tasks")}
    behavior_tasks = {
        "architecture-summary",
        "atif-trajectory-format",
        "config-env-vars",
        "one-shot-safety-policy",
        "tool-registry-location",
    }

    for task_id in behavior_tasks:
        read_file_requirements = [
            check
            for check in tasks[task_id].rubric
            if check.type == "tool_called" and check.params.get("name") == "read_file"
        ]
        assert read_file_requirements == []


@pytest.mark.parametrize("field", ["id", "description", "prompt", "rubric"])
def test_missing_required_fields_raise_value_error(tmp_path, field):
    payload = _task_payload()
    payload.pop(field)
    _write_json(tmp_path / "task.json", payload)

    with pytest.raises(ValueError, match=field):
        load_tasks(tmp_path / "task.json")


def test_invalid_task_id_rejected(tmp_path):
    _write_json(tmp_path / "task.json", _task_payload(id="Invalid Id"))

    with pytest.raises(ValueError, match="id"):
        load_tasks(tmp_path / "task.json")


def test_invalid_rubric_type_rejected(tmp_path):
    _write_json(
        tmp_path / "task.json",
        _task_payload(rubric=[{"type": "does_not_exist"}]),
    )

    with pytest.raises(ValueError, match="rubric"):
        load_tasks(tmp_path / "task.json")


def test_empty_rubric_list_rejected(tmp_path):
    _write_json(tmp_path / "task.json", _task_payload(rubric=[]))

    with pytest.raises(ValueError, match="rubric"):
        load_tasks(tmp_path / "task.json")


def test_unknown_top_level_key_rejected(tmp_path):
    _write_json(tmp_path / "task.json", _task_payload(unexpected=123))

    with pytest.raises(ValueError, match="unexpected"):
        load_tasks(tmp_path / "task.json")


def test_setup_files_and_directories_parse(tmp_path):
    payload = _task_payload(
        setup={
            "directories": ["workspace", "workspace/src"],
            "files": [
                {"path": "workspace/src/main.py", "content": "print('ok')\n"},
                {"path": "workspace/README.md", "content": "# fixture\n"},
            ],
            "fixture_dir": "fixtures/basic",
            "copy_repo": True,
        }
    )
    _write_json(tmp_path / "task.json", payload)

    task = load_tasks(tmp_path / "task.json")[0]

    assert task.setup == TaskSetup(
        directories=("workspace", "workspace/src"),
        files=(
            SetupFile(path="workspace/src/main.py", content="print('ok')\n"),
            SetupFile(path="workspace/README.md", content="# fixture\n"),
        ),
        fixture_dir="fixtures/basic",
        copy_repo=True,
    )


@pytest.mark.parametrize("directory", ["/absolute/path", "workspace/../escape"])
def test_setup_directories_reject_absolute_and_traversal(tmp_path, directory):
    _write_json(tmp_path / "task.json", _task_payload(setup={"directories": [directory]}))

    with pytest.raises(ValueError, match="setup.directories"):
        load_tasks(tmp_path / "task.json")


@pytest.mark.parametrize("file_path", ["/absolute/file.txt", "workspace/../escape.txt"])
def test_setup_file_paths_reject_absolute_and_traversal(tmp_path, file_path):
    _write_json(
        tmp_path / "task.json",
        _task_payload(setup={"files": [{"path": file_path, "content": "data"}]}),
    )

    with pytest.raises(ValueError, match="setup.files"):
        load_tasks(tmp_path / "task.json")


@pytest.mark.parametrize("fixture_dir", ["/absolute/fixture", "fixtures/../escape"])
def test_setup_fixture_dir_rejects_absolute_and_traversal(tmp_path, fixture_dir):
    _write_json(tmp_path / "task.json", _task_payload(setup={"fixture_dir": fixture_dir}))

    with pytest.raises(ValueError, match="setup.fixture_dir"):
        load_tasks(tmp_path / "task.json")


def test_setup_copy_repo_must_be_boolean(tmp_path):
    _write_json(tmp_path / "task.json", _task_payload(setup={"copy_repo": "yes"}))

    with pytest.raises(ValueError, match="setup.copy_repo"):
        load_tasks(tmp_path / "task.json")


def test_rubric_params_are_immutable(tmp_path):
    _write_json(tmp_path / "task.json", _task_payload(rubric=[{"type": "tool_called", "name": "read_file"}]))

    task = load_tasks(tmp_path / "task.json")[0]

    with pytest.raises(TypeError):
        task.rubric[0].params["name"] = "write_file"


def test_apply_setup_creates_declared_directories_and_files(tmp_path):
    task = EvalTask(
        id="task-setup",
        description="setup test",
        prompt="prompt",
        rubric=(),
        setup=TaskSetup(
            directories=("workspace/src", "workspace/tests"),
            files=(
                SetupFile(path="workspace/src/main.py", content="print('hi')\n"),
                SetupFile(path="workspace/tests/test_main.py", content="assert True\n"),
            ),
        ),
    )
    sandbox = tmp_path / "sandbox"

    _apply_setup(task, sandbox)

    assert (sandbox / "workspace/src").is_dir()
    assert (sandbox / "workspace/tests").is_dir()
    assert (sandbox / "workspace/src/main.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert (sandbox / "workspace/tests/test_main.py").read_text(encoding="utf-8") == "assert True\n"


@pytest.mark.parametrize("relative", ["/abs/path", "../escape", "safe/../../escape"])
def test_resolve_sandbox_path_rejects_escape_and_absolute(tmp_path, relative):
    with pytest.raises(ValueError):
        _resolve_sandbox_path(tmp_path, relative)


def test_apply_setup_copies_fixture_dir_relative_to_task_source(tmp_path):
    task_file = tmp_path / "tasks" / "task.json"
    task_file.parent.mkdir(parents=True)
    fixture_root = task_file.parent / "fixtures" / "basic"
    fixture_root.mkdir(parents=True)
    (fixture_root / "README.txt").write_text("fixture\n", encoding="utf-8")
    (fixture_root / "nested").mkdir()
    (fixture_root / "nested" / "file.txt").write_text("nested\n", encoding="utf-8")
    sandbox = tmp_path / "sandbox"

    task = EvalTask(
        id="task-fixture",
        description="fixture copy",
        prompt="prompt",
        rubric=(),
        setup=TaskSetup(fixture_dir="fixtures/basic"),
        source_path=task_file,
    )

    _apply_setup(task, sandbox)

    assert (sandbox / "README.txt").read_text(encoding="utf-8") == "fixture\n"
    assert (sandbox / "nested/file.txt").read_text(encoding="utf-8") == "nested\n"


def test_apply_setup_rejects_fixture_source_escaping_task_parent(tmp_path):
    task_file = tmp_path / "tasks" / "task.json"
    task_file.parent.mkdir(parents=True)
    (tmp_path / "outside").mkdir()

    task = EvalTask(
        id="task-fixture-escape",
        description="fixture escape",
        prompt="prompt",
        rubric=(),
        setup=TaskSetup(fixture_dir="../outside"),
        source_path=task_file,
    )

    with pytest.raises(ValueError):
        _apply_setup(task, tmp_path / "sandbox")


def test_apply_setup_rejects_symlink_in_fixture_source(tmp_path):
    task_file = tmp_path / "tasks" / "task.json"
    task_file.parent.mkdir(parents=True)
    fixture_root = task_file.parent / "fixtures" / "basic"
    fixture_root.mkdir(parents=True)
    real_file = fixture_root / "real.txt"
    real_file.write_text("real\n", encoding="utf-8")

    symlink_path = fixture_root / "link.txt"
    try:
        symlink_path.symlink_to(real_file)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks not supported in this environment")

    task = EvalTask(
        id="task-fixture-symlink",
        description="fixture symlink",
        prompt="prompt",
        rubric=(),
        setup=TaskSetup(fixture_dir="fixtures/basic"),
        source_path=task_file,
    )

    with pytest.raises(ValueError, match="symlink"):
        _apply_setup(task, tmp_path / "sandbox")


def test_apply_setup_copy_repo_copies_tracked_files_to_clean_git_sandbox(tmp_path):
    repo = _init_repo_fixture(tmp_path)
    task_file = repo / "evals" / "tasks" / "task.json"
    task_file.write_text("{}", encoding="utf-8")
    (repo / "README.md").write_text("# Project\n", encoding="utf-8")
    (repo / "baremetal_agent").mkdir()
    (repo / "baremetal_agent" / "config.py").write_text("AGENT_MODEL = 'model'\n", encoding="utf-8")
    (repo / "untracked-secret.txt").write_text("do not copy\n", encoding="utf-8")
    _run_git(repo, "add", "README.md", "baremetal_agent/config.py", "evals/tasks/task.json")

    task = EvalTask(
        id="task-copy-repo",
        description="copy repo",
        prompt="prompt",
        rubric=(),
        setup=TaskSetup(copy_repo=True),
        source_path=task_file,
    )
    sandbox = tmp_path / "sandbox"

    _apply_setup(task, sandbox)

    assert (sandbox / "README.md").read_text(encoding="utf-8") == "# Project\n"
    assert (sandbox / "baremetal_agent" / "config.py").read_text(encoding="utf-8") == "AGENT_MODEL = 'model'\n"
    assert not (sandbox / "untracked-secret.txt").exists()
    assert _run_git(sandbox, "status", "--short") == ""
    assert all(".git/" not in path for path in _workspace_snapshot(sandbox).files)


def test_apply_setup_copy_repo_excludes_tracked_env_file_from_sandbox(tmp_path):
    repo = _init_repo_fixture(tmp_path)
    task_file = repo / "evals" / "tasks" / "task.json"
    task_file.write_text("{}", encoding="utf-8")
    (repo / "README.md").write_text("# Project\n", encoding="utf-8")
    (repo / ".env").write_text("GITHUB_TOKEN=ghp_abcdef123456\n", encoding="utf-8")
    _run_git(repo, "add", "README.md", ".env", "evals/tasks/task.json")

    task = EvalTask(
        id="task-copy-repo-env",
        description="copy repo excludes env",
        prompt="prompt",
        rubric=(),
        setup=TaskSetup(copy_repo=True),
        source_path=task_file,
    )
    sandbox = tmp_path / "sandbox"

    _apply_setup(task, sandbox)

    assert not (sandbox / ".env").exists()
    assert ".env" not in _workspace_snapshot(sandbox).files
    assert _run_git(sandbox, "status", "--short") == ""


def test_apply_setup_copy_repo_commits_declarative_setup_to_clean_git_sandbox(tmp_path):
    repo = _init_repo_fixture(tmp_path)
    task_file = repo / "evals" / "tasks" / "task.json"
    task_file.write_text("{}", encoding="utf-8")
    (repo / "README.md").write_text("# Project\n", encoding="utf-8")
    _run_git(repo, "add", "README.md", "evals/tasks/task.json")

    task = EvalTask(
        id="task-copy-repo-with-file",
        description="copy repo with file",
        prompt="prompt",
        rubric=(),
        setup=TaskSetup(
            files=(SetupFile(path="inputs/question.txt", content="what changed?\n"),),
            copy_repo=True,
        ),
        source_path=task_file,
    )
    sandbox = tmp_path / "sandbox"

    _apply_setup(task, sandbox)

    assert (sandbox / "inputs" / "question.txt").read_text(encoding="utf-8") == "what changed?\n"
    assert _run_git(sandbox, "status", "--short") == ""


def test_apply_setup_copy_repo_disables_user_configured_git_hooks(tmp_path, monkeypatch):
    repo = _init_repo_fixture(tmp_path)
    task_file = repo / "evals" / "tasks" / "task.json"
    task_file.write_text("{}", encoding="utf-8")
    (repo / "README.md").write_text("# Project\n", encoding="utf-8")
    _run_git(repo, "add", "README.md", "evals/tasks/task.json")

    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    pre_commit = hooks_dir / "pre-commit"
    pre_commit.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    pre_commit.chmod(0o755)
    global_config = tmp_path / "global-gitconfig"
    global_config.write_text(f"[core]\n\thooksPath = {hooks_dir.as_posix()}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

    task = EvalTask(
        id="task-copy-repo-hooks",
        description="copy repo ignores hooks",
        prompt="prompt",
        rubric=(),
        setup=TaskSetup(copy_repo=True),
        source_path=task_file,
    )
    sandbox = tmp_path / "sandbox"

    _apply_setup(task, sandbox)

    assert _run_git(sandbox, "status", "--short") == ""


def test_workspace_snapshot_includes_regular_files_with_posix_relative_paths(tmp_path):
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "file.txt").write_text("hello\n", encoding="utf-8")

    snapshot = _workspace_snapshot(tmp_path)

    assert snapshot.files["dir/file.txt"].exists is True
    assert snapshot.files["dir/file.txt"].content == "hello\n"
    assert snapshot.files["dir/file.txt"].truncated is False


def test_workspace_snapshot_skips_symlinks(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("target\n", encoding="utf-8")
    symlink = tmp_path / "link.txt"
    try:
        symlink.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks not supported in this environment")

    snapshot = _workspace_snapshot(tmp_path)

    assert "target.txt" in snapshot.files
    assert "link.txt" not in snapshot.files


def test_workspace_snapshot_redacts_and_marks_truncated_content(tmp_path):
    large_secret = "token=supersecret" + ("a" * (DEFAULT_MAX_CHARS + 100))
    (tmp_path / "large.txt").write_text(large_secret, encoding="utf-8")

    snapshot = _workspace_snapshot(tmp_path)
    file_snapshot = snapshot.files["large.txt"]

    assert file_snapshot.exists is True
    assert file_snapshot.content is not None
    assert "token=supersecret" not in file_snapshot.content
    assert "[truncated: eval_file exceeded " in file_snapshot.content
    assert file_snapshot.truncated is True


def test_workspace_snapshot_literal_marker_substring_is_not_truncated(tmp_path):
    literal = "note: [truncated: eval_file exceeded 999 chars; omitted 100 chars]"
    (tmp_path / "literal.txt").write_text(literal, encoding="utf-8")

    snapshot = _workspace_snapshot(tmp_path)
    file_snapshot = snapshot.files["literal.txt"]

    assert file_snapshot.content == literal
    assert file_snapshot.truncated is False


def _eval_task_with_rubric(*rubric: dict[str, object]) -> EvalTask:
    return EvalTask(
        id="task-rubric",
        description="rubric test",
        prompt="prompt",
        rubric=tuple(
            RubricCheck(type=check["type"], params={k: v for k, v in check.items() if k != "type"}) for check in rubric
        ),
        setup=TaskSetup(),
    )


def test_final_response_contains_passes_for_substring_and_regex():
    task = _eval_task_with_rubric(
        {"type": "final_response_contains", "contains": "great"},
        {"type": "final_response_contains", "regex": r"done\.$"},
    )
    atif = {"steps": [{"source": "agent", "message": "That looks great, done."}]}

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert len(results) == 2
    assert all(result.passed for result in results)


def test_final_response_contains_fails_when_message_does_not_match():
    task = _eval_task_with_rubric({"type": "final_response_contains", "contains": "missing"})
    atif = {"steps": [{"source": "agent", "message": "no match here"}]}

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert len(results) == 1
    assert results[0].passed is False


def test_final_response_contains_invalid_regex_fails_with_message():
    task = _eval_task_with_rubric({"type": "final_response_contains", "regex": "("})
    atif = {"steps": [{"source": "agent", "message": "anything"}]}

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert results[0].passed is False
    assert "invalid regex" in results[0].message.lower()


def test_trajectory_atif_passes_from_artifact_without_final_metrics_prose():
    task = _eval_task_with_rubric(
        {"type": "trajectory_atif", "schema_version": "ATIF-v1.4", "require_final_metrics": True}
    )
    atif = _sample_atif(message="The exported trajectory uses ATIF-v1.4.")

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert len(results) == 1
    assert results[0].passed is True


@pytest.mark.parametrize(("schema_version", "expected_actual"), [("ATIF-v1.3", "ATIF-v1.3"), (None, "(missing)")])
def test_trajectory_atif_schema_mismatch_fails_with_useful_message(schema_version, expected_actual):
    task = _eval_task_with_rubric(
        {"type": "trajectory_atif", "schema_version": "ATIF-v1.4", "require_final_metrics": True}
    )
    atif = _sample_atif()
    if schema_version is None:
        del atif["schema_version"]
    else:
        atif["schema_version"] = schema_version

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert results[0].passed is False
    assert "schema_version" in results[0].message
    assert "expected ATIF-v1.4" in results[0].message
    assert f"actual {expected_actual}" in results[0].message


def test_trajectory_atif_missing_final_metrics_fails_with_useful_message():
    task = _eval_task_with_rubric(
        {"type": "trajectory_atif", "schema_version": "ATIF-v1.4", "require_final_metrics": True}
    )
    atif = _sample_atif()
    del atif["final_metrics"]

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert results[0].passed is False
    assert "final_metrics" in results[0].message
    assert "missing" in results[0].message


def test_trajectory_atif_missing_final_metric_fields_fails_with_useful_message():
    task = _eval_task_with_rubric(
        {"type": "trajectory_atif", "schema_version": "ATIF-v1.4", "require_final_metrics": True}
    )
    atif = _sample_atif()
    del atif["final_metrics"]["total_cached_tokens"]

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert results[0].passed is False
    assert "final_metrics" in results[0].message
    assert "total_cached_tokens" in results[0].message


def test_trajectory_atif_uses_exporter_final_metric_field_list(monkeypatch):
    task = _eval_task_with_rubric(
        {"type": "trajectory_atif", "schema_version": "ATIF-v1.4", "require_final_metrics": True}
    )
    monkeypatch.setattr(
        eval_module.trajectory,
        "FINAL_METRIC_FIELDS",
        ("total_prompt_tokens", "total_completion_tokens", "total_cached_tokens", "total_steps", "total_cost_units"),
        raising=False,
    )

    results = evaluate_rubric(task, _sample_atif(), WorkspaceSnapshot(files={}))

    assert results[0].passed is False
    assert "total_cost_units" in results[0].message


def test_starter_atif_task_uses_artifact_check_for_metrics_not_final_prose():
    task = {loaded.id: loaded for loaded in load_tasks(Path(__file__).resolve().parents[1] / "evals" / "tasks")}[
        "atif-trajectory-format"
    ]

    final_metric_prose_checks = [
        check
        for check in task.rubric
        if check.type == "final_response_contains" and check.params.get("contains") == "final_metrics"
    ]
    trajectory_checks = [check for check in task.rubric if check.type == "trajectory_atif"]

    assert final_metric_prose_checks == []
    assert len(trajectory_checks) == 1
    assert trajectory_checks[0].params == {"schema_version": "ATIF-v1.4", "require_final_metrics": True}
    assert "where do final token metrics live" not in task.prompt
    assert "how are final token metrics represented" not in task.prompt


def test_starter_atif_task_prompt_points_to_concrete_files_not_ambiguous_path():
    task = {loaded.id: loaded for loaded in load_tasks(Path(__file__).resolve().parents[1] / "evals" / "tasks")}[
        "atif-trajectory-format"
    ]

    assert "trajectory/eval-related" not in task.prompt
    assert "baremetal_agent/trajectory.py" in task.prompt
    assert "SCHEMA_VERSION" in task.prompt


def test_starter_atif_task_allows_reasonable_artifact_inspection_budget():
    task = {loaded.id: loaded for loaded in load_tasks(Path(__file__).resolve().parents[1] / "evals" / "tasks")}[
        "atif-trajectory-format"
    ]

    max_step_checks = [check for check in task.rubric if check.type == "max_agent_steps"]

    assert max_step_checks == [] or max_step_checks[0].params["value"] >= 6


def test_tool_called_passes_for_matching_function_name():
    task = _eval_task_with_rubric({"type": "tool_called", "name": "read_file"})
    atif = {
        "steps": [{"source": "agent", "tool_calls": [{"function_name": "read_file", "arguments": {"path": "a.txt"}}]}]
    }

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert results[0].passed is True


def test_tool_called_with_passes_for_recursive_subset_and_fails_for_mismatch():
    task = _eval_task_with_rubric(
        {"type": "tool_called_with", "name": "write_file", "arguments": {"path": "a.txt", "opts": {"mode": "w"}}},
        {"type": "tool_called_with", "name": "write_file", "arguments": {"opts": {"mode": "x"}}},
        {"type": "tool_called_with", "name": "write_file", "arguments": {"operations": [{"path": "src/main.py"}]}},
        {
            "type": "tool_called_with",
            "name": "write_file",
            "arguments": {"operations": [{"path": "src/main.py"}, {"path": "src/other.py"}]},
        },
    )
    atif = {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "function_name": "write_file",
                        "arguments": {
                            "path": "a.txt",
                            "opts": {"mode": "w", "encoding": "utf-8"},
                            "operations": [{"path": "src/main.py", "content": "hello"}],
                        },
                    }
                ],
            }
        ]
    }

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert results[0].passed is True
    assert results[1].passed is False
    assert results[2].passed is True
    assert results[3].passed is False


def test_tool_not_called_passes_and_fails_correctly():
    task = _eval_task_with_rubric(
        {"type": "tool_not_called", "name": "shell_exec"},
        {"type": "tool_not_called", "name": "read_file"},
    )
    atif = {"steps": [{"source": "agent", "tool_calls": [{"function_name": "read_file", "arguments": {}}]}]}

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert results[0].passed is True
    assert results[1].passed is False


def test_max_agent_steps_only_counts_agent_steps():
    task = _eval_task_with_rubric({"type": "max_agent_steps", "value": 2})
    atif = {
        "steps": [
            {"source": "user", "message": "first"},
            {"source": "agent", "message": "step 1", "tool_calls": [{"function_name": "a"}, {"function_name": "b"}]},
            {"source": "agent", "message": "step 2", "tool_calls": [{"function_name": "c"}]},
            {"source": "user", "message": "second"},
        ]
    }

    results = evaluate_rubric(task, atif, WorkspaceSnapshot(files={}))

    assert results[0].passed is True


def test_file_exists_and_file_contains_use_workspace_snapshot_not_live_filesystem(tmp_path):
    live_file = tmp_path / "example.txt"
    live_file.write_text("live content", encoding="utf-8")

    task = _eval_task_with_rubric(
        {"type": "file_exists", "path": "example.txt"},
        {"type": "file_contains", "path": "example.txt", "contains": "snapshot"},
    )
    workspace = WorkspaceSnapshot(
        files={"example.txt": FileSnapshot(exists=True, content="snapshot content", truncated=False)}
    )

    results = evaluate_rubric(task, {"steps": []}, workspace)

    assert results[0].passed is True
    assert results[1].passed is True


def test_file_contains_invalid_regex_fails_with_invalid_regex_message():
    task = _eval_task_with_rubric({"type": "file_contains", "path": "a.txt", "regex": "("})
    workspace = WorkspaceSnapshot(files={"a.txt": FileSnapshot(exists=True, content="x", truncated=False)})

    results = evaluate_rubric(task, {"steps": []}, workspace)

    assert results[0].passed is False
    assert "invalid regex" in results[0].message.lower()


def test_missing_or_wrong_params_fail_check_without_exception():
    task = _eval_task_with_rubric(
        {"type": "final_response_contains"},
        {"type": "max_agent_steps", "value": "3"},
    )

    results = evaluate_rubric(task, {"steps": [{"source": "agent", "message": "ok"}]}, WorkspaceSnapshot(files={}))

    assert results[0].passed is False
    assert results[1].passed is False


def test_execute_task_default_policy_uses_read_only_tools_and_confirmation_deny(tmp_path, monkeypatch, cfg):
    captured: dict[str, object] = {}

    def fake_run(prompt, history, api_responses, **kwargs):
        captured["prompt"] = prompt
        captured["history"] = list(history)
        captured["api_responses"] = list(api_responses)
        captured["kwargs"] = kwargs
        history.append({"role": "assistant", "content": "done"})
        return agent.AgentTurnResult(content="done", status=agent.STATUS_OK)

    monkeypatch.setattr(eval_module.agent, "run_agent_turn", fake_run)

    task = EvalTask(id="task-default", description="desc", prompt="run", rubric=(), setup=TaskSetup())
    execute_task(task, tmp_path / "sandbox", cfg=cfg)

    kwargs = captured["kwargs"]
    assert kwargs["confirmer"] is agent.auto_deny_confirmer
    assert kwargs["tool_names"] == tools.get_read_only_tool_names()
    assert isinstance(kwargs["renderer"], NullRenderer)
    assert captured["history"] == [{"role": "system", "content": cfg.system_prompt}]
    assert captured["api_responses"] == []


def test_execute_task_requires_writes_allows_write_file_but_denies_shell_exec(tmp_path, monkeypatch, cfg):
    captured: dict[str, object] = {}

    def fake_run(_prompt, history, _api_responses, **kwargs):
        captured["kwargs"] = kwargs
        history.append({"role": "assistant", "content": "ok"})
        return agent.AgentTurnResult(content="ok", status=agent.STATUS_OK)

    monkeypatch.setattr(eval_module.agent, "run_agent_turn", fake_run)

    task = EvalTask(
        id="task-write",
        description="desc",
        prompt="run",
        rubric=(),
        setup=TaskSetup(),
        requires_writes=True,
    )
    execute_task(task, tmp_path / "sandbox", cfg=cfg)

    kwargs = captured["kwargs"]
    assert kwargs["confirmer"] is agent.auto_approve_confirmer
    assert "write_file" in kwargs["tool_names"]
    assert "shell_exec" not in kwargs["tool_names"]


def test_execute_task_returns_atif_and_workspace_snapshot(tmp_path, monkeypatch, cfg):
    sandbox = tmp_path / "sandbox"

    def fake_run(_prompt, history, api_responses, **_kwargs):
        history.append({"role": "user", "content": "hello"})
        history.append({"role": "assistant", "content": "all done"})
        api_responses.append(
            {"model": "mock-model", "created": 1, "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
        )
        return agent.AgentTurnResult(content="all done", status=agent.STATUS_OK)

    monkeypatch.setattr(eval_module.agent, "run_agent_turn", fake_run)

    task = EvalTask(
        id="task-atif",
        description="desc",
        prompt="run",
        rubric=(),
        setup=TaskSetup(files=(SetupFile(path="artifact.txt", content="value"),)),
    )
    atif, workspace = execute_task(task, sandbox, cfg=cfg)

    assert isinstance(workspace, WorkspaceSnapshot)
    assert workspace.files["artifact.txt"].exists is True
    assert atif["schema_version"] == "ATIF-v1.4"
    assert any(step.get("source") == "agent" and step.get("message") == "all done" for step in atif["steps"])


def test_execute_task_propagates_failure(tmp_path, monkeypatch, cfg):
    """A non-OK agent turn raises ValueError describing the status."""

    def fake_non_ok(_prompt, _history, _api_responses, **_kwargs):
        return agent.AgentTurnResult(content="api down", status=agent.STATUS_API_ERROR)

    monkeypatch.setattr(eval_module.agent, "run_agent_turn", fake_non_ok)
    task = EvalTask(id="task-non-ok", description="desc", prompt="run", rubric=(), setup=TaskSetup())

    with pytest.raises(ValueError, match="status=api_error"):
        execute_task(task, tmp_path / "sandbox-status", cfg=cfg)


def test_run_task_non_ok_status_returns_error(monkeypatch, tmp_path, cfg):
    task = EvalTask(id="task-run-error", description="desc", prompt="run", rubric=(), setup=TaskSetup())

    def fake_execute(_task, _sandbox, *, cfg):
        raise ValueError("agent turn failed with status=api_error")

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)

    result = run_task(task, tmp_path / "cache" / "task.json", cfg=cfg)

    assert isinstance(result, EvalResult)
    assert result.status == STATUS_ERROR
    assert "api_error" in (result.error or "")


def test_run_task_skip_returns_skipped_without_calling_agent(monkeypatch, tmp_path, cfg):
    task = EvalTask(
        id="task-skip",
        description="desc",
        prompt="run",
        rubric=(),
        setup=TaskSetup(),
        skip="not supported yet",
    )

    def fake_execute(_task, _sandbox, *, cfg):
        raise AssertionError("execute_task should not be called for skipped task")

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    result = run_task(task, tmp_path / "cache" / "task.json", cfg=cfg)

    assert result.status == STATUS_SKIPPED
    assert result.error == "not supported yet"


@pytest.mark.parametrize(
    ("checks", "expected_status"),
    [
        ((CheckResult(type="tool_called", passed=True, message="ok"),), STATUS_PASS),
        (
            (
                CheckResult(type="tool_called", passed=True, message="ok"),
                CheckResult(type="file_exists", passed=False, message="x"),
            ),
            STATUS_FAIL,
        ),
    ],
)
def test_run_task_sets_pass_fail_from_rubric_results(monkeypatch, tmp_path, checks, expected_status, cfg):
    task = EvalTask(id="task-pass-fail", description="desc", prompt="run", rubric=(), setup=TaskSetup())

    def fake_execute(_task, _sandbox, *, cfg):
        return {"steps": []}, WorkspaceSnapshot(files={})

    def fake_eval(_task, _atif, _workspace):
        return checks

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    monkeypatch.setattr(eval_module.runner, "evaluate_rubric", fake_eval)

    result = run_task(task, tmp_path / "cache" / "task.json", cfg=cfg)

    assert result.status == expected_status
    assert result.checks == checks


def test_run_task_relative_cache_sandbox_supports_tool_path_safety(monkeypatch, tmp_path, cfg):
    monkeypatch.chdir(tmp_path)
    task = EvalTask(
        id="task-relative-cache",
        description="desc",
        prompt="read input",
        rubric=(RubricCheck(type="final_response_contains", params={"contains": "fixture content"}),),
        setup=TaskSetup(files=(SetupFile(path="input.txt", content="fixture content"),)),
    )

    def fake_run(_prompt, history, api_responses, *, cfg, **_kwargs):
        ctx = tools.ToolContext(working_dir=cfg.working_dir)
        content = tools.execute_tool("read_file", {"path": "input.txt"}, ctx=ctx)
        history.append({"role": "assistant", "content": content})
        api_responses.append({"created": 1, "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return agent.AgentTurnResult(content=content, status=agent.STATUS_OK)

    monkeypatch.setattr(eval_module.agent, "run_agent_turn", fake_run)

    result = run_task(task, Path(".baremetal-eval-cache"), cfg=cfg)

    assert result.status == STATUS_PASS
    assert result.checks[0].passed is True


def test_run_task_cache_miss_writes_artifacts_and_sets_trajectory_path(monkeypatch, tmp_path, cfg):
    task = EvalTask(
        id="task-cache-write",
        description="desc",
        prompt="run",
        rubric=(RubricCheck(type="file_contains", params={"path": "artifact.txt", "contains": "value"}),),
        setup=TaskSetup(),
    )
    atif = _sample_atif()
    workspace = WorkspaceSnapshot(files={"artifact.txt": FileSnapshot(exists=True, content="value", truncated=False)})

    def fake_execute(_task, _sandbox, *, cfg):
        return atif, workspace

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    cache_root = tmp_path / ".baremetal-eval-cache"

    result = run_task(task, cache_root, cfg=cfg)

    assert result.status == STATUS_PASS
    assert result.cached is False
    cache_dirs = sorted(cache_root.glob(f"{task.id}-*"))
    assert len(cache_dirs) == 1
    cache_dir = cache_dirs[0]
    assert (cache_dir / "trajectory.json").is_file()
    assert (cache_dir / "workspace.json").is_file()
    assert (cache_dir / "metadata.json").is_file()
    assert result.trajectory_path == str(cache_dir / "trajectory.json")


def test_run_task_second_run_reuses_cache_with_rubric_only_change(monkeypatch, tmp_path, cfg):
    cache_root = tmp_path / ".baremetal-eval-cache"
    base_task = EvalTask(
        id="task-cache-reuse",
        description="desc",
        prompt="run",
        rubric=(RubricCheck(type="tool_called", params={"name": "write_file"}),),
        setup=TaskSetup(),
    )
    atif = _sample_atif()
    atif["steps"][0]["tool_calls"] = [{"function_name": "write_file", "arguments": {"path": "artifact.txt"}}]
    workspace = WorkspaceSnapshot(files={"artifact.txt": FileSnapshot(exists=True, content="value", truncated=False)})

    execute_calls = {"count": 0}

    def first_execute(_task, _sandbox, *, cfg):
        execute_calls["count"] += 1
        return atif, workspace

    monkeypatch.setattr(eval_module.runner, "execute_task", first_execute)
    first = run_task(base_task, cache_root, cfg=cfg)
    assert first.cached is False
    assert execute_calls["count"] == 1

    def must_not_run(_task, _sandbox, *, cfg):
        raise AssertionError("execute_task should not run on cache hit")

    monkeypatch.setattr(eval_module.runner, "execute_task", must_not_run)
    changed_rubric_task = EvalTask(
        id=base_task.id,
        description=base_task.description,
        prompt=base_task.prompt,
        rubric=(RubricCheck(type="file_contains", params={"path": "artifact.txt", "contains": "value"}),),
        setup=base_task.setup,
    )
    second = run_task(changed_rubric_task, cache_root, cfg=cfg)

    assert second.status == STATUS_PASS
    assert second.cached is True
    assert second.checks[0].type == "file_contains"


def test_run_task_prompt_or_setup_change_invalidates_cache(monkeypatch, tmp_path, cfg):
    cache_root = tmp_path / ".baremetal-eval-cache"
    base = EvalTask(id="task-key", description="desc", prompt="run", rubric=(), setup=TaskSetup())
    calls = {"count": 0}

    def fake_execute(_task, _sandbox, *, cfg):
        calls["count"] += 1
        return _sample_atif(), WorkspaceSnapshot(files={})

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)

    first = run_task(base, cache_root, cfg=cfg)
    second = run_task(EvalTask(**{**base.__dict__, "prompt": "run changed"}), cache_root, cfg=cfg)
    third = run_task(
        EvalTask(**{**base.__dict__, "setup": TaskSetup(files=(SetupFile(path="a.txt", content="x"),))}),
        cache_root,
        cfg=cfg,
    )

    assert first.cached is False
    assert second.cached is False
    assert third.cached is False
    assert calls["count"] == 3
    assert len(list(cache_root.glob(f"{base.id}-*"))) == 3


def test_run_task_system_prompt_change_invalidates_cache(monkeypatch, tmp_path, cfg):
    cache_root = tmp_path / ".baremetal-eval-cache"
    task = EvalTask(id="task-system-prompt", description="desc", prompt="run", rubric=(), setup=TaskSetup())
    calls = {"count": 0}

    def fake_execute(_task, _sandbox, *, cfg):
        calls["count"] += 1
        return _sample_atif(), WorkspaceSnapshot(files={})

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    first = run_task(task, cache_root, cfg=cfg)
    assert first.cached is False

    second = run_task(task, cache_root, cfg=replace(cfg, system_prompt="system prompt changed"))

    assert second.cached is False
    assert calls["count"] == 2
    assert len(list(cache_root.glob(f"{task.id}-*"))) == 2


def test_run_task_atomic_cache_write_no_partial_dir_on_failure(monkeypatch, tmp_path, cfg):
    cache_root = tmp_path / ".baremetal-eval-cache"
    task = EvalTask(id="task-atomic-write", description="desc", prompt="run", rubric=(), setup=TaskSetup())
    atif = _sample_atif()
    workspace = WorkspaceSnapshot(files={"artifact.txt": FileSnapshot(exists=True, content="value", truncated=False)})
    execution_hash, _ = eval_module.cache._execution_hash(task, cfg)
    final_dir = cache_root / f"{task.id}-{execution_hash}"

    def fake_execute(_task, _sandbox, *, cfg):
        return atif, workspace

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    original_save_workspace = eval_module.runner._save_workspace_snapshot

    def fail_save_workspace(_workspace, _path):
        raise OSError("disk full")

    monkeypatch.setattr(eval_module.runner, "_save_workspace_snapshot", fail_save_workspace)
    first = run_task(task, cache_root, cfg=cfg)

    assert first.status == STATUS_ERROR
    assert not final_dir.exists()

    monkeypatch.setattr(eval_module.runner, "_save_workspace_snapshot", original_save_workspace)
    second = run_task(task, cache_root, cfg=cfg)

    assert second.status in {STATUS_PASS, STATUS_FAIL}
    assert second.cached is False
    assert final_dir.is_dir()
    assert (final_dir / "trajectory.json").is_file()
    assert (final_dir / "workspace.json").is_file()
    assert (final_dir / "metadata.json").is_file()


def test_run_task_corrupt_cached_trajectory_returns_error_without_rerun(monkeypatch, tmp_path, cfg):
    cache_root = tmp_path / ".baremetal-eval-cache"
    task = EvalTask(id="task-corrupt-traj", description="desc", prompt="run", rubric=(), setup=TaskSetup())

    def fake_execute(_task, _sandbox, *, cfg):
        return _sample_atif(), WorkspaceSnapshot(files={})

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    first = run_task(task, cache_root, cfg=cfg)
    assert first.status in {STATUS_PASS, STATUS_FAIL}

    trajectory_path = Path(first.trajectory_path)
    trajectory_path.write_text("{not-json", encoding="utf-8")

    def must_not_run(_task, _sandbox, *, cfg):
        raise AssertionError("execute_task should not run with corrupt cache")

    monkeypatch.setattr(eval_module.runner, "execute_task", must_not_run)
    second = run_task(task, cache_root, cfg=cfg)

    assert second.status == STATUS_ERROR
    assert "trajectory" in (second.error or "")


def test_run_task_corrupt_cached_workspace_returns_error_without_rerun(monkeypatch, tmp_path, cfg):
    cache_root = tmp_path / ".baremetal-eval-cache"
    task = EvalTask(id="task-corrupt-workspace", description="desc", prompt="run", rubric=(), setup=TaskSetup())

    def fake_execute(_task, _sandbox, *, cfg):
        return _sample_atif(), WorkspaceSnapshot(files={})

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    run_task(task, cache_root, cfg=cfg)
    workspace_path = next(cache_root.glob(f"{task.id}-*/workspace.json"))
    workspace_path.write_text('{"files": {"x": "bad"}}', encoding="utf-8")

    def must_not_run(_task, _sandbox, *, cfg):
        raise AssertionError("execute_task should not run with corrupt cache")

    monkeypatch.setattr(eval_module.runner, "execute_task", must_not_run)
    second = run_task(task, cache_root, cfg=cfg)

    assert second.status == STATUS_ERROR
    assert "workspace" in (second.error or "")


def test_run_task_fixture_digest_change_invalidates_cache(monkeypatch, tmp_path, cfg):
    task_file = tmp_path / "tasks" / "task.json"
    task_file.parent.mkdir(parents=True)
    fixture_root = task_file.parent / "fixtures" / "basic"
    fixture_root.mkdir(parents=True)
    fixture_file = fixture_root / "data.txt"
    fixture_file.write_text("one", encoding="utf-8")
    task = EvalTask(
        id="task-fixture-digest",
        description="desc",
        prompt="run",
        rubric=(),
        setup=TaskSetup(fixture_dir="fixtures/basic"),
        source_path=task_file,
    )
    cache_root = tmp_path / ".baremetal-eval-cache"
    calls = {"count": 0}

    def fake_execute(_task, _sandbox, *, cfg):
        calls["count"] += 1
        return _sample_atif(), WorkspaceSnapshot(files={})

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    first = run_task(task, cache_root, cfg=cfg)
    assert first.cached is False

    fixture_file.write_text("two", encoding="utf-8")
    second = run_task(task, cache_root, cfg=cfg)

    assert second.cached is False
    assert calls["count"] == 2
    assert len(list(cache_root.glob(f"{task.id}-*"))) == 2


def test_run_task_copy_repo_digest_change_invalidates_cache(monkeypatch, tmp_path, cfg):
    repo = _init_repo_fixture(tmp_path)
    task_file = repo / "evals" / "tasks" / "task.json"
    task_file.write_text("{}", encoding="utf-8")
    source_file = repo / "README.md"
    source_file.write_text("one", encoding="utf-8")
    _run_git(repo, "add", "README.md", "evals/tasks/task.json")
    task = EvalTask(
        id="task-copy-repo-digest",
        description="desc",
        prompt="run",
        rubric=(),
        setup=TaskSetup(copy_repo=True),
        source_path=task_file,
    )
    cache_root = tmp_path / ".baremetal-eval-cache"
    calls = {"count": 0}

    def fake_execute(_task, _sandbox, *, cfg):
        calls["count"] += 1
        return _sample_atif(), WorkspaceSnapshot(files={})

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    first = run_task(task, cache_root, cfg=cfg)
    assert first.cached is False

    source_file.write_text("two", encoding="utf-8")
    second = run_task(task, cache_root, cfg=cfg)

    assert second.cached is False
    assert calls["count"] == 2
    assert len(list(cache_root.glob(f"{task.id}-*"))) == 2


def test_run_task_metadata_hash_excludes_description_and_rubric(monkeypatch, tmp_path, cfg):
    cache_root = tmp_path / ".baremetal-eval-cache"
    base_task = EvalTask(
        id="task-meta",
        description="original desc",
        prompt="run",
        rubric=(RubricCheck(type="tool_called", params={"name": "read_file"}),),
        setup=TaskSetup(),
    )

    def fake_execute(_task, _sandbox, *, cfg):
        return _sample_atif(), WorkspaceSnapshot(files={})

    monkeypatch.setattr(eval_module.runner, "execute_task", fake_execute)
    first = run_task(base_task, cache_root, cfg=cfg)
    assert first.cached is False
    metadata_path = next(cache_root.glob(f"{base_task.id}-*/metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hash_inputs = metadata["execution_hash_inputs"]
    assert "description" not in hash_inputs
    assert "rubric" not in hash_inputs

    def must_not_run(_task, _sandbox, *, cfg):
        raise AssertionError("description/rubric changes should not rerun")

    monkeypatch.setattr(eval_module.runner, "execute_task", must_not_run)
    second_task = EvalTask(
        id=base_task.id,
        description="new desc",
        prompt=base_task.prompt,
        rubric=(RubricCheck(type="final_response_contains", params={"contains": "done"}),),
        setup=base_task.setup,
    )
    second = run_task(second_task, cache_root, cfg=cfg)
    assert second.cached is True


def test_gitignore_contains_eval_cache_entry():
    gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"
    contents = gitignore_path.read_text(encoding="utf-8")
    assert ".baremetal-eval-cache/" in contents


def test_run_task_concurrent_execution_is_isolated(tmp_path, monkeypatch, cfg):
    """Two ``run_task`` calls running on different sandboxes must not bleed
    files or trajectories into each other.

    This is the headline guardrail for dropping ``_EXECUTE_TASK_LOCK``: with
    ``ToolContext`` threaded through the agent, the only sandbox state lives
    on each thread's stack (``ctx.working_dir``), so concurrent execution is
    safe even when both agents touch ``write_file``/``read_file`` tools.
    """

    # Use real tool execution against a fake "model" that drives one
    # ``write_file`` call per task with task-specific content. If isolation is
    # broken, the writes will land in the wrong sandbox.
    barrier = threading.Barrier(2)
    call_history: dict[str, list[str]] = {}

    def fake_run(prompt, history, api_responses, *, cfg, **_kwargs):
        # Force overlap so any shared mutable state would corrupt at least
        # one trajectory. Each task writes its own marker file via the real
        # tool dispatcher (so we exercise the ToolContext path end-to-end).
        marker = prompt  # task prompt is the marker we write
        ctx = tools.ToolContext(working_dir=cfg.working_dir)
        barrier.wait(timeout=5)
        result_str = tools.execute_tool(
            "write_file",
            {"path": "marker.txt", "content": marker},
            ctx=ctx,
        )
        # Re-read via ctx to confirm round-trip isolation inside the same call.
        readback = tools.execute_tool("read_file", {"path": "marker.txt"}, ctx=ctx)
        history.append({"role": "assistant", "content": readback})
        api_responses.append({"created": 1, "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        call_history.setdefault(marker, []).append(result_str)
        return agent.AgentTurnResult(content=readback, status=agent.STATUS_OK)

    monkeypatch.setattr(eval_module.agent, "run_agent_turn", fake_run)

    task_a = EvalTask(
        id="concurrent-a",
        description="task A",
        prompt="MARKER-A",
        rubric=(RubricCheck(type="file_contains", params={"path": "marker.txt", "contains": "MARKER-A"}),),
        setup=TaskSetup(),
        requires_writes=True,
    )
    task_b = EvalTask(
        id="concurrent-b",
        description="task B",
        prompt="MARKER-B",
        rubric=(RubricCheck(type="file_contains", params={"path": "marker.txt", "contains": "MARKER-B"}),),
        setup=TaskSetup(),
        requires_writes=True,
    )

    cache_root = tmp_path / "cache"

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(run_task, task_a, cache_root, cfg=cfg)
        future_b = executor.submit(run_task, task_b, cache_root, cfg=cfg)
        result_a = future_a.result(timeout=5)
        result_b = future_b.result(timeout=5)

    # Both tasks succeeded.
    assert result_a.status == STATUS_PASS, result_a
    assert result_b.status == STATUS_PASS, result_b

    # Sandboxes are torn down by run_task, so reach into the on-disk
    # trajectories to confirm each task's history saw only its own marker.
    traj_a = json.loads(Path(result_a.trajectory_path).read_text())
    traj_b = json.loads(Path(result_b.trajectory_path).read_text())

    def _assistant_messages(atif: dict) -> list[str]:
        return [step.get("message", "") for step in atif["steps"] if step.get("source") == "agent"]

    a_msgs = _assistant_messages(traj_a)
    b_msgs = _assistant_messages(traj_b)
    assert any("MARKER-A" in m for m in a_msgs), a_msgs
    assert all("MARKER-B" not in m for m in a_msgs), a_msgs
    assert any("MARKER-B" in m for m in b_msgs), b_msgs
    assert all("MARKER-A" not in m for m in b_msgs), b_msgs

    # Each task triggered exactly one fake_run invocation (no cross-bleed
    # of cfg.working_dir into the wrong call).
    assert len(call_history["MARKER-A"]) == 1, call_history
    assert len(call_history["MARKER-B"]) == 1, call_history


def _result_samples() -> list[EvalResult]:
    return [
        EvalResult(
            task_id="task-pass",
            description="passes checks",
            status=STATUS_PASS,
            checks=(
                CheckResult(type="tool_called", passed=True, message="write_file was called"),
                CheckResult(type="file_exists", passed=True, message="artifact.txt exists"),
            ),
            trajectory_path="cache/task-pass/trajectory.json",
            cached=True,
        ),
        EvalResult(
            task_id="task-fail",
            description="fails checks",
            status=STATUS_FAIL,
            checks=(CheckResult(type="file_contains", passed=False, message="missing expected text"),),
            trajectory_path="cache/task-fail/trajectory.json",
            cached=False,
        ),
        EvalResult(
            task_id="task-error",
            description="errors",
            status=STATUS_ERROR,
            error="agent crashed",
            checks=(),
        ),
        EvalResult(
            task_id="task-skip",
            description="skipped",
            status=STATUS_SKIPPED,
            error="unsupported platform",
            checks=(),
        ),
    ]


def test_render_markdown_report_includes_summary_and_task_details(monkeypatch, cfg):
    cfg = replace(cfg, model="mock-model")
    report = render_markdown_report(_result_samples(), cfg=cfg)

    assert "# Baremetal Agent Eval Report" in report
    assert "- Model: `mock-model`" in report
    assert "- Total tasks: 4" in report
    assert "- pass: 1" in report
    assert "- fail: 1" in report
    assert "- error: 1" in report
    assert "- skipped: 1" in report
    assert "| Task | Status | Checks | Cache |" in report
    assert "| task-pass | pass | 2/2 | hit |" in report
    assert "| task-fail | fail | 0/1 | miss |" in report
    assert "## task-pass — passes checks" in report
    assert "- Trajectory: `cache/task-pass/trajectory.json`" in report
    assert "- ✅ tool_called: write_file was called" in report
    assert "- ❌ file_contains: missing expected text" in report
    assert "- Error: agent crashed" in report
    assert "- Error: unsupported platform" in report


def test_render_json_report_has_expected_shape(monkeypatch, cfg):
    cfg = replace(cfg, model="mock-model")
    report = render_json_report(_result_samples(), cfg=cfg)

    assert report["model"] == "mock-model"
    assert report["summary"] == {"total": 4, "pass": 1, "fail": 1, "error": 1, "skipped": 1}
    assert report["results"][0] == {
        "task_id": "task-pass",
        "description": "passes checks",
        "status": "pass",
        "cached": True,
        "trajectory_path": "cache/task-pass/trajectory.json",
        "checks": [
            {"type": "tool_called", "passed": True, "message": "write_file was called"},
            {"type": "file_exists", "passed": True, "message": "artifact.txt exists"},
        ],
        "error": None,
    }
    assert report["results"][3]["status"] == "skipped"
    assert report["results"][3]["error"] == "unsupported platform"


def test_report_rendering_sanitizes_check_messages_and_errors(cfg):
    raw_secret = "token=abcdef123456"
    result = EvalResult(
        task_id="task-secret",
        description=f"desc {raw_secret}",
        status=STATUS_FAIL,
        checks=(CheckResult(type="file_contains", passed=False, message=f"missing {raw_secret}"),),
        error=f"error {raw_secret}",
    )

    markdown = render_markdown_report([result], cfg=cfg)
    json_report = render_json_report([result], cfg=cfg)
    serialized_json_report = json.dumps(json_report)

    assert raw_secret not in markdown
    assert raw_secret not in serialized_json_report
    assert "token=abcdef******" in markdown
    assert "token=abcdef******" in serialized_json_report


def test_run_eval_suite_writes_reports_and_creates_parent_dirs(monkeypatch, tmp_path, cfg):
    tasks = [
        EvalTask(id="task-pass", description="p", prompt="run", rubric=(), setup=TaskSetup()),
        EvalTask(id="task-skip", description="s", prompt="run", rubric=(), setup=TaskSetup()),
    ]
    results = {
        "task-pass": EvalResult(task_id="task-pass", description="p", status=STATUS_PASS),
        "task-skip": EvalResult(task_id="task-skip", description="s", status=STATUS_SKIPPED, error="skip"),
    }
    cache_dirs: list[Path] = []

    def fake_load_tasks(path):
        assert path == tmp_path / "tasks"
        return tasks

    def fake_run_task(task, cache_dir, *, cfg):
        cache_dirs.append(cache_dir)
        return results[task.id]

    monkeypatch.setattr(eval_module.runner, "load_tasks", fake_load_tasks)
    monkeypatch.setattr(eval_module.runner, "run_task", fake_run_task)
    cfg = replace(cfg, model="mock-model")
    monkeypatch.chdir(tmp_path)

    markdown_out = tmp_path / "nested" / "reports" / "report.md"
    json_out = tmp_path / "nested" / "reports" / "report.json"
    exit_code = run_eval_suite(tmp_path / "tasks", markdown_out, json_out, cfg=cfg)

    assert exit_code == 0
    assert markdown_out.is_file()
    assert json_out.is_file()
    assert markdown_out.read_text(encoding="utf-8").startswith("# Baremetal Agent Eval Report")
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["summary"]["skipped"] == 1
    assert cache_dirs == [Path(".baremetal-eval-cache"), Path(".baremetal-eval-cache")]


def test_run_eval_suite_returns_1_when_any_task_fails_or_errors(monkeypatch, tmp_path, cfg):
    tasks = [
        EvalTask(id="task-pass", description="p", prompt="run", rubric=(), setup=TaskSetup()),
        EvalTask(id="task-fail", description="f", prompt="run", rubric=(), setup=TaskSetup()),
    ]

    def fake_load_tasks(_path):
        return tasks

    def fake_run_task(task, _cache_dir, *, cfg):
        if task.id == "task-pass":
            return EvalResult(task_id=task.id, description=task.description, status=STATUS_PASS)
        return EvalResult(task_id=task.id, description=task.description, status=STATUS_FAIL)

    monkeypatch.setattr(eval_module.runner, "load_tasks", fake_load_tasks)
    monkeypatch.setattr(eval_module.runner, "run_task", fake_run_task)

    exit_code = run_eval_suite(tmp_path / "tasks", tmp_path / "report.md", tmp_path / "report.json", cfg=cfg)
    assert exit_code == 1


def test_run_eval_suite_returns_1_when_any_task_errors(monkeypatch, tmp_path, cfg):
    tasks = [EvalTask(id="task-error", description="e", prompt="run", rubric=(), setup=TaskSetup())]

    monkeypatch.setattr(eval_module.runner, "load_tasks", lambda _path: tasks)
    monkeypatch.setattr(
        eval_module.runner,
        "run_task",
        lambda task, _cache_dir, **_kwargs: EvalResult(
            task_id=task.id,
            description=task.description,
            status=STATUS_ERROR,
            error="boom",
        ),
    )

    exit_code = run_eval_suite(tmp_path / "tasks", tmp_path / "report.md", tmp_path / "report.json", cfg=cfg)
    assert exit_code == 1


def test_run_eval_suite_handles_load_errors_without_writing_reports(monkeypatch, tmp_path, capsys, cfg):
    markdown_out = tmp_path / "out" / "report.md"
    json_out = tmp_path / "out" / "report.json"

    monkeypatch.setattr(eval_module.runner, "load_tasks", lambda _path: (_ for _ in ()).throw(ValueError("bad input")))
    exit_code = run_eval_suite(tmp_path / "missing", markdown_out, json_out, cfg=cfg)

    assert exit_code == 1
    assert not markdown_out.exists()
    assert not json_out.exists()
    assert "eval: bad input" in capsys.readouterr().err


def test_run_eval_suite_reports_output_path_errors_without_traceback(monkeypatch, tmp_path, capsys, cfg):
    tasks = [EvalTask(id="task-pass", description="p", prompt="run", rubric=(), setup=TaskSetup())]
    parent_file = tmp_path / "report-parent-file"
    parent_file.write_text("not a directory", encoding="utf-8")

    monkeypatch.setattr(eval_module.runner, "load_tasks", lambda _path: tasks)
    monkeypatch.setattr(
        eval_module.runner,
        "run_task",
        lambda task, _cache_dir, **_kwargs: EvalResult(
            task_id=task.id, description=task.description, status=STATUS_PASS
        ),
    )

    exit_code = run_eval_suite(
        tmp_path / "tasks",
        parent_file / "out.md",
        tmp_path / "report.json",
        cfg=cfg,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("eval: ")
    assert "Traceback" not in captured.err


def test_run_eval_suite_rejects_non_positive_workers(monkeypatch, tmp_path, capsys, cfg):
    """``run_eval_suite(workers=0)`` should error out before loading tasks."""

    def explode(_path):
        raise AssertionError("load_tasks must not be called when workers is invalid")

    monkeypatch.setattr(eval_module.runner, "load_tasks", explode)

    exit_code = run_eval_suite(
        tmp_path / "tasks",
        tmp_path / "report.md",
        tmp_path / "report.json",
        cfg=cfg,
        workers=0,
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "workers must be >= 1" in captured.err
    assert "(got 0)" in captured.err


def test_run_eval_suite_workers_gt_1_uses_threadpool_and_preserves_order(monkeypatch, tmp_path, cfg):
    """With ``workers>1``, tasks run on a ``ThreadPoolExecutor`` and the
    final results list still reflects the load order so the report is
    deterministic across runs."""
    tasks = [
        EvalTask(id=f"task-{i}", description=f"task {i}", prompt="run", rubric=(), setup=TaskSetup()) for i in range(5)
    ]

    monkeypatch.setattr(eval_module.runner, "load_tasks", lambda _path: tasks)

    state_lock = threading.Lock()
    state = {"in_flight": 0, "max_in_flight": 0}
    seen_threads: set[int] = set()

    def fake_run_task(task, _cache_dir, *, cfg):
        with state_lock:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            seen_threads.add(threading.get_ident())
        # Sleep briefly so concurrent submissions actually overlap; without
        # this, the executor may run tasks back-to-back on a single worker.
        time.sleep(0.05)
        with state_lock:
            state["in_flight"] -= 1
        return EvalResult(task_id=task.id, description=task.description, status=STATUS_PASS)

    monkeypatch.setattr(eval_module.runner, "run_task", fake_run_task)

    exit_code = run_eval_suite(
        tmp_path / "tasks",
        tmp_path / "report.md",
        tmp_path / "report.json",
        cfg=cfg,
        workers=3,
    )

    assert exit_code == 0
    # Multiple worker threads were used and at least 2 ran concurrently —
    # proves the threadpool path was taken instead of the serial fallback.
    assert len(seen_threads) >= 2
    assert state["max_in_flight"] >= 2

    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    # Result order in the report matches load order regardless of worker
    # finish order — this is the determinism guarantee the design requires.
    assert [r["task_id"] for r in data["results"]] == [t.id for t in tasks]
    assert data["summary"][STATUS_PASS] == 5


def test_run_eval_suite_workers_gt_1_yields_same_pass_fail_set_as_serial(monkeypatch, tmp_path, cfg):
    """``workers=2`` and ``workers=1`` must yield identical aggregate
    pass/fail/error/skipped counts and the same per-task statuses (modulo
    the timing differences ``ThreadPoolExecutor`` introduces)."""
    statuses = {
        "task-a": STATUS_PASS,
        "task-b": STATUS_FAIL,
        "task-c": STATUS_PASS,
        "task-d": STATUS_ERROR,
        "task-e": STATUS_SKIPPED,
    }
    tasks = [EvalTask(id=tid, description=tid, prompt="run", rubric=(), setup=TaskSetup()) for tid in statuses]

    monkeypatch.setattr(eval_module.runner, "load_tasks", lambda _path: tasks)
    monkeypatch.setattr(
        eval_module.runner,
        "run_task",
        lambda task, _cache_dir, **_kwargs: EvalResult(
            task_id=task.id,
            description=task.description,
            status=statuses[task.id],
            error="boom" if statuses[task.id] == STATUS_ERROR else None,
        ),
    )

    serial_md = tmp_path / "serial.md"
    serial_json = tmp_path / "serial.json"
    parallel_md = tmp_path / "parallel.md"
    parallel_json = tmp_path / "parallel.json"

    serial_exit = run_eval_suite(tmp_path / "tasks", serial_md, serial_json, cfg=cfg, workers=1)
    parallel_exit = run_eval_suite(tmp_path / "tasks", parallel_md, parallel_json, cfg=cfg, workers=4)

    assert serial_exit == parallel_exit == 1  # task-b fails, task-d errors
    serial_data = json.loads(serial_json.read_text(encoding="utf-8"))
    parallel_data = json.loads(parallel_json.read_text(encoding="utf-8"))
    assert serial_data == parallel_data


def test_run_eval_suite_workers_eq_1_stays_on_serial_path(monkeypatch, tmp_path, cfg):
    """``workers=1`` must not spin up a ``ThreadPoolExecutor`` — the common
    case stays on the synchronous code path so tracebacks, profiling, and
    interactive debugging behave exactly as before."""
    tasks = [
        EvalTask(id="task-a", description="a", prompt="run", rubric=(), setup=TaskSetup()),
        EvalTask(id="task-b", description="b", prompt="run", rubric=(), setup=TaskSetup()),
    ]

    monkeypatch.setattr(eval_module.runner, "load_tasks", lambda _path: tasks)

    sentinel: list[bool] = []

    class _Boom:
        def __init__(self, *_, **__):
            sentinel.append(True)
            raise AssertionError("workers=1 must not construct a ThreadPoolExecutor")

    monkeypatch.setattr(eval_module.runner, "ThreadPoolExecutor", _Boom)
    monkeypatch.setattr(
        eval_module.runner,
        "run_task",
        lambda task, _cache_dir, **_kwargs: EvalResult(
            task_id=task.id, description=task.description, status=STATUS_PASS
        ),
    )

    exit_code = run_eval_suite(tmp_path / "tasks", tmp_path / "report.md", tmp_path / "report.json", cfg=cfg, workers=1)

    assert exit_code == 0
    assert sentinel == []


def test_render_reports_are_order_independent_for_aggregates(cfg):
    """Report aggregation (totals, per-status counts) is order-independent,
    which is what makes the parallel runner safe to feed an out-of-order
    result list. Reordering EvalResults must not change the JSON summary
    counts or the Markdown header line totals."""
    results = _result_samples()
    reversed_results = list(reversed(results))

    forward_json = render_json_report(results, cfg=cfg)
    reversed_json = render_json_report(reversed_results, cfg=cfg)

    assert forward_json["summary"] == reversed_json["summary"]
    forward_md = render_markdown_report(results, cfg=cfg)
    reversed_md = render_markdown_report(reversed_results, cfg=cfg)
    # Header section (totals + per-status counts) is identical regardless
    # of the order tasks finished in.
    assert forward_md.splitlines()[:9] == reversed_md.splitlines()[:9]


def test_eval_public_api_has_no_underscore_leaks():
    """The eval package public API must not re-export private helpers.

    Private helpers (``_execution_hash``, ``_apply_setup``, etc.) live in
    submodules and tests import them from there directly. Re-exporting them
    on ``baremetal_agent.eval`` leaks the test boundary into the public API.
    """
    underscore_exports = [name for name in dir(eval_module) if name.startswith("_") and not name.startswith("__")]
    assert underscore_exports == [], f"baremetal_agent.eval re-exports private names: {underscore_exports}"

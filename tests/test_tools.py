"""Tests for tools.py — validation, path safety, and tool execution."""

import subprocess
from pathlib import Path

import pytest

from baremetal_agent import safety, tools

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _working_dir(tmp_path, monkeypatch):
    """Run every tools test inside a fresh CWD.

    Tools fall back to ``Path.cwd()`` when no ``ctx`` is passed; chdir-ing
    here keeps test bodies free of explicit ctx wiring while still isolating
    each test.
    """
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# _validate_args
# ---------------------------------------------------------------------------


class TestValidateArgs:
    SCHEMA = {
        "properties": {
            "path": {"type": "string"},
            "count": {"type": "integer"},
            "verbose": {"type": "boolean"},
        },
        "required": ["path"],
    }

    def test_valid_args(self, tmp_path):
        assert tools._validate_args("t", {"path": "a.txt"}, self.SCHEMA) is None

    def test_missing_required(self, tmp_path):
        err = tools._validate_args("t", {}, self.SCHEMA)
        assert err is not None and "path" in err

    def test_wrong_type(self, tmp_path):
        err = tools._validate_args("t", {"path": 123}, self.SCHEMA)
        assert err is not None and "string" in err

    def test_bool_not_accepted_as_integer(self, tmp_path):
        err = tools._validate_args("t", {"path": "a", "count": True}, self.SCHEMA)
        assert err is not None

    def test_extra_args_ignored(self, tmp_path):
        assert tools._validate_args("t", {"path": "a", "extra": 1}, self.SCHEMA) is None


# ---------------------------------------------------------------------------
# _resolve_safe
# ---------------------------------------------------------------------------


class TestResolveSafe:
    def test_normal_path(self, tmp_path):
        result = tools._resolve_safe("foo.txt")
        assert isinstance(result, Path)
        assert result.parent == tmp_path.resolve()

    def test_traversal_blocked(self, tmp_path):
        with pytest.raises(tools.PathEscape, match="escapes"):
            tools._resolve_safe("../../etc/passwd")

    def test_absolute_outside_blocked(self, tmp_path):
        with pytest.raises(tools.PathEscape, match="escapes"):
            tools._resolve_safe("/etc/passwd")


# ---------------------------------------------------------------------------
# read_file / write_file / list_directory
# ---------------------------------------------------------------------------


class TestFileTools:
    def test_read_existing_file(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hello world")
        result = tools.read_file(path="hello.txt")
        assert result == "hello world"

    def test_read_missing_file(self, tmp_path):
        result = tools.read_file(path="nope.txt")
        assert "not found" in result.lower()

    def test_read_file_redacts_and_truncates_large_content(self, tmp_path):
        raw_secret = "token=abcdef123456"
        (tmp_path / "secret.txt").write_text(raw_secret + "\n" + ("x" * safety.DEFAULT_MAX_CHARS))

        result = tools.read_file(path="secret.txt")

        assert raw_secret not in result
        assert "token=abcdef******" in result
        assert "[truncated: read_file exceeded" in result

    def test_write_and_read(self, tmp_path):
        tools.write_file(path="out.txt", content="data")
        assert (tmp_path / "out.txt").read_text() == "data"

    def test_write_creates_parents(self, tmp_path):
        tools.write_file(path="sub/dir/f.txt", content="nested")
        assert (tmp_path / "sub/dir/f.txt").read_text() == "nested"

    def test_list_directory(self, tmp_path):
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        result = tools.list_directory(path=".")
        assert "a.py" in result and "b.py" in result

    def test_list_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        result = tools.list_directory(path="empty")
        assert "empty" in result.lower()

    def test_list_not_a_directory(self, tmp_path):
        (tmp_path / "file.txt").touch()
        result = tools.list_directory(path="file.txt")
        assert "not a directory" in result.lower()

    def test_list_directory_redacts_and_truncates_large_listing(self, tmp_path):
        for index in range(140):
            filename = f"file_{index:03d}_token=abcdef123456_" + ("x" * 60)
            (tmp_path / filename).touch()

        result = tools.list_directory(path=".")

        assert "token=abcdef123456" not in result
        assert "token=abcdef******" in result
        assert "[truncated: list_directory exceeded" in result


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------


class TestExecuteTool:
    def test_unknown_tool(self, tmp_path):
        result = tools.execute_tool("nonexistent", {})
        assert "unknown tool" in result.lower()

    def test_valid_tool_call(self, tmp_path):
        (tmp_path / "test.txt").write_text("content")
        result = tools.execute_tool("read_file", {"path": "test.txt"})
        assert result == "content"

    def test_missing_required_arg(self, tmp_path):
        result = tools.execute_tool("read_file", {})
        assert "invalid arguments" in result.lower()


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------


class TestSearchCode:
    def test_finds_pattern(self, tmp_path):
        (tmp_path / "code.py").write_text("def hello():\n    pass\n")
        result = tools.search_code(pattern="def hello")
        assert "code.py" in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "code.py").write_text("nothing here\n")
        result = tools.search_code(pattern="zzzzz")
        assert "no matches" in result.lower()

    def test_invalid_regex(self, tmp_path):
        result = tools.search_code(pattern="[invalid")
        assert "invalid regex" in result.lower()

    def test_search_code_redacts_and_truncates_large_matches(self, tmp_path):
        raw_secret = "password=abcdef123456"
        (tmp_path / "code.py").write_text("needle " + raw_secret + " " + ("x" * safety.DEFAULT_MAX_CHARS))

        result = tools.search_code(pattern="needle", file_glob="*.py")

        assert raw_secret not in result
        assert "password=abcdef******" in result
        assert "[truncated: search_code exceeded" in result


class TestCommandTools:
    def test_shell_exec_redacts_and_truncates_large_output(self, tmp_path):
        raw_secret = "token=abcdef123456"
        command = f"printf '{raw_secret} '; head -c {safety.DEFAULT_MAX_CHARS} /dev/zero | tr '\\0' x"

        result = tools.shell_exec(command=command)

        assert raw_secret not in result
        assert "token=abcdef******" in result
        assert "[truncated: shell_exec exceeded" in result

    def test_git_diff_redacts_and_truncates_large_diff(self, tmp_path):
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked = tmp_path / "tracked.txt"
        tracked.write_text("base\n")
        subprocess.run(
            ["git", "add", "tracked.txt"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        raw_secret = "token=abcdef123456"
        tracked.write_text(raw_secret + "\n" + ("x" * safety.DEFAULT_MAX_CHARS))

        result = tools.git_diff()

        assert raw_secret not in result
        assert "token=abcdef******" in result
        assert "[truncated: git_diff exceeded" in result


# ---------------------------------------------------------------------------
# @tool decorator registry
# ---------------------------------------------------------------------------


def test_tool_decorator_registers_all_eight_tools():
    """The @tool decorator should register all 8 tools, with confirmation only on write_file/shell_exec."""
    expected = {
        "read_file",
        "write_file",
        "list_directory",
        "search_code",
        "shell_exec",
        "git_status",
        "git_diff",
        "git_log",
    }
    assert set(tools.get_tool_names()) == expected

    confirming = {name for name, entry in tools.TOOLS.items() if entry["requires_confirmation"]}
    assert confirming == {"write_file", "shell_exec"}


# ---------------------------------------------------------------------------
# ToolContext threading
# ---------------------------------------------------------------------------


class TestToolContextDivergence:
    """When ``ctx.working_dir`` differs from the process CWD, every
    path/cwd-sensitive tool must operate against ``ctx.working_dir``.

    These tests are the primary guardrail: they would fail if a tool fell
    back to ``Path.cwd()`` even when an explicit ctx was supplied.
    """

    def test_read_file_uses_ctx_not_global(self, tmp_path, monkeypatch):
        a = tmp_path / "global-A"
        b = tmp_path / "ctx-B"
        a.mkdir()
        b.mkdir()
        (a / "shared.txt").write_text("from-A")
        (b / "shared.txt").write_text("from-B")
        monkeypatch.chdir(a)
        ctx = tools.ToolContext(working_dir=b)
        assert tools.execute_tool("read_file", {"path": "shared.txt"}, ctx=ctx) == "from-B"

    def test_write_file_uses_ctx_not_global(self, tmp_path, monkeypatch):
        a = tmp_path / "global-A"
        b = tmp_path / "ctx-B"
        a.mkdir()
        b.mkdir()
        monkeypatch.chdir(a)
        ctx = tools.ToolContext(working_dir=b)
        tools.execute_tool("write_file", {"path": "out.txt", "content": "hi"}, ctx=ctx)
        assert (b / "out.txt").read_text() == "hi"
        assert not (a / "out.txt").exists()

    def test_list_directory_uses_ctx_not_global(self, tmp_path, monkeypatch):
        a = tmp_path / "global-A"
        b = tmp_path / "ctx-B"
        a.mkdir()
        b.mkdir()
        (a / "in-a.txt").touch()
        (b / "in-b.txt").touch()
        monkeypatch.chdir(a)
        ctx = tools.ToolContext(working_dir=b)
        result = tools.execute_tool("list_directory", {"path": "."}, ctx=ctx)
        assert "in-b.txt" in result
        assert "in-a.txt" not in result

    def test_search_code_uses_ctx_for_relative_paths(self, tmp_path, monkeypatch):
        a = tmp_path / "global-A"
        b = tmp_path / "ctx-B"
        a.mkdir()
        b.mkdir()
        (b / "code.py").write_text("def needle():\n    pass\n")
        monkeypatch.chdir(a)
        ctx = tools.ToolContext(working_dir=b)
        result = tools.execute_tool("search_code", {"pattern": "needle", "file_glob": "*.py"}, ctx=ctx)
        # Match must reference the file under ctx.working_dir (relative path),
        # not under cwd. If search_code consulted Path.cwd() instead, it would
        # either find nothing or render an absolute path under A.
        assert "code.py" in result

    def test_shell_exec_uses_ctx_cwd(self, tmp_path, monkeypatch):
        a = tmp_path / "global-A"
        b = tmp_path / "ctx-B"
        a.mkdir()
        b.mkdir()
        monkeypatch.chdir(a)
        ctx = tools.ToolContext(working_dir=b)
        result = tools.execute_tool("shell_exec", {"command": "pwd"}, ctx=ctx)
        # macOS resolves /tmp through /private/tmp; compare resolved paths.
        assert str(b.resolve()) in str(Path(result.strip()).resolve())

    def test_str_replace_via_write_file_uses_ctx(self, tmp_path, monkeypatch):
        # No dedicated str_replace tool in this repo — write_file covers the
        # write path; this test layer exercises the read+write round-trip
        # purely through ctx, asserting the global is not consulted.
        a = tmp_path / "global-A"
        b = tmp_path / "ctx-B"
        a.mkdir()
        b.mkdir()
        (b / "doc.txt").write_text("before")
        monkeypatch.chdir(a)
        ctx = tools.ToolContext(working_dir=b)
        assert tools.execute_tool("read_file", {"path": "doc.txt"}, ctx=ctx) == "before"
        tools.execute_tool("write_file", {"path": "doc.txt", "content": "after"}, ctx=ctx)
        assert (b / "doc.txt").read_text() == "after"


class TestToolContextSchema:
    """``ctx`` is a reserved internal kwarg and must never appear in the
    model-facing JSON schema."""

    def test_no_tool_schema_declares_ctx(self, tmp_path):
        for name, entry in tools.TOOLS.items():
            params = entry["definition"]["function"]["parameters"]
            properties = params.get("properties", {})
            assert "ctx" not in properties, f"tool {name!r} leaked 'ctx' into its JSON schema"
            assert "ctx" not in params.get("required", [])

    def test_tool_decorator_rejects_ctx_in_schema(self, tmp_path):
        with pytest.raises(ValueError, match="reserved internal kwarg"):

            @tools.tool(
                requires_confirmation=False,
                parameters={
                    "type": "object",
                    "properties": {"ctx": {"type": "string"}},
                    "required": [],
                },
            )
            def _bad():
                """noop"""
                return ""

    def test_tool_decorator_rejects_ctx_in_required(self, tmp_path):
        with pytest.raises(ValueError, match="reserved internal kwarg"):

            @tools.tool(
                requires_confirmation=False,
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": ["ctx"],
                },
            )
            def _bad_required():
                """noop"""
                return ""


class TestToolContextInjectionGuard:
    """A model-supplied ``"ctx"`` in arguments must never override the real
    ctx threaded by the dispatcher."""

    def test_caller_supplied_ctx_in_arguments_is_stripped(self, tmp_path, monkeypatch):
        a = tmp_path / "global-A"
        b = tmp_path / "real-ctx-B"
        a.mkdir()
        b.mkdir()
        (b / "real.txt").write_text("real-ctx-content")
        monkeypatch.chdir(a)
        real_ctx = tools.ToolContext(working_dir=b)

        # The model-style payload tries to smuggle a forged "ctx". The
        # dispatcher must drop it and use real_ctx instead.
        result = tools.execute_tool(
            "read_file",
            {"ctx": "i-am-not-a-toolcontext", "path": "real.txt"},
            ctx=real_ctx,
        )
        assert result == "real-ctx-content"

    def test_caller_supplied_ctx_does_not_break_validation(self, tmp_path, monkeypatch):
        # Even when no real ctx is passed, the smuggled key must not trip the
        # validator or surface as a TypeError from the handler.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "f.txt").write_text("ok")
        result = tools.execute_tool(
            "read_file",
            {"ctx": {"working_dir": "/etc"}, "path": "f.txt"},
        )
        assert result == "ok"

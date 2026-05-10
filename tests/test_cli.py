"""Tests for cli.py — command parsing edge cases."""

import json
from pathlib import Path
from unittest.mock import ANY, patch

import pytest

from baremetal_agent import agent, cli, tools


class TestModelCommand:
    def test_bare_model_shows_current(self, capsys, monkeypatch):
        """Typing 'model' alone should show current model, not send to LLM."""
        monkeypatch.setenv("AGENT_MODEL", "test-model-123")

        # Simulate: user types "model", then "exit"
        inputs = iter(["model", "exit"])
        with patch("builtins.input", side_effect=inputs):
            cli.run()

        output = capsys.readouterr().out
        assert "Current model: test-model-123" in output

    def test_model_with_name_switches(self, capsys, monkeypatch):
        """Typing 'model foo' should switch model."""
        monkeypatch.setenv("AGENT_MODEL", "old-model")

        inputs = iter(["model new-model", "exit"])
        with patch("builtins.input", side_effect=inputs):
            cli.run()

        output = capsys.readouterr().out
        assert "old-model → new-model" in output


class TestVerboseCommand:
    def test_verbose_toggles_both_render_and_log_payloads(self, capsys, monkeypatch):
        """REPL `verbose` is a one-knob alias that flips both flags together."""
        monkeypatch.delenv("AGENT_VERBOSE", raising=False)
        monkeypatch.delenv("AGENT_RENDER_VERBOSE", raising=False)
        monkeypatch.delenv("AGENT_LOG_PAYLOADS", raising=False)

        seen: list[tuple[bool, bool]] = []

        original_make_renderer = cli.make_renderer

        def spy_make_renderer(cfg):
            seen.append((cfg.render_verbose, cfg.log_payloads))
            return original_make_renderer(cfg)

        inputs = iter(["verbose", "verbose", "exit"])
        with (
            patch("builtins.input", side_effect=inputs),
            patch.object(cli, "make_renderer", side_effect=spy_make_renderer),
            patch.object(agent, "run_agent_turn", side_effect=AssertionError("verbose called agent")),
        ):
            result = cli.run()

        output = capsys.readouterr().out
        assert result == 0
        # Initial render at startup, then after first toggle, then after second.
        assert seen[0] == (False, False)
        assert seen[1] == (True, True)
        assert seen[2] == (False, False)
        assert "Verbose: on (raw API payloads)" in output
        assert "Verbose: off (rich visualization)" in output

    def test_verbose_collapses_drifted_state(self, capsys, monkeypatch):
        """If only one flag is on (e.g. via env), `verbose` collapses both to the inverse."""
        from baremetal_agent.config import AgentConfig

        cfg = AgentConfig(
            model="test-model",
            api_url="https://example.test/chat/completions",
            max_iterations=10,
            working_dir=Path("/tmp/baremetal-test"),
            render_verbose=True,  # drifted: only renderer on
            log_payloads=False,
            stream=False,
            system_prompt="test",
        )

        seen: list[tuple[bool, bool]] = []
        original_make_renderer = cli.make_renderer

        def spy_make_renderer(c):
            seen.append((c.render_verbose, c.log_payloads))
            return original_make_renderer(c)

        inputs = iter(["verbose", "exit"])
        with (
            patch("builtins.input", side_effect=inputs),
            patch.object(cli, "make_renderer", side_effect=spy_make_renderer),
            patch.object(agent, "run_agent_turn", side_effect=AssertionError("verbose called agent")),
        ):
            cli.run(cfg=cfg)

        # Startup render then post-toggle: not render_verbose => both False.
        assert seen[0] == (True, False)
        assert seen[1] == (False, False)


class TestStreamCommand:
    def test_stream_toggles_on_and_off_without_calling_agent(self, capsys, monkeypatch):
        """Typing 'stream' should be a local REPL command, not a model prompt."""
        monkeypatch.setenv("AGENT_STREAM", "0")
        inputs = iter(["stream", "stream", "exit"])

        with (
            patch("builtins.input", side_effect=inputs),
            patch.object(agent, "run_agent_turn", side_effect=AssertionError("stream command called agent")),
        ):
            result = cli.run()

        output = capsys.readouterr().out
        assert result == 0
        assert "Streaming: on" in output
        assert "Streaming: off" in output

    def test_help_lists_stream_toggle(self, capsys):
        """Help should make the REPL streaming toggle discoverable."""
        inputs = iter(["help", "exit"])

        with patch("builtins.input", side_effect=inputs):
            cli.run()

        output = capsys.readouterr().out
        assert "stream" in output
        assert "Toggle streaming" in output


class TestOneShotCli:
    def test_main_with_no_args_launches_repl(self):
        """No CLI args should preserve the current interactive REPL entry."""
        with patch.object(cli, "run", return_value=0) as run:
            result = cli.main([])

        assert result == 0
        assert run.call_count == 1

    def test_prompt_runs_one_turn_and_prints_final_response(self, capsys):
        """One-shot mode should print only the final assistant text."""

        def fake_turn(user_message, history, api_responses, **_kwargs):
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "hello back"})
            return agent.AgentTurnResult(content="hello back", status=agent.STATUS_OK)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["-p", "hello"])

        assert result == 0
        assert capsys.readouterr().out == "hello back\n"

    def test_prompt_keeps_agent_diagnostics_off_stdout(self, capsys):
        """One-shot stdout should remain safe for shell pipelines."""

        def fake_turn(user_message, history, api_responses, **_kwargs):
            print("diagnostic from lower layer")
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "final only"})
            return agent.AgentTurnResult(content="final only", status=agent.STATUS_OK)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["-p", "hello"])

        captured = capsys.readouterr()
        assert result == 0
        assert captured.out == "final only\n"
        assert captured.err == "diagnostic from lower layer\n"

    def test_prompt_stream_flag_opts_one_shot_into_streaming(self):
        """--stream should opt a one-shot prompt into the streaming agent path."""
        seen = {}

        def fake_turn(user_message, history, api_responses, **kwargs):
            seen["stream"] = kwargs["cfg"].stream
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "streamed"})
            return agent.AgentTurnResult(content="streamed", status=agent.STATUS_OK)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["--stream", "-p", "hello"])

        assert result == 0
        assert seen["stream"] is True

    def test_prompt_uses_agent_stream_env_by_default(self, monkeypatch):
        """AGENT_STREAM=1 should opt one-shot prompts into streaming without a flag."""
        monkeypatch.setenv("AGENT_STREAM", "1")
        seen = {}

        def fake_turn(user_message, history, api_responses, **kwargs):
            seen["stream"] = kwargs["cfg"].stream
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "streamed"})
            return agent.AgentTurnResult(content="streamed", status=agent.STATUS_OK)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["-p", "hello"])

        assert result == 0
        assert seen["stream"] is True

    def test_prompt_file_reads_prompt_text(self, tmp_path):
        """--prompt-file should read UTF-8 prompt text and pass it to the agent."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("from file\n", encoding="utf-8")
        seen = {}

        def fake_turn(user_message, history, api_responses, **_kwargs):
            seen["prompt"] = user_message
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "ok"})
            return agent.AgentTurnResult(content="ok", status=agent.STATUS_OK)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["--prompt-file", str(prompt_file)])

        assert result == 0
        assert seen["prompt"] == "from file\n"

    def test_prompt_file_decode_error_exits_cleanly(self, tmp_path, capsys):
        """A non-UTF-8 prompt file should fail like an invalid CLI argument."""
        prompt_file = tmp_path / "prompt.bin"
        prompt_file.write_bytes(b"\xff")

        with pytest.raises(SystemExit) as exc:
            cli.main(["--prompt-file", str(prompt_file)])

        captured = capsys.readouterr()
        assert exc.value.code == 2
        assert captured.out == ""
        assert "could not read prompt file" in captured.err

    def test_prompt_and_prompt_file_are_mutually_exclusive(self):
        """Argparse should reject ambiguous prompt sources."""
        with pytest.raises(SystemExit) as exc:
            cli.main(["--prompt", "hi", "--prompt-file", "prompt.txt"])

        assert exc.value.code == 2

    def test_trajectory_out_writes_atif_for_one_shot(self, tmp_path):
        """One-shot mode can persist the same ATIF format as the REPL command."""
        out = tmp_path / "trajectory.json"

        def fake_turn(user_message, history, api_responses, **_kwargs):
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "done"})
            api_responses.append({"created": 1700000000, "usage": {"prompt_tokens": 2, "completion_tokens": 3}})
            return agent.AgentTurnResult(content="done", status=agent.STATUS_OK)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["--prompt", "hello", "--trajectory-out", str(out)])

        assert result == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["schema_version"] == "ATIF-v1.4"
        assert data["final_metrics"]["total_prompt_tokens"] == 2
        assert data["final_metrics"]["total_completion_tokens"] == 3

    def test_api_error_status_returns_nonzero(self, capsys):
        """Script callers need a non-zero exit when the model call fails."""

        def fake_turn(user_message, history, api_responses, **_kwargs):
            history.append({"role": "user", "content": user_message})
            return agent.AgentTurnResult(content="API error: boom", status=agent.STATUS_API_ERROR)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["--prompt", "hello"])

        captured = capsys.readouterr()
        assert result == 1
        assert captured.out == ""
        assert captured.err == "API error: boom\n"

    def test_max_iterations_status_returns_nonzero(self):
        """Max-iteration failures should be distinct from successful final answers."""

        def fake_turn(user_message, history, api_responses, **_kwargs):
            history.append({"role": "user", "content": user_message})
            return agent.AgentTurnResult(
                content="Reached maximum iteration limit (1).",
                status=agent.STATUS_MAX_ITERATIONS,
            )

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["--prompt", "hello"])

        assert result == 1

    def test_default_one_shot_denies_dangerous_tools(self):
        """One-shot mode should not prompt or execute dangerous tools by default."""
        seen = {}

        def fake_turn(_user_message, _history, _api_responses, **kwargs):
            seen["confirmer"] = kwargs["confirmer"]
            return agent.AgentTurnResult(content="ok", status=agent.STATUS_OK)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["--prompt", "hello"])

        assert result == 0
        assert seen["confirmer"] is agent.auto_deny_confirmer

    def test_allow_dangerous_tools_switches_confirmation_policy(self):
        """The opt-in flag should permit confirmation-required tools in one-shot mode."""
        seen = {}

        def fake_turn(_user_message, _history, _api_responses, **kwargs):
            seen["confirmer"] = kwargs["confirmer"]
            return agent.AgentTurnResult(content="ok", status=agent.STATUS_OK)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["--prompt", "hello", "--allow-dangerous-tools"])

        assert result == 0
        assert seen["confirmer"] is agent.auto_approve_confirmer

    def test_read_only_exposes_only_non_confirmation_tools(self):
        """--read-only should hide write/exec tools from the model."""
        seen = {}

        def fake_turn(_user_message, _history, _api_responses, **kwargs):
            seen["tool_names"] = kwargs["tool_names"]
            return agent.AgentTurnResult(content="ok", status=agent.STATUS_OK)

        with patch.object(agent, "run_agent_turn", side_effect=fake_turn):
            result = cli.main(["--prompt", "hello", "--read-only"])

        assert result == 0
        assert seen["tool_names"] == tools.get_read_only_tool_names()
        assert "write_file" not in seen["tool_names"]
        assert "shell_exec" not in seen["tool_names"]

    def test_read_only_and_allow_dangerous_are_mutually_exclusive(self):
        """Read-only mode and dangerous-tool allowance are contradictory."""
        with pytest.raises(SystemExit) as exc:
            cli.main(["--prompt", "hello", "--read-only", "--allow-dangerous-tools"])

        assert exc.value.code == 2

    def test_stream_requires_one_shot_prompt(self):
        """--stream has no standalone behavior outside one-shot mode."""
        with pytest.raises(SystemExit) as exc:
            cli.main(["--stream"])

        assert exc.value.code == 2


class TestReplayCli:
    def test_replay_subcommand_renders_trajectory_step(self, tmp_path, capsys):
        """Replay should inspect saved ATIF without invoking the agent loop."""
        path = tmp_path / "trajectory.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "ATIF-v1.4",
                    "session_id": "offline-session",
                    "steps": [
                        {"step_id": 1, "source": "user", "message": "hello"},
                        {"step_id": 2, "source": "agent", "message": "hello back"},
                    ],
                    "final_metrics": {
                        "total_prompt_tokens": 1,
                        "total_completion_tokens": 2,
                        "total_cached_tokens": 0,
                        "total_steps": 2,
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch.object(agent, "run_agent_turn") as run_turn:
            result = cli.main(["replay", str(path), "--step", "2"])

        captured = capsys.readouterr()
        assert result == 0
        assert "Trajectory offline-session (ATIF-v1.4)" in captured.out
        assert "Step 2 [agent]" in captured.out
        assert "hello back" in captured.out
        assert captured.err == ""
        run_turn.assert_not_called()

    def test_replay_subcommand_reports_load_errors_to_stderr(self, tmp_path, capsys):
        """Replay errors should be script-friendly and return non-zero."""
        path = tmp_path / "trajectory.json"
        path.write_text(json.dumps({"schema_version": "ATIF-v1.3", "steps": []}), encoding="utf-8")

        result = cli.main(["replay", str(path)])

        captured = capsys.readouterr()
        assert result == 1
        assert captured.out == ""
        assert "replay: unsupported trajectory schema_version" in captured.err

    def test_replay_rejects_empty_prompt_option(self, tmp_path):
        """Replay should reject one-shot options even when their value is falsey."""
        path = tmp_path / "trajectory.json"
        path.write_text(json.dumps({"schema_version": "ATIF-v1.4", "steps": []}), encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            cli.main(["--prompt", "", "replay", str(path)])

        assert exc.value.code == 2

    def test_replay_rejects_stream_option(self, tmp_path):
        """Replay should reject the one-shot streaming option."""
        path = tmp_path / "trajectory.json"
        path.write_text(json.dumps({"schema_version": "ATIF-v1.4", "steps": []}), encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            cli.main(["--stream", "replay", str(path)])

        assert exc.value.code == 2


class TestEvalCli:
    def test_eval_subcommand_routes_to_eval_suite(self):
        """Eval should route to the eval suite and return its exit code."""
        with patch.object(cli.eval_harness, "run_eval_suite", return_value=3) as run_eval_suite:
            result = cli.main(["eval", "--tasks", "custom/tasks", "--out", "out.md", "--json-out", "out.json"])

        assert result == 3
        run_eval_suite.assert_called_once_with("custom/tasks", "out.md", "out.json", cfg=ANY, workers=1)

    def test_eval_subcommand_uses_default_paths(self):
        """Eval should provide default paths when flags are omitted."""
        with patch.object(cli.eval_harness, "run_eval_suite", return_value=0) as run_eval_suite:
            result = cli.main(["eval"])

        assert result == 0
        run_eval_suite.assert_called_once_with("evals/tasks", "report.md", "report.json", cfg=ANY, workers=1)

    def test_eval_subcommand_passes_workers_flag(self):
        """``--workers`` should be forwarded to ``run_eval_suite``."""
        with patch.object(cli.eval_harness, "run_eval_suite", return_value=0) as run_eval_suite:
            result = cli.main(["eval", "--workers", "4"])

        assert result == 0
        run_eval_suite.assert_called_once_with("evals/tasks", "report.md", "report.json", cfg=ANY, workers=4)

    @pytest.mark.parametrize("bad_value", ["0", "-1", "-100"])
    def test_eval_rejects_non_positive_workers(self, bad_value):
        """``--workers`` must be >= 1; non-positive values exit with code 2."""
        with (
            patch.object(cli.eval_harness, "run_eval_suite", return_value=0) as run_eval_suite,
            pytest.raises(SystemExit) as exc,
        ):
            cli.main(["eval", "--workers", bad_value])

        assert exc.value.code == 2
        run_eval_suite.assert_not_called()

    def test_eval_rejects_empty_prompt_option(self):
        """Eval should reject one-shot options even when their value is falsey."""
        with pytest.raises(SystemExit) as exc:
            cli.main(["--prompt", "", "eval"])

        assert exc.value.code == 2

    def test_eval_rejects_stream_option(self):
        """Eval should reject the one-shot streaming option."""
        with pytest.raises(SystemExit) as exc:
            cli.main(["--stream", "eval"])

        assert exc.value.code == 2

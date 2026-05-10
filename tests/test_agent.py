"""Tests for agent.py — agent loop edge cases."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from baremetal_agent import agent, tools, visualizer
from baremetal_agent.agent import Message


@pytest.fixture
def cfg(make_cfg):
    return make_cfg()


def _kwargs(**extra):
    """Default required kwargs for run_agent_turn, with overrides."""
    base = {"renderer": visualizer.NullRenderer(), "confirmer": agent.auto_deny_confirmer}
    base.update(extra)
    return base


class TestRunAgentTurn:
    def test_empty_choices_returns_error(self, monkeypatch, cfg):
        """API response with empty choices should return error, not crash."""
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []

        empty_choices_response = {"choices": [], "usage": {}}
        with patch.object(agent.client, "chat_completion", return_value=empty_choices_response):
            result = agent.run_agent_turn("hello", history, api_responses, cfg=cfg, **_kwargs())

        assert "no choices" in result.content.lower()
        assert len(history) == 1

    def test_missing_choices_key_returns_error(self, monkeypatch, cfg):
        """API response missing 'choices' key should return error, not crash."""
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []

        bad_response = {"usage": {}}
        with patch.object(agent.client, "chat_completion", return_value=bad_response):
            result = agent.run_agent_turn("hello", history, api_responses, cfg=cfg, **_kwargs())

        assert "no choices" in result.content.lower()
        assert len(history) == 1

    def test_api_responses_appended_to_passed_list(self, monkeypatch, cfg):
        """Responses are appended to the caller-owned list, not module state."""
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []

        response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {},
        }
        with patch.object(agent.client, "chat_completion", return_value=response):
            agent.run_agent_turn("hello", history, api_responses, cfg=cfg, **_kwargs())

        assert len(api_responses) == 1

    def test_api_responses_rolled_back_on_error(self, monkeypatch, cfg):
        """Responses appended during a failed turn are removed on rollback."""
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []

        with patch.object(agent.client, "chat_completion", side_effect=RuntimeError("boom")):
            agent.run_agent_turn("hello", history, api_responses, cfg=cfg, **_kwargs())

        assert len(api_responses) == 0

    def test_no_module_level_api_responses(self):
        """Module-level mutable api_responses list must not exist."""
        assert not hasattr(agent, "api_responses")

    def test_two_turns_use_independent_lists(self, monkeypatch, cfg):
        """Two callers with separate lists don't share state."""
        response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {},
        }

        list_a: list[dict] = []
        list_b: list[dict] = []
        history_a: list[Message] = [{"role": "system", "content": "sys"}]
        history_b: list[Message] = [{"role": "system", "content": "sys"}]

        with patch.object(agent.client, "chat_completion", return_value=response):
            agent.run_agent_turn("hello", history_a, list_a, cfg=cfg, **_kwargs())
            agent.run_agent_turn("hello", history_b, list_b, cfg=cfg, **_kwargs())

        assert len(list_a) == 1
        assert len(list_b) == 1
        assert list_a is not list_b

    def test_noninteractive_deny_policy_denies_dangerous_tool_without_input(self, monkeypatch, cfg):
        """One-shot default must deny confirmation tools without prompting."""
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []

        tool_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_write",
                                "function": {
                                    "name": "write_file",
                                    "arguments": '{"path": "out.txt", "content": "hello"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        final_response = {
            "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {},
        }

        with (
            patch.object(agent.client, "chat_completion", side_effect=[tool_response, final_response]),
            patch("builtins.input", side_effect=AssertionError("input() must not be called")),
        ):
            result = agent.run_agent_turn(
                "write it",
                history,
                api_responses,
                confirmer=agent.auto_deny_confirmer,
                renderer=visualizer.NullRenderer(),
                cfg=cfg,
            )

        assert result.status == agent.STATUS_OK
        denied_messages = [msg for msg in history if msg["role"] == "tool"]
        assert denied_messages[0]["content"] == "Tool execution denied."

    def test_allow_policy_executes_dangerous_tool_without_prompting(self, monkeypatch, cfg):
        """Explicit one-shot opt-in should execute confirmation tools without input()."""
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []

        tool_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_shell",
                                "function": {"name": "shell_exec", "arguments": '{"command": "echo hi"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        final_response = {
            "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {},
        }

        with (
            patch.object(agent.client, "chat_completion", side_effect=[tool_response, final_response]),
            patch.object(agent.tools, "execute_tool", return_value="executed") as execute_tool,
            patch("builtins.input", side_effect=AssertionError("input() must not be called")),
        ):
            result = agent.run_agent_turn(
                "run it",
                history,
                api_responses,
                confirmer=agent.auto_approve_confirmer,
                renderer=visualizer.NullRenderer(),
                cfg=cfg,
            )

        assert result.status == agent.STATUS_OK
        execute_tool.assert_called_once_with(
            "shell_exec", {"command": "echo hi"}, ctx=tools.ToolContext(working_dir=cfg.working_dir)
        )

    def test_tool_name_filter_limits_exposed_tool_definitions(self, monkeypatch, cfg):
        """Callers should be able to expose a read-only subset without mutating TOOLS."""
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []
        response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {},
        }

        with patch.object(agent.client, "chat_completion", return_value=response) as chat_completion:
            agent.run_agent_turn(
                "hello",
                history,
                api_responses,
                tool_names=["read_file", "git_status"],
                renderer=visualizer.NullRenderer(),
                confirmer=agent.auto_deny_confirmer,
                cfg=cfg,
            )

        exposed = [tool["function"]["name"] for tool in chat_completion.call_args.args[1]]
        assert exposed == ["read_file", "git_status"]

    def test_tool_name_filter_denies_unadvertised_tool_execution(self, monkeypatch, cfg):
        """Tool filtering should be enforced when executing model-returned calls."""
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []
        tool_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_write",
                                "function": {
                                    "name": "write_file",
                                    "arguments": '{"path": "out.txt", "content": "hello"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        final_response = {
            "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {},
        }

        with (
            patch.object(agent.client, "chat_completion", side_effect=[tool_response, final_response]),
            patch.object(agent.tools, "execute_tool", return_value="executed") as execute_tool,
            patch("builtins.input", side_effect=AssertionError("input() must not be called")),
        ):
            result = agent.run_agent_turn(
                "write it",
                history,
                api_responses,
                confirmer=agent.auto_approve_confirmer,
                tool_names=["read_file"],
                renderer=visualizer.NullRenderer(),
                cfg=cfg,
            )

        assert result.status == agent.STATUS_OK
        execute_tool.assert_not_called()
        denied_messages = [msg for msg in history if msg["role"] == "tool"]
        assert denied_messages[0]["content"] == "Tool execution denied by policy: write_file is not enabled."

    def test_max_iteration_failure_has_structured_status(self, monkeypatch, cfg):
        """Script mode needs a reliable max-iteration status for exit codes."""
        cfg = replace(cfg, max_iterations=1)
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []
        tool_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_read",
                                "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }

        with (
            patch.object(agent.client, "chat_completion", return_value=tool_response),
            patch.object(agent.tools, "execute_tool", return_value="contents"),
        ):
            result = agent.run_agent_turn(
                "keep going",
                history,
                api_responses,
                confirmer=agent.auto_approve_confirmer,
                renderer=visualizer.NullRenderer(),
                cfg=cfg,
            )

        assert result.status == agent.STATUS_MAX_ITERATIONS
        assert "maximum iteration limit" in result.content

    def test_streaming_path_uses_stream_client_and_preserves_history_shape(self, monkeypatch, cfg):
        """Streaming should assemble into the same message shape before tool execution."""
        cfg = replace(cfg, stream=True)
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []
        tool_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_read",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        final_response = {
            "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {},
        }

        with (
            patch.object(agent.client, "chat_completion") as chat_completion,
            patch.object(agent.client, "chat_completion_stream", side_effect=[tool_response, final_response]),
            patch.object(agent.tools, "execute_tool", return_value="contents") as execute_tool,
            patch("builtins.input", side_effect=AssertionError("input() must not be called")),
        ):
            result = agent.run_agent_turn(
                "read it",
                history,
                api_responses,
                confirmer=agent.auto_approve_confirmer,
                renderer=visualizer.NullRenderer(),
                cfg=cfg,
            )

        assert result.status == agent.STATUS_OK
        chat_completion.assert_not_called()
        execute_tool.assert_called_once_with(
            "read_file", {"path": "README.md"}, ctx=tools.ToolContext(working_dir=cfg.working_dir)
        )
        assert history[2] == {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
                }
            ],
            "_model": cfg.model,
        }

    def test_streaming_waits_for_complete_tool_call_before_confirmation(self, monkeypatch, cfg):
        """Dangerous tools are confirmed only after the streamed arguments are complete."""
        cfg = replace(cfg, stream=True)
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []
        tool_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_shell",
                                "type": "function",
                                "function": {"name": "shell_exec", "arguments": '{"command": "echo hi"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        final_response = {
            "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {},
        }

        confirmer_calls = []

        def confirmer(name, args):
            confirmer_calls.append((name, args))
            return False

        with (
            patch.object(agent.client, "chat_completion_stream", side_effect=[tool_response, final_response]),
            patch.object(agent.tools, "execute_tool") as execute_tool,
        ):
            result = agent.run_agent_turn(
                "run it",
                history,
                api_responses,
                confirmer=confirmer,
                renderer=visualizer.NullRenderer(),
                cfg=cfg,
            )

        assert result.status == agent.STATUS_OK
        assert confirmer_calls == [("shell_exec", {"command": "echo hi"})]
        execute_tool.assert_not_called()


class TestDispatchToolCall:
    """Direct tests for the per-tool-call dispatch helper."""

    def _call(self, name, args_json, call_id="call_1"):
        return {"id": call_id, "function": {"name": name, "arguments": args_json}}

    def test_read_only_filter_denies_unadvertised_tool(self):
        result, tool_msg = agent._dispatch_tool_call(
            self._call("write_file", '{"path": "x", "content": "y"}'),
            confirmer=agent.auto_approve_confirmer,
            allowed_tool_names={"read_file"},
        )
        assert result["denied"] is True
        assert result["name"] == "write_file"
        assert "not enabled" in result["result"]
        assert tool_msg == {"role": "tool", "tool_call_id": "call_1", "content": result["result"]}

    def test_dangerous_tool_denied_by_policy(self):
        with patch.object(agent.tools, "execute_tool") as execute_tool:
            result, tool_msg = agent._dispatch_tool_call(
                self._call("write_file", '{"path": "x", "content": "y"}'),
                confirmer=agent.auto_deny_confirmer,
                allowed_tool_names=None,
            )
        execute_tool.assert_not_called()
        assert result["denied"] is True
        assert result["result"] == "Tool execution denied."
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"

    def test_invalid_json_arguments_returns_error_result(self):
        result, tool_msg = agent._dispatch_tool_call(
            self._call("read_file", "{not json"),
            confirmer=agent.auto_approve_confirmer,
            allowed_tool_names=None,
        )
        assert result["denied"] is False
        assert result["args"] == {}
        assert "Could not parse arguments" in result["result"]
        assert tool_msg["content"] == result["result"]

    def test_non_object_arguments_returns_error_result(self):
        result, _ = agent._dispatch_tool_call(
            self._call("read_file", "[1, 2, 3]"),
            confirmer=agent.auto_approve_confirmer,
            allowed_tool_names=None,
        )
        assert "must be a JSON object" in result["result"]
        assert result["args"] == {}

    def test_dangerous_tool_denied_by_user_via_confirmer(self):
        confirmer_calls = []

        def confirmer(name, args):
            confirmer_calls.append((name, args))
            return False

        result, tool_msg = agent._dispatch_tool_call(
            self._call("write_file", '{"path": "x", "content": "y"}'),
            confirmer=confirmer,
            allowed_tool_names=None,
        )
        assert confirmer_calls == [("write_file", {"path": "x", "content": "y"})]
        assert result["denied"] is True
        assert result["result"] == "Tool execution denied."
        assert tool_msg["content"] == "Tool execution denied."

    def test_successful_execution_records_duration_and_result(self):
        with patch.object(agent.tools, "execute_tool", return_value="contents") as execute_tool:
            result, tool_msg = agent._dispatch_tool_call(
                self._call("read_file", '{"path": "README.md"}'),
                confirmer=agent.auto_approve_confirmer,
                allowed_tool_names={"read_file"},
            )
        execute_tool.assert_called_once_with("read_file", {"path": "README.md"}, ctx=None)
        assert result["denied"] is False
        assert result["result"] == "contents"
        assert result["duration_ms"] >= 0
        assert tool_msg["content"] == "contents"


class TestInjectionSilentAndSafe:
    """End-to-end: NullRenderer + auto_deny_confirmer => no output, no execution."""

    def test_null_renderer_and_auto_deny_silence_and_block_dangerous_tools(self, capsys, cfg):
        history: list[Message] = [{"role": "system", "content": "sys"}]
        api_responses: list[dict] = []
        tool_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_shell",
                                "function": {"name": "shell_exec", "arguments": '{"command": "rm -rf /"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        final_response = {
            "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {},
        }

        with (
            patch.object(agent.client, "chat_completion", side_effect=[tool_response, final_response]),
            patch.object(agent.tools, "execute_tool") as execute_tool,
            patch("builtins.input", side_effect=AssertionError("input() must not be called")),
        ):
            result = agent.run_agent_turn(
                "delete everything",
                history,
                api_responses,
                cfg=cfg,
                renderer=visualizer.NullRenderer(),
                confirmer=agent.auto_deny_confirmer,
            )

        # No dangerous execution, no terminal output.
        execute_tool.assert_not_called()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert result.status == agent.STATUS_OK
        denied = [m for m in history if m["role"] == "tool"]
        assert denied[0]["content"] == "Tool execution denied."


class TestMessageTypedDict:
    def test_system_message(self):
        msg: Message = {"role": "system", "content": "you are an agent"}
        assert msg["role"] == "system"

    def test_user_message(self):
        msg: Message = {"role": "user", "content": "hello"}
        assert msg["role"] == "user"

    def test_tool_message(self):
        msg: Message = {"role": "tool", "tool_call_id": "call_123", "content": "result"}
        assert msg["role"] == "tool"

"""Tests for trajectory.py — ATIF export and version consistency."""

import json

from baremetal_agent import __version__, safety, trajectory


class TestTrajectoryExport:
    def test_empty_history_produces_valid_atif(self):
        history = [{"role": "system", "content": "You are helpful."}]
        atif = trajectory.history_to_atif(history, [], "test-model")
        assert atif["schema_version"] == "ATIF-v1.4"
        assert atif["agent"]["name"] == "baremetal-agent"
        assert atif["steps"] == []
        assert atif["final_metrics"]["total_steps"] == 0

    def test_version_matches_package(self):
        atif = trajectory.history_to_atif([], [], "test-model")
        assert atif["agent"]["version"] == __version__

    def test_user_message_becomes_step(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        atif = trajectory.history_to_atif(history, [], "test-model")
        assert len(atif["steps"]) == 1
        assert atif["steps"][0]["source"] == "user"
        assert atif["steps"][0]["message"] == "hello"

    def test_assistant_text_response(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello back"},
        ]
        api_responses = [{"created": 1700000000, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}]
        atif = trajectory.history_to_atif(history, api_responses, "test-model")
        agent_step = [s for s in atif["steps"] if s["source"] == "agent"][0]
        assert agent_step["message"] == "hello back"
        assert agent_step["metrics"]["prompt_tokens"] == 10

    def test_tool_call_and_observation(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
        ]
        api_responses = [{"created": 1700000000, "usage": {"prompt_tokens": 20, "completion_tokens": 10}}]
        atif = trajectory.history_to_atif(history, api_responses, "test-model")
        agent_step = [s for s in atif["steps"] if s["source"] == "agent"][0]
        assert len(agent_step["tool_calls"]) == 1
        assert agent_step["tool_calls"][0]["function_name"] == "read_file"
        assert agent_step["observation"]["results"][0]["content"] == "file contents"

    def test_history_to_atif_redacts_user_and_assistant_messages(self):
        user_secret = "token=abcdef123456"
        assistant_secret = "password=abcdef123456"
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"my token is {user_secret}"},
            {"role": "assistant", "content": f"found {assistant_secret}"},
        ]
        api_responses = [{"created": 1700000000, "usage": {}}]

        atif = trajectory.history_to_atif(history, api_responses, "test-model")
        serialized = json.dumps(atif)

        assert user_secret not in serialized
        assert assistant_secret not in serialized
        assert "token=abcdef******" in serialized
        assert "password=abcdef******" in serialized

    def test_history_to_atif_redacts_tool_arguments_recursively(self):
        raw_secret = "sk-abcdef123456"
        history = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search_code",
                            "arguments": json.dumps({"pattern": raw_secret, "nested": ["token=abcdef123456"]}),
                        },
                    }
                ],
            },
        ]

        atif = trajectory.history_to_atif(history, [], "test-model")
        args = atif["steps"][0]["tool_calls"][0]["arguments"]

        assert raw_secret not in json.dumps(args)
        assert "token=abcdef123456" not in json.dumps(args)
        assert args["pattern"] == "sk-abcdef******"
        assert args["nested"] == ["token=abcdef******"]

    def test_history_to_atif_redacts_and_truncates_observations(self):
        raw_secret = "password=abcdef123456"
        history = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": raw_secret + "\n" + ("x" * safety.DEFAULT_MAX_CHARS)},
        ]

        atif = trajectory.history_to_atif(history, [], "test-model")
        content = atif["steps"][0]["observation"]["results"][0]["content"]

        assert raw_secret not in content
        assert "password=abcdef******" in content
        assert "[truncated: trajectory exceeded" in content

    def test_final_metrics_aggregate(self):
        api_responses = [
            {"usage": {"prompt_tokens": 100, "completion_tokens": 50}},
            {"usage": {"prompt_tokens": 200, "completion_tokens": 80}},
        ]
        atif = trajectory.history_to_atif([], api_responses, "m")
        assert atif["final_metrics"]["total_prompt_tokens"] == 300
        assert atif["final_metrics"]["total_completion_tokens"] == 130

    def test_save_trajectory(self, tmp_path):
        atif = trajectory.history_to_atif([], [], "m")
        out = tmp_path / "out.json"
        trajectory.save_trajectory(atif, str(out))
        assert out.exists()

        data = json.loads(out.read_text())
        assert data["schema_version"] == "ATIF-v1.4"

    def test_save_trajectory_redacts_raw_input_before_writing(self, tmp_path):
        raw_secret = "token=abcdef123456"
        raw_trajectory = {
            "schema_version": "ATIF-v1.4",
            "steps": [{"message": raw_secret}],
            "final_metrics": {"total_steps": 1},
        }
        out = tmp_path / "out.json"

        trajectory.save_trajectory(raw_trajectory, str(out))

        data = json.loads(out.read_text())
        serialized = json.dumps(data)
        assert raw_secret not in serialized
        assert "token=abcdef******" in serialized


class TestPerTurnModel:
    """Each assistant turn records the model that produced it."""

    def test_single_model_trajectory_records_same_model_per_turn(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "first", "_model": "model-a"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "second", "_model": "model-a"},
        ]
        api_responses = [
            {"created": 1700000000, "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            {"created": 1700000001, "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ]
        atif = trajectory.history_to_atif(history, api_responses, "model-a")

        agent_steps = [s for s in atif["steps"] if s["source"] == "agent"]
        assert len(agent_steps) == 2
        assert all(s["model_name"] == "model-a" for s in agent_steps)
        assert atif["agent"]["model_name"] == "model-a"

    def test_multi_model_trajectory_records_each_turn_model(self):
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "first", "_model": "model-a"},
            {"role": "user", "content": "switch"},
            {"role": "assistant", "content": "second", "_model": "model-b"},
        ]
        api_responses = [
            {"created": 1700000000, "model": "model-a", "usage": {}},
            {"created": 1700000001, "model": "model-b", "usage": {}},
        ]
        # The trajectory-level fallback is intentionally a third model to
        # confirm the per-turn ``_model`` wins over the fallback.
        atif = trajectory.history_to_atif(history, api_responses, "fallback-model")

        agent_steps = [s for s in atif["steps"] if s["source"] == "agent"]
        assert [s["model_name"] for s in agent_steps] == ["model-a", "model-b"]
        # Top-level reflects the most recent turn's resolved model.
        assert atif["agent"]["model_name"] == "model-b"

    def test_backward_compat_missing_per_turn_model_falls_back(self):
        # Trajectories produced before this change have no ``_model`` field
        # on assistant messages. Ensure the fallback chain still works:
        # response ``model`` first, then the function ``model`` parameter.
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "with-resp-model"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "no-resp-model"},
        ]
        api_responses = [
            {"created": 1700000000, "model": "echoed-model", "usage": {}},
            {"created": 1700000001, "usage": {}},  # no model key
        ]
        atif = trajectory.history_to_atif(history, api_responses, "fallback-model")

        agent_steps = [s for s in atif["steps"] if s["source"] == "agent"]
        assert agent_steps[0]["model_name"] == "echoed-model"
        assert agent_steps[1]["model_name"] == "fallback-model"
        # Top-level reflects the latest turn's resolved model.
        assert atif["agent"]["model_name"] == "fallback-model"

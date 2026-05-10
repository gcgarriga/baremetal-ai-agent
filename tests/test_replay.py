"""Tests for replay.py — offline ATIF replay and diff inspection."""

import json

import pytest

from baremetal_agent import replay, safety


def _trajectory(
    *,
    final_message: str = "Done.",
    tool_name: str = "read_file",
    observation: str = "file contents",
) -> dict:
    return {
        "schema_version": "ATIF-v1.4",
        "session_id": "session-123",
        "agent": {"name": "baremetal-agent", "version": "0.1.0", "model_name": "test-model"},
        "steps": [
            {"step_id": 1, "timestamp": "2026-01-01T00:00:00Z", "source": "user", "message": "read it"},
            {
                "step_id": 2,
                "timestamp": "2026-01-01T00:00:01Z",
                "source": "agent",
                "model_name": "test-model",
                "metrics": {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 2},
                "tool_calls": [
                    {
                        "tool_call_id": "call_1",
                        "function_name": tool_name,
                        "arguments": {"path": "a.txt"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "call_1", "content": observation}]},
            },
            {
                "step_id": 3,
                "timestamp": "2026-01-01T00:00:02Z",
                "source": "agent",
                "model_name": "test-model",
                "metrics": {"prompt_tokens": 4, "completion_tokens": 6},
                "message": final_message,
            },
        ],
        "final_metrics": {
            "total_prompt_tokens": 14,
            "total_completion_tokens": 11,
            "total_cached_tokens": 2,
            "total_steps": 3,
        },
    }


class TestLoadTrajectory:
    def test_load_valid_trajectory(self, tmp_path):
        path = tmp_path / "trajectory.json"
        expected = _trajectory()
        path.write_text(json.dumps(expected), encoding="utf-8")

        assert replay.load_trajectory(str(path)) == expected

    def test_load_rejects_missing_schema(self, tmp_path):
        path = tmp_path / "trajectory.json"
        path.write_text(json.dumps({"steps": []}), encoding="utf-8")

        with pytest.raises(ValueError, match="schema_version"):
            replay.load_trajectory(str(path))

    def test_load_rejects_invalid_schema(self, tmp_path):
        path = tmp_path / "trajectory.json"
        path.write_text(json.dumps({"schema_version": "ATIF-v1.3", "steps": []}), encoding="utf-8")

        with pytest.raises(ValueError, match="ATIF-v1.4"):
            replay.load_trajectory(str(path))


class TestRender:
    def test_render_single_step_uses_atif_step_id(self, capsys):
        replay.render(_trajectory(), step_id=2)

        output = capsys.readouterr().out
        assert "Filtered to ATIF step_id=2" in output
        assert "Step 2 [agent]" in output
        assert 'read_file(path="a.txt") id=call_1' in output
        assert "file contents" in output
        assert "Step 1 [user]" not in output
        assert "Step 3 [agent]" not in output


class TestDiff:
    def test_diff_includes_required_comparison_sections(self):
        long_a = "alpha\n" + ("a" * 300)
        long_b = "beta\n" + ("b" * 300)

        result = replay.diff(
            _trajectory(final_message="Done.", tool_name="read_file", observation=long_a),
            _trajectory(final_message="Changed.", tool_name="search_code", observation=long_b),
        )

        assert "A: steps=3, tokens=25 (prompt=14, completion=11, cached=2)" in result
        assert "B: steps=3, tokens=25 (prompt=14, completion=11, cached=2)" in result
        assert 'A: present "Done."' in result
        assert 'B: present "Changed."' in result
        assert "Tool-call sequence by index:" in result
        assert "1. A=step 2 read_file | B=step 2 search_code" in result
        assert "read_file: A=1 B=0" in result
        assert "search_code: A=0 B=1" in result
        assert "Observation/result differences:" in result
        assert "sha256=" in result
        assert "a" * 120 not in result
        assert "b" * 120 not in result

    def test_diff_ignores_nondeterministic_observation_call_ids(self):
        left = _trajectory(observation="same content")
        right = _trajectory(observation="same content")
        left["steps"][1]["tool_calls"][0]["tool_call_id"] = "call_left"
        left["steps"][1]["observation"]["results"][0]["source_call_id"] = "call_left"
        right["steps"][1]["tool_calls"][0]["tool_call_id"] = "call_right"
        right["steps"][1]["observation"]["results"][0]["source_call_id"] = "call_right"

        result = replay.diff(left, right)

        assert "Observation/result differences:\n  none" in result

    def test_diff_detects_observation_changes_after_display_truncation_boundary(self):
        common_prefix = "x" * safety.DEFAULT_MAX_CHARS
        left = _trajectory(observation=common_prefix + "left-only")
        right = _trajectory(observation=common_prefix + "right-only")

        result = replay.diff(left, right)

        assert "Observation/result differences:\n  1." in result
        assert "sha256=" in result
        assert "left-only" not in result
        assert "right-only" not in result

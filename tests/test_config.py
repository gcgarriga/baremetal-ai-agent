"""Tests for config.py — .env loading and configuration."""

import os
import subprocess
import sys
from pathlib import Path

from baremetal_agent import config
from baremetal_agent.config import load_config


def _config_subprocess_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "AGENT_DOTENV": str(tmp_path / "missing.env"),
        "AGENT_MAX_ITERATIONS": "10",
        "AGENT_MODEL": "openai/gpt-4.1",
        "AGENT_VERBOSE": "false",
        "AGENT_WORKING_DIR": str(tmp_path),
    }
    env.update(overrides)
    return env


class TestLoadDotenv:
    def test_loads_from_cwd(self, tmp_path, monkeypatch):
        """config._load_dotenv() should read .env from CWD."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_DOTENV_VAR=hello_from_cwd\n")
        monkeypatch.chdir(tmp_path)
        # Remove if already set, so setdefault in _load_dotenv takes effect
        monkeypatch.delenv("TEST_DOTENV_VAR", raising=False)
        monkeypatch.delenv("AGENT_DOTENV", raising=False)
        config._load_dotenv()
        assert os.environ["TEST_DOTENV_VAR"] == "hello_from_cwd"

    def test_agent_dotenv_override(self, tmp_path, monkeypatch):
        """AGENT_DOTENV env var should override the default .env path."""
        custom_env = tmp_path / "custom.env"
        custom_env.write_text("TEST_CUSTOM_VAR=from_override\n")
        monkeypatch.setenv("AGENT_DOTENV", str(custom_env))
        monkeypatch.delenv("TEST_CUSTOM_VAR", raising=False)
        config._load_dotenv()
        assert os.environ["TEST_CUSTOM_VAR"] == "from_override"

    def test_missing_env_file_no_error(self, tmp_path, monkeypatch):
        """No .env file at all should not raise."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AGENT_DOTENV", raising=False)
        # Should not raise
        config._load_dotenv()

    def test_quoted_values_stripped(self, tmp_path, monkeypatch):
        """Quotes around values in .env should be stripped."""
        env_file = tmp_path / ".env"
        env_file.write_text('TEST_QUOTED_VAR="quoted_value"\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TEST_QUOTED_VAR", raising=False)
        monkeypatch.delenv("AGENT_DOTENV", raising=False)
        config._load_dotenv()
        assert os.environ["TEST_QUOTED_VAR"] == "quoted_value"

    def test_comments_and_blank_lines_ignored(self, tmp_path, monkeypatch):
        """Comments and blank lines in .env should be skipped."""
        env_file = tmp_path / ".env"
        env_file.write_text("# this is a comment\n\nTEST_COMMENT_VAR=works\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TEST_COMMENT_VAR", raising=False)
        monkeypatch.delenv("AGENT_DOTENV", raising=False)
        config._load_dotenv()
        assert os.environ["TEST_COMMENT_VAR"] == "works"

    def test_existing_env_not_overwritten(self, tmp_path, monkeypatch):
        """setdefault should not overwrite already-set env vars."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_EXISTING_VAR=from_file\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TEST_EXISTING_VAR", "already_set")
        monkeypatch.delenv("AGENT_DOTENV", raising=False)
        config._load_dotenv()
        assert os.environ["TEST_EXISTING_VAR"] == "already_set"


class TestLazyTokenValidation:
    def test_import_without_github_token_succeeds(self, tmp_path):
        """GITHUB_TOKEN should not be required until a model call is made."""
        cmd = "from baremetal_agent import config; print('GITHUB_TOKEN' not in __import__('os').environ)"
        result = subprocess.run(
            [sys.executable, "-c", cmd],
            capture_output=True,
            text=True,
            env=_config_subprocess_env(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"


class TestApiUrl:
    def test_default_api_url_is_github_models_endpoint(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", "from baremetal_agent.config import load_config; print(load_config().api_url)"],
            capture_output=True,
            text=True,
            env=_config_subprocess_env(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "https://models.github.ai/inference/chat/completions"

    def test_agent_api_url_overrides_default(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", "from baremetal_agent.config import load_config; print(load_config().api_url)"],
            capture_output=True,
            text=True,
            env=_config_subprocess_env(tmp_path, AGENT_API_URL="https://example.test/custom"),
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "https://example.test/custom"


class TestStreamFlag:
    def test_agent_stream_defaults_false(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", "from baremetal_agent.config import load_config; print(load_config().stream)"],
            capture_output=True,
            text=True,
            env=_config_subprocess_env(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"

    def test_agent_stream_truthy_values_enable_streaming(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", "from baremetal_agent.config import load_config; print(load_config().stream)"],
            capture_output=True,
            text=True,
            env=_config_subprocess_env(tmp_path, AGENT_STREAM="1"),
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"


class TestVerboseSplit:
    """`AGENT_VERBOSE` is the legacy one-knob flag; granular vars override it."""

    def _load(self, tmp_path: Path, **overrides: str):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from baremetal_agent.config import load_config; "
                "c = load_config(); "
                "print(c.render_verbose, c.log_payloads)",
            ],
            capture_output=True,
            text=True,
            env=_config_subprocess_env(tmp_path, **overrides),
        )
        assert result.returncode == 0, result.stderr
        render, log = result.stdout.strip().split()
        return render == "True", log == "True"

    def test_default_both_false(self, tmp_path):
        assert self._load(tmp_path) == (False, False)

    def test_agent_verbose_sets_both(self, tmp_path):
        assert self._load(tmp_path, AGENT_VERBOSE="1") == (True, True)

    def test_render_verbose_alone_does_not_log_payloads(self, tmp_path):
        assert self._load(tmp_path, AGENT_RENDER_VERBOSE="1") == (True, False)

    def test_log_payloads_alone_does_not_render_verbose(self, tmp_path):
        assert self._load(tmp_path, AGENT_LOG_PAYLOADS="1") == (False, True)

    def test_render_verbose_overrides_legacy_when_off(self, tmp_path):
        # AGENT_VERBOSE=1 would set both, but explicit AGENT_RENDER_VERBOSE=0
        # disables only the renderer flag.
        assert self._load(tmp_path, AGENT_VERBOSE="1", AGENT_RENDER_VERBOSE="0") == (False, True)

    def test_log_payloads_overrides_legacy_when_off(self, tmp_path):
        assert self._load(tmp_path, AGENT_VERBOSE="1", AGENT_LOG_PAYLOADS="0") == (True, False)

    def test_granular_can_extend_legacy(self, tmp_path):
        # Both granular vars set, AGENT_VERBOSE off.
        assert self._load(tmp_path, AGENT_VERBOSE="0", AGENT_RENDER_VERBOSE="1", AGENT_LOG_PAYLOADS="1") == (True, True)


class TestWorkingDir:
    def test_working_dir_is_path(self):
        assert isinstance(load_config().working_dir, Path)

    def test_working_dir_is_resolved(self):
        assert load_config().working_dir.is_absolute()


class TestMaxIterationsValidation:
    def test_invalid_max_iterations_exits_with_message(self, tmp_path):
        """Non-integer AGENT_MAX_ITERATIONS should exit with a friendly error."""
        result = subprocess.run(
            [sys.executable, "-c", "from baremetal_agent.config import load_config; load_config()"],
            capture_output=True,
            text=True,
            env=_config_subprocess_env(tmp_path, GITHUB_TOKEN="test-token", AGENT_MAX_ITERATIONS="notanumber"),
        )
        assert result.returncode != 0
        assert "AGENT_MAX_ITERATIONS must be an integer" in result.stderr

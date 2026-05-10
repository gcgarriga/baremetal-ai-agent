"""Pytest configuration — runs before any test imports."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

# Set known-good env vars before config.py is imported.
os.environ["GITHUB_TOKEN"] = "test-token-for-testing"
os.environ["AGENT_MAX_ITERATIONS"] = "10"
os.environ["AGENT_WORKING_DIR"] = "/tmp/baremetal-test"


_DEFAULT_CFG_KWARGS = dict(
    model="test-model",
    api_url="https://example.test/chat/completions",
    max_iterations=10,
    working_dir=Path("/tmp/baremetal-test"),
    render_verbose=False,
    log_payloads=False,
    stream=False,
    system_prompt="test system prompt",
)


@pytest.fixture
def make_cfg():
    """Build an AgentConfig with sensible test defaults; overrides via kwargs."""
    from baremetal_agent.config import AgentConfig

    base = AgentConfig(**_DEFAULT_CFG_KWARGS)

    def _make(**overrides):
        return replace(base, **overrides) if overrides else base

    return _make


@pytest.fixture
def null_renderer():
    """A NullRenderer instance for tests that don't care about output."""
    from baremetal_agent.visualizer import NullRenderer

    return NullRenderer()

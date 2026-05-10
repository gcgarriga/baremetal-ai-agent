"""Tests for shared output redaction and truncation helpers."""

from baremetal_agent import safety


def test_redact_secrets_masks_known_secret_patterns():
    text = "ghp_abcdef123456 token=abcdef123456 password=abcdef123456 github_pat_abcdef123456 password=uvwxyz"

    redacted = safety.redact_secrets(text)

    assert "abcdef123456" not in redacted
    assert "password=uvwxyz" not in redacted
    assert "ghp_abcdef******" in redacted
    assert "token=abcdef******" in redacted
    assert "password=abcdef******" in redacted
    assert "github_pat_abcdef******" in redacted
    assert "password=******" in redacted


def test_truncate_text_adds_visible_marker_and_omitted_count():
    result = safety.truncate_text("abcdefghij", max_chars=6, label="unit")

    assert result == "abcdef\n\n[truncated: unit exceeded 6 chars; omitted 4 chars]"


def test_truncate_text_leaves_exact_limit_unchanged():
    assert safety.truncate_text("abcdef", max_chars=6, label="unit") == "abcdef"


def test_sanitize_text_redacts_before_truncating():
    raw_secret = "token=abcdef123456"

    result = safety.sanitize_text(f"{raw_secret} trailing text", max_chars=20, label="unit")

    assert raw_secret not in result
    assert "token=abcdef******" in result
    assert "[truncated: unit exceeded 20 chars;" in result


def test_sanitize_text_does_not_truncate_existing_marker_again():
    once = safety.sanitize_text("x" * 30, max_chars=20, label="unit")

    twice = safety.sanitize_text(once, max_chars=20, label="unit")

    assert twice == once


def test_sanitize_json_value_recurses_into_nested_string_values():
    value = {
        "message": "password=abcdef123456",
        "items": ["safe", "sk-abcdef123456"],
        "number": 7,
        "enabled": True,
        "nothing": None,
    }

    result = safety.sanitize_json_value(value, max_string_chars=100, label="json")

    assert result == {
        "message": "password=abcdef******",
        "items": ["safe", "sk-abcdef******"],
        "number": 7,
        "enabled": True,
        "nothing": None,
    }

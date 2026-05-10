"""Shared helpers for redacting and bounding persisted or replayed text."""

import re
from collections.abc import Mapping, Sequence

DEFAULT_MAX_CHARS = 10_000

_SECRET_RE = re.compile(
    r"(ghp_|gho_|ghu_|github_pat_|sk-|key-|token[=: ]+|password[=: ]+)"
    r"([A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)
_TRUNCATION_MARKER_RE = re.compile(
    r"\n\n\[truncated: .+ exceeded (?P<max_chars>\d+) chars; omitted (?P<omitted>\d+) chars\]\Z"
)


def redact_secrets(text: str) -> str:
    """Replace likely secrets in text, keeping only a short non-secret prefix."""

    def _mask(m: re.Match) -> str:
        prefix = m.group(1)
        secret = m.group(2)
        visible = secret[:6] if len(secret) > 6 else ""
        hidden_len = len(secret) - len(visible)
        return f"{prefix}{visible}{'*' * hidden_len}"

    return _SECRET_RE.sub(_mask, text)


def truncate_text(text: str, *, max_chars: int = DEFAULT_MAX_CHARS, label: str = "text") -> str:
    """Truncate long text and append a visible marker."""
    marker_match = _TRUNCATION_MARKER_RE.search(text)
    if marker_match and len(text[: marker_match.start()]) == int(marker_match.group("max_chars")):
        return text
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    marker = f"[truncated: {label} exceeded {max_chars} chars; omitted {omitted} chars]"
    return f"{text[:max_chars]}\n\n{marker}"


def sanitize_text(text: str, *, max_chars: int = DEFAULT_MAX_CHARS, label: str = "text") -> str:
    """Redact likely secrets, then truncate the redacted text."""
    return truncate_text(redact_secrets(text), max_chars=max_chars, label=label)


def sanitize_json_value(
    value: object,
    *,
    max_string_chars: int = DEFAULT_MAX_CHARS,
    label: str = "json",
) -> object:
    """Recursively sanitize string values in JSON-like data."""
    if isinstance(value, str):
        return sanitize_text(value, max_chars=max_string_chars, label=label)
    if isinstance(value, Mapping):
        return {
            key: sanitize_json_value(item, max_string_chars=max_string_chars, label=label)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_json_value(item, max_string_chars=max_string_chars, label=label) for item in value]
    return value

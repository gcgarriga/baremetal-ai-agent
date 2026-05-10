"""Tests for client.py — HTTP behavior, retries, and redaction."""

from dataclasses import replace

import httpx
import pytest

from baremetal_agent import client, safety


def _completion_response() -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {}}


@pytest.fixture
def cfg(make_cfg):
    return make_cfg()


@pytest.fixture
def install_transport(monkeypatch):
    clients: list[httpx.Client] = []

    def _install(handler):
        mocked_client = httpx.Client(transport=httpx.MockTransport(handler), timeout=120.0)
        clients.append(mocked_client)
        monkeypatch.setattr(client, "_client", mocked_client)

    yield _install

    for mocked_client in clients:
        mocked_client.close()


def test_chat_completion_returns_200_json(install_transport, cfg):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_completion_response())

    install_transport(handler)

    response = client.chat_completion([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert response == _completion_response()
    assert len(requests) == 1
    assert str(requests[0].url) == cfg.api_url
    assert requests[0].headers["authorization"] == "Bearer test-token-for-testing"


def test_chat_completion_rejects_invalid_200_json(install_transport, cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    install_transport(handler)

    with pytest.raises(RuntimeError, match="Invalid JSON in 200 response"):
        client.chat_completion([{"role": "user", "content": "hello"}], [], cfg=cfg)


def test_chat_completion_requires_token_before_request(monkeypatch, install_transport, cfg):
    requests: list[httpx.Request] = []
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_completion_response())

    install_transport(handler)

    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        client.chat_completion([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert requests == []


def test_chat_completion_raises_clear_401_error(install_transport, cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad token")

    install_transport(handler)

    with pytest.raises(RuntimeError, match="Authentication failed \\(401\\).*GITHUB_TOKEN"):
        client.chat_completion([{"role": "user", "content": "hello"}], [], cfg=cfg)


def test_chat_completion_retries_429_then_succeeds(monkeypatch, install_transport, cfg):
    attempts = 0
    sleeps: list[int] = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(200, json=_completion_response())

    install_transport(handler)

    response = client.chat_completion([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert response == _completion_response()
    assert attempts == 2
    assert sleeps == [0]


def test_chat_completion_retries_5xx_then_succeeds(monkeypatch, install_transport, cfg):
    statuses = [500, 502, 200]
    sleeps: list[int] = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(status, json=_completion_response())
        return httpx.Response(status, text="server error")

    install_transport(handler)

    response = client.chat_completion([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert response == _completion_response()
    assert statuses == []
    assert sleeps == [1, 2]


def test_chat_completion_retries_request_errors_then_succeeds(monkeypatch, install_transport, cfg):
    attempts = 0
    sleeps: list[int] = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("network down", request=request)
        return httpx.Response(200, json=_completion_response())

    install_transport(handler)

    response = client.chat_completion([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert response == _completion_response()
    assert attempts == 3
    assert sleeps == [1, 2]


def test_verbose_logging_uses_shared_redactor(monkeypatch, install_transport, capsys, cfg):
    cfg = replace(cfg, log_payloads=True)
    monkeypatch.setattr(safety, "redact_secrets", lambda text: text.replace("hello", "shared-redactor-used"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion_response())

    install_transport(handler)

    client.chat_completion([{"role": "user", "content": "hello"}], [], cfg=cfg)

    output = capsys.readouterr().out
    assert "shared-redactor-used" in output
    assert "hello" not in output


def test_redact_masks_known_secret_patterns():
    text = "ghp_abcdef123456 token=abcdef123456 password=abcdef123456 github_pat_abcdef123456 password=uvwxyz"

    redacted = safety.redact_secrets(text)

    assert "abcdef123456" not in redacted
    assert "password=uvwxyz" not in redacted
    assert "ghp_abcdef******" in redacted
    assert "token=abcdef******" in redacted
    assert "password=abcdef******" in redacted
    assert "github_pat_abcdef******" in redacted
    assert "password=******" in redacted


def test_chat_completion_stream_posts_streaming_body_and_returns_openai_shape(install_transport, cfg):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"stream"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":" ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    install_transport(handler)

    response = client.chat_completion_stream([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert response["choices"][0]["message"] == {"role": "assistant", "content": "stream ok"}
    assert response["choices"][0]["finish_reason"] == "stop"
    assert len(requests) == 1
    assert requests[0].read() == (
        b'{"model":"test-model","messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto",'
        b'"stream":true,"stream_options":{"include_usage":true}}'
    )


def test_chat_completion_stream_preserves_usage_chunks(install_transport, cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    install_transport(handler)

    response = client.chat_completion_stream([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert response["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}


def test_chat_completion_stream_uses_shared_redactor_for_verbose_logs(monkeypatch, install_transport, capsys, cfg):
    cfg = replace(cfg, log_payloads=True)
    monkeypatch.setattr(safety, "redact_secrets", lambda text: text.replace("token=abcdef123456", "token=redacted"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"token=abcdef123456"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    install_transport(handler)

    response = client.chat_completion_stream([{"role": "user", "content": "hello"}], [], cfg=cfg)

    output = capsys.readouterr().out
    assert response["choices"][0]["message"]["content"] == "token=abcdef123456"
    assert "token=redacted" in output
    assert "token=abcdef123456" not in output


def test_chat_completion_stream_raises_clear_401_error(install_transport, cfg):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad token")

    install_transport(handler)

    with pytest.raises(RuntimeError, match="Authentication failed \\(401\\).*GITHUB_TOKEN"):
        client.chat_completion_stream([{"role": "user", "content": "hello"}], [], cfg=cfg)


def test_chat_completion_stream_retries_429_then_succeeds(monkeypatch, install_transport, cfg):
    attempts = 0
    sleeps: list[int] = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow down")
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')

    install_transport(handler)

    response = client.chat_completion_stream([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert attempts == 2
    assert sleeps == [0]


def test_chat_completion_stream_retries_5xx_then_succeeds(monkeypatch, install_transport, cfg):
    statuses = [500, 502, 200]
    sleeps: list[int] = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(status, content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(status, text="server error")

    install_transport(handler)

    response = client.chat_completion_stream([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert statuses == []
    assert sleeps == [1, 2]


def test_chat_completion_stream_retries_request_errors_then_succeeds(monkeypatch, install_transport, cfg):
    attempts = 0
    sleeps: list[int] = []
    monkeypatch.setattr(client.time, "sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("network down", request=request)
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')

    install_transport(handler)

    response = client.chat_completion_stream([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert response["choices"][0]["message"]["content"] == "ok"
    assert attempts == 3
    assert sleeps == [1, 2]


def test_chat_completion_stream_does_not_retry_malformed_stream_after_200(install_transport, cfg):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, content=b"data: {not-json}\n\n")

    install_transport(handler)

    with pytest.raises(RuntimeError, match="Streaming response failed: Invalid JSON in streaming response"):
        client.chat_completion_stream([{"role": "user", "content": "hello"}], [], cfg=cfg)

    assert attempts == 1

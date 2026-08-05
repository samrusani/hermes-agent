"""Regressions for callable API keys on custom auxiliary endpoints."""

from unittest.mock import MagicMock, patch

from agent.auxiliary_client import resolve_provider_client


def test_callable_explicit_api_key_is_resolved_for_async_vision(monkeypatch):
    """Callable keys are resolved before building an async vision client."""
    import agent.auxiliary_client as ac

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict = {}
    calls = 0
    sync_client = MagicMock()
    async_client = MagicMock()

    def key_provider():
        nonlocal calls
        calls += 1
        return "  short-lived-token  "

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return sync_client

    with patch.object(ac, "_create_openai_client", side_effect=_capture_create), \
         patch.object(
             ac,
             "_to_async_client",
             return_value=(async_client, "vision-model"),
         ) as mock_to_async:
        client, model = resolve_provider_client(
            "custom",
            model="vision-model",
            async_mode=True,
            explicit_base_url="https://gw.example.com/v1",
            explicit_api_key=key_provider,
            is_vision=True,
        )

    assert (client, model) == (async_client, "vision-model")
    assert calls == 1
    assert captured.get("api_key") == "short-lived-token"
    assert captured.get("api_key") is not key_provider
    assert captured.get("api_key") != str(key_provider)
    mock_to_async.assert_called_once_with(sync_client, "vision-model", is_vision=True)


def test_callable_main_runtime_api_key_is_resolved():
    """The adjacent main-runtime custom path resolves callable keys too."""
    import agent.auxiliary_client as ac

    captured: dict = {}
    calls = 0

    def key_provider():
        nonlocal calls
        calls += 1
        return "  runtime-token  "

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch.object(ac, "_create_openai_client", side_effect=_capture_create):
        resolve_provider_client(
            "custom",
            model="vision-model",
            main_runtime={
                "provider": "custom",
                "model": "vision-model",
                "base_url": "https://runtime.example.com/v1",
                "api_key": key_provider,
            },
            is_vision=True,
        )

    assert calls == 1
    assert captured.get("api_key") == "runtime-token"
    assert captured.get("api_key") is not key_provider
    assert captured.get("api_key") != str(key_provider)

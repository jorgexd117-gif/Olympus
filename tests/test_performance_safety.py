"""Performance and thermal safety unit tests."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from src import nodes
from src.thermal import ThermalRegulator


def test_deepseek_short_circuits_when_local_backend_unavailable(monkeypatch):
    state = {
        "messages": [
            HumanMessage(content="consulta"),
            AIMessage(content="[OpenAI pensamiento resumido]\nNo se pudo usar el backend gratis local."),
            AIMessage(content="[Anthropic contraste]\nSe omite llamada de contraste."),
        ]
    }

    async def _should_not_run(**_: str) -> str:
        raise AssertionError("action stage should not call backend when failure already known")

    monkeypatch.setenv("LOCAL_RESCUE_ENABLED", "false")
    monkeypatch.setattr(nodes, "_call_free_oss_chat", _should_not_run)
    result = asyncio.run(nodes.deepseek_action_node(state))
    content = str(result["messages"][0].content)

    assert "Se omitieron llamadas adicionales" in content
    assert "No se pudo usar el backend gratis local" in content


def test_deepseek_uses_rescue_mode_when_enabled(monkeypatch):
    state = {
        "messages": [
            HumanMessage(content="necesito una accion concreta"),
            AIMessage(content="[OpenAI pensamiento resumido]\nNo se pudo usar el backend gratis local."),
            AIMessage(content="[Anthropic contraste]\nSe omite llamada de contraste."),
        ]
    }

    async def _rescue_stub(**_: str) -> str:
        return "Plan de rescate local: ejecuta comando X y valida salida."

    monkeypatch.setenv("LOCAL_RESCUE_ENABLED", "true")
    monkeypatch.setattr(nodes, "_call_free_oss_chat", _rescue_stub)
    result = asyncio.run(nodes.deepseek_action_node(state))
    content = str(result["messages"][0].content)

    assert "Modo rescate local activado" in content
    assert "Plan de rescate local" in content


def test_thermal_timeout_scales_down_under_critical(monkeypatch):
    monkeypatch.setenv("THERMAL_TIMEOUT_SCALE_WARNING", "1.0")
    monkeypatch.setenv("THERMAL_TIMEOUT_SCALE_CRITICAL", "0.5")
    monkeypatch.setenv("THERMAL_TIMEOUT_MIN_SECONDS", "10")
    regulator = ThermalRegulator()

    assert regulator.request_timeout(base_timeout_s=60, level="warning") == 60
    assert regulator.request_timeout(base_timeout_s=60, level="critical") == 30
    assert regulator.request_timeout(base_timeout_s=5, level="critical") == 10


def test_friendly_provider_error_masks_raw_404_not_found():
    raw_error = (
        "Client error '404 Not Found' for url 'http://localhost:11434/api/generate' "
        "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404"
    )
    text = nodes._friendly_provider_error(raw_error)

    assert "Not Found" not in text
    assert "/api/generate" in text

"""Test suite for the LangGraph agent."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from src import nodes
from src.graph import AgentState, compile_graph
from src.multi_agent import AgentRunResult, MultiAgentCoordinator
from src.persistence import AgentDatabase


async def _stub_free_chat(*, model: str, user_prompt: str, system_prompt: str, provider: str | None = None) -> str:
    del model, system_prompt, provider
    if "Responde SOLO con JSON" in user_prompt:
        return '{"final_response":"ok","actions":[]}'
    if "Contrasta el análisis" in user_prompt:
        return "- validado\n- sin riesgos criticos"
    return "- objetivo claro\n- siguiente paso inmediato"


def test_agent_graph_pipeline_is_stable(monkeypatch):
    """Run full graph quickly with local chat calls stubbed."""
    monkeypatch.setenv("FORCE_FREE_LLM", "true")
    monkeypatch.setattr(nodes, "_call_free_oss_chat", _stub_free_chat)

    agent = compile_graph()
    initial_state: AgentState = {"messages": [HumanMessage(content="Test message")]}

    result = asyncio.run(agent.ainvoke(initial_state))

    assert len(result["messages"]) >= 4
    assert isinstance(result["messages"][-1], AIMessage)
    assert "Respuesta final" in str(result["messages"][-1].content)


def test_review_short_circuits_when_backend_is_down(monkeypatch):
    """Avoid extra heavy calls when previous stage already reported local backend failure."""
    initial_state: AgentState = {
        "messages": [
            HumanMessage(content="consulta"),
            AIMessage(content="[OpenAI pensamiento resumido]\nNo se pudo usar el backend gratis local."),
        ]
    }

    async def _should_not_run(**_: str) -> str:
        raise AssertionError("review stage should not call local backend when already unavailable")

    monkeypatch.setattr(nodes, "_call_free_oss_chat", _should_not_run)
    result = asyncio.run(nodes.anthropic_review_node(initial_state))

    assert "Se omite llamada de contraste" in str(result["messages"][0].content)


def test_prompt_translation_machine_ir_for_command(tmp_path):
    """Natural-language command requests are translated into machine-readable IR."""
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    coordinator = MultiAgentCoordinator(database)
    try:
        ir = coordinator._translate_prompt_to_machine_ir(
            text="comando: pytest -q",
            project_id=3,
            command_text="pytest -q",
            game_folder=None,
            control_kind=None,
        )
    finally:
        database.close()

    assert ir["schema"] == "prompt-ir/v1"
    assert ir["intent"] == "command_execution"
    assert ir["route"]["target"] == "command_runner"
    assert ir["args"]["command_text"] == "pytest -q"
    assert any(item.get("op") == "EXECUTE_COMMAND" for item in ir["ops"])


def test_free_provider_huggingface_branch(monkeypatch):
    async def _stub_hf_chat(*, model: str, user_prompt: str, system_prompt: str) -> str:
        del model, user_prompt, system_prompt
        return "hf-ok"

    monkeypatch.setattr(nodes, "_call_huggingface_chat", _stub_hf_chat)
    result = asyncio.run(
        nodes._call_free_oss_chat(
            model="Qwen/Qwen2.5-7B-Instruct",
            user_prompt="hola",
            system_prompt="responde breve",
            provider="huggingface",
        )
    )

    assert result == "hf-ok"


def test_time_request_forces_full_flow_before_reply(tmp_path, monkeypatch):
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    coordinator = MultiAgentCoordinator(database)
    monkeypatch.setenv("FREE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LOCAL_SINGLE_PASS_ENABLED", "true")

    marks = {"single_pass_called": False, "run_agent_called": False}

    async def _stub_single_pass_reply(*, project_id: int | None, user_prompt: str) -> str | None:
        del project_id, user_prompt
        marks["single_pass_called"] = True
        return "single-pass"

    async def _stub_run_agent(*, project_id: int | None, user_prompt: str) -> AgentRunResult:
        del project_id, user_prompt
        marks["run_agent_called"] = True
        return AgentRunResult(
            messages=[AIMessage(content="[DeepSeek ejecución]\nRespuesta final:\nfull-flow")],
            sections={"action": "full-flow"},
            final_output="full-flow",
        )

    coordinator._single_pass_reply = _stub_single_pass_reply  # type: ignore[assignment]
    coordinator.run_agent = _stub_run_agent  # type: ignore[assignment]
    try:
        result = asyncio.run(coordinator.assistant_turn(project_id=None, user_prompt="que hora es ahora"))
    finally:
        database.close()

    assert marks["run_agent_called"] is True
    assert marks["single_pass_called"] is False
    assert result.source == "agent"
    assert result.reply == "full-flow"
    assert result.machine_translation["execution_policy"]["forced_flow"] == "multi_stage"


def test_assistant_detects_missing_project_context_for_project_task(tmp_path):
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    coordinator = MultiAgentCoordinator(database)
    try:
        result = asyncio.run(
            coordinator.assistant_turn(
                project_id=None,
                user_prompt="ejecuta pruebas del proyecto y corrige fallos",
            )
        )
    finally:
        database.close()

    assert result.source == "context-request"
    assert "Falta contexto" in result.reply
    context = result.machine_translation["context_requirements"]
    assert context["is_blocking"] is True
    assert "project_context" in context["missing"]

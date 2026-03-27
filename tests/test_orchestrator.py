"""Tests for process orchestrator workflows."""

import asyncio

from src.multi_agent import AssistantTurnResult, MultiAgentCoordinator
from src.persistence import AgentDatabase
from src.process_orchestrator import ProcessOrchestrator


def test_orchestrator_lists_workflows(tmp_path):
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    coordinator = MultiAgentCoordinator(database)
    orchestrator = ProcessOrchestrator(database, coordinator)
    try:
        workflows = orchestrator.list_workflows()
    finally:
        database.close()

    ids = {item.workflow_id for item in workflows}
    assert "diagnostic" in ids
    assert "assistant_quick" in ids
    assert "agent_full" in ids


def test_orchestrator_runs_assistant_quick_with_stub(tmp_path):
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    coordinator = MultiAgentCoordinator(database)
    orchestrator = ProcessOrchestrator(database, coordinator)

    async def _stub_assistant_turn(*, project_id: int | None, user_prompt: str) -> AssistantTurnResult:
        del project_id, user_prompt
        return AssistantTurnResult(
            reply="ok",
            source="stub",
            project_id=None,
            sections={"action": "ok"},
            machine_translation={},
        )

    coordinator.assistant_turn = _stub_assistant_turn  # type: ignore[assignment]
    try:
        result = asyncio.run(
            orchestrator.run_workflow(
                workflow_id="assistant_quick",
                project_id=None,
                user_prompt="prueba",
            )
        )
    finally:
        database.close()

    assert result.status == "completed"
    assert any(step.step == "assistant" and step.status == "ok" for step in result.steps)
    assert result.output.get("assistant", {}).get("reply") == "ok"

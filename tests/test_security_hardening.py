"""Security and hardening tests for coordinator/database."""

from __future__ import annotations

import pytest

from src.multi_agent import MultiAgentCoordinator
from src.persistence import AgentDatabase


def test_command_blocks_dangerous_pattern(tmp_path):
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    coordinator = MultiAgentCoordinator(database)
    try:
        code, output = coordinator.execute_project_command(project_id=None, command_text="rm -rf /")
    finally:
        database.close()

    assert code == 126
    assert "bloqueado" in output.lower()


def test_command_blocks_shell_operators_by_default(tmp_path):
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    coordinator = MultiAgentCoordinator(database)
    try:
        code, output = coordinator.execute_project_command(
            project_id=None,
            command_text="echo ok | cat",
        )
    finally:
        database.close()

    assert code == 126
    assert "operadores de shell" in output.lower()


def test_command_simple_execution_works_without_shell(tmp_path):
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    coordinator = MultiAgentCoordinator(database)
    try:
        code, output = coordinator.execute_project_command(project_id=None, command_text="echo hola")
    finally:
        database.close()

    assert code == 0
    assert "hola" in output.lower()


def test_command_execution_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMAND_EXECUTION_ENABLED", "false")
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    coordinator = MultiAgentCoordinator(database)
    try:
        code, output = coordinator.execute_project_command(project_id=None, command_text="echo hola")
    finally:
        database.close()

    assert code == 126
    assert "deshabilitada" in output.lower()


def test_project_root_collision_raises_value_error(tmp_path):
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    try:
        database.upsert_project(name="a", root_path=str(tmp_path / "same"), description="")
        with pytest.raises(ValueError):
            database.upsert_project(name="b", root_path=str(tmp_path / "same"), description="")
    finally:
        database.close()


def test_mongodb_backend_falls_back_to_sqlite_when_disabled_or_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("MONGODB_FALLBACK_TO_SQLITE", "true")
    database = AgentDatabase(db_path=tmp_path / "agent.db", backend="mongodb")
    try:
        assert database.backend == "sqlite"
        assert database.mongo_init_error
    finally:
        database.close()


def test_mongodb_backend_raises_when_fallback_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MONGODB_FALLBACK_TO_SQLITE", "false")
    with pytest.raises(Exception):
        AgentDatabase(db_path=tmp_path / "agent.db", backend="mongodb")

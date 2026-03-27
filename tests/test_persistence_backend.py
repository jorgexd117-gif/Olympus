"""Tests for persistence backend selection."""

from __future__ import annotations

import pytest

from src.persistence import AgentDatabase


def test_sqlite_backend_is_default(tmp_path):
    database = AgentDatabase(db_path=tmp_path / "agent.db")
    try:
        assert database.backend == "sqlite"
    finally:
        database.close()


def test_invalid_backend_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        AgentDatabase(db_path=tmp_path / "agent.db", backend="invalid")

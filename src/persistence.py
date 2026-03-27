"""Persistence layer for projects, profiles, memory, and runs.

Supports two backends:
- sqlite (default)
- mongodb (optional via AGENT_DB_BACKEND=mongodb)
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_string(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=True)


@dataclass
class ProjectRecord:
    """Project row model."""

    id: int
    name: str
    root_path: str
    description: str
    created_at: str


@dataclass
class AgentProfileRecord:
    """Agent profile row model."""

    id: int
    agent_key: str
    display_name: str
    role: str
    system_prompt: str
    model_name: str
    is_enabled: bool
    created_at: str
    updated_at: str


@dataclass
class MemoryRecord:
    """Memory row model."""

    id: int
    project_id: int | None
    memory_type: str
    content: str
    metadata_json: str
    relevance: float
    created_at: str


@dataclass
class ConversationRecord:
    """Conversation row model."""

    id: int
    project_id: int | None
    user_input: str
    assistant_output: str
    trace_json: str
    created_at: str


class AgentDatabase:
    """Low-level storage for the external human-facing control center."""

    def __init__(
        self,
        db_path: str | Path = "data/agent_memory.db",
        backend: str | None = None,
    ) -> None:
        self.backend = (backend or os.getenv("AGENT_DB_BACKEND", "sqlite")).strip().lower()
        self.db_path: str | Path = Path(db_path)
        self.connection: sqlite3.Connection | None = None
        self.mongo_client: Any = None
        self.mongo_db: Any = None
        self._mongo_return_document: Any = None
        self.mongo_init_error: str | None = None

        if self.backend == "sqlite":
            self._init_sqlite(db_path=db_path)
        elif self.backend == "mongodb":
            try:
                self._init_mongodb()
            except Exception as exc:
                if self._env_bool("MONGODB_FALLBACK_TO_SQLITE", True):
                    self.mongo_init_error = str(exc)
                    self.backend = "sqlite"
                    self._init_sqlite(db_path=db_path)
                else:
                    raise
        else:
            raise ValueError("AGENT_DB_BACKEND invalido. Usa 'sqlite' o 'mongodb'.")

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name, "true" if default else "false").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _init_sqlite(self, *, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_sqlite_schema()
        self._seed_profiles()

    def _init_mongodb(self) -> None:
        try:
            from pymongo import ASCENDING, DESCENDING, MongoClient
            from pymongo import ReturnDocument as MongoReturnDocument
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MongoDB backend requiere pymongo. Instala con: pip install pymongo"
            ) from exc

        uri = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017").strip()
        db_name = os.getenv("MONGODB_DATABASE", "langgraph_agent").strip() or "langgraph_agent"
        timeout_ms = int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "3000"))

        self.mongo_client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        # Fail fast if connection is unavailable.
        self.mongo_client.admin.command("ping")
        self.mongo_db = self.mongo_client[db_name]
        self.db_path = f"{uri}/{db_name}"
        self._mongo_return_document = MongoReturnDocument

        self.mongo_db.projects.create_index([("name", ASCENDING)], unique=True)
        self.mongo_db.projects.create_index([("root_path", ASCENDING)], unique=True)
        self.mongo_db.agent_profiles.create_index([("agent_key", ASCENDING)], unique=True)
        self.mongo_db.memories.create_index([("project_id", ASCENDING), ("created_at", DESCENDING)])
        self.mongo_db.conversations.create_index(
            [("project_id", ASCENDING), ("created_at", DESCENDING)]
        )
        self.mongo_db.command_runs.create_index(
            [("project_id", ASCENDING), ("created_at", DESCENDING)]
        )
        self._seed_profiles()

    def _create_sqlite_schema(self) -> None:
        connection = self._sqlite_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                root_path TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_key TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                model_name TEXT NOT NULL,
                is_enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                relevance REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NULL,
                user_input TEXT NOT NULL,
                assistant_output TEXT NOT NULL,
                trace_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS command_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NULL,
                command_text TEXT NOT NULL,
                return_code INTEGER NOT NULL,
                output_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id)
            )
            """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_project_time ON memories(project_id, created_at DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_project_time ON conversations(project_id, created_at DESC)"
        )

        connection.commit()

    def _seed_profiles(self) -> None:
        defaults = [
            (
                "thought",
                "Thought Agent",
                "analyst",
                (
                    "You are a strategic analyst. Return concise bullets only with goal, "
                    "assumptions, risks, missing data, and next step."
                ),
                "qwen2.5:7b",
            ),
            (
                "review",
                "Review Agent",
                "critic",
                (
                    "You are a critical reviewer. Challenge weak assumptions and prioritize "
                    "the most important correction."
                ),
                "qwen2.5:7b",
            ),
            (
                "action",
                "Action Agent",
                "executor",
                (
                    "You are an execution planner. Produce practical answers and optionally "
                    "return machine-readable local actions."
                ),
                "deepseek-r1:8b",
            ),
        ]
        now = _utc_now()

        if self.backend == "sqlite":
            connection = self._sqlite_connection()
            cursor = connection.cursor()
            for key, name, role, prompt, model in defaults:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO agent_profiles
                    (agent_key, display_name, role, system_prompt, model_name, is_enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (key, name, role, prompt, model, now, now),
                )
            connection.commit()
            return

        profiles = self._mongo_db().agent_profiles
        for key, name, role, prompt, model in defaults:
            existing = profiles.find_one({"agent_key": key}, {"_id": 1})
            if existing:
                continue
            profiles.insert_one(
                {
                    "_id": self._mongo_next_id("agent_profiles"),
                    "agent_key": key,
                    "display_name": name,
                    "role": role,
                    "system_prompt": prompt,
                    "model_name": model,
                    "is_enabled": True,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def upsert_project(self, *, name: str, root_path: str, description: str = "") -> ProjectRecord:
        now = _utc_now()
        clean_name = name.strip()
        clean_path = root_path.strip()
        clean_description = description.strip()

        if self.backend == "sqlite":
            connection = self._sqlite_connection()
            cursor = connection.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO projects(name, root_path, description, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        root_path=excluded.root_path,
                        description=excluded.description
                    """,
                    (clean_name, clean_path, clean_description, now),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "No se pudo guardar el proyecto: root_path ya esta siendo usado por otro proyecto."
                ) from exc
            row = cursor.execute(
                "SELECT * FROM projects WHERE name = ?",
                (clean_name,),
            ).fetchone()
            return self._row_to_project(row)

        projects = self._mongo_db().projects
        existing = projects.find_one({"name": clean_name})
        if existing:
            try:
                projects.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {"root_path": clean_path, "description": clean_description}},
                )
            except Exception as exc:
                if "duplicate key error" in str(exc).lower():
                    raise ValueError(
                        "No se pudo guardar el proyecto: root_path ya esta siendo usado por otro proyecto."
                    ) from exc
                raise
            existing["root_path"] = clean_path
            existing["description"] = clean_description
            return self._doc_to_project(existing)

        doc = {
            "_id": self._mongo_next_id("projects"),
            "name": clean_name,
            "root_path": clean_path,
            "description": clean_description,
            "created_at": now,
        }
        try:
            projects.insert_one(doc)
        except Exception as exc:
            if "duplicate key error" in str(exc).lower():
                raise ValueError(
                    "No se pudo guardar el proyecto: nombre o root_path duplicado."
                ) from exc
            raise
        return self._doc_to_project(doc)

    def get_project(self, project_id: int) -> ProjectRecord | None:
        if self.backend == "sqlite":
            row = self._sqlite_connection().execute(
                "SELECT * FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
            return self._row_to_project(row) if row else None

        doc = self._mongo_db().projects.find_one({"_id": int(project_id)})
        return self._doc_to_project(doc) if doc else None

    def get_project_by_name(self, name: str) -> ProjectRecord | None:
        clean_name = name.strip()
        if self.backend == "sqlite":
            row = self._sqlite_connection().execute(
                "SELECT * FROM projects WHERE name = ?",
                (clean_name,),
            ).fetchone()
            return self._row_to_project(row) if row else None

        doc = self._mongo_db().projects.find_one({"name": clean_name})
        return self._doc_to_project(doc) if doc else None

    def list_projects(self) -> list[ProjectRecord]:
        if self.backend == "sqlite":
            rows = self._sqlite_connection().execute(
                "SELECT * FROM projects ORDER BY created_at ASC"
            ).fetchall()
            return [self._row_to_project(row) for row in rows]

        docs = self._mongo_db().projects.find({}, sort=[("created_at", 1)])
        return [self._doc_to_project(doc) for doc in docs]

    def update_agent_profile(
        self,
        *,
        agent_key: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        is_enabled: bool | None = None,
    ) -> None:
        clean_key = agent_key.strip()

        if self.backend == "sqlite":
            updates: list[str] = []
            values: list[Any] = []
            if system_prompt is not None:
                updates.append("system_prompt = ?")
                values.append(system_prompt.strip())
            if model_name is not None:
                updates.append("model_name = ?")
                values.append(model_name.strip())
            if is_enabled is not None:
                updates.append("is_enabled = ?")
                values.append(1 if is_enabled else 0)
            if not updates:
                return

            updates.append("updated_at = ?")
            values.append(_utc_now())
            values.append(clean_key)

            sql = f"UPDATE agent_profiles SET {', '.join(updates)} WHERE agent_key = ?"
            connection = self._sqlite_connection()
            connection.execute(sql, tuple(values))
            connection.commit()
            return

        changes: dict[str, Any] = {}
        if system_prompt is not None:
            changes["system_prompt"] = system_prompt.strip()
        if model_name is not None:
            changes["model_name"] = model_name.strip()
        if is_enabled is not None:
            changes["is_enabled"] = bool(is_enabled)
        if not changes:
            return

        changes["updated_at"] = _utc_now()
        self._mongo_db().agent_profiles.update_one({"agent_key": clean_key}, {"$set": changes})

    def create_agent_profile(
        self,
        *,
        agent_key: str,
        display_name: str,
        role: str,
        system_prompt: str,
        model_name: str,
        is_enabled: bool = True,
    ) -> AgentProfileRecord:
        clean_key = agent_key.strip().lower()
        clean_name = display_name.strip()
        clean_role = role.strip()
        clean_prompt = system_prompt.strip()
        clean_model = model_name.strip()
        now = _utc_now()

        if self.backend == "sqlite":
            connection = self._sqlite_connection()
            cursor = connection.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO agent_profiles
                    (agent_key, display_name, role, system_prompt, model_name, is_enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_key,
                        clean_name,
                        clean_role,
                        clean_prompt,
                        clean_model,
                        1 if is_enabled else 0,
                        now,
                        now,
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"No se pudo crear el agente '{clean_key}': agent_key duplicado."
                ) from exc

            row = cursor.execute(
                "SELECT * FROM agent_profiles WHERE agent_key = ?",
                (clean_key,),
            ).fetchone()
            if row is None:
                raise RuntimeError("No se pudo leer el perfil recien creado.")
            return self._row_to_profile(row)

        doc = {
            "_id": self._mongo_next_id("agent_profiles"),
            "agent_key": clean_key,
            "display_name": clean_name,
            "role": clean_role,
            "system_prompt": clean_prompt,
            "model_name": clean_model,
            "is_enabled": bool(is_enabled),
            "created_at": now,
            "updated_at": now,
        }
        try:
            self._mongo_db().agent_profiles.insert_one(doc)
        except Exception as exc:
            if "duplicate key error" in str(exc).lower():
                raise ValueError(
                    f"No se pudo crear el agente '{clean_key}': agent_key duplicado."
                ) from exc
            raise
        return self._doc_to_profile(doc)

    def get_agent_profiles(self) -> dict[str, AgentProfileRecord]:
        if self.backend == "sqlite":
            rows = self._sqlite_connection().execute(
                "SELECT * FROM agent_profiles ORDER BY id ASC"
            ).fetchall()
            profiles = [self._row_to_profile(row) for row in rows]
            return {profile.agent_key: profile for profile in profiles}

        docs = self._mongo_db().agent_profiles.find({}, sort=[("_id", 1)])
        profiles = [self._doc_to_profile(doc) for doc in docs]
        return {profile.agent_key: profile for profile in profiles}

    def add_memory(
        self,
        *,
        project_id: int | None,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        relevance: float = 1.0,
    ) -> None:
        metadata_json = _json_string(metadata)

        if self.backend == "sqlite":
            connection = self._sqlite_connection()
            connection.execute(
                """
                INSERT INTO memories(project_id, memory_type, content, metadata_json, relevance, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    memory_type.strip(),
                    content.strip(),
                    metadata_json,
                    relevance,
                    _utc_now(),
                ),
            )
            connection.commit()
            return

        self._mongo_db().memories.insert_one(
            {
                "_id": self._mongo_next_id("memories"),
                "project_id": int(project_id) if project_id is not None else None,
                "memory_type": memory_type.strip(),
                "content": content.strip(),
                "metadata_json": metadata_json,
                "relevance": float(relevance),
                "created_at": _utc_now(),
            }
        )

    def recent_memories(
        self,
        *,
        project_id: int | None,
        limit: int = 12,
    ) -> list[MemoryRecord]:
        if self.backend == "sqlite":
            rows = self._sqlite_connection().execute(
                """
                SELECT * FROM memories
                WHERE project_id IS ? OR project_id IS NULL
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
            return [self._row_to_memory(row) for row in rows]

        if project_id is None:
            query: dict[str, Any] = {"project_id": None}
        else:
            query = {"$or": [{"project_id": int(project_id)}, {"project_id": None}]}
        docs = self._mongo_db().memories.find(query, sort=[("created_at", -1)], limit=limit)
        return [self._doc_to_memory(doc) for doc in docs]

    def add_conversation(
        self,
        *,
        project_id: int | None,
        user_input: str,
        assistant_output: str,
        trace: dict[str, Any] | None = None,
    ) -> None:
        trace_json = _json_string(trace)

        if self.backend == "sqlite":
            connection = self._sqlite_connection()
            connection.execute(
                """
                INSERT INTO conversations(project_id, user_input, assistant_output, trace_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    user_input.strip(),
                    assistant_output.strip(),
                    trace_json,
                    _utc_now(),
                ),
            )
            connection.commit()
            return

        self._mongo_db().conversations.insert_one(
            {
                "_id": self._mongo_next_id("conversations"),
                "project_id": int(project_id) if project_id is not None else None,
                "user_input": user_input.strip(),
                "assistant_output": assistant_output.strip(),
                "trace_json": trace_json,
                "created_at": _utc_now(),
            }
        )

    def recent_conversations(
        self,
        *,
        project_id: int | None,
        limit: int = 20,
    ) -> list[ConversationRecord]:
        if self.backend == "sqlite":
            rows = self._sqlite_connection().execute(
                """
                SELECT * FROM conversations
                WHERE project_id IS ? OR project_id IS NULL
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
            return [self._row_to_conversation(row) for row in rows]

        if project_id is None:
            query: dict[str, Any] = {"project_id": None}
        else:
            query = {"$or": [{"project_id": int(project_id)}, {"project_id": None}]}
        docs = self._mongo_db().conversations.find(
            query, sort=[("created_at", -1)], limit=limit
        )
        return [self._doc_to_conversation(doc) for doc in docs]

    def log_command_run(
        self,
        *,
        project_id: int | None,
        command_text: str,
        return_code: int,
        output_text: str,
    ) -> None:
        if self.backend == "sqlite":
            connection = self._sqlite_connection()
            connection.execute(
                """
                INSERT INTO command_runs(project_id, command_text, return_code, output_text, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, command_text, return_code, output_text, _utc_now()),
            )
            connection.commit()
            return

        self._mongo_db().command_runs.insert_one(
            {
                "_id": self._mongo_next_id("command_runs"),
                "project_id": int(project_id) if project_id is not None else None,
                "command_text": command_text,
                "return_code": int(return_code),
                "output_text": output_text,
                "created_at": _utc_now(),
            }
        )

    def close(self) -> None:
        if self.backend == "sqlite":
            if self.connection is not None:
                self.connection.close()
            return

        if self.mongo_client is not None:
            self.mongo_client.close()

    def _sqlite_connection(self) -> sqlite3.Connection:
        if self.connection is None:
            raise RuntimeError("SQLite connection is not initialized.")
        return self.connection

    def _mongo_db(self) -> Any:
        if self.mongo_db is None:
            raise RuntimeError("MongoDB client is not initialized.")
        return self.mongo_db

    def _mongo_next_id(self, counter_name: str) -> int:
        counters = self._mongo_db().counters
        doc = counters.find_one_and_update(
            {"_id": counter_name},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=self._mongo_return_document.AFTER,
        )
        return int(doc["seq"])

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            id=int(row["id"]),
            name=str(row["name"]),
            root_path=str(row["root_path"]),
            description=str(row["description"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> AgentProfileRecord:
        return AgentProfileRecord(
            id=int(row["id"]),
            agent_key=str(row["agent_key"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
            system_prompt=str(row["system_prompt"]),
            model_name=str(row["model_name"]),
            is_enabled=bool(row["is_enabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=int(row["id"]),
            project_id=int(row["project_id"]) if row["project_id"] is not None else None,
            memory_type=str(row["memory_type"]),
            content=str(row["content"]),
            metadata_json=str(row["metadata_json"]),
            relevance=float(row["relevance"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
        return ConversationRecord(
            id=int(row["id"]),
            project_id=int(row["project_id"]) if row["project_id"] is not None else None,
            user_input=str(row["user_input"]),
            assistant_output=str(row["assistant_output"]),
            trace_json=str(row["trace_json"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _doc_to_project(doc: dict[str, Any]) -> ProjectRecord:
        return ProjectRecord(
            id=int(doc["_id"]),
            name=str(doc.get("name", "")),
            root_path=str(doc.get("root_path", "")),
            description=str(doc.get("description", "")),
            created_at=str(doc.get("created_at", "")),
        )

    @staticmethod
    def _doc_to_profile(doc: dict[str, Any]) -> AgentProfileRecord:
        return AgentProfileRecord(
            id=int(doc["_id"]),
            agent_key=str(doc.get("agent_key", "")),
            display_name=str(doc.get("display_name", "")),
            role=str(doc.get("role", "")),
            system_prompt=str(doc.get("system_prompt", "")),
            model_name=str(doc.get("model_name", "")),
            is_enabled=bool(doc.get("is_enabled", True)),
            created_at=str(doc.get("created_at", "")),
            updated_at=str(doc.get("updated_at", "")),
        )

    @staticmethod
    def _doc_to_memory(doc: dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=int(doc["_id"]),
            project_id=int(doc["project_id"]) if doc.get("project_id") is not None else None,
            memory_type=str(doc.get("memory_type", "")),
            content=str(doc.get("content", "")),
            metadata_json=str(doc.get("metadata_json", "{}")),
            relevance=float(doc.get("relevance", 1.0)),
            created_at=str(doc.get("created_at", "")),
        )

    @staticmethod
    def _doc_to_conversation(doc: dict[str, Any]) -> ConversationRecord:
        return ConversationRecord(
            id=int(doc["_id"]),
            project_id=int(doc["project_id"]) if doc.get("project_id") is not None else None,
            user_input=str(doc.get("user_input", "")),
            assistant_output=str(doc.get("assistant_output", "")),
            trace_json=str(doc.get("trace_json", "{}")),
            created_at=str(doc.get("created_at", "")),
        )

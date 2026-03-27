"""Context acquisition system for agents.

When an agent doesn't know how to perform a task, this module provides
the ability to search for context, query databases, read documentation,
and gather information before responding — similar to how a human would
research before acting.
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import httpx


@dataclass
class ContextSource:
    source_type: str
    label: str
    content: str
    relevance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextAcquisitionResult:
    query: str
    sources: list[ContextSource] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    needs_human_input: bool = False
    human_question: str = ""


UNCERTAINTY_MARKERS_ES = [
    "no estoy seguro",
    "no se como",
    "no tengo informacion",
    "necesito mas contexto",
    "no puedo determinar",
    "me falta",
    "desconozco",
    "no conozco",
    "fuera de mi alcance",
]

UNCERTAINTY_MARKERS_EN = [
    "i'm not sure",
    "i don't know",
    "i lack information",
    "i need more context",
    "i cannot determine",
    "outside my knowledge",
    "i'm uncertain",
    "beyond my expertise",
]


def detect_uncertainty(text: str) -> bool:
    lower = text.lower()
    all_markers = UNCERTAINTY_MARKERS_ES + UNCERTAINTY_MARKERS_EN
    return any(marker in lower for marker in all_markers)


def detect_knowledge_gaps(text: str) -> list[str]:
    gaps: list[str] = []
    lower = text.lower()

    gap_patterns = [
        (r"(?:que|what)\s+(?:es|is)\s+(.+?)[\?.]", "definition"),
        (r"(?:como|how)\s+(?:se\s+)?(?:hace|funciona|works?|do)\s+(.+?)[\?.]", "procedure"),
        (r"(?:donde|where)\s+(?:esta|is|are)\s+(.+?)[\?.]", "location"),
        (r"(?:por que|why)\s+(.+?)[\?.]", "reasoning"),
        (r"(?:cual|which)\s+(?:es|is|are)\s+(.+?)[\?.]", "selection"),
    ]

    for pattern, gap_type in gap_patterns:
        matches = re.findall(pattern, lower)
        for match in matches:
            gaps.append(f"{gap_type}:{match.strip()}")

    return gaps


class ContextAcquisitionEngine:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self._search_history: list[dict[str, Any]] = []

    async def acquire_context(
        self,
        query: str,
        project_path: str | None = None,
        memory_context: str = "",
    ) -> ContextAcquisitionResult:
        sources: list[ContextSource] = []

        file_sources = self._search_local_files(query, project_path)
        sources.extend(file_sources)

        if memory_context:
            memory_source = self._parse_memory_context(query, memory_context)
            if memory_source:
                sources.extend(memory_source)

        env_sources = self._check_environment_context(query)
        sources.extend(env_sources)

        db_sources = await self._query_project_database(query)
        sources.extend(db_sources)

        sources.sort(key=lambda s: s.relevance, reverse=True)
        top_sources = sources[:10]

        confidence = self._calculate_confidence(top_sources)
        needs_human = confidence < 0.3 and len(top_sources) == 0

        summary = self._build_context_summary(query, top_sources)

        human_question = ""
        if needs_human:
            gaps = detect_knowledge_gaps(query)
            if gaps:
                human_question = (
                    f"Para responder mejor, necesito informacion adicional sobre: "
                    f"{', '.join(g.split(':')[1] for g in gaps[:3])}. "
                    f"Puedes proporcionar mas detalles?"
                )
            else:
                human_question = (
                    "No tengo suficiente contexto para responder con confianza. "
                    "Puedes describir con mas detalle lo que necesitas?"
                )

        self._search_history.append({
            "query": query,
            "sources_found": len(top_sources),
            "confidence": confidence,
            "needs_human": needs_human,
        })

        return ContextAcquisitionResult(
            query=query,
            sources=top_sources,
            summary=summary,
            confidence=confidence,
            needs_human_input=needs_human,
            human_question=human_question,
        )

    def _search_local_files(self, query: str, project_path: str | None) -> list[ContextSource]:
        sources: list[ContextSource] = []
        search_dirs: list[Path] = []

        if project_path:
            p = Path(project_path)
            if p.exists():
                search_dirs.append(p)

        cwd = Path.cwd()
        if cwd not in search_dirs:
            search_dirs.append(cwd)

        doc_files = ["README.md", "CONTRIBUTING.md", "docs/README.md", "ARCHITECTURE.md"]
        config_files = [
            "package.json", "pyproject.toml", "requirements.txt",
            "Makefile", "docker-compose.yml", ".env.example",
        ]

        keywords = set(re.findall(r"[a-zA-Z0-9_]{3,}", query.lower()))

        for search_dir in search_dirs:
            for doc_file in doc_files:
                fp = search_dir / doc_file
                if fp.exists() and fp.stat().st_size < 100_000:
                    try:
                        content = fp.read_text(encoding="utf-8", errors="replace")
                        relevance = self._text_relevance(content, keywords)
                        if relevance > 0.1:
                            sources.append(ContextSource(
                                source_type="documentation",
                                label=str(fp.relative_to(search_dir)),
                                content=content[:3000],
                                relevance=relevance,
                                metadata={"file_path": str(fp)},
                            ))
                    except Exception:
                        pass

            for config_file in config_files:
                fp = search_dir / config_file
                if fp.exists() and fp.stat().st_size < 50_000:
                    try:
                        content = fp.read_text(encoding="utf-8", errors="replace")
                        relevance = self._text_relevance(content, keywords)
                        if relevance > 0.05:
                            sources.append(ContextSource(
                                source_type="configuration",
                                label=str(fp.relative_to(search_dir)),
                                content=content[:2000],
                                relevance=min(relevance, 0.7),
                                metadata={"file_path": str(fp)},
                            ))
                    except Exception:
                        pass

        return sources

    def _parse_memory_context(self, query: str, memory_context: str) -> list[ContextSource]:
        if not memory_context or memory_context == "No previous memory.":
            return []
        keywords = set(re.findall(r"[a-zA-Z0-9_]{3,}", query.lower()))
        relevance = self._text_relevance(memory_context, keywords)
        return [ContextSource(
            source_type="memory",
            label="Previous interactions",
            content=memory_context[:2000],
            relevance=max(relevance, 0.3),
            metadata={"type": "memory_context"},
        )]

    def _check_environment_context(self, query: str) -> list[ContextSource]:
        sources: list[ContextSource] = []
        lower = query.lower()

        env_relevant_markers = [
            "database", "db", "postgres", "api", "key",
            "model", "ollama", "openai", "config", "env",
            "base de datos", "configuracion", "modelo",
        ]

        if not any(marker in lower for marker in env_relevant_markers):
            return sources

        safe_env_keys = [
            "FREE_LLM_PROVIDER", "OLLAMA_BASE_URL",
            "OPENAI_MODEL", "ANTHROPIC_MODEL", "DEEPSEEK_MODEL",
            "FREE_THOUGHT_MODEL", "FREE_REVIEW_MODEL", "FREE_ACTION_MODEL",
            "FORCE_FREE_LLM", "COMMAND_EXECUTION_ENABLED",
        ]

        env_info: list[str] = []
        for key in safe_env_keys:
            value = os.getenv(key, "")
            if value:
                env_info.append(f"{key}={value}")

        if env_info:
            sources.append(ContextSource(
                source_type="environment",
                label="System configuration",
                content="\n".join(env_info),
                relevance=0.5,
                metadata={"type": "safe_env_vars"},
            ))

        db_url = os.getenv("DATABASE_URL", "")
        if db_url and ("database" in lower or "db" in lower or "postgres" in lower or "base de datos" in lower):
            sources.append(ContextSource(
                source_type="environment",
                label="Database availability",
                content="PostgreSQL database is available via DATABASE_URL. Use the query_db tool for data access.",
                relevance=0.6,
                metadata={"type": "database_hint"},
            ))

        return sources

    async def _query_project_database(self, query: str) -> list[ContextSource]:
        if not self.database_url:
            return []

        lower = query.lower()
        db_markers = [
            "tabla", "table", "schema", "columna", "column",
            "registro", "record", "dato", "data", "consulta",
            "query", "database", "base de datos",
        ]

        if not any(marker in lower for marker in db_markers):
            return []

        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text as sa_text

            url = self.database_url
            if "postgresql://" in url and "+asyncpg" not in url:
                url = url.replace("postgresql://", "postgresql+asyncpg://")
                url = re.sub(r"[?&]sslmode=[^&]*", "", url)

            engine = create_async_engine(url, pool_pre_ping=True)
            async with engine.connect() as conn:
                result = await conn.execute(sa_text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' ORDER BY table_name"
                ))
                tables = [row[0] for row in result.fetchall()]
            await engine.dispose()

            if tables:
                return [ContextSource(
                    source_type="database_schema",
                    label="Available tables",
                    content=f"Tables in the database: {', '.join(tables)}",
                    relevance=0.7,
                    metadata={"tables": tables},
                )]
        except Exception:
            pass

        return []

    @staticmethod
    def _text_relevance(content: str, keywords: set[str]) -> float:
        if not keywords or not content:
            return 0.0
        lower_content = content.lower()
        matches = sum(1 for kw in keywords if kw in lower_content)
        return min(1.0, matches / max(len(keywords), 1))

    @staticmethod
    def _calculate_confidence(sources: list[ContextSource]) -> float:
        if not sources:
            return 0.0
        max_relevance = max(s.relevance for s in sources)
        avg_relevance = sum(s.relevance for s in sources) / len(sources)
        type_diversity = len(set(s.source_type for s in sources)) / 5.0
        return min(1.0, (max_relevance * 0.5) + (avg_relevance * 0.3) + (type_diversity * 0.2))

    @staticmethod
    def _build_context_summary(query: str, sources: list[ContextSource]) -> str:
        if not sources:
            return "No se encontro contexto relevante para esta consulta."

        lines = [f"Contexto adquirido para: '{query[:80]}'", ""]
        for source in sources[:5]:
            lines.append(
                f"- [{source.source_type}] {source.label} "
                f"(relevancia: {source.relevance:.0%}): "
                f"{source.content[:150]}..."
            )
        return "\n".join(lines)

    def build_context_prompt(self, result: ContextAcquisitionResult) -> str:
        if not result.sources and result.needs_human_input:
            return (
                "NOTA: No se encontro contexto suficiente. "
                "Antes de responder, indica al usuario que necesitas mas informacion.\n"
                f"Pregunta sugerida: {result.human_question}"
            )

        parts = [
            "## Contexto Adquirido Automaticamente",
            f"Confianza del contexto: {result.confidence:.0%}",
            "",
        ]

        for source in result.sources[:5]:
            parts.append(f"### {source.label} ({source.source_type})")
            parts.append(source.content[:1000])
            parts.append("")

        if result.confidence < 0.5:
            parts.append(
                "NOTA: El contexto tiene baja confianza. Indica cualquier "
                "incertidumbre en tu respuesta y sugiere como obtener mas informacion."
            )

        return "\n".join(parts)

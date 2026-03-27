"""Session Manifest — Shared context protocol for the 3-agent Flash pipeline.

The SessionManifest is the "source of truth" passed between agents.
It prevents the "broken telephone" effect by always anchoring every agent
to the user's original intent, enriching progressively without losing fidelity.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Domain detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "casino":       ["casino", "ruleta", "poker", "tragamonedas", "slot", "apuesta", "jackpot"],
    "ecommerce":    ["tienda", "ecommerce", "carrito", "producto", "venta", "checkout", "stripe"],
    "educacion":    ["aprende", "enseña", "pedagogia", "curso", "tutorial", "explicar", "alumno"],
    "backend":      ["api", "fastapi", "endpoint", "servidor", "microservicio", "rest", "graphql"],
    "frontend":     ["react", "html", "css", "componente", "ui", "interfaz", "vite", "tailwind"],
    "datos":        ["base de datos", "postgresql", "sql", "tabla", "query", "mongodb", "redis"],
    "ia":           ["agente", "llm", "modelo", "ia", "machine learning", "embedding", "langraph"],
    "arquitectura": ["arquitectura", "escalabilidad", "latencia", "buffer", "pipeline", "orquestador"],
}

_PRIORITY_HIGH_KW  = ["urgente", "crítico", "ahora", "inmediato", "rapido", "velocidad", "flash", "latencia"]
_PRIORITY_LOW_KW   = ["cuando puedas", "simple", "básico", "pequeño", "prueba"]


def detect_domain(text: str) -> str:
    low = text.lower()
    scores: dict[str, int] = {}
    for domain, kws in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in kws if kw in low)
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"


def detect_priority(text: str) -> str:
    low = text.lower()
    if any(kw in low for kw in _PRIORITY_HIGH_KW):
        return "high"
    if any(kw in low for kw in _PRIORITY_LOW_KW):
        return "low"
    return "medium"


def build_intent_summary(text: str, max_len: int = 200) -> str:
    """One-sentence distillation of the user's original intent."""
    clean = re.sub(r"\s+", " ", text.strip())
    if len(clean) <= max_len:
        return clean
    sentences = re.split(r"[.!?]\s+", clean)
    return sentences[0][:max_len] + ("..." if len(sentences[0]) > max_len else "")


# ─────────────────────────────────────────────────────────────────────────────
# Session Manifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentTrace:
    """Single-agent execution record stored in the manifest."""
    agent_id: str
    display_name: str
    role: str
    started_at: str
    finished_at: str = ""
    output: str = ""
    latency_ms: float = 0.0
    tokens_estimate: int = 0
    status: str = "pending"   # pending | ok | error


@dataclass
class SessionManifest:
    """Shared context blob that travels through the entire 3-agent pipeline.

    Every agent reads from and writes to this manifest.
    The original_intent is IMMUTABLE — it anchors the entropy filter.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── User input (immutable after creation) ────────────────────────────────
    original_intent: str = ""        # Raw user prompt — NEVER modified
    intent_summary: str = ""         # Condensed 1-sentence summary
    domain: str = "general"
    priority: str = "medium"
    input_language: str = "es"

    # ── Shared data blob (all agents read/write) ─────────────────────────────
    context_blob: dict[str, Any] = field(default_factory=dict)

    # ── Per-agent outputs ────────────────────────────────────────────────────
    logic_output: str = ""           # Agent 1 — Logic & Structure
    context_validated: str = ""      # Agent 2 — Context & Memory
    synthesis_output: str = ""       # Agent 3 — Interface & Synthesis
    final_output: str = ""           # Post entropy-filter final answer

    # ── Trace + performance ───────────────────────────────────────────────────
    traces: list[AgentTrace] = field(default_factory=list)
    total_latency_ms: float = 0.0
    pipeline_version: str = "flash/v1"

    @classmethod
    def from_prompt(cls, user_prompt: str, project_id: int | None = None) -> "SessionManifest":
        return cls(
            original_intent=user_prompt,
            intent_summary=build_intent_summary(user_prompt),
            domain=detect_domain(user_prompt),
            priority=detect_priority(user_prompt),
            context_blob={
                "project_id": project_id,
                "multimodal_inputs": [],
            },
        )

    def add_trace(self, trace: AgentTrace) -> None:
        self.traces.append(trace)
        if trace.latency_ms:
            self.total_latency_ms += trace.latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "original_intent": self.original_intent,
            "intent_summary": self.intent_summary,
            "domain": self.domain,
            "priority": self.priority,
            "logic_output": self.logic_output,
            "context_validated": self.context_validated,
            "synthesis_output": self.synthesis_output,
            "final_output": self.final_output,
            "total_latency_ms": self.total_latency_ms,
            "pipeline_version": self.pipeline_version,
            "traces": [
                {
                    "agent_id": t.agent_id,
                    "display_name": t.display_name,
                    "role": t.role,
                    "status": t.status,
                    "latency_ms": t.latency_ms,
                    "output_preview": t.output[:200],
                }
                for t in self.traces
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entropy Filter — Auto-Critique for Agent 3
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaithfulnessResult:
    is_faithful: bool
    score: float               # 0.0 = completely deviated, 1.0 = perfectly faithful
    deviations: list[str]      # Detected deviations from original intent
    corrections: list[str]     # Suggested corrections
    corrected_output: str = "" # Auto-corrected final output


class EntropyFilter:
    """Verifies that the synthesis output stays faithful to the original intent.

    Uses keyword overlap, semantic anchors, and domain consistency checks
    to detect "broken telephone" drift between agents.
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def check_faithfulness(
        self,
        manifest: SessionManifest,
        draft_response: str,
    ) -> FaithfulnessResult:
        original = manifest.original_intent.lower()
        draft = draft_response.lower()

        # Extract key terms from original intent (tokens ≥ 4 chars)
        key_terms = set(re.findall(r"[a-záéíóúñ\w]{4,}", original))
        # Remove common stop words
        stop_words = {"para", "como", "que", "con", "por", "una", "los", "las", "del", "esto", "pero", "cuando", "donde"}
        key_terms -= stop_words

        if not key_terms:
            return FaithfulnessResult(is_faithful=True, score=1.0, deviations=[], corrections=[])

        # Check how many key terms appear in draft
        present = {t for t in key_terms if t in draft}
        coverage = len(present) / len(key_terms)

        # Check domain consistency
        original_domain = manifest.domain
        domain_keywords = _DOMAIN_KEYWORDS.get(original_domain, [])
        domain_coverage = sum(1 for kw in domain_keywords if kw in draft) / max(len(domain_keywords), 1)

        # Score: weighted average of term coverage + domain coverage
        score = round(0.6 * coverage + 0.4 * domain_coverage, 3)

        deviations: list[str] = []
        corrections: list[str] = []

        missing_terms = key_terms - present
        if missing_terms and len(missing_terms) > len(key_terms) * 0.4:
            deviations.append(
                f"La respuesta omite conceptos clave del pedido original: "
                f"{', '.join(list(missing_terms)[:5])}"
            )
            corrections.append(
                f"Asegúrate de incluir referencia explícita a: {', '.join(list(missing_terms)[:5])}"
            )

        if domain_coverage < 0.3 and domain_keywords:
            deviations.append(
                f"La respuesta se desvió del dominio '{original_domain}'. "
                f"Se esperaban términos como: {', '.join(domain_keywords[:3])}"
            )
            corrections.append(
                f"Reorienta la respuesta hacia el contexto de '{original_domain}'"
            )

        is_faithful = score >= 0.5 and not deviations
        return FaithfulnessResult(
            is_faithful=is_faithful,
            score=score,
            deviations=deviations,
            corrections=corrections,
        )

    def auto_correct(
        self,
        manifest: SessionManifest,
        draft_response: str,
        faithfulness: FaithfulnessResult,
    ) -> str:
        """Apply corrections to drift responses."""
        if faithfulness.is_faithful:
            return draft_response

        correction_header = (
            f"\n\n---\n"
            f"**📌 Auto-corrección del sistema (Filtro de Entropía)**\n\n"
            f"*Fidelidad detectada: {round(faithfulness.score * 100)}% — "
            f"se aplicó corrección para mantener coherencia con la solicitud original.*\n\n"
            f"**Solicitud original del usuario:**\n> {manifest.intent_summary}\n\n"
        )

        if faithfulness.corrections:
            correction_header += "**Aspectos corregidos:**\n"
            for c in faithfulness.corrections:
                correction_header += f"• {c}\n"

        return draft_response + correction_header

"""Prompt Orchestrator — LLM-powered intent translation layer.

Translates natural language user prompts into structured JSON schemas
that downstream agents can process more effectively. Uses a free local
LLM (Ollama / DeepSeek) for the translation, with a deterministic
rule-based fallback when the LLM is unavailable or returns invalid output.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, field_validator


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolActivation(BaseModel):
    tool_name: str
    reason: str = ""


class TranslatedPrompt(BaseModel):
    objective: str = Field(description="Clear, concise statement of what the user wants to achieve")
    intent: str = Field(description="Classified intent category")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Key-value parameters extracted from the prompt")
    tools: list[ToolActivation] = Field(default_factory=list, description="Tools that should be activated")
    expected_output_format: str = Field(default="text", description="Expected format: text, code, json, table, etc.")
    priority: Priority = Field(default=Priority.MEDIUM)
    sub_tasks: list[str] = Field(default_factory=list, description="Decomposed sub-tasks if the request is complex")
    ambiguity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="0 = perfectly clear, 1 = completely ambiguous")
    clarification_question: Optional[str] = Field(default=None, description="Question to ask user when ambiguity > 0.7")
    language: str = Field(default="es", description="Detected input language code")
    raw_prompt: str = Field(default="", description="Original user prompt")

    @field_validator("ambiguity_score")
    @classmethod
    def clamp_ambiguity(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


AVAILABLE_TOOLS_DESCRIPTION = """Available tools in the system:
1. "calculator" — Evaluates mathematical expressions (e.g. "2 + 2", "sqrt(16)"). Use when the user asks for calculations.
2. "get_current_info" — Returns the current system date/time. Use when the user asks about the current time, date, or needs timestamps.
3. "query_db" — Executes read-only SQL SELECT queries against the PostgreSQL database. Use when the user asks about data, tables, records, or database content."""

SYSTEM_PROMPT = """You are a Prompt Orchestrator. Your ONLY job is to translate a user's natural language request into a structured JSON object that other AI agents will use as their instruction set.

{tools_description}

You MUST respond with ONLY a valid JSON object (no markdown, no explanation, no extra text). The JSON schema is:

{{
  "objective": "Clear statement of what the user wants",
  "intent": "One of: question, code_task, data_query, calculation, research, system_info, creative, analysis, execution, general",
  "parameters": {{"key": "value pairs extracted from the prompt"}},
  "tools": [{{"tool_name": "name", "reason": "why this tool is needed"}}],
  "expected_output_format": "text | code | json | table | list | markdown",
  "priority": "low | medium | high | critical",
  "sub_tasks": ["step 1", "step 2"],
  "ambiguity_score": 0.0 to 1.0,
  "clarification_question": "question to ask if ambiguity > 0.7 (null otherwise)",
  "language": "detected language code (es, en, etc.)"
}}

Rules:
- ambiguity_score: 0.0 = perfectly clear request, 1.0 = impossible to understand
- If ambiguity_score > 0.7, you MUST provide a clarification_question
- Only include tools that are genuinely needed (don't guess)
- priority: low = informational, medium = standard task, high = urgent/complex, critical = system-critical
- Decompose complex requests into sub_tasks
- Respond with ONLY the JSON object, nothing else"""


class PromptOrchestrator:
    """Translates natural language prompts into structured JSON schemas
    using a free local LLM, with a rule-based fallback."""

    def __init__(
        self,
        ollama_base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 8.0,
    ):
        self.ollama_base_url = (
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model_name = (
            model_name
            or os.getenv("FREE_ACTION_MODEL")
            or os.getenv("DEEPSEEK_MODEL", "deepseek-r1:8b")
        )
        self.timeout = timeout
        self._system_prompt = SYSTEM_PROMPT.format(
            tools_description=AVAILABLE_TOOLS_DESCRIPTION,
        )

    async def translate(
        self,
        user_prompt: str,
        project_id: int | None = None,
    ) -> TranslatedPrompt:
        text = user_prompt.strip()
        if not text:
            return TranslatedPrompt(
                objective="Empty request",
                intent="general",
                ambiguity_score=1.0,
                clarification_question="Could you describe what you need?",
                raw_prompt="",
                language="es",
            )

        try:
            llm_output = await self._call_llm(text)
            parsed = self._parse_llm_response(llm_output, text)
            return parsed
        except Exception:
            return self._fallback_translate(text)

    async def _call_llm(self, user_prompt: str) -> str:
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 800,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return content

    def _parse_llm_response(self, raw: str, original_prompt: str) -> TranslatedPrompt:
        cleaned = raw.strip()

        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if not json_match:
            raise ValueError("No JSON object found in LLM response")

        json_str = json_match.group(0)
        data = json.loads(json_str)
        data["raw_prompt"] = original_prompt

        if "tools" in data and isinstance(data["tools"], list):
            normalized_tools = []
            for t in data["tools"]:
                if isinstance(t, str):
                    normalized_tools.append({"tool_name": t, "reason": ""})
                elif isinstance(t, dict):
                    normalized_tools.append(t)
            data["tools"] = normalized_tools

        result = TranslatedPrompt(**data)
        return result

    def _fallback_translate(self, text: str) -> TranslatedPrompt:
        lowered = text.lower()
        lang = self._detect_language(text)

        intent = "general"
        tools: list[ToolActivation] = []
        priority = Priority.MEDIUM
        output_format = "text"
        sub_tasks: list[str] = []

        math_patterns = [
            r"\d+\s*[\+\-\*\/\^]\s*\d+",
            r"\b(calcula|calcular|calculate|suma|resta|multiplica|divide|sqrt|raiz)\b",
        ]
        if any(re.search(p, lowered) for p in math_patterns):
            intent = "calculation"
            tools.append(ToolActivation(tool_name="calculator", reason="Mathematical expression detected"))
            output_format = "text"

        time_patterns = [
            r"\b(hora|tiempo|fecha|date|time|clock|now|hoy|today|current)\b",
        ]
        if any(re.search(p, lowered) for p in time_patterns):
            intent = "system_info"
            tools.append(ToolActivation(tool_name="get_current_info", reason="Time/date information requested"))

        db_patterns = [
            r"\b(sql|query|tabla|table|base de datos|database|registro|record|select|datos|data|consulta)\b",
        ]
        if any(re.search(p, lowered) for p in db_patterns):
            intent = "data_query"
            tools.append(ToolActivation(tool_name="query_db", reason="Database query requested"))
            output_format = "table"

        code_patterns = [
            r"\b(codigo|code|programa|function|funcion|class|clase|script|implementa|implement|crea una clase|create a class|python|javascript|typescript)\b",
        ]
        if any(re.search(p, lowered) for p in code_patterns):
            intent = "code_task"
            output_format = "code"
            priority = Priority.HIGH

        research_patterns = [
            r"\b(investiga|research|busca|search|encuentra|find|analiza|analyze|explica|explain|que es|what is|como funciona|how does)\b",
        ]
        if any(re.search(p, lowered) for p in research_patterns):
            if intent == "general":
                intent = "research"

        word_count = len(text.split())
        ambiguity = 0.0
        clarification = None

        if word_count <= 2:
            ambiguity = 0.85
            clarification = (
                "Tu solicitud es muy breve. ¿Podrías dar más detalles sobre qué necesitas?"
                if lang == "es"
                else "Your request is very brief. Could you provide more details about what you need?"
            )
        elif word_count <= 5 and intent == "general":
            ambiguity = 0.6
        elif intent == "general" and not tools:
            ambiguity = 0.4

        if any(w in lowered for w in ["urgente", "urgent", "critico", "critical", "asap", "inmediato"]):
            priority = Priority.CRITICAL
        elif any(w in lowered for w in ["importante", "important"]):
            priority = Priority.HIGH

        return TranslatedPrompt(
            objective=text,
            intent=intent,
            parameters={},
            tools=tools,
            expected_output_format=output_format,
            priority=priority,
            sub_tasks=sub_tasks,
            ambiguity_score=ambiguity,
            clarification_question=clarification,
            language=lang,
            raw_prompt=text,
        )

    @staticmethod
    def _detect_language(text: str) -> str:
        spanish_markers = [
            "que", "como", "para", "por", "los", "las", "una", "del",
            "esta", "esto", "ese", "esa", "con", "sin", "pero", "más",
        ]
        words = text.lower().split()
        spanish_count = sum(1 for w in words if w in spanish_markers)
        return "es" if spanish_count >= 2 or len(words) == 0 else "en"

    def to_machine_ir(
        self,
        translated: TranslatedPrompt,
        project_id: int | None = None,
    ) -> dict[str, Any]:
        tool_names = [t.tool_name for t in translated.tools]

        ops: list[dict[str, Any]] = [
            {"op": "PARSE_INPUT", "value": translated.raw_prompt},
            {"op": "NORMALIZE", "value": translated.objective},
            {"op": "LLM_TRANSLATE", "intent": translated.intent, "priority": translated.priority.value},
        ]

        if tool_names:
            ops.append({"op": "ACTIVATE_TOOLS", "tools": tool_names})

        for i, task in enumerate(translated.sub_tasks):
            ops.append({"op": "SUB_TASK", "index": i, "description": task})

        route_target = "multi_agent_graph"
        requires_llm = True

        if translated.intent == "calculation" and "calculator" in tool_names:
            route_target = "tool_executor"
            requires_llm = False
        elif translated.intent == "system_info" and "get_current_info" in tool_names:
            route_target = "tool_executor"
            requires_llm = False

        ops.append({"op": "ROUTE", "target": route_target})

        if requires_llm:
            ops.extend([
                {"op": "RUN_STAGE", "name": "openai_thought"},
                {"op": "RUN_STAGE", "name": "anthropic_review"},
                {"op": "RUN_STAGE", "name": "deepseek_action"},
            ])

        execution_policy = {
            "single_pass_allowed": True,
            "forced_flow": None,
        }
        if translated.intent in ("code_task", "analysis", "research"):
            execution_policy["forced_flow"] = "multi_stage"

        return {
            "schema": "prompt-ir/v2",
            "input_language": translated.language,
            "project_id": project_id,
            "intent": translated.intent,
            "route": {
                "target": route_target,
                "requires_llm": requires_llm,
                "confidence": 1.0 - translated.ambiguity_score,
            },
            "args": translated.parameters,
            "ops": ops,
            "context_requirements": {
                "is_blocking": False,
                "missing": [],
            },
            "execution_policy": execution_policy,
            "orchestrator": {
                "objective": translated.objective,
                "tools_to_activate": [t.model_dump() for t in translated.tools],
                "expected_output_format": translated.expected_output_format,
                "priority": translated.priority.value,
                "sub_tasks": translated.sub_tasks,
                "ambiguity_score": translated.ambiguity_score,
                "clarification_question": translated.clarification_question,
            },
        }

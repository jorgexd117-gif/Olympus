"""Sub-agent definitions for the multi-agent system.

Defines specialized sub-agents that can collaborate on complex tasks,
similar to how modern AI assistants use internal specialization.
Each sub-agent has a specific role, system prompt, and capabilities.
"""

from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import httpx


class SubAgentRole(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    EXECUTOR = "executor"
    SYNTHESIZER = "synthesizer"


@dataclass
class SubAgentConfig:
    role: SubAgentRole
    display_name: str
    description: str
    system_prompt: str
    model_name: str = ""
    max_tokens: int = 500
    temperature: float = 0.2
    capabilities: list[str] = field(default_factory=list)


@dataclass
class SubAgentResult:
    role: str
    display_name: str
    status: str
    output: str
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentPipelineResult:
    task: str
    steps: list[SubAgentResult]
    final_output: str
    total_agents: int
    successful_agents: int
    pipeline_type: str = "sequential"


DEFAULT_SUBAGENTS: dict[SubAgentRole, SubAgentConfig] = {
    SubAgentRole.PLANNER: SubAgentConfig(
        role=SubAgentRole.PLANNER,
        display_name="Planificador",
        description="Descompone tareas complejas en pasos ejecutables",
        system_prompt=(
            "Eres un planificador estrategico. Tu trabajo es:\n"
            "1. Analizar la solicitud del usuario y descomponerla en subtareas claras\n"
            "2. Identificar dependencias entre subtareas\n"
            "3. Priorizar por impacto y viabilidad\n"
            "4. Estimar la complejidad de cada paso\n\n"
            "Responde con un plan estructurado usando este formato:\n"
            "## Plan de Ejecucion\n"
            "### Paso N: [nombre]\n"
            "- Objetivo: [que lograr]\n"
            "- Entrada: [que necesita]\n"
            "- Salida: [que produce]\n"
            "- Complejidad: [baja/media/alta]\n"
        ),
        capabilities=["task_decomposition", "dependency_analysis", "prioritization"],
    ),
    SubAgentRole.RESEARCHER: SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        display_name="Investigador",
        description="Recopila informacion y contexto relevante",
        system_prompt=(
            "Eres un investigador. Tu trabajo es:\n"
            "1. Identificar que informacion se necesita para completar la tarea\n"
            "2. Buscar en el contexto disponible (archivos, base de datos, memoria)\n"
            "3. Sintetizar hallazgos relevantes\n"
            "4. Identificar brechas de conocimiento\n\n"
            "Responde con:\n"
            "## Hallazgos\n"
            "- [hallazgo relevante]\n\n"
            "## Brechas de Conocimiento\n"
            "- [informacion que falta]\n\n"
            "## Recomendaciones\n"
            "- [como obtener la informacion faltante]\n"
        ),
        capabilities=["information_gathering", "synthesis", "gap_analysis"],
    ),
    SubAgentRole.CODER: SubAgentConfig(
        role=SubAgentRole.CODER,
        display_name="Programador",
        description="Genera y modifica codigo fuente",
        system_prompt=(
            "Eres un programador experto. Tu trabajo es:\n"
            "1. Escribir codigo limpio, bien documentado y funcional\n"
            "2. Seguir las convenciones del proyecto existente\n"
            "3. Manejar errores de forma explicita\n"
            "4. Incluir tipos y validaciones apropiadas\n\n"
            "Reglas:\n"
            "- Prefiere editar codigo existente sobre crear archivos nuevos\n"
            "- No uses datos mockeados; implementa funcionalidad real\n"
            "- Incluye manejo de errores explicito\n"
            "- Sigue el estilo del proyecto\n"
        ),
        capabilities=["code_generation", "code_modification", "debugging"],
        temperature=0.1,
    ),
    SubAgentRole.REVIEWER: SubAgentConfig(
        role=SubAgentRole.REVIEWER,
        display_name="Revisor",
        description="Revisa calidad, seguridad y correccion",
        system_prompt=(
            "Eres un revisor de calidad. Tu trabajo es:\n"
            "1. Verificar la correccion logica del trabajo\n"
            "2. Identificar problemas de seguridad\n"
            "3. Evaluar si cumple con los requisitos\n"
            "4. Sugerir mejoras concretas\n\n"
            "Responde con:\n"
            "## Veredicto: [APROBADO/REQUIERE_CAMBIOS/RECHAZADO]\n\n"
            "## Hallazgos Criticos\n"
            "- [problema critico si existe]\n\n"
            "## Sugerencias\n"
            "- [mejora sugerida]\n"
        ),
        capabilities=["code_review", "security_audit", "quality_check"],
    ),
    SubAgentRole.EXECUTOR: SubAgentConfig(
        role=SubAgentRole.EXECUTOR,
        display_name="Ejecutor",
        description="Ejecuta acciones y herramientas del sistema",
        system_prompt=(
            "Eres un agente ejecutor. Tu trabajo es:\n"
            "1. Tomar las decisiones de los agentes anteriores y ejecutarlas\n"
            "2. Usar las herramientas disponibles (calculator, get_current_info, query_db)\n"
            "3. Reportar resultados de forma clara\n"
            "4. Manejar errores y proponer alternativas\n\n"
            "Responde SOLO con JSON:\n"
            '{"final_response": "texto", "actions": [{"tool": "nombre", "input": "valor"}]}\n'
        ),
        capabilities=["tool_execution", "action_planning", "error_handling"],
        temperature=0.1,
    ),
    SubAgentRole.SYNTHESIZER: SubAgentConfig(
        role=SubAgentRole.SYNTHESIZER,
        display_name="Sintetizador",
        description="Integra resultados de multiples agentes en una respuesta coherente",
        system_prompt=(
            "Eres un sintetizador. Tu trabajo es:\n"
            "1. Integrar los resultados de todos los agentes anteriores\n"
            "2. Resolver contradicciones entre diferentes perspectivas\n"
            "3. Producir una respuesta final clara y accionable\n"
            "4. Destacar los puntos mas importantes\n\n"
            "Tu respuesta debe ser:\n"
            "- Coherente: integra todas las perspectivas\n"
            "- Accionable: con pasos claros siguientes\n"
            "- Honesta: indica areas de incertidumbre\n"
        ),
        capabilities=["synthesis", "conflict_resolution", "summarization"],
    ),
}


PIPELINE_TEMPLATES: dict[str, list[SubAgentRole]] = {
    "full_analysis": [
        SubAgentRole.PLANNER,
        SubAgentRole.RESEARCHER,
        SubAgentRole.REVIEWER,
        SubAgentRole.SYNTHESIZER,
    ],
    "code_task": [
        SubAgentRole.PLANNER,
        SubAgentRole.CODER,
        SubAgentRole.REVIEWER,
        SubAgentRole.SYNTHESIZER,
    ],
    "research": [
        SubAgentRole.RESEARCHER,
        SubAgentRole.REVIEWER,
        SubAgentRole.SYNTHESIZER,
    ],
    "quick_answer": [
        SubAgentRole.RESEARCHER,
        SubAgentRole.SYNTHESIZER,
    ],
    "execute": [
        SubAgentRole.PLANNER,
        SubAgentRole.EXECUTOR,
        SubAgentRole.SYNTHESIZER,
    ],
}


def detect_pipeline_type(query: str) -> str:
    lower = query.lower()

    code_markers = [
        "codigo", "code", "programa", "function", "clase", "class",
        "implementa", "implement", "debug", "fix", "corrige", "refactor",
        "api", "endpoint", "componente", "component",
    ]
    if any(marker in lower for marker in code_markers):
        return "code_task"

    execute_markers = [
        "ejecuta", "execute", "run", "corre", "comando", "command",
        "instala", "install", "deploy", "despliega",
    ]
    if any(marker in lower for marker in execute_markers):
        return "execute"

    research_markers = [
        "investiga", "research", "busca", "search", "encuentra", "find",
        "explica", "explain", "que es", "what is", "como funciona", "how does",
    ]
    if any(marker in lower for marker in research_markers):
        return "research"

    if len(lower.split()) < 10:
        return "quick_answer"

    return "full_analysis"


class SubAgentOrchestrator:
    def __init__(self) -> None:
        self.configs: dict[SubAgentRole, SubAgentConfig] = dict(DEFAULT_SUBAGENTS)
        self.execution_log: list[dict[str, Any]] = []

    def get_subagent_configs(self) -> list[dict[str, Any]]:
        return [
            {
                "role": config.role.value,
                "display_name": config.display_name,
                "description": config.description,
                "capabilities": config.capabilities,
                "model_name": config.model_name or self._default_model(),
            }
            for config in self.configs.values()
        ]

    async def run_pipeline(
        self,
        task: str,
        pipeline_type: str | None = None,
        context: str = "",
        ethics_prompt: str = "",
    ) -> SubAgentPipelineResult:
        if pipeline_type is None:
            pipeline_type = detect_pipeline_type(task)

        roles = PIPELINE_TEMPLATES.get(pipeline_type, PIPELINE_TEMPLATES["full_analysis"])
        steps: list[SubAgentResult] = []

        for role in roles:
            config = self.configs.get(role)
            if config is None:
                continue

            previous_output = "\n\n".join(
                f"--- {s.display_name} ({s.role}) ---\n{s.output}"
                for s in steps
            )

            result = await self._execute_subagent(
                config=config,
                task=task,
                context=context,
                previous_output=previous_output,
                ethics_prompt=ethics_prompt,
            )
            steps.append(result)

        successful = sum(1 for s in steps if s.status == "ok")
        final_output = steps[-1].output if steps else "No se genero salida."

        return SubAgentPipelineResult(
            task=task,
            steps=steps,
            final_output=final_output,
            total_agents=len(steps),
            successful_agents=successful,
            pipeline_type=pipeline_type,
        )

    async def run_single_subagent(
        self,
        role: SubAgentRole,
        task: str,
        context: str = "",
        ethics_prompt: str = "",
    ) -> SubAgentResult:
        config = self.configs.get(role)
        if config is None:
            return SubAgentResult(
                role=role.value,
                display_name="Unknown",
                status="error",
                output=f"Sub-agent role not found: {role.value}",
            )

        return await self._execute_subagent(
            config=config,
            task=task,
            context=context,
            previous_output="",
            ethics_prompt=ethics_prompt,
        )

    async def _execute_subagent(
        self,
        config: SubAgentConfig,
        task: str,
        context: str,
        previous_output: str,
        ethics_prompt: str,
    ) -> SubAgentResult:
        system_parts = []
        if ethics_prompt:
            system_parts.append(ethics_prompt)
        system_parts.append(config.system_prompt)
        system_prompt = "\n\n".join(system_parts)

        user_parts = []
        if context:
            user_parts.append(f"Contexto disponible:\n{context}")
        if previous_output:
            user_parts.append(f"Trabajo previo de otros agentes:\n{previous_output}")
        user_parts.append(f"Tarea:\n{task}")
        user_prompt = "\n\n".join(user_parts)

        model = config.model_name or self._default_model()

        try:
            output = await self._call_model(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            status = "ok"
            confidence = 0.8
        except Exception as exc:
            output = f"Error al ejecutar {config.display_name}: {exc}"
            status = "error"
            confidence = 0.0

        result = SubAgentResult(
            role=config.role.value,
            display_name=config.display_name,
            status=status,
            output=output,
            confidence=confidence,
            metadata={"model": model, "pipeline_position": config.role.value},
        )

        self.execution_log.append({
            "role": config.role.value,
            "status": status,
            "model": model,
            "task_preview": task[:100],
        })

        return result

    async def _call_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        provider = os.getenv("FREE_LLM_PROVIDER", "ollama").strip().lower()

        if provider == "ollama":
            return await self._call_ollama(model, system_prompt, user_prompt, temperature)

        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            return await self._call_openai_compatible(
                api_key=openai_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                temperature=temperature,
            )

        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if deepseek_key:
            return await self._call_openai_compatible(
                api_key=deepseek_key,
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                temperature=temperature,
            )

        return await self._call_ollama(model, system_prompt, user_prompt, temperature)

    async def _call_ollama(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": temperature},
        }
        timeout = float(os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS", "120"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            content = data.get("response", "")
        return str(content).strip()

    async def _call_openai_compatible(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        base_url: str,
        temperature: float,
    ) -> str:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _default_model() -> str:
        return os.getenv("FREE_ACTION_MODEL", "") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

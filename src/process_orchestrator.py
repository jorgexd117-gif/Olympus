"""LangChain-style process orchestrator for operational workflows."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain_core.runnables import RunnableLambda

from src.multi_agent import MultiAgentCoordinator
from src.persistence import AgentDatabase


@dataclass
class WorkflowDefinition:
    workflow_id: str
    name: str
    description: str
    steps: list[str]


@dataclass
class WorkflowStepResult:
    step: str
    status: str
    detail: str
    duration_ms: int


@dataclass
class WorkflowRunResult:
    workflow_id: str
    status: str
    summary: str
    steps: list[WorkflowStepResult]
    output: dict[str, Any]


class ProcessOrchestrator:
    """Runs operational workflows using Runnable pipelines."""

    def __init__(self, database: AgentDatabase, coordinator: MultiAgentCoordinator) -> None:
        self.database = database
        self.coordinator = coordinator
        self._workflows: dict[str, WorkflowDefinition] = {
            "diagnostic": WorkflowDefinition(
                workflow_id="diagnostic",
                name="Diagnostico General",
                description="Valida salud del sistema, modelos disponibles y estado de memoria.",
                steps=["health", "models", "project", "memory"],
            ),
            "assistant_quick": WorkflowDefinition(
                workflow_id="assistant_quick",
                name="Asistente Rapido",
                description="Ejecuta consulta asistida en modo rapido con contexto del proyecto.",
                steps=["health", "project", "assistant"],
            ),
            "agent_full": WorkflowDefinition(
                workflow_id="agent_full",
                name="Pipeline Completo",
                description="Ejecuta flujo multi-agente completo (thought/review/action).",
                steps=["health", "project", "agent_full"],
            ),
        }

    def list_workflows(self) -> list[WorkflowDefinition]:
        return list(self._workflows.values())

    async def run_workflow(
        self,
        *,
        workflow_id: str,
        project_id: int | None,
        user_prompt: str,
    ) -> WorkflowRunResult:
        if workflow_id not in self._workflows:
            raise ValueError(f"Unknown workflow: {workflow_id}")

        workflow = self._workflows[workflow_id]
        state: dict[str, Any] = {
            "workflow_id": workflow_id,
            "project_id": project_id,
            "user_prompt": user_prompt.strip(),
            "status": "running",
            "steps": [],
            "output": {},
        }

        chain: RunnableLambda = RunnableLambda(lambda payload: payload)
        for step_name in workflow.steps:
            chain = chain | RunnableLambda(self._build_step_runner(step_name))

        final_state = await chain.ainvoke(state)
        status = str(final_state.get("status", "completed"))
        if status == "running":
            status = "completed"

        summary = "Workflow completado."
        if status != "completed":
            summary = str(final_state.get("summary", "Workflow con errores."))
        elif workflow_id == "assistant_quick":
            summary = "Consulta asistida ejecutada."
        elif workflow_id == "agent_full":
            summary = "Pipeline completo ejecutado."

        return WorkflowRunResult(
            workflow_id=workflow_id,
            status=status,
            summary=summary,
            steps=list(final_state.get("steps", [])),
            output=dict(final_state.get("output", {})),
        )

    def _build_step_runner(self, step_name: str) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        step_map: dict[str, Callable[[dict[str, Any]], Awaitable[str]]] = {
            "health": self._step_health,
            "models": self._step_models,
            "project": self._step_project,
            "memory": self._step_memory,
            "assistant": self._step_assistant,
            "agent_full": self._step_agent_full,
        }
        if step_name not in step_map:
            raise ValueError(f"Unknown step: {step_name}")
        step_fn = step_map[step_name]

        async def _runner(state: dict[str, Any]) -> dict[str, Any]:
            if state.get("status") == "error":
                return state

            started = time.perf_counter()
            try:
                detail = await step_fn(state)
                duration_ms = int((time.perf_counter() - started) * 1000)
                step_result = WorkflowStepResult(
                    step=step_name,
                    status="ok",
                    detail=detail,
                    duration_ms=duration_ms,
                )
                state.setdefault("steps", []).append(step_result)
                return state
            except Exception as exc:  # pragma: no cover - defensive
                duration_ms = int((time.perf_counter() - started) * 1000)
                step_result = WorkflowStepResult(
                    step=step_name,
                    status="error",
                    detail=str(exc),
                    duration_ms=duration_ms,
                )
                state.setdefault("steps", []).append(step_result)
                state["status"] = "error"
                state["summary"] = f"Fallo en paso '{step_name}': {exc}"
                return state

        return _runner

    async def _step_health(self, state: dict[str, Any]) -> str:
        db_path = str(self.database.db_path)
        state_out = {"db_path": db_path}
        return self._store_output_and_return("health", state, state_out, f"DB activo en {db_path}.")

    async def _step_models(self, state: dict[str, Any]) -> str:
        profiles = self.coordinator.get_profiles()
        models = sorted(
            {
                profile.model_name.strip()
                for profile in profiles.values()
                if profile.model_name.strip()
            }
        )
        hf_model = os.getenv("HUGGINGFACE_MODEL", "").strip()
        if hf_model and hf_model not in models:
            models.append(hf_model)
        state_out = {"count": len(models), "models": models}
        return self._store_output_and_return("models", state, state_out, f"{len(models)} modelos listados.")

    async def _step_project(self, state: dict[str, Any]) -> str:
        project_id = state.get("project_id")
        project = self.database.get_project(project_id) if project_id else None
        if project is None:
            state_out = {"project_id": None, "name": None}
            return self._store_output_and_return(
                "project",
                state,
                state_out,
                "Sin proyecto activo; se usara contexto global.",
            )

        state_out = {"project_id": project.id, "name": project.name, "root_path": project.root_path}
        return self._store_output_and_return(
            "project",
            state,
            state_out,
            f"Proyecto activo: {project.name}.",
        )

    async def _step_memory(self, state: dict[str, Any]) -> str:
        project_id = state.get("project_id")
        memories = self.database.recent_memories(project_id=project_id, limit=5)
        preview = [item.content[:180] for item in memories]
        state_out = {"count": len(memories), "preview": preview}
        return self._store_output_and_return(
            "memory",
            state,
            state_out,
            f"{len(memories)} memorias recientes encontradas.",
        )

    async def _step_assistant(self, state: dict[str, Any]) -> str:
        prompt = str(state.get("user_prompt", "")).strip()
        if not prompt:
            prompt = "Resume el estado operativo en 3 lineas."
        project_id = state.get("project_id")
        result = await self.coordinator.assistant_turn(project_id=project_id, user_prompt=prompt)
        state_out = {"source": result.source, "reply": result.reply, "sections": result.sections}
        return self._store_output_and_return(
            "assistant",
            state,
            state_out,
            f"Asistente completado via {result.source}.",
        )

    async def _step_agent_full(self, state: dict[str, Any]) -> str:
        prompt = str(state.get("user_prompt", "")).strip()
        if not prompt:
            prompt = "Genera un resumen operativo del proyecto."
        project_id = state.get("project_id")
        result = await self.coordinator.run_agent(project_id=project_id, user_prompt=prompt)
        state_out = {"final_output": result.final_output, "sections": result.sections}
        return self._store_output_and_return(
            "agent_full",
            state,
            state_out,
            "Pipeline multi-agente finalizado.",
        )

    @staticmethod
    def _store_output_and_return(
        key: str,
        state: dict[str, Any],
        value: dict[str, Any],
        detail: str,
    ) -> str:
        output = state.setdefault("output", {})
        output[key] = value
        return detail

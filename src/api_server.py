"""HTTP API for the LangGraph multi-agent control layer."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload

from src.database import AsyncSessionLocal, engine
from src.models import Agent, AgentAssignment, Folder, ProcessType
from src.multi_agent import MultiAgentCoordinator
from src.persistence import AgentDatabase, AgentProfileRecord, ConversationRecord, MemoryRecord, ProjectRecord
from src.process_orchestrator import ProcessOrchestrator

load_dotenv()
load_dotenv(".env.local", override=True)


def _project_to_dict(project: ProjectRecord) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "root_path": project.root_path,
        "description": project.description,
        "created_at": project.created_at,
    }


def _profile_to_dict(profile: AgentProfileRecord) -> dict[str, Any]:
    return {
        "id": profile.id,
        "agent_key": profile.agent_key,
        "display_name": profile.display_name,
        "role": profile.role,
        "system_prompt": profile.system_prompt,
        "model_name": profile.model_name,
        "is_enabled": profile.is_enabled,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def _memory_to_dict(item: MemoryRecord) -> dict[str, Any]:
    try:
        metadata = json.loads(item.metadata_json)
        if not isinstance(metadata, dict):
            metadata = {}
    except json.JSONDecodeError:
        metadata = {}
    return {
        "id": item.id,
        "project_id": item.project_id,
        "memory_type": item.memory_type,
        "content": item.content,
        "metadata": metadata,
        "relevance": item.relevance,
        "created_at": item.created_at,
    }


def _conversation_to_dict(item: ConversationRecord) -> dict[str, Any]:
    try:
        trace = json.loads(item.trace_json)
        if not isinstance(trace, dict):
            trace = {}
    except json.JSONDecodeError:
        trace = {}
    return {
        "id": item.id,
        "project_id": item.project_id,
        "user_input": item.user_input,
        "assistant_output": item.assistant_output,
        "trace": trace,
        "created_at": item.created_at,
    }


def _fetch_ollama_models() -> list[str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return []
    except Exception:
        return []

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return []

    models = decoded.get("models")
    if not isinstance(models, list):
        return []

    output: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name", "")).strip()
        if name:
            output.append(name)
    return output


def _build_available_models(database: AgentDatabase) -> list[str]:
    discovered: set[str] = set()
    for env_key in (
        "FREE_THOUGHT_MODEL",
        "FREE_REVIEW_MODEL",
        "FREE_ACTION_MODEL",
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "DEEPSEEK_MODEL",
        "HUGGINGFACE_MODEL",
    ):
        value = os.getenv(env_key, "").strip()
        if value:
            discovered.add(value)

    for profile in database.get_agent_profiles().values():
        model_name = profile.model_name.strip()
        if model_name:
            discovered.add(model_name)

    for model in _fetch_ollama_models():
        discovered.add(model)

    return sorted(discovered, key=str.lower)


class AppState:
    """Runtime state for API resources."""

    def __init__(self) -> None:
        db_path = Path(os.getenv("AGENT_DB_PATH", "data/agent_memory.db"))
        self.database = AgentDatabase(db_path=db_path)
        self.coordinator = MultiAgentCoordinator(self.database)
        self.orchestrator = ProcessOrchestrator(self.database, self.coordinator)

    def close(self) -> None:
        self.database.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.runtime = AppState()
    try:
        yield
    finally:
        _app.state.runtime.close()


app = FastAPI(
    title="LangGraph Agent API",
    version="0.1.0",
    description="API facade for projects, profiles, memory, and multi-agent execution.",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
allow_all_origins = "*" in cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    # Browsers reject credentials for wildcard origins.
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _runtime() -> AppState:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=500, detail="Runtime is not initialized.")
    return runtime


class HealthResponse(BaseModel):
    status: str
    db_path: str
    db_backend: str
    db_notice: str | None = None


class ApiRootResponse(BaseModel):
    status: str
    message: str
    health_url: str
    docs_url: str
    endpoints: list[str]


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    root_path: str = Field(min_length=1, max_length=1024)
    description: str = Field(default="", max_length=4000)


class ProjectResponse(BaseModel):
    id: int
    name: str
    root_path: str
    description: str
    created_at: str


class ProfileResponse(BaseModel):
    id: int
    agent_key: str
    display_name: str
    role: str
    system_prompt: str
    model_name: str
    is_enabled: bool
    created_at: str
    updated_at: str


class ProfileUpdateRequest(BaseModel):
    system_prompt: str | None = Field(default=None, max_length=12000)
    model_name: str | None = Field(default=None, max_length=200)
    is_enabled: bool | None = None


class ProfileCreateRequest(BaseModel):
    agent_key: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str = Field(min_length=2, max_length=120)
    role: str = Field(min_length=2, max_length=120)
    system_prompt: str = Field(min_length=1, max_length=12000)
    model_name: str = Field(min_length=1, max_length=200)
    is_enabled: bool = True


class AgentRunRequest(BaseModel):
    user_prompt: str = Field(min_length=1, max_length=12000)
    project_id: int | None = None


class AgentRunResponse(BaseModel):
    sections: dict[str, str]
    final_output: str


class AssistantChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    project_id: int | None = None


class AssistantChatResponse(BaseModel):
    reply: str
    source: str
    project_id: int | None
    sections: dict[str, str]
    machine_translation: dict[str, Any]


class CommandRunRequest(BaseModel):
    command_text: str = Field(min_length=1, max_length=2000)
    project_id: int | None = None
    timeout_seconds: int = Field(default=120, ge=1, le=900)


class CommandRunResponse(BaseModel):
    return_code: int
    output: str


class OrchestratorWorkflowResponse(BaseModel):
    workflow_id: str
    name: str
    description: str
    steps: list[str]


class OrchestratorStepResponse(BaseModel):
    step: str
    status: str
    detail: str
    duration_ms: int


class OrchestratorRunRequest(BaseModel):
    workflow_id: str = Field(min_length=1, max_length=100)
    project_id: int | None = None
    user_prompt: str = Field(default="", max_length=12000)


class OrchestratorRunResponse(BaseModel):
    workflow_id: str
    status: str
    summary: str
    steps: list[OrchestratorStepResponse]
    output: dict[str, Any]


class TeamRunRequest(BaseModel):
    user_prompt: str = Field(min_length=1, max_length=12000)
    project_id: int | None = None
    agent_keys: list[str] = Field(default_factory=list)


class TeamRunStepResponse(BaseModel):
    agent_key: str
    display_name: str
    role: str
    model_name: str
    status: str
    output: str


class TeamRunResponse(BaseModel):
    project_id: int | None
    user_prompt: str
    final_output: str
    steps: list[TeamRunStepResponse]


class MemoryResponse(BaseModel):
    id: int
    project_id: int | None
    memory_type: str
    content: str
    metadata: dict[str, Any]
    relevance: float
    created_at: str


class ConversationResponse(BaseModel):
    id: int
    project_id: int | None
    user_input: str
    assistant_output: str
    trace: dict[str, Any]
    created_at: str


class AvailableModelsResponse(BaseModel):
    models: list[str]


# Weather tool models and cache
class WeatherResponse(BaseModel):
    city: str = Field(min_length=1, max_length=64)
    temperature_c: float
    units: Literal["C"] = "C"
    condition: str = Field(min_length=1, max_length=60)
    source: Literal["cache", "api"] = "api"

_WEATHER_CACHE: dict[str, tuple[float, WeatherResponse]] = {}
WEATHER_TTL_SECONDS = int(os.getenv("WEATHER_TTL_SECONDS", "300"))


async def _fetch_weather(city: str) -> WeatherResponse:
    """Fetch current weather for a city. Placeholder for real API integration."""
    normalized = city.strip()
    # TODO: Integrate with a real provider via httpx.AsyncClient using env credentials.
    # Deterministic stub for now
    condition = "Sunny"
    temp_c = 22.0
    return WeatherResponse(city=normalized, temperature_c=temp_c, condition=condition, units="C", source="api")


@app.get(
    "/tools/weather",
    response_model=WeatherResponse,
    summary="Obtiene el clima actual por ciudad",
    tags=["tools"],
)
async def tools_weather(
    city: str = Query(..., min_length=1, max_length=64, pattern=r"^[A-Za-zÀ-ÿ .'-]+$")
) -> WeatherResponse:
    key = city.strip().lower()
    if not key:
        raise HTTPException(status_code=400, detail="City must not be empty")

    now = time.time()
    cached = _WEATHER_CACHE.get(key)
    if cached and cached[0] > now:
        data = cached[1]
        # Return a copy with source annotated as cache
        return WeatherResponse(**data.dict(exclude={"source"}), source="cache")

    try:
        data = await _fetch_weather(city)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Weather provider unavailable") from exc

    _WEATHER_CACHE[key] = (now + WEATHER_TTL_SECONDS, data)
    return data


@app.get("/healthz")
async def healthz():
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        return {"status": "degraded"}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    runtime = _runtime()
    return HealthResponse(
        status="ok",
        db_path=str(runtime.database.db_path),
        db_backend=str(runtime.database.backend),
        db_notice=runtime.database.mongo_init_error,
    )


@app.get("/api/status", response_model=ApiRootResponse)
async def api_status() -> ApiRootResponse:
    return ApiRootResponse(
        status="ok",
        message="LangGraph Agent API running.",
        health_url="/health",
        docs_url="/docs",
        endpoints=[
            "/health",
            "/projects",
            "/profiles",
            "/profiles (POST create)",
            "/agent/run",
            "/agents/team/run",
            "/assistant/chat",
            "/commands/run",
            "/orchestrator/workflows",
            "/orchestrator/run",
            "/memories",
            "/conversations",
            "/models/available",
            "/tools/weather",
        ],
    )


@app.get("/", include_in_schema=False)
async def root_handler():
    index = Path("ui/dist/index.html")
    if index.is_file():
        return FileResponse(str(index))
    from fastapi.responses import JSONResponse
    return JSONResponse({
        "status": "ok",
        "message": "LangGraph Agent API running.",
        "docs_url": "/docs",
    })


@app.get("/projects", response_model=list[ProjectResponse])
async def list_projects() -> list[ProjectResponse]:
    runtime = _runtime()
    projects = runtime.coordinator.list_projects()
    return [ProjectResponse(**_project_to_dict(item)) for item in projects]


@app.post("/projects", response_model=ProjectResponse)
async def create_or_update_project(payload: ProjectCreateRequest) -> ProjectResponse:
    runtime = _runtime()
    try:
        record = runtime.coordinator.ensure_project(
            name=payload.name,
            root_path=payload.root_path,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProjectResponse(**_project_to_dict(record))


@app.get("/profiles", response_model=list[ProfileResponse])
async def list_profiles() -> list[ProfileResponse]:
    runtime = _runtime()
    profiles = runtime.coordinator.get_profiles().values()
    return [ProfileResponse(**_profile_to_dict(item)) for item in profiles]


@app.post("/profiles", response_model=ProfileResponse)
async def create_profile(payload: ProfileCreateRequest) -> ProfileResponse:
    runtime = _runtime()
    try:
        profile = runtime.coordinator.create_profile(
            agent_key=payload.agent_key,
            display_name=payload.display_name,
            role=payload.role,
            system_prompt=payload.system_prompt,
            model_name=payload.model_name,
            is_enabled=payload.is_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProfileResponse(**_profile_to_dict(profile))


@app.patch("/profiles/{agent_key}", response_model=ProfileResponse)
async def update_profile(agent_key: str, payload: ProfileUpdateRequest) -> ProfileResponse:
    runtime = _runtime()
    profiles = runtime.coordinator.get_profiles()
    if agent_key not in profiles:
        raise HTTPException(status_code=404, detail=f"Unknown agent key: {agent_key}")

    runtime.coordinator.update_profile(
        agent_key=agent_key,
        system_prompt=payload.system_prompt,
        model_name=payload.model_name,
        is_enabled=payload.is_enabled,
    )

    updated = runtime.coordinator.get_profiles().get(agent_key)
    if updated is None:
        raise HTTPException(status_code=500, detail="Profile update failed.")
    return ProfileResponse(**_profile_to_dict(updated))


@app.post("/agent/run", response_model=AgentRunResponse)
async def run_agent(payload: AgentRunRequest) -> AgentRunResponse:
    runtime = _runtime()
    result = await runtime.coordinator.run_agent(
        project_id=payload.project_id,
        user_prompt=payload.user_prompt,
    )
    return AgentRunResponse(sections=result.sections, final_output=result.final_output)


def _auto_learn_from_chat(runtime: Any, user_msg: str, reply: str, project_id: int | None) -> None:
    """Extract and store a memory entry from a chat interaction."""
    import re as _re_al
    # Detect topic from user message
    msg_l = user_msg.lower()
    prefix_match = _re_al.match(r"^\[(THINK|PLAN|ACT)\]\s*", msg_l)
    mode = prefix_match.group(1) if prefix_match else None
    clean = user_msg[prefix_match.end():].strip() if prefix_match else user_msg.strip()

    if any(kw in msg_l for kw in ("casino", "ruleta", "poker", "tragamonedas", "slot")):
        topic = "casino"
    elif any(kw in msg_l for kw in ("tienda", "ecommerce", "venta")):
        topic = "ecommerce"
    elif any(kw in msg_l for kw in ("api", "backend", "endpoint")):
        topic = "backend"
    elif any(kw in msg_l for kw in ("python", "javascript", "código", "codigo", "programa")):
        topic = "programacion"
    elif any(kw in msg_l for kw in ("web", "html", "css", "frontend")):
        topic = "frontend"
    else:
        topic = "general"

    mem_type = f"{mode.lower()}_{topic}" if mode else f"chat_{topic}"
    summary = f"Q: {clean[:120]}\nA: {reply[:300]}"
    runtime.database.add_memory(
        project_id=project_id,
        memory_type=mem_type,
        content=summary,
        metadata={"source": "auto_learn", "mode": mode or "chat", "topic": topic},
        relevance=0.8,
    )


class MemoryLearnIn(BaseModel):
    content: str
    topic: str = "general"
    project_id: int | None = None


class MemoryStatsResponse(BaseModel):
    total_memories: int
    total_conversations: int
    unique_topics: int
    knowledge_level: float
    top_topics: list[dict[str, Any]]
    level_label: str


@app.post("/api/memory/learn")
async def learn_memory(payload: MemoryLearnIn):
    runtime = _runtime()
    runtime.database.add_memory(
        project_id=payload.project_id,
        memory_type=f"manual_{payload.topic}",
        content=payload.content.strip(),
        metadata={"source": "manual", "topic": payload.topic},
        relevance=1.0,
    )
    return {"ok": True, "message": "Aprendizaje almacenado correctamente."}


@app.get("/api/memory/stats", response_model=MemoryStatsResponse)
async def memory_stats(project_id: int | None = None):
    runtime = _runtime()
    # Fetch all memories to compute stats
    memories = runtime.database.recent_memories(project_id=project_id, limit=500)
    conversations = runtime.database.recent_conversations(project_id=project_id, limit=500)

    total_memories = len(memories)
    total_conversations = len(conversations)

    topic_counts: dict[str, int] = {}
    for m in memories:
        mtype = m.memory_type or "general"
        base = mtype.split("_")[-1] if "_" in mtype else mtype
        topic_counts[base] = topic_counts.get(base, 0) + 1

    unique_topics = len(topic_counts)
    top_topics = sorted(
        [{"topic": k, "count": v} for k, v in topic_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:5]

    # Knowledge level: logarithmic scale capped at 100
    import math as _math
    raw = (total_memories * 3 + total_conversations * 1.5 + unique_topics * 5)
    knowledge_level = min(100.0, round(_math.log1p(raw) / _math.log1p(500) * 100, 1))

    if knowledge_level < 20:
        label = "Aprendiz"
    elif knowledge_level < 40:
        label = "Explorador"
    elif knowledge_level < 60:
        label = "Especialista"
    elif knowledge_level < 80:
        label = "Experto"
    else:
        label = "Maestro"

    return MemoryStatsResponse(
        total_memories=total_memories,
        total_conversations=total_conversations,
        unique_topics=unique_topics,
        knowledge_level=knowledge_level,
        top_topics=top_topics,
        level_label=label,
    )


@app.post("/assistant/chat", response_model=AssistantChatResponse)
async def assistant_chat(payload: AssistantChatRequest) -> AssistantChatResponse:
    runtime = _runtime()
    result = await runtime.coordinator.assistant_turn(
        project_id=payload.project_id,
        user_prompt=payload.message,
    )
    # Auto-learn: store a condensed memory entry from each interaction
    try:
        _auto_learn_from_chat(runtime, payload.message, result.reply, payload.project_id)
    except Exception:
        pass
    return AssistantChatResponse(
        reply=result.reply,
        source=result.source,
        project_id=result.project_id,
        sections=result.sections,
        machine_translation=result.machine_translation,
    )


@app.post("/agents/team/run", response_model=TeamRunResponse)
async def run_agent_team(payload: TeamRunRequest) -> TeamRunResponse:
    runtime = _runtime()
    try:
        result = await runtime.coordinator.run_custom_team(
            project_id=payload.project_id,
            user_prompt=payload.user_prompt,
            agent_keys=payload.agent_keys,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TeamRunResponse(
        project_id=result.project_id,
        user_prompt=result.user_prompt,
        final_output=result.final_output,
        steps=[
            TeamRunStepResponse(
                agent_key=item.agent_key,
                display_name=item.display_name,
                role=item.role,
                model_name=item.model_name,
                status=item.status,
                output=item.output,
            )
            for item in result.steps
        ],
    )


@app.post("/commands/run", response_model=CommandRunResponse)
async def run_command(payload: CommandRunRequest) -> CommandRunResponse:
    runtime = _runtime()
    code, output = runtime.coordinator.execute_project_command(
        project_id=payload.project_id,
        command_text=payload.command_text,
        timeout_seconds=payload.timeout_seconds,
    )
    return CommandRunResponse(return_code=code, output=output)


@app.get("/orchestrator/workflows", response_model=list[OrchestratorWorkflowResponse])
async def orchestrator_workflows() -> list[OrchestratorWorkflowResponse]:
    runtime = _runtime()
    workflows = runtime.orchestrator.list_workflows()
    return [
        OrchestratorWorkflowResponse(
            workflow_id=item.workflow_id,
            name=item.name,
            description=item.description,
            steps=item.steps,
        )
        for item in workflows
    ]


@app.post("/orchestrator/run", response_model=OrchestratorRunResponse)
async def orchestrator_run(payload: OrchestratorRunRequest) -> OrchestratorRunResponse:
    runtime = _runtime()
    try:
        result = await runtime.orchestrator.run_workflow(
            workflow_id=payload.workflow_id,
            project_id=payload.project_id,
            user_prompt=payload.user_prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return OrchestratorRunResponse(
        workflow_id=result.workflow_id,
        status=result.status,
        summary=result.summary,
        steps=[
            OrchestratorStepResponse(
                step=step.step,
                status=step.status,
                detail=step.detail,
                duration_ms=step.duration_ms,
            )
            for step in result.steps
        ],
        output=result.output,
    )


@app.get("/memories", response_model=list[MemoryResponse])
async def recent_memories(project_id: int | None = None, limit: int = 20) -> list[MemoryResponse]:
    runtime = _runtime()
    clamped_limit = max(1, min(limit, 200))
    memories = runtime.database.recent_memories(project_id=project_id, limit=clamped_limit)
    return [MemoryResponse(**_memory_to_dict(item)) for item in memories]


@app.get("/conversations", response_model=list[ConversationResponse])
async def recent_conversations(
    project_id: int | None = None,
    limit: int = 20,
) -> list[ConversationResponse]:
    runtime = _runtime()
    clamped_limit = max(1, min(limit, 200))
    conversations = runtime.database.recent_conversations(
        project_id=project_id,
        limit=clamped_limit,
    )
    return [ConversationResponse(**_conversation_to_dict(item)) for item in conversations]


@app.get("/models/available", response_model=AvailableModelsResponse)
async def available_models() -> AvailableModelsResponse:
    runtime = _runtime()
    return AvailableModelsResponse(models=_build_available_models(runtime.database))


class AgentOut(BaseModel):
    id: int
    key: str
    display_name: str
    role: str
    system_prompt: str
    model_name: str
    is_enabled: bool
    created_at: str
    updated_at: str


class AgentCreateIn(BaseModel):
    key: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str = Field(min_length=2, max_length=120)
    role: str = Field(min_length=2, max_length=120)
    system_prompt: str = Field(default="", max_length=12000)
    model_name: str = Field(default="", max_length=200)
    is_enabled: bool = True


class AgentUpdateIn(BaseModel):
    display_name: str | None = None
    role: str | None = None
    system_prompt: str | None = None
    model_name: str | None = None
    is_enabled: bool | None = None


class FolderOut(BaseModel):
    id: int
    name: str
    description: str
    parent_id: int | None
    created_at: str
    children: list["FolderOut"] = []
    assignments: list["AssignmentOut"] = []


class FolderCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    parent_id: int | None = None


class AssignmentOut(BaseModel):
    id: int
    folder_id: int
    agent_id: int
    agent_key: str
    agent_display_name: str
    process_type: str
    created_at: str


class AssignmentCreateIn(BaseModel):
    agent_id: int
    process_type: str = Field(pattern=r"^(planning|thinking|action)$")


FolderOut.model_rebuild()


def _agent_to_out(a: Agent) -> AgentOut:
    return AgentOut(
        id=a.id, key=a.key, display_name=a.display_name, role=a.role,
        system_prompt=a.system_prompt, model_name=a.model_name,
        is_enabled=a.is_enabled, created_at=str(a.created_at), updated_at=str(a.updated_at),
    )


def _folder_to_out(f: Folder) -> FolderOut:
    return FolderOut(
        id=f.id, name=f.name, description=f.description, parent_id=f.parent_id,
        created_at=str(f.created_at),
        children=[_folder_to_out(c) for c in (f.children or [])],
        assignments=[_assignment_to_out(a) for a in (f.assignments or [])],
    )


def _assignment_to_out(a: AgentAssignment) -> AssignmentOut:
    return AssignmentOut(
        id=a.id, folder_id=a.folder_id, agent_id=a.agent_id,
        agent_key=a.agent.key if a.agent else "",
        agent_display_name=a.agent.display_name if a.agent else "",
        process_type=a.process_type.value if isinstance(a.process_type, ProcessType) else a.process_type,
        created_at=str(a.created_at),
    )


@app.get("/api/agents", response_model=list[AgentOut])
async def list_agents():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Agent).order_by(Agent.display_name))
        agents = result.scalars().all()
        return [_agent_to_out(a) for a in agents]


@app.post("/api/agents", response_model=AgentOut)
async def create_agent(payload: AgentCreateIn):
    async with AsyncSessionLocal() as session:
        agent = Agent(
            key=payload.key, display_name=payload.display_name, role=payload.role,
            system_prompt=payload.system_prompt, model_name=payload.model_name,
            is_enabled=payload.is_enabled,
        )
        session.add(agent)
        await session.commit()
        await session.refresh(agent)
        return _agent_to_out(agent)


@app.patch("/api/agents/{agent_id}", response_model=AgentOut)
async def update_agent(agent_id: int, payload: AgentUpdateIn):
    async with AsyncSessionLocal() as session:
        agent = await session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        if payload.display_name is not None:
            agent.display_name = payload.display_name
        if payload.role is not None:
            agent.role = payload.role
        if payload.system_prompt is not None:
            agent.system_prompt = payload.system_prompt
        if payload.model_name is not None:
            agent.model_name = payload.model_name
        if payload.is_enabled is not None:
            agent.is_enabled = payload.is_enabled
        await session.commit()
        await session.refresh(agent)
        return _agent_to_out(agent)


@app.get("/api/folders", response_model=list[FolderOut])
async def list_folders():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Folder)
            .where(Folder.parent_id.is_(None))
            .options(
                selectinload(Folder.children).selectinload(Folder.assignments).selectinload(AgentAssignment.agent),
                selectinload(Folder.assignments).selectinload(AgentAssignment.agent),
            )
            .order_by(Folder.name)
        )
        folders = result.scalars().unique().all()
        return [_folder_to_out(f) for f in folders]


@app.post("/api/folders", response_model=FolderOut)
async def create_folder(payload: FolderCreateIn):
    async with AsyncSessionLocal() as session:
        if payload.parent_id is not None:
            parent = await session.get(Folder, payload.parent_id)
            if not parent:
                raise HTTPException(status_code=404, detail="Parent folder not found")
        folder = Folder(name=payload.name, description=payload.description, parent_id=payload.parent_id)
        session.add(folder)
        await session.commit()
        await session.refresh(folder)
        return FolderOut(
            id=folder.id, name=folder.name, description=folder.description,
            parent_id=folder.parent_id, created_at=str(folder.created_at),
        )


@app.delete("/api/folders/{folder_id}")
async def delete_folder(folder_id: int):
    async with AsyncSessionLocal() as session:
        folder = await session.get(Folder, folder_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        await session.delete(folder)
        await session.commit()
        return {"ok": True}


@app.get("/api/folders/{folder_id}/assignments", response_model=list[AssignmentOut])
async def list_folder_assignments(folder_id: int):
    async with AsyncSessionLocal() as session:
        folder = await session.get(Folder, folder_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        result = await session.execute(
            select(AgentAssignment)
            .where(AgentAssignment.folder_id == folder_id)
            .options(selectinload(AgentAssignment.agent))
        )
        assignments = result.scalars().all()
        return [_assignment_to_out(a) for a in assignments]


@app.post("/api/folders/{folder_id}/assignments", response_model=AssignmentOut)
async def create_assignment(folder_id: int, payload: AssignmentCreateIn):
    async with AsyncSessionLocal() as session:
        folder = await session.get(Folder, folder_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        agent = await session.get(Agent, payload.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        assignment = AgentAssignment(
            folder_id=folder_id, agent_id=payload.agent_id,
            process_type=ProcessType(payload.process_type),
        )
        session.add(assignment)
        await session.commit()
        await session.refresh(assignment)
        assignment.agent = agent
        return _assignment_to_out(assignment)


@app.delete("/api/assignments/{assignment_id}")
async def delete_assignment(assignment_id: int):
    async with AsyncSessionLocal() as session:
        assignment = await session.get(AgentAssignment, assignment_id)
        if not assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")
        await session.delete(assignment)
        await session.commit()
        return {"ok": True}


class SubAgentPipelineRequest(BaseModel):
    user_prompt: str = Field(min_length=1, max_length=12000)
    project_id: int | None = None
    pipeline_type: str | None = Field(default=None, pattern=r"^(full_analysis|code_task|research|quick_answer|execute)?$")


@app.post("/api/subagents/run")
async def run_subagent_pipeline(payload: SubAgentPipelineRequest):
    runtime = _runtime()
    result = await runtime.coordinator.run_subagent_pipeline(
        project_id=payload.project_id,
        user_prompt=payload.user_prompt,
        pipeline_type=payload.pipeline_type,
    )
    return result


@app.get("/api/subagents/configs")
async def get_subagent_configs():
    runtime = _runtime()
    return runtime.coordinator.subagent_orchestrator.get_subagent_configs()


@app.get("/api/subagents/pipelines")
async def get_pipeline_templates():
    from src.subagents import PIPELINE_TEMPLATES
    return {
        name: [role.value for role in roles]
        for name, roles in PIPELINE_TEMPLATES.items()
    }


@app.get("/api/ethics/principles")
async def get_ethics_principles():
    runtime = _runtime()
    return {
        "principles": runtime.coordinator.ethics.principles,
        "audit_summary": runtime.coordinator.ethics.get_audit_summary(),
    }


class EthicsCheckRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    check_type: str = Field(default="input", pattern=r"^(input|output)$")


@app.post("/api/ethics/check")
async def check_ethics(payload: EthicsCheckRequest):
    runtime = _runtime()
    if payload.check_type == "input":
        result = runtime.coordinator.ethics.check_input(payload.text)
    else:
        result = runtime.coordinator.ethics.check_output(payload.text)
    return {
        "is_safe": result.is_safe,
        "violations": result.violations,
        "warnings": result.warnings,
        "applied_rules": result.applied_rules,
    }


class ContextAcquireRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    project_id: int | None = None


@app.post("/api/context/acquire")
async def acquire_context(payload: ContextAcquireRequest):
    runtime = _runtime()
    project = runtime.database.get_project(payload.project_id) if payload.project_id else None
    result = await runtime.coordinator.context_engine.acquire_context(
        query=payload.query,
        project_path=project.root_path if project else None,
    )
    return {
        "query": result.query,
        "confidence": result.confidence,
        "needs_human_input": result.needs_human_input,
        "human_question": result.human_question,
        "summary": result.summary,
        "sources": [
            {
                "source_type": s.source_type,
                "label": s.label,
                "content": s.content[:500],
                "relevance": s.relevance,
            }
            for s in result.sources
        ],
    }


@app.post("/api/query-db")
async def api_query_db(payload: dict):
    from src.tools import run_query_db
    sql = payload.get("sql", "")
    if not sql:
        raise HTTPException(status_code=400, detail="SQL query is required")
    result = run_query_db(sql)
    return {"result": result}


@app.post("/api/flash/run")
async def api_flash_run(payload: dict):
    """Run the 3-agent Flash pipeline with Session Manifest + Entropy Filter."""
    from src.flash_orchestrator import FlashOrchestrator
    prompt = payload.get("prompt", "").strip()
    project_id = payload.get("project_id")
    multimodal_inputs = payload.get("multimodal_inputs", [])
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    fo = FlashOrchestrator()
    result = await fo.run(prompt, project_id, multimodal_inputs or None)
    return {
        "session_id": result.session_id,
        "final_output": result.final_output,
        "logic_output": result.logic_output,
        "context_output": result.context_output,
        "synthesis_output": result.synthesis_output,
        "faithfulness": result.faithfulness,
        "manifest": result.manifest,
        "total_latency_ms": result.total_latency_ms,
        "agent_latencies": result.agent_latencies,
        "pipeline_version": result.pipeline_version,
    }


@app.post("/api/orchestrator/translate")
async def api_orchestrator_translate(payload: dict):
    prompt = payload.get("prompt", "").strip()
    project_id = payload.get("project_id")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    po = _runtime().coordinator.prompt_orchestrator
    translated = await po.translate(prompt, project_id)
    machine_ir = po.to_machine_ir(translated, project_id)
    return {
        "translated": translated.model_dump(),
        "machine_ir": machine_ir,
    }


# ── Serve built frontend in production ────────────────────────────────────────
# When ui/dist exists (post-build), mount static assets and catch-all for SPA.
_ui_dist = Path("ui/dist")
if _ui_dist.is_dir():
    _assets = _ui_dist / "assets"
    if _assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="ui-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _serve_frontend(full_path: str) -> FileResponse:
        index = _ui_dist / "index.html"
        return FileResponse(str(index))

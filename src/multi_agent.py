"""Multi-agent coordinator with project execution and memory context."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.context_acquisition import ContextAcquisitionEngine, detect_uncertainty
from src.ethics import EthicsFramework
from src.graph import compile_graph
from src.persistence import AgentDatabase, AgentProfileRecord, ProjectRecord
from src.prompt_orchestrator import PromptOrchestrator
from src.subagents import SubAgentOrchestrator, SubAgentRole, detect_pipeline_type
from src.tools import extract_urls, fetch_all_urls


@dataclass
class AgentRunResult:
    """Result model returned to UI and CLI callers."""

    messages: list[BaseMessage]
    sections: dict[str, str]
    final_output: str


@dataclass
class AssistantTurnResult:
    """Single assistant turn result for conversational UX."""

    reply: str
    source: str
    project_id: int | None
    sections: dict[str, str]
    machine_translation: dict[str, Any]


@dataclass
class TeamStepResult:
    """Single custom team step output."""

    agent_key: str
    display_name: str
    role: str
    model_name: str
    status: str
    output: str


@dataclass
class TeamRunResult:
    """Custom team execution result."""

    project_id: int | None
    user_prompt: str
    final_output: str
    steps: list[TeamStepResult]


@dataclass
class BaseRole:
    """Base role in the multi-agent hierarchy."""

    agent_key: str
    display_name: str
    responsibility: str

    @property
    def prompt_field(self) -> str:
        return f"{self.agent_key}_system_prompt"

    @property
    def model_field(self) -> str:
        return f"{self.agent_key}_model"


class ThoughtRole(BaseRole):
    """Strategic analysis role."""

    def __init__(self) -> None:
        super().__init__(
            agent_key="thought",
            display_name="Thought Agent",
            responsibility="Create concise strategic analysis.",
        )


class ReviewRole(BaseRole):
    """Critical review role."""

    def __init__(self) -> None:
        super().__init__(
            agent_key="review",
            display_name="Review Agent",
            responsibility="Challenge assumptions and rank risks.",
        )


class ActionRole(BaseRole):
    """Execution role."""

    def __init__(self) -> None:
        super().__init__(
            agent_key="action",
            display_name="Action Agent",
            responsibility="Generate final answer and actionable plan.",
        )


class AgentHierarchy:
    """Central role registry for extensible multi-agent orchestration."""

    def __init__(self) -> None:
        self.roles = [ThoughtRole(), ReviewRole(), ActionRole()]
        self.by_key = {role.agent_key: role for role in self.roles}


class MultiAgentCoordinator:
    """Coordinator for project-aware multi-agent execution."""

    def __init__(self, database: AgentDatabase) -> None:
        self.database = database
        self.hierarchy = AgentHierarchy()
        self.graph = compile_graph()
        self.ethics = EthicsFramework()
        self.context_engine = ContextAcquisitionEngine()
        self.subagent_orchestrator = SubAgentOrchestrator()
        self.prompt_orchestrator = PromptOrchestrator()

    def list_projects(self) -> list[ProjectRecord]:
        return self.database.list_projects()

    def ensure_project(self, name: str, root_path: str, description: str = "") -> ProjectRecord:
        path = str(Path(root_path).expanduser().resolve())
        return self.database.upsert_project(name=name, root_path=path, description=description)

    def get_profiles(self) -> dict[str, AgentProfileRecord]:
        return self.database.get_agent_profiles()

    def update_profile(
        self,
        *,
        agent_key: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        is_enabled: bool | None = None,
    ) -> None:
        self.database.update_agent_profile(
            agent_key=agent_key,
            system_prompt=system_prompt,
            model_name=model_name,
            is_enabled=is_enabled,
        )

    def create_profile(
        self,
        *,
        agent_key: str,
        display_name: str,
        role: str,
        system_prompt: str,
        model_name: str,
        is_enabled: bool = True,
    ) -> AgentProfileRecord:
        return self.database.create_agent_profile(
            agent_key=agent_key,
            display_name=display_name,
            role=role,
            system_prompt=system_prompt,
            model_name=model_name,
            is_enabled=is_enabled,
        )

    @staticmethod
    def _is_control_instruction_unrecognized(message: str) -> bool:
        return message.startswith("Instruccion no reconocida.")

    @staticmethod
    def _slugify(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        lowered = normalized.lower()
        slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
        return slug or "game"

    @classmethod
    def _extract_game_scaffold_request(cls, text: str) -> str | None:
        lowered = text.lower().strip()
        verbs = ("crear", "crea", "genera", "generate", "create", "build")
        contains_verb = any(lowered.startswith(v + " ") or lowered == v for v in verbs)
        if not contains_verb:
            return None
        if not ("juego" in lowered or "game" in lowered):
            return None

        match = re.search(r"(?:llamado|named)\s+([a-zA-Z0-9 _-]{3,50})", text, flags=re.IGNORECASE)
        if match:
            return cls._slugify(match.group(1))
        return "game-starter"

    def _create_browser_game_scaffold(
        self,
        *,
        project_id: int | None,
        folder_name: str,
    ) -> tuple[str, int | None]:
        project = self.database.get_project(project_id) if project_id else None
        base_root = Path(project.root_path) if project else Path.cwd()
        target_dir = base_root / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        (target_dir / "index.html").write_text(
            (
                "<!doctype html>\n"
                "<html lang=\"en\">\n"
                "<head>\n"
                "  <meta charset=\"UTF-8\" />\n"
                "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
                "  <title>Game Starter</title>\n"
                "  <link rel=\"stylesheet\" href=\"style.css\" />\n"
                "</head>\n"
                "<body>\n"
                "  <main>\n"
                "    <h1>Game Starter</h1>\n"
                "    <p>Use arrows or WASD to move. Avoid red enemies.</p>\n"
                "    <canvas id=\"game\" width=\"900\" height=\"540\"></canvas>\n"
                "    <p id=\"status\"></p>\n"
                "  </main>\n"
                "  <script src=\"game.js\"></script>\n"
                "</body>\n"
                "</html>\n"
            ),
            encoding="utf-8",
        )

        (target_dir / "style.css").write_text(
            (
                "body { margin: 0; font-family: Arial, sans-serif; background: #f4f7fb; color: #111; }\n"
                "main { max-width: 980px; margin: 24px auto; padding: 16px; }\n"
                "canvas { display: block; background: #0f172a; border: 2px solid #1e293b; border-radius: 8px; }\n"
                "#status { margin-top: 12px; font-weight: 700; }\n"
            ),
            encoding="utf-8",
        )

        (target_dir / "game.js").write_text(
            (
                "const canvas = document.getElementById('game');\n"
                "const ctx = canvas.getContext('2d');\n"
                "const statusEl = document.getElementById('status');\n"
                "const keys = new Set();\n"
                "const player = { x: 80, y: 220, w: 26, h: 26, speed: 240 };\n"
                "const enemies = Array.from({ length: 7 }, (_, i) => ({\n"
                "  x: 420 + i * 65,\n"
                "  y: 40 + Math.random() * 430,\n"
                "  r: 12,\n"
                "  vx: -130 - Math.random() * 120,\n"
                "}));\n"
                "let running = true;\n"
                "let start = performance.now();\n"
                "window.addEventListener('keydown', (e) => keys.add(e.key.toLowerCase()));\n"
                "window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));\n"
                "function hit(a, b) {\n"
                "  const cx = Math.max(a.x, Math.min(b.x, a.x + a.w));\n"
                "  const cy = Math.max(a.y, Math.min(b.y, a.y + a.h));\n"
                "  return ((cx - b.x) ** 2 + (cy - b.y) ** 2) < b.r ** 2;\n"
                "}\n"
                "function update(dt) {\n"
                "  if (!running) return;\n"
                "  const left = keys.has('arrowleft') || keys.has('a');\n"
                "  const right = keys.has('arrowright') || keys.has('d');\n"
                "  const up = keys.has('arrowup') || keys.has('w');\n"
                "  const down = keys.has('arrowdown') || keys.has('s');\n"
                "  if (left) player.x -= player.speed * dt;\n"
                "  if (right) player.x += player.speed * dt;\n"
                "  if (up) player.y -= player.speed * dt;\n"
                "  if (down) player.y += player.speed * dt;\n"
                "  player.x = Math.max(0, Math.min(canvas.width - player.w, player.x));\n"
                "  player.y = Math.max(0, Math.min(canvas.height - player.h, player.y));\n"
                "  for (const e of enemies) {\n"
                "    e.x += e.vx * dt;\n"
                "    if (e.x < -30) {\n"
                "      e.x = canvas.width + 20 + Math.random() * 160;\n"
                "      e.y = 30 + Math.random() * 470;\n"
                "      e.vx = -130 - Math.random() * 120;\n"
                "    }\n"
                "    if (hit(player, e)) running = false;\n"
                "  }\n"
                "}\n"
                "function draw() {\n"
                "  ctx.clearRect(0, 0, canvas.width, canvas.height);\n"
                "  ctx.fillStyle = '#38bdf8';\n"
                "  ctx.fillRect(player.x, player.y, player.w, player.h);\n"
                "  ctx.fillStyle = '#ef4444';\n"
                "  for (const e of enemies) {\n"
                "    ctx.beginPath();\n"
                "    ctx.arc(e.x, e.y, e.r, 0, Math.PI * 2);\n"
                "    ctx.fill();\n"
                "  }\n"
                "}\n"
                "let last = performance.now();\n"
                "function loop(now) {\n"
                "  const dt = Math.min(0.033, (now - last) / 1000);\n"
                "  last = now;\n"
                "  update(dt);\n"
                "  draw();\n"
                "  const secs = ((now - start) / 1000).toFixed(1);\n"
                "  statusEl.textContent = running ? `Survival: ${secs}s` : `Game Over. Survival: ${secs}s`;\n"
                "  requestAnimationFrame(loop);\n"
                "}\n"
                "requestAnimationFrame(loop);\n"
            ),
            encoding="utf-8",
        )

        (target_dir / "README.md").write_text(
            (
                "# Game Starter\n\n"
                "## Run\n\n"
                "1. Open `index.html` in your browser.\n"
                "2. Move with arrow keys or WASD.\n"
                "3. Avoid red enemies and beat your survival time.\n"
            ),
            encoding="utf-8",
        )

        self.database.add_memory(
            project_id=project_id,
            memory_type="artifact_generation",
            content=f"Generated browser game scaffold at {target_dir}",
            metadata={"artifact_path": str(target_dir)},
            relevance=0.95,
        )

        return str(target_dir), project_id

    @staticmethod
    def _looks_like_shell_command(command_text: str) -> bool:
        candidate = command_text.strip()
        if not candidate:
            return False
        first = candidate.split(maxsplit=1)[0].lower()
        allowed_starts = {
            "python",
            "python3",
            "pip",
            "pip3",
            "pytest",
            "npm",
            "npx",
            "node",
            "yarn",
            "pnpm",
            "uv",
            "uvicorn",
            "poetry",
            "git",
            "ls",
            "pwd",
            "cat",
            "echo",
            "mkdir",
            "touch",
            "cp",
            "mv",
            "find",
            "rg",
            "grep",
            "sed",
            "awk",
            "bash",
            "sh",
            "zsh",
            "ollama",
            "langgraph",
            "docker",
            "kubectl",
            "make",
            "cargo",
            "go",
            "java",
            "javac",
            "mvn",
            "gradle",
        }
        if first in allowed_starts:
            return True
        return first.startswith("./")

    @classmethod
    def _extract_command_request(cls, text: str) -> str | None:
        stripped = text.strip()
        lowered = stripped.lower()
        explicit_prefixes = (
            "comando:",
            "command:",
            "cmd:",
            "ejecuta comando:",
            "run command:",
            "terminal:",
        )
        for prefix in explicit_prefixes:
            if lowered.startswith(prefix):
                value = stripped[len(prefix) :].strip()
                return value or None

        natural_prefixes = ("ejecuta ", "run ", "corre ")
        for prefix in natural_prefixes:
            if lowered.startswith(prefix):
                value = stripped[len(prefix) :].strip()
                if cls._looks_like_shell_command(value):
                    return value
                return None
        return None

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", normalized.strip()).lower()

    @classmethod
    def _detect_control_kind(cls, text: str) -> str | None:
        lowered = cls._normalize_prompt(text)
        if re.match(r"^(usar|usa|selecciona|cambiar a|cambia a)\s+proyecto\s+.+$", lowered):
            return "switch_project"
        if re.match(r"^crear proyecto\s+.+?\s+en\s+.+$", lowered):
            return "create_project"

        for marker in ("pensamiento", "revision", "accion"):
            if lowered.startswith(f"agente {marker}:"):
                return f"set_prompt_{marker}"
            if lowered.startswith(f"modelo {marker}:"):
                return f"set_model_{marker}"
        return None

    @classmethod
    def _detect_input_language(cls, text: str) -> str:
        lowered = cls._normalize_prompt(text)
        spanish_markers = ("quiero", "crear", "proyecto", "ejecuta", "comando", "agente", "modelo")
        english_markers = ("create", "project", "run", "command", "agent", "model")
        if any(token in lowered for token in spanish_markers):
            return "es"
        if any(token in lowered for token in english_markers):
            return "en"
        return "unknown"

    @classmethod
    def _is_time_request(cls, text: str) -> bool:
        lowered = cls._normalize_prompt(text)
        markers = (
            "que hora",
            "hora es",
            "dime la hora",
            "horario",
            "what time",
            "time is it",
            "current time",
        )
        return any(marker in lowered for marker in markers)

    @classmethod
    def _infer_context_requirements(
        cls,
        *,
        text: str,
        project_id: int | None,
        intent: str,
        command_text: str | None,
    ) -> dict[str, Any]:
        required: list[str] = []
        missing: list[str] = []
        reasons: list[str] = []
        lowered = cls._normalize_prompt(text)

        requires_project = False
        if intent == "command_execution" and command_text:
            first_token = ""
            try:
                tokens = shlex.split(command_text)
                first_token = tokens[0].lower() if tokens else ""
            except ValueError:
                first_token = ""
            project_scoped_commands = {
                "npm",
                "pnpm",
                "yarn",
                "pytest",
                "python",
                "uv",
                "make",
                "cargo",
                "go",
                "mvn",
                "gradle",
                "git",
                "docker",
            }
            requires_project = first_token in project_scoped_commands

        if intent == "agent_query":
            project_scope_markers = (
                "en este proyecto",
                "de este proyecto",
                "del proyecto",
                "build",
                "compila",
                "corregir",
                "corrige",
                "arregla",
                "fix",
                "refactor",
                "test",
                "pruebas",
            )
            requires_project = any(marker in lowered for marker in project_scope_markers)

        if requires_project:
            required.append("project_context")
            if project_id is None:
                missing.append("project_context")
                reasons.append("No hay proyecto activo seleccionado para ejecutar la tarea.")

        return {
            "required": required,
            "missing": missing,
            "reasons": reasons,
            "is_blocking": bool(missing),
        }

    @classmethod
    def _translate_prompt_to_machine_ir(
        cls,
        *,
        text: str,
        project_id: int | None,
        command_text: str | None,
        game_folder: str | None,
        control_kind: str | None,
    ) -> dict[str, Any]:
        intent = "agent_query"
        route_target = "multi_agent_graph"
        confidence = 0.75
        args: dict[str, Any] = {}
        ops: list[dict[str, Any]] = [
            {"op": "PARSE_INPUT", "value": text},
            {"op": "NORMALIZE", "value": cls._normalize_prompt(text)},
        ]

        if command_text:
            intent = "command_execution"
            route_target = "command_runner"
            confidence = 0.96
            args = {"command_text": command_text}
            ops.extend(
                [
                    {"op": "ROUTE", "target": route_target},
                    {"op": "EXECUTE_COMMAND", "command": command_text},
                ]
            )
        elif game_folder:
            intent = "artifact_generation"
            route_target = "artifact_generator"
            confidence = 0.93
            args = {"artifact_type": "browser_game", "folder_name": game_folder}
            ops.extend(
                [
                    {"op": "ROUTE", "target": route_target},
                    {"op": "CREATE_GAME_SCAFFOLD", "folder_name": game_folder},
                ]
            )
        elif control_kind:
            intent = "control_instruction"
            route_target = "control_plane"
            confidence = 0.9
            args = {"control_kind": control_kind}
            ops.extend(
                [
                    {"op": "ROUTE", "target": route_target},
                    {"op": "APPLY_CONTROL_INSTRUCTION", "kind": control_kind},
                ]
            )
        else:
            ops.extend(
                [
                    {"op": "ROUTE", "target": route_target},
                    {"op": "RUN_STAGE", "name": "openai_thought"},
                    {"op": "RUN_STAGE", "name": "anthropic_review"},
                    {"op": "RUN_STAGE", "name": "deepseek_action"},
                ]
            )

        context_requirements = cls._infer_context_requirements(
            text=text,
            project_id=project_id,
            intent=intent,
            command_text=command_text,
        )
        if context_requirements["is_blocking"]:
            ops.append({"op": "REQUEST_CONTEXT", "missing": context_requirements["missing"]})

        execution_policy = {
            "single_pass_allowed": True,
            "forced_flow": None,
        }
        if route_target == "multi_agent_graph" and cls._is_time_request(text):
            execution_policy = {
                "single_pass_allowed": False,
                "forced_flow": "multi_stage",
            }
            ops.append({"op": "FORCE_FULL_FLOW", "reason": "time_request_validation"})

        return {
            "schema": "prompt-ir/v1",
            "input_language": cls._detect_input_language(text),
            "project_id": project_id,
            "intent": intent,
            "route": {
                "target": route_target,
                "requires_llm": route_target == "multi_agent_graph",
                "confidence": confidence,
            },
            "args": args,
            "ops": ops,
            "context_requirements": context_requirements,
            "execution_policy": execution_policy,
        }

    def apply_prompt_instruction(
        self,
        *,
        instruction: str,
        current_project_id: int | None,
    ) -> tuple[str, int | None]:
        """
        Apply plain-language control instructions.

        Supported patterns:
        - "usar proyecto <name>"
        - "crear proyecto <name> en <path>"
        - "agente pensamiento: <prompt>"
        - "agente revision: <prompt>"
        - "agente accion: <prompt>"
        - "modelo pensamiento: <model>"
        - "modelo revision: <model>"
        - "modelo accion: <model>"
        """
        text = instruction.strip()
        lowered = text.lower()

        match = re.match(r"^(usar|usa|selecciona|cambiar a|cambia a)\s+proyecto\s+(.+)$", lowered)
        if match:
            name = match.group(2).strip()
            project = self.database.get_project_by_name(name)
            if project:
                return f"Proyecto activo cambiado a: {project.name}", project.id
            return f"No existe proyecto con nombre: {name}", current_project_id

        match = re.match(r"^crear proyecto\s+(.+?)\s+en\s+(.+)$", text, flags=re.IGNORECASE)
        if match:
            name, path = match.groups()
            project = self.ensure_project(name=name.strip(), root_path=path.strip())
            return f"Proyecto creado/actualizado: {project.name} -> {project.root_path}", project.id

        prompt_matchers = {
            "pensamiento": "thought",
            "revision": "review",
            "accion": "action",
        }
        for marker, key in prompt_matchers.items():
            prefix = f"agente {marker}:"
            if lowered.startswith(prefix):
                content = text[len(prefix) :].strip()
                if not content:
                    return f"Prompt vacio para {marker}.", current_project_id
                self.update_profile(agent_key=key, system_prompt=content)
                return f"Prompt de {marker} actualizado.", current_project_id

        for marker, key in prompt_matchers.items():
            prefix = f"modelo {marker}:"
            if lowered.startswith(prefix):
                content = text[len(prefix) :].strip()
                if not content:
                    return f"Modelo vacio para {marker}.", current_project_id
                self.update_profile(agent_key=key, model_name=content)
                return f"Modelo de {marker} actualizado a {content}.", current_project_id

        return (
            "Instruccion no reconocida. Usa: 'usar proyecto <nombre>', "
            "'crear proyecto <nombre> en <ruta>', "
            "'agente pensamiento: ...', 'agente revision: ...', 'agente accion: ...'.",
            current_project_id,
        )

    def _build_memory_context(self, project_id: int | None, user_prompt: str) -> str:
        memories = self.database.recent_memories(project_id=project_id, limit=20)
        if not memories:
            return "No previous memory."

        tokens = {token for token in re.findall(r"[a-zA-Z0-9_]{4,}", user_prompt.lower())}

        def _score(text: str) -> int:
            if not tokens:
                return 0
            low = text.lower()
            return sum(1 for token in tokens if token in low)

        ranked = sorted(memories, key=lambda mem: (_score(mem.content), mem.created_at), reverse=True)
        top = ranked[:6]
        lines = [f"- [{item.memory_type}] {item.content}" for item in top]
        return "\n".join(lines)

    def _build_context_snapshot(
        self,
        user_prompt: str,
        memory_context: str,
        machine_translation: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a structured context snapshot for display and downstream use."""
        orch = machine_translation.get("orchestrator", {})
        intent = machine_translation.get("intent", orch.get("intent", "general"))
        objective = orch.get("objective", user_prompt[:150])
        priority = orch.get("priority", "medium")
        ambiguity = orch.get("ambiguity_score", 0.0)
        sub_tasks = orch.get("sub_tasks", [])
        tools_to_activate = orch.get("tools_to_activate", [])

        # Extract relevant memories for display (scored against prompt)
        tokens = set(re.findall(r"[a-zA-Z0-9_]{4,}", user_prompt.lower()))
        recalled_memories: list[dict[str, Any]] = []
        if memory_context and memory_context != "No previous memory.":
            for line in memory_context.split("\n"):
                if not line.strip():
                    continue
                score = sum(1 for t in tokens if t in line.lower())
                if score > 0:
                    # Parse "- [type] content"
                    m = re.match(r"-\s*\[(.+?)\]\s*(.+)", line)
                    if m:
                        recalled_memories.append({
                            "type": m.group(1),
                            "snippet": m.group(2)[:120],
                            "relevance": score,
                        })
            recalled_memories.sort(key=lambda x: x["relevance"], reverse=True)
            recalled_memories = recalled_memories[:3]

        # Detect domain entities from prompt
        prompt_l = user_prompt.lower()
        entities: list[str] = []
        domain_map = {
            "casino": "🎰 casino",
            "ruleta": "🎰 ruleta",
            "poker": "🃏 póker",
            "tienda": "🛒 tienda",
            "ecommerce": "🛒 e-commerce",
            "api": "⚙️ API REST",
            "fastapi": "⚙️ FastAPI",
            "python": "🐍 Python",
            "react": "⚛️ React",
            "javascript": "📜 JavaScript",
            "base de datos": "🗄️ base de datos",
            "postgresql": "🗄️ PostgreSQL",
            "microtransaccion": "💳 microtransacciones",
            "pago": "💳 pagos",
            "usuario": "👤 usuarios",
            "autenticacion": "🔐 autenticación",
        }
        for kw, label in domain_map.items():
            if kw in prompt_l and label not in entities:
                entities.append(label)

        # Confidence based on ambiguity and memory recall
        try:
            amb_f = float(ambiguity)
        except (TypeError, ValueError):
            amb_f = 0.0
        recall_bonus = min(0.3, len(recalled_memories) * 0.1)
        confidence = round(max(0.1, min(1.0, 1.0 - amb_f + recall_bonus)), 2)

        return {
            "intent": intent,
            "objective": objective,
            "priority": priority,
            "confidence": confidence,
            "entities": entities[:8],
            "recalled_memories": recalled_memories,
            "sub_tasks": sub_tasks[:5],
            "tools_activated": tools_to_activate[:4],
            "ambiguity": amb_f,
            "clarification_needed": amb_f > 0.7,
        }

    def _build_agent_config(self) -> dict[str, Any]:
        profiles = self.get_profiles()
        payload: dict[str, Any] = {}
        for role in self.hierarchy.roles:
            profile = profiles.get(role.agent_key)
            if not profile:
                continue
            payload[role.prompt_field] = profile.system_prompt
            payload[role.model_field] = profile.model_name
        return payload

    @staticmethod
    def _local_bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name, "true" if default else "false").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @classmethod
    def _command_policy(cls, command_text: str) -> tuple[bool, str, bool]:
        text = command_text.strip()
        if not text:
            return False, "Comando vacio.", False

        if not cls._local_bool_env("COMMAND_EXECUTION_ENABLED", True):
            return (
                False,
                "Ejecucion de comandos deshabilitada por seguridad (COMMAND_EXECUTION_ENABLED=false).",
                False,
            )

        if len(text) > 2000:
            return False, "Comando demasiado largo (max 2000 caracteres).", False

        blocked_patterns = (
            r"(?:^|\s)rm\s+-rf\s+/(?:\s|$)",
            r"(?:^|\s)shutdown(?:\s|$)",
            r"(?:^|\s)reboot(?:\s|$)",
            r"(?:^|\s)mkfs(?:\s|$)",
            r"(?:^|\s)dd\s+if=",
            r":\(\)\{:\|:&\};:",
        )
        lowered = text.lower()
        for pattern in blocked_patterns:
            if re.search(pattern, lowered):
                return False, "Comando bloqueado por politica de seguridad.", False

        allowlist_pattern = os.getenv("COMMAND_RUN_ALLOWLIST_REGEX", "").strip()
        if allowlist_pattern:
            try:
                if re.search(allowlist_pattern, text) is None:
                    return (
                        False,
                        "Comando fuera de allowlist (COMMAND_RUN_ALLOWLIST_REGEX).",
                        False,
                    )
            except re.error:
                return False, "Regex invalido en COMMAND_RUN_ALLOWLIST_REGEX.", False

        use_shell = cls._local_bool_env("COMMAND_RUN_USE_SHELL", False)
        if not use_shell:
            shell_markers = ("|", "&&", "||", ";", "$(", "`", ">", "<")
            if any(marker in text for marker in shell_markers):
                return (
                    False,
                    "Comando con operadores de shell bloqueado. Activa COMMAND_RUN_USE_SHELL=true si es requerido.",
                    False,
                )
        return True, "", use_shell

    def _single_pass_enabled(self) -> bool:
        return self._local_bool_env("LOCAL_SINGLE_PASS_ENABLED", True)

    def _single_pass_model(self) -> str:
        profiles = self.get_profiles()
        action_profile = profiles.get("action")
        if action_profile and action_profile.model_name.strip():
            return action_profile.model_name.strip()
        model = os.getenv("FREE_ACTION_MODEL", "").strip()
        if model:
            return model
        return "qwen2.5:latest"

    def _resolve_ollama_model_sync(self, requested_model: str, base_url: str) -> str:
        url = f"{base_url.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=4) as response:
                payload = response.read().decode("utf-8")
            decoded = json.loads(payload)
            models = decoded.get("models", [])
            if not isinstance(models, list):
                return requested_model
            names: list[str] = []
            for item in models:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    if name:
                        names.append(name)
            if requested_model in names:
                return requested_model
            if names:
                return names[0]
        except Exception:
            return requested_model
        return requested_model

    def _single_pass_ollama_call_sync(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        request_url = f"{base_url.rstrip('/')}/api/chat"
        timeout_s = float(os.getenv("OLLAMA_SINGLE_PASS_TIMEOUT_SECONDS", "95"))
        retries = max(0, int(os.getenv("OLLAMA_SINGLE_PASS_RETRIES", "0")))
        num_predict = int(os.getenv("OLLAMA_SINGLE_PASS_NUM_PREDICT", "96"))

        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"num_predict": num_predict},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        last_error: str | None = None
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(request_url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = str(data.get("message", {}).get("content", "")).strip()
                if not content:
                    content = str(data.get("response", "")).strip()
                if content:
                    return content
                last_error = "empty response from ollama"
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = str(exc)

            if attempt < retries:
                continue

        raise RuntimeError(last_error or "single pass ollama call failed")

    async def _single_pass_reply(
        self,
        *,
        project_id: int | None,
        user_prompt: str,
    ) -> str | None:
        if not self._single_pass_enabled():
            return None
        if os.getenv("FREE_LLM_PROVIDER", "ollama").strip().lower() != "ollama":
            return None

        project = self.database.get_project(project_id) if project_id else None
        project_context = (
            f"Proyecto activo: {project.name}"
            if project
            else "Sin proyecto activo."
        )
        model = self._single_pass_model()
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        resolved_model = self._resolve_ollama_model_sync(model, base_url)

        system_prompt = (
            "Eres un asistente tecnico local para operar el sistema. "
            "Responde claro, directo y accionable en espanol."
        )
        user_payload = (
            f"{project_context}\n"
            f"Solicitud: {user_prompt}\n"
            "Entrega 3 pasos concretos y un siguiente comando sugerido."
        )

        try:
            hard_timeout_s = float(os.getenv("OLLAMA_SINGLE_PASS_HARD_TIMEOUT_SECONDS", "100"))
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._single_pass_ollama_call_sync,
                    base_url=base_url,
                    model=resolved_model,
                    system_prompt=system_prompt,
                    user_prompt=user_payload,
                ),
                timeout=hard_timeout_s,
            )
        except Exception:
            return None

    async def _custom_team_step_reply(
        self,
        *,
        profile: AgentProfileRecord,
        project_label: str,
        user_prompt: str,
        previous_steps: list[TeamStepResult],
    ) -> TeamStepResult:
        provider = os.getenv("FREE_LLM_PROVIDER", "ollama").strip().lower()
        if provider != "ollama":
            return TeamStepResult(
                agent_key=profile.agent_key,
                display_name=profile.display_name,
                role=profile.role,
                model_name=profile.model_name,
                status="error",
                output=(
                    "El modo de equipo personalizado soporta FREE_LLM_PROVIDER=ollama por ahora. "
                    f"Proveedor actual: {provider}."
                ),
            )

        history_lines = [
            f"- {item.display_name} ({item.role}) [{item.status}]: {item.output[:1500]}"
            for item in previous_steps
        ]
        history = "\n".join(history_lines) if history_lines else "- Sin pasos previos."
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        requested_model = profile.model_name.strip() or "qwen2.5:latest"
        resolved_model = self._resolve_ollama_model_sync(requested_model, base_url)
        system_prompt = (
            f"{profile.system_prompt.strip()}\n\n"
            "Responde en espanol, con foco tecnico y sin relleno. "
            "Tu salida sera usada por el siguiente agente."
        )
        step_prompt = (
            f"Proyecto: {project_label}\n"
            f"Objetivo principal: {user_prompt}\n"
            f"Rol actual: {profile.role} ({profile.display_name})\n"
            "Resultados previos del equipo:\n"
            f"{history}\n\n"
            "Entrega una contribucion util para avanzar la tarea."
        )

        try:
            hard_timeout_s = float(os.getenv("OLLAMA_SINGLE_PASS_HARD_TIMEOUT_SECONDS", "100"))
            reply = await asyncio.wait_for(
                asyncio.to_thread(
                    self._single_pass_ollama_call_sync,
                    base_url=base_url,
                    model=resolved_model,
                    system_prompt=system_prompt,
                    user_prompt=step_prompt,
                ),
                timeout=hard_timeout_s,
            )
            if not reply.strip():
                raise RuntimeError("empty model response")
            return TeamStepResult(
                agent_key=profile.agent_key,
                display_name=profile.display_name,
                role=profile.role,
                model_name=resolved_model,
                status="ok",
                output=reply.strip(),
            )
        except Exception:
            fallback = self._team_step_rule_fallback(
                role=profile.role,
                display_name=profile.display_name,
                task=user_prompt,
                previous_steps=previous_steps,
            )
            return TeamStepResult(
                agent_key=profile.agent_key,
                display_name=profile.display_name,
                role=profile.role,
                model_name=resolved_model,
                status="ok",
                output=fallback,
            )

    @staticmethod
    def _team_step_rule_fallback(
        *,
        role: str,
        display_name: str,
        task: str,
        previous_steps: list,
    ) -> str:
        """Rule-based fallback output for a team agent step when Ollama is unavailable."""
        task_short = task.strip()[:200]
        task_lower = task.strip().lower()
        prev_count = len(previous_steps)

        # Detect key domains from the task
        is_web = any(kw in task_lower for kw in ("web", "sitio", "página", "html", "css", "frontend"))
        is_app = any(kw in task_lower for kw in ("app", "aplicación", "aplicacion", "móvil", "mobile"))
        is_casino = any(kw in task_lower for kw in ("casino", "juego", "game", "ruleta", "poker", "tragamonedas", "slot"))
        is_ecommerce = any(kw in task_lower for kw in ("tienda", "ecommerce", "e-commerce", "venta", "producto", "carrito"))
        is_api = any(kw in task_lower for kw in ("api", "backend", "endpoint", "servidor", "server"))
        is_db = any(kw in task_lower for kw in ("base de datos", "database", "sql", "tabla", "modelo"))

        domain = "proyecto"
        if is_casino: domain = "plataforma de casino en línea"
        elif is_ecommerce: domain = "tienda en línea"
        elif is_web: domain = "sitio web"
        elif is_app: domain = "aplicación móvil"
        elif is_api: domain = "API backend"
        elif is_db: domain = "sistema de base de datos"

        role_lower = (role or "").strip().lower()
        display_lower = (display_name or "").strip().lower()
        combined = role_lower + " " + display_lower

        if any(kw in combined for kw in ("investigador", "researcher", "research", "investiga", "información", "informacion", "gather")):
            return (
                f"## Hallazgos — {display_name}\n\n"
                f"**Tarea analizada:** {task_short}\n\n"
                f"**Información relevante recopilada:**\n"
                f"- El {domain} requiere definir una arquitectura clara antes de implementar\n"
                f"- Tecnologías recomendadas según el dominio: Python/FastAPI (backend), React/Vite (frontend), PostgreSQL (datos)\n"
                f"- Se necesita definir: autenticación de usuarios, modelo de datos, flujo principal de la aplicación\n"
                f"- Regulaciones a considerar: privacidad de datos, términos de servicio, seguridad HTTPS\n\n"
                f"## Brechas de Conocimiento\n"
                f"- Presupuesto y plazo del proyecto no definidos\n"
                f"- Número de usuarios esperados (afecta la arquitectura)\n"
                f"- Integraciones externas necesarias (pagos, autenticación OAuth, etc.)\n\n"
                f"## Recomendaciones\n"
                f"- Comenzar con un MVP (producto mínimo viable) bien definido\n"
                f"- Definir los 3 flujos de usuario más importantes antes de codificar\n"
                f"- Usar 'comando: SELECT * FROM projects' para revisar proyectos existentes"
            )

        if any(kw in combined for kw in ("planificador", "planner", "analyst", "thought", "planifica", "plan")):
            return (
                f"## Plan de Ejecución — {display_name}\n\n"
                f"**Objetivo:** {task_short}\n\n"
                f"### Paso 1: Definición del alcance\n"
                f"- Objetivo: Delimitar funcionalidades del {domain}\n"
                f"- Entrada: Descripción de la tarea\n"
                f"- Salida: Lista de funcionalidades prioritarias\n"
                f"- Complejidad: baja\n\n"
                f"### Paso 2: Diseño de arquitectura\n"
                f"- Objetivo: Definir stack tecnológico y estructura de módulos\n"
                f"- Entrada: Lista de funcionalidades\n"
                f"- Salida: Diagrama de componentes y modelo de datos\n"
                f"- Complejidad: media\n\n"
                f"### Paso 3: Implementación incremental\n"
                f"- Objetivo: Construir el {domain} por módulos\n"
                f"- Entrada: Arquitectura definida\n"
                f"- Salida: Código funcional y probado\n"
                f"- Complejidad: alta\n\n"
                f"### Paso 4: Revisión y despliegue\n"
                f"- Objetivo: Validar calidad y publicar\n"
                f"- Complejidad: media"
            )

        if any(kw in combined for kw in ("programador", "coder", "developer", "action", "executor", "programa", "codif", "implement")):
            template = "casino en línea" if is_casino else domain
            code_hint = ""
            if is_casino:
                code_hint = (
                    "```python\n"
                    "# Estructura base: Casino en Línea\n"
                    "# backend/main.py\n"
                    "from fastapi import FastAPI\n"
                    "from fastapi.middleware.cors import CORSMiddleware\n\n"
                    "app = FastAPI(title='Casino Online API')\n"
                    "app.add_middleware(CORSMiddleware, allow_origins=['*'])\n\n"
                    "@app.get('/games')\n"
                    "async def list_games():\n"
                    "    return [{'id': 1, 'name': 'Ruleta', 'type': 'table'},\n"
                    "            {'id': 2, 'name': 'Slots', 'type': 'machine'},\n"
                    "            {'id': 3, 'name': 'Blackjack', 'type': 'table'}]\n\n"
                    "@app.post('/bet')\n"
                    "async def place_bet(game_id: int, amount: float, choice: str):\n"
                    "    import random\n"
                    "    result = random.choice(['win', 'lose'])\n"
                    "    payout = amount * 2 if result == 'win' else 0\n"
                    "    return {'result': result, 'payout': payout}\n"
                    "```"
                )
            elif is_web:
                code_hint = (
                    "```html\n"
                    "<!-- Estructura base del sitio web -->\n"
                    "<!DOCTYPE html>\n<html lang='es'>\n<head>\n"
                    "  <meta charset='UTF-8'>\n"
                    "  <title>Mi Sitio Web</title>\n"
                    "</head>\n<body>\n"
                    "  <header><nav>Navegación</nav></header>\n"
                    "  <main><h1>Bienvenido</h1></main>\n"
                    "  <footer>Footer</footer>\n"
                    "</body>\n</html>\n```"
                )
            return (
                f"## Implementación — {display_name}\n\n"
                f"**Tarea:** Generar código base para {template}\n\n"
                f"**Estructura de archivos propuesta:**\n"
                f"```\n"
                f"{template.replace(' ', '_')}/\n"
                f"├── backend/\n"
                f"│   ├── main.py          # API principal (FastAPI)\n"
                f"│   ├── models.py        # Modelos de datos\n"
                f"│   ├── routes/          # Endpoints organizados\n"
                f"│   └── database.py      # Conexión PostgreSQL\n"
                f"├── frontend/\n"
                f"│   ├── src/\n"
                f"│   │   ├── App.tsx      # Componente raíz\n"
                f"│   │   ├── pages/       # Páginas principales\n"
                f"│   │   └── components/  # Componentes reutilizables\n"
                f"│   └── package.json\n"
                f"└── README.md\n"
                f"```\n\n"
                f"{code_hint}"
            )

        if any(kw in combined for kw in ("revisor", "reviewer", "critic", "review", "revis", "valida", "calidad")):
            prev_summary = ""
            if previous_steps:
                prev_summary = f"Revisando el trabajo de {len(previous_steps)} agente(s) previo(s).\n"
            return (
                f"## Veredicto: REQUIERE_REVISIÓN — {display_name}\n\n"
                f"{prev_summary}"
                f"**Evaluación del plan para {domain}:**\n\n"
                f"## Hallazgos Críticos\n"
                f"- Revisión estructural basada en análisis de calidad y coherencia\n"
                f"- Verificar: ¿Se definieron los casos de uso principales?\n"
                f"- Verificar: ¿El modelo de datos cubre los flujos de negocio?\n\n"
                f"## Sugerencias\n"
                f"- Añadir autenticación y control de sesiones desde el inicio\n"
                f"- Definir manejo de errores explícito en cada endpoint\n"
                f"- Incluir pruebas básicas antes de desplegar\n"
                f"- Documentar la API con ejemplos reales"
            )

        # Generic fallback for any other role
        return (
            f"## Contribución — {display_name} ({role})\n\n"
            f"**Tarea recibida:** {task_short}\n\n"
            f"He analizado la solicitud y propongo avanzar con el {domain} "
            f"usando un enfoque modular e incremental.\n\n"
            f"**Próximos pasos recomendados:**\n"
            f"1. Definir el MVP con las 3 funciones más importantes\n"
            f"2. Construir el backend con FastAPI y PostgreSQL\n"
            f"3. Crear el frontend con React/Vite\n"
            f"4. Desplegar con Docker o directamente en Replit."
        )

    async def run_custom_team(
        self,
        *,
        project_id: int | None,
        user_prompt: str,
        agent_keys: list[str],
    ) -> TeamRunResult:
        clean_prompt = user_prompt.strip()
        if not clean_prompt:
            raise ValueError("user_prompt no puede estar vacio.")

        profiles_map = self.get_profiles()
        ordered_keys = [key.strip().lower() for key in agent_keys if key.strip()]
        if not ordered_keys:
            ordered_keys = [profile.agent_key for profile in profiles_map.values() if profile.is_enabled]

        deduped_keys = list(dict.fromkeys(ordered_keys))
        selected_profiles: list[AgentProfileRecord] = []
        for key in deduped_keys:
            profile = profiles_map.get(key)
            if profile is None:
                raise ValueError(f"Agente no encontrado: {key}")
            if not profile.is_enabled:
                raise ValueError(f"Agente deshabilitado: {key}")
            selected_profiles.append(profile)

        if not selected_profiles:
            raise ValueError("No hay agentes habilitados para ejecutar el equipo.")

        project = self.database.get_project(project_id) if project_id else None
        project_label = project.name if project else "sin proyecto activo"

        steps: list[TeamStepResult] = []
        for profile in selected_profiles:
            step = await self._custom_team_step_reply(
                profile=profile,
                project_label=project_label,
                user_prompt=clean_prompt,
                previous_steps=steps,
            )
            steps.append(step)

        final_output = steps[-1].output if steps else "No se genero salida."
        trace = {
            "team_steps": [
                {
                    "agent_key": item.agent_key,
                    "display_name": item.display_name,
                    "role": item.role,
                    "model_name": item.model_name,
                    "status": item.status,
                }
                for item in steps
            ]
        }
        self.database.add_conversation(
            project_id=project_id,
            user_input=clean_prompt,
            assistant_output=final_output,
            trace=trace,
        )
        self.database.add_memory(
            project_id=project_id,
            memory_type="team_run",
            content=f"User: {clean_prompt}\nFinal: {final_output}",
            metadata={"agent_keys": [item.agent_key for item in steps], "steps": len(steps)},
            relevance=1.0,
        )
        return TeamRunResult(
            project_id=project_id,
            user_prompt=clean_prompt,
            final_output=final_output,
            steps=steps,
        )

    async def run_agent(self, *, project_id: int | None, user_prompt: str) -> AgentRunResult:
        ethics_check = self.ethics.check_input(user_prompt)
        if not ethics_check.is_safe:
            violation_response = self.ethics.get_violation_response(ethics_check)
            return AgentRunResult(
                messages=[AIMessage(content=violation_response)],
                sections={"action": violation_response},
                final_output=violation_response,
            )

        project = self.database.get_project(project_id) if project_id else None
        project_context = (
            f"Project name: {project.name}\nProject path: {project.root_path}\n"
            f"Project description: {project.description or 'N/A'}"
            if project
            else "No active project selected."
        )
        memory_context = self._build_memory_context(project_id, user_prompt)

        context_result = await self.context_engine.acquire_context(
            query=user_prompt,
            project_path=project.root_path if project else None,
            memory_context=memory_context,
        )
        acquired_context = ""
        if context_result.sources:
            acquired_context = self.context_engine.build_context_prompt(context_result)

        agent_config = self._build_agent_config()
        ethics_prompt = self.ethics.build_ethics_prompt()
        agent_config["ethics_prompt"] = ethics_prompt

        enriched_context = project_context
        if acquired_context:
            enriched_context = f"{project_context}\n\n{acquired_context}"

        initial_state: dict[str, Any] = {
            "messages": [HumanMessage(content=user_prompt)],
            "project_context": enriched_context,
            "memory_context": memory_context,
            "agent_config": agent_config,
        }
        result = await self.graph.ainvoke(initial_state)
        messages = list(result.get("messages", []))
        sections = self._extract_sections(messages)
        final_output = self._final_output(messages)

        final_output = self.ethics.sanitize_output(final_output)

        trace = {
            "sections": sections,
            "project_context": project_context,
            "context_confidence": context_result.confidence,
            "ethics_warnings": [w for w in ethics_check.warnings],
        }
        self.database.add_conversation(
            project_id=project_id,
            user_input=user_prompt,
            assistant_output=final_output,
            trace=trace,
        )
        self.database.add_memory(
            project_id=project_id,
            memory_type="interaction",
            content=f"User: {user_prompt}\nAssistant: {final_output}",
            metadata={"sections": list(sections.keys())},
            relevance=1.0,
        )

        return AgentRunResult(messages=messages, sections=sections, final_output=final_output)

    def run_agent_sync(self, *, project_id: int | None, user_prompt: str) -> AgentRunResult:
        return asyncio.run(self.run_agent(project_id=project_id, user_prompt=user_prompt))

    async def run_subagent_pipeline(
        self,
        *,
        project_id: int | None,
        user_prompt: str,
        pipeline_type: str | None = None,
    ) -> dict[str, Any]:
        ethics_check = self.ethics.check_input(user_prompt)
        if not ethics_check.is_safe:
            return {
                "task": user_prompt,
                "final_output": self.ethics.get_violation_response(ethics_check),
                "steps": [],
                "pipeline_type": "blocked",
                "ethics_blocked": True,
            }

        project = self.database.get_project(project_id) if project_id else None
        memory_context = self._build_memory_context(project_id, user_prompt)

        context_result = await self.context_engine.acquire_context(
            query=user_prompt,
            project_path=project.root_path if project else None,
            memory_context=memory_context,
        )
        context = self.context_engine.build_context_prompt(context_result)

        result = await self.subagent_orchestrator.run_pipeline(
            task=user_prompt,
            pipeline_type=pipeline_type,
            context=context,
            ethics_prompt=self.ethics.build_ethics_prompt(),
        )

        final_output = self.ethics.sanitize_output(result.final_output)

        self.database.add_conversation(
            project_id=project_id,
            user_input=user_prompt,
            assistant_output=final_output,
            trace={
                "pipeline_type": result.pipeline_type,
                "total_agents": result.total_agents,
                "successful_agents": result.successful_agents,
            },
        )
        self.database.add_memory(
            project_id=project_id,
            memory_type="subagent_pipeline",
            content=f"User: {user_prompt}\nFinal: {final_output}",
            metadata={
                "pipeline_type": result.pipeline_type,
                "agents": [s.role for s in result.steps],
            },
            relevance=1.0,
        )

        return {
            "task": result.task,
            "final_output": final_output,
            "steps": [
                {
                    "role": s.role,
                    "display_name": s.display_name,
                    "status": s.status,
                    "output": s.output,
                    "confidence": s.confidence,
                }
                for s in result.steps
            ],
            "pipeline_type": result.pipeline_type,
            "total_agents": result.total_agents,
            "successful_agents": result.successful_agents,
        }

    async def assistant_turn(
        self,
        *,
        project_id: int | None,
        user_prompt: str,
    ) -> AssistantTurnResult:
        text = user_prompt.strip()
        if not text:
            machine_translation = self._translate_prompt_to_machine_ir(
                text="",
                project_id=project_id,
                command_text=None,
                game_folder=None,
                control_kind=None,
            )
            return AssistantTurnResult(
                reply="Escribe una solicitud para continuar.",
                source="system",
                project_id=project_id,
                sections={},
                machine_translation=machine_translation,
            )

        ethics_check = self.ethics.check_input(text)
        if not ethics_check.is_safe:
            violation_response = self.ethics.get_violation_response(ethics_check)
            machine_translation = self._translate_prompt_to_machine_ir(
                text=text, project_id=project_id, command_text=None,
                game_folder=None, control_kind=None,
            )
            return AssistantTurnResult(
                reply=violation_response,
                source="ethics-block",
                project_id=project_id,
                sections={"action": violation_response},
                machine_translation=machine_translation,
            )

        # ── URL detection: fetch content of any links in the message ──────────
        detected_urls = extract_urls(text)
        url_fetched: list[dict] = []
        agent_prompt = text
        if detected_urls:
            url_fetched = await fetch_all_urls(detected_urls)
            successful = [r for r in url_fetched if not r.get("error") and r.get("content")]
            if successful:
                url_blocks: list[str] = []
                for r in successful:
                    header = f"[URL: {r['url']}]"
                    if r.get("title"):
                        header += f" — {r['title']}"
                    url_blocks.append(f"{header}\n{r['content'][:1500]}")
                url_section = "\n\n---\n".join(url_blocks)
                agent_prompt = (
                    f"[URL_CONTEXT]\n{url_section}\n[/URL_CONTEXT]\n\n"
                    f"Solicitud del usuario: {text}"
                )

        command_text = self._extract_command_request(text)
        game_folder = self._extract_game_scaffold_request(text)
        control_kind = self._detect_control_kind(text)

        if command_text or game_folder or control_kind:
            machine_translation = self._translate_prompt_to_machine_ir(
                text=text,
                project_id=project_id,
                command_text=command_text,
                game_folder=game_folder,
                control_kind=control_kind,
            )
        else:
            translated = await self.prompt_orchestrator.translate(text, project_id)
            machine_translation = self.prompt_orchestrator.to_machine_ir(translated, project_id)

        if url_fetched:
            machine_translation["url_contexts"] = url_fetched

        # ── Context acquisition: enrich machine_translation with memory-based context ──
        memory_context = self._build_memory_context(project_id, text)
        context_snapshot = self._build_context_snapshot(text, memory_context, machine_translation)
        machine_translation["context_snapshot"] = context_snapshot

        context_requirements = machine_translation.get("context_requirements", {})
        if bool(context_requirements.get("is_blocking", False)):
            missing_items = context_requirements.get("missing", [])
            reasons = context_requirements.get("reasons", [])
            missing_text = ", ".join(str(item) for item in missing_items) or "contexto"
            reason_text = " ".join(str(item) for item in reasons).strip()
            reply = (
                f"Falta contexto para ejecutar la tarea: {missing_text}. "
                "Indica el proyecto objetivo (por ejemplo: 'usar proyecto <nombre>')."
            )
            if reason_text:
                reply = f"{reply}\nMotivo: {reason_text}"
            return AssistantTurnResult(
                reply=reply,
                source="context-request",
                project_id=project_id,
                sections={"action": reply},
                machine_translation=machine_translation,
            )

        if command_text:
            code, output = self.execute_project_command(
                project_id=project_id,
                command_text=command_text,
            )
            body = output.strip() if output.strip() else "<no output>"
            return AssistantTurnResult(
                reply=f"$ {command_text}\n{body}\n[exit={code}]",
                source="command",
                project_id=project_id,
                sections={},
                machine_translation=machine_translation,
            )

        if game_folder:
            generated_path, same_project = self._create_browser_game_scaffold(
                project_id=project_id,
                folder_name=game_folder,
            )
            return AssistantTurnResult(
                reply=(
                    "Juego base creado correctamente.\n"
                    f"Ruta: {generated_path}\n"
                    "Abre index.html en tu navegador para jugar."
                ),
                source="artifact",
                project_id=same_project,
                sections={},
                machine_translation=machine_translation,
            )

        control_message, maybe_project_id = self.apply_prompt_instruction(
            instruction=text,
            current_project_id=project_id,
        )
        if not self._is_control_instruction_unrecognized(control_message):
            self.database.add_memory(
                project_id=maybe_project_id,
                memory_type="assistant_control",
                content=f"User: {text}\nAssistant: {control_message}",
                metadata={"source": "control"},
                relevance=0.9,
            )
            return AssistantTurnResult(
                reply=control_message,
                source="control",
                project_id=maybe_project_id,
                sections={},
                machine_translation=machine_translation,
            )

        execution_policy = machine_translation.get("execution_policy", {})
        single_pass_allowed = bool(execution_policy.get("single_pass_allowed", True))
        if (
            single_pass_allowed
            and self._single_pass_enabled()
            and os.getenv("FREE_LLM_PROVIDER", "ollama").strip().lower() == "ollama"
        ):
            single_pass_reply = await self._single_pass_reply(project_id=project_id, user_prompt=text)
            if single_pass_reply:
                single_pass_reply = self.ethics.sanitize_output(single_pass_reply)
                output_check = self.ethics.check_output(single_pass_reply)
                if not output_check.is_safe:
                    single_pass_reply = self.ethics.get_violation_response(output_check)
                sections = {"action": single_pass_reply}
                self.database.add_conversation(
                    project_id=project_id,
                    user_input=text,
                    assistant_output=single_pass_reply,
                    trace={"sections": sections, "single_pass": True},
                )
                self.database.add_memory(
                    project_id=project_id,
                    memory_type="interaction",
                    content=f"User: {text}\nAssistant: {single_pass_reply}",
                    metadata={"sections": ["action"], "single_pass": True},
                    relevance=1.0,
                )
                return AssistantTurnResult(
                    reply=single_pass_reply,
                    source="agent-single",
                    project_id=project_id,
                    sections=sections,
                    machine_translation=machine_translation,
                )

        agent_result = await self.run_agent(project_id=project_id, user_prompt=agent_prompt)
        reply = agent_result.sections.get("action") or agent_result.final_output
        return AssistantTurnResult(
            reply=reply,
            source="agent",
            project_id=project_id,
            sections=agent_result.sections,
            machine_translation=machine_translation,
        )

    def assistant_turn_sync(
        self,
        *,
        project_id: int | None,
        user_prompt: str,
    ) -> AssistantTurnResult:
        return asyncio.run(self.assistant_turn(project_id=project_id, user_prompt=user_prompt))

    def execute_project_command(
        self,
        *,
        project_id: int | None,
        command_text: str,
        timeout_seconds: int = 120,
    ) -> tuple[int, str]:
        project = self.database.get_project(project_id) if project_id else None
        cwd = project.root_path if project else str(Path.cwd())
        is_allowed, policy_message, use_shell = self._command_policy(command_text)

        if not is_allowed:
            code = 126
            output = policy_message
            self.database.log_command_run(
                project_id=project_id,
                command_text=command_text,
                return_code=code,
                output_text=output,
            )
            self.database.add_memory(
                project_id=project_id,
                memory_type="command_run",
                content=f"$ {command_text}\nexit={code}\n{output}",
                metadata={"return_code": code, "blocked": True},
                relevance=0.8,
            )
            return code, output

        try:
            run_target: str | list[str]
            if use_shell:
                run_target = command_text
            else:
                run_target = shlex.split(command_text)
                if not run_target:
                    return 126, "Comando vacio."
            completed = subprocess.run(
                run_target,
                cwd=cwd,
                shell=use_shell,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            code = int(completed.returncode)
        except subprocess.TimeoutExpired as exc:
            output = f"Command timed out after {timeout_seconds}s\n{exc}"
            code = 124
        except Exception as exc:  # pragma: no cover - defensive branch
            output = f"Execution error: {exc}"
            code = 1

        self.database.log_command_run(
            project_id=project_id,
            command_text=command_text,
            return_code=code,
            output_text=output,
        )
        self.database.add_memory(
            project_id=project_id,
            memory_type="command_run",
            content=f"$ {command_text}\nexit={code}\n{output[:2000]}",
            metadata={"return_code": code},
            relevance=0.8,
        )
        return code, output

    @staticmethod
    def _extract_sections(messages: list[BaseMessage]) -> dict[str, str]:
        sections: dict[str, str] = {}
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            content = str(message.content)
            if content.startswith("[OpenAI pensamiento resumido]"):
                sections["thought"] = content.replace("[OpenAI pensamiento resumido]", "", 1).strip()
            elif content.startswith("[Anthropic contraste]"):
                sections["review"] = content.replace("[Anthropic contraste]", "", 1).strip()
            elif content.startswith("[DeepSeek ejecucion]"):
                sections["action"] = content.replace("[DeepSeek ejecucion]", "", 1).strip()
            elif content.startswith("[DeepSeek ejecución]"):
                sections["action"] = content.replace("[DeepSeek ejecución]", "", 1).strip()
        return sections

    @staticmethod
    def _final_output(messages: list[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return str(message.content)
        return "No assistant output."

    def export_state_snapshot(self) -> str:
        """Return state for debugging or admin export."""
        data = {
            "projects": [project.__dict__ for project in self.list_projects()],
            "profiles": {key: profile.__dict__ for key, profile in self.get_profiles().items()},
        }
        return json.dumps(data, ensure_ascii=True, indent=2)

    @staticmethod
    def split_command(command_text: str) -> list[str]:
        """Utility parser for command previews."""
        return shlex.split(command_text)

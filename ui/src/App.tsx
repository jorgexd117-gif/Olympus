import { Dispatch, FormEvent, SetStateAction, useEffect, useMemo, useRef, useState } from "react";

import {
  assistantChat,
  availableModels,
  checkEthics,
  createProfile,
  getApiBaseUrl,
  getEthicsPrinciples,
  getMemoryStats,
  getResolvedApiBaseUrl,
  health,
  learnMemory,
  listAgents,
  listFolders,
  listOrchestratorWorkflows,
  listMemories,
  listProfiles,
  listProjects,
  queryDatabase,
  runAgentTeam,
  runFlashPipeline,
  runOrchestratorWorkflow,
  updateProfile,
  upsertProject,
} from "./api";
import type {
  AgentRecord,
  AgentRunResponse,
  EthicsCheckResult,
  EthicsPrinciples,
  FlashResult,
  Folder,
  HealthResponse,
  MemoryRecord,
  MemoryStats,
  OrchestratorRunResponse,
  OrchestratorWorkflow,
  Profile,
  Project,
  TeamRunResponse,
} from "./types";
import AgentList from "./components/AgentList";
import FolderManager from "./components/FolderManager";
import SubAgentPanel from "./components/SubAgentPanel";

type NavSection =
  | "dashboard"
  | "assistant"
  | "agents"
  | "folders"
  | "subagents"
  | "orchestrator"
  | "flash"
  | "profiles"
  | "memory"
  | "ethics"
  | "database";

type ProfileEdits = {
  model_name: string;
  system_prompt: string;
};

type AssistantMessage = {
  id: number;
  role: "user" | "assistant";
  text: string;
  source?: string;
};

function normalizeProfileEdits(items: Profile[]): Record<string, ProfileEdits> {
  const output: Record<string, ProfileEdits> = {};
  for (const profile of items) {
    output[profile.agent_key] = {
      model_name: profile.model_name,
      system_prompt: profile.system_prompt,
    };
  }
  return output;
}

function appendMessage(
  setter: Dispatch<SetStateAction<AssistantMessage[]>>,
  role: "user" | "assistant",
  text: string,
  source?: string
): void {
  setter((previous) => [
    ...previous,
    {
      id: Date.now() + Math.floor(Math.random() * 100000),
      role,
      text,
      source,
    },
  ]);
}

const NAV_ITEMS: { id: NavSection; label: string; icon: string }[] = [
  { id: "dashboard", label: "Dashboard", icon: "⊞" },
  { id: "assistant", label: "Asistente IA", icon: "💬" },
  { id: "agents", label: "Agentes", icon: "🤖" },
  { id: "folders", label: "Carpetas", icon: "📁" },
  { id: "subagents", label: "Sub-Agentes", icon: "⚡" },
  { id: "orchestrator", label: "Orquestador", icon: "🔀" },
  { id: "flash", label: "Flash Pipeline", icon: "⚡🧠" },
  { id: "ethics", label: "Ética", icon: "⚖️" },
  { id: "database", label: "Base de Datos", icon: "🗄️" },
  { id: "profiles", label: "Perfiles", icon: "⚙️" },
  { id: "memory", label: "Memoria", icon: "🧠" },
];

export default function App() {
  const [activeNav, setActiveNav] = useState<NavSection>("dashboard");
  const [healthState, setHealthState] = useState<HealthResponse | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profileEdits, setProfileEdits] = useState<Record<string, ProfileEdits>>({});
  const [models, setModels] = useState<string[]>([]);
  const [workflows, setWorkflows] = useState<OrchestratorWorkflow[]>([]);
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null);
  const [learnInput, setLearnInput] = useState("");
  const [learnTopic, setLearnTopic] = useState("general");
  const [learnStatus, setLearnStatus] = useState<string | null>(null);
  const [agentRecords, setAgentRecords] = useState<AgentRecord[]>([]);
  const [folders, setFolders] = useState<Folder[]>([]);
  const [machineTranslation, setMachineTranslation] = useState<Record<string, unknown> | null>(null);
  const [workflowResult, setWorkflowResult] = useState<OrchestratorRunResponse | null>(null);
  const [teamRunResult, setTeamRunResult] = useState<TeamRunResponse | null>(null);
  const [ethicsData, setEthicsData] = useState<EthicsPrinciples | null>(null);
  const [ethicsCheckText, setEthicsCheckText] = useState("");
  const [ethicsCheckResult, setEthicsCheckResult] = useState<EthicsCheckResult | null>(null);
  const [pipelineRunCount, setPipelineRunCount] = useState(0);
  const [dbSqlInput, setDbSqlInput] = useState("SELECT name FROM sqlite_master WHERE type='table' LIMIT 20;");
  const [dbQueryResult, setDbQueryResult] = useState<string | null>(null);
  const [dbQueryError, setDbQueryError] = useState<string | null>(null);

  const [thinkInput, setThinkInput] = useState("");
  const [planInput, setPlanInput] = useState("");
  const [actInput, setActInput] = useState("");
  const [thinkOutput, setThinkOutput] = useState<string | null>(null);
  const [planOutput, setPlanOutput] = useState<string | null>(null);
  const [actOutput, setActOutput] = useState<string | null>(null);

  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectPath, setNewProjectPath] = useState("");
  const [newProjectDescription, setNewProjectDescription] = useState("");
  const [workflowInput, setWorkflowInput] = useState("");
  const [selectedWorkflowId, setSelectedWorkflowId] = useState("assistant_quick");
  const [assistantInput, setAssistantInput] = useState("");
  const [teamTaskInput, setTeamTaskInput] = useState("");
  const [selectedTeamAgentKeys, setSelectedTeamAgentKeys] = useState<string[]>([]);
  const [newAgentKey, setNewAgentKey] = useState("");
  const [newAgentName, setNewAgentName] = useState("");
  const [newAgentRole, setNewAgentRole] = useState("");
  const [newAgentModel, setNewAgentModel] = useState("");
  const [newAgentPrompt, setNewAgentPrompt] = useState("");
  const [isTeamRunning, setIsTeamRunning] = useState(false);
  const [assistantMessages, setAssistantMessages] = useState<AssistantMessage[]>([
    {
      id: 1,
      role: "assistant",
      source: "system",
      text: "Asistente listo. Puedes pedirme cambios, preguntas tecnicas o usar 'comando: <instruccion>'.",
    },
  ]);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  const [statusText, setStatusText] = useState("Cargando...");
  const [isBusy, setIsBusy] = useState(false);
  const [connectionState, setConnectionState] = useState<"checking" | "online" | "degraded" | "offline">("checking");
  const [resolvedApiBaseUrl, setResolvedApiBaseUrl] = useState(getApiBaseUrl());
  const [lastHealthCheckAt, setLastHealthCheckAt] = useState("");
  const [isTaskRunning, setIsTaskRunning] = useState(false);
  const [isWorkflowRunning, setIsWorkflowRunning] = useState(false);
  const [taskProgress, setTaskProgress] = useState(0);
  const [taskStage, setTaskStage] = useState("Esperando solicitud.");
  const [taskStatus, setTaskStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const reconnectSyncInFlight = useRef(false);

  const [flashInput, setFlashInput] = useState("");
  const [flashResult, setFlashResult] = useState<FlashResult | null>(null);
  const [flashLoading, setFlashLoading] = useState(false);
  const [flashError, setFlashError] = useState<string | null>(null);
  const [flashActiveTab, setFlashActiveTab] = useState<"synthesis" | "logic" | "context" | "manifest">("synthesis");

  const activeProject = useMemo(
    () => projects.find((item) => item.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void refreshHealth(true);
    }, 8000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    void refreshMemories(selectedProjectId);
  }, [selectedProjectId]);

  useEffect(() => {
    const enabledKeys = profiles.filter((p) => p.is_enabled).map((p) => p.agent_key);
    setSelectedTeamAgentKeys((prev) => {
      const kept = prev.filter((item) => enabledKeys.includes(item));
      return kept.length > 0 ? kept : enabledKeys.slice(0, 3);
    });
  }, [profiles]);

  useEffect(() => {
    if (!isTaskRunning) return undefined;
    const timer = window.setInterval(() => {
      setTaskProgress((prev) => {
        if (prev >= 92) return prev;
        if (prev < 35) return Math.min(92, prev + 7);
        if (prev < 70) return Math.min(92, prev + 5);
        return Math.min(92, prev + 2);
      });
    }, 450);
    return () => window.clearInterval(timer);
  }, [isTaskRunning]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [assistantMessages]);

  async function reloadWorkspaceData(): Promise<boolean> {
    let hasPartialFailure = false;
    const [projectsRes, profilesRes, modelsRes, workflowsRes, agentsRes, foldersRes] = await Promise.allSettled([
      listProjects(),
      listProfiles(),
      availableModels(),
      listOrchestratorWorkflows(),
      listAgents(),
      listFolders(),
    ]);

    if (projectsRes.status === "fulfilled") {
      setProjects(projectsRes.value);
      if (projectsRes.value.length > 0) {
        setSelectedProjectId((prev) => prev ?? projectsRes.value[0].id);
      }
    } else hasPartialFailure = true;

    if (profilesRes.status === "fulfilled") {
      setProfiles(profilesRes.value);
      setProfileEdits(normalizeProfileEdits(profilesRes.value));
    } else hasPartialFailure = true;

    if (modelsRes.status === "fulfilled") {
      setModels(modelsRes.value.models);
    } else hasPartialFailure = true;

    if (workflowsRes.status === "fulfilled") {
      setWorkflows(workflowsRes.value);
      if (workflowsRes.value.length > 0) {
        const exists = workflowsRes.value.some((w) => w.workflow_id === selectedWorkflowId);
        if (!exists) setSelectedWorkflowId(workflowsRes.value[0].workflow_id);
      }
    } else hasPartialFailure = true;

    if (agentsRes.status === "fulfilled") setAgentRecords(agentsRes.value);
    else hasPartialFailure = true;

    if (foldersRes.status === "fulfilled") setFolders(foldersRes.value);
    else hasPartialFailure = true;

    return !hasPartialFailure;
  }

  async function syncWorkspaceAfterReconnect(): Promise<void> {
    if (reconnectSyncInFlight.current) return;
    reconnectSyncInFlight.current = true;
    try {
      const fullReload = await reloadWorkspaceData();
      if (fullReload) {
        setStatusText("Conexión recuperada. Datos sincronizados.");
      } else {
        setConnectionState("degraded");
        setStatusText("Conexión recuperada con carga parcial.");
      }
    } catch (error) {
      setConnectionState("degraded");
      setStatusText(`Error en sincronización: ${String(error)}`);
    } finally {
      reconnectSyncInFlight.current = false;
    }
  }

  async function refreshHealth(silent = false, syncOnReconnect = true): Promise<boolean> {
    try {
      const healthRes = await health();
      setHealthState(healthRes);
      let recoveredNow = false;
      setConnectionState((prev) => {
        recoveredNow = prev !== "online";
        return "online";
      });
      setResolvedApiBaseUrl(getResolvedApiBaseUrl());
      setLastHealthCheckAt(new Date().toLocaleTimeString());
      if (syncOnReconnect && recoveredNow) void syncWorkspaceAfterReconnect();
      return true;
    } catch (error) {
      setConnectionState((prev) => (prev === "online" || prev === "degraded" ? "degraded" : "offline"));
      setResolvedApiBaseUrl(getApiBaseUrl());
      if (!silent) setStatusText(`Error de conexión: ${String(error)}`);
      return false;
    }
  }

  async function bootstrap(): Promise<void> {
    try {
      setIsBusy(true);
      const isHealthy = await refreshHealth(true, false);
      const fullReload = isHealthy ? await reloadWorkspaceData() : false;
      if (isHealthy && fullReload) {
        setStatusText("API y UI conectadas correctamente.");
      } else if (isHealthy) {
        setConnectionState("degraded");
        setStatusText("API en linea con carga parcial.");
      } else {
        setStatusText("Error de conexión: health check fallido.");
      }
    } catch (error) {
      setStatusText(`Error de conexión: ${String(error)}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function refreshMemories(projectId: number | null): Promise<void> {
    try {
      const [response, stats] = await Promise.all([
        listMemories({ project_id: projectId, limit: 20 }),
        getMemoryStats(projectId),
      ]);
      setMemories(response);
      setMemoryStats(stats);
      setConnectionState((prev) => (prev === "offline" ? "degraded" : prev));
    } catch {
      setMemories([]);
      void refreshHealth(true);
    }
  }

  async function refreshModels(): Promise<void> {
    try {
      const response = await availableModels();
      setModels(response.models);
      setStatusText("Catálogo de modelos actualizado.");
    } catch (error) {
      setStatusText(`No se pudo actualizar modelos: ${String(error)}`);
    }
  }

  async function handleRunWorkflow(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!selectedWorkflowId) { setStatusText("Selecciona un workflow."); return; }
    try {
      setIsWorkflowRunning(true);
      const response = await runOrchestratorWorkflow({
        workflow_id: selectedWorkflowId,
        project_id: selectedProjectId,
        user_prompt: workflowInput.trim(),
      });
      setWorkflowResult(response);
      setPipelineRunCount((c) => c + 1);
      if (response.output?.assistant && typeof response.output.assistant === "object") {
        const assistantOutput = response.output.assistant as { reply?: unknown; source?: unknown };
        if (typeof assistantOutput.reply === "string") {
          appendMessage(
            setAssistantMessages,
            "assistant",
            assistantOutput.reply,
            typeof assistantOutput.source === "string" ? assistantOutput.source : "orchestrator"
          );
        }
      }
      setStatusText(`Workflow ${response.workflow_id}: ${response.status}.`);
    } catch (error) {
      setStatusText(`Error en workflow: ${String(error)}`);
    } finally {
      setIsWorkflowRunning(false);
    }
  }

  async function handleCreateProject(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!newProjectName.trim() || !newProjectPath.trim()) { setStatusText("Nombre y ruta son requeridos."); return; }
    try {
      setIsBusy(true);
      const project = await upsertProject({
        name: newProjectName.trim(),
        root_path: newProjectPath.trim(),
        description: newProjectDescription.trim(),
      });
      const updatedProjects = await listProjects();
      setProjects(updatedProjects);
      setSelectedProjectId(project.id);
      setNewProjectName("");
      setNewProjectPath("");
      setNewProjectDescription("");
      setStatusText(`Proyecto guardado: ${project.name}`);
    } catch (error) {
      setStatusText(`Error guardando proyecto: ${String(error)}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function runTypedAgent(kind: "think" | "plan" | "act", text: string): Promise<string> {
    const prefix = kind === "think" ? "[THINK] " : kind === "plan" ? "[PLAN] " : "[ACT] ";
    const res = await assistantChat({ message: `${prefix}${text}`.trim(), project_id: selectedProjectId });
    return res.reply;
  }

  async function handleThinkSubmit(e: FormEvent<HTMLFormElement>): Promise<void> {
    e.preventDefault();
    const text = thinkInput.trim();
    if (!text) return;
    try { setIsBusy(true); setThinkOutput(await runTypedAgent("think", text)); setStatusText("Pensar: completado."); }
    catch (err) { setStatusText(`Pensar error: ${String(err)}`); }
    finally { setIsBusy(false); }
  }

  async function handlePlanSubmit(e: FormEvent<HTMLFormElement>): Promise<void> {
    e.preventDefault();
    const text = planInput.trim();
    if (!text) return;
    try { setIsBusy(true); setPlanOutput(await runTypedAgent("plan", text)); setStatusText("Planificar: completado."); }
    catch (err) { setStatusText(`Planificar error: ${String(err)}`); }
    finally { setIsBusy(false); }
  }

  async function handleActSubmit(e: FormEvent<HTMLFormElement>): Promise<void> {
    e.preventDefault();
    const text = actInput.trim();
    if (!text) return;
    try { setIsBusy(true); setActOutput(await runTypedAgent("act", text)); setStatusText("Acción: completado."); }
    catch (err) { setStatusText(`Acción error: ${String(err)}`); }
    finally { setIsBusy(false); }
  }

  async function handleSaveProfile(agentKey: string): Promise<void> {
    const edit = profileEdits[agentKey];
    if (!edit) return;
    try {
      setIsBusy(true);
      await updateProfile(agentKey, { model_name: edit.model_name, system_prompt: edit.system_prompt });
      const freshProfiles = await listProfiles();
      setProfiles(freshProfiles);
      setProfileEdits(normalizeProfileEdits(freshProfiles));
      setStatusText(`Perfil actualizado: ${agentKey}`);
    } catch (error) {
      setStatusText(`Error actualizando perfil: ${String(error)}`);
    } finally {
      setIsBusy(false);
    }
  }

  function toggleTeamAgent(agentKey: string): void {
    setSelectedTeamAgentKeys((prev) =>
      prev.includes(agentKey) ? prev.filter((item) => item !== agentKey) : [...prev, agentKey]
    );
  }

  async function handleCreateAgent(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const key = newAgentKey.trim().toLowerCase();
    const name = newAgentName.trim();
    const role = newAgentRole.trim();
    const model = newAgentModel.trim();
    const prompt = newAgentPrompt.trim();
    if (!key || !name || !role || !model || !prompt) {
      setStatusText("Completa todos los campos del agente.");
      return;
    }
    try {
      setIsBusy(true);
      await createProfile({ agent_key: key, display_name: name, role, model_name: model, system_prompt: prompt, is_enabled: true });
      const freshProfiles = await listProfiles();
      setProfiles(freshProfiles);
      setProfileEdits(normalizeProfileEdits(freshProfiles));
      setNewAgentKey(""); setNewAgentName(""); setNewAgentRole(""); setNewAgentModel(""); setNewAgentPrompt("");
      setStatusText(`Agente creado: ${key}`);
    } catch (error) {
      setStatusText(`No se pudo crear el agente: ${String(error)}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleRunAgentTeam(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const prompt = teamTaskInput.trim();
    if (!prompt) { setStatusText("Escribe la tarea del equipo."); return; }
    if (selectedTeamAgentKeys.length === 0) { setStatusText("Selecciona al menos un agente."); return; }
    try {
      setIsTeamRunning(true);
      const response = await runAgentTeam({ user_prompt: prompt, project_id: selectedProjectId, agent_keys: selectedTeamAgentKeys });
      setTeamRunResult(response);
      setPipelineRunCount((c) => c + 1);
      appendMessage(setAssistantMessages, "assistant", response.final_output, "team");
      setStatusText(`Equipo ejecutado: ${response.steps.length} agentes.`);
    } catch (error) {
      setStatusText(`Error en equipo: ${String(error)}`);
    } finally {
      setIsTeamRunning(false);
    }
  }

  async function handleAssistantSend(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const text = assistantInput.trim();
    if (!text) return;

    appendMessage(setAssistantMessages, "user", text);
    setAssistantInput("");
    setIsTaskRunning(true);
    setTaskStatus("running");
    setTaskProgress(10);
    setTaskStage("Traduciendo solicitud...");

    try {
      setIsBusy(true);
      const response = await assistantChat({ message: text, project_id: selectedProjectId });
      setTaskProgress((prev) => Math.max(prev, 80));
      setTaskStage("Procesando respuesta...");

      appendMessage(setAssistantMessages, "assistant", response.reply, response.source);
      setMachineTranslation(response.machine_translation ?? null);

      const nextProjectId = response.project_id;
      if (nextProjectId !== selectedProjectId) setSelectedProjectId(nextProjectId);

      if (response.source === "control") {
        const [updatedProjects, updatedProfiles] = await Promise.all([listProjects(), listProfiles()]);
        setProjects(updatedProjects);
        setProfiles(updatedProfiles);
        setProfileEdits(normalizeProfileEdits(updatedProfiles));
      }

      await refreshMemories(nextProjectId);
      const intent = typeof response.machine_translation?.intent === "string" ? response.machine_translation.intent : "n/a";
      setTaskProgress(100);
      setTaskStatus("done");
      setTaskStage(`Completado: ${intent}.`);
      setStatusText(`Asistente respondió via ${response.source}. Intención: ${intent}.`);
    } catch (error) {
      const errorText = `Error del asistente: ${String(error)}`;
      appendMessage(setAssistantMessages, "assistant", errorText, "error");
      setStatusText(errorText);
      void refreshHealth(true);
      setTaskProgress(100);
      setTaskStatus("error");
      setTaskStage("Error al resolver la tarea.");
    } finally {
      setIsBusy(false);
      setIsTaskRunning(false);
      window.setTimeout(() => {
        setTaskProgress(0);
        setTaskStatus("idle");
        setTaskStage("Esperando solicitud.");
      }, 1400);
    }
  }

  /* ── Render sections ────────────────────────────────────────────────────── */

  function renderDashboard() {
    const enabledAgents = profiles.filter((p) => p.is_enabled);
    const enabledCount = enabledAgents.length;
    const totalAgents = profiles.length + agentRecords.length;
    const activeModels = [...new Set(enabledAgents.map((p) => p.model_name).filter(Boolean))];
    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Dashboard</h2>
            <p className="section-subtitle">Vista general del sistema multi-agente</p>
          </div>
          <button className="btn btn-primary" onClick={() => setActiveNav("assistant")}>
            💬 Iniciar Chat
          </button>
        </div>

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-icon purple">🤖</div>
            <div className="metric-label">Agentes Activos</div>
            <div className="metric-value">{enabledCount}</div>
            {activeModels.length > 0 ? (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.25rem", justifyContent: "center", marginTop: "0.25rem" }}>
                {activeModels.map((m) => (
                  <span key={m} style={{ fontSize: "0.65rem", fontFamily: "var(--font-mono)", background: "rgba(124,58,237,0.1)", color: "var(--primary)", padding: "0.1rem 0.4rem", borderRadius: "4px", whiteSpace: "nowrap" }}>
                    {m}
                  </span>
                ))}
              </div>
            ) : (
              <div className="metric-sub">{totalAgents} total registrados</div>
            )}
          </div>
          <div className="metric-card">
            <div className="metric-icon green">📁</div>
            <div className="metric-label">Carpetas</div>
            <div className="metric-value">{folders.length}</div>
            <div className="metric-sub">con asignaciones de proceso</div>
          </div>
          <div className="metric-card">
            <div className="metric-icon blue">🔀</div>
            <div className="metric-label">Ejecuciones Pipeline</div>
            <div className="metric-value">{pipelineRunCount}</div>
            <div className="metric-sub">{workflows.length} pipelines disponibles</div>
          </div>
          <div className="metric-card">
            <div className="metric-icon orange">⚖️</div>
            <div className="metric-label">Verificaciones Éticas</div>
            <div className="metric-value">{ethicsData?.audit_summary?.total_checks ?? 0}</div>
            <div className="metric-sub">del marco de ética activo</div>
          </div>
          <div className="metric-card" style={{ gridColumn: "1 / -1" }}>
            <div className="metric-icon" style={{ background: "rgba(16,185,129,0.15)", color: "#10b981" }}>🧠</div>
            <div className="metric-label">Nivel de Conocimiento de la IA</div>
            <div className="metric-value" style={{ fontSize: "1.4rem" }}>
              {memoryStats ? `${memoryStats.level_label} · ${memoryStats.knowledge_level}%` : "Calculando..."}
            </div>
            <div style={{ width: "100%", maxWidth: "420px", margin: "0.5rem auto 0.25rem" }}>
              <div style={{ background: "rgba(255,255,255,0.08)", borderRadius: "999px", height: "12px", overflow: "hidden", position: "relative" }}>
                <div style={{
                  height: "100%",
                  width: `${memoryStats?.knowledge_level ?? 0}%`,
                  background: "linear-gradient(90deg, #7C3AED, #10b981)",
                  borderRadius: "999px",
                  transition: "width 1.2s cubic-bezier(0.4,0,0.2,1)",
                  boxShadow: "0 0 10px rgba(124,58,237,0.5)",
                }} />
              </div>
            </div>
            <div className="metric-sub">
              {memoryStats
                ? `${memoryStats.total_memories} memorias · ${memoryStats.unique_topics} temas · ${memoryStats.total_conversations} conversaciones`
                : "Sin datos aún"}
            </div>
          </div>
        </div>

        <div className="two-col">
          <div className="card">
            <div className="card-header">
              <span className="card-title"><span className="card-icon">🟢</span> Estado del Sistema</span>
            </div>
            <div className="system-status-rows">
              <div className="system-status-row">
                <span className="system-status-row-label">Backend API</span>
                <span className={`badge-${connectionState}`}>
                  {connectionState === "online" ? "En línea" : connectionState === "degraded" ? "Degradado" : connectionState === "offline" ? "Sin conexión" : "Verificando"}
                </span>
              </div>
              <div className="system-status-row">
                <span className="system-status-row-label">Base de Datos</span>
                <span className={healthState ? "badge-online" : "badge-checking"}>
                  {healthState?.db_path ?? "—"}
                </span>
              </div>
              <div className="system-status-row">
                <span className="system-status-row-label">Última verificación</span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{lastHealthCheckAt || "Pendiente..."}</span>
              </div>
              <div className="system-status-row">
                <span className="system-status-row-label">URL API</span>
                <span style={{ fontSize: "0.72rem", fontFamily: "JetBrains Mono, monospace", color: "var(--text-muted)" }}>
                  {resolvedApiBaseUrl || "proxy local"}
                </span>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title"><span className="card-icon">⚡</span> Acciones Rápidas</span>
            </div>
            <div className="quick-actions">
              <button className="quick-action-btn" onClick={() => setActiveNav("assistant")}>
                💬 Abrir Asistente
              </button>
              <button className="quick-action-btn" onClick={() => setActiveNav("agents")}>
                🤖 Gestionar Agentes
              </button>
              <button className="quick-action-btn" onClick={() => setActiveNav("subagents")}>
                ⚡ Ejecutar Pipeline
              </button>
              <button className="quick-action-btn" onClick={() => setActiveNav("orchestrator")}>
                🔀 Lanzar Workflow
              </button>
              <button className="quick-action-btn" onClick={() => void reloadWorkspaceData()}>
                🔄 Sincronizar Todo
              </button>
            </div>

            <hr className="divider" />

            <div className="card-title" style={{ fontSize: "0.82rem", marginBottom: "0.5rem" }}>
              Proyecto activo
            </div>
            {activeProject ? (
              <div style={{ fontSize: "0.82rem" }}>
                <strong>{activeProject.name}</strong>
                <div style={{ color: "var(--text-muted)", fontSize: "0.73rem", marginTop: "0.2rem", fontFamily: "JetBrains Mono, monospace" }}>
                  {activeProject.root_path}
                </div>
              </div>
            ) : (
              <p className="muted">Sin proyecto seleccionado.</p>
            )}
          </div>
        </div>

        {profiles.length > 0 && (
          <div className="card">
            <div className="card-header">
              <span className="card-title"><span className="card-icon">🤖</span> Agentes registrados</span>
              <button className="btn btn-secondary btn-sm" onClick={() => setActiveNav("agents")}>Ver todos</button>
            </div>
            <div className="agent-grid">
              {profiles.slice(0, 6).map((p) => (
                <div key={p.agent_key} className={`agent-card ${p.is_enabled ? "" : "disabled"}`}>
                  <div className="agent-card-header">
                    <strong>{p.display_name}</strong>
                    <span className={`status-dot ${p.is_enabled ? "active" : "inactive"}`} />
                  </div>
                  <div className="agent-card-meta">
                    <span className="agent-key">{p.agent_key}</span>
                    <span className="agent-role">{p.role}</span>
                  </div>
                  {p.model_name && <div className="agent-model">{p.model_name}</div>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderAssistant() {
    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Asistente IA</h2>
            <p className="section-subtitle">
              Proyecto: <strong>{activeProject?.name ?? "ninguno"}</strong>
            </p>
          </div>
        </div>

        <div className="assistant-layout">
          <div className="chat-card">
            <div className="chat-card-header">
              <span className="chat-card-title">💬 Copiloto Asistente</span>
              <span className={`status-chip ${connectionState}`}>
                <span className="status-chip-dot" />
                {connectionState === "online" ? "Conectado" : connectionState === "degraded" ? "Degradado" : connectionState === "offline" ? "Sin conexión" : "Verificando"}
              </span>
            </div>

            <div className="chat-messages">
              {assistantMessages.map((message) => (
                <div key={message.id} className={`chat-bubble ${message.role}`}>
                  <div className="chat-avatar">
                    {message.role === "user" ? "U" : "AI"}
                  </div>
                  <div className="chat-content">
                    <div className="chat-meta">
                      <strong>{message.role === "user" ? "Tú" : "Asistente"}</strong>
                      {message.source && <span>· {message.source}</span>}
                    </div>
                    <div className="chat-text">{message.text}</div>
                  </div>
                </div>
              ))}
              <div ref={chatBottomRef} />
            </div>

            <div className="chat-input-area">
              <article className={`task-progress-card ${taskStatus}`}>
                <header>
                  <strong>Progreso</strong>
                  <span>{taskProgress}%</span>
                </header>
                <div className="task-progress-track">
                  <div className="task-progress-fill" style={{ width: `${taskProgress}%` }} />
                </div>
                <p>{taskStage}</p>
              </article>

              <form className="chat-input-row" onSubmit={handleAssistantSend}>
                <textarea
                  className="chat-input"
                  value={assistantInput}
                  onChange={(e) => setAssistantInput(e.target.value)}
                  rows={2}
                  placeholder="Escribe en lenguaje natural. También puedes usar: comando: <instruccion>"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      e.currentTarget.form?.requestSubmit();
                    }
                  }}
                />
                <button className="btn btn-primary chat-send-btn" disabled={isBusy} type="submit">
                  Enviar
                </button>
              </form>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div className="card">
              <div className="card-header">
                <span className="card-title">🧭 Contexto Detectado por la IA</span>
                {machineTranslation?.context_snapshot != null && (
                  <span style={{
                    fontSize: "0.72rem",
                    padding: "0.15rem 0.55rem",
                    borderRadius: "999px",
                    background: "rgba(16,185,129,0.15)",
                    color: "#10b981",
                    fontWeight: 600,
                  }}>
                    Confianza {Math.round(((machineTranslation.context_snapshot as Record<string,unknown>).confidence as number ?? 0) * 100)}%
                  </span>
                )}
              </div>
              {machineTranslation?.context_snapshot ? (() => {
                const ctx = machineTranslation.context_snapshot as Record<string, unknown>;
                const intent = String(ctx.intent ?? "general");
                const objective = String(ctx.objective ?? "—");
                const confidence = (ctx.confidence as number) ?? 0;
                const entities = (ctx.entities as string[]) ?? [];
                const recalled = (ctx.recalled_memories as {type:string; snippet:string; relevance:number}[]) ?? [];
                const subTasks = (ctx.sub_tasks as string[]) ?? [];
                const clarificationNeeded = Boolean(ctx.clarification_needed);
                const ambiguity = (ctx.ambiguity as number) ?? 0;

                const intentIcons: Record<string, string> = {
                  code_task: "💻", research: "🔍", data_query: "🗄️",
                  calculation: "🧮", question: "❓", analysis: "📊",
                  creative: "✨", system_info: "ℹ️", execution: "⚡", general: "💬",
                };

                return (
                  <div style={{ padding: "0.75rem 1rem 1rem" }}>
                    {/* Intent + Objective */}
                    <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.75rem", flexWrap: "wrap" }}>
                      <span style={{ fontSize: "1.2rem" }}>{intentIcons[intent] ?? "💬"}</span>
                      <span style={{ fontSize: "0.75rem", padding: "0.2rem 0.65rem", borderRadius: "999px", background: "rgba(124,58,237,0.15)", color: "var(--primary)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em" }}>
                        {intent}
                      </span>
                      <span style={{ fontSize: "0.75rem", padding: "0.2rem 0.65rem", borderRadius: "999px", background: "rgba(59,130,246,0.12)", color: "#60a5fa", fontWeight: 600 }}>
                        {String(ctx.priority ?? "medium")}
                      </span>
                    </div>

                    {/* Objective */}
                    <div style={{ marginBottom: "0.75rem" }}>
                      <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.2rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Lo que la IA entendió</p>
                      <p style={{ fontSize: "0.88rem", color: "var(--text)", lineHeight: 1.5, padding: "0.5rem 0.75rem", background: "rgba(255,255,255,0.04)", borderRadius: "6px", borderLeft: "3px solid var(--primary)" }}>
                        {objective}
                      </p>
                    </div>

                    {/* Confidence bar */}
                    <div style={{ marginBottom: "0.75rem" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                        <span>Confianza de interpretación</span>
                        <span style={{ color: confidence > 0.7 ? "#10b981" : confidence > 0.4 ? "#F59E0B" : "#EF4444", fontWeight: 600 }}>
                          {Math.round(confidence * 100)}%
                        </span>
                      </div>
                      <div style={{ background: "rgba(255,255,255,0.08)", borderRadius: "999px", height: "6px", overflow: "hidden" }}>
                        <div style={{
                          height: "100%",
                          width: `${Math.round(confidence * 100)}%`,
                          background: confidence > 0.7 ? "#10b981" : confidence > 0.4 ? "#F59E0B" : "#EF4444",
                          borderRadius: "999px",
                          transition: "width 0.8s ease",
                        }} />
                      </div>
                    </div>

                    {/* Entities detected */}
                    {entities.length > 0 && (
                      <div style={{ marginBottom: "0.75rem" }}>
                        <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.3rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Entidades detectadas</p>
                        <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
                          {entities.map((e, i) => (
                            <span key={i} style={{ fontSize: "0.75rem", padding: "0.15rem 0.55rem", borderRadius: "999px", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", color: "var(--text)" }}>
                              {e}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Recalled memories */}
                    {recalled.length > 0 && (
                      <div style={{ marginBottom: "0.75rem" }}>
                        <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.3rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>🧠 Memorias relevantes activadas</p>
                        <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                          {recalled.map((r, i) => (
                            <div key={i} style={{ fontSize: "0.78rem", padding: "0.4rem 0.6rem", background: "rgba(124,58,237,0.08)", borderRadius: "6px", borderLeft: "2px solid var(--primary)", color: "var(--text-muted)" }}>
                              <span style={{ fontSize: "0.65rem", color: "var(--primary)", fontWeight: 600, marginRight: "0.4rem" }}>[{r.type}]</span>
                              {r.snippet}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Sub-tasks */}
                    {subTasks.length > 0 && (
                      <div style={{ marginBottom: "0.75rem" }}>
                        <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "0.3rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Desglose de tareas</p>
                        <ol style={{ paddingLeft: "1.2rem", margin: 0, display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                          {subTasks.map((task, i) => (
                            <li key={i} style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>{task}</li>
                          ))}
                        </ol>
                      </div>
                    )}

                    {/* Clarification warning */}
                    {clarificationNeeded && (
                      <div style={{ padding: "0.5rem 0.75rem", background: "rgba(245,158,11,0.12)", borderRadius: "6px", borderLeft: "3px solid #F59E0B", marginBottom: "0.5rem" }}>
                        <p style={{ fontSize: "0.78rem", color: "#F59E0B", margin: 0 }}>
                          ⚠️ La solicitud es ambigua ({Math.round(ambiguity * 100)}%). La IA hará lo posible pero podría necesitar más detalles.
                        </p>
                      </div>
                    )}

                    {/* Technical details collapsible */}
                    <details style={{ marginTop: "0.5rem" }}>
                      <summary style={{ fontSize: "0.7rem", color: "var(--text-muted)", cursor: "pointer", userSelect: "none" }}>Ver datos técnicos (JSON)</summary>
                      <pre style={{ fontSize: "0.65rem", marginTop: "0.5rem", overflow: "auto", maxHeight: "200px" }}>{JSON.stringify(machineTranslation, null, 2)}</pre>
                    </details>
                  </div>
                );
              })() : (
                <p className="muted" style={{ padding: "1rem" }}>Envía un mensaje al asistente para ver cómo la IA interpreta tu solicitud.</p>
              )}
            </div>

            <div className="card">
              <div className="card-header">
                <span className="card-title">📁 Proyecto</span>
              </div>
              <div className="project-list">
                {projects.map((project) => (
                  <button
                    key={project.id}
                    className={`project-item ${project.id === selectedProjectId ? "active" : ""}`}
                    onClick={() => setSelectedProjectId(project.id)}
                    type="button"
                  >
                    <div>
                      <strong>{project.name}</strong>
                      <span style={{ display: "block" }}>{project.root_path}</span>
                    </div>
                  </button>
                ))}
                {projects.length === 0 && <p className="muted">Sin proyectos.</p>}
              </div>
              <form className="stack" onSubmit={handleCreateProject} style={{ marginTop: "0.75rem" }}>
                <input value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} placeholder="Nombre del proyecto" />
                <input value={newProjectPath} onChange={(e) => setNewProjectPath(e.target.value)} placeholder="Ruta absoluta" />
                <textarea value={newProjectDescription} onChange={(e) => setNewProjectDescription(e.target.value)} placeholder="Descripción" rows={2} />
                <button className="btn btn-primary" disabled={isBusy} type="submit">Guardar Proyecto</button>
              </form>
            </div>
          </div>
        </div>
      </div>
    );
  }

  function renderAgents() {
    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Agentes</h2>
            <p className="section-subtitle">Gestiona agentes y ejecuta tareas en equipo</p>
          </div>
        </div>

        <AgentList agents={agentRecords} onRefresh={() => void reloadWorkspaceData()} />

        <div className="two-col">
          <div className="card">
            <div className="card-header">
              <span className="card-title">➕ Crear Perfil de Agente</span>
            </div>
            <form className="stack" onSubmit={handleCreateAgent}>
              <input value={newAgentKey} onChange={(e) => setNewAgentKey(e.target.value)} placeholder="agent_key (ej: researcher)" />
              <input value={newAgentName} onChange={(e) => setNewAgentName(e.target.value)} placeholder="Nombre visible" />
              <input value={newAgentRole} onChange={(e) => setNewAgentRole(e.target.value)} placeholder="Rol (ej: investigador)" />
              <select value={newAgentModel} onChange={(e) => setNewAgentModel(e.target.value)}>
                <option value="">Selecciona modelo</option>
                {models.map((model) => <option key={model} value={model}>{model}</option>)}
              </select>
              <textarea value={newAgentPrompt} onChange={(e) => setNewAgentPrompt(e.target.value)} rows={4} placeholder="Prompt del sistema" />
              <button className="btn btn-primary" disabled={isBusy} type="submit">Crear Agente</button>
            </form>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">👥 Equipo de Agentes</span>
              <button className="btn btn-secondary btn-sm" disabled={isBusy} onClick={() => void refreshModels()}>
                Actualizar Modelos
              </button>
            </div>
            <form className="stack" onSubmit={handleRunAgentTeam}>
              <div className="agent-check-list">
                {profiles.map((profile) => (
                  <label key={profile.agent_key} className={`agent-check ${!profile.is_enabled ? "disabled" : ""}`}>
                    <input
                      checked={selectedTeamAgentKeys.includes(profile.agent_key)}
                      disabled={!profile.is_enabled}
                      onChange={() => toggleTeamAgent(profile.agent_key)}
                      type="checkbox"
                    />
                    <span>{profile.display_name} ({profile.role})</span>
                  </label>
                ))}
                {profiles.length === 0 && <p className="muted">Sin perfiles.</p>}
              </div>
              <textarea
                value={teamTaskInput}
                onChange={(e) => setTeamTaskInput(e.target.value)}
                rows={3}
                placeholder="Tarea para el equipo de agentes"
              />
              <button className="btn btn-primary" disabled={isBusy || isTeamRunning || selectedTeamAgentKeys.length === 0} type="submit">
                {isTeamRunning ? "Ejecutando..." : "Ejecutar Equipo"}
              </button>
            </form>
            {teamRunResult && (
              <pre style={{ marginTop: "0.75rem", maxHeight: "200px", overflow: "auto" }}>
                {JSON.stringify({ final_output: teamRunResult.final_output, steps: teamRunResult.steps.length }, null, 2)}
              </pre>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">⚙️ Agentes Pensar / Planificar / Actuar</span>
          </div>
          <div className="three-col">
            <form className="stack" onSubmit={handleThinkSubmit}>
              <label style={{ fontWeight: 600, fontSize: "0.85rem" }}>Pensar</label>
              <textarea value={thinkInput} onChange={(e) => setThinkInput(e.target.value)} rows={3} placeholder="Prompt para el agente Pensar" />
              <button className="btn btn-primary" disabled={isBusy} type="submit">Ejecutar</button>
              {thinkOutput && <pre style={{ maxHeight: 150, overflow: "auto" }}>{thinkOutput}</pre>}
            </form>
            <form className="stack" onSubmit={handlePlanSubmit}>
              <label style={{ fontWeight: 600, fontSize: "0.85rem" }}>Planificar</label>
              <textarea value={planInput} onChange={(e) => setPlanInput(e.target.value)} rows={3} placeholder="Prompt para el agente Planificar" />
              <button className="btn btn-primary" disabled={isBusy} type="submit">Ejecutar</button>
              {planOutput && <pre style={{ maxHeight: 150, overflow: "auto" }}>{planOutput}</pre>}
            </form>
            <form className="stack" onSubmit={handleActSubmit}>
              <label style={{ fontWeight: 600, fontSize: "0.85rem" }}>Acción</label>
              <textarea value={actInput} onChange={(e) => setActInput(e.target.value)} rows={3} placeholder="Prompt para el agente Acción" />
              <button className="btn btn-primary" disabled={isBusy} type="submit">Ejecutar</button>
              {actOutput && <pre style={{ maxHeight: 150, overflow: "auto" }}>{actOutput}</pre>}
            </form>
          </div>
        </div>
      </div>
    );
  }

  function renderFolders() {
    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Carpetas</h2>
            <p className="section-subtitle">Organiza agentes por carpetas y procesos</p>
          </div>
        </div>
        <FolderManager folders={folders} agents={agentRecords} onRefresh={() => void reloadWorkspaceData()} />
      </div>
    );
  }

  function renderSubAgents() {
    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Sub-Agentes y Herramientas</h2>
            <p className="section-subtitle">Ejecuta pipelines de múltiples agentes y consultas a la base de datos</p>
          </div>
        </div>
        <SubAgentPanel projectId={selectedProjectId} />
      </div>
    );
  }

  function renderOrchestrator() {
    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Orquestador LangChain</h2>
            <p className="section-subtitle">Ejecuta flujos operativos: diagnóstico, asistente rápido, pipeline completo</p>
          </div>
        </div>

        <div className="two-col">
          <div className="card">
            <div className="card-header">
              <span className="card-title">🔀 Ejecutar Workflow</span>
            </div>
            <form className="stack" onSubmit={handleRunWorkflow}>
              <div className="form-group">
                <label className="form-label">Workflow</label>
                <select value={selectedWorkflowId} onChange={(e) => setSelectedWorkflowId(e.target.value)}>
                  {workflows.map((workflow) => (
                    <option key={workflow.workflow_id} value={workflow.workflow_id}>{workflow.name}</option>
                  ))}
                  {workflows.length === 0 && <option value="">Sin workflows</option>}
                </select>
              </div>
              <textarea value={workflowInput} onChange={(e) => setWorkflowInput(e.target.value)} rows={3} placeholder="Prompt opcional para el workflow" />
              <button className="btn btn-primary" disabled={isWorkflowRunning || isBusy || workflows.length === 0} type="submit">
                {isWorkflowRunning ? "Ejecutando..." : "Lanzar Workflow"}
              </button>
            </form>
            {workflowResult && (
              <pre style={{ maxHeight: 250, overflow: "auto" }}>
                {JSON.stringify({ workflow: workflowResult.workflow_id, status: workflowResult.status, summary: workflowResult.summary }, null, 2)}
              </pre>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">📊 Último resultado</span>
            </div>
            {workflowResult ? (
              <pre style={{ maxHeight: 350, overflow: "auto" }}>
                {JSON.stringify(workflowResult, null, 2)}
              </pre>
            ) : (
              <p className="muted">Sin ejecuciones de workflow aún.</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  function renderProfiles() {
    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Perfiles y Modelos</h2>
            <p className="section-subtitle">Configura modelos y system prompts de cada agente</p>
          </div>
          <button className="btn btn-secondary" disabled={isBusy} onClick={() => void refreshModels()}>
            🔄 Actualizar Modelos
          </button>
        </div>

        <div className="profiles-grid">
          {profiles.map((profile) => {
            const edit = profileEdits[profile.agent_key];
            const options = Array.from(new Set([...(models || []), edit?.model_name || ""])).filter(Boolean).sort((a, b) => a.localeCompare(b));
            return (
              <div className="profile-card" key={profile.agent_key}>
                <h3>
                  {profile.display_name}
                  <span className={`status-dot ${profile.is_enabled ? "active" : "inactive"}`} style={{ marginLeft: "0.6rem" }} />
                </h3>
                <div style={{ display: "flex", gap: "0.4rem", marginBottom: "0.75rem" }}>
                  <span className="agent-key">{profile.agent_key}</span>
                  <span className="agent-role">{profile.role}</span>
                </div>
                <label>
                  Modelo
                  <select
                    value={edit?.model_name ?? ""}
                    onChange={(e) => setProfileEdits((prev) => ({
                      ...prev,
                      [profile.agent_key]: { model_name: e.target.value, system_prompt: prev[profile.agent_key]?.system_prompt ?? "" },
                    }))}
                  >
                    {options.map((option) => <option key={option} value={option}>{option}</option>)}
                  </select>
                </label>
                <label>
                  System Prompt
                  <textarea
                    value={edit?.system_prompt ?? ""}
                    onChange={(e) => setProfileEdits((prev) => ({
                      ...prev,
                      [profile.agent_key]: { model_name: prev[profile.agent_key]?.model_name ?? "", system_prompt: e.target.value },
                    }))}
                    rows={4}
                  />
                </label>
                <button className="btn btn-primary" disabled={isBusy} onClick={() => void handleSaveProfile(profile.agent_key)} type="button">
                  Guardar Perfil
                </button>
              </div>
            );
          })}
          {profiles.length === 0 && <p className="muted">Sin perfiles configurados.</p>}
        </div>
      </div>
    );
  }

  async function handleLearnSubmit(e: FormEvent<HTMLFormElement>): Promise<void> {
    e.preventDefault();
    if (!learnInput.trim()) return;
    try {
      setIsBusy(true);
      await learnMemory(learnInput.trim(), learnTopic, selectedProjectId);
      setLearnStatus("✅ Aprendizaje almacenado. La IA se ha nutrido de este conocimiento.");
      setLearnInput("");
      await refreshMemories(selectedProjectId);
    } catch {
      setLearnStatus("❌ Error al guardar el aprendizaje. Intenta de nuevo.");
    } finally {
      setIsBusy(false);
    }
  }

  function renderMemory() {
    const stats = memoryStats;
    const levelColors = ["#7C3AED", "#3B82F6", "#10b981", "#F59E0B", "#EF4444"];
    const topicLabels: Record<string, string> = {
      casino: "🎰 Casino",
      programacion: "💻 Programación",
      frontend: "🌐 Frontend",
      backend: "⚙️ Backend",
      ecommerce: "🛒 E-Commerce",
      general: "💬 General",
    };

    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Memoria & Aprendizaje</h2>
            <p className="section-subtitle">La IA aprende de cada interacción y se vuelve más inteligente</p>
          </div>
          <button className="btn btn-secondary" onClick={() => void refreshMemories(selectedProjectId)}>
            🔄 Actualizar
          </button>
        </div>

        {/* Knowledge progress card */}
        <div className="card" style={{ marginBottom: "1.25rem" }}>
          <div className="card-header">
            <span className="card-title">🧠 Nivel de Inteligencia Acumulada</span>
            {stats && (
              <span style={{
                fontSize: "0.8rem",
                background: "rgba(124,58,237,0.15)",
                color: "var(--primary)",
                padding: "0.2rem 0.7rem",
                borderRadius: "999px",
                fontWeight: 600,
              }}>
                {stats.level_label}
              </span>
            )}
          </div>
          <div style={{ padding: "0.5rem 1rem 1.25rem" }}>
            {/* Main progress bar */}
            <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "0.75rem" }}>
              <div style={{ flex: 1, background: "rgba(255,255,255,0.08)", borderRadius: "999px", height: "18px", overflow: "hidden" }}>
                <div style={{
                  height: "100%",
                  width: `${stats?.knowledge_level ?? 0}%`,
                  background: "linear-gradient(90deg, #7C3AED 0%, #3B82F6 50%, #10b981 100%)",
                  borderRadius: "999px",
                  transition: "width 1.4s cubic-bezier(0.4,0,0.2,1)",
                  boxShadow: "0 0 14px rgba(124,58,237,0.45)",
                  position: "relative",
                }} />
              </div>
              <span style={{ fontWeight: 700, color: "var(--primary)", minWidth: "3rem", textAlign: "right" }}>
                {stats?.knowledge_level ?? 0}%
              </span>
            </div>

            {/* Stats row */}
            <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
              {[
                { label: "Memorias", value: stats?.total_memories ?? 0, icon: "🗂" },
                { label: "Conversaciones", value: stats?.total_conversations ?? 0, icon: "💬" },
                { label: "Temas dominados", value: stats?.unique_topics ?? 0, icon: "📚" },
              ].map(({ label, value, icon }) => (
                <div key={label} style={{ textAlign: "center", flex: "1", minWidth: "80px" }}>
                  <div style={{ fontSize: "1.5rem", marginBottom: "0.15rem" }}>{icon}</div>
                  <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--text)" }}>{value}</div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{label}</div>
                </div>
              ))}
            </div>

            {/* Top topics */}
            {stats && stats.top_topics.length > 0 && (
              <div>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>Temas aprendidos:</p>
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  {stats.top_topics.map((t, i) => (
                    <span key={t.topic} style={{
                      fontSize: "0.75rem",
                      padding: "0.2rem 0.65rem",
                      borderRadius: "999px",
                      background: `${levelColors[i % levelColors.length]}22`,
                      color: levelColors[i % levelColors.length],
                      border: `1px solid ${levelColors[i % levelColors.length]}44`,
                      fontWeight: 600,
                    }}>
                      {topicLabels[t.topic] ?? t.topic} ({t.count})
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Manual learning form */}
        <div className="card" style={{ marginBottom: "1.25rem" }}>
          <div className="card-header">
            <span className="card-title">📥 Enseñar a la IA</span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Ingresa conocimiento directamente</span>
          </div>
          <form onSubmit={(e) => void handleLearnSubmit(e)} style={{ padding: "0.75rem 1rem 1rem" }}>
            <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem" }}>
              <select
                value={learnTopic}
                onChange={(e) => setLearnTopic(e.target.value)}
                style={{ padding: "0.4rem 0.7rem", borderRadius: "6px", background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)", fontSize: "0.85rem" }}
              >
                <option value="general">💬 General</option>
                <option value="casino">🎰 Casino</option>
                <option value="programacion">💻 Programación</option>
                <option value="frontend">🌐 Frontend</option>
                <option value="backend">⚙️ Backend</option>
                <option value="ecommerce">🛒 E-Commerce</option>
              </select>
            </div>
            <textarea
              value={learnInput}
              onChange={(e) => setLearnInput(e.target.value)}
              placeholder="Escribe el conocimiento que quieres que aprenda la IA... Ejemplo: 'El sistema de pagos debe usar Stripe con webhooks para confirmar microtransacciones.'"
              rows={4}
              style={{ width: "100%", resize: "vertical", padding: "0.6rem 0.75rem", borderRadius: "6px", background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)", fontSize: "0.85rem", fontFamily: "inherit", boxSizing: "border-box" }}
            />
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginTop: "0.5rem" }}>
              <button type="submit" className="btn btn-primary" disabled={isBusy || !learnInput.trim()}>
                {isBusy ? "Aprendiendo..." : "🧠 Enseñar"}
              </button>
              {learnStatus && <span style={{ fontSize: "0.8rem", color: learnStatus.startsWith("✅") ? "#10b981" : "#EF4444" }}>{learnStatus}</span>}
            </div>
          </form>
        </div>

        {/* Memory records */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">🗂 Memorias Recientes</span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{memories.length} registros</span>
          </div>
          <div className="memory-list">
            {memories.map((item) => (
              <div key={item.id} className="memory-item">
                <header>
                  <strong style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                    <span style={{ fontSize: "0.7rem", background: "rgba(124,58,237,0.12)", color: "var(--primary)", padding: "0.1rem 0.45rem", borderRadius: "4px" }}>
                      {item.memory_type}
                    </span>
                  </strong>
                  <span>{new Date(item.created_at).toLocaleString()}</span>
                </header>
                <pre>{item.content}</pre>
              </div>
            ))}
            {memories.length === 0 && <p className="muted">Sin registros de memoria. Interactúa con los agentes para que la IA aprenda.</p>}
          </div>
        </div>
      </div>
    );
  }

  function renderEthics() {
    const principles = ethicsData?.principles as Record<string, string> | undefined;
    const principleEntries = principles ? Object.entries(principles) : [];
    const audit = ethicsData?.audit_summary;

    async function loadEthics() {
      try {
        const data = await getEthicsPrinciples();
        setEthicsData(data);
      } catch {
        /* silent */
      }
    }

    async function runEthicsCheck(e: FormEvent) {
      e.preventDefault();
      if (!ethicsCheckText.trim()) return;
      try {
        const result = await checkEthics({ text: ethicsCheckText.trim(), check_type: "input" });
        setEthicsCheckResult(result);
      } catch {
        setEthicsCheckResult(null);
      }
    }

    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Marco de Ética</h2>
            <p className="section-subtitle">Principios, auditoría y verificación de contenido</p>
          </div>
          <button className="btn btn-secondary" onClick={() => void loadEthics()}>
            🔄 Cargar Principios
          </button>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem" }}>
          <div className="card">
            <div className="card-header">
              <span className="card-title">⚖️ Principios Éticos</span>
              {audit && (
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  {audit.total_checks} verificaciones
                </span>
              )}
            </div>
            {!ethicsData && (
              <p className="muted" style={{ padding: "1rem" }}>Haz clic en "Cargar Principios" para ver el marco ético activo.</p>
            )}
            {principleEntries.length > 0 && (
              <ul style={{ listStyle: "none", padding: "0.5rem 1rem", margin: 0, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                {principleEntries.map(([key, value]) => (
                  <li key={key} style={{ padding: "0.625rem", background: "var(--bg-page)", borderRadius: "6px", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                    <strong style={{ color: "var(--text-primary)" }}>{key}:</strong> {value}
                  </li>
                ))}
              </ul>
            )}
            {audit && (
              <div style={{ padding: "0.75rem 1rem", borderTop: "1px solid var(--border)", display: "flex", gap: "1rem", fontSize: "0.8rem" }}>
                <span style={{ color: "var(--success)" }}>✓ Permitidos: {audit.total_checks - audit.blocked}</span>
                <span style={{ color: "var(--danger)" }}>✗ Bloqueados: {audit.blocked}</span>
                <span style={{ color: "var(--text-muted)" }}>Total: {audit.total_checks}</span>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">🔍 Verificar Contenido</span>
            </div>
            <form onSubmit={(e) => void runEthicsCheck(e)} style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              <textarea
                value={ethicsCheckText}
                onChange={(e) => setEthicsCheckText(e.target.value)}
                placeholder="Introduce texto para verificar si cumple el marco ético..."
                rows={4}
                style={{ width: "100%", resize: "vertical", padding: "0.625rem", borderRadius: "6px", border: "1px solid var(--border)", fontSize: "0.875rem", fontFamily: "inherit", boxSizing: "border-box" }}
              />
              <button type="submit" className="btn btn-primary" disabled={!ethicsCheckText.trim()}>
                ⚖️ Verificar
              </button>
              {ethicsCheckResult && (
                <div style={{ padding: "0.75rem", borderRadius: "8px", background: !ethicsCheckResult.is_safe ? "#FEF2F2" : "#F0FDF4", border: `1px solid ${!ethicsCheckResult.is_safe ? "var(--danger)" : "var(--success)"}` }}>
                  <strong style={{ color: !ethicsCheckResult.is_safe ? "var(--danger)" : "var(--success)" }}>
                    {!ethicsCheckResult.is_safe ? "✗ No seguro" : "✓ Seguro"}
                  </strong>
                  {ethicsCheckResult.violations.length > 0 && (
                    <ul style={{ margin: "0.25rem 0 0", paddingLeft: "1rem", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                      {ethicsCheckResult.violations.map((v, i) => <li key={i}>{v.type}: {v.detail}</li>)}
                    </ul>
                  )}
                  {ethicsCheckResult.warnings.length > 0 && (
                    <p style={{ margin: "0.25rem 0 0", fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                      ⚠️ {ethicsCheckResult.warnings.join(", ")}
                    </p>
                  )}
                </div>
              )}
            </form>
          </div>
        </div>
      </div>
    );
  }

  function renderDatabase() {
    async function runQuery(e: FormEvent) {
      e.preventDefault();
      if (!dbSqlInput.trim()) return;
      setDbQueryError(null);
      setDbQueryResult(null);
      try {
        const res = await queryDatabase(dbSqlInput.trim());
        setDbQueryResult(res.result);
      } catch (err: unknown) {
        setDbQueryError(err instanceof Error ? err.message : String(err));
      }
    }

    return (
      <div className="page-content">
        <div className="section-header">
          <div>
            <h2 className="section-title">Base de Datos</h2>
            <p className="section-subtitle">Estado y consultas directas a la base de datos del sistema</p>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem", marginBottom: "1.25rem" }}>
          <div className="card">
            <div className="card-header"><span className="card-title">🗄️ Estado de la Base de Datos</span></div>
            <div style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.625rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem", background: "var(--bg-page)", borderRadius: "6px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Backend</span>
                <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>{healthState?.db_backend ?? "—"}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem", background: "var(--bg-page)", borderRadius: "6px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Ruta</span>
                <span style={{ fontSize: "0.8rem", fontFamily: "var(--font-mono)" }}>{healthState?.db_path ?? "—"}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem", background: "var(--bg-page)", borderRadius: "6px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Estado</span>
                <span className={`status-chip ${connectionState === "online" ? "online" : "offline"}`}>
                  {connectionState === "online" ? "Conectada" : "Sin conexión"}
                </span>
              </div>
              {healthState?.db_notice && (
                <div style={{ padding: "0.5rem", background: "#FFFBEB", borderRadius: "6px", fontSize: "0.8rem", color: "#92400E" }}>
                  ⚠️ {healthState.db_notice}
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header"><span className="card-title">📊 Resumen del Sistema</span></div>
            <div style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.625rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem", background: "var(--bg-page)", borderRadius: "6px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Agentes registrados</span>
                <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>{agentRecords.length}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem", background: "var(--bg-page)", borderRadius: "6px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Carpetas activas</span>
                <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>{folders.length}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem", background: "var(--bg-page)", borderRadius: "6px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Memorias almacenadas</span>
                <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>{memories.length}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.5rem", background: "var(--bg-page)", borderRadius: "6px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Proyectos</span>
                <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>{projects.length}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">💾 Consulta SQL</span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Solo lectura recomendada</span>
          </div>
          <form onSubmit={(e) => void runQuery(e)} style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            <textarea
              value={dbSqlInput}
              onChange={(e) => setDbSqlInput(e.target.value)}
              rows={3}
              style={{ width: "100%", resize: "vertical", padding: "0.625rem", borderRadius: "6px", border: "1px solid var(--border)", fontSize: "0.85rem", fontFamily: "var(--font-mono)", boxSizing: "border-box" }}
            />
            <button type="submit" className="btn btn-primary" style={{ alignSelf: "flex-start" }}>
              ▶ Ejecutar Consulta
            </button>
            {dbQueryError && (
              <div style={{ padding: "0.75rem", background: "#FEF2F2", borderRadius: "8px", color: "var(--danger)", fontSize: "0.85rem" }}>
                ✗ {dbQueryError}
              </div>
            )}
            {dbQueryResult !== null && (
              <pre style={{ margin: 0, padding: "0.75rem", background: "var(--bg-page)", borderRadius: "8px", fontSize: "0.8rem", fontFamily: "var(--font-mono)", overflowX: "auto", whiteSpace: "pre-wrap", maxHeight: "300px", overflowY: "auto" }}>
                {dbQueryResult}
              </pre>
            )}
          </form>
        </div>
      </div>
    );
  }

  async function handleFlashRun(e: FormEvent) {
    e.preventDefault();
    if (!flashInput.trim()) return;
    setFlashLoading(true);
    setFlashError(null);
    setFlashResult(null);
    try {
      const res = await runFlashPipeline(flashInput.trim(), selectedProjectId);
      setFlashResult(res);
      setFlashActiveTab("synthesis");
    } catch (err) {
      setFlashError(err instanceof Error ? err.message : "Error al ejecutar el pipeline");
    } finally {
      setFlashLoading(false);
    }
  }

  function renderFlash() {
    const faithScore = flashResult ? Math.round(flashResult.faithfulness.score * 100) : 0;
    const faithColor = faithScore >= 80 ? "#22c55e" : faithScore >= 50 ? "#f59e0b" : "#ef4444";

    const agentCards = [
      {
        id: "agent1",
        icon: "🧠",
        name: "Agente de Lógica y Estructura",
        role: "Arquitectura · Código · Pedagogía Técnica",
        color: "#3b82f6",
        latencyKey: "agent1_logic_ms",
      },
      {
        id: "agent2",
        icon: "💾",
        name: "Agente de Contexto y Memoria",
        role: "Buffer 1M tokens · Validación · Anti-Teléfono-Descompuesto",
        color: "#8b5cf6",
        latencyKey: "agent2_context_ms",
      },
      {
        id: "agent3",
        icon: "🎯",
        name: "Agente de Interfaz y Síntesis",
        role: "Respuesta fluida · Multimodal · Filtro de Entropía",
        color: "#10b981",
        latencyKey: "agent3_synthesis_ms",
      },
    ];

    const tabs: { id: typeof flashActiveTab; label: string }[] = [
      { id: "synthesis", label: "Respuesta Final" },
      { id: "logic", label: "Lógica (Agente 1)" },
      { id: "context", label: "Contexto (Agente 2)" },
      { id: "manifest", label: "Manifiesto" },
    ];

    return (
      <div className="section-content">
        <h2 className="section-title">⚡ Flash Pipeline — 3 Agentes Especializados</h2>
        <p style={{ color: "var(--text-muted)", marginBottom: "1.5rem", lineHeight: 1.6 }}>
          Arquitectura multi-agente con <strong>Manifiesto de Sesión</strong>, procesamiento
          asíncrono tipo Flash y <strong>Filtro de Entropía</strong> para garantizar fidelidad
          al intent original.
        </p>

        {/* Architecture diagram */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
          {agentCards.map((agent, idx) => (
            <div key={agent.id} style={{
              background: "var(--surface)",
              border: `1px solid ${agent.color}40`,
              borderRadius: "12px",
              padding: "1rem",
              position: "relative",
            }}>
              <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>{agent.icon}</div>
              <div style={{ fontWeight: 600, fontSize: "0.85rem", color: agent.color, marginBottom: "0.25rem" }}>
                Agente {idx + 1}
              </div>
              <div style={{ fontWeight: 700, fontSize: "0.9rem", marginBottom: "0.5rem" }}>
                {agent.name}
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", lineHeight: 1.4 }}>
                {agent.role}
              </div>
              {flashResult && (
                <div style={{
                  marginTop: "0.75rem",
                  padding: "0.25rem 0.5rem",
                  background: `${agent.color}20`,
                  borderRadius: "6px",
                  fontSize: "0.75rem",
                  color: agent.color,
                  fontWeight: 600,
                }}>
                  ✓ {Math.round(flashResult.agent_latencies[agent.latencyKey] ?? 0)}ms
                </div>
              )}
              {idx < 2 && (
                <div style={{
                  position: "absolute",
                  right: "-1.5rem",
                  top: "50%",
                  transform: "translateY(-50%)",
                  color: "var(--text-muted)",
                  fontSize: "1.2rem",
                  zIndex: 1,
                }}>→</div>
              )}
            </div>
          ))}
        </div>

        {/* Input form */}
        <form onSubmit={(e) => { void handleFlashRun(e); }} style={{ marginBottom: "1.5rem" }}>
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
            <textarea
              value={flashInput}
              onChange={(e) => setFlashInput(e.target.value)}
              placeholder="Escribe tu solicitud... (ej: 'Genera un casino online con microtransacciones')"
              rows={3}
              style={{
                flex: 1,
                padding: "0.75rem 1rem",
                borderRadius: "8px",
                border: "1px solid var(--border)",
                background: "var(--surface)",
                color: "var(--text)",
                fontSize: "0.9rem",
                resize: "vertical",
                fontFamily: "inherit",
              }}
            />
            <button
              type="submit"
              disabled={flashLoading || !flashInput.trim()}
              style={{
                padding: "0.75rem 1.5rem",
                background: flashLoading ? "var(--border)" : "linear-gradient(135deg, #3b82f6, #8b5cf6)",
                color: "#fff",
                border: "none",
                borderRadius: "8px",
                fontWeight: 600,
                cursor: flashLoading ? "not-allowed" : "pointer",
                whiteSpace: "nowrap",
                minWidth: "120px",
              }}
            >
              {flashLoading ? "⚡ Ejecutando..." : "⚡ Ejecutar Flash"}
            </button>
          </div>
        </form>

        {flashError && (
          <div style={{
            background: "#ef444420",
            border: "1px solid #ef4444",
            borderRadius: "8px",
            padding: "1rem",
            color: "#ef4444",
            marginBottom: "1rem",
          }}>
            ❌ {flashError}
          </div>
        )}

        {flashResult && (
          <>
            {/* Performance summary */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: "0.75rem",
              marginBottom: "1.5rem",
            }}>
              {[
                { label: "Latencia Total", value: `${Math.round(flashResult.total_latency_ms)}ms`, color: "#3b82f6" },
                { label: "Dominio Detectado", value: flashResult.manifest.domain, color: "#8b5cf6" },
                { label: "Prioridad", value: flashResult.manifest.priority.toUpperCase(), color: "#f59e0b" },
                { label: "Fidelidad", value: `${faithScore}%`, color: faithColor },
              ].map((stat) => (
                <div key={stat.label} style={{
                  background: "var(--surface)",
                  border: `1px solid ${stat.color}30`,
                  borderRadius: "10px",
                  padding: "0.75rem 1rem",
                  textAlign: "center",
                }}>
                  <div style={{ fontSize: "1.25rem", fontWeight: 700, color: stat.color }}>{stat.value}</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>{stat.label}</div>
                </div>
              ))}
            </div>

            {/* Faithfulness / Entropy filter */}
            <div style={{
              background: flashResult.faithfulness.is_faithful ? "#22c55e15" : "#f59e0b15",
              border: `1px solid ${flashResult.faithfulness.is_faithful ? "#22c55e" : "#f59e0b"}50`,
              borderRadius: "10px",
              padding: "1rem 1.25rem",
              marginBottom: "1.5rem",
              display: "flex",
              alignItems: "center",
              gap: "1rem",
            }}>
              <div style={{ fontSize: "1.5rem" }}>
                {flashResult.faithfulness.is_faithful ? "✅" : "⚠️"}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700, marginBottom: "0.25rem" }}>
                  Filtro de Entropía — Auto-Crítica del Agente 3
                </div>
                <div style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                  {flashResult.faithfulness.is_faithful
                    ? `Respuesta 100% fiel al intent original. Score: ${faithScore}%`
                    : `Desviaciones detectadas y corregidas. Score: ${faithScore}%`}
                </div>
                {flashResult.faithfulness.deviations.length > 0 && (
                  <div style={{ marginTop: "0.5rem" }}>
                    {flashResult.faithfulness.deviations.map((d, i) => (
                      <div key={i} style={{ fontSize: "0.8rem", color: "#f59e0b", marginTop: "0.25rem" }}>
                        • {d}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {/* Fidelity bar */}
              <div style={{ textAlign: "right", minWidth: "80px" }}>
                <div style={{
                  height: "8px",
                  background: "var(--border)",
                  borderRadius: "4px",
                  overflow: "hidden",
                  width: "80px",
                }}>
                  <div style={{
                    height: "100%",
                    width: `${faithScore}%`,
                    background: faithColor,
                    transition: "width 0.5s ease",
                  }} />
                </div>
                <div style={{ fontSize: "0.75rem", color: faithColor, marginTop: "0.25rem", fontWeight: 700 }}>
                  {faithScore}%
                </div>
              </div>
            </div>

            {/* Session Manifest badge */}
            <div style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "8px",
              padding: "0.75rem 1rem",
              marginBottom: "1.5rem",
              display: "flex",
              alignItems: "center",
              gap: "0.75rem",
              fontSize: "0.85rem",
            }}>
              <span style={{ color: "#8b5cf6", fontWeight: 700 }}>📋 Manifiesto de Sesión</span>
              <span style={{ color: "var(--text-muted)" }}>ID:</span>
              <code style={{ color: "var(--text)", fontFamily: "monospace", fontSize: "0.8rem" }}>
                {flashResult.manifest.session_id}
              </code>
              <span style={{ marginLeft: "auto", color: "var(--text-muted)" }}>
                {flashResult.pipeline_version}
              </span>
            </div>

            {/* Tabs */}
            <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setFlashActiveTab(tab.id)}
                  style={{
                    padding: "0.5rem 1rem",
                    borderRadius: "8px",
                    border: "none",
                    cursor: "pointer",
                    fontWeight: 600,
                    fontSize: "0.85rem",
                    background: flashActiveTab === tab.id
                      ? "linear-gradient(135deg, #3b82f6, #8b5cf6)"
                      : "var(--surface)",
                    color: flashActiveTab === tab.id ? "#fff" : "var(--text-muted)",
                    transition: "all 0.2s",
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "12px",
              padding: "1.5rem",
              minHeight: "300px",
            }}>
              {flashActiveTab === "synthesis" && (
                <pre style={{
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  fontFamily: "inherit",
                  fontSize: "0.9rem",
                  lineHeight: 1.7,
                  margin: 0,
                  color: "var(--text)",
                }}>
                  {flashResult.final_output}
                </pre>
              )}
              {flashActiveTab === "logic" && (
                <pre style={{
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  fontFamily: "inherit",
                  fontSize: "0.88rem",
                  lineHeight: 1.7,
                  margin: 0,
                  color: "var(--text)",
                }}>
                  {flashResult.logic_output}
                </pre>
              )}
              {flashActiveTab === "context" && (
                <pre style={{
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  fontFamily: "inherit",
                  fontSize: "0.88rem",
                  lineHeight: 1.7,
                  margin: 0,
                  color: "var(--text)",
                }}>
                  {flashResult.context_output}
                </pre>
              )}
              {flashActiveTab === "manifest" && (
                <div>
                  {[
                    { label: "Intent Original", value: flashResult.manifest.original_intent },
                    { label: "Resumen", value: flashResult.manifest.intent_summary },
                    { label: "Dominio", value: flashResult.manifest.domain },
                    { label: "Prioridad", value: flashResult.manifest.priority },
                    { label: "Sesión", value: flashResult.manifest.session_id },
                    { label: "Versión", value: flashResult.manifest.pipeline_version },
                  ].map((item) => (
                    <div key={item.label} style={{
                      display: "grid",
                      gridTemplateColumns: "160px 1fr",
                      gap: "0.5rem",
                      padding: "0.6rem 0",
                      borderBottom: "1px solid var(--border)",
                    }}>
                      <span style={{ color: "var(--text-muted)", fontSize: "0.85rem", fontWeight: 600 }}>
                        {item.label}
                      </span>
                      <span style={{ color: "var(--text)", fontSize: "0.9rem" }}>{item.value}</span>
                    </div>
                  ))}
                  <div style={{ marginTop: "1rem" }}>
                    <div style={{ fontWeight: 700, marginBottom: "0.75rem", fontSize: "0.9rem" }}>
                      Trazas de Agentes
                    </div>
                    {flashResult.manifest.traces.map((trace) => (
                      <div key={trace.agent_id} style={{
                        background: "var(--bg)",
                        border: "1px solid var(--border)",
                        borderRadius: "8px",
                        padding: "0.75rem 1rem",
                        marginBottom: "0.5rem",
                      }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.25rem" }}>
                          <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>{trace.display_name}</span>
                          <span style={{
                            padding: "0.15rem 0.5rem",
                            borderRadius: "4px",
                            fontSize: "0.75rem",
                            background: trace.status === "ok" ? "#22c55e20" : "#ef444420",
                            color: trace.status === "ok" ? "#22c55e" : "#ef4444",
                          }}>
                            {trace.status}
                          </span>
                        </div>
                        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                          {trace.latency_ms > 0 ? `${Math.round(trace.latency_ms)}ms · ` : ""}{trace.role}
                        </div>
                        {trace.output_preview && (
                          <div style={{
                            marginTop: "0.5rem",
                            fontSize: "0.8rem",
                            color: "var(--text)",
                            fontStyle: "italic",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}>
                            {trace.output_preview}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {!flashResult && !flashLoading && (
          <div style={{
            textAlign: "center",
            padding: "3rem 1rem",
            color: "var(--text-muted)",
          }}>
            <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>⚡</div>
            <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Pipeline Flash listo</div>
            <div style={{ fontSize: "0.9rem" }}>
              Escribe una solicitud arriba para activar los 3 agentes en paralelo
            </div>
          </div>
        )}

        {flashLoading && (
          <div style={{
            textAlign: "center",
            padding: "3rem 1rem",
            color: "var(--text-muted)",
          }}>
            <div style={{ fontSize: "2rem", marginBottom: "1rem", animation: "spin 1s linear infinite" }}>⚡</div>
            <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Agentes procesando en paralelo...</div>
            <div style={{ fontSize: "0.85rem", marginTop: "0.5rem" }}>
              Agente 1 (Lógica) + Agente 2 (Contexto) → Agente 3 (Síntesis)
            </div>
          </div>
        )}
      </div>
    );
  }

  function renderActiveSection() {
    switch (activeNav) {
      case "dashboard": return renderDashboard();
      case "assistant": return renderAssistant();
      case "agents": return renderAgents();
      case "folders": return renderFolders();
      case "subagents": return renderSubAgents();
      case "orchestrator": return renderOrchestrator();
      case "flash": return renderFlash();
      case "profiles": return renderProfiles();
      case "memory": return renderMemory();
      case "ethics": return renderEthics();
      case "database": return renderDatabase();
      default: return renderDashboard();
    }
  }

  const activeNavItem = NAV_ITEMS.find((n) => n.id === activeNav);

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-mark">
            <div className="sidebar-logo-icon">🧠</div>
            <div className="sidebar-logo-text">
              <span className="sidebar-logo-name">LangGraph</span>
              <span className="sidebar-logo-sub">Control Surface</span>
            </div>
          </div>
        </div>

        <div className="sidebar-section-label">Principal</div>
        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${activeNav === item.id ? "active" : ""}`}
              onClick={() => setActiveNav(item.id)}
              type="button"
            >
              <span className="nav-item-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-status-badge">
            <span className={`sidebar-status-dot ${connectionState}`} />
            <span>
              {connectionState === "online"
                ? "Sistema en línea"
                : connectionState === "degraded"
                ? "Sistema degradado"
                : connectionState === "offline"
                ? "Sin conexión"
                : "Verificando..."}
            </span>
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="main-area">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-breadcrumb">
            <span>Consola</span>
            <span>›</span>
            <strong>{activeNavItem?.label ?? "Dashboard"}</strong>
          </div>
          <div className="topbar-right">
            <span className={`status-chip ${connectionState}`}>
              <span className="status-chip-dot" />
              {connectionState === "online"
                ? `API en línea · ${healthState?.status ?? "ok"}`
                : connectionState === "degraded"
                ? "API degradada"
                : connectionState === "offline"
                ? "Sin conexión"
                : "Verificando..."}
            </span>
            <span className="status-chip">
              DB: {healthState?.db_path ?? "n/a"}
            </span>
            {lastHealthCheckAt && (
              <span className="status-chip">
                {lastHealthCheckAt}
              </span>
            )}
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => void refreshHealth(false)}
              disabled={isBusy}
              type="button"
            >
              🔄
            </button>
          </div>
        </header>

        {/* Page content */}
        {renderActiveSection()}

        {/* Status bar */}
        <footer className="status-bar">{statusText}</footer>
      </div>
    </div>
  );
}

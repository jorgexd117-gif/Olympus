import type {
  AgentAssignment,
  AgentRecord,
  AgentRunResponse,
  AssistantChatResponse,
  AvailableModelsResponse,
  ContextAcquisitionResult,
  EthicsCheckResult,
  EthicsPrinciples,
  FlashResult,
  Folder,
  HealthResponse,
  MemoryRecord,
  OrchestratorRunResponse,
  OrchestratorTranslateResponse,
  OrchestratorWorkflow,
  Profile,
  Project,
  SubAgentConfig,
  SubAgentPipelineResult,
  TeamRunResponse,
} from "./types";

const _rawEnvUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim().replace(/\/$/, "") ?? "";
const API_BASE_URL = _rawEnvUrl || "";
let LAST_SUCCESSFUL_API_BASE_URL = API_BASE_URL;

function apiCandidates(): string[] {
  return [API_BASE_URL];
}

function friendlyApiError(status: number, path: string, body: string): string {
  if (status === 404) {
    return `Endpoint no encontrado (${path}). Verifica que el backend esté actualizado.`;
  }
  if (status >= 500) {
    return `Error interno del backend (${status}). Intenta de nuevo en unos segundos.`;
  }
  return `Error API ${status} en ${path}: ${body || "sin detalle"}`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  for (const baseUrl of apiCandidates()) {
    let response: Response;
    try {
      response = await fetch(`${baseUrl}${path}`, {
        headers: {
          "Content-Type": "application/json",
          ...(init?.headers || {}),
        },
        ...init,
      });
    } catch {
      continue;
    }

    if (!response.ok) {
      const body = (await response.text()).slice(0, 280);
      throw new Error(friendlyApiError(response.status, path, body));
    }

    LAST_SUCCESSFUL_API_BASE_URL = baseUrl;
    return (await response.json()) as T;
  }
  throw new Error(
    `No se pudo conectar con la API${API_BASE_URL ? ` en ${API_BASE_URL}` : " (proxy local)"}. ` +
      "Confirma que el backend esté encendido."
  );
}

export function getApiBaseUrl(): string {
  return API_BASE_URL || "(proxy local)";
}

export function getResolvedApiBaseUrl(): string {
  return LAST_SUCCESSFUL_API_BASE_URL || "(proxy local)";
}

export function health(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export function listProjects(): Promise<Project[]> {
  return requestJson<Project[]>("/projects");
}

export function upsertProject(payload: {
  name: string;
  root_path: string;
  description: string;
}): Promise<Project> {
  return requestJson<Project>("/projects", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listProfiles(): Promise<Profile[]> {
  return requestJson<Profile[]>("/profiles");
}

export function updateProfile(
  agentKey: string,
  payload: {
    system_prompt?: string;
    model_name?: string;
    is_enabled?: boolean;
  }
): Promise<Profile> {
  return requestJson<Profile>(`/profiles/${agentKey}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function createProfile(payload: {
  agent_key: string;
  display_name: string;
  role: string;
  system_prompt: string;
  model_name: string;
  is_enabled?: boolean;
}): Promise<Profile> {
  return requestJson<Profile>("/profiles", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runAgent(payload: {
  user_prompt: string;
  project_id: number | null;
}): Promise<AgentRunResponse> {
  return requestJson<AgentRunResponse>("/agent/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function assistantChat(payload: {
  message: string;
  project_id: number | null;
}): Promise<AssistantChatResponse> {
  return requestJson<AssistantChatResponse>("/assistant/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listMemories(params: {
  project_id: number | null;
  limit?: number;
}): Promise<MemoryRecord[]> {
  const query = new URLSearchParams();
  if (params.project_id !== null) {
    query.set("project_id", String(params.project_id));
  }
  query.set("limit", String(params.limit ?? 20));
  return requestJson<MemoryRecord[]>(`/memories?${query.toString()}`);
}

export function availableModels(): Promise<AvailableModelsResponse> {
  return requestJson<AvailableModelsResponse>("/models/available");
}

export function listOrchestratorWorkflows(): Promise<OrchestratorWorkflow[]> {
  return requestJson<OrchestratorWorkflow[]>("/orchestrator/workflows");
}

export function runOrchestratorWorkflow(payload: {
  workflow_id: string;
  project_id: number | null;
  user_prompt: string;
}): Promise<OrchestratorRunResponse> {
  return requestJson<OrchestratorRunResponse>("/orchestrator/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runAgentTeam(payload: {
  user_prompt: string;
  project_id: number | null;
  agent_keys: string[];
}): Promise<TeamRunResponse> {
  return requestJson<TeamRunResponse>("/agents/team/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listAgents(): Promise<AgentRecord[]> {
  return requestJson<AgentRecord[]>("/api/agents");
}

export function createAgent(payload: {
  key: string;
  display_name: string;
  role: string;
  system_prompt?: string;
  model_name?: string;
  is_enabled?: boolean;
}): Promise<AgentRecord> {
  return requestJson<AgentRecord>("/api/agents", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateAgent(
  agentId: number,
  payload: {
    display_name?: string;
    role?: string;
    system_prompt?: string;
    model_name?: string;
    is_enabled?: boolean;
  }
): Promise<AgentRecord> {
  return requestJson<AgentRecord>(`/api/agents/${agentId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function listFolders(): Promise<Folder[]> {
  return requestJson<Folder[]>("/api/folders");
}

export function createFolder(payload: {
  name: string;
  description?: string;
  parent_id?: number | null;
}): Promise<Folder> {
  return requestJson<Folder>("/api/folders", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function deleteFolder(folderId: number): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>(`/api/folders/${folderId}`, {
    method: "DELETE",
  });
}

export function listFolderAssignments(folderId: number): Promise<AgentAssignment[]> {
  return requestJson<AgentAssignment[]>(`/api/folders/${folderId}/assignments`);
}

export function createAssignment(
  folderId: number,
  payload: { agent_id: number; process_type: string }
): Promise<AgentAssignment> {
  return requestJson<AgentAssignment>(`/api/folders/${folderId}/assignments`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function deleteAssignment(assignmentId: number): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>(`/api/assignments/${assignmentId}`, {
    method: "DELETE",
  });
}

export function runSubAgentPipeline(payload: {
  user_prompt: string;
  project_id: number | null;
  pipeline_type?: string | null;
}): Promise<SubAgentPipelineResult> {
  return requestJson<SubAgentPipelineResult>("/api/subagents/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getSubAgentConfigs(): Promise<SubAgentConfig[]> {
  return requestJson<SubAgentConfig[]>("/api/subagents/configs");
}

export function getPipelineTemplates(): Promise<Record<string, string[]>> {
  return requestJson<Record<string, string[]>>("/api/subagents/pipelines");
}

export function getEthicsPrinciples(): Promise<EthicsPrinciples> {
  return requestJson<EthicsPrinciples>("/api/ethics/principles");
}

export function checkEthics(payload: {
  text: string;
  check_type?: string;
}): Promise<EthicsCheckResult> {
  return requestJson<EthicsCheckResult>("/api/ethics/check", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function acquireContext(payload: {
  query: string;
  project_id?: number | null;
}): Promise<ContextAcquisitionResult> {
  return requestJson<ContextAcquisitionResult>("/api/context/acquire", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function queryDatabase(sql: string): Promise<{ result: string }> {
  return requestJson<{ result: string }>("/api/query-db", {
    method: "POST",
    body: JSON.stringify({ sql }),
  });
}

export function getMemoryStats(projectId?: number | null): Promise<import("./types").MemoryStats> {
  const qs = projectId ? `?project_id=${projectId}` : "";
  return requestJson(`/api/memory/stats${qs}`);
}

export function learnMemory(content: string, topic: string, projectId?: number | null): Promise<{ ok: boolean; message: string }> {
  return requestJson("/api/memory/learn", {
    method: "POST",
    body: JSON.stringify({ content, topic, project_id: projectId ?? null }),
  });
}

export function translatePrompt(
  prompt: string,
  projectId?: number | null
): Promise<OrchestratorTranslateResponse> {
  return requestJson<OrchestratorTranslateResponse>("/api/orchestrator/translate", {
    method: "POST",
    body: JSON.stringify({ prompt, project_id: projectId ?? null }),
  });
}

export function runFlashPipeline(
  prompt: string,
  projectId?: number | null,
  multimodalInputs?: unknown[]
): Promise<FlashResult> {
  return requestJson<FlashResult>("/api/flash/run", {
    method: "POST",
    body: JSON.stringify({
      prompt,
      project_id: projectId ?? null,
      multimodal_inputs: multimodalInputs ?? [],
    }),
  });
}


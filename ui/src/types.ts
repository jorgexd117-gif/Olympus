export type HealthResponse = {
  status: string;
  db_path: string;
  db_backend?: string;
  db_notice?: string | null;
};

export type Project = {
  id: number;
  name: string;
  root_path: string;
  description: string;
  created_at: string;
};

export type Profile = {
  id: number;
  agent_key: string;
  display_name: string;
  role: string;
  system_prompt: string;
  model_name: string;
  is_enabled: boolean;
  created_at: string;
  updated_at: string;
};

export type MemoryRecord = {
  id: number;
  project_id: number | null;
  memory_type: string;
  content: string;
  metadata: Record<string, unknown>;
  relevance: number;
  created_at: string;
};

export type MemoryStats = {
  total_memories: number;
  total_conversations: number;
  unique_topics: number;
  knowledge_level: number;
  top_topics: { topic: string; count: number }[];
  level_label: string;
};

export type AgentRunResponse = {
  sections: Record<string, string>;
  final_output: string;
};

export type AssistantChatResponse = {
  reply: string;
  source: string;
  project_id: number | null;
  sections: Record<string, string>;
  machine_translation: Record<string, unknown>;
};

export type AvailableModelsResponse = {
  models: string[];
};

export type OrchestratorWorkflow = {
  workflow_id: string;
  name: string;
  description: string;
  steps: string[];
};

export type OrchestratorStepResult = {
  step: string;
  status: string;
  detail: string;
  duration_ms: number;
};

export type OrchestratorRunResponse = {
  workflow_id: string;
  status: string;
  summary: string;
  steps: OrchestratorStepResult[];
  output: Record<string, unknown>;
};

export type TeamRunStep = {
  agent_key: string;
  display_name: string;
  role: string;
  model_name: string;
  status: string;
  output: string;
};

export type TeamRunResponse = {
  project_id: number | null;
  user_prompt: string;
  final_output: string;
  steps: TeamRunStep[];
};

export type AgentRecord = {
  id: number;
  key: string;
  display_name: string;
  role: string;
  system_prompt: string;
  model_name: string;
  is_enabled: boolean;
  created_at: string;
  updated_at: string;
};

export type AgentAssignment = {
  id: number;
  folder_id: number;
  agent_id: number;
  agent_key: string;
  agent_display_name: string;
  process_type: "planning" | "thinking" | "action";
  created_at: string;
};

export type Folder = {
  id: number;
  name: string;
  description: string;
  parent_id: number | null;
  created_at: string;
  children: Folder[];
  assignments: AgentAssignment[];
};

export type SubAgentConfig = {
  role: string;
  display_name: string;
  description: string;
  capabilities: string[];
  model_name: string;
};

export type SubAgentStep = {
  role: string;
  display_name: string;
  status: string;
  output: string;
  confidence: number;
};

export type SubAgentPipelineResult = {
  task: string;
  final_output: string;
  steps: SubAgentStep[];
  pipeline_type: string;
  total_agents: number;
  successful_agents: number;
  ethics_blocked?: boolean;
};

export type EthicsPrinciples = {
  principles: Record<string, string>;
  audit_summary: {
    total_checks: number;
    blocked: number;
    warned: number;
    passed: number;
  };
};

export type EthicsCheckResult = {
  is_safe: boolean;
  violations: Array<{ type: string; detail: string }>;
  warnings: string[];
  applied_rules: string[];
};

export type ContextSource = {
  source_type: string;
  label: string;
  content: string;
  relevance: number;
};

export type ContextAcquisitionResult = {
  query: string;
  confidence: number;
  needs_human_input: boolean;
  human_question: string;
  summary: string;
  sources: ContextSource[];
};

export type ToolActivation = {
  tool_name: string;
  reason: string;
};

export type TranslatedPrompt = {
  objective: string;
  intent: string;
  parameters: Record<string, unknown>;
  tools: ToolActivation[];
  expected_output_format: string;
  priority: "low" | "medium" | "high" | "critical";
  sub_tasks: string[];
  ambiguity_score: number;
  clarification_question: string | null;
  language: string;
  raw_prompt: string;
};

export type OrchestratorTranslateResponse = {
  translated: TranslatedPrompt;
  machine_ir: Record<string, unknown>;
};

export type EthicsPrinciple = {
  id?: string;
  description?: string;
  rule?: string;
  category?: string;
};

export type EthicsAuditSummary = {
  total_checks?: number;
  blocked?: number;
  allowed?: number;
  last_check?: string | null;
};

export type EthicsPrinciplesResponse = {
  principles: EthicsPrinciple[] | string[] | Record<string, unknown>;
  audit_summary: EthicsAuditSummary | Record<string, unknown>;
};

export type FlashAgentTrace = {
  agent_id: string;
  display_name: string;
  role: string;
  status: string;
  latency_ms: number;
  output_preview: string;
};

export type FlashFaithfulness = {
  is_faithful: boolean;
  score: number;
  deviations: string[];
  corrections: string[];
};

export type FlashManifest = {
  session_id: string;
  created_at: string;
  original_intent: string;
  intent_summary: string;
  domain: string;
  priority: string;
  total_latency_ms: number;
  pipeline_version: string;
  traces: FlashAgentTrace[];
};

export type FlashResult = {
  session_id: string;
  final_output: string;
  logic_output: string;
  context_output: string;
  synthesis_output: string;
  faithfulness: FlashFaithfulness;
  manifest: FlashManifest;
  total_latency_ms: number;
  agent_latencies: Record<string, number>;
  pipeline_version: string;
};

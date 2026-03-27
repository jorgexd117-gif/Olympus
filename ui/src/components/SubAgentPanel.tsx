import { useState } from "react";
import type { SubAgentConfig, SubAgentPipelineResult, EthicsPrinciples } from "../types";
import {
  runSubAgentPipeline,
  getSubAgentConfigs,
  getPipelineTemplates,
  getEthicsPrinciples,
  queryDatabase,
} from "../api";

type Props = {
  projectId: number | null;
};

export default function SubAgentPanel({ projectId }: Props) {
  const [prompt, setPrompt] = useState("");
  const [pipelineType, setPipelineType] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SubAgentPipelineResult | null>(null);
  const [configs, setConfigs] = useState<SubAgentConfig[]>([]);
  const [pipelines, setPipelines] = useState<Record<string, string[]>>({});
  const [ethics, setEthics] = useState<EthicsPrinciples | null>(null);
  const [showEthics, setShowEthics] = useState(false);
  const [showConfigs, setShowConfigs] = useState(false);
  const [dbQuery, setDbQuery] = useState("");
  const [dbResult, setDbResult] = useState("");
  const [dbRunning, setDbRunning] = useState(false);
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  async function loadConfigs() {
    if (configs.length > 0) {
      setShowConfigs(!showConfigs);
      return;
    }
    try {
      const [c, p] = await Promise.all([getSubAgentConfigs(), getPipelineTemplates()]);
      setConfigs(c);
      setPipelines(p);
      setShowConfigs(true);
    } catch {}
  }

  async function loadEthics() {
    if (ethics) {
      setShowEthics(!showEthics);
      return;
    }
    try {
      const e = await getEthicsPrinciples();
      setEthics(e);
      setShowEthics(true);
    } catch {}
  }

  async function handleRun() {
    if (!prompt.trim() || running) return;
    setRunning(true);
    setResult(null);
    setExpandedStep(null);
    try {
      const r = await runSubAgentPipeline({
        user_prompt: prompt.trim(),
        project_id: projectId,
        pipeline_type: pipelineType || null,
      });
      setResult(r);
    } catch (err) {
      setResult({
        task: prompt,
        final_output: `Error: ${err instanceof Error ? err.message : String(err)}`,
        steps: [],
        pipeline_type: pipelineType || "auto",
        total_agents: 0,
        successful_agents: 0,
      });
    } finally {
      setRunning(false);
    }
  }

  async function handleDbQuery() {
    if (!dbQuery.trim() || dbRunning) return;
    setDbRunning(true);
    setDbResult("");
    try {
      const r = await queryDatabase(dbQuery.trim());
      setDbResult(r.result);
    } catch (err) {
      setDbResult(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setDbRunning(false);
    }
  }

  return (
    <section className="panel subagent-panel">
      <div className="panel-header">
        <h3>Sub-Agentes y Herramientas</h3>
        <div className="panel-header-actions">
          <button className="btn btn-sm" onClick={loadConfigs}>
            {showConfigs ? "Ocultar Agentes" : "Ver Agentes"}
          </button>
          <button className="btn btn-sm" onClick={loadEthics}>
            {showEthics ? "Ocultar Etica" : "Marco Etico"}
          </button>
        </div>
      </div>

      {showConfigs && configs.length > 0 && (
        <div className="subagent-configs">
          <h4>Sub-agentes disponibles</h4>
          <div className="config-grid">
            {configs.map((c) => (
              <div key={c.role} className="config-card">
                <div className="config-role">{c.display_name}</div>
                <div className="config-desc">{c.description}</div>
                <div className="config-caps">
                  {c.capabilities.map((cap) => (
                    <span key={cap} className="cap-tag">{cap}</span>
                  ))}
                </div>
                <div className="config-model">{c.model_name}</div>
              </div>
            ))}
          </div>
          {Object.keys(pipelines).length > 0 && (
            <div className="pipeline-templates">
              <h4>Pipelines disponibles</h4>
              {Object.entries(pipelines).map(([name, roles]) => (
                <div key={name} className="pipeline-item">
                  <span className="pipeline-name">{name}</span>
                  <span className="pipeline-roles">{roles.join(" → ")}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {showEthics && ethics && (
        <div className="ethics-panel">
          <h4>Principios Eticos</h4>
          <div className="ethics-principles">
            {Object.entries(ethics.principles).map(([key, desc]) => (
              <div key={key} className="ethics-item">
                <span className="ethics-key">{key.replace(/_/g, " ")}</span>
                <span className="ethics-desc">{desc}</span>
              </div>
            ))}
          </div>
          <div className="ethics-audit">
            <span>Verificaciones: {ethics.audit_summary.total_checks}</span>
            <span>Bloqueadas: {ethics.audit_summary.blocked}</span>
            <span>Advertencias: {ethics.audit_summary.warned}</span>
            <span>Aprobadas: {ethics.audit_summary.passed}</span>
          </div>
        </div>
      )}

      <div className="subagent-form">
        <div className="subagent-input-row">
          <select
            className="pipeline-select"
            value={pipelineType}
            onChange={(e) => setPipelineType(e.target.value)}
          >
            <option value="">Auto-detectar pipeline</option>
            <option value="full_analysis">Analisis completo</option>
            <option value="code_task">Tarea de codigo</option>
            <option value="research">Investigacion</option>
            <option value="quick_answer">Respuesta rapida</option>
            <option value="execute">Ejecucion</option>
          </select>
        </div>
        <textarea
          className="subagent-textarea"
          placeholder="Describe la tarea para los sub-agentes..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
        />
        <button
          className="btn"
          onClick={handleRun}
          disabled={running || !prompt.trim()}
        >
          {running ? "Ejecutando pipeline..." : "Ejecutar Sub-Agentes"}
        </button>
      </div>

      {result && (
        <div className="subagent-result">
          <div className="result-header">
            <span className="result-pipeline">Pipeline: {result.pipeline_type}</span>
            <span className="result-stats">
              {result.successful_agents}/{result.total_agents} agentes exitosos
            </span>
          </div>

          {result.ethics_blocked && (
            <div className="ethics-blocked-notice">
              Solicitud bloqueada por el marco etico
            </div>
          )}

          {result.steps.length > 0 && (
            <div className="subagent-steps">
              {result.steps.map((step, i) => (
                <div key={i} className="step-card">
                  <div
                    className="step-header"
                    onClick={() => setExpandedStep(expandedStep === i ? null : i)}
                  >
                    <span className={`step-status-dot ${step.status}`} />
                    <span className="step-name">{step.display_name}</span>
                    <span className="step-role">({step.role})</span>
                    <span className="step-confidence">
                      {(step.confidence * 100).toFixed(0)}%
                    </span>
                    <span className="step-expand">{expandedStep === i ? "▾" : "▸"}</span>
                  </div>
                  {expandedStep === i && (
                    <pre className="step-output">{step.output}</pre>
                  )}
                </div>
              ))}
            </div>
          )}

          <div className="final-output">
            <h4>Resultado Final</h4>
            <pre>{result.final_output}</pre>
          </div>
        </div>
      )}

      <div className="db-query-section">
        <h4>Consulta a Base de Datos</h4>
        <div className="db-query-form">
          <textarea
            className="db-query-input"
            placeholder="SELECT * FROM agents LIMIT 10"
            value={dbQuery}
            onChange={(e) => setDbQuery(e.target.value)}
            rows={2}
          />
          <button
            className="btn btn-sm"
            onClick={handleDbQuery}
            disabled={dbRunning || !dbQuery.trim()}
          >
            {dbRunning ? "Ejecutando..." : "Ejecutar SQL"}
          </button>
        </div>
        {dbResult && <pre className="db-query-result">{dbResult}</pre>}
      </div>
    </section>
  );
}

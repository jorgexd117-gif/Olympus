import { FormEvent, useState } from "react";
import type { AgentRecord } from "../types";
import { createAgent, updateAgent } from "../api";

type Props = {
  agents: AgentRecord[];
  onRefresh: () => void;
};

export default function AgentList({ agents, onRefresh }: Props) {
  const [showForm, setShowForm] = useState(false);
  const [key, setKey] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [role, setRole] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [modelName, setModelName] = useState("");
  const [saving, setSaving] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editEnabled, setEditEnabled] = useState<boolean>(true);

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    if (!key || !displayName || !role) return;
    setSaving(true);
    try {
      await createAgent({ key, display_name: displayName, role, system_prompt: systemPrompt, model_name: modelName });
      setKey("");
      setDisplayName("");
      setRole("");
      setSystemPrompt("");
      setModelName("");
      setShowForm(false);
      onRefresh();
    } finally {
      setSaving(false);
    }
  }

  async function toggleEnabled(agent: AgentRecord) {
    await updateAgent(agent.id, { is_enabled: !agent.is_enabled });
    onRefresh();
  }

  return (
    <div className="panel agent-list-panel">
      <div className="panel-header">
        <h3>Agentes Disponibles</h3>
        <button className="btn btn-sm" onClick={() => setShowForm(!showForm)}>
          {showForm ? "Cancelar" : "+ Nuevo Agente"}
        </button>
      </div>

      {showForm && (
        <form className="agent-form" onSubmit={handleCreate}>
          <input placeholder="Clave (ej: planner)" value={key} onChange={(e) => setKey(e.target.value)} required />
          <input placeholder="Nombre" value={displayName} onChange={(e) => setDisplayName(e.target.value)} required />
          <input placeholder="Rol" value={role} onChange={(e) => setRole(e.target.value)} required />
          <input placeholder="Modelo (ej: gpt-4o-mini)" value={modelName} onChange={(e) => setModelName(e.target.value)} />
          <textarea placeholder="Prompt del sistema" value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} rows={3} />
          <button className="btn" type="submit" disabled={saving}>{saving ? "Creando..." : "Crear Agente"}</button>
        </form>
      )}

      <div className="agent-grid">
        {agents.length === 0 && <p className="empty-text">No hay agentes creados.</p>}
        {agents.map((agent) => (
          <div key={agent.id} className={`agent-card ${agent.is_enabled ? "" : "disabled"}`}>
            <div className="agent-card-header">
              <strong>{agent.display_name}</strong>
              <span className={`status-dot ${agent.is_enabled ? "active" : "inactive"}`} />
            </div>
            <div className="agent-card-meta">
              <span className="agent-key">{agent.key}</span>
              <span className="agent-role">{agent.role}</span>
            </div>
            {agent.model_name && <div className="agent-model">{agent.model_name}</div>}
            <div className="agent-card-actions">
              <button className="btn btn-xs" onClick={() => toggleEnabled(agent)}>
                {agent.is_enabled ? "Desactivar" : "Activar"}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

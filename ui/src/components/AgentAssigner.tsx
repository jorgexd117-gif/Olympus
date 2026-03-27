import { useState } from "react";
import type { AgentRecord, Folder } from "../types";
import { createAssignment, deleteAssignment } from "../api";

type Props = {
  folder: Folder;
  agents: AgentRecord[];
  onRefresh: () => void;
};

const PROCESS_TYPES = [
  { key: "planning", label: "Planificacion" },
  { key: "thinking", label: "Pensamiento" },
  { key: "action", label: "Accion" },
] as const;

export default function AgentAssigner({ folder, agents, onRefresh }: Props) {
  const [saving, setSaving] = useState(false);

  function getAssignment(processType: string) {
    return folder.assignments?.find((a) => a.process_type === processType) ?? null;
  }

  async function handleAssign(processType: string, agentId: number) {
    if (!agentId) return;
    const existing = getAssignment(processType);
    setSaving(true);
    try {
      if (existing) {
        await deleteAssignment(existing.id);
      }
      await createAssignment(folder.id, { agent_id: agentId, process_type: processType });
      onRefresh();
    } finally {
      setSaving(false);
    }
  }

  async function handleRemove(assignmentId: number) {
    setSaving(true);
    try {
      await deleteAssignment(assignmentId);
      onRefresh();
    } finally {
      setSaving(false);
    }
  }

  const enabledAgents = agents.filter((a) => a.is_enabled);

  return (
    <div className="assigner-container">
      <h4>Asignar Agentes a Procesos</h4>
      <div className="assignment-slots">
        {PROCESS_TYPES.map(({ key, label }) => {
          const assignment = getAssignment(key);
          return (
            <div key={key} className="assignment-slot">
              <div className="slot-label">{label}</div>
              {assignment ? (
                <div className="slot-assigned">
                  <span className="assigned-agent-name">{assignment.agent_display_name}</span>
                  <button
                    className="btn btn-xs btn-danger"
                    onClick={() => handleRemove(assignment.id)}
                    disabled={saving}
                  >
                    Quitar
                  </button>
                </div>
              ) : (
                <select
                  className="slot-select"
                  value=""
                  onChange={(e) => handleAssign(key, Number(e.target.value))}
                  disabled={saving}
                >
                  <option value="">Seleccionar agente...</option>
                  {enabledAgents.map((agent) => (
                    <option key={agent.id} value={agent.id}>
                      {agent.display_name} ({agent.role})
                    </option>
                  ))}
                </select>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

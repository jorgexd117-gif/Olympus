import { FormEvent, useState } from "react";
import type { AgentRecord, Folder } from "../types";
import { createFolder, deleteFolder } from "../api";
import AgentAssigner from "./AgentAssigner";

type Props = {
  folders: Folder[];
  agents: AgentRecord[];
  onRefresh: () => void;
};

export default function FolderManager({ folders, agents, onRefresh }: Props) {
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [parentId, setParentId] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    if (!name) return;
    setSaving(true);
    try {
      await createFolder({ name, description, parent_id: parentId });
      setName("");
      setDescription("");
      setParentId(null);
      setShowForm(false);
      onRefresh();
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(folderId: number) {
    if (!confirm("Eliminar esta carpeta y sus asignaciones?")) return;
    await deleteFolder(folderId);
    if (expandedId === folderId) setExpandedId(null);
    onRefresh();
  }

  function flatFolders(items: Folder[]): { id: number; name: string }[] {
    const result: { id: number; name: string }[] = [];
    for (const f of items) {
      result.push({ id: f.id, name: f.name });
      if (f.children) result.push(...flatFolders(f.children));
    }
    return result;
  }

  function renderFolder(folder: Folder, depth: number = 0) {
    const isExpanded = expandedId === folder.id;
    return (
      <div key={folder.id} className="folder-item" style={{ marginLeft: depth * 16 }}>
        <div className="folder-row" onClick={() => setExpandedId(isExpanded ? null : folder.id)}>
          <span className="folder-icon">{isExpanded ? "📂" : "📁"}</span>
          <span className="folder-name">{folder.name}</span>
          {folder.description && <span className="folder-desc">{folder.description}</span>}
          <button className="btn btn-xs btn-danger" onClick={(e) => { e.stopPropagation(); handleDelete(folder.id); }}>
            Eliminar
          </button>
        </div>
        {isExpanded && (
          <div className="folder-detail">
            <AgentAssigner folder={folder} agents={agents} onRefresh={onRefresh} />
          </div>
        )}
        {folder.children?.map((child) => renderFolder(child, depth + 1))}
      </div>
    );
  }

  const allFolders = flatFolders(folders);

  return (
    <div className="panel folder-panel">
      <div className="panel-header">
        <h3>Carpetas</h3>
        <button className="btn btn-sm" onClick={() => setShowForm(!showForm)}>
          {showForm ? "Cancelar" : "+ Nueva Carpeta"}
        </button>
      </div>

      {showForm && (
        <form className="folder-form" onSubmit={handleCreate}>
          <input placeholder="Nombre de carpeta" value={name} onChange={(e) => setName(e.target.value)} required />
          <input placeholder="Descripcion (opcional)" value={description} onChange={(e) => setDescription(e.target.value)} />
          <select value={parentId ?? ""} onChange={(e) => setParentId(e.target.value ? Number(e.target.value) : null)}>
            <option value="">Sin carpeta padre</option>
            {allFolders.map((f) => (
              <option key={f.id} value={f.id}>{f.name}</option>
            ))}
          </select>
          <button className="btn" type="submit" disabled={saving}>{saving ? "Creando..." : "Crear Carpeta"}</button>
        </form>
      )}

      <div className="folder-tree">
        {folders.length === 0 && <p className="empty-text">No hay carpetas.</p>}
        {folders.map((folder) => renderFolder(folder))}
      </div>
    </div>
  );
}

"""External visual control center for human-friendly multi-agent operation."""

from __future__ import annotations

import os
import threading
import tkinter as tk
import urllib.error
import urllib.request
import json
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from dotenv import load_dotenv

from src.multi_agent import AssistantTurnResult, MultiAgentCoordinator
from src.persistence import AgentDatabase
from src.thermal import get_thermal_regulator


class ControlCenterApp:
    """Desktop application to manage prompts, projects, memory, and execution."""

    def __init__(self) -> None:
        load_dotenv()
        load_dotenv(".env.local", override=True)
        db_path = os.getenv("AGENT_DB_PATH", "data/agent_memory.db")
        self.database = AgentDatabase(db_path=db_path)
        self.coordinator = MultiAgentCoordinator(self.database)
        self.thermal = get_thermal_regulator()
        self.thermal.start()

        self.root = tk.Tk()
        self.root.title("Agent Control Center")
        self.root.geometry("1440x900")
        self.root.minsize(1200, 760)
        self.root.configure(bg="#ffffff")

        self.current_project_id: int | None = None
        self.project_ids_by_index: list[int] = []
        self.profile_prompt_widgets: dict[str, tk.Text] = {}
        self.profile_model_vars: dict[str, tk.StringVar] = {}
        self.profile_model_selectors: dict[str, ttk.Combobox] = {}

        self.is_running_agent = False
        self.is_running_command = False
        self.thermal_status_var = tk.StringVar(value="Thermal monitor starting...")

        self._build_ui()
        self._load_projects()
        self._refresh_model_options()
        self._load_profiles()
        self._refresh_memory_panel()
        self._refresh_thermal_panel()

        self._append_chat(
            "System",
            (
                "Asistente listo. Puedes escribir en lenguaje natural para ajustar el sistema "
                "o pedir una respuesta tecnica. Usa 'comando: <instruccion>' para ejecutar "
                "comandos en el proyecto activo."
            ),
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabelframe", background="#ffffff", foreground="#111827")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#111827")
        style.configure("TLabel", background="#ffffff", foreground="#111827")
        style.configure("TButton", padding=6)
        style.configure("TNotebook", background="#ffffff")
        style.configure("TNotebook.Tab", padding=(10, 6))

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(main, width=320)
        center = ttk.Frame(main, width=740)
        right = ttk.Frame(main, width=360)

        main.add(left, weight=1)
        main.add(center, weight=3)
        main.add(right, weight=2)

        self._build_left_panel(left)
        self._build_center_panel(center)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        projects_box = ttk.LabelFrame(parent, text="Projects", padding=8)
        projects_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.project_list = tk.Listbox(
            projects_box,
            exportselection=False,
            bg="#ffffff",
            fg="#111827",
            selectbackground="#dbeafe",
            selectforeground="#111827",
            activestyle="none",
            height=12,
        )
        self.project_list.pack(fill=tk.BOTH, expand=True)
        self.project_list.bind("<<ListboxSelect>>", self._on_project_select)

        form = ttk.Frame(projects_box)
        form.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Label(form, text="Path").grid(row=1, column=0, sticky="w")
        ttk.Label(form, text="Desc").grid(row=2, column=0, sticky="w")

        self.project_name_var = tk.StringVar()
        self.project_path_var = tk.StringVar()
        self.project_desc_var = tk.StringVar()

        ttk.Entry(form, textvariable=self.project_name_var).grid(
            row=0, column=1, sticky="ew", padx=(6, 0), pady=2
        )
        ttk.Entry(form, textvariable=self.project_path_var).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=2
        )
        ttk.Entry(form, textvariable=self.project_desc_var).grid(
            row=2, column=1, sticky="ew", padx=(6, 0), pady=2
        )
        form.columnconfigure(1, weight=1)

        buttons = ttk.Frame(projects_box)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Browse", command=self._browse_project_path).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(buttons, text="Save Project", command=self._save_project).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(buttons, text="Refresh", command=self._load_projects).pack(side=tk.LEFT)

        command_box = ttk.LabelFrame(parent, text="Project Command Runner", padding=8)
        command_box.pack(fill=tk.BOTH, expand=True)

        self.command_var = tk.StringVar()
        cmd_row = ttk.Frame(command_box)
        cmd_row.pack(fill=tk.X)
        ttk.Entry(cmd_row, textvariable=self.command_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.command_button = ttk.Button(cmd_row, text="Run", command=self._run_command)
        self.command_button.pack(side=tk.LEFT, padx=(6, 0))

        self.command_output = ScrolledText(
            command_box,
            wrap=tk.WORD,
            height=14,
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
        )
        self.command_output.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.command_output.insert(
            tk.END,
            "Selecciona un proyecto y ejecuta comandos para trabajar entre repos.\n",
        )
        self.command_output.configure(state=tk.DISABLED)

    def _build_center_panel(self, parent: ttk.Frame) -> None:
        control_box = ttk.LabelFrame(parent, text="Prompt Control", padding=8)
        control_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            control_box,
            text=(
                "Ejemplos: 'usar proyecto demo', "
                "'agente pensamiento: ...', "
                "'crear proyecto api en /ruta/proyecto', "
                "'comando: pytest -q'"
            ),
        ).pack(anchor="w")

        control_row = ttk.Frame(control_box)
        control_row.pack(fill=tk.X, pady=(6, 0))
        self.control_prompt_var = tk.StringVar()
        ttk.Entry(control_row, textvariable=self.control_prompt_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(control_row, text="Apply", command=self._apply_control_prompt).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        chat_box = ttk.LabelFrame(parent, text="Agent Workspace", padding=8)
        chat_box.pack(fill=tk.BOTH, expand=True)

        self.chat_log = ScrolledText(
            chat_box,
            wrap=tk.WORD,
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
        )
        self.chat_log.pack(fill=tk.BOTH, expand=True)
        self.chat_log.configure(state=tk.DISABLED)

        send_box = ttk.Frame(chat_box)
        send_box.pack(fill=tk.X, pady=(8, 0))

        self.user_prompt = tk.Text(
            send_box,
            height=4,
            wrap=tk.WORD,
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
        )
        self.user_prompt.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_prompt.bind("<Control-Return>", lambda _e: self._send_user_prompt())

        self.send_button = ttk.Button(send_box, text="Send Prompt", command=self._send_user_prompt)
        self.send_button.pack(side=tk.LEFT, padx=(8, 0))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        thermal_box = ttk.LabelFrame(parent, text="Thermal Regulator", padding=8)
        thermal_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(thermal_box, textvariable=self.thermal_status_var, justify="left").pack(
            anchor="w"
        )

        profile_box = ttk.LabelFrame(parent, text="Multi-Agent Profiles", padding=8)
        profile_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        models_row = ttk.Frame(profile_box)
        models_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(models_row, text="Available models").pack(side=tk.LEFT)
        ttk.Button(
            models_row,
            text="Refresh Models",
            command=self._refresh_model_options,
        ).pack(side=tk.RIGHT)

        self.profile_notebook = ttk.Notebook(profile_box)
        self.profile_notebook.pack(fill=tk.BOTH, expand=True)
        self._build_profile_tab("thought", "Thought Agent")
        self._build_profile_tab("review", "Review Agent")
        self._build_profile_tab("action", "Action Agent")

        memory_box = ttk.LabelFrame(parent, text="Project Memory", padding=8)
        memory_box.pack(fill=tk.BOTH, expand=True)
        self.memory_text = ScrolledText(
            memory_box,
            wrap=tk.WORD,
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
        )
        self.memory_text.pack(fill=tk.BOTH, expand=True)
        self.memory_text.configure(state=tk.DISABLED)

    def _build_profile_tab(self, agent_key: str, title: str) -> None:
        tab = ttk.Frame(self.profile_notebook, padding=8)
        self.profile_notebook.add(tab, text=title)

        ttk.Label(tab, text="Model").pack(anchor="w")
        model_var = tk.StringVar()
        self.profile_model_vars[agent_key] = model_var
        selector = ttk.Combobox(
            tab,
            textvariable=model_var,
            state="readonly",
            values=(),
        )
        selector.pack(fill=tk.X, pady=(2, 8))
        self.profile_model_selectors[agent_key] = selector

        ttk.Label(tab, text="System Prompt").pack(anchor="w")
        prompt_widget = tk.Text(
            tab,
            wrap=tk.WORD,
            height=12,
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
        )
        prompt_widget.pack(fill=tk.BOTH, expand=True)
        self.profile_prompt_widgets[agent_key] = prompt_widget

        ttk.Button(tab, text="Save Profile", command=lambda key=agent_key: self._save_profile(key)).pack(
            pady=(8, 0), anchor="e"
        )

    def _append_chat(self, role: str, text: str) -> None:
        self.chat_log.configure(state=tk.NORMAL)
        self.chat_log.insert(tk.END, f"{role}:\n{text}\n\n")
        self.chat_log.see(tk.END)
        self.chat_log.configure(state=tk.DISABLED)

    def _append_command_output(self, text: str) -> None:
        self.command_output.configure(state=tk.NORMAL)
        self.command_output.insert(tk.END, text + "\n")
        self.command_output.see(tk.END)
        self.command_output.configure(state=tk.DISABLED)

    def _set_agent_busy(self, busy: bool) -> None:
        self.is_running_agent = busy
        self.send_button.configure(state=tk.DISABLED if busy else tk.NORMAL)

    def _set_command_busy(self, busy: bool) -> None:
        self.is_running_command = busy
        self.command_button.configure(state=tk.DISABLED if busy else tk.NORMAL)

    def _browse_project_path(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.project_path_var.set(selected)

    def _save_project(self) -> None:
        name = self.project_name_var.get().strip()
        path = self.project_path_var.get().strip()
        description = self.project_desc_var.get().strip()
        if not name or not path:
            messagebox.showwarning("Missing fields", "Name and Path are required.")
            return
        try:
            project = self.coordinator.ensure_project(name=name, root_path=path, description=description)
            self.current_project_id = project.id
            self._append_chat("System", f"Proyecto guardado: {project.name} -> {project.root_path}")
            self._load_projects()
        except Exception as exc:  # pragma: no cover - UI branch
            messagebox.showerror("Project error", str(exc))

    def _load_projects(self) -> None:
        projects = self.coordinator.list_projects()
        self.project_ids_by_index = [project.id for project in projects]
        self.project_list.delete(0, tk.END)
        selected_index = None

        for idx, project in enumerate(projects):
            label = f"{project.name}  |  {project.root_path}"
            self.project_list.insert(tk.END, label)
            if self.current_project_id == project.id:
                selected_index = idx

        if selected_index is None and projects and self.current_project_id is None:
            self.current_project_id = projects[0].id
            selected_index = 0

        if selected_index is not None:
            self.project_list.selection_clear(0, tk.END)
            self.project_list.selection_set(selected_index)
            self.project_list.activate(selected_index)

    def _on_project_select(self, _event: object) -> None:
        selection = self.project_list.curselection()
        if not selection:
            return
        index = selection[0]
        if index >= len(self.project_ids_by_index):
            return
        self.current_project_id = self.project_ids_by_index[index]
        project = self.database.get_project(self.current_project_id)
        if not project:
            return
        self.project_name_var.set(project.name)
        self.project_path_var.set(project.root_path)
        self.project_desc_var.set(project.description)
        self._append_chat("System", f"Proyecto activo: {project.name}")
        self._refresh_memory_panel()

    def _fetch_ollama_models(self) -> list[str]:
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

    def _refresh_model_options(self) -> None:
        discovered = set()

        for env_key in (
            "FREE_THOUGHT_MODEL",
            "FREE_REVIEW_MODEL",
            "FREE_ACTION_MODEL",
            "OPENAI_MODEL",
            "ANTHROPIC_MODEL",
            "DEEPSEEK_MODEL",
        ):
            value = os.getenv(env_key, "").strip()
            if value:
                discovered.add(value)

        for profile in self.coordinator.get_profiles().values():
            model_name = profile.model_name.strip()
            if model_name:
                discovered.add(model_name)

        for model in self._fetch_ollama_models():
            discovered.add(model)

        sorted_models = sorted(discovered, key=str.lower)

        for agent_key, selector in self.profile_model_selectors.items():
            current_value = self.profile_model_vars[agent_key].get().strip()
            if current_value and current_value not in discovered:
                selector.configure(values=sorted([*sorted_models, current_value], key=str.lower))
            else:
                selector.configure(values=sorted_models)

    def _load_profiles(self) -> None:
        self._refresh_model_options()
        profiles = self.coordinator.get_profiles()
        for key, profile in profiles.items():
            model_var = self.profile_model_vars.get(key)
            prompt_widget = self.profile_prompt_widgets.get(key)
            if not model_var or not prompt_widget:
                continue
            model_var.set(profile.model_name)
            prompt_widget.delete("1.0", tk.END)
            prompt_widget.insert(tk.END, profile.system_prompt)

    def _save_profile(self, agent_key: str) -> None:
        prompt_widget = self.profile_prompt_widgets[agent_key]
        model_name = self.profile_model_vars[agent_key].get().strip()
        prompt_text = prompt_widget.get("1.0", tk.END).strip()
        if not prompt_text:
            messagebox.showwarning("Missing prompt", f"System prompt for {agent_key} is empty.")
            return
        self.coordinator.update_profile(
            agent_key=agent_key,
            system_prompt=prompt_text,
            model_name=model_name if model_name else None,
        )
        self._append_chat("System", f"Perfil actualizado: {agent_key}")

    def _apply_control_prompt(self) -> None:
        text = self.control_prompt_var.get().strip()
        if not text:
            return
        message, project_id = self.coordinator.apply_prompt_instruction(
            instruction=text,
            current_project_id=self.current_project_id,
        )
        self.current_project_id = project_id
        self._append_chat("Control", message)
        self.control_prompt_var.set("")
        self._load_projects()
        self._load_profiles()
        self._refresh_memory_panel()

    def _send_user_prompt(self) -> None:
        if self.is_running_agent:
            return
        text = self.user_prompt.get("1.0", tk.END).strip()
        if not text:
            return
        self.user_prompt.delete("1.0", tk.END)
        self._append_chat("You", text)
        self._set_agent_busy(True)

        thread = threading.Thread(
            target=self._run_agent_worker,
            args=(text,),
            daemon=True,
        )
        thread.start()

    def _run_agent_worker(self, text: str) -> None:
        try:
            result = self.coordinator.assistant_turn_sync(
                project_id=self.current_project_id,
                user_prompt=text,
            )
            self.root.after(0, self._handle_assistant_result, result)
        except Exception as exc:  # pragma: no cover - UI branch
            self.root.after(0, self._handle_agent_error, str(exc))

    def _handle_assistant_result(self, result: AssistantTurnResult) -> None:
        previous_project_id = self.current_project_id
        self.current_project_id = result.project_id
        sections = result.sections

        if result.source == "agent":
            if sections.get("thought"):
                self._append_chat("Thought Agent", sections["thought"])
            if sections.get("review"):
                self._append_chat("Review Agent", sections["review"])
            if sections.get("action"):
                self._append_chat("Action Agent", sections["action"])
            if not sections:
                self._append_chat("Assistant", result.reply)
        elif result.source == "command":
            self._append_chat("Assistant Command", result.reply)
        else:
            self._append_chat("Assistant", result.reply)

        if self.current_project_id != previous_project_id:
            self._load_projects()
        if result.source == "control":
            self._load_profiles()

        self._set_agent_busy(False)
        self._refresh_memory_panel()

    def _handle_agent_error(self, text: str) -> None:
        self._append_chat("Error", text)
        self._set_agent_busy(False)

    def _run_command(self) -> None:
        if self.is_running_command:
            return
        command_text = self.command_var.get().strip()
        if not command_text:
            return
        self.command_var.set("")
        self._append_command_output(f"$ {command_text}")
        self._set_command_busy(True)

        thread = threading.Thread(
            target=self._run_command_worker,
            args=(command_text,),
            daemon=True,
        )
        thread.start()

    def _run_command_worker(self, command_text: str) -> None:
        code, output = self.coordinator.execute_project_command(
            project_id=self.current_project_id,
            command_text=command_text,
        )
        self.root.after(0, self._handle_command_result, code, output)

    def _handle_command_result(self, code: int, output: str) -> None:
        self._append_command_output(output.rstrip() if output.strip() else "<no output>")
        self._append_command_output(f"[exit={code}]")
        self._append_command_output("-" * 60)
        self._set_command_busy(False)
        self._refresh_memory_panel()

    def _refresh_memory_panel(self) -> None:
        memories = self.database.recent_memories(project_id=self.current_project_id, limit=12)
        self.memory_text.configure(state=tk.NORMAL)
        self.memory_text.delete("1.0", tk.END)
        if not memories:
            self.memory_text.insert(tk.END, "No memory rows yet.\n")
        else:
            for item in memories:
                self.memory_text.insert(
                    tk.END,
                    f"[{item.created_at}] ({item.memory_type})\n{item.content}\n\n",
                )
        self.memory_text.configure(state=tk.DISABLED)

    def _refresh_thermal_panel(self) -> None:
        snapshot = self.thermal.current_snapshot()
        temp_text = (
            f"{snapshot.cpu_temp_c:.1f} C ({snapshot.source})"
            if snapshot.cpu_temp_c is not None
            else f"N/A ({snapshot.source})"
        )
        self.thermal_status_var.set(
            "\n".join(
                [
                    f"Temperature: {temp_text}",
                    f"Load ratio: {snapshot.load_ratio * 100:.0f}%",
                    f"Level: {snapshot.level.upper()}",
                    f"Auto cooldown: {snapshot.recommended_cooldown_s:.1f}s per call",
                ]
            )
        )
        self.root.after(1500, self._refresh_thermal_panel)

    def _on_close(self) -> None:
        try:
            self.database.close()
            self.thermal.stop()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = ControlCenterApp()
    app.run()


if __name__ == "__main__":
    main()

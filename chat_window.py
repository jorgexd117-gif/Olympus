"""Simple desktop chat window for interacting with the LangGraph agent."""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from tkinter import ttk

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.graph import compile_graph


class AgentChatWindow:
    """Tkinter UI for chatting with the LangGraph agent."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("LangGraph Agent Chat")
        self.root.geometry("860x620")

        self.agent = compile_graph()
        self.messages: list[BaseMessage] = []
        self.is_running = False

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        self.chat_log = tk.Text(container, wrap="word", state="disabled", height=24)
        self.chat_log.pack(fill="both", expand=True)

        prompts_frame = ttk.LabelFrame(container, text="Quick Prompts", padding=8)
        prompts_frame.pack(fill="x", pady=(10, 6))

        self.quick_prompt_var = tk.StringVar(
            value=(
                "What business is most profitable online in 2025?",
                "Give me 3 low-cost startup ideas.",
                "Create a one-week launch plan for a small online business.",
                "How can I validate a business idea with no budget?",
            )
        )
        self.quick_prompt_list = tk.Listbox(
            prompts_frame,
            listvariable=self.quick_prompt_var,
            height=4,
            exportselection=False,
        )
        self.quick_prompt_list.pack(fill="x")
        self.quick_prompt_list.bind("<Double-Button-1>", self._use_quick_prompt)

        input_frame = ttk.Frame(container)
        input_frame.pack(fill="x")

        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side="left", fill="x", expand=True)
        self.user_input.bind("<Return>", lambda _event: self.send_prompt())

        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_prompt)
        self.send_button.pack(side="left", padx=(8, 0))

        self.clear_button = ttk.Button(input_frame, text="Clear", command=self.clear_chat)
        self.clear_button.pack(side="left", padx=(8, 0))

        self._append_log("System", "Ready. Type a prompt and press Enter.")

    def _append_log(self, role: str, text: str) -> None:
        self.chat_log.configure(state="normal")
        self.chat_log.insert("end", f"{role}: {text}\n\n")
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self.is_running = busy
        state = "disabled" if busy else "normal"
        self.send_button.configure(state=state)
        self.user_input.configure(state=state)

    def _use_quick_prompt(self, _event: object) -> None:
        selection = self.quick_prompt_list.curselection()
        if not selection:
            return

        prompt = self.quick_prompt_list.get(selection[0])
        self.user_input.delete(0, "end")
        self.user_input.insert(0, prompt)
        self.send_prompt()

    def clear_chat(self) -> None:
        if self.is_running:
            return
        self.messages = []
        self.chat_log.configure(state="normal")
        self.chat_log.delete("1.0", "end")
        self.chat_log.configure(state="disabled")
        self._append_log("System", "Chat cleared.")

    def send_prompt(self) -> None:
        if self.is_running:
            return

        prompt = self.user_input.get().strip()
        if not prompt:
            return

        self.user_input.delete(0, "end")
        self._append_log("You", prompt)
        self._set_busy(True)

        thread = threading.Thread(target=self._invoke_agent, args=(prompt,), daemon=True)
        thread.start()

    def _invoke_agent(self, prompt: str) -> None:
        try:
            local_state = {"messages": [*self.messages, HumanMessage(content=prompt)]}
            result = asyncio.run(self.agent.ainvoke(local_state))
            updated_messages = result.get("messages", [])

            previous_len = len(self.messages)
            self.messages = updated_messages
            new_messages = updated_messages[previous_len:]

            replies: list[tuple[str, str]] = []
            for message in new_messages:
                if isinstance(message, AIMessage):
                    replies.append(self._format_ai_message(message.content))

            if not replies:
                replies = [("Agent", "No response generated.")]

            self.root.after(0, self._finish_request_batch, replies)
        except Exception as exc:
            self.root.after(0, self._finish_request_batch, [("Error", str(exc))])

    def _format_ai_message(self, content: str) -> tuple[str, str]:
        if content.startswith("[OpenAI pensamiento resumido]"):
            return ("OpenAI", content.replace("[OpenAI pensamiento resumido]", "", 1).strip())
        if content.startswith("[Anthropic contraste]"):
            return ("Anthropic", content.replace("[Anthropic contraste]", "", 1).strip())
        if content.startswith("[DeepSeek ejecución]"):
            return ("DeepSeek", content.replace("[DeepSeek ejecución]", "", 1).strip())
        return ("Agent", content)

    def _finish_request_batch(self, replies: list[tuple[str, str]]) -> None:
        for role, text in replies:
            self._append_log(role, text)
        self._set_busy(False)
        self.user_input.focus_set()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = AgentChatWindow()
    app.run()


if __name__ == "__main__":
    main()

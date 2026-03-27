"""Utilities to verify and auto-start a local Ollama runtime."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import time
import json
import urllib.error
import urllib.request

_START_LOCK = threading.Lock()
_LAST_START_ATTEMPT_TS = 0.0


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _can_probe(base_url: str, timeout_s: float) -> bool:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            if not (200 <= int(response.status) < 300):
                return False
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False
    except Exception:
        return False

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return False

    models = decoded.get("models") if isinstance(decoded, dict) else None
    return isinstance(models, list)


def _try_start_ollama_cli() -> bool:
    if shutil.which("ollama") is None:
        return False
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:
        return False


def _try_open_ollama_app_macos() -> bool:
    if platform.system().lower() != "darwin":
        return False
    try:
        completed = subprocess.run(
            ["open", "-a", "Ollama"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode == 0
    except Exception:
        return False


def ensure_ollama_ready(*, base_url: str | None = None) -> tuple[bool, str]:
    """
    Ensure local Ollama server is available.

    Returns:
        (is_ready, detail_message)
    """
    target_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).strip()
    if not target_url:
        target_url = "http://localhost:11434"

    probe_timeout_s = float(os.getenv("OLLAMA_PROBE_TIMEOUT_SECONDS", "1.5"))
    if _can_probe(target_url, timeout_s=probe_timeout_s):
        return True, "Ollama is already running."

    if not _env_bool("OLLAMA_AUTO_START", True):
        return (
            False,
            (
                "Ollama is not reachable and OLLAMA_AUTO_START=false. "
                "Start Ollama manually or set OLLAMA_AUTO_START=true."
            ),
        )

    startup_timeout_s = float(os.getenv("OLLAMA_AUTO_START_TIMEOUT_SECONDS", "25"))
    retry_window_s = float(os.getenv("OLLAMA_AUTO_START_RETRY_WINDOW_SECONDS", "15"))

    with _START_LOCK:
        global _LAST_START_ATTEMPT_TS
        now = time.time()
        should_attempt = (now - _LAST_START_ATTEMPT_TS) >= retry_window_s
        if should_attempt:
            started_by_cli = _try_start_ollama_cli()
            started_by_app = False if started_by_cli else _try_open_ollama_app_macos()
            _LAST_START_ATTEMPT_TS = now
        else:
            started_by_cli = False
            started_by_app = False

    start_source = (
        "ollama serve"
        if started_by_cli
        else ("open -a Ollama" if started_by_app else "no launcher or recent attempt reused")
    )

    deadline = time.time() + max(1.0, startup_timeout_s)
    while time.time() < deadline:
        if _can_probe(target_url, timeout_s=probe_timeout_s):
            return True, f"Ollama started successfully ({start_source})."
        time.sleep(1.0)

    return (
        False,
        (
            "Ollama is still unavailable after auto-start attempt "
            f"({start_source}). Verify installation and OLLAMA_BASE_URL={target_url}."
        ),
    )

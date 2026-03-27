#!/usr/bin/env python3
"""Keep API and UI alive with secure localhost bindings and auto-restart."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from urllib.error import URLError
from urllib.request import urlopen


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _health_ok(url: str, timeout_s: float) -> bool:
    try:
        with urlopen(url, timeout=timeout_s) as response:  # nosec B310 - local healthcheck URL
            status = getattr(response, "status", 200)
            return 200 <= status < 500
    except (URLError, TimeoutError, ValueError):
        return False
    except Exception:
        return False


@dataclass
class Service:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    log_path: Path
    health_url: str | None = None
    health_timeout_s: float = 2.0
    health_failures_before_restart: int = 4
    health_interval_s: float = 5.0
    startup_grace_s: float = 12.0
    max_backoff_s: float = 30.0
    process: subprocess.Popen[str] | None = field(default=None, init=False)
    _log_handle: object | None = field(default=None, init=False)
    _next_start_at: float = field(default=0.0, init=False)
    _backoff_s: float = field(default=1.0, init=False)
    _health_failures: int = field(default=0, init=False)
    _next_health_at: float = field(default=0.0, init=False)
    _health_grace_until: float = field(default=0.0, init=False)

    def start(self, now: float) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._log_handle.write(f"\n[{_ts()}] START {self.name}: {' '.join(self.cmd)}\n")
        self._log_handle.flush()
        self.process = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            env=self.env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._next_health_at = now + self.health_interval_s
        self._health_grace_until = now + self.startup_grace_s
        self._health_failures = 0
        _log(f"{self.name} started (pid={self.process.pid}).")

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None

    def schedule_restart(self, now: float, reason: str) -> None:
        self._next_start_at = now + self._backoff_s
        _log(f"{self.name} restart scheduled in {self._backoff_s:.0f}s ({reason}).")
        self._backoff_s = min(self.max_backoff_s, self._backoff_s * 2)

    def stop(self, force: bool = False) -> None:
        if self.process is None:
            self._close_log()
            return
        proc = self.process
        self.process = None
        try:
            if proc.poll() is None:
                if force:
                    proc.kill()
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        finally:
            self._close_log()

    def tick(self, now: float) -> None:
        if self.process is None:
            if now >= self._next_start_at:
                self.start(now)
            return

        code = self.process.poll()
        if code is not None:
            self.stop(force=True)
            self.schedule_restart(now, f"exit code {code}")
            return

        if self.health_url and now >= self._next_health_at:
            self._next_health_at = now + self.health_interval_s
            if now < self._health_grace_until:
                return
            if _health_ok(self.health_url, self.health_timeout_s):
                self._health_failures = 0
                self._backoff_s = 1.0
            else:
                self._health_failures += 1
                _log(
                    f"{self.name} health failed ({self._health_failures}/"
                    f"{self.health_failures_before_restart}): {self.health_url}"
                )
                if self._health_failures >= self.health_failures_before_restart:
                    self.stop(force=True)
                    self.schedule_restart(now, "healthcheck failure threshold reached")


def _python_has_module(python_bin: Path | str, module_name: str) -> bool:
    try:
        result = subprocess.run(
            [str(python_bin), "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=4,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _find_python(root_dir: Path) -> str:
    candidates = [
        root_dir / ".venv" / "bin" / "python",
        root_dir / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists() and _python_has_module(candidate, "uvicorn"):
            return str(candidate)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if _python_has_module(sys.executable, "uvicorn"):
        return sys.executable
    return sys.executable


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    # Buffer logs line-by-line for better observability.
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-restart supervisor for LangGraph API + UI.")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8010)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=5173)
    parser.add_argument("--no-ui", action="store_true", help="Run only API.")
    parser.add_argument("--logs-dir", default="logs/keepalive")
    parser.add_argument("--max-backoff", type=float, default=30.0)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--health-failures", type=int, default=4)
    parser.add_argument("--startup-grace", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    python_bin = _find_python(root_dir)
    logs_dir = (root_dir / args.logs_dir).resolve()

    cors_origins = ",".join(
        [
            f"http://{args.ui_host}:{args.ui_port}",
            f"http://127.0.0.1:{args.ui_port}",
            f"http://localhost:{args.ui_port}",
        ]
    )

    api_env = _base_env()
    api_env.setdefault("CORS_ALLOW_ORIGINS", cors_origins)
    api_env.setdefault("COMMAND_EXECUTION_ENABLED", "false")

    api_service = Service(
        name="api",
        cmd=[
            python_bin,
            "-m",
            "uvicorn",
            "src.api_server:app",
            "--host",
            args.api_host,
            "--port",
            str(args.api_port),
        ],
        cwd=root_dir,
        env=api_env,
        log_path=logs_dir / "api.log",
        health_url=f"http://{args.api_host}:{args.api_port}/health",
        health_interval_s=args.health_interval,
        health_failures_before_restart=args.health_failures,
        startup_grace_s=args.startup_grace,
        max_backoff_s=args.max_backoff,
    )

    services: list[Service] = [api_service]
    if not args.no_ui:
        ui_env = _base_env()
        ui_service = Service(
            name="ui",
            cmd=[
                "npm",
                "run",
                "dev",
                "--",
                "--host",
                args.ui_host,
                "--port",
                str(args.ui_port),
                "--strictPort",
            ],
            cwd=root_dir / "ui",
            env=ui_env,
            log_path=logs_dir / "ui.log",
            health_url=f"http://{args.ui_host}:{args.ui_port}",
            health_interval_s=args.health_interval,
            health_failures_before_restart=args.health_failures,
            startup_grace_s=args.startup_grace,
            max_backoff_s=args.max_backoff,
        )
        services.append(ui_service)

    running = True

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal running
        _log(f"signal {signum} received, shutting down...")
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _log("keepalive supervisor started.")
    _log(f"api url: http://{args.api_host}:{args.api_port}")
    if not args.no_ui:
        _log(f"ui url: http://{args.ui_host}:{args.ui_port}")
    _log(f"logs: {logs_dir}")

    try:
        while running:
            now = time.time()
            for service in services:
                service.tick(now)
            time.sleep(1.0)
    finally:
        for service in services:
            service.stop(force=True)
        _log("keepalive supervisor stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

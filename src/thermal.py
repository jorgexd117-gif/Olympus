"""Live thermal monitoring and soft regulation for stable runtime performance."""

from __future__ import annotations

import asyncio
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ThermalSnapshot:
    """Snapshot of current thermal and load conditions."""

    timestamp: str
    cpu_temp_c: float | None
    source: str
    load_ratio: float
    level: str
    recommended_cooldown_s: float


class ThermalRegulator:
    """Runtime-friendly thermal manager that throttles work under heat."""

    def __init__(self) -> None:
        self.enabled = os.getenv("THERMAL_MONITOR_ENABLED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.warning_c = float(os.getenv("THERMAL_WARNING_C", "75"))
        self.critical_c = float(os.getenv("THERMAL_CRITICAL_C", "85"))
        self.cooldown_warn_s = float(os.getenv("THERMAL_COOLDOWN_WARN_SECONDS", "0.8"))
        self.cooldown_critical_s = float(os.getenv("THERMAL_COOLDOWN_CRITICAL_SECONDS", "2.5"))
        self.poll_interval_s = float(os.getenv("THERMAL_POLL_INTERVAL_SECONDS", "2.0"))
        self.timeout_scale_warning = float(os.getenv("THERMAL_TIMEOUT_SCALE_WARNING", "1.0"))
        self.timeout_scale_critical = float(os.getenv("THERMAL_TIMEOUT_SCALE_CRITICAL", "0.85"))
        self.timeout_min_s = float(os.getenv("THERMAL_TIMEOUT_MIN_SECONDS", "12.0"))

        self._lock = threading.Lock()
        self._latest: ThermalSnapshot | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _poll_loop(self) -> None:
        while self._running:
            self.sample()
            time.sleep(self.poll_interval_s)

    def current_snapshot(self) -> ThermalSnapshot:
        with self._lock:
            cached = self._latest
        if cached:
            return cached
        return self.sample()

    def sample(self) -> ThermalSnapshot:
        if not self.enabled:
            snapshot = ThermalSnapshot(
                timestamp=_utc_now(),
                cpu_temp_c=None,
                source="disabled",
                load_ratio=self._load_ratio(),
                level="normal",
                recommended_cooldown_s=0.0,
            )
            with self._lock:
                self._latest = snapshot
            return snapshot

        temp_c, source = self._read_temperature()
        load_ratio = self._load_ratio()
        level = self._derive_level(temp_c=temp_c, load_ratio=load_ratio)
        cooldown = self._cooldown_for(level=level)
        snapshot = ThermalSnapshot(
            timestamp=_utc_now(),
            cpu_temp_c=temp_c,
            source=source,
            load_ratio=load_ratio,
            level=level,
            recommended_cooldown_s=cooldown,
        )
        with self._lock:
            self._latest = snapshot
        return snapshot

    async def throttle_if_needed(self) -> ThermalSnapshot:
        snapshot = self.sample()
        if snapshot.recommended_cooldown_s > 0:
            await asyncio.sleep(snapshot.recommended_cooldown_s)
        return snapshot

    def request_timeout(self, *, base_timeout_s: float, level: str) -> float:
        if level == "critical":
            scaled = base_timeout_s * max(0.1, self.timeout_scale_critical)
            return max(self.timeout_min_s, scaled)
        if level == "warning":
            scaled = base_timeout_s * max(0.1, self.timeout_scale_warning)
            return max(self.timeout_min_s, scaled)
        return max(self.timeout_min_s, base_timeout_s)

    def _cooldown_for(self, *, level: str) -> float:
        if level == "critical":
            return self.cooldown_critical_s
        if level == "warning":
            return self.cooldown_warn_s
        return 0.0

    def _derive_level(self, *, temp_c: float | None, load_ratio: float) -> str:
        if temp_c is not None:
            if temp_c >= self.critical_c:
                return "critical"
            if temp_c >= self.warning_c:
                return "warning"
            return "normal"

        # Fallback when no temperature sensor is available.
        if load_ratio >= 0.95:
            return "critical"
        if load_ratio >= 0.80:
            return "warning"
        return "normal"

    @staticmethod
    def _load_ratio() -> float:
        try:
            load1, _load5, _load15 = os.getloadavg()
            cpus = max(1, os.cpu_count() or 1)
            return max(0.0, load1 / cpus)
        except Exception:
            return 0.0

    def _read_temperature(self) -> tuple[float | None, str]:
        system = platform.system().lower()
        if system == "darwin":
            temp = self._read_temp_macos()
            if temp is not None:
                return temp, "osx-cpu-temp"
        if system == "linux":
            temp = self._read_temp_linux()
            if temp is not None:
                return temp, "sysfs"
        return None, "unavailable"

    @staticmethod
    def _read_temp_macos() -> float | None:
        # Optional helper: brew install osx-cpu-temp
        try:
            completed = subprocess.run(
                ["osx-cpu-temp"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        match = re.search(r"(-?\d+(?:\.\d+)?)", completed.stdout)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _read_temp_linux() -> float | None:
        zones = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
        values: list[float] = []
        for zone_file in zones:
            try:
                raw = zone_file.read_text(encoding="utf-8").strip()
                if not raw:
                    continue
                milli = float(raw)
                values.append(milli / 1000.0)
            except Exception:
                continue
        if not values:
            return None
        return sum(values) / len(values)


_REGULATOR: ThermalRegulator | None = None


def get_thermal_regulator() -> ThermalRegulator:
    """Return a singleton thermal regulator."""
    global _REGULATOR
    if _REGULATOR is None:
        _REGULATOR = ThermalRegulator()
    return _REGULATOR

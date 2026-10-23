"""Hardware detection: RAM, CPU, and optional NVIDIA GPU info.

Stdlib only. Every detector is a small function so tests can mock the
underlying data sources (proc files, sysctl output, nvidia-smi output).

Copyright (c) 2026 Asdale Dynamics, LLC
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field

GIB = 1024**3


@dataclass
class GpuInfo:
    name: str
    vram_bytes: int
    source: str = "nvidia-smi"

    def vram_gib(self) -> float:
        return self.vram_bytes / GIB


@dataclass
class Hardware:
    total_ram_bytes: int = 0
    available_ram_bytes: int = 0
    cpu_count: int = 0
    cpu_model: str = "unknown CPU"
    gpus: list[GpuInfo] = field(default_factory=list)
    platform_name: str = field(
        default_factory=lambda: f"{platform.system()} {platform.machine()}"
    )

    def total_ram_gib(self) -> float:
        return self.total_ram_bytes / GIB

    def available_ram_gib(self) -> float:
        return self.available_ram_bytes / GIB

    def total_vram_bytes(self) -> int:
        return sum(g.vram_bytes for g in self.gpus)

    def describe(self) -> str:
        cpu = f"{self.cpu_count} CPU cores ({self.cpu_model})"
        ram = f"{self.total_ram_gib():.1f} GiB RAM"
        if self.available_ram_bytes:
            ram += f", {self.available_ram_gib():.1f} GiB free right now"
        lines = [f"Platform: {self.platform_name}", f"CPU: {cpu}", f"RAM: {ram}"]
        if self.gpus:
            vram = self.total_vram_bytes() / GIB
            names = ", ".join(f"{g.name}" for g in self.gpus)
            lines.append(f"GPU: {names} with {vram:.1f} GiB VRAM total")
        else:
            lines.append("GPU: no CUDA GPU detected (CPU-only inference)")
        return "\n".join(lines)


_NUM_KB = re.compile(r"(\d+)\s*kB", re.IGNORECASE)
_CPU_MODEL_RE = re.compile(r"^model name\s*:\s*(.+)$", re.MULTILINE)


def read_ram_from_procfs(meminfo_text: str) -> tuple[int, int]:
    """Parse /proc/meminfo-style text -> (total_bytes, available_bytes).

    MemAvailable may be missing on very old kernels; approximate it with
    MemFree + buffers/cache if so.
    """
    values_gb: dict[str, float] = {}
    for line in meminfo_text.splitlines():
        m = _NUM_KB.search(line)
        if not m:
            continue
        key, _, _ = line.partition(":")
        values_gb[key.strip()] = int(m.group(1)) * 1024
    total = int(values_gb.get("MemTotal", 0))
    if "MemAvailable" in values_gb:
        available = int(values_gb["MemAvailable"])
    else:
        fallback = (
            values_gb.get("MemFree", 0)
            + values_gb.get("Buffers", 0)
            + values_gb.get("Cached", 0)
        )
        available = int(fallback)
    # Never claim more free RAM than there is total.
    available = min(available, total)
    return total, available


def detect_ram() -> tuple[int, int]:
    """Return (total_bytes, available_bytes). Best-effort, cross-platform."""
    proc_path = "/proc/meminfo"
    if os.path.exists(proc_path):
        try:
            with open(proc_path, encoding="utf-8") as fh:
                return read_ram_from_procfs(fh.read())
        except OSError:
            pass
    if hasattr(os, "sysconf"):
        try:
            pg_size = os.sysconf("SC_PAGE_SIZE")
            total = os.sysconf("SC_PHYS_PAGES") * pg_size
            avail = os.sysconf("SC_AVPHYS_PAGES") * pg_size
            return int(total), int(avail)
        except (ValueError, OSError):
            pass
    if platform.system() == "Windows":  # pragma: no cover - platform specific
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
        except Exception:  # noqa: BLE001, S110 - best-effort detection must never crash
            pass
    if platform.system() == "Darwin":  # pragma: no cover - platform specific
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            total = int(out.stdout.strip())
            return total, total
        except Exception:  # noqa: BLE001, S110 - best-effort detection must never crash
            pass
    return 0, 0


def detect_cpu() -> tuple[int, str]:
    """Return (core_count, model_name-ish string)."""
    count = os.cpu_count() or 1
    model = platform.processor() or platform.machine() or "unknown CPU"
    proc_path = "/proc/cpuinfo"
    if os.path.exists(proc_path):
        try:
            with open(proc_path, encoding="utf-8") as fh:
                m = _CPU_MODEL_RE.search(fh.read())
        except OSError:
            m = None
        if m:
            model = m.group(1).strip()
    return count, model


_NVIDIA_CSV_RE = re.compile(
    r"^(?P<name>[^,]+),\s*(?P<vram>\d+(?:\.\d+)?)(?P<unit> MiB| MB)\s*$"
)


def parse_nvidia_smi_output(text: str) -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`."""
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _NVIDIA_CSV_RE.match(line)
        if not m:
            continue
        value = float(m.group("vram"))
        unit = m.group("unit").strip()
        if unit.lower() == "mb":
            # Decimal megabytes.
            mib = value * 1e6 / (1024 * 1024)
        else:
            mib = round(value)
        gpus.append(
            GpuInfo(name=m.group("name").strip(), vram_bytes=int(mib * 1024 * 1024))
        )
    return gpus


NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader",
]


def detect_gpus(runner=subprocess.run, which=shutil.which) -> list[GpuInfo]:
    """Optional GPU detection via nvidia-smi. Returns [] if unavailable."""
    if not which("nvidia-smi"):
        return []
    try:
        out = runner(
            NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=10, check=True
        )
        return parse_nvidia_smi_output(out.stdout)
    except (OSError, subprocess.SubprocessError):
        return []


def detect_hardware() -> Hardware:
    total, available = detect_ram()
    cores, model = detect_cpu()
    return Hardware(
        total_ram_bytes=total,
        available_ram_bytes=available,
        cpu_count=cores,
        cpu_model=model,
        gpus=detect_gpus(),
    )

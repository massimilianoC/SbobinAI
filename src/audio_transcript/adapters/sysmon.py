"""OS resource samplers: CPU, RAM, process memory and NVIDIA GPU telemetry.

Standard library only (``ctypes`` on Windows, ``/proc`` on Linux); ``psutil`` is used when it
is importable. The GPU is read from one long-lived ``nvidia-smi ... -lms`` subprocess. The
GPU is shared by every process, so its numbers are system-wide, not attributable to this run.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable

MIB = 1024 * 1024
NVIDIA_FIELDS = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"


def _float(text: str) -> float | None:
    text = text.strip()
    if not text or text.startswith("[") or text.upper() in {"N/A", "NAN"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value == value and abs(value) != float("inf") else None


# ---------------------------------------------------------------------------- parsers


def parse_nvidia_smi_line(line: str) -> dict | None:
    """Parse one ``--format=csv,noheader,nounits`` line of :data:`NVIDIA_FIELDS`."""
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) < 6:
        return None
    index = _float(parts[0])
    if index is None:
        return None
    return {
        "gpu_index": int(index),
        "gpu_util_percent": _float(parts[1]),
        "gpu_mem_used_mb": _float(parts[2]),
        "gpu_mem_total_mb": _float(parts[3]),
        "gpu_power_w": _float(parts[4]),
        "gpu_temp_c": _float(parts[5]),
    }


def parse_proc_stat(text: str) -> tuple[int, int] | None:
    """Return ``(idle, total)`` jiffies from the first ``cpu`` line of ``/proc/stat``."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            try:
                values = [int(item) for item in line.split()[1:]]
            except ValueError:
                return None
            if len(values) < 4:
                return None
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return idle, sum(values[:8])
    return None


def parse_meminfo(text: str) -> tuple[float, float] | None:
    """Return ``(used_mb, total_mb)`` from ``/proc/meminfo`` text."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached"}:
            try:
                found[key] = int(rest.split()[0])
            except (ValueError, IndexError):
                return None
    total = found.get("MemTotal")
    if total is None:
        return None
    available = found.get("MemAvailable")
    if available is None:
        available = found.get("MemFree", 0) + found.get("Buffers", 0) + found.get("Cached", 0)
    return (total - available) / 1024, total / 1024


def parse_vmrss(text: str) -> float | None:
    """Resident set size in MiB from ``/proc/<pid>/status`` text."""
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            try:
                return int(line.split()[1]) / 1024
            except (ValueError, IndexError):
                return None
    return None


# ---------------------------------------------------------------------------- host samplers


class _CpuDelta:
    def __init__(self) -> None:
        self._last: tuple[float, float] | None = None

    def percent(self, idle: float, total: float) -> float | None:
        last, self._last = self._last, (idle, total)
        if last is None:
            return None
        d_total = total - last[1]
        d_idle = idle - last[0]
        if d_total <= 0:
            return None
        return round(max(0.0, min(100.0, 100.0 * (1 - d_idle / d_total))), 1)


class PsutilHost:
    label = "psutil"

    def __init__(self, psutil_module) -> None:
        self._psutil = psutil_module
        self._process = psutil_module.Process()
        psutil_module.cpu_percent(None)

    def sample(self) -> dict:
        memory = self._psutil.virtual_memory()
        return {
            "cpu_percent": round(float(self._psutil.cpu_percent(None)), 1),
            "ram_used_mb": round((memory.total - memory.available) / MIB, 1),
            "ram_total_mb": round(memory.total / MIB, 1),
            "process_rss_mb": round(self._process.memory_info().rss / MIB, 1),
        }

    def close(self) -> None:
        return None


class LinuxHost:
    label = "proc"

    def __init__(self, read: Callable[[str], str] | None = None) -> None:
        self._read = read or _read_text
        self._cpu = _CpuDelta()

    def sample(self) -> dict:
        out: dict = {}
        stat = parse_proc_stat(self._read("/proc/stat"))
        if stat is not None:
            out["cpu_percent"] = self._cpu.percent(*stat)
        memory = parse_meminfo(self._read("/proc/meminfo"))
        if memory is not None:
            out["ram_used_mb"] = round(memory[0], 1)
            out["ram_total_mb"] = round(memory[1], 1)
        try:
            rss = parse_vmrss(self._read("/proc/self/status"))
        except OSError:
            rss = None
        if rss is not None:
            out["process_rss_mb"] = round(rss, 1)
        return out

    def close(self) -> None:
        return None


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as stream:
        return stream.read()


class WindowsHost:
    """``GetSystemTimes``, ``GlobalMemoryStatusEx`` and ``GetProcessMemoryInfo`` via ctypes."""

    label = "windows"

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._cpu = _CpuDelta()

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        class ProcessMemory(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        self._memory_status = MemoryStatus
        self._process_memory = ProcessMemory
        self._kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        try:
            self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
            self._psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemory),
                wintypes.DWORD,
            ]
        except OSError:
            self._psapi = None

    def _filetime(self, value) -> int:
        return (value.dwHighDateTime << 32) | value.dwLowDateTime

    def sample(self) -> dict:
        ctypes, wintypes = self._ctypes, self._wintypes
        out: dict = {}
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if self._kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            idle_t, total = self._filetime(idle), self._filetime(kernel) + self._filetime(user)
            out["cpu_percent"] = self._cpu.percent(idle_t, total)  # kernel includes idle
        status = self._memory_status()
        status.dwLength = ctypes.sizeof(status)
        if self._kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            out["ram_total_mb"] = round(status.ullTotalPhys / MIB, 1)
            out["ram_used_mb"] = round((status.ullTotalPhys - status.ullAvailPhys) / MIB, 1)
        if self._psapi is not None:
            counters = self._process_memory()
            counters.cb = ctypes.sizeof(counters)
            if self._psapi.GetProcessMemoryInfo(
                self._kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                out["process_rss_mb"] = round(counters.WorkingSetSize / MIB, 1)
        return out

    def close(self) -> None:
        return None


def make_host_sampler(platform: str | None = None, *, import_module=importlib.import_module):
    """Best available host sampler, or ``None`` when no source works."""
    try:
        return PsutilHost(import_module("psutil"))
    except Exception:
        pass
    platform = platform or sys.platform
    try:
        if platform == "win32":
            return WindowsHost()
        if platform.startswith("linux"):
            return LinuxHost()
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------- GPU sampler


class NvidiaSmiStream:
    """One long-lived ``nvidia-smi -lms`` process feeding the latest GPU reading.

    Restart-safe: if the process exits while running, it is started again (at most
    ``max_restarts`` times, one second apart). :meth:`sample` returns an empty dict when no
    fresh reading exists.
    """

    label = "nvidia-smi"

    def __init__(
        self,
        executable: str,
        interval: float = 1.0,
        *,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        max_restarts: int = 3,
        stale_after: float | None = None,
    ):
        self.executable = executable
        self.interval_ms = max(100, int(interval * 1000))
        self._popen = popen
        self._max_restarts = max_restarts
        self._stale_after = stale_after if stale_after is not None else max(5.0, interval * 5)
        self._lock = threading.Lock()
        self._gpus: dict[int, dict] = {}
        self._stamp = 0.0
        self._process = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.restarts = 0

    def command(self) -> list[str]:
        return [
            self.executable,
            f"--query-gpu={NVIDIA_FIELDS}",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="nvidia-smi", daemon=True)
        self._thread.start()

    def _spawn(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return self._popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            text=True,
            creationflags=flags,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._process = self._spawn()
            except (OSError, ValueError):
                return
            try:
                for line in self._process.stdout:
                    if self._stop.is_set():
                        return
                    self.feed(line)
            except (OSError, ValueError):
                pass
            if self._stop.is_set() or self.restarts >= self._max_restarts:
                return
            self.restarts += 1
            if self._stop.wait(1.0):
                return

    def feed(self, line: str) -> None:
        parsed = parse_nvidia_smi_line(line)
        if parsed is None:
            return
        with self._lock:
            self._gpus[parsed["gpu_index"]] = parsed
            self._stamp = time.monotonic()

    def sample(self) -> dict:
        if self._thread is None:
            self.start()
        with self._lock:
            if not self._gpus or time.monotonic() - self._stamp > self._stale_after:
                return {}
            first = self._gpus[min(self._gpus)]
        return {key: value for key, value in first.items() if key != "gpu_index"}

    def close(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            except OSError:
                pass


def make_gpu_sampler(interval: float = 1.0, *, which=shutil.which):
    """An :class:`NvidiaSmiStream` when ``nvidia-smi`` is on PATH, else ``None``."""
    executable = which("nvidia-smi")
    if not executable:
        return None
    return NvidiaSmiStream(executable, interval)


def make_samplers(interval: float = 1.0) -> list:
    samplers = []
    host = make_host_sampler()
    if host is not None:
        samplers.append(host)
    gpu = make_gpu_sampler(interval)
    if gpu is not None:
        samplers.append(gpu)
    return samplers

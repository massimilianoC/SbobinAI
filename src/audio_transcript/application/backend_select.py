"""Inference backend selection (CUDA, Vulkan, CPU) with an explicit, visible fallback.

Pure logic: the only side effect is reading the runtime folders and one injected probe
that lists the devices of a runtime (``llama-server --list-devices``; see
``adapters.devices``). The result is a ``Selection``; it is never part of the job
fingerprint (the measured transcripts are identical across backends) but it is recorded
in the run provenance so versions made on different backends stay distinguishable.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from ..config import ServerConfig

EXECUTABLES = ("llama-server.exe", "llama-server")
LIBRARIES = {
    "cuda": ("ggml-cuda.dll", "libggml-cuda.so", "libggml-cuda.dylib"),
    "vulkan": ("ggml-vulkan.dll", "libggml-vulkan.so", "libggml-vulkan.dylib"),
}
DEVICE_PREFIX = {"cuda": "CUDA", "vulkan": "Vulkan"}
# A Vulkan CPU renderer would be slower than the optimised CPU path.
SOFTWARE_DEVICE = re.compile(
    r"llvmpipe|lavapipe|swiftshader|software rasterizer|basic render", re.I
)
# Measured on the reference machine (RTX 5070 Ti, 12-core CPU); other hardware differs.
SLOWDOWN = {
    "vulkan": "about 2x slower than CUDA",
    "cpu": "about 9x slower than CUDA",
}
_DEVICE_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z]*\d+)\s*:\s*(\S.*?)\s*$")
_MEMORY_SUFFIX = re.compile(r"\s*\(\s*\d+\s*MiB\b.*\)\s*$")


class BackendSelectionError(RuntimeError):
    """No acceptable backend; ``tried`` lists why each candidate was rejected."""

    def __init__(self, message: str, tried: tuple[Attempt, ...] = ()):
        super().__init__(message)
        self.tried = tried


@dataclass(frozen=True)
class Device:
    id: str  # as printed by llama-server, e.g. "CUDA0" or "Vulkan1"
    description: str  # e.g. "NVIDIA GeForce RTX 5070 Ti"


@dataclass(frozen=True)
class DeviceListing:
    """Outcome of one ``--list-devices`` run: devices, or why it could not be done."""

    devices: tuple[Device, ...] = ()
    error: str | None = None


Probe = Callable[[Path], DeviceListing]


@dataclass(frozen=True)
class Attempt:
    backend: str
    ok: bool
    reason: str | None


@dataclass(frozen=True)
class Selection:
    backend: str
    device: str  # "CUDA0", "Vulkan0" or "none" (CPU)
    device_name: str
    runtime_dir: Path
    requested: str  # the configured [server].backend ("auto" or a fixed backend)
    fallback_used: bool
    tried: tuple[Attempt, ...]
    threads: int | None
    runtime_tag: str | None = None
    warning: str | None = None

    @property
    def skipped(self) -> list[dict]:
        return [
            {"backend": item.backend, "reason": item.reason} for item in self.tried if not item.ok
        ]

    def to_dict(self) -> dict:
        """Full description for ``server-profile`` and ``doctor`` (includes the runtime path)."""
        return {
            "backend": self.backend,
            "device": self.device,
            "device_name": self.device_name,
            "runtime_dir": str(self.runtime_dir),
            "runtime_tag": self.runtime_tag,
            "requested": self.requested,
            "fallback_used": self.fallback_used,
            "threads": self.threads,
            "tried": [
                {"backend": item.backend, "ok": item.ok, "reason": item.reason}
                for item in self.tried
            ],
            "warning": self.warning,
            "warning_lines": warning_lines(self),
        }

    def run_info(self) -> dict:
        """Path-free record for provenance, reports, events and the console."""
        return {
            "backend": self.backend,
            "device": self.device,
            "device_name": self.device_name,
            "fallback_used": self.fallback_used,
            "requested": self.requested,
            "threads": self.threads,
            "runtime_tag": self.runtime_tag,
            "skipped": self.skipped,
        }


def parse_devices(text: str) -> tuple[Device, ...]:
    """Devices printed by ``llama-server --list-devices`` (``Vulkan0: NAME (… MiB, … free)``)."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("available devices"):
            lines = lines[index + 1 :]
            break
    devices = []
    for line in lines:
        matched = _DEVICE_LINE.match(line)
        if matched:
            description = _MEMORY_SUFFIX.sub("", matched.group(2)).strip()
            devices.append(Device(matched.group(1), description))
    return tuple(devices)


def find_executable(folder: Path) -> Path | None:
    for name in EXECUTABLES:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def runtime_tag(folder: Path) -> str | None:
    """Release tag from the ``runtime-manifest.json`` written by the installers, if any."""
    try:
        data = json.loads((folder / "runtime-manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    tag = data.get("tag") if isinstance(data, dict) else None
    return tag if isinstance(tag, str) and tag else None


def default_threads() -> int:
    """Fallback when the physical core count is unknown: half the logical processors."""
    return max(1, (os.cpu_count() or 2) // 2)


def short_device_name(name: str) -> str:
    """``NVIDIA GeForce RTX 5070 Ti`` -> ``RTX 5070 Ti`` (console header)."""
    cleaned = re.sub(r"^(NVIDIA|AMD|Intel\(R\)|Intel)\s+", "", name.strip(), flags=re.I)
    cleaned = re.sub(r"^(GeForce|Radeon|Arc)\s+", "", cleaned, flags=re.I) or cleaned
    return re.sub(r"\s*\((TM|R)\)", "", cleaned).strip() or name


def fallback_reasons(info: dict) -> str:
    """``no CUDA device reported; vulkan: runtime not installed`` (first backend is implied)."""
    skipped = info.get("skipped") or []
    parts = []
    for index, item in enumerate(skipped):
        reason = item.get("reason") or "unavailable"
        parts.append(reason if index == 0 else f"{item['backend']}: {reason}")
    return "; ".join(parts)


def describe_runtime(info: dict | None) -> str | None:
    """One status line, e.g. ``Backend: vulkan Vulkan0 NVIDIA ... (fallback from cuda: ...)``."""
    if not info or not info.get("backend"):
        return None
    parts = [f"Backend: {info['backend']}"]
    if info.get("device") and info["device"] != "none":
        parts.append(str(info["device"]))
    if info.get("device_name") and info["backend"] != "cpu":
        parts.append(str(info["device_name"]))
    if info["backend"] == "cpu" and info.get("threads"):
        parts.append(f"{info['threads']} threads")
    line = " ".join(parts)
    if info.get("fallback_used"):
        skipped = info.get("skipped") or []
        origin = skipped[0]["backend"] if skipped else "the preferred backend"
        line += f" (fallback from {origin}: {fallback_reasons(info) or 'unavailable'})"
    return line


def header_label(info: dict | None, sep: str = " · ") -> str | None:
    """Short console-header text: ``CUDA0 · RTX 5070 Ti`` or ``CPU (fallback)``."""
    if not info or not info.get("backend"):
        return None
    if info["backend"] == "cpu":
        label = "CPU"
    else:
        label = str(info.get("device") or info["backend"])
        if info.get("device_name"):
            label += f"{sep}{short_device_name(str(info['device_name']))}"
    return label + (" (fallback)" if info.get("fallback_used") else "")


def warning_lines(selection: Selection) -> list[str]:
    """Multi-line banner for the launcher; empty when no fallback was needed."""
    if not selection.fallback_used:
        return []
    chosen = selection.backend
    if selection.device != "none":
        chosen += f" ({selection.device}, {selection.device_name})"
    else:
        chosen += f" ({selection.threads} threads)"
    lines = [f"Inference backend fallback: using {chosen}"]
    for item in selection.tried:
        if not item.ok:
            lines.append(f"  skipped {item.backend}: {item.reason}")
    slowdown = SLOWDOWN.get(selection.backend)
    if slowdown:
        lines.append(
            f"  expected speed: {slowdown} (measured on the reference machine; hardware differs)"
        )
    lines.append(
        "  the transcript is unaffected (identical output was measured on all backends); "
        "see docs/runtime-setup.md to install or force a backend"
    )
    return lines


def _check_gpu_backend(
    backend: str, folder: Path, server: ServerConfig, probe: Probe, cache: dict
) -> tuple[Device | None, str | None]:
    """Return ``(device, None)`` when usable or ``(None, reason)``."""
    if not any((folder / name).is_file() for name in LIBRARIES[backend]):
        return None, f"{LIBRARIES[backend][0]} missing"
    prefix = DEVICE_PREFIX[backend]
    if folder not in cache:
        cache[folder] = probe(folder)
    listing = cache[folder]
    if listing.error:
        return None, listing.error
    matching = [d for d in listing.devices if re.fullmatch(rf"{prefix}\d+", d.id)]
    if not matching:
        return None, f"no {prefix} device reported"
    hardware = [d for d in matching if not SOFTWARE_DEVICE.search(d.description)]
    if not hardware:
        return None, f"only a software {prefix} device ({matching[0].description})"
    wanted = server.device if server.device and server.device.startswith(prefix) else None
    if wanted:
        for device in hardware:
            if device.id == wanted:
                return device, None
        return None, f"device {wanted} not reported"
    return hardware[0], None


def select_backend(
    server: ServerConfig,
    probe: Probe,
    *,
    threads_default: Callable[[], int] = default_threads,
) -> Selection:
    """Choose the backend for ``server``; raise ``BackendSelectionError`` when none works.

    ``backend = "auto"`` walks ``server.fallback`` and takes the first backend whose runtime
    folder exists and (for cuda and vulkan) lists a real device; ``cpu`` needs only a
    runtime. A fixed backend is the only candidate: it never falls back.
    """
    candidates = list(server.fallback) if server.backend == "auto" else [server.backend]
    tried: list[Attempt] = []
    cache: dict[Path, DeviceListing] = {}
    for backend in candidates:
        folder = server.runtimes.get(backend)
        if folder is None or find_executable(folder) is None:
            if backend == "cpu":
                # Every llama.cpp build contains the CPU path: reuse any installed runtime.
                spare = next(
                    (
                        (name, other)
                        for name, other in server.runtimes.items()
                        if name != "cpu" and find_executable(other) is not None
                    ),
                    None,
                )
                if spare is not None:
                    folder = spare[1]
                    note = f"no CPU runtime installed; using the CPU path of the {spare[0]} runtime"
                    threads = server.threads or threads_default()
                    return _finish(
                        server, "cpu", "none", "CPU", folder, tried, note, candidates, threads
                    )
            where = folder.name if folder is not None else "not configured"
            tried.append(Attempt(backend, False, f"runtime not installed ({where})"))
            continue
        if backend == "cpu":
            threads = server.threads or threads_default()
            return _finish(server, "cpu", "none", "CPU", folder, tried, None, candidates, threads)
        device, reason = _check_gpu_backend(backend, folder, server, probe, cache)
        if device is None:
            tried.append(Attempt(backend, False, reason))
            continue
        return _finish(
            server, backend, device.id, device.description, folder, tried, None, candidates, None
        )
    detail = "; ".join(f"{item.backend}: {item.reason}" for item in tried)
    if server.backend == "auto":
        raise BackendSelectionError(
            f"No usable inference backend ({detail}). Install a runtime with "
            "'sbobinai setup' (see docs/runtime-setup.md).",
            tuple(tried),
        )
    raise BackendSelectionError(
        f'[server].backend = "{server.backend}" is not usable ({detail}). A fixed backend never '
        'falls back; fix it or set backend = "auto".',
        tuple(tried),
    )


def _finish(
    server: ServerConfig,
    backend: str,
    device: str,
    device_name: str,
    folder: Path,
    tried: list[Attempt],
    note: str | None,
    candidates: list[str],
    threads: int | None,
) -> Selection:
    attempts = (*tried, Attempt(backend, True, note))
    fallback_used = server.backend == "auto" and backend != candidates[0]
    selection = Selection(
        backend=backend,
        device=device,
        device_name=device_name,
        runtime_dir=folder,
        requested=server.backend,
        fallback_used=fallback_used,
        tried=attempts,
        threads=threads,
        runtime_tag=runtime_tag(folder),
    )
    if fallback_used:
        lines = warning_lines(selection)[:-1]
        selection = replace(selection, warning="; ".join(line.strip() for line in lines))
    return selection

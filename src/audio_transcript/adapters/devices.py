"""Hardware probes: ``llama-server --list-devices``, NVIDIA presence and CPU core count."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..application.backend_select import DeviceListing, find_executable, parse_devices

LIST_DEVICES_TIMEOUT = 30.0


def list_devices(runtime_dir: Path, timeout: float = LIST_DEVICES_TIMEOUT) -> DeviceListing:
    """Run ``llama-server --list-devices`` of a runtime; failures become ``error`` text.

    The environment is inherited, so ``CUDA_VISIBLE_DEVICES`` and ``GGML_VK_VISIBLE_DEVICES``
    hide devices exactly as they do for the real server (useful to simulate a missing GPU).
    """
    executable = find_executable(runtime_dir)
    if executable is None:
        return DeviceListing(error="llama-server executable missing")
    try:
        result = subprocess.run(
            [str(executable), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(runtime_dir),
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return DeviceListing(error=f"--list-devices timed out after {timeout:g} s")
    except OSError as exc:
        return DeviceListing(error=f"--list-devices could not run ({type(exc).__name__})")
    if result.returncode != 0:
        return DeviceListing(error=f"--list-devices failed (exit {result.returncode})")
    return DeviceListing(
        devices=parse_devices((result.stdout or "") + "\n" + (result.stderr or ""))
    )


def nvidia_gpu_present(timeout: float = 15.0) -> bool:
    """True when ``nvidia-smi -L`` lists at least one GPU."""
    located = shutil.which("nvidia-smi")
    if located is None:
        return False
    try:
        result = subprocess.run(
            [located, "-L"], capture_output=True, text=True, timeout=timeout, errors="replace"
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU " in (result.stdout or "")


def physical_cores() -> int | None:
    """Physical CPU cores on Windows (None elsewhere or on failure)."""
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        length = ctypes.c_ulong(0)
        # RelationProcessorCore = 0; the first call reports the buffer size.
        kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
        buffer = ctypes.create_string_buffer(length.value)
        if not kernel32.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(length)):
            return None
        offset = count = 0
        raw = buffer.raw
        while offset < length.value:
            size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
            if size <= 0:
                return None
            count += 1
            offset += size
        return count or None
    except (OSError, AttributeError, ValueError):
        return None


def default_cpu_threads() -> int:
    """Threads for CPU inference: physical cores, else half the logical processors."""
    return physical_cores() or max(1, (os.cpu_count() or 2) // 2)

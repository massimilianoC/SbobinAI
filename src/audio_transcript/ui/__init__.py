"""Console presentation of a run (standard library only).

``resolve_ui_mode`` decides between the live ANSI renderer and the plain status lines; the
renderer itself is a pure function of a ``RunState`` (see ``live.render``).
"""

from __future__ import annotations

import os
import sys

UI_MODES = ("auto", "live", "plain", "jsonl")


def enable_vt_processing() -> bool:
    """Turn on ANSI escape handling of the Windows console; ``True`` where it already works."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enable_virtual_terminal_processing = 0x0004
        if mode.value & enable_virtual_terminal_processing:
            return True
        return bool(
            kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing)
        )
    except Exception:
        return False


def resolve_ui_mode(
    requested: str = "auto",
    *,
    isatty: bool | None = None,
    environ: dict | None = None,
    enable_vt=enable_vt_processing,
) -> str:
    """Return ``live``, ``plain`` or ``jsonl``.

    ``auto`` is ``live`` only when stdout is a terminal, ``TERM`` is not ``dumb``, no ``CI``
    variable is set and the console can render escape sequences; anything else (pipes, logs,
    CI, tests) gets ``plain``, which prints exactly the status lines. An explicit ``live``
    still falls back to ``plain`` when escape sequences cannot be enabled.
    """
    env = os.environ if environ is None else environ
    if requested in {"plain", "jsonl"}:
        return requested
    if requested == "live":
        return "live" if enable_vt() else "plain"
    if isatty is None:
        isatty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    if not isatty:
        return "plain"
    if env.get("TERM", "").lower() == "dumb" or env.get("CI"):
        return "plain"
    return "live" if enable_vt() else "plain"

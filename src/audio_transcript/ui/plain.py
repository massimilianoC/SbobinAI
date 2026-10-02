"""Plain console output: the status lines exactly as the pipeline words them."""

from __future__ import annotations

import sys

from ..application.progress import RunState
from ..application.session import summary_lines


def status_printer(stream=None):
    """``print`` with an immediate flush so lines appear while a long run is in progress."""

    def show(message: str) -> None:
        out = stream if stream is not None else sys.stdout
        print(message, file=out, flush=True)

    return show


def final_summary(state: RunState, paths: dict | None = None) -> list[str]:
    """Lines printed after a live run, in place of the suppressed status lines."""
    lines = ["", "Run summary"]
    lines += [f"  {line}" for line in summary_lines(state)]
    if paths:
        for label, value in paths.items():
            lines.append(f"  {label}: {value}")
    return lines

"""Interactive shell (``sbobinai repl``): a stateful front end for the same commands."""

from __future__ import annotations

import contextlib
import os
import shlex
import sys
from pathlib import Path

from . import __version__, clidoc

# Subcommands that accept --config and --json (derived from the option catalogue).
_TAKES_CONFIG = {
    c.name for c in clidoc.COMMANDS if any(o.flags == ("--config",) for o in c.options)
}
_TAKES_JSON = {c.name for c in clidoc.COMMANDS if any(o.flags == ("--json",) for o in c.options)}


def _enable_history() -> bool:
    """Line editing and history through readline when the platform has it."""
    try:
        import readline  # noqa: F401  (importing it activates history for input())
    except ImportError:
        return False
    return True


def split_line(line: str) -> list[str]:
    """Split a command line; on Windows keep backslashes in paths."""
    if os.name == "nt":
        tokens = shlex.split(line, posix=False)
        return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "'\"" else t for t in tokens]
    return shlex.split(line)


def _model_label(config_path: Path | None) -> str:
    from .config import load_config

    try:
        config = load_config(config_path)
    except ValueError as exc:
        return f"configuration error: {exc}"
    return f"{config.model} ({config.backend})"


def banner(config_path: Path | None) -> str:
    rule = "-" * 60
    config = str(config_path) if config_path else "(none: built-in defaults)"
    return "\n".join(
        [
            rule,
            f"  SbobinAI {__version__} - local transcription shell",
            f"  model : {_model_label(config_path)}",
            f"  config: {config}",
            "  Type a command (run, doctor, events ...), help, or exit.",
            rule,
        ]
    )


def _prompt(color: bool) -> str:
    if color:
        # \001 and \002 keep readline from counting the escape codes as width.
        return "\001\033[1;36m\002sbobinai>\001\033[0m\002 "
    return "sbobinai> "


def _color_enabled() -> bool:
    from .ui.live import color_enabled

    try:
        return bool(sys.stdout.isatty()) and color_enabled(sys.stdout)
    except Exception:
        return False


def _help_text() -> str:
    lines = ["Commands (any subcommand takes its usual arguments):"]
    for command in clidoc.COMMANDS:
        if command.name != "repl":
            lines.append(f"  {command.name:<15}{command.summary}")
    lines += [
        "Built-ins:",
        "  help [command] show this list or the help of a command",
        "  config [PATH]  show or set the --config used for every command",
        "  exit, quit     leave the shell",
    ]
    return "\n".join(lines)


def run_repl(config: Path | None = None, as_json: bool = False) -> int:
    from .cli import main

    _enable_history()
    session_config = config
    color = _color_enabled()
    if not as_json:
        print(banner(session_config))
    while True:
        try:
            line = input(_prompt(color) if sys.stdin.isatty() else "")
        except EOFError:
            print()
            return clidoc.EXIT_OK
        except KeyboardInterrupt:
            print()
            continue
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = split_line(line)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            continue
        word, rest = tokens[0], tokens[1:]
        if word in {"exit", "quit"}:
            return clidoc.EXIT_OK
        if word == "config":
            if rest:
                session_config = Path(rest[0])
            print(f"config: {session_config if session_config else '(none)'}")
            continue
        if word == "help" and not rest:
            print(_help_text())
            continue
        if word == "repl":
            print("ERROR: already inside the shell", file=sys.stderr)
            continue
        argv = [word, *rest]
        if word in _TAKES_CONFIG and session_config is not None and "--config" not in rest:
            argv += ["--config", str(session_config)]
        if as_json and word in _TAKES_JSON and "--json" not in rest:
            argv.append("--json")
        status = clidoc.EXIT_OK
        try:
            status = main(argv)
        except SystemExit as exc:  # argparse usage errors and --help
            status = exc.code if isinstance(exc.code, int) else clidoc.EXIT_OK
        except KeyboardInterrupt:
            status = clidoc.EXIT_INTERRUPTED
            print()
        if status and not as_json:
            print(f"[exit status {status}]")
        with contextlib.suppress(Exception):
            sys.stdout.flush()

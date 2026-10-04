"""Renderers of the command line documentation (help text, manual, Markdown, JSON, skill).

All content comes from ``clidoc``; nothing here is hand-written per command.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from . import clidoc
from .clidoc import (
    AGENT_PRINCIPLES,
    ALIASES,
    COMMAND_BY_NAME,
    COMMAND_GROUPS,
    COMMANDS,
    CONFIG_KEYS,
    ENVIRONMENT,
    ERROR_CODES,
    EVENT_TYPES,
    EXIT_CODES,
    FILES,
    GLOBAL_NOTES,
    HELP_WIDTH,
    PROGRAM,
    RECIPES,
    SCHEMA_VERSION,
    SEE_ALSO,
    TOP_DESCRIPTION,
    WORKFLOW,
    Command,
    Opt,
    config_default,
    config_flag,
    grouped_options,
    visible_options,
)

_JSON_TYPES = {
    "str": "string",
    "path": "path",
    "int": "integer",
    "float": "number",
    "floatlist": "list of numbers",
}


_SHORT_EXIT = {
    0: "success (jobs completed or already done)",
    1: "job failed or incomplete, or runtime error",
    2: "usage or configuration error",
    130: "interrupted (Ctrl+C)",
}


def wrap(text: str, indent: int = 0, width: int = HELP_WIDTH, subsequent: int | None = None) -> str:
    pad = " " * indent
    sub = " " * (indent if subsequent is None else subsequent)
    return textwrap.fill(
        text, width=width, initial_indent=pad, subsequent_indent=sub, break_long_words=False
    )


def paragraphs(items, indent: int = 0, width: int = HELP_WIDTH) -> str:
    return "\n\n".join(wrap(p, indent, width) for p in items)


# ---------------------------------------------------------------- option formatting


def opt_type(opt: Opt) -> str:
    if opt.kind in {"flag", "bool"}:
        return "boolean"
    return _JSON_TYPES[opt.type]


def flag_string(opt: Opt) -> str:
    """Flags with metavar as shown in the manual, for example ``--file PATH``."""
    if opt.positional:
        return opt.metavar or opt.name.upper()
    if opt.kind == "bool":
        base = opt.long_flag
        return f"{base}, --no-{base[2:]}"
    if opt.kind == "flag":
        return ", ".join(opt.flags)
    if opt.choices and not opt.metavar:
        value = "{" + ",".join(opt.choices) + "}"
    else:
        value = opt.metavar or opt.name.upper()
    return f"{opt.flags[0]} {value}"


def option_help(opt: Opt, with_default: bool = True) -> str:
    text = opt.help
    shown = opt.shown_default() if with_default else None
    if shown is not None and opt.kind != "flag":
        text = f"{text} [default: {shown}]"
    return text


# ---------------------------------------------------------------- per-command help


def command_description(cmd: Command) -> str:
    return wrap(cmd.summary) + "\n\n" + paragraphs(cmd.description)


def _example_lines(cmd: Command, indent: int = 2) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    for ex in cmd.examples:
        lines.append(f"{pad}# {ex.title}")
        if ex.posix is not None and ex.posix != ex.command:
            lines.append(f"{pad}Windows: {ex.command}")
            lines.append(f"{pad}POSIX:   {ex.posix}")
        else:
            lines.append(f"{pad}{ex.command}")
        lines.append("")
    if lines:
        lines.pop()
    return lines


def command_epilog(cmd: Command) -> str:
    parts: list[str] = []
    if cmd.positional_help:
        parts.append(wrap(cmd.positional_help))
    if cmd.notes:
        parts.append("Notes:\n" + "\n".join(wrap("- " + n, 2, subsequent=4) for n in cmd.notes))
    if cmd.files:
        parts.append("Files:\n" + "\n".join(wrap(f, 2) for f in cmd.files))
    if cmd.json_output:
        parts.append("With --json:\n" + wrap(cmd.json_output, 2))
    if cmd.related:
        parts.append("See also:\n" + wrap(", ".join(f"{PROGRAM} help {r}" for r in cmd.related), 2))
    parts.append("Examples:\n" + "\n".join(_example_lines(cmd)))
    codes = "\n".join(wrap(f"{code:<4}{text}", 2, subsequent=6) for code, text in cmd.exit_status)
    parts.append("Exit status:\n" + codes)
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------- top level overview


def overview() -> str:
    out = [f"usage: {PROGRAM} [--version] [-h] <command> [options]", ""]
    out.append(wrap(f"{PROGRAM} - private, local transcription of long audio and video."))
    out.append("")
    out.append(paragraphs(TOP_DESCRIPTION))
    out.append("")
    out.append("Typical workflow:")
    for number, step in enumerate(WORKFLOW, 1):
        out.append(f"  {number}. {step}")
    out.append("")
    for group in COMMAND_GROUPS:
        out.append(f"{group}:")
        for cmd in COMMANDS:
            if cmd.group == group:
                out.append(wrap(f"{cmd.name:<15}{cmd.summary}", 2, subsequent=17))
        out.append("")
    out.append("Global notes:")
    for note in GLOBAL_NOTES:
        out.append(wrap("- " + note, 2, subsequent=4))
    out.append("")
    out.append("More help:")
    out.append(f"  {PROGRAM} help <command>     same as <command> --help (options, examples)")
    out.append(f"  {PROGRAM} man                full manual; --format markdown|json|skill")
    out.append(f"  {PROGRAM} man --format json  machine-readable interface description")
    out.append("")
    out.append("Examples:")
    out.append("  sbobinai doctor --config config.local.toml")
    out.append("  sbobinai run --config config.local.toml --file input/talk.mp4 --json")
    out.append("")
    out.append("Exit status:")
    for code, _name, _text in EXIT_CODES:
        out.append(f"  {code:<4}{_SHORT_EXIT[code]}")
    out.append("  (full table: sbobinai man)")
    out.append("")
    out.append(wrap(f"Aliases of the command: {', '.join(ALIASES)}."))
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- synopsis


def synopsis(cmd: Command) -> str:
    parts = [PROGRAM, cmd.name]
    for opt in cmd.options:
        if opt.positional:
            parts.append(f"[{opt.metavar or opt.name.upper()}]")
        elif opt.required and not opt.hidden:
            parts.append(f"{opt.long_flag} {opt.metavar or opt.name.upper()}")
    if any(not o.positional and not o.required and not o.hidden for o in cmd.options):
        parts.append("[options]")
    return " ".join(parts)


# ---------------------------------------------------------------- manual (text)


def _option_block(opt: Opt, indent: int = 4) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}{flag_string(opt)}"]
    extra = []
    if opt.config:
        extra.append(f"Config key: {opt.config}.")
    if opt.kind == "append":
        extra.append("May be repeated.")
    if opt.required:
        extra.append("Required.")
    text = option_help(opt)
    if extra:
        text = text + " " + " ".join(extra)
    lines.append(wrap(text, indent + 4))
    return lines


def _config_row(key) -> tuple[str, str]:
    if key.section == "audio_transcript":
        return config_flag(key.key) or "file only", config_default(key.key)
    return "file only", ("unset" if key.key == "gpu_layers" else "required")


def _config_lines() -> list[str]:
    lines: list[str] = []
    for key in CONFIG_KEYS:
        flag, default = _config_row(key)
        lines.append(f"  [{key.section}] {key.key}")
        lines.append(
            wrap(
                f"{key.description} Type: {key.type}. Default: {default}. "
                f"CLI: {flag}. Job identity: {key.identity}.",
                6,
            )
        )
    return lines


def man_text(command: str | None = None) -> str:
    commands = [COMMAND_BY_NAME[command]] if command else list(COMMANDS)
    out = ["NAME", f"  {PROGRAM} - private, local transcription of long recordings", ""]
    out.append("SYNOPSIS")
    out.extend(f"  {synopsis(c)}" for c in commands)
    out.append("")
    out.append("DESCRIPTION")
    out.append(paragraphs(TOP_DESCRIPTION, 2))
    out.append("")
    out.append("  Typical workflow:")
    out.extend(f"    {n}. {s}" for n, s in enumerate(WORKFLOW, 1))
    out.append("")
    out.extend(wrap("- " + n, 2, subsequent=4) for n in GLOBAL_NOTES)
    out.append("")
    out.append("COMMANDS")
    for group in COMMAND_GROUPS:
        listed = [c for c in commands if c.group == group]
        if listed:
            out.append(f"  {group}")
            out.extend(wrap(f"{c.name:<15}{c.summary}", 4, subsequent=19) for c in listed)
    out.append("")
    out.append("OPTIONS")
    for cmd in commands:
        out.append(f"  {PROGRAM} {cmd.name}")
        out.append(paragraphs(cmd.description, 4))
        out.append("")
        for group, opts in grouped_options(cmd):
            out.append(f"    {group}")
            for opt in opts:
                out.extend(_option_block(opt, 6))
            out.append("")
        for note in cmd.notes:
            out.append(wrap("- " + note, 4, subsequent=6))
        if cmd.json_output:
            out.append(wrap("With --json: " + cmd.json_output, 4, subsequent=6))
        out.append("")
        out.append("    Examples")
        out.extend(_example_lines(cmd, 6))
        out.append("")
        out.append("    Exit status")
        out.extend(wrap(f"{c:<4}{t}", 6, subsequent=10) for c, t in cmd.exit_status)
        out.append("")
    if not command:
        out.append("CONFIGURATION")
        out.append(
            wrap(
                "Settings live in a TOML file given with --config: an [audio_transcript] "
                "table (or top-level keys) plus optional [resources] and [server] tables. "
                "Unknown keys are errors. Command line options override the file. "
                "`Job identity` says whether changing the value creates a new version of a "
                "result (yes) or only changes how the run behaves (no).",
                2,
            )
        )
        out.append("")
        out.extend(_config_lines())
        out.append("")
        out.append("FILES")
        for path, text in FILES:
            out.append(f"  {path}")
            out.append(wrap(text, 6))
        out.append("")
        out.append("ENVIRONMENT")
        for name, text in ENVIRONMENT:
            out.append(f"  {name}")
            out.append(wrap(text, 6))
        out.append("")
    out.append("EXIT STATUS")
    for code, name, text in EXIT_CODES:
        out.append(wrap(f"{code:<4}{name}: {text}", 2, subsequent=6))
    out.append("")
    out.append("ERRORS")
    out.append(
        wrap(
            'In JSON mode a failure is {"ok": false, "error": {"code", "message", '
            '"exit_status"}} on stdout. The codes are stable:',
            2,
        )
    )
    for code, status, text in ERROR_CODES:
        out.append(wrap(f"{code:<22}{status:<5}{text}", 4, subsequent=31))
    out.append("")
    if not command:
        out.append("EVENTS")
        out.append(wrap("Event types of events.jsonl (see docs/events.md):", 2))
        for kind, payload in EVENT_TYPES:
            out.append(wrap(f"{kind:<16}{payload}", 4, subsequent=20))
        out.append("")
        out.append("EXAMPLES")
        for rec in RECIPES:
            out.append(f"  # {rec.title}")
            out.extend("  " + line for line in rec.posix.split("\n"))
            out.append("")
        out.append("AGENT USAGE")
        out.extend(wrap("- " + p, 2, subsequent=4) for p in AGENT_PRINCIPLES)
        out.append("")
        out.append("SEE ALSO")
        out.extend(f"  {s}" for s in SEE_ALSO)
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- Markdown


def _md_code(lines: list[str], lang: str = "text") -> list[str]:
    return [f"```{lang}", *lines, "```", ""]


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def markdown(command: str | None = None) -> str:
    commands = [COMMAND_BY_NAME[command]] if command else list(COMMANDS)
    workflow = []
    for number, step in enumerate(WORKFLOW, 1):
        first, _, rest = step.partition(" ")
        workflow.append(f"{number}. `{first}` {rest.strip()}")
    out = [
        "# SbobinAI command-line reference",
        "",
        "<!-- Generated by `sbobinai man --format markdown`; do not edit by hand. "
        "Regenerate with `python scripts/gen-cli-docs.py`. -->",
        "",
        f"`{PROGRAM}` (aliases: {', '.join(f'`{a}`' for a in ALIASES)}) - "
        "private, local transcription of long recordings.",
        "",
        "## Contents",
        "",
        "- [Description](#description)",
        "- [Commands](#commands)",
        "- [Options per command](#options-per-command)",
        "- [Configuration](#configuration)",
        "- [Files](#files)",
        "- [Environment](#environment)",
        "- [Exit status](#exit-status)",
        "- [Errors in JSON mode](#errors-in-json-mode)",
        "- [Events](#events)",
        "- [Agent usage](#agent-usage)",
        "- [Recipes](#recipes)",
        "- [See also](#see-also)",
        "",
        "## Description",
        "",
        *TOP_DESCRIPTION,
        "",
        "Typical workflow:",
        "",
        *workflow,
        "",
        *[f"- {n}" for n in GLOBAL_NOTES],
        "",
        "## Commands",
        "",
        "| Group | Command | Summary |",
        "| --- | --- | --- |",
    ]
    for group in COMMAND_GROUPS:
        for cmd in commands:
            if cmd.group == group:
                out.append(f"| {group} | [`{cmd.name}`](#{cmd.name}) | {_md_cell(cmd.summary)} |")
    out += ["", "## Options per command", ""]
    for cmd in commands:
        out += [f"### {cmd.name}", "", cmd.summary, "", *_md_code([synopsis(cmd)])]
        out += [*cmd.description, ""]
        for group, opts in grouped_options(cmd):
            out += [f"**{group}**", "", "| Option | Meaning | Config key |", "| --- | --- | --- |"]
            for opt in opts:
                extra = " Repeatable." if opt.kind == "append" else ""
                extra += " Required." if opt.required else ""
                key = f"`{opt.config}`" if opt.config else ""
                out.append(
                    f"| `{flag_string(opt)}` | {_md_cell(option_help(opt) + extra)} | {key} |"
                )
            out.append("")
        if cmd.notes:
            out += ["Notes:", "", *[f"- {n}" for n in cmd.notes], ""]
        if cmd.files:
            out += ["Files:", "", *[f"- {f}" for f in cmd.files], ""]
        if cmd.json_output:
            out += [f"With `--json`: {cmd.json_output}", ""]
        out += ["Examples:", ""]
        out += _md_code(_example_lines(cmd, 0), "text")
        out += ["Exit status:", "", "| Code | Meaning |", "| --- | --- |"]
        out += [f"| {c} | {_md_cell(t)} |" for c, t in cmd.exit_status]
        out.append("")
        if cmd.related:
            out += ["See also: " + ", ".join(f"[`{r}`](#{r})" for r in cmd.related), ""]
    out += [
        "## Configuration",
        "",
        "Settings live in a TOML file given with `--config`: an `[audio_transcript]` table "
        "(or top-level keys) plus optional `[resources]` and `[server]` tables. Unknown keys "
        "are errors. Command-line options override the file. **Job identity** says whether "
        "changing a value creates a new version of a result (yes) or only changes how the "
        "run behaves (no).",
        "",
        "| Table | Key | Type | Default | CLI flag | Job identity |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key in CONFIG_KEYS:
        flag, default = _config_row(key)
        shown_flag = f"`{flag}`" if flag != "file only" else "file only"
        out.append(
            f"| `[{key.section}]` | `{key.key}` | {key.type} | {_md_cell(default)} | "
            f"{shown_flag} | {key.identity} |"
        )
    out += ["", "## Files", "", "| Path | Content |", "| --- | --- |"]
    out += [f"| `{p}` | {_md_cell(t)} |" for p, t in FILES]
    out += ["", "## Environment", "", "| Variable | Effect |", "| --- | --- |"]
    out += [f"| `{n}` | {t} |" for n, t in ENVIRONMENT]
    out += ["", "## Exit status", "", "| Code | Name | Meaning |", "| --- | --- | --- |"]
    out += [f"| {c} | {n} | {_md_cell(t)} |" for c, n, t in EXIT_CODES]
    out += [
        "",
        "## Errors in JSON mode",
        "",
        'With `--json` a failure is `{"ok": false, "error": {"code": "...", "message": '
        '"...", "exit_status": N}}` on stdout; stderr stays empty. The codes are stable.',
        "",
        "| Code | Exit status | Meaning |",
        "| --- | --- | --- |",
    ]
    out += [f"| `{c}` | {s} | {_md_cell(t)} |" for c, s, t in ERROR_CODES]
    out += ["", "## Events", "", "| Type | Payload |", "| --- | --- |"]
    out += [f"| `{k}` | {_md_cell(p)} |" for k, p in EVENT_TYPES]
    out += ["", "Details: [events.md](events.md).", "", "## Agent usage", ""]
    out += [f"- {p}" for p in AGENT_PRINCIPLES]
    out += ["", "## Recipes", ""]
    for rec in RECIPES:
        out += [f"**{rec.title}**", "", *_md_code(rec.posix.split("\n"), "sh")]
        if rec.powershell:
            out += _md_code(rec.powershell.split("\n"), "powershell")
    out += ["## See also", ""]
    out += [f"- {s}" for s in SEE_ALSO]
    return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------- JSON


def _json_default(opt: Opt):
    if opt.kind == "flag":
        return False
    if opt.config is not None:
        value = getattr(clidoc.AppConfig, opt.config, None)
    elif opt.default is clidoc.NO_DEFAULT:
        return None
    else:
        value = opt.default
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return list(value)
    return value


def option_dict(opt: Opt) -> dict:
    return {
        "name": opt.name,
        "flags": list(opt.flags),
        "type": opt_type(opt),
        "kind": opt.kind,
        "default": _json_default(opt),
        "default_text": opt.shown_default(),
        "choices": list(opt.choices) if opt.choices else None,
        "help": opt.help,
        "group": opt.group,
        "metavar": opt.metavar,
        "config_key": opt.config,
        "required": opt.required,
        "repeatable": opt.kind == "append",
        "negatable": opt.kind == "bool",
        "positional": opt.positional,
    }


def command_dict(cmd: Command) -> dict:
    return {
        "name": cmd.name,
        "group": cmd.group,
        "summary": cmd.summary,
        "synopsis": synopsis(cmd),
        "description": list(cmd.description),
        "notes": list(cmd.notes),
        "options": [option_dict(o) for o in visible_options(cmd)],
        "examples": [
            {"title": e.title, "windows": e.command, "posix": e.posix or e.command}
            for e in cmd.examples
        ],
        "exit_status": [{"code": c, "meaning": t} for c, t in cmd.exit_status],
        "files": list(cmd.files),
        "related": list(cmd.related),
        "json_output": cmd.json_output,
    }


def interface(command: str | None = None) -> dict:
    commands = [COMMAND_BY_NAME[command]] if command else list(COMMANDS)
    config = []
    for key in CONFIG_KEYS:
        flag, default = _config_row(key)
        config.append(
            {
                "table": key.section,
                "key": key.key,
                "type": key.type,
                "default": None if key.section != "audio_transcript" else default,
                "cli_flag": None if flag == "file only" else flag,
                "job_identity": key.identity,
                "description": key.description,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "program": PROGRAM,
        "aliases": list(ALIASES),
        "summary": TOP_DESCRIPTION[0],
        "commands": [command_dict(c) for c in commands],
        "exit_codes": [{"code": c, "name": n, "meaning": t} for c, n, t in EXIT_CODES],
        "error_codes": [{"code": c, "exit_status": s, "meaning": t} for c, s, t in ERROR_CODES],
        "config": config,
        "environment": [{"name": n, "meaning": t} for n, t in ENVIRONMENT],
        "files": [{"path": p, "meaning": t} for p, t in FILES],
        "event_types": [{"type": k, "payload": p} for k, p in EVENT_TYPES],
        "agent_principles": list(AGENT_PRINCIPLES),
        "recipes": [
            {"title": r.title, "posix": r.posix, "powershell": r.powershell} for r in RECIPES
        ],
    }


def json_text(command: str | None = None) -> str:
    return json.dumps(interface(command), indent=2, ensure_ascii=True) + "\n"


# ---------------------------------------------------------------- SKILL.md

SKILL_DESCRIPTION = (
    "Transcribe long local audio and video files with the SbobinAI command line "
    "(sbobinai): run unattended jobs, check readiness, read transcripts and catalogs, "
    "follow progress events. Use when asked to transcribe a recording, to inspect or "
    "re-run transcription results, or to set up or diagnose the local transcription runtime."
)


def skill_text() -> str:
    aliases = ", ".join(f"`{a}`" for a in ALIASES)
    out = [
        "---",
        f"name: {PROGRAM}",
        f"description: {SKILL_DESCRIPTION}",
        "---",
        "",
        "<!-- Generated by `sbobinai man --format skill`; do not edit by hand. -->",
        "",
        "# SbobinAI command line",
        "",
        *TOP_DESCRIPTION,
        "",
        f"Invoke it as `sbobinai` (aliases: {aliases}), or `python -m audio_transcript` from "
        "the repository environment. Pass `--config config.local.toml` to every command that "
        "accepts it.",
        "",
        "## Workflow",
        "",
        *[f"{n}. {s}" for n, s in enumerate(WORKFLOW, 1)],
        "",
        "## Commands",
        "",
    ]
    for group in COMMAND_GROUPS:
        out += [f"### {group}", ""]
        for cmd in COMMANDS:
            if cmd.group == group:
                out.append(f"- `{cmd.name}` - {cmd.summary} Usage: `{synopsis(cmd)}`")
        out.append("")
    out += ["## Agent guidance", "", *[f"- {p}" for p in AGENT_PRINCIPLES], ""]
    out += ["## Common workflows", ""]
    for rec in RECIPES:
        out += [f"**{rec.title}**", "", *_md_code(rec.posix.split("\n"), "sh")]
    out += [
        "## Exit status",
        "",
        "| Code | Meaning |",
        "| --- | --- |",
        *[f"| {c} | {n}: {_md_cell(t)} |" for c, n, t in EXIT_CODES],
        "",
        "## JSON errors",
        "",
        'With `--json`: `{"ok": false, "error": {"code", "message", "exit_status"}}` on stdout.',
        "",
        "| Code | Exit status | Meaning |",
        "| --- | --- | --- |",
        *[f"| `{c}` | {s} | {_md_cell(t)} |" for c, s, t in ERROR_CODES],
        "",
        "## Events",
        "",
        "`events.jsonl` lines are JSON objects with `schema_version`, `seq`, `ts`, `run_id`, "
        "`type` and `data`. Types: " + ", ".join(f"`{k}`" for k, _ in EVENT_TYPES) + ". "
        "Events never contain transcript text.",
        "",
        "## Where results are",
        "",
        *[f"- `{p}` - {t}" for p, t in FILES],
        "",
        "Full manual: `sbobinai man`; machine-readable: `sbobinai man --format json`.",
    ]
    return "\n".join(out).rstrip("\n") + "\n"


def render(fmt: str, command: str | None = None) -> str:
    if fmt == "text":
        return man_text(command)
    if fmt == "markdown":
        return markdown(command)
    if fmt == "json":
        return json_text(command)
    if fmt == "skill":
        return skill_text()
    raise ValueError(f"unknown format: {fmt}")

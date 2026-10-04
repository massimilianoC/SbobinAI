"""Single source of truth for the command line documentation.

Everything a human or an agent can read about the interface lives here: the option
catalogue (from which ``cli.py`` builds argparse), command descriptions, examples, exit
and error codes, the configuration key table, files, environment and the agent guide.
``climan.py`` renders it as ``--help`` text, the manual, Markdown, JSON and SKILL.md, so
the documents can never drift from the parser. Text is ASCII only (Windows consoles).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig

PROGRAM = "sbobinai"
ALIASES = ("audio-transcript", "cli-anything-sbobinai")
SCHEMA_VERSION = 1
HELP_WIDTH = 78

# Exit status: the single table for the whole tool (cli.py returns only these values).
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

EXIT_CODES = (
    (
        EXIT_OK,
        "success",
        "Success: every job completed or was skipped as already done, the command "
        "finished, or the user cancelled before anything ran (wizard). Ctrl+C ending "
        "`watch` or `events --follow` is a normal stop and also returns 0.",
    ),
    (
        EXIT_FAILED,
        "failed",
        "A job failed or ended incomplete, a check failed (doctor), a resource or "
        "runtime problem occurred (backend unavailable, another run holds the lock, "
        "no such run, download failure, output file exists).",
    ),
    (
        EXIT_USAGE,
        "usage",
        "Usage or configuration error: unknown or invalid option, invalid or "
        "unreadable configuration or answers file, missing [server] table, wizard "
        "without a terminal.",
    ),
    (
        EXIT_INTERRUPTED,
        "interrupted",
        "Interrupted by Ctrl+C during a run; job state stays recoverable.",
    ),
)

# Stable machine-readable error codes (JSON mode: {"ok": false, "error": {"code": ...}}).
ERROR_CODES = (
    ("usage", EXIT_USAGE, "Invalid command line (unknown command or option, bad value)."),
    ("config_invalid", EXIT_USAGE, "The configuration file or a value overriding it is invalid."),
    ("answers_invalid", EXIT_USAGE, "The wizard answers file or an answer is invalid."),
    ("interactive_required", EXIT_USAGE, "The wizard needs a terminal or --answers."),
    ("server_not_configured", EXIT_USAGE, "The configuration has no [server] table."),
    ("backend_unavailable", EXIT_FAILED, "The transcription backend or a tool it needs failed."),
    ("lock_held", EXIT_FAILED, "Another sbobinai process holds the process lock."),
    ("job_failed", EXIT_FAILED, "At least one job failed (see jobs[].error)."),
    ("job_incomplete", EXIT_FAILED, "At least one job ended incomplete; re-run to retry gaps."),
    ("checks_failed", EXIT_FAILED, "doctor found at least one failing check."),
    ("run_not_found", EXIT_FAILED, "events: no run exists (for example latest with no runs yet)."),
    ("events_missing", EXIT_FAILED, "events: the requested run id has no events file."),
    ("config_exists", EXIT_FAILED, "init-config: the output file exists (use --force)."),
    ("resource_conflict", EXIT_FAILED, "setup: a file exists but does not match the pinned one."),
    ("insufficient_space", EXIT_FAILED, "setup: not enough free disk space at the store."),
    ("download_failed", EXIT_FAILED, "setup: at least one download failed verification."),
    ("migration_failed", EXIT_FAILED, "migrate-layout: at least one job could not be moved."),
    ("runtime_error", EXIT_FAILED, "Any other runtime error (message holds the details)."),
    ("interrupted", EXIT_INTERRUPTED, "Interrupted by Ctrl+C."),
)

GROUP_ORDER = (
    "Inputs",
    "Transcription",
    "Speech detection",
    "Recovery",
    "Output and UI",
    "Server/runtime",
    "Paths",
    "Behaviour",
)

NO_DEFAULT = object()


@dataclass(frozen=True)
class Opt:
    """One command line option; ``config`` links it to an ``AppConfig`` field."""

    flags: tuple[str, ...]
    group: str
    help: str
    kind: str = "value"  # value | flag | bool | append
    type: str = "str"  # str | path | int | float | floatlist
    dest: str | None = None
    default: object = NO_DEFAULT
    default_text: str | None = None
    choices: tuple[str, ...] | None = None
    metavar: str | None = None
    config: str | None = None
    required: bool = False
    hidden: bool = False
    positional: bool = False
    nargs: str | None = None

    @property
    def name(self) -> str:
        return (self.dest or self.flags[0].lstrip("-").replace("-", "_")).replace("-", "_")

    @property
    def long_flag(self) -> str:
        return next((f for f in self.flags if f.startswith("--")), self.flags[0])

    def shown_default(self) -> str | None:
        """Default as documented text, or None when there is nothing to show."""
        if self.default_text is not None:
            return self.default_text
        if self.config is not None:
            return format_default(getattr(AppConfig, self.config, None))
        if self.default is NO_DEFAULT or self.default is None or self.kind in {"flag", "append"}:
            return None
        return format_default(self.default)


def format_default(value: object) -> str:
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple | list):
        return ",".join(str(v) for v in value) if value else "none"
    return str(value)


@dataclass(frozen=True)
class Example:
    title: str
    command: str
    posix: str | None = None  # POSIX form when it differs from ``command`` (Windows form)


@dataclass(frozen=True)
class Command:
    name: str
    group: str
    summary: str
    description: tuple[str, ...]
    options: tuple[Opt, ...]
    examples: tuple[Example, ...]
    exit_status: tuple[tuple[int, str], ...]
    notes: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    json_output: str = ""
    positional_help: str = ""


def _cfg(flags, group, help, **kw):
    """Option backed by an AppConfig field named after its first long flag."""
    flag = next(f for f in flags if f.startswith("--"))
    key = kw.pop("config", flag.lstrip("-").replace("-", "_"))
    return Opt(tuple(flags), group, help, config=key, **kw)


# --------------------------------------------------------------------------------------
# Option catalogue shared by run, watch, wizard and doctor
# --------------------------------------------------------------------------------------

CONFIG_OPT = Opt(
    ("--config",),
    "Paths",
    "Path to a TOML configuration file (for example config.local.toml). Optional: "
    "there is no automatic discovery, so without it the built-in defaults apply. "
    "Every other option overrides the same-named key of this file.",
    type="path",
    metavar="PATH",
)
JSON_OPT = Opt(
    ("--json",),
    "Output and UI",
    "Machine-readable output: one JSON object on stdout (run/watch/wizard stream "
    'JSON Lines events and end with a {"type": "result"} object). Errors become '
    '{"ok": false, "error": {...}} on stdout and nothing human is written to stderr.',
    kind="flag",
)

SHARED_OPTS: tuple[Opt, ...] = (
    # ---- Inputs
    Opt(
        ("--file",),
        "Inputs",
        "Process this media file instead of discovering --input-dir; repeatable. "
        "Files outside the input folder are transcribed but never archived.",
        kind="append",
        type="path",
        metavar="PATH",
    ),
    Opt(
        ("--limit",),
        "Inputs",
        "Process at most N of the discovered files (a positive integer; run only).",
        type="int",
        metavar="N",
        default_text="all files",
    ),
    _cfg(
        ("--max-duration",),
        "Inputs",
        "Transcribe only the first SECONDS of each file (a bounded quick check). "
        "It creates a separate version and keeps the input file in place.",
        type="float",
        metavar="SECONDS",
        default_text="unlimited",
    ),
    _cfg(
        ("--max-file-size",),
        "Inputs",
        "Reject input files larger than BYTES (the job is recorded as failed).",
        type="int",
        metavar="BYTES",
        default_text="unlimited",
    ),
    _cfg(
        ("--sample-rate",),
        "Inputs",
        "Sample rate in Hz of the mono audio FFmpeg extracts for the model.",
        type="int",
        metavar="HZ",
    ),
    # ---- Transcription
    _cfg(
        ("--language",),
        "Transcription",
        "Spoken language code (for example it, en, de) or auto for no hint. "
        "Qwen3-ASR constrains its output to it unless --no-force-language.",
        metavar="CODE",
        default_text="auto (no hint)",
    ),
    _cfg(
        ("--chunk-seconds",),
        "Transcription",
        "Maximum chunk length in seconds; the speech detector cuts at pauses and "
        "never exceeds it. Subtitle times are chunk boundaries, not word timings.",
        type="float",
        metavar="SECONDS",
    ),
    _cfg(
        ("--prompt",),
        "Transcription",
        "Context or instructions sent with every chunk (topic, names, jargon); "
        "overrides the built-in prompt. Part of the version identity.",
        metavar="TEXT",
        default_text="built-in transcript-only prompt",
    ),
    Opt(
        ("--prompt-file",),
        "Transcription",
        "Read the prompt from a UTF-8 file; mutually exclusive with --prompt.",
        type="path",
        metavar="PATH",
    ),
    _cfg(
        ("--response-mode",),
        "Transcription",
        "How the model is asked to answer: json or plain for generic models, "
        "qwen3-asr for Qwen3-ASR GGUF on llama.cpp.",
        choices=("json", "plain", "qwen3-asr"),
    ),
    _cfg(
        ("--temperature",),
        "Transcription",
        "Sampling temperature, 0 to 2 (0 is greedy and reproducible).",
        type="float",
        metavar="T",
    ),
    _cfg(
        ("--seed",),
        "Transcription",
        "Sampling seed, 0 to 4294967295.",
        type="int",
        metavar="N",
    ),
    _cfg(
        ("--max-tokens",),
        "Transcription",
        "Upper bound of generated tokens per chunk (llama.cpp).",
        type="int",
        metavar="N",
    ),
    _cfg(
        ("--min-tokens",),
        "Transcription",
        "Token cap floor per chunk: cap = min(max-tokens, min-tokens + "
        "tokens-per-second * chunk seconds).",
        type="int",
        metavar="N",
    ),
    _cfg(
        ("--tokens-per-second",),
        "Transcription",
        "Token allowance per second of audio in the per-chunk cap (see --min-tokens).",
        type="float",
        metavar="RATE",
    ),
    _cfg(
        ("--compression-ratio-threshold",),
        "Transcription",
        "Whisper-style gzip compression ratio above which a transcript counts as "
        "a loop and triggers the recovery ladder.",
        type="float",
        metavar="RATIO",
    ),
    _cfg(
        ("--repeat-penalty",),
        "Transcription",
        "llama.cpp repetition penalty; 1.0 disables it.",
        type="float",
        metavar="X",
    ),
    _cfg(
        ("--dry-multiplier",),
        "Transcription",
        "llama.cpp DRY sampler multiplier; 0 disables it.",
        type="float",
        metavar="X",
    ),
    _cfg(
        ("--force-language",),
        "Transcription",
        "Qwen3-ASR: constrain output to --language instead of auto-detection "
        "(--no-force-language disables).",
        kind="bool",
    ),
    _cfg(
        ("--collect-logprobs",),
        "Transcription",
        "Request token log-probabilities for the uncalibrated confidence proxy "
        "(--no-collect-logprobs keeps responses small and faster).",
        kind="bool",
    ),
    _cfg(
        ("--parallel-requests",),
        "Transcription",
        "Concurrent chunk requests, 1 to 8; must not exceed the server parallel "
        "slots. Values above 1 create a distinct version.",
        type="int",
        metavar="N",
    ),
    # ---- Speech detection
    _cfg(
        ("--vad",),
        "Speech detection",
        "Speech detector: silero (ONNX model file), energy (FFmpeg only) or none "
        "(fixed-length chunks, silence reaches the model).",
        choices=("silero", "energy", "none"),
    ),
    _cfg(
        ("--vad-model-path",),
        "Speech detection",
        "Silero ONNX model file; a relative config value resolves against [resources].store.",
        type="path",
        metavar="PATH",
    ),
    _cfg(
        ("--vad-threshold",),
        "Speech detection",
        "Speech probability threshold, greater than 0 and less than 1.",
        type="float",
        metavar="P",
    ),
    _cfg(
        ("--vad-min-speech-seconds",),
        "Speech detection",
        "Discard speech segments shorter than SECONDS.",
        type="float",
        metavar="SECONDS",
    ),
    _cfg(
        ("--vad-min-silence-seconds",),
        "Speech detection",
        "Minimum silence in seconds that ends a speech segment.",
        type="float",
        metavar="SECONDS",
    ),
    _cfg(
        ("--vad-speech-pad-seconds",),
        "Speech detection",
        "Padding in seconds added around each speech segment.",
        type="float",
        metavar="SECONDS",
    ),
    _cfg(
        ("--vad-max-merge-gap-seconds",),
        "Speech detection",
        "Merge neighbouring segments separated by less than SECONDS (within the chunk limit).",
        type="float",
        metavar="SECONDS",
    ),
    _cfg(
        ("--vad-energy-margin-db",),
        "Speech detection",
        "Energy detector only: dB above the noise floor that counts as speech.",
        type="float",
        metavar="DB",
    ),
    # ---- Recovery
    _cfg(
        ("--fallback-temperatures",),
        "Recovery",
        "Comma-separated temperatures (0 to 2) retried after degenerate output; "
        "an empty value disables them. Example: 0.2,0.4",
        type="floatlist",
        metavar="LIST",
    ),
    _cfg(
        ("--split-on-failure",),
        "Recovery",
        "Split a still-failing chunk once at its quietest pause (--no-split-on-failure disables).",
        kind="bool",
    ),
    _cfg(
        ("--context-free-fallback",),
        "Recovery",
        "Retry a chunk without the prompt/context when the model echoed it "
        "(--no-context-free-fallback disables).",
        kind="bool",
    ),
    _cfg(
        ("--retries",),
        "Recovery",
        "Extra attempts after a transient request failure, 0 or more (not part of "
        "the version identity).",
        type="int",
        metavar="N",
    ),
    # ---- Output and UI
    _cfg(
        ("--ui",),
        "Output and UI",
        "Console output: live (progress display), plain (status lines), jsonl "
        "(event stream on stdout) or auto (live only on an interactive terminal).",
        choices=("auto", "live", "plain", "jsonl"),
    ),
    _cfg(
        ("--monitor",),
        "Output and UI",
        "Sample CPU, RAM and NVIDIA GPU use during the run (--no-monitor disables).",
        kind="bool",
    ),
    _cfg(
        ("--monitor-interval",),
        "Output and UI",
        "Seconds between resource samples, 0.1 to 3600.",
        type="float",
        metavar="SECONDS",
    ),
    _cfg(
        ("--intermediate-interval-seconds",),
        "Output and UI",
        "Minimum seconds between rewrites of the intermediate exports; 0 rewrites "
        "after every chunk. The checkpoint is always written per chunk.",
        type="float",
        metavar="SECONDS",
    ),
    # ---- Server/runtime
    _cfg(
        ("--backend",),
        "Server/runtime",
        "Inference backend: llamacpp (local llama-server over HTTP), nexa "
        "(historical Nexa runtime) or mock (tests; never real transcripts).",
        choices=("nexa", "mock", "llamacpp"),
    ),
    _cfg(
        ("--model",),
        "Server/runtime",
        "Model name: the llama-server alias for llamacpp. Default depends on the "
        "backend (qwen2-audio-7b for llamacpp, qwen2audio for nexa, mock for mock).",
        metavar="NAME",
        default_text="depends on backend",
    ),
    _cfg(
        ("--base-url",),
        "Server/runtime",
        "llama.cpp server base URL, for example http://127.0.0.1:8088.",
        metavar="URL",
    ),
    _cfg(
        ("--timeout",),
        "Server/runtime",
        "llama.cpp request timeout in seconds.",
        type="float",
        metavar="SECONDS",
    ),
    _cfg(
        ("--device",),
        "Server/runtime",
        "Device name passed to the legacy Nexa backend (auto, cuda, ...).",
        metavar="NAME",
    ),
    # ---- Paths
    _cfg(
        ("--input-dir",),
        "Paths",
        "Operational queue: media files waiting to be transcribed.",
        type="path",
        metavar="DIR",
    ),
    Opt(
        ("--input",),
        "Paths",
        "Alias of --input-dir.",
        type="path",
        dest="input_dir",
        metavar="DIR",
        hidden=True,
    ),
    _cfg(
        ("--process-dir",),
        "Paths",
        "Working files, checkpoints, raw model responses and run logs.",
        type="path",
        metavar="DIR",
    ),
    _cfg(
        ("--processed-dir",),
        "Paths",
        "Archive for fully transcribed queued sources.",
        type="path",
        metavar="DIR",
    ),
    _cfg(
        ("--output-dir",),
        "Paths",
        "Results: transcripts, reports, source.json and catalog.json.",
        type="path",
        metavar="DIR",
    ),
    _cfg(
        ("--ffmpeg",),
        "Paths",
        "FFmpeg executable name or path.",
        metavar="EXE",
    ),
    _cfg(
        ("--ffprobe",),
        "Paths",
        "FFprobe executable name or path.",
        metavar="EXE",
    ),
    _cfg(
        ("--model-path",),
        "Paths",
        "Existing local model file for the legacy Nexa backend (no download).",
        type="path",
        metavar="PATH",
        default_text="unset",
    ),
    _cfg(
        ("--projector-path",),
        "Paths",
        "Existing local audio projector file required by the legacy Nexa backend.",
        type="path",
        metavar="PATH",
        default_text="unset",
    ),
    # ---- Behaviour
    _cfg(
        ("--force",),
        "Behaviour",
        "Recompute versions that are already complete instead of skipping them "
        "(--no-force restores the default).",
        kind="bool",
    ),
    _cfg(
        ("--prepare-only",),
        "Behaviour",
        "Only probe, detect speech and cut chunks; no model is contacted and nothing is archived.",
        kind="bool",
    ),
    _cfg(
        ("--archive-inputs",),
        "Behaviour",
        "Move fully transcribed queued sources to --processed-dir "
        "(--no-archive-inputs keeps them in the input folder).",
        kind="bool",
    ),
    _cfg(
        ("--watch-interval",),
        "Behaviour",
        "watch: seconds between scans of the input folder.",
        type="float",
        metavar="SECONDS",
    ),
    _cfg(
        ("--stable-scans",),
        "Behaviour",
        "watch: scans, at least 2, a file must keep its size and mtime before it "
        "is processed (avoids half-copied files).",
        type="int",
        metavar="N",
    ),
)

DIR_OPTS = tuple(
    o
    for o in SHARED_OPTS
    if o.name in {"input_dir", "process_dir", "processed_dir", "output_dir"} and not o.hidden
)
_HIDDEN_FOR_NON_RUN = {"file", "limit"}


def _without(opts, names):
    return tuple(o for o in opts if o.name not in names)


def _hidden(opts, names):
    from dataclasses import replace

    return tuple(replace(o, hidden=True) if o.name in names else o for o in opts)


RESOURCE_OPTS = (
    Opt(
        ("--profile",),
        "Behaviour",
        "Profile name from resources.json.",
        metavar="NAME",
        default_text="the manifest default profile",
    ),
    Opt(
        ("--manifest",),
        "Paths",
        "Path to resources.json.",
        type="path",
        metavar="PATH",
        default_text="resources.json next to the package",
    ),
)

_GOOD = ((0, "All jobs completed or were skipped as already done."),)
_RUN_EXIT = (
    (0, "Every job completed, was skipped as already done, or was prepared only."),
    (
        1,
        "A job failed or ended incomplete, the backend is unavailable, or another "
        "sbobinai process holds the lock.",
    ),
    (2, "Invalid option, configuration file or prompt file."),
    (130, "Interrupted by Ctrl+C; re-run to resume."),
)


def _opts(*groups):
    out = []
    for g in groups:
        out.extend(g)
    return tuple(out)


COMMANDS: tuple[Command, ...] = (
    Command(
        name="run",
        group="Transcribe",
        summary="Transcribe the media files in the input folder (unattended).",
        description=(
            "Discovers media files in --input-dir (or takes the files given with "
            "--file), prepares the audio, detects speech, sends chunks to the "
            "configured backend and writes transcripts and reports under "
            "--output-dir. It never asks questions; it is the command for scripts "
            "and agents.",
            "Each combination of file and settings is a version. Re-running skips "
            "versions that are already complete and resumes incomplete ones; "
            "--force recomputes. A fully transcribed queued source is archived to "
            "--processed-dir unless --no-archive-inputs.",
        ),
        options=_opts((CONFIG_OPT,), SHARED_OPTS, (JSON_OPT,)),
        examples=(
            Example(
                "Transcribe everything in input/ with your local settings",
                "sbobinai run --config config.local.toml",
            ),
            Example(
                "One file, Italian, keep it in place",
                "sbobinai run --config config.local.toml --file input\\talk.mp4 "
                "--language it --no-archive-inputs",
                "sbobinai run --config config.local.toml --file input/talk.mp4 "
                "--language it --no-archive-inputs",
            ),
            Example(
                "Quick check of the first 10 minutes (separate version)",
                "sbobinai run --config config.local.toml --max-duration 600 --limit 1",
            ),
            Example(
                "Validate preparation only (no model needed)",
                "sbobinai run --prepare-only --vad energy --limit 1",
            ),
            Example(
                "Stream machine events, then read the result line",
                "sbobinai run --config config.local.toml --json",
            ),
            Example(
                "Recompute an already finished version",
                "sbobinai run --config config.local.toml --file input/talk.mp4 --force",
            ),
        ),
        exit_status=_RUN_EXIT,
        notes=(
            "Options override the configuration file; the file overrides the "
            "defaults. Anything not part of job identity (paths, ui, monitor, retries) "
            "never creates a new version.",
            "With no command at all, or when the first word is an option, sbobinai runs `run`.",
            "A job that cannot transcribe some chunks ends `incomplete`: the partial "
            "result lists the gaps, the file stays in the input folder and exit "
            "status is 1.",
        ),
        files=(
            "input/ (queue), output/<source>/<version>/ (results), process/ (working "
            "files), processed/ (archive), process/runs/<run-id>/ (events, status, log).",
        ),
        related=("watch", "wizard", "doctor", "events", "catalog"),
        json_output=(
            "stdout is JSON Lines: one event per line (see EVENTS) ending with "
            '{"type": "result", "schema_version": 1, "ok": bool, '
            '"command": "run", "exit_status": N, "jobs": [{"job_id", '
            '"source", "status", "transcript"?, "error"?}]} and, when not ok, '
            'an "error" object with code job_failed or job_incomplete.'
        ),
    ),
    Command(
        name="watch",
        group="Transcribe",
        summary="Keep watching the input folder and transcribe new files.",
        description=(
            "Scans --input-dir every --watch-interval seconds and processes files "
            "whose size and modification time stayed unchanged for --stable-scans "
            "scans, so half-copied files are not read. Runs until Ctrl+C.",
            "Use it for a drop folder on a workstation. For one-off batches prefer `run`.",
        ),
        options=_opts((CONFIG_OPT,), _hidden(SHARED_OPTS, _HIDDEN_FOR_NON_RUN), (JSON_OPT,)),
        examples=(
            Example(
                "Watch the input folder until Ctrl+C", "sbobinai watch --config config.local.toml"
            ),
            Example(
                "Scan every 10 s, require 3 stable scans, plain output",
                "sbobinai watch --config config.local.toml --watch-interval 10 "
                "--stable-scans 3 --ui plain",
            ),
        ),
        exit_status=(
            (0, "Stopped with Ctrl+C (the normal way to end a watch)."),
            (1, "Backend unavailable or another sbobinai process holds the lock."),
            (
                2,
                "Invalid option or configuration file (--file and --limit are not "
                "supported by watch).",
            ),
        ),
        notes=("Job failures do not stop the watcher; they appear in the output and events.",),
        files=("Same as `run`.",),
        related=("run", "events"),
        json_output=(
            'Events stream as JSON Lines; after Ctrl+C a {"type": "result", '
            '"command": "watch", "ok": true, "exit_status": 0} line is written.'
        ),
    ),
    Command(
        name="wizard",
        group="Transcribe",
        summary="Guided interactive run: asks language, context and scope.",
        description=(
            "Shows the queue with durations and an estimate, asks the spoken "
            "language, optional context and scope (whole file or first N minutes), "
            "confirms, runs and prints a results table. Your last answers are "
            "remembered. The only command that asks questions.",
            "It needs an interactive terminal. For automation pass --answers with a "
            "JSON file; the questions are skipped.",
        ),
        options=_opts(
            (CONFIG_OPT,),
            _hidden(SHARED_OPTS, {"limit"}),
            (
                Opt(
                    ("--answers",),
                    "Inputs",
                    "JSON file with the answers (automation and tests); skips the "
                    "questions. Keys: language, context, max_minutes, confirm.",
                    type="path",
                    metavar="PATH",
                    default=None,
                ),
                JSON_OPT,
            ),
        ),
        examples=(
            Example("Interactive session", "sbobinai wizard --config config.local.toml"),
            Example(
                "Scripted answers (no terminal needed)",
                "sbobinai wizard --config config.local.toml --answers answers.json",
            ),
            Example(
                "Only one file",
                "sbobinai wizard --config config.local.toml --file input\\talk.mp4",
                "sbobinai wizard --config config.local.toml --file input/talk.mp4",
            ),
        ),
        exit_status=(
            (0, "All jobs completed or were skipped, or you cancelled before anything ran."),
            (1, "A job failed or ended incomplete, or another process holds the lock."),
            (2, "No terminal and no --answers, or an invalid answers file or answer."),
            (130, "Interrupted by Ctrl+C."),
        ),
        notes=(
            "Answers that change the output (language, context, scope) create a new "
            "version; earlier versions are kept.",
            "With --json the questions are not asked, --answers is required and the "
            "wizard's chatter goes to stderr.",
        ),
        files=(".local/wizard/last-answers.json (remembered answers), plus those of `run`.",),
        related=("run", "doctor"),
        json_output=(
            'Requires --answers. Events stream as JSON Lines, then {"type": '
            '"result", "command": "wizard", "ok": bool, "exit_status": N}.'
        ),
    ),
    Command(
        name="doctor",
        group="Inspect",
        summary="Check that FFmpeg, the speech detector and the backend are ready.",
        description=(
            "Resolves the configuration like `run` does and verifies the media "
            "tools (ffmpeg, ffprobe), the speech detector and the transcription "
            "backend (for llamacpp: the server is reachable and the model loaded). "
            "With a [server] table it also shows which inference backend (cuda, "
            "vulkan or cpu) would run and warns when that is a fallback. "
            "Nothing is transcribed and no media is read.",
            "Accepts the same options as `run` so you can check exactly the "
            "configuration a run would use.",
        ),
        options=_opts((CONFIG_OPT,), _hidden(SHARED_OPTS, _HIDDEN_FOR_NON_RUN), (JSON_OPT,)),
        examples=(
            Example("Check the local configuration", "sbobinai doctor --config config.local.toml"),
            Example(
                "Structured readiness for scripts and agents",
                "sbobinai doctor --config config.local.toml --json",
            ),
            Example(
                "Check a different backend without a config file",
                "sbobinai doctor --backend mock --vad none",
            ),
        ),
        exit_status=(
            (0, "Every check passed."),
            (1, "At least one check failed (details on stderr, or in the JSON checks)."),
            (2, "Invalid option or configuration file."),
        ),
        files=("None written.",),
        related=("setup", "init-config", "run"),
        json_output=(
            '{"ok": bool, "checks": [{"name", "ok", "detail"}], "warnings": [text]} '
            "where name is ffmpeg, ffprobe, speech_detector, inference_backend (only "
            'with a [server] table; carries the full backend "selection" object), '
            "or backend (plus backend_cleanup on a cleanup failure). warnings is "
            "non-empty when the inference backend is a fallback. When not ok an "
            '"error" object (checks_failed) is added.'
        ),
    ),
    Command(
        name="events",
        group="Inspect",
        summary="Print a run's events as JSONL (tail -f style with --follow).",
        description=(
            "Reads process/runs/<run-id>/events.jsonl and prints it line by line. "
            "With --follow it keeps polling until the run emits run.finished, so "
            "another terminal or an agent can watch a run in progress.",
            "Events carry numbers and labels only, never transcript or prompt text.",
        ),
        options=(
            CONFIG_OPT,
            Opt(
                ("--process-dir",),
                "Paths",
                "Folder holding runs/ (see run); default process.",
                type="path",
                metavar="DIR",
                config="process_dir",
            ),
            Opt(
                ("--run",),
                "Inputs",
                "Run id (for example 20261002T211415Z-58da) or latest.",
                default="latest",
                metavar="ID",
            ),
            Opt(
                ("--follow",),
                "Behaviour",
                "Keep polling until the run emits run.finished.",
                kind="flag",
            ),
            Opt(
                ("--type",),
                "Inputs",
                "Only events whose type starts with this prefix "
                "(progress, chunk, job, run.finished, ...).",
                metavar="PREFIX",
            ),
            Opt(
                ("--poll-interval",),
                "Behaviour",
                "Seconds between polls.",
                type="float",
                default=0.5,
                hidden=True,
            ),
            JSON_OPT,
        ),
        examples=(
            Example("Follow the latest run until it finishes", "sbobinai events --follow"),
            Example(
                "Only progress events of the latest run", "sbobinai events --follow --type progress"
            ),
            Example("Replay a finished run", "sbobinai events --run 20261002T211415Z-58da"),
        ),
        exit_status=(
            (0, "Events printed (also when stopped with Ctrl+C or a closed pipe)."),
            (1, "No such run, or the run has no events file."),
            (2, "Invalid option or configuration file."),
        ),
        files=("process/runs/<run-id>/events.jsonl, process/runs/latest.json.",),
        related=("run", "watch"),
        json_output=(
            "The output is already JSON Lines (one event per line); --json is "
            "accepted for uniformity and only makes errors structured."
        ),
    ),
    Command(
        name="server-profile",
        group="Inspect",
        summary="Print the resolved [server] profile as JSON (absolute paths).",
        description=(
            "Loads the configuration and prints the llama.cpp server profile "
            "(runtime, model and projector paths, alias, port, context size, slots, "
            "GPU layers, base URL) as one JSON object. It also chooses the inference "
            "backend: with backend = auto the runtimes are tried in the fallback "
            "order (cuda, vulkan, cpu) and the first usable one wins; the result, "
            "including why other backends were skipped, is in the selection object. "
            "The PowerShell launchers use it; agents can use it to learn how the "
            "local server is set up.",
        ),
        options=(
            Opt(
                ("--config",),
                "Paths",
                "Path to the TOML configuration file (required).",
                type="path",
                metavar="PATH",
                required=True,
            ),
            Opt(
                ("--optional",),
                "Behaviour",
                "Print null and exit 0 when the configuration has no [server] table.",
                kind="flag",
            ),
            JSON_OPT,
        ),
        examples=(
            Example(
                "Show the server profile", "sbobinai server-profile --config config.local.toml"
            ),
            Example(
                "Tolerate a configuration without [server]",
                "sbobinai server-profile --config config.local.toml --optional",
            ),
        ),
        exit_status=(
            (0, "Profile (or null with --optional) printed."),
            (1, "No usable inference backend (a fixed backend is unusable, or nothing works)."),
            (2, "Invalid configuration, or no [server] table without --optional."),
        ),
        related=("init-config", "doctor"),
        json_output=(
            "The output is always one JSON object (or null); --json only structures "
            "errors. Keys: runtime_dir (of the selected backend), backend, device, "
            "threads, model_path, projector_path, alias, state_dir, port, context_size, "
            "parallel, gpu_layers, base_url and selection {backend, device, "
            "device_name, runtime_dir, runtime_tag, requested, fallback_used, threads, "
            "tried: [{backend, ok, reason}], warning, warning_lines}."
        ),
    ),
    Command(
        name="init-config",
        group="Setup",
        summary="Write a ready-to-use local configuration for a profile.",
        description=(
            "Renders config.example.toml for a profile of resources.json with all "
            "paths pointing into your resource store, validates the result and "
            "writes it (default config.local.toml, which is git-ignored). An "
            "existing file is never overwritten without --force.",
        ),
        options=_opts(
            RESOURCE_OPTS,
            (
                Opt(
                    ("--store",),
                    "Paths",
                    "Resource store folder for models, VAD and runtimes (required).",
                    type="path",
                    metavar="DIR",
                    required=True,
                ),
                Opt(
                    ("--output",),
                    "Paths",
                    "File to write.",
                    type="path",
                    default=Path("config.local.toml"),
                    metavar="PATH",
                ),
                Opt(("--template",), "Paths", "Template file.", type="path", hidden=True),
                Opt(("--force",), "Behaviour", "Overwrite an existing --output file.", kind="flag"),
                JSON_OPT,
            ),
        ),
        examples=(
            Example(
                "Create config.local.toml for the default profile",
                "sbobinai init-config --store D:\\ai-models --profile qwen3-asr-1.7b-q8",
                "sbobinai init-config --store ~/ai-models --profile qwen3-asr-1.7b-q8",
            ),
            Example(
                "Regenerate it, replacing the old one",
                "sbobinai init-config --store D:\\ai-models --force",
                "sbobinai init-config --store ~/ai-models --force",
            ),
        ),
        exit_status=(
            (0, "File written."),
            (1, "The output file exists (use --force)."),
            (2, "Unknown profile, unreadable manifest or template, or invalid result."),
        ),
        files=("Writes --output (default ./config.local.toml).",),
        related=("setup", "doctor", "server-profile"),
        json_output=('{"ok": true, "command": "init-config", "output", "profile", "store"}.'),
    ),
    Command(
        name="setup",
        group="Setup",
        summary="Download the pinned resources of a profile into the resource store.",
        description=(
            "Prints the plan (files, sizes, status) of a resources.json profile, "
            "asks for confirmation, downloads missing or partial files with resume, "
            "verifies size and SHA-256 and records provenance. Existing files that "
            "do not match are never overwritten. The Vulkan and CPU llama.cpp runtimes "
            "are downloaded, verified and extracted by setup itself (runtime-manifest.json "
            "and PROVENANCE.txt are written; a verified installation is skipped). The "
            "CUDA runtime is large and has its own PowerShell installer; setup prints "
            "the exact command. --backends chooses which runtimes are installed.",
        ),
        options=_opts(
            (
                Opt(
                    ("--config",),
                    "Paths",
                    "Local configuration providing [resources].store.",
                    type="path",
                    metavar="PATH",
                ),
            ),
            RESOURCE_OPTS,
            (
                Opt(
                    ("--store",),
                    "Paths",
                    "Resource store folder (overrides [resources].store).",
                    type="path",
                    metavar="DIR",
                ),
                Opt(
                    ("--backends",),
                    "Server/runtime",
                    "Comma-separated runtimes to install: cuda, vulkan, cpu.",
                    metavar="LIST",
                    default_text=(
                        "automatic: vulkan and cpu, plus cuda when nvidia-smi finds a GPU "
                        "(a fixed [server].backend installs only that one)"
                    ),
                ),
                Opt(("--yes",), "Behaviour", "Do not ask for confirmation.", kind="flag"),
                Opt(
                    ("--dry-run",),
                    "Behaviour",
                    "Only print the plan; download nothing.",
                    kind="flag",
                ),
                JSON_OPT,
            ),
        ),
        examples=(
            Example("Show the plan only", "sbobinai setup --config config.local.toml --dry-run"),
            Example(
                "Install only the fallback runtimes",
                "sbobinai setup --config config.local.toml --backends vulkan,cpu --yes",
            ),
            Example("Download after confirming", "sbobinai setup --config config.local.toml"),
            Example(
                "Unattended download with a JSON summary",
                "sbobinai setup --config config.local.toml --yes --json",
            ),
        ),
        exit_status=(
            (0, "Everything is installed (or the dry run found no conflict)."),
            (1, "Conflict, not enough disk space, declined confirmation or a failed download."),
            (2, "No store, unknown profile, invalid --backends or invalid configuration."),
        ),
        notes=("With --json there is no prompt: pass --yes (or --dry-run).",),
        files=(
            "<store>/... downloaded files and their PROVENANCE.txt records.",
            "<store>/runtimes/<runtime>/runtime-manifest.json and PROVENANCE.txt "
            "(Vulkan and CPU runtimes).",
        ),
        related=("init-config", "doctor"),
        json_output=(
            '{"ok": bool, "command": "setup", "profile", "store", "dry_run", '
            '"backends": [names], "plan": [lines], "downloaded": [paths], '
            '"installed_runtimes": [folders], "failed": [ids]}.'
        ),
    ),
    Command(
        name="catalog",
        group="Maintenance",
        summary="Rebuild catalog.json and every source.json from process/ and output/.",
        description=(
            "Regenerates output/catalog.json and each output/<source>/source.json "
            "from the job metadata, newest first. It is normally kept up to date "
            "automatically; run it after moving folders by hand or after an "
            "interrupted run. Transcripts are not touched.",
        ),
        options=_opts((CONFIG_OPT,), DIR_OPTS, (JSON_OPT,)),
        examples=(
            Example("Rebuild the overview", "sbobinai catalog --config config.local.toml"),
            Example(
                "Machine-readable summary", "sbobinai catalog --config config.local.toml --json"
            ),
        ),
        exit_status=(
            (0, "Catalog rebuilt."),
            (1, "Another process holds the lock or a file could not be written."),
            (2, "Invalid configuration."),
        ),
        files=("Writes output/catalog.json and output/<source>/source.json.",),
        related=("migrate-layout", "run"),
        json_output='{"ok": true, "command": "catalog", "sources": N, "versions": N}.',
    ),
    Command(
        name="migrate-layout",
        group="Maintenance",
        summary="Move legacy flat job folders into the per-source layout (dry run).",
        description=(
            "Plans the move of old flat job folders into process/<source>/<version> "
            "and output/<source>/<version>. Without --apply only the plan is "
            "printed. Conflicting folders are refused, never overwritten; a move "
            "log is written and the catalog rebuilt.",
        ),
        options=_opts(
            (CONFIG_OPT,),
            DIR_OPTS,
            (
                Opt(
                    ("--apply",),
                    "Behaviour",
                    "Perform the moves; without it only the plan is printed.",
                    kind="flag",
                ),
                JSON_OPT,
            ),
        ),
        examples=(
            Example("Preview the migration", "sbobinai migrate-layout --config config.local.toml"),
            Example("Apply it", "sbobinai migrate-layout --config config.local.toml --apply"),
        ),
        exit_status=(
            (0, "Plan printed, or every move succeeded."),
            (1, "At least one move failed, or the lock is held."),
            (2, "Invalid configuration."),
        ),
        files=(
            "Moves folders under process/ and output/, writes a move log, rebuilds the catalog.",
        ),
        related=("catalog",),
        json_output=(
            '{"ok": bool, "command": "migrate-layout", "dry_run", "plan": [lines], '
            '"to_migrate", "refused", "untouched", "migrated"?, "failed"?, "log"?}.'
        ),
    ),
    Command(
        name="help",
        group="Help and shell",
        summary="Show the help of a command (same as `<command> --help`).",
        description=(
            "Without an argument it prints the overview; with a command name it "
            "prints exactly what `sbobinai <command> --help` prints.",
        ),
        options=(
            Opt(
                ("command",),
                "Arguments",
                "Command to describe.",
                positional=True,
                nargs="?",
                metavar="COMMAND",
            ),
            JSON_OPT,
        ),
        examples=(
            Example("Overview", "sbobinai help"),
            Example("Help of one command", "sbobinai help run"),
        ),
        exit_status=((0, "Help printed."), (2, "Unknown command.")),
        related=("man",),
        json_output="Prints the machine-readable description of the command (see man --format json).",
    ),
    Command(
        name="man",
        group="Help and shell",
        summary="Print the full manual (text, markdown, json or skill).",
        description=(
            "Prints the manual generated from the same source as --help: NAME, "
            "SYNOPSIS, DESCRIPTION, COMMANDS, OPTIONS, CONFIGURATION, FILES, "
            "ENVIRONMENT, EXIT STATUS, ERRORS, EXAMPLES, AGENT USAGE and SEE ALSO. "
            "--format json is the machine-readable interface description; "
            "--format skill prints the SKILL.md agent manifest.",
        ),
        options=(
            Opt(
                ("command",),
                "Arguments",
                "Limit the manual to one command.",
                positional=True,
                nargs="?",
                metavar="COMMAND",
            ),
            Opt(
                ("--format",),
                "Output and UI",
                "Output format.",
                default="text",
                choices=("text", "markdown", "json", "skill"),
            ),
            JSON_OPT,
        ),
        examples=(
            Example("Read the manual", "sbobinai man"),
            Example(
                "Regenerate the committed reference",
                "sbobinai man --format markdown > docs/cli-reference.md",
            ),
            Example("Interface description for tools", "sbobinai man --format json"),
            Example("Manual of one command", "sbobinai man run"),
        ),
        exit_status=((0, "Manual printed."), (2, "Unknown command or format.")),
        notes=("--json is a shortcut for --format json.",),
        related=("help",),
        json_output="Same as --format json.",
    ),
    Command(
        name="repl",
        group="Help and shell",
        summary="Interactive shell that runs the commands above with a remembered config.",
        description=(
            "Starts a small shell with a banner, a prompt and (where the platform "
            "provides readline) command history. Type any subcommand with its "
            "arguments, for example `doctor` or `run --limit 1`. The --config given "
            "to repl (or set with `config PATH`) is added to every command that "
            "accepts it. Built-ins: help [command], config [PATH], exit, quit.",
            "Starting sbobinai with no arguments still runs `run`; the shell only "
            "starts with this command.",
        ),
        options=(
            CONFIG_OPT,
            Opt(
                ("--json",),
                "Output and UI",
                "Run every command inside the shell with --json.",
                kind="flag",
            ),
        ),
        examples=(
            Example("Start the shell", "sbobinai repl --config config.local.toml"),
            Example("Scripted session", "echo doctor | sbobinai repl --config config.local.toml"),
        ),
        exit_status=((0, "Shell ended (exit, quit or end of input)."),),
        related=("help", "run", "doctor"),
        json_output="Every command typed inside the shell runs with --json.",
    ),
)

COMMAND_BY_NAME = {c.name: c for c in COMMANDS}
COMMAND_GROUPS = ("Transcribe", "Inspect", "Setup", "Maintenance", "Help and shell")


# --------------------------------------------------------------------------------------
# Configuration key table
# --------------------------------------------------------------------------------------

YES = "yes"
NO = "no"


@dataclass(frozen=True)
class ConfigKey:
    key: str
    type: str
    description: str
    identity: str
    section: str = "audio_transcript"


_CONFIG_ROWS = (
    ("input_dir", "path", "Queue folder with media waiting to be transcribed.", NO),
    ("processed_dir", "path", "Archive folder for fully transcribed queued sources.", NO),
    ("process_dir", "path", "Working files, checkpoints and run logs.", NO),
    ("output_dir", "path", "Transcripts, reports, source.json and catalog.json.", NO),
    ("backend", "string", "Inference backend: llamacpp, nexa or mock.", YES),
    ("model", "string", "Model name or llama-server alias (default depends on backend).", YES),
    (
        "base_url",
        "string",
        "llama.cpp server URL (http or https, no credentials).",
        "yes (llamacpp)",
    ),
    ("timeout", "number", "llama.cpp request timeout in seconds (positive).", "yes (llamacpp)"),
    (
        "max_tokens",
        "integer",
        "Generated tokens per chunk, upper bound (positive).",
        "yes (llamacpp)",
    ),
    ("prompt", "string or null", "Context/instructions for the model.", "yes (llamacpp)"),
    ("temperature", "number", "Sampling temperature, 0 to 2.", "yes (llamacpp)"),
    ("seed", "integer", "Sampling seed, 0 to 4294967295.", "yes (llamacpp)"),
    ("response_mode", "string", "json, plain or qwen3-asr.", "yes (llamacpp)"),
    ("model_path", "path or null", "Local model file (Nexa backend).", YES),
    ("projector_path", "path or null", "Local audio projector file (Nexa backend).", YES),
    ("device", "string", "Device for the Nexa backend.", YES),
    ("ffmpeg", "string", "FFmpeg executable name or path.", NO),
    ("ffprobe", "string", "FFprobe executable name or path.", NO),
    ("language", "string or null", "Spoken language code; auto or null means no hint.", YES),
    ("chunk_seconds", "number", "Maximum chunk length in seconds (positive).", YES),
    ("sample_rate", "integer", "Extracted audio sample rate in Hz (positive).", YES),
    ("max_duration", "number or null", "Transcribe only the first N seconds.", YES),
    ("max_file_size", "integer or null", "Reject larger input files (bytes).", NO),
    ("retries", "integer", "Extra attempts after transient failures (0 or more).", NO),
    ("force", "boolean", "Recompute already complete versions.", NO),
    ("prepare_only", "boolean", "Prepare chunks only; no model contact.", NO),
    ("archive_inputs", "boolean", "Archive fully transcribed queued sources.", NO),
    ("watch_interval", "number", "Seconds between input scans in watch.", NO),
    ("stable_scans", "integer", "Unchanged scans before a file is processed (2 or more).", NO),
    ("vad", "string", "Speech detector: silero, energy or none.", YES),
    ("vad_model_path", "path", "Silero ONNX model file.", "yes (silero only)"),
    ("vad_threshold", "number", "Speech probability threshold, 0 < x < 1.", YES),
    ("vad_min_speech_seconds", "number", "Shortest kept speech segment.", YES),
    ("vad_min_silence_seconds", "number", "Silence that ends a segment.", YES),
    ("vad_speech_pad_seconds", "number", "Padding around speech segments.", YES),
    ("vad_max_merge_gap_seconds", "number", "Merge segments closer than this.", YES),
    ("vad_energy_margin_db", "number", "Energy detector margin above noise (dB).", YES),
    ("fallback_temperatures", "list of numbers", "Retry temperatures (each 0 to 2).", YES),
    ("split_on_failure", "boolean", "Split a failing chunk once at its quietest pause.", YES),
    (
        "context_free_fallback",
        "boolean",
        "Retry without prompt when the model echoed it.",
        "yes (only when false)",
    ),
    ("min_tokens", "integer", "Per-chunk token cap floor (positive).", "yes (llamacpp)"),
    ("tokens_per_second", "number", "Token allowance per audio second.", "yes (llamacpp)"),
    ("compression_ratio_threshold", "number", "Loop detection ratio (positive).", "yes (llamacpp)"),
    ("repeat_penalty", "number", "Repetition penalty (positive, 1.0 disables).", "yes (llamacpp)"),
    ("dry_multiplier", "number", "DRY sampler multiplier (0 disables).", "yes (llamacpp)"),
    ("force_language", "boolean", "Qwen3-ASR: constrain output to language.", "yes (llamacpp)"),
    (
        "collect_logprobs",
        "boolean",
        "Record token log-probabilities for confidence.",
        "yes (llamacpp, only when true)",
    ),
    (
        "parallel_requests",
        "integer",
        "Concurrent chunk requests, 1 to 8.",
        "yes (llamacpp, only above 1)",
    ),
    (
        "intermediate_interval_seconds",
        "number",
        "Seconds between intermediate export rewrites.",
        NO,
    ),
    ("ui", "string", "Console output: auto, live, plain or jsonl.", NO),
    ("monitor", "boolean", "Sample CPU, RAM and GPU use.", NO),
    ("monitor_interval", "number", "Seconds between samples (0.1 to 3600).", NO),
)

CONFIG_KEYS = tuple(ConfigKey(*row) for row in _CONFIG_ROWS) + (
    ConfigKey(
        "store",
        "path",
        "Root folder for downloaded resources (required in [resources]).",
        NO,
        "resources",
    ),
    ConfigKey(
        "runtime_dir",
        "path",
        "CUDA llama.cpp runtime folder (relative: against the store); optional when "
        "[server.runtimes] names it.",
        NO,
        "server",
    ),
    ConfigKey(
        "backend",
        "string",
        'Inference backend: "auto" (try fallback in order), "cuda", "vulkan" or "cpu" '
        "(a fixed backend never falls back). Default auto.",
        NO,
        "server",
    ),
    ConfigKey(
        "fallback",
        "list",
        'Order tried by backend = "auto". Default ["cuda", "vulkan", "cpu"].',
        NO,
        "server",
    ),
    ConfigKey(
        "runtimes",
        "table",
        "[server.runtimes]: backend -> runtime folder (cuda, vulkan, cpu; relative: against "
        "the store). Missing backends default to the folders setup installs into.",
        NO,
        "server",
    ),
    ConfigKey(
        "device",
        "string",
        'Optional llama.cpp device, for example "Vulkan1" or "CUDA1" (see '
        "llama-server --list-devices).",
        NO,
        "server",
    ),
    ConfigKey(
        "threads",
        "integer",
        "CPU backend threads (1-1024). Default: physical cores.",
        NO,
        "server",
    ),
    ConfigKey("model_path", "path", "Model GGUF file.", NO, "server"),
    ConfigKey("projector_path", "path", "Audio projector GGUF file.", NO, "server"),
    ConfigKey("alias", "string", "Server model alias; must equal model.", NO, "server"),
    ConfigKey("state_dir", "path", "Folder for server pid, logs and state.", NO, "server"),
    ConfigKey("port", "integer", "Server port (1-65535); must match base_url.", NO, "server"),
    ConfigKey(
        "context_size",
        "integer",
        "Server context tokens (1-1048576), shared by slots.",
        NO,
        "server",
    ),
    ConfigKey(
        "parallel", "integer", "Server slots (1-8); at least parallel_requests.", NO, "server"
    ),
    ConfigKey(
        "gpu_layers", "integer", "Layers offloaded to the GPU (1-999, optional).", NO, "server"
    ),
)


def config_flag(key: str) -> str | None:
    """CLI flag for a [audio_transcript] key, or None when it is file-only."""
    for opt in SHARED_OPTS:
        if opt.config == key and not opt.hidden:
            if opt.kind == "bool":
                return f"{opt.long_flag}/--no-{opt.long_flag[2:]}"
            return opt.long_flag
    return None


def config_default(key: str) -> str:
    for opt in SHARED_OPTS:
        if opt.config == key and not opt.hidden and opt.default_text is not None:
            return opt.default_text
    return format_default(getattr(AppConfig, key, None))


# --------------------------------------------------------------------------------------
# Environment, files, events, agent guide
# --------------------------------------------------------------------------------------

ENVIRONMENT = (
    ("NO_COLOR", "When set (any value), the live console uses no colors."),
    ("CI", "When set, --ui auto never picks the live display (plain status lines)."),
    ("TERM", "TERM=dumb disables the live display under --ui auto."),
    ("PATH", "Used to find ffmpeg and ffprobe when --ffmpeg/--ffprobe are plain names."),
    ("SILERO_VAD_MODEL", "Tests only: Silero ONNX file for the optional real-model tests."),
)

FILES = (
    ("input/", "Operational queue; failed, incomplete, bounded and prepare-only jobs stay."),
    ("processed/<source>/<file>", "Original file after a complete transcription."),
    (
        "process/<source>/<version>/",
        "Working files: chunks, checkpoint, metadata.json, raw "
        "responses, events.jsonl of that job.",
    ),
    ("process/runs/<run-id>/", "events.jsonl, status.json and pipeline.log of one run."),
    ("process/runs/latest.json", "Pointer to the latest run id and its state."),
    ("process/.audio-transcript.lock", "Process lock: one run at a time."),
    ("output/catalog.json", "Every input file, latest activity first."),
    ("output/<source>/source.json", "All versions of one input, newest first."),
    ("output/<source>/<version>/transcript.{txt,md,srt,vtt,json}", "The transcript."),
    ("output/<source>/<version>/report.json", "Settings, timings, counts, confidence."),
    ("output/<source>/<version>/run-report.md", "Human-readable report."),
    ("output/<source>/<version>/intermediate/", "Partial results while a job runs."),
    ("config.local.toml", "Your local configuration (git-ignored); see config.example.toml."),
    (".local/wizard/last-answers.json", "Answers remembered by the wizard."),
)

EVENT_TYPES = (
    (
        "run.started",
        "command, model, backend, response_mode, language, parallel_requests, queue_size, "
        "ui, runtime? (inference backend: backend, device, device_name, fallback_used, "
        "requested, threads, runtime_tag, skipped)",
    ),
    ("job.started", "source, version, scope, index, total"),
    ("stage.started", "stage"),
    ("stage.finished", "stage, seconds, reused?"),
    ("job.prepared", "analysed_seconds, speech_seconds, chunk_count, detector, reused"),
    ("chunk.started", "index, part, audio_seconds, in_flight"),
    ("chunk.attempt", "index, stage, temperature, outcome, wall_seconds, completion_tokens"),
    ("chunk.finished", "index, outcome, attempts, wall_seconds, running counters"),
    ("progress", "done_chunks, total_chunks, percent, eta_s, throughput_x, ok, failed"),
    ("resource.sample", "cpu_percent, ram_used_mb, gpu_util_percent, gpu_mem_used_mb, ..."),
    ("log", "level, message"),
    ("warning", "code, message"),
    ("job.finished", "source, status, counts, timings, confidence, artifacts, archived"),
    ("run.finished", "exit_status, jobs, wall_seconds"),
)

TOP_DESCRIPTION = (
    "SbobinAI transcribes long local audio and video recordings on your own "
    "NVIDIA GPU: speech detection cuts the audio at pauses, a local model "
    "transcribes each chunk, and you get TXT, Markdown, SRT, VTT and JSON plus a "
    "run report. Nothing leaves your machine.",
)

WORKFLOW = (
    "setup         download the pinned models into a resource store",
    "init-config   write config.local.toml pointing at that store",
    "doctor        verify FFmpeg, speech detector and backend",
    "run | wizard  transcribe the files in input/",
    "read output   output/<source>/<version>/transcript.txt, catalog.json",
)

GLOBAL_NOTES = (
    "Configuration: pass --config PATH (for example config.local.toml). There is no "
    "automatic discovery; without --config the built-in defaults are used. Command "
    "line options override the file; the file overrides the defaults.",
    "Non-interactive by default: only `wizard` (and `setup` without --yes) ever "
    "prompt. Use --json for machine-readable output.",
    "Running `sbobinai` with no command, or starting with an option, runs `run`.",
)

AGENT_PRINCIPLES = (
    "Non-interactive by default. Only `wizard` asks questions (needs a terminal or "
    "--answers) and `setup` confirms unless --yes; never call them without those.",
    "Use --json on every command. One-shot commands print one JSON object; "
    "run/watch/wizard stream JSON Lines events and end with a result object. "
    "`--ui jsonl` streams events without the result line; `--ui plain` gives "
    "stable text lines.",
    "Branch on the exit status (0, 1, 2, 130; see EXIT STATUS) and, in JSON mode, "
    'on error.code. Errors are {"ok": false, "error": {"code", "message", '
    '"exit_status"}} on stdout; stderr stays empty.',
    "Read results from files, not from console text: output/catalog.json (all "
    "sources), output/<source>/source.json (versions, newest first) and "
    "output/<source>/<version>/transcript.* and report.json.",
    "Idempotent: re-running skips versions that are already complete and resumes "
    "incomplete ones; --force recomputes. Different settings create a new version, "
    "never overwrite an old one.",
    "Bound the work with --file, --limit and --max-duration (a bounded run is a "
    "separate version and keeps the input file).",
    "Follow a run with `events --follow` (JSON Lines, ends at run.finished) or poll "
    "process/runs/latest.json and process/runs/<run-id>/status.json.",
    "One run at a time: a second process fails with exit status 1 (lock_held). "
    "Check readiness first with `doctor --json`; do not start or stop the model "
    "server yourself unless the user asks.",
)


@dataclass(frozen=True)
class Recipe:
    title: str
    posix: str
    powershell: str | None = None


RECIPES = (
    Recipe(
        "Check readiness as JSON",
        "sbobinai doctor --config config.local.toml --json   # exit 0 and ok=true when ready",
    ),
    Recipe(
        "Transcribe one file and print the transcript path",
        "sbobinai run --config config.local.toml --json --file input/talk.mp4 \\\n"
        "  | tail -n 1 | jq -r '.jobs[0].transcript'",
        "sbobinai run --config config.local.toml --json --file input\\talk.mp4 |\n"
        "  Select-Object -Last 1 | ConvertFrom-Json | ForEach-Object { $_.jobs[0].transcript }",
    ),
    Recipe(
        "Branch on the outcome of a run",
        "sbobinai run --config config.local.toml --json > run.jsonl; case $? in 0) echo done;;\n"
        "  1) tail -n 1 run.jsonl | jq '.error';; 2) echo fix the command line;; esac",
    ),
    Recipe(
        "List every transcribed source, latest activity first",
        "jq -r '.sources[] | [.original_name, .latest_version.status] | @tsv' output/catalog.json",
        "(Get-Content output\\catalog.json -Raw | ConvertFrom-Json).sources |\n"
        '  ForEach-Object { "$($_.original_name) $($_.latest_version.status)" }',
    ),
    Recipe(
        "List the versions of one source, newest first",
        "jq -r '.versions[] | [.version, .status, .completion_percent] | @tsv' \\\n"
        "  output/<source>/source.json",
        "(Get-Content output\\<source>\\source.json -Raw | ConvertFrom-Json).versions |\n"
        "  Select-Object version, status, completion_percent",
    ),
    Recipe(
        "Follow progress of a running job until it finishes",
        "sbobinai events --follow --type progress     # ends after run.finished",
    ),
    Recipe(
        "Quick bounded check: first 10 minutes, keep the file in input",
        "sbobinai run --config config.local.toml --json --limit 1 --max-duration 600 "
        "--no-archive-inputs",
    ),
    Recipe(
        "Recompute a version that is already complete",
        "sbobinai run --config config.local.toml --json --file input/talk.mp4 --force",
    ),
    Recipe(
        "Preview a download without changing anything",
        "sbobinai setup --config config.local.toml --dry-run --json",
    ),
    Recipe(
        "Learn how the local server is configured",
        "sbobinai server-profile --config config.local.toml --optional",
    ),
)

SEE_ALSO = (
    "README.md (installation and daily use)",
    "docs/cli-reference.md (this manual, generated)",
    "docs/events.md (event schema, status files, resource monitor)",
    "docs/inference-controls.md (segmentation, recovery ladder, response modes)",
    "config.example.toml (every configuration key with comments)",
)


def visible_options(command: Command) -> tuple[Opt, ...]:
    return tuple(o for o in command.options if not o.hidden)


def grouped_options(command: Command) -> list[tuple[str, list[Opt]]]:
    """Visible options by group in the canonical group order (positionals first)."""
    result: list[tuple[str, list[Opt]]] = []
    seen: dict[str, list[Opt]] = {}
    for opt in visible_options(command):
        seen.setdefault(opt.group, []).append(opt)
    for group in GROUP_ORDER:
        if group in seen:
            result.append((group, seen.pop(group)))
    result.extend(seen.items())
    return result

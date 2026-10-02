"""Interactive, guided run: ask the run parameters, confirm, process, summarise.

The module holds pure prompting logic. ``input``, ``print``, the clock, the pipeline factory,
the process lock and the folder opener are injected so the whole flow is testable without a
console. Answers that change the output (language, context, scope) become configuration
overrides and therefore part of the job identity; the remembered answers in
``.local/wizard/last-answers.json`` are a convenience only and never part of identity.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from ..config import AppConfig
from .catalog import SOURCE_NAME, version_summary
from .pipeline import TranscriptionPipeline
from .reporting import format_duration, format_timing
from .state import read_json

MAX_CONTEXT_CHARS = 2000
DEFAULT_REAL_TIME_FACTOR = 0.04
PREVIEW_CHARS = 60
LAST_ANSWERS_PATH = Path(".local") / "wizard" / "last-answers.json"
_GENERIC_LANGUAGE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})?$")
_GOOD_STATUSES = {"completed", "skipped"}


class _Cancelled(Exception):
    """The user ended the session (EOF) before confirming."""


@dataclass(frozen=True)
class Answers:
    """Run parameters chosen for one file."""

    language: str | None  # None = auto-detect
    context_input: str  # exactly what was typed ("" = none, "@file" = file reference)
    context: str | None  # resolved text
    max_minutes: float | None  # None = full file


def language_label(language: str | None) -> str:
    return language if language else "auto-detect"


def context_preview(text: str | None) -> str:
    if not text:
        return "(none)"
    flat = " ".join(text.split())
    shown = flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 3] + "..."
    return f'"{shown}"'


def scope_text(max_minutes: float | None) -> str:
    return "full file" if max_minutes is None else f"first {max_minutes:g} min only"


def validate_language(text: str, response_mode: str) -> tuple[str | None, str | None]:
    """Return ``(language, error)``; ``language`` None means auto-detect."""
    token = text.strip().casefold()
    if token in {"", "auto"}:
        return None, None
    if response_mode == "qwen3-asr":
        module = importlib.import_module("..adapters.llamacpp", __package__)
        if module.qwen3_asr_language(token) is None:
            codes = ", ".join(sorted(module.QWEN3_ASR_LANGUAGES))
            return None, (
                f"'{text.strip()}' is not a language the Qwen3-ASR model supports. "
                f"Supported codes: {codes}. Type auto to let the model detect it."
            )
        return token, None
    if not _GENERIC_LANGUAGE.match(token):
        return None, f"'{text.strip()}' does not look like a language code (for example it, en)."
    return token, None


def resolve_context(raw: str) -> tuple[str | None, str | None]:
    """Return ``(text, error)`` for a typed line or ``@path`` UTF-8 file reference."""
    raw = raw.strip()
    if not raw:
        return None, None
    if raw.startswith("@"):
        path = Path(raw[1:].strip().strip('"')).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return None, f"Could not read the UTF-8 context file {path}: {exc}"
        text = text.strip()
    else:
        text = raw
    if not text:
        return None, None
    if len(text) > MAX_CONTEXT_CHARS:
        return None, (
            f"The context has {len(text)} characters; the limit is {MAX_CONTEXT_CHARS}. "
            "Shorten it (names, topic and key terms are enough)."
        )
    return text, None


def parse_minutes(value: object) -> tuple[float | None, str | None]:
    """Return ``(minutes, error)``; ``None`` minutes means the full file."""
    if value is None:
        return None, None
    if isinstance(value, str):
        token = value.strip().casefold()
        if token in {"", "full", "f", "all"}:
            return None, None
        try:
            number = float(token.replace(",", "."))
        except ValueError:
            return (
                None,
                f"'{value.strip()}' is not a number of minutes; type full for the whole file.",
            )
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "The scope must be a number of minutes or full."
    else:
        number = float(value)
    if not number > 0 or number == float("inf"):
        return None, "The number of minutes must be greater than zero."
    return number, None


def load_last_answers(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_last_answers(path: Path, answers: Answers) -> None:
    payload = {
        "language": answers.language or "auto",
        "context": answers.context_input,
        "max_minutes": answers.max_minutes,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # a convenience only


def estimate_rtf(config: AppConfig) -> tuple[float, bool]:
    """Real-time factor of the most recent completed version, and whether it is measured."""
    catalog = read_json(config.output_dir / "catalog.json", {})
    sources = catalog.get("sources") if isinstance(catalog, dict) else None
    for entry in (sources if isinstance(sources, list) else [])[:25]:
        folder = entry.get("folder") if isinstance(entry, dict) else None
        if not isinstance(folder, str):
            continue
        source = read_json(config.output_dir / folder / SOURCE_NAME, {})
        versions = source.get("versions") if isinstance(source, dict) else None
        for version in versions if isinstance(versions, list) else []:
            rtf = version.get("real_time_factor") if isinstance(version, dict) else None
            if (
                version.get("status") == "completed"
                and isinstance(rtf, (int, float))
                and not isinstance(rtf, bool)
                and rtf > 0
            ):
                return float(rtf), True
    return DEFAULT_REAL_TIME_FACTOR, False


class Wizard:
    def __init__(
        self,
        config: AppConfig,
        make_pipeline: Callable[[AppConfig], TranscriptionPipeline],
        *,
        ask: Callable[[str], str] = input,
        say: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
        lock_factory: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext,
        last_answers_path: Path | None = LAST_ANSWERS_PATH,
        open_folder: Callable[[Path], None] | None = None,
        scripted: dict | None = None,
        sources: Iterable[Path] | None = None,
    ) -> None:
        self.config = config
        self.make_pipeline = make_pipeline
        self._ask = ask
        self.say = say
        self.clock = clock
        self.lock_factory = lock_factory
        self.last_answers_path = last_answers_path
        self.open_folder = open_folder
        self.scripted = scripted
        self.sources = None if sources is None else list(sources)
        self.uses_context = config.backend == "llamacpp"

    # ------------------------------------------------------------------ prompting
    def ask(self, prompt: str) -> str:
        try:
            return self._ask(prompt)
        except EOFError as exc:
            raise _Cancelled from exc

    def _queue(self) -> list[tuple[Path, float | None, int | None]]:
        pipeline = self.make_pipeline(self.config)
        try:
            sources = self.sources if self.sources is not None else pipeline.discover()
            items = []
            for source in sources:
                try:
                    size = source.stat().st_size
                except OSError:
                    size = None
                try:
                    duration = float(pipeline.processor.probe(source).duration)
                except Exception:
                    duration = None
                items.append((source, duration, size))
            return items
        finally:
            backend = pipeline.backend
            if backend is not None:
                with contextlib.suppress(Exception):
                    backend.close()

    def _show_queue(self, items) -> None:
        rtf, measured = estimate_rtf(self.config)
        self.say(f"Queue: {len(items)} file(s)")
        total = 0.0
        for number, (source, duration, size) in enumerate(items, 1):
            size_text = "size unknown" if size is None else f"{size / 1_000_000:.1f} MB"
            if duration is None:
                self.say(f"  {number}. {source.name}  ({size_text}, duration unknown)")
                continue
            total += duration
            self.say(
                f"  {number}. {source.name}  ({size_text}, {format_duration(duration)}, "
                f"estimated {format_duration(duration * rtf)})"
            )
        basis = (
            "from the most recent completed run"
            if measured
            else f"rough estimate with a default real-time factor of {rtf:g}; no completed run found"
        )
        self.say(
            f"Estimated inference time for the whole queue: {format_duration(total * rtf)} "
            f"({basis}). Preparation and the first model load are not included."
        )

    def _ask_language(self, default: str | None) -> str | None:
        while True:
            reply = self.ask(
                f"Spoken language (ISO code such as it or en, or auto) [{language_label(default)}]: "
            )
            if not reply.strip():
                return default
            language, error = validate_language(reply, self.config.response_mode)
            if error is None:
                return language
            self.say(error)

    def _ask_transcript_language(self) -> None:
        self.say(
            "The transcript is written in the spoken language. Translation is not supported by "
            "the current ASR model (Qwen3-ASR only transcribes); it may become a separate "
            "derived stage later."
        )
        while True:
            reply = self.ask("Transcript language [same as spoken]: ").strip().casefold()
            if reply in {"", "same", "same as spoken", "s"}:
                return
            self.say(
                f"Cannot produce a transcript in '{reply}': translation is not supported yet. "
                "Press Enter to keep the spoken language."
            )

    def _context_help(self) -> str:
        if self.config.response_mode == "qwen3-asr":
            return (
                "Optional context is sent to the model as system context (names, topic, "
                "special terms); it biases recognition and does not instruct the model."
            )
        return (
            "Optional context replaces the built-in instruction prompt (response mode "
            f"{self.config.response_mode}), so write it as a complete instruction."
        )

    def _ask_context(self, previous: str) -> tuple[str, str | None]:
        reuse = (
            f" Type = to reuse the previous one {context_preview(resolve_context(previous)[0])}."
            if previous.strip()
            else ""
        )
        while True:
            reply = self.ask(
                "Context / instructions (one line, or @path\\to\\file.txt for a UTF-8 file; "
                f"max {MAX_CONTEXT_CHARS} chars; Enter = none).{reuse}\n> "
            ).strip()
            if reply == "=" and previous.strip():
                reply = previous.strip()
            text, error = resolve_context(reply)
            if error is None:
                if text:
                    self.say(f"Context accepted: {len(text)} characters, {context_preview(text)}")
                return reply, text
            self.say(error)

    def _ask_scope(self, default: float | None) -> float | None:
        label = "full" if default is None else f"{default:g}"
        while True:
            reply = self.ask(
                f"Scope: full file, or only the first N minutes (a number) [{label}]: "
            )
            if not reply.strip():
                return default
            minutes, error = parse_minutes(reply)
            if error is None:
                return minutes
            self.say(error)

    def _ask_answers(self, defaults: Answers, previous_context: str) -> Answers:
        language = self._ask_language(defaults.language)
        self._ask_transcript_language()
        if self.uses_context:
            self.say(self._context_help())
            context_input, context = self._ask_context(previous_context)
        else:
            self.say(
                f"The {self.config.backend} backend does not use context or instructions; "
                "that question is skipped."
            )
            context_input, context = "", None
        return Answers(language, context_input, context, self._ask_scope(defaults.max_minutes))

    def _scripted_answers(self, mapping: dict, defaults: Answers) -> Answers:
        language = defaults.language
        if "language" in mapping:
            language, error = validate_language(str(mapping["language"]), self.config.response_mode)
            if error:
                raise ValueError(error)
        context_input, context = defaults.context_input, defaults.context
        if "context" in mapping:
            context_input = str(mapping["context"] or "")
            context, error = resolve_context(context_input)
            if error:
                raise ValueError(error)
        max_minutes = defaults.max_minutes
        if "max_minutes" in mapping:
            max_minutes, error = parse_minutes(mapping["max_minutes"])
            if error:
                raise ValueError(error)
        return Answers(language, context_input, context, max_minutes)

    def _collect(self, items) -> dict[Path, Answers] | None:
        last = load_last_answers(self.last_answers_path) if self.last_answers_path else {}
        language, _ = validate_language(
            str(last.get("language") or "auto"), self.config.response_mode
        )
        if "language" not in last:
            language = self.config.language
        minutes, _ = parse_minutes(last.get("max_minutes"))
        previous_context = last.get("context") if isinstance(last.get("context"), str) else ""
        defaults = Answers(language, "", None, minutes)
        if self.scripted is not None:
            base = self._scripted_answers(self.scripted, defaults)
            per_file = self.scripted.get("files") if isinstance(self.scripted, dict) else None
            per_file = per_file if isinstance(per_file, dict) else {}
            return {
                source: self._scripted_answers(per_file.get(source.name, {}), base)
                for source, _duration, _size in items
            }
        self.say("")
        self.say("Run parameters (press Enter to accept the value in brackets)")
        answers = self._ask_answers(defaults, previous_context)
        chosen = {items[0][0]: answers}
        if len(items) > 1:
            same = self.ask(f"Apply these answers to all {len(items)} files? [Y/n]: ")
            if same.strip().casefold() in {"", "y", "yes"}:
                chosen = {source: answers for source, _d, _s in items}
            else:
                for source, _d, _s in items[1:]:
                    self.say(f"Answers for {source.name}")
                    answers = self._ask_answers(answers, answers.context_input or previous_context)
                    chosen[source] = answers
        if self.last_answers_path is not None:
            save_last_answers(self.last_answers_path, answers)
        return chosen

    def _config_for(self, answers: Answers) -> AppConfig:
        return replace(
            self.config,
            language=answers.language,
            prompt=answers.context if answers.context else self.config.prompt,
            max_duration=None if answers.max_minutes is None else answers.max_minutes * 60,
        )

    def _confirm(self, plan: dict[Path, Answers]) -> bool:
        config = self.config
        self.say("")
        self.say("Summary")
        for source, answers in plan.items():
            length = len(answers.context) if answers.context else 0
            self.say(f"  {source.name}")
            self.say(f"    spoken language : {language_label(answers.language)}")
            self.say("    transcript      : same as spoken (translation is not supported yet)")
            self.say(f"    context         : {context_preview(answers.context)} ({length} chars)")
            self.say(f"    scope           : {scope_text(answers.max_minutes)}")
        self.say(
            f"  model {config.model} | backend {config.backend} | response mode "
            f"{config.response_mode} | parallel requests {config.parallel_requests}"
        )
        self.say(
            "A different language, context or scope creates a new version in "
            "output/<source>/<version>/; earlier versions are kept."
        )
        if any(a.max_minutes is not None for a in plan.values()):
            self.say("Bounded runs keep the file in the input folder.")
        if self.scripted is not None:
            return bool(self.scripted.get("confirm", True))
        reply = self.ask("Start processing? [Y/n]: ").strip().casefold()
        return reply in {"", "y", "yes"}

    # ------------------------------------------------------------------ running
    def _run_file(self, source: Path, answers: Answers) -> dict:
        config = self._config_for(answers)
        started = self.clock()
        try:
            pipeline = self.make_pipeline(config)
            results = pipeline.run([source])
            result = dict(results[0]) if results else {"status": "failed", "error": "no result"}
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        result["elapsed"] = self.clock() - started
        result["source_name"] = source.name
        result["_config"] = config
        return result

    def _result_lines(self, result: dict) -> list[str]:
        config: AppConfig = result["_config"]
        status = str(result.get("status"))
        lines = [f"  {result['source_name']}", f"    status        : {status}"]
        job_id = result.get("job_id")
        summary = None
        base = None
        if isinstance(job_id, str) and "/" in job_id:
            folder, version = job_id.split("/", 1)
            try:
                summary = version_summary(config, folder, version)
            except Exception:
                summary = None
            base = config.output_dir / folder / version
            lines.append(f"    version       : {job_id}")
        if summary is not None:
            chunks = summary["chunks"]
            lines.append(
                f"    chunks        : {_n(chunks['ok'])} ok, {_n(chunks['failed'])} failed, "
                f"{_n(chunks['no_speech'])} no speech (of {_n(chunks['total'])})"
            )
            confidence = summary.get("confidence")
            mean = confidence.get("mean") if isinstance(confidence, dict) else None
            lines.append(
                f"    confidence    : {_n(mean)} mean token probability (uncalibrated proxy)"
            )
        lines.append(f"    total time    : {format_timing(result['elapsed'])}")
        if base is not None:
            for label, name in (("transcript", "transcript.txt"), ("run report", "run-report.md")):
                for candidate in (base / name, base / "intermediate" / name):
                    if candidate.is_file():
                        lines.append(f"    {label:<13} : {candidate}")
                        break
        if result.get("error"):
            lines.append(f"    error         : {result['error']}")
        if result.get("archive_warning"):
            lines.append(f"    note          : {result['archive_warning']}")
        return lines

    def run(self) -> int:
        try:
            return self._run()
        except _Cancelled:
            self.say("Input ended; cancelled. Nothing was processed.")
            return 0
        except ValueError as exc:
            self.say(f"ERROR: {exc}")
            return 2

    def _run(self) -> int:
        items = self._queue()
        if not items:
            self.say(
                f"No media files found in {self.config.input_dir}. Copy audio or video files "
                "there and start the wizard again."
            )
            return 0
        self._show_queue(items)
        plan = self._collect(items)
        if plan is None or not self._confirm(plan):
            self.say("Cancelled. Nothing was processed.")
            return 0
        results = []
        try:
            with self.lock_factory():
                for number, (source, answers) in enumerate(plan.items(), 1):
                    self.say("")
                    self.say(f"[{number}/{len(plan)}] {source.name}")
                    results.append(self._run_file(source, answers))
        except RuntimeError as exc:  # process lock held by another run
            self.say(f"ERROR: {exc}")
            return 1
        self.say("")
        self.say("Results")
        for result in results:
            for line in self._result_lines(result):
                self.say(line)
        if self.open_folder is not None:
            reply = ""
            if self.scripted is None:
                with contextlib.suppress(_Cancelled):
                    reply = self.ask("Open the output folder? [y/N]: ")
            if reply.strip().casefold() in {"y", "yes"}:
                with contextlib.suppress(Exception):
                    self.open_folder(self.config.output_dir)
        return 0 if all(r.get("status") in _GOOD_STATUSES for r in results) else 1


def _n(value: object) -> str:
    return (
        "n/a" if value is None else f"{value:g}" if isinstance(value, (int, float)) else str(value)
    )

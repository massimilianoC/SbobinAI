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

from ..config import AFTER_SUCCESS_CHOICES, AppConfig
from ..languages import UI_LANGUAGES, language_code, own_name
from .catalog import SOURCE_NAME, version_summary
from .pipeline import TranscriptionPipeline
from .reporting import format_duration, format_timing
from .state import read_json
from .wizard_text import FULL_WORDS, YES_WORDS, Texts

MAX_CONTEXT_CHARS = 2000
DEFAULT_REAL_TIME_FACTOR = 0.04
PREVIEW_CHARS = 60
LAST_ANSWERS_PATH = Path(".local") / "wizard" / "last-answers.json"
_GENERIC_LANGUAGE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})?$")
_AUTO_WORDS = {
    "",
    "auto",
    "automatic",
    "automatico",
    "automatica",
    "automático",
    "automatique",
    "automatisch",
    "detect",
    "rileva",
}
# Answers to "what happens to the original file?" in every guide language.
_AFTER_SUCCESS_WORDS = {
    "keep-all": {"1", "keep", "keep-all", "tieni", "tutto", "conservar", "garder", "behalten",
                 "manter"},
    "keep-audio": {"2", "audio", "áudio", "keep-audio"},
    "delete-all": {"3", "delete", "delete-all", "elimina", "borrar", "supprimer", "löschen",
                   "loschen", "apagar"},
}  # fmt: skip
# Size of 16 kHz mono speech as FLAC, for the estimate shown before deleting
# (measured 2026-10-04: 2 h 6 min webinar -> 104 MB, about 13.8 kB/s).
FLAC_BYTES_PER_SECOND = 14_000
_GOOD_STATUSES = {"completed", "skipped"}
_ENGLISH = Texts("en")


class _Cancelled(Exception):
    """The user ended the session (EOF) before confirming."""


@dataclass(frozen=True)
class Answers:
    """Run parameters chosen for one file."""

    language: str | None  # None = auto-detect
    context_input: str  # exactly what was typed ("" = none, "@file" = file reference)
    context: str | None  # resolved text
    max_minutes: float | None  # None = full file
    after_success: str = "keep-all"  # never remembered between sessions


def language_label(language: str | None, t: Texts = _ENGLISH) -> str:
    if not language:
        return t("auto_detect")
    name = own_name(language)
    return f"{language} ({name})" if name else language


def after_success_label(policy: str, t: Texts = _ENGLISH) -> str:
    return t("retention_" + policy.replace("-", "_"))


def parse_after_success(value: object) -> str | None:
    """Policy for a typed number or word (``2``, ``audio``, ``elimina``), else None."""
    token = str(value).strip().casefold()
    for policy, words in _AFTER_SUCCESS_WORDS.items():
        if token in words:
            return policy
    return None


def context_preview(text: str | None, t: Texts = _ENGLISH) -> str:
    if not text:
        return t("none")
    flat = " ".join(text.split())
    shown = flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 3] + "..."
    return f'"{shown}"'


def scope_text(max_minutes: float | None, t: Texts = _ENGLISH) -> str:
    return t("scope_full") if max_minutes is None else t("scope_first", minutes=f"{max_minutes:g}")


def validate_language(
    text: str, response_mode: str, t: Texts = _ENGLISH
) -> tuple[str | None, str | None]:
    """Return ``(language, error)``; ``language`` None means auto-detect.

    Codes and names (``it``, ``Italian``, ``italiano``, ``it-IT``) become the ISO code, so
    the same language always gives the same job identity.
    """
    token = text.strip().casefold()
    if token in _AUTO_WORDS:
        return None, None
    code = language_code(token)
    if response_mode == "qwen3-asr":
        module = importlib.import_module("..adapters.llamacpp", __package__)
        if code not in module.QWEN3_ASR_LANGUAGES:
            codes = ", ".join(sorted(module.QWEN3_ASR_LANGUAGES))
            return None, t("language_unsupported", text=text.strip(), codes=codes)
        return code, None
    if code is not None:
        return code, None
    if not _GENERIC_LANGUAGE.match(token):
        return None, t("language_invalid", text=text.strip())
    return token, None


def resolve_context(raw: str, t: Texts = _ENGLISH) -> tuple[str | None, str | None]:
    """Return ``(text, error)`` for a typed line or ``@path`` UTF-8 file reference."""
    raw = raw.strip()
    if not raw:
        return None, None
    if raw.startswith("@"):
        path = Path(raw[1:].strip().strip('"')).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return None, t("context_unreadable", path=path, error=exc)
        text = text.strip()
    else:
        text = raw
    if not text:
        return None, None
    if len(text) > MAX_CONTEXT_CHARS:
        return None, t("context_too_long", length=len(text), limit=MAX_CONTEXT_CHARS)
    return text, None


def parse_minutes(value: object, t: Texts = _ENGLISH) -> tuple[float | None, str | None]:
    """Return ``(minutes, error)``; ``None`` minutes means the full file."""
    if value is None:
        return None, None
    if isinstance(value, str):
        token = value.strip().casefold()
        if token == "" or token in FULL_WORDS:
            return None, None
        try:
            number = float(token.replace(",", "."))
        except ValueError:
            return None, t("minutes_not_number", text=value.strip())
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, t("minutes_type")
    else:
        number = float(value)
    if not number > 0 or number == float("inf"):
        return None, t("minutes_positive")
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
        session_factory: Callable[[int], object] | None = None,
        after_session: Callable[[object], None] | None = None,
        texts: Texts | None = None,
        language_detected: bool | None = None,
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
        self.session_factory = session_factory
        self.after_session = after_session
        self.uses_context = config.backend == "llamacpp"
        self.t = texts or _ENGLISH
        # None: do not announce the guide language (tests, embedding); True/False: say
        # whether it came from the operating system or from the setting.
        self.language_detected = language_detected
        self._queue_seconds = 0.0
        self._queued_in_input = True  # only files in the input folder are archived

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
        t = self.t
        self.say(t("queue_header", count=len(items)))
        total = 0.0
        for number, (source, duration, size) in enumerate(items, 1):
            size_text = t("size_unknown") if size is None else f"{size / 1_000_000:.1f} MB"
            if duration is None:
                self.say(
                    t("queue_item_no_duration", number=number, name=source.name, size=size_text)
                )
                continue
            total += duration
            self.say(
                t(
                    "queue_item",
                    number=number,
                    name=source.name,
                    size=size_text,
                    duration=format_duration(duration),
                    estimate=format_duration(duration * rtf),
                )
            )
        self._queue_seconds = total
        input_root = self.config.input_dir.resolve()
        self._queued_in_input = any(
            source.parent.resolve() == input_root for source, _d, _s in items
        )
        basis = t("estimate_measured") if measured else t("estimate_default", rtf=f"{rtf:g}")
        self.say(t("queue_total", total=format_duration(total * rtf), basis=basis))

    def _language_menu(self) -> list[str]:
        """auto, the guide's language, then the other translated languages."""
        first = [self.t.language] if self.t.language in UI_LANGUAGES else []
        return ["auto", *first, *(code for code in UI_LANGUAGES if code not in first)]

    def _ask_language(self, default: str | None) -> str | None:
        t = self.t
        menu = self._language_menu()
        self.say(t("language_intro"))
        for number, code in enumerate(menu, 1):
            label = t("auto_detect_long") if code == "auto" else own_name(code)
            self.say(f"   {number}  {code:<5} {label}")
        if self.config.response_mode == "qwen3-asr":
            module = importlib.import_module("..adapters.llamacpp", __package__)
            others = sorted(set(module.QWEN3_ASR_LANGUAGES) - set(menu))
            self.say(t("language_other_codes", codes=" ".join(others)))
        else:
            self.say(t("language_any_code"))
        self.say(t("language_examples"))
        while True:
            reply = self.ask(t("ask_language", default=language_label(default, t))).strip()
            if not reply:
                return default
            if reply.isdigit():
                number = int(reply)
                if 1 <= number <= len(menu):
                    code = menu[number - 1]
                    return None if code == "auto" else code
                self.say(t("menu_number_invalid", number=number, count=len(menu)))
                continue
            language, error = validate_language(reply, self.config.response_mode, t)
            if error is None:
                return language
            self.say(error)

    def _context_help(self) -> str:
        if self.config.response_mode == "qwen3-asr":
            return self.t("context_intro", limit=MAX_CONTEXT_CHARS)
        return self.t("context_help_prompt", mode=self.config.response_mode)

    def _ask_context(self, previous: str) -> tuple[str, str | None]:
        t = self.t
        if previous.strip():
            preview = context_preview(resolve_context(previous, t)[0], t)
            self.say(t("context_reuse", preview=preview))
        while True:
            reply = self.ask(t("ask_context")).strip()
            if reply == "=" and previous.strip():
                reply = previous.strip()
            text, error = resolve_context(reply, t)
            if error is None:
                if text:
                    self.say(
                        t("context_accepted", length=len(text), preview=context_preview(text, t))
                    )
                return reply, text
            self.say(error)

    def _ask_scope(self, default: float | None) -> float | None:
        full = self.t("scope_default_full")
        label = full if default is None else f"{default:g}"
        self.say(self.t("scope_intro", full=full))
        while True:
            reply = self.ask(self.t("ask_scope", default=label))
            if not reply.strip():
                return default
            minutes, error = parse_minutes(reply, self.t)
            if error is None:
                return minutes
            self.say(error)

    def _ask_after_success(self, default: str) -> str:
        t = self.t
        self.say(t("after_success_intro"))
        for number, policy in enumerate(AFTER_SUCCESS_CHOICES, 1):
            marker = t("default_marker") if policy == default else ""
            self.say(f"   {number}  {after_success_label(policy, t)}{marker}")
        if self._queue_seconds > 0:
            size = max(1, round(self._queue_seconds * FLAC_BYTES_PER_SECOND / 1_000_000))
            self.say(t("after_success_audio_estimate", size=size))
        self.say(t("after_success_note"))
        default_number = AFTER_SUCCESS_CHOICES.index(default) + 1
        while True:
            reply = self.ask(t("ask_after_success", default=default_number)).strip()
            if not reply:
                return default
            policy = parse_after_success(reply)
            if policy is not None:
                if policy != "keep-all":
                    self.say(t("retention_warning"))
                return policy
            self.say(t("after_success_invalid"))

    def _ask_answers(self, defaults: Answers, previous_context: str) -> Answers:
        language = self._ask_language(defaults.language)
        self.say(self.t("transcript_language_note"))
        if self.uses_context:
            self.say(self._context_help())
            context_input, context = self._ask_context(previous_context)
        else:
            self.say(self.t("context_backend_skipped", backend=self.config.backend))
            context_input, context = "", None
        max_minutes = self._ask_scope(defaults.max_minutes)
        after_success = defaults.after_success
        if max_minutes is None and self.config.archive_inputs and self._queued_in_input:
            after_success = self._ask_after_success(defaults.after_success)
        return Answers(language, context_input, context, max_minutes, after_success)

    def _scripted_answers(self, mapping: dict, defaults: Answers) -> Answers:
        language = defaults.language
        if "language" in mapping:
            language, error = validate_language(
                str(mapping["language"]), self.config.response_mode, self.t
            )
            if error:
                raise ValueError(error)
        context_input, context = defaults.context_input, defaults.context
        if "context" in mapping:
            context_input = str(mapping["context"] or "")
            context, error = resolve_context(context_input, self.t)
            if error:
                raise ValueError(error)
        max_minutes = defaults.max_minutes
        if "max_minutes" in mapping:
            max_minutes, error = parse_minutes(mapping["max_minutes"], self.t)
            if error:
                raise ValueError(error)
        after_success = defaults.after_success
        if "after_success" in mapping:
            after_success = parse_after_success(mapping["after_success"])
            if after_success is None:
                raise ValueError(self.t("after_success_invalid"))
        return Answers(language, context_input, context, max_minutes, after_success)

    def _collect(self, items) -> dict[Path, Answers] | None:
        last = load_last_answers(self.last_answers_path) if self.last_answers_path else {}
        language, _ = validate_language(
            str(last.get("language") or "auto"), self.config.response_mode
        )
        if "language" not in last:
            language = self.config.language
        minutes, _ = parse_minutes(last.get("max_minutes"))
        previous_context = last.get("context") if isinstance(last.get("context"), str) else ""
        defaults = Answers(language, "", None, minutes, self.config.after_success)
        if self.scripted is not None:
            base = self._scripted_answers(self.scripted, defaults)
            per_file = self.scripted.get("files") if isinstance(self.scripted, dict) else None
            per_file = per_file if isinstance(per_file, dict) else {}
            return {
                source: self._scripted_answers(per_file.get(source.name, {}), base)
                for source, _duration, _size in items
            }
        self.say("")
        if self.language_detected is not None:
            origin = "origin_system" if self.language_detected else "origin_setting"
            self.say(self.t("guide_language", name=self.t.name, origin=self.t(origin)))
        self.say(self.t("parameters_header"))
        answers = self._ask_answers(defaults, previous_context)
        chosen = {items[0][0]: answers}
        if len(items) > 1:
            same = self.ask(self.t("ask_same_for_all", count=len(items)))
            if same.strip().casefold() in YES_WORDS | {""}:
                chosen = {source: answers for source, _d, _s in items}
            else:
                for source, _d, _s in items[1:]:
                    self.say(self.t("answers_for", name=source.name))
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
            after_success=answers.after_success,
        )

    def _confirm(self, plan: dict[Path, Answers]) -> bool:
        config = self.config
        t = self.t
        self.say("")
        self.say(t("summary_header"))
        for source, answers in plan.items():
            length = len(answers.context) if answers.context else 0
            self.say(f"  {source.name}")
            self.say(t("summary_language", value=language_label(answers.language, t)))
            self.say(t("summary_transcript"))
            self.say(
                t("summary_context", preview=context_preview(answers.context, t), length=length)
            )
            self.say(t("summary_scope", value=scope_text(answers.max_minutes, t)))
            if answers.max_minutes is None and config.archive_inputs and self._queued_in_input:
                value = after_success_label(answers.after_success, t)
                self.say(t("summary_after_success", value=value))
        self.say(
            t(
                "summary_model",
                model=config.model,
                backend=config.backend,
                mode=config.response_mode,
                parallel=config.parallel_requests,
            )
        )
        self.say(t("summary_versions"))
        if any(a.max_minutes is not None for a in plan.values()):
            self.say(t("bounded_note"))
        if (
            config.archive_inputs
            and self._queued_in_input
            and any(a.max_minutes is None and a.after_success != "keep-all" for a in plan.values())
        ):
            self.say(t("retention_warning"))
        if self.scripted is not None:
            return bool(self.scripted.get("confirm", True))
        reply = self.ask(t("ask_start")).strip().casefold()
        return reply == "" or reply in YES_WORDS

    # ------------------------------------------------------------------ running
    def _run_file(self, source: Path, answers: Answers, session=None) -> dict:
        config = self._config_for(answers)
        started = self.clock()
        try:
            pipeline = self.make_pipeline(config)
            if session is not None and callable(getattr(pipeline, "attach", None)):
                pipeline.attach(events=session.bus, status=session.status_out)
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
        t = self.t
        status = str(result.get("status"))
        lines = [f"  {result['source_name']}", t("result_status", value=status)]
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
            lines.append(t("result_version", value=job_id))
        if summary is not None:
            chunks = summary["chunks"]
            na = t("not_available")
            lines.append(
                t(
                    "result_chunks",
                    ok=_n(chunks["ok"], na),
                    failed=_n(chunks["failed"], na),
                    no_speech=_n(chunks["no_speech"], na),
                    total=_n(chunks["total"], na),
                )
            )
            confidence = summary.get("confidence")
            mean = confidence.get("mean") if isinstance(confidence, dict) else None
            lines.append(t("result_confidence", value=_n(mean, na)))
        lines.append(t("result_time", value=format_timing(result["elapsed"])))
        if base is not None:
            for key, name in (
                ("result_transcript", "transcript.txt"),
                ("result_report", "run-report.md"),
            ):
                for candidate in (base / name, base / "intermediate" / name):
                    if candidate.is_file():
                        lines.append(t(key, path=candidate))
                        break
        disposition = result.get("source_disposition")
        if isinstance(disposition, dict) and disposition.get("audio_path"):
            lines.append(t("result_original_audio", path=disposition["audio_path"]))
        elif isinstance(disposition, dict):
            lines.append(t("result_original_deleted"))
        elif result.get("archived_source_path"):
            lines.append(t("result_original_archived", path=result["archived_source_path"]))
        if result.get("error"):
            lines.append(t("result_error", value=result["error"]))
        for key in ("archive_warning", "disposition_warning"):
            if result.get(key):
                lines.append(t("result_note", value=result[key]))
        return lines

    def run(self) -> int:
        try:
            return self._run()
        except _Cancelled:
            self.say(self.t("cancelled_eof"))
            return 0
        except ValueError as exc:
            self.say(self.t("error", message=exc))
            return 2

    def _run(self) -> int:
        items = self._queue()
        if not items:
            self.say(self.t("no_media", folder=self.config.input_dir))
            return 0
        self._show_queue(items)
        plan = self._collect(items)
        if plan is None or not self._confirm(plan):
            self.say(self.t("cancelled"))
            return 0
        results = []
        session = None
        try:
            with self.lock_factory():
                # The run session (event files, live display) covers only the processing
                # phase, so it never paints over the questions.
                session = self.session_factory(len(plan)) if self.session_factory else None
                try:
                    if session is not None:
                        session.start()
                    for number, (source, answers) in enumerate(plan.items(), 1):
                        if session is None or session.ui_mode == "plain":
                            self.say("")
                            self.say(f"[{number}/{len(plan)}] {source.name}")
                        results.append(self._run_file(source, answers, session))
                finally:
                    if session is not None:
                        good = all(r.get("status") in _GOOD_STATUSES for r in results)
                        session.finish(0 if good else 1, results)
                        session.close()
                        if self.after_session is not None:
                            self.after_session(session)
        except RuntimeError as exc:  # process lock held by another run
            self.say(self.t("error", message=exc))
            return 1
        self.say("")
        self.say(self.t("results_header"))
        for result in results:
            for line in self._result_lines(result):
                self.say(line)
        if self.open_folder is not None:
            reply = ""
            if self.scripted is None:
                with contextlib.suppress(_Cancelled):
                    reply = self.ask(self.t("ask_open_folder"))
            if reply.strip().casefold() in YES_WORDS:
                with contextlib.suppress(Exception):
                    self.open_folder(self.config.output_dir)
        return 0 if all(r.get("status") in _GOOD_STATUSES for r in results) else 1


def _n(value: object, missing: str = "n/a") -> str:
    return (
        missing
        if value is None
        else f"{value:g}"
        if isinstance(value, (int, float))
        else str(value)
    )

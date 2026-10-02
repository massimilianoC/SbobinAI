"""Command line interface for local media transcription."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .application.pipeline import ProcessLock, TranscriptionPipeline, run_watch
from .application.session import RunSession, exit_status_of
from .config import AppConfig, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbobinai",
        description="SbobinAI: transcribe local audio and video files.",
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in ("run", "watch", "doctor", "wizard"):
        sub = subparsers.add_parser(
            command,
            help="Guided interactive run: asks language, context and scope"
            if command == "wizard"
            else None,
        )
        _add_options(sub)
        if command == "wizard":
            sub.add_argument(
                "--answers",
                type=Path,
                default=None,
                help="JSON file with the answers (automation and tests); skips the questions",
            )
    for command, help_text in (
        ("catalog", "Rebuild catalog.json and every source.json from process/ and output/"),
        ("migrate-layout", "Move legacy flat job folders into the per-source layout (dry run)"),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument("--config", type=Path, help="Path to an optional TOML configuration file")
        sub.add_argument("--input-dir", type=Path, default=None)
        sub.add_argument("--process-dir", type=Path, default=None)
        sub.add_argument("--processed-dir", type=Path, default=None)
        sub.add_argument("--output-dir", type=Path, default=None)
        if command == "migrate-layout":
            sub.add_argument(
                "--apply",
                action="store_true",
                help="Perform the moves; without it only the plan is printed",
            )
    sub = subparsers.add_parser(
        "events", help="Print a run's events as JSONL (tail -f style with --follow)"
    )
    sub.add_argument("--config", type=Path, help="Path to an optional TOML configuration file")
    sub.add_argument("--process-dir", type=Path, default=None)
    sub.add_argument("--run", default="latest", help="Run id or 'latest' (default)")
    sub.add_argument(
        "--follow", action="store_true", help="Keep polling until the run emits run.finished"
    )
    sub.add_argument("--type", default=None, help="Only events whose type starts with this prefix")
    sub.add_argument("--poll-interval", type=float, default=0.5, help=argparse.SUPPRESS)
    sub = subparsers.add_parser(
        "server-profile", help="Print the resolved [server] profile as JSON (absolute paths)"
    )
    sub.add_argument("--config", type=Path, required=True)
    sub.add_argument(
        "--optional",
        action="store_true",
        help="Print null and exit 0 when the configuration has no [server] table",
    )
    sub = subparsers.add_parser(
        "setup", help="Download the pinned resources of a profile into the resource store"
    )
    sub.add_argument("--config", type=Path, help="Local configuration providing [resources].store")
    _add_resource_options(sub)
    sub.add_argument("--store", type=Path, default=None, help="Resource store folder")
    sub.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    sub.add_argument("--dry-run", action="store_true", help="Only print the plan")
    sub = subparsers.add_parser(
        "init-config", help="Write a ready-to-use local configuration for a profile"
    )
    _add_resource_options(sub)
    sub.add_argument("--store", type=Path, required=True, help="Resource store folder")
    sub.add_argument("--output", type=Path, default=Path("config.local.toml"))
    sub.add_argument("--template", type=Path, default=None, help=argparse.SUPPRESS)
    sub.add_argument("--force", action="store_true", help="Overwrite an existing file")
    return parser


def _add_resource_options(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--profile", default=None, help="Profile from resources.json")
    sub.add_argument("--manifest", type=Path, default=None, help="Path to resources.json")


def _float_list(value: str) -> list[float]:
    try:
        return [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of numbers: {value}"
        ) from exc


def _add_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="Path to an optional TOML configuration file")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--input", dest="input_dir", type=Path, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--process-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=("nexa", "mock", "llamacpp"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--base-url",
        default=None,
        help="llama.cpp server base URL, for example http://127.0.0.1:8088",
    )
    parser.add_argument(
        "--timeout", type=float, default=None, help="llama.cpp request timeout in seconds"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Maximum generated tokens for llama.cpp"
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--response-mode", choices=("json", "plain", "qwen3-asr"), default=None)
    prompt_options = parser.add_mutually_exclusive_group()
    prompt_options.add_argument(
        "--prompt", default=None, help="Override the llama.cpp transcription prompt"
    )
    prompt_options.add_argument(
        "--prompt-file", type=Path, default=None, help="Read a UTF-8 llama.cpp prompt file"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Use an existing local Nexa model path without downloading weights",
    )
    parser.add_argument(
        "--projector-path",
        type=Path,
        default=None,
        help="Existing local projector file required by the legacy Nexa backend",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument(
        "--language",
        default=None,
        help="Spoken language code, or auto for no hint (overrides a language set in the config)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Maximum chunk length in seconds; the VAD cuts at pauses",
    )
    parser.add_argument("--vad", choices=("silero", "energy", "none"), default=None)
    parser.add_argument("--vad-model-path", type=Path, default=None)
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--vad-min-speech-seconds", type=float, default=None)
    parser.add_argument("--vad-min-silence-seconds", type=float, default=None)
    parser.add_argument("--vad-speech-pad-seconds", type=float, default=None)
    parser.add_argument("--vad-max-merge-gap-seconds", type=float, default=None)
    parser.add_argument("--vad-energy-margin-db", type=float, default=None)
    parser.add_argument(
        "--fallback-temperatures",
        type=_float_list,
        default=None,
        help="Comma-separated temperatures tried after degenerate output; empty disables",
    )
    parser.add_argument("--split-on-failure", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--context-free-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retry a chunk without the configured prompt/context when the model echoed it",
    )
    parser.add_argument(
        "--force-language",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Qwen3-ASR: constrain output to the configured --language instead of auto-detection",
    )
    parser.add_argument(
        "--collect-logprobs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request token log-probabilities for the uncalibrated confidence proxy",
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=None,
        help="Concurrent chunk requests (1-8); must not exceed the server's -Parallel slots",
    )
    parser.add_argument(
        "--intermediate-interval-seconds",
        type=float,
        default=None,
        help="Minimum seconds between intermediate export rewrites; 0 rewrites after every chunk",
    )
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--tokens-per-second", type=float, default=None)
    parser.add_argument("--compression-ratio-threshold", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--dry-multiplier", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument(
        "--max-file-size", type=int, default=None, help="Maximum input size in bytes"
    )
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--watch-interval", type=float, default=None)
    parser.add_argument("--stable-scans", type=int, default=None)
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        default=None,
        help="Process this media file instead of discovering the input directory; repeatable",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of discovered files to process"
    )
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prepare-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--archive-inputs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--ui",
        choices=("auto", "live", "plain", "jsonl"),
        default=None,
        help="Console output: live (progress display), plain (status lines), jsonl (event "
        "stream on stdout) or auto (live only on an interactive terminal)",
    )
    parser.add_argument(
        "--monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sample CPU, RAM and NVIDIA GPU use during the run (--no-monitor disables)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=None,
        help="Seconds between resource samples (default 1.0)",
    )


def _make_detector(config: AppConfig):
    """Build the configured speech detector, or None when VAD is disabled."""
    make_detector = importlib.import_module(".adapters.vad", "audio_transcript").make_detector
    return make_detector(
        config.vad,
        model_path=config.vad_model_path,
        threshold=config.vad_threshold,
        energy_margin_db=config.vad_energy_margin_db,
        ffmpeg=config.ffmpeg,
    )


def _make_processor(config: AppConfig):
    cls = importlib.import_module(".adapters.ffmpeg", "audio_transcript").FFmpegProcessor
    return cls(ffmpeg=config.ffmpeg, ffprobe=config.ffprobe, detector=_make_detector(config))


def _make_backend(config: AppConfig):
    module, class_name = {
        "nexa": (".adapters.nexa", "NexaBackend"),
        "mock": (".adapters.mock", "MockBackend"),
        "llamacpp": (".adapters.llamacpp", "LlamaCppBackend"),
    }[config.backend]
    cls = getattr(importlib.import_module(module, "audio_transcript"), class_name)
    if config.backend == "nexa":
        return cls(
            model=config.model,
            device=config.device,
            local_path=config.model_path,
            projector_local_path=config.projector_path,
        )
    if config.backend == "mock":
        return cls(model=config.model)
    return cls(
        model=config.model,
        base_url=config.base_url,
        timeout=config.timeout,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        seed=config.seed,
        response_mode=config.response_mode,
        prompt=config.prompt,
        min_tokens=config.min_tokens,
        tokens_per_second=config.tokens_per_second,
        compression_ratio_threshold=config.compression_ratio_threshold,
        repeat_penalty=config.repeat_penalty,
        dry_multiplier=config.dry_multiplier,
        force_language=config.force_language,
        collect_logprobs=config.collect_logprobs,
    )


def _make_exporter():
    cls = importlib.import_module(".adapters.exporters", "audio_transcript").FileExporter
    return cls()


def _doctor(config: AppConfig) -> int:
    problems = []
    for executable in (config.ffmpeg, config.ffprobe):
        located = shutil.which(executable)
        if located is None and not Path(executable).is_file():
            problems.append(f"Missing executable: {executable}")
            continue
        try:
            result = subprocess.run(
                [executable, "-version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode:
                problems.append(
                    f"Executable check failed for {executable}: {result.stderr.strip() or result.returncode}"
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"Could not run {executable}: {exc}")
    try:
        detector = _make_detector(config)
        if detector is not None:
            detector.check()
            print(f"Speech detector available: {config.vad}")
    except Exception as exc:
        problems.append(f"Speech detector unavailable ({config.vad}): {type(exc).__name__}: {exc}")
    backend = None
    try:
        backend = _make_backend(config)
        backend.check()
        print(f"Backend available: {config.backend} ({config.model})")
    except Exception as exc:
        problems.append(f"Backend unavailable ({config.backend}): {type(exc).__name__}: {exc}")
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                problems.append(f"Backend cleanup failed: {exc}")
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print(f"Media tools available: {config.ffmpeg}, {config.ffprobe}")
    return 0


def _layout_command(args, command: str) -> int:
    from .application.catalog import rebuild_all
    from .application.migration import apply_migration, format_plan, plan_migration

    try:
        config = load_config(
            args.config,
            {
                key: getattr(args, key)
                for key in ("input_dir", "process_dir", "processed_dir", "output_dir")
            },
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        config.process_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if command == "catalog":
            with ProcessLock(config.process_dir / ".audio-transcript.lock"):
                summary = rebuild_all(config)
            print(
                f"Catalog rebuilt: {summary['sources']} source(s), {summary['versions']} version(s)"
            )
            return 0
        items = plan_migration(config)
        for line in format_plan(items):
            print(line)
        movable = [item for item in items if item.status == "move"]
        refused = [item for item in items if item.status == "conflict"]
        print(
            f"{len(movable)} job(s) to migrate, {len(refused)} refused, "
            f"{len(items) - len(movable) - len(refused)} left untouched"
        )
        if not args.apply:
            print("Dry run: nothing was moved. Re-run with --apply to perform the moves.")
            return 0
        with ProcessLock(config.process_dir / ".audio-transcript.lock"):
            log_path, summary = apply_migration(config, items)
        print(
            f"Migrated {summary['migrated']} job(s), {summary['failed']} failed, "
            f"{summary['refused']} refused. Move log: {log_path}"
        )
        print(
            f"Catalog rebuilt: {summary['catalog']['sources']} source(s), "
            f"{summary['catalog']['versions']} version(s)"
        )
        return 1 if summary["failed"] else 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _server_profile_command(args) -> int:
    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    server = config.server
    if server is None:
        if args.optional:
            print("null")
            return 0
        print(
            f"ERROR: {args.config} has no [server] table; add one (see config.example.toml)",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "runtime_dir": str(server.runtime_dir),
                "model_path": str(server.model_path),
                "projector_path": str(server.projector_path),
                "alias": server.alias,
                "state_dir": str(server.state_dir),
                "port": server.port,
                "context_size": server.context_size,
                "parallel": server.parallel,
                "gpu_layers": server.gpu_layers,
                "base_url": config.base_url,
            },
            indent=2,
        )
    )
    return 0


def _progress_printer(name: str):
    def show(done: int, total: int, speed: float) -> None:
        sys.stdout.write(
            f"\r  {name}: {100 * done / total:5.1f}%  {speed / 1_000_000:6.1f} MB/s   "
        )
        sys.stdout.flush()

    return show


def _setup_command(args) -> int:
    from .application import resources as res

    download = importlib.import_module(".adapters.download", "audio_transcript")
    try:
        config = load_config(args.config) if args.config else None
        manifest = res.load_manifest(args.manifest)
        profile = res.get_profile(manifest, args.profile)
        if args.store is not None:
            store = args.store.expanduser().resolve()
        elif config is not None and config.resources is not None:
            store = config.resources.store
        else:
            raise ValueError("No resource store: pass --store or add [resources].store to --config")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Profile {profile.name}: {profile.description}")
    items = res.build_plan(manifest, profile, store, hasher=download.sha256_file)
    for line in res.format_plan(items, store, res.free_space(store)):
        print(line)
    conflicts = [item for item in items if item.status == "conflict"]
    for item in conflicts:
        print(
            f"ERROR: {item.path} exists but does not match the pinned file; it is never "
            "overwritten. Move it away and run setup again.",
            file=sys.stderr,
        )
    todo = [item for item in items if item.status in {"download", "resume"}]
    if any(item.status == "installer" for item in items):
        runtime_dir = config.server.runtime_dir if config and config.server else None
        print(f"\nInstall the llama.cpp runtime with:\n  {res.runtime_command(store, runtime_dir)}")
    if args.dry_run:
        print("\nDry run: nothing was downloaded.")
        return 1 if conflicts else 0
    if conflicts:
        return 1
    if not todo:
        print("\nNothing to download.")
        return 0
    total = sum(item.download_bytes for item in todo)
    if total > res.free_space(store):
        print("ERROR: not enough free disk space at the store.", file=sys.stderr)
        return 1
    if not args.yes:
        try:
            answer = input(f"\nDownload {total / 1_000_000:,.1f} MB into {store}? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled; nothing was downloaded.")
            return 1
    failures = 0
    for item in todo:
        try:
            download.download_verified(
                item.resource.url,
                item.path,
                sha256=item.resource.sha256,
                size_bytes=item.resource.size_bytes,
                progress=_progress_printer(item.path.name),
            )
            res.write_provenance(item.path, item.resource)
            print(f"\n  verified and installed: {item.path}")
        except (download.DownloadError, OSError) as exc:
            failures += 1
            print(f"\nERROR: {item.resource.id}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def _init_config_command(args) -> int:
    import tempfile

    from .application import resources as res

    try:
        manifest = res.load_manifest(args.manifest)
        profile = res.get_profile(manifest, args.profile)
        template_path = args.template or res.default_manifest_path().with_name(
            "config.example.toml"
        )
        try:
            template = template_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Could not read the config template {template_path}: {exc}") from exc
        text = res.render_config(template, manifest, profile, args.store.expanduser())
        if args.output.exists() and not args.force:
            print(
                f"ERROR: {args.output} already exists; use --force to overwrite it",
                file=sys.stderr,
            )
            return 1
        with tempfile.TemporaryDirectory() as scratch:
            probe = Path(scratch) / "config.toml"
            probe.write_text(text, encoding="utf-8")
            load_config(probe)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output} for profile {profile.name} (store {args.store.resolve()})")
    print(f"Next: audio-transcript setup --config {args.output} --profile {profile.name}")
    return 0


def _monitor_factory(config: AppConfig):
    """Build the resource monitor lazily so the OS samplers load only when enabled."""

    def make(bus):
        from .application.monitor import ResourceMonitor

        sysmon = importlib.import_module(".adapters.sysmon", "audio_transcript")
        return ResourceMonitor(
            bus, sysmon.make_samplers(config.monitor_interval), config.monitor_interval
        )

    return make


def _open_session(config: AppConfig, command: str, queue_size: int | None) -> RunSession:
    """Create the observable run: event files, plain log, console renderer, monitor."""
    from .ui import resolve_ui_mode
    from .ui.plain import status_printer

    mode = resolve_ui_mode(config.ui)
    console = None
    if mode == "live":
        from .ui.live import LiveConsole

        console = LiveConsole()
    return RunSession(
        config,
        command=command,
        ui_mode=mode,
        queue_size=queue_size,
        console_sink=console,
        monitor_factory=_monitor_factory(config),
        status_out=status_printer(),
    )


def _attach(pipeline, session: RunSession) -> None:
    attach = getattr(pipeline, "attach", None)
    if callable(attach):
        attach(events=session.bus, status=session.status_out)


def _queue_size(pipeline, sources, limit: int | None) -> int | None:
    try:
        size = len(sources) if sources is not None else len(pipeline.discover())
    except Exception:
        return None
    return size if sources is not None or limit is None else min(size, limit)


def _after_session(session: RunSession) -> None:
    """Lines printed once the console is released (live runs hid the status lines)."""
    if session.ui_mode != "live":
        return
    from .ui.plain import final_summary

    for line in final_summary(
        session.state,
        {"log": session.paths.log, "events": session.paths.events},
    ):
        print(line)


def _say(session: RunSession | None, text: str) -> None:
    """Print a message without disturbing the console mode of the run."""
    if session is not None and session.ui_mode == "jsonl":
        print(text, file=sys.stderr)
    else:
        print(text)


def _events_command(args) -> int:
    from .application.eventlog import events_path, follow_events

    try:
        config = load_config(args.config, {"process_dir": args.process_dir})
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    path = events_path(config.process_dir, args.run)
    if path is None:
        print(
            f"ERROR: no run found for '{args.run}' in {config.process_dir / 'runs'}",
            file=sys.stderr,
        )
        return 1
    try:
        code = follow_events(
            path,
            lambda line: print(line, flush=True),
            type_prefix=args.type,
            follow=args.follow,
            poll_interval=max(0.05, args.poll_interval),
        )
    except KeyboardInterrupt:
        return 0
    except BrokenPipeError:
        return 0
    if code:
        print(f"ERROR: no events file at {path}", file=sys.stderr)
    return code


def _make_pipeline(config: AppConfig) -> TranscriptionPipeline:
    processor = _make_processor(config)
    backend = None if config.prepare_only else _make_backend(config)
    exporter = None if config.prepare_only else _make_exporter()
    return TranscriptionPipeline(config, processor, backend, exporter)


def _open_folder(path: Path) -> None:
    import os

    opener = getattr(os, "startfile", None)
    if opener is not None:
        opener(str(path))


def _wizard_command(args, config: AppConfig) -> int:
    from .application.wizard import Wizard

    scripted = None
    if args.answers is not None:
        try:
            scripted = json.loads(args.answers.read_text(encoding="utf-8"))
            if not isinstance(scripted, dict):
                raise ValueError("the answers file must contain a JSON object")
        except (OSError, ValueError) as exc:
            print(f"ERROR: Could not read the answers file {args.answers}: {exc}", file=sys.stderr)
            return 2
    elif not (sys.stdin and sys.stdin.isatty()):
        print(
            "ERROR: the wizard needs an interactive console. Use "
            "'audio-transcript run' for unattended runs, or pass --answers <json file>.",
            file=sys.stderr,
        )
        return 2
    lock_path = config.process_dir / ".audio-transcript.lock"
    wizard = Wizard(
        config,
        _make_pipeline,
        lock_factory=lambda: ProcessLock(lock_path),
        open_folder=_open_folder if scripted is None and sys.platform == "win32" else None,
        scripted=scripted,
        sources=[path.expanduser() for path in args.file] if args.file else None,
        session_factory=lambda queue_size: _open_session(config, "wizard", queue_size),
        after_session=_after_session,
    )
    try:
        return wizard.run()
    except KeyboardInterrupt:
        print("\nInterrupted; job state remains recoverable.", file=sys.stderr)
        return 130


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["run"]
    if raw[0] not in {
        "run",
        "watch",
        "wizard",
        "doctor",
        "catalog",
        "migrate-layout",
        "events",
        "server-profile",
        "setup",
        "init-config",
        "-h",
        "--help",
    }:
        raw.insert(0, "run")
    parser = _parser()
    args = parser.parse_args(raw)
    command = args.command or "run"
    if command in {"catalog", "migrate-layout"}:
        return _layout_command(args, command)
    if command == "events":
        return _events_command(args)
    if command == "server-profile":
        return _server_profile_command(args)
    if command == "setup":
        return _setup_command(args)
    if command == "init-config":
        return _init_config_command(args)
    try:
        if args.prompt_file is not None:
            try:
                prompt_text = args.prompt_file.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                print(
                    f"ERROR: Could not read UTF-8 prompt file {args.prompt_file}: {exc}",
                    file=sys.stderr,
                )
                return 2
        else:
            prompt_text = args.prompt
        config_values = {
            key: value
            for key, value in vars(args).items()
            if key not in {"command", "config", "file", "limit", "prompt_file", "answers"}
        }
        config_values["prompt"] = prompt_text
        config = load_config(args.config, config_values)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if command == "doctor":
        return _doctor(config)
    if command == "wizard":
        return _wizard_command(args, config)
    try:
        processor = _make_processor(config)
        backend = None if config.prepare_only else _make_backend(config)
        exporter = None if config.prepare_only else _make_exporter()
        pipeline = TranscriptionPipeline(config, processor, backend, exporter)
        lock_path = config.process_dir / ".audio-transcript.lock"
        with ProcessLock(lock_path):
            if command == "watch":
                if args.file or args.limit is not None:
                    print(
                        "ERROR: --file and --limit are only supported by the run command",
                        file=sys.stderr,
                    )
                    return 2
                session = _open_session(config, "watch", None)
                try:
                    with session:
                        _attach(pipeline, session)
                        try:
                            run_watch(
                                pipeline,
                                interval=config.watch_interval,
                                stable_scans=config.stable_scans,
                            )
                        except KeyboardInterrupt:
                            session.finish(0)
                        finally:
                            if backend is not None:
                                backend.close()
                finally:
                    _after_session(session)
                _say(session, "Watch stopped")
                return 0
            sources = [path.expanduser() for path in args.file] if args.file else None
            queue_size = _queue_size(pipeline, sources, args.limit)
            session = _open_session(config, "run", queue_size)
            try:
                with session:
                    _attach(pipeline, session)
                    results = pipeline.run(sources, limit=args.limit)
                    session.finish(exit_status_of(results), results)
            finally:
                _after_session(session)
            if not results and sources is None:
                _say(
                    session,
                    f"No media files found in {config.input_dir}; "
                    "put audio/video files there and run again.",
                )
        return exit_status_of(results)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: interrupted; job state remains recoverable", file=sys.stderr)
        return 130

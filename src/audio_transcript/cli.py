"""Command line interface for local media transcription."""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

from .application.pipeline import ProcessLock, TranscriptionPipeline, run_watch
from .config import AppConfig, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-transcript", description="Transcribe local audio and video files."
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in ("run", "watch", "doctor"):
        sub = subparsers.add_parser(command)
        _add_options(sub)
    return parser


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
    parser.add_argument("--response-mode", choices=("json", "plain"), default=None)
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
    parser.add_argument("--language", default=None)
    parser.add_argument("--chunk-seconds", type=float, default=None)
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


def _make_processor(config: AppConfig):
    cls = importlib.import_module(".adapters.ffmpeg", "audio_transcript").FFmpegProcessor
    return cls(ffmpeg=config.ffmpeg, ffprobe=config.ffprobe)


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


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["run"]
    if raw[0] not in {"run", "watch", "doctor", "-h", "--help"}:
        raw.insert(0, "run")
    parser = _parser()
    args = parser.parse_args(raw)
    command = args.command or "run"
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
            if key not in {"command", "config", "file", "limit", "prompt_file"}
        }
        config_values["prompt"] = prompt_text
        config = load_config(args.config, config_values)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if command == "doctor":
        return _doctor(config)
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
                try:
                    run_watch(
                        pipeline, interval=config.watch_interval, stable_scans=config.stable_scans
                    )
                except KeyboardInterrupt:
                    print("Watch stopped")
                finally:
                    if backend is not None:
                        backend.close()
                return 0
            sources = [path.expanduser() for path in args.file] if args.file else None
            results = pipeline.run(sources, limit=args.limit)
        return 1 if any(result["status"] == "failed" for result in results) else 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: interrupted; job state remains recoverable", file=sys.stderr)
        return 130

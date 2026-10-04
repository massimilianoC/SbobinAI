"""Command line interface for local media transcription.

The parser is built from ``clidoc`` (option catalogue and command text), so ``--help``,
the manual and the machine-readable description cannot drift apart.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__, clidoc, climan
from .application.backend_select import (
    BackendSelectionError,
    Selection,
    describe_runtime,
    select_backend,
)
from .application.pipeline import ProcessLock, TranscriptionPipeline, run_watch
from .application.session import RunSession, exit_status_of
from .clidoc import EXIT_FAILED, EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE
from .config import AppConfig, load_config

COMMAND_NAMES = tuple(command.name for command in clidoc.COMMANDS)
_ERROR_STATUS = {code: status for code, status, _text in clidoc.ERROR_CODES}
_RUN_COMMANDS = {"run", "watch", "wizard"}


# ------------------------------------------------------------------ structured output


def _print_json(payload) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _fail(as_json: bool, code: str, message: str, extra: dict | None = None) -> int:
    """Report an error and return its exit status (JSON on stdout, or text on stderr)."""
    status = _ERROR_STATUS[code]
    if as_json:
        payload = {"ok": False}
        payload.update(extra or {})
        payload["error"] = {"code": code, "message": message, "exit_status": status}
        _print_json(payload)
    else:
        print(f"ERROR: {message}", file=sys.stderr)
    return status


def _write_stdout(text: str) -> None:
    """Write exact bytes (LF newlines, so `> file` redirection is reproducible)."""
    buffer = getattr(sys.stdout, "buffer", None)
    try:
        if buffer is not None:
            sys.stdout.flush()
            buffer.write(text.encode("utf-8"))
            buffer.flush()
        else:
            sys.stdout.write(text)
    except BrokenPipeError:
        pass


# ------------------------------------------------------------------ parser


class _Formatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog, indent_increment=2, max_help_position=30, width=None):
        super().__init__(prog, indent_increment, max_help_position, clidoc.HELP_WIDTH)


class _Parser(argparse.ArgumentParser):
    """ArgumentParser with fixed-width plain help and JSON usage errors."""

    json_errors = False
    overview = False

    def __init__(self, *args, **kwargs):
        if sys.version_info >= (3, 14):
            kwargs.setdefault("color", False)
        super().__init__(*args, **kwargs)

    def format_help(self) -> str:
        return climan.overview() if self.overview else super().format_help()

    def error(self, message: str):
        if _Parser.json_errors:
            _fail(True, "usage", message)
            raise SystemExit(EXIT_USAGE)
        super().error(message)


def _float_list(value: str) -> list[float]:
    try:
        return [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of numbers: {value}"
        ) from exc


_CONVERTERS = {"str": None, "path": Path, "int": int, "float": float, "floatlist": _float_list}


def _add_option(target, opt: clidoc.Opt) -> None:
    help_text = argparse.SUPPRESS if opt.hidden else climan.option_help(opt).replace("%", "%%")
    if opt.positional:
        target.add_argument(
            "topic",  # not "command": that is the dest of the subcommand itself
            nargs=opt.nargs,
            choices=COMMAND_NAMES,
            metavar=opt.metavar,
            help=help_text,
        )
        return
    default = None if opt.default is clidoc.NO_DEFAULT else opt.default
    if opt.kind == "flag":
        target.add_argument(*opt.flags, dest=opt.dest, action="store_true", help=help_text)
    elif opt.kind == "bool":
        target.add_argument(
            *opt.flags,
            dest=opt.dest,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=help_text,
        )
    elif opt.kind == "append":
        target.add_argument(
            *opt.flags,
            dest=opt.dest,
            action="append",
            type=_CONVERTERS[opt.type],
            default=None,
            metavar=opt.metavar,
            help=help_text,
        )
    else:
        target.add_argument(
            *opt.flags,
            dest=opt.dest,
            type=_CONVERTERS[opt.type],
            default=default,
            choices=opt.choices,
            metavar=opt.metavar,
            required=opt.required,
            help=help_text,
        )


def _build_command(subparsers, command: clidoc.Command) -> None:
    sub = subparsers.add_parser(
        command.name,
        help=command.summary,
        usage=climan.synopsis(command),
        description=climan.command_description(command),
        epilog=climan.command_epilog(command),
        formatter_class=_Formatter,
    )
    for opt in command.options:
        if opt.positional:
            _add_option(sub, opt)
    for opt in command.options:
        if opt.hidden:
            _add_option(sub, opt)
    for title, opts in climan.grouped_options(command):
        if title == "Arguments":
            continue
        group = sub.add_argument_group(title)
        exclusive = None
        for opt in opts:
            if opt.name in {"prompt", "prompt_file"}:
                if exclusive is None:
                    exclusive = group.add_mutually_exclusive_group()
                _add_option(exclusive, opt)
            else:
                _add_option(group, opt)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog=clidoc.PROGRAM, formatter_class=_Formatter, add_help=False)
    parser.overview = True
    parser.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version=f"sbobinai {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for command in clidoc.COMMANDS:
        _build_command(subparsers, command)
    return parser


def _subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name]
    raise KeyError(name)


# ------------------------------------------------------------------ factories


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


# ------------------------------------------------------------------ backend selection


def _select_backend(config: AppConfig) -> Selection:
    """Choose cuda, vulkan or cpu for the configured [server] (runs ``--list-devices``)."""
    devices = importlib.import_module(".adapters.devices", "audio_transcript")
    if config.server is None:
        raise BackendSelectionError("The configuration has no [server] table")
    return select_backend(
        config.server, devices.list_devices, threads_default=devices.default_cpu_threads
    )


def _runtime_info(config: AppConfig) -> dict | None:
    """Path-free backend record for a run; None for an unmanaged server or a failed selection.

    The server is already started by the launcher, so a failure here must not stop the run;
    it only means the reports will not name the backend.
    """
    if config.server is None or config.backend != "llamacpp" or config.prepare_only:
        return None
    try:
        return _select_backend(config).run_info()
    except Exception:
        return None


# ------------------------------------------------------------------ doctor


def _tool_check(name: str, executable: str) -> dict:
    """Check that an external tool (FFmpeg, FFprobe) exists and runs ``-version``."""
    located = shutil.which(executable)
    if located is None and not Path(executable).is_file():
        return {"name": name, "ok": False, "detail": f"Missing executable: {executable}"}
    try:
        result = subprocess.run(
            [executable, "-version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"name": name, "ok": False, "detail": f"Could not run {executable}: {exc}"}
    if result.returncode:
        detail = (
            f"Executable check failed for {executable}: "
            f"{result.stderr.strip() or result.returncode}"
        )
        return {"name": name, "ok": False, "detail": detail}
    return {"name": name, "ok": True, "detail": str(located or executable)}


def _doctor_checks(config: AppConfig) -> list[dict]:
    """Run the readiness checks; every check is {name, ok, detail}."""
    checks: list[dict] = [
        _tool_check(name, executable)
        for name, executable in (("ffmpeg", config.ffmpeg), ("ffprobe", config.ffprobe))
    ]
    try:
        detector = _make_detector(config)
        if detector is not None:
            detector.check()
            detail = f"Speech detector available: {config.vad}"
        else:
            detail = "Speech detection disabled (vad = none)"
        checks.append({"name": "speech_detector", "ok": True, "detail": detail})
    except Exception as exc:
        detail = f"Speech detector unavailable ({config.vad}): {type(exc).__name__}: {exc}"
        checks.append({"name": "speech_detector", "ok": False, "detail": detail})
    if config.server is not None and config.backend == "llamacpp" and not config.prepare_only:
        try:
            selection = _select_backend(config)
            line = describe_runtime(selection.run_info()) or "Backend: selected"
            detail = line.replace("Backend:", "Inference backend:", 1)
            checks.append(
                {
                    "name": "inference_backend",
                    "ok": True,
                    "detail": detail,
                    "selection": selection.to_dict(),
                }
            )
        except BackendSelectionError as exc:
            checks.append(
                {
                    "name": "inference_backend",
                    "ok": False,
                    "detail": str(exc),
                    "selection": {
                        "requested": config.server.backend,
                        "tried": [
                            {"backend": a.backend, "ok": a.ok, "reason": a.reason}
                            for a in exc.tried
                        ],
                    },
                }
            )
    backend = None
    try:
        backend = _make_backend(config)
        backend.check()
        detail = f"Backend available: {config.backend} ({config.model})"
        checks.append({"name": "backend", "ok": True, "detail": detail})
    except Exception as exc:
        detail = f"Backend unavailable ({config.backend}): {type(exc).__name__}: {exc}"
        checks.append({"name": "backend", "ok": False, "detail": detail})
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                checks.append(
                    {
                        "name": "backend_cleanup",
                        "ok": False,
                        "detail": f"Backend cleanup failed: {exc}",
                    }
                )
    return checks


def _doctor_warnings(checks: list[dict]) -> list[str]:
    warnings = []
    for check in checks:
        selection = check.get("selection")
        if check["ok"] and isinstance(selection, dict) and selection.get("fallback_used"):
            warnings.append(selection.get("warning") or check["detail"])
    return warnings


def _doctor(config: AppConfig, as_json: bool = False) -> int:
    checks = _doctor_checks(config)
    failed = [check for check in checks if not check["ok"]]
    warnings = _doctor_warnings(checks)
    if as_json:
        payload: dict = {"ok": not failed, "checks": checks, "warnings": warnings}
        if failed:
            payload["error"] = {
                "code": "checks_failed",
                "message": "; ".join(check["detail"] for check in failed),
                "exit_status": EXIT_FAILED,
            }
        _print_json(payload)
        return EXIT_FAILED if failed else EXIT_OK
    for check in checks:
        if check["ok"] and check["name"] == "speech_detector" and config.vad != "none":
            print(check["detail"])
        elif check["ok"] and check["name"] in {"backend", "inference_backend"}:
            print(check["detail"])
    for warning in warnings:
        print(f"WARNING: {warning}")
    if failed:
        for check in failed:
            print(f"ERROR: {check['detail']}", file=sys.stderr)
        return EXIT_FAILED
    print(f"Media tools available: {config.ffmpeg}, {config.ffprobe}")
    return EXIT_OK


# ------------------------------------------------------------------ catalog / migrate


def _layout_command(args, command: str) -> int:
    from .application.catalog import rebuild_all
    from .application.migration import apply_migration, format_plan, plan_migration

    as_json = args.json
    try:
        config = load_config(
            args.config,
            {
                key: getattr(args, key)
                for key in ("input_dir", "process_dir", "processed_dir", "output_dir")
            },
        )
    except ValueError as exc:
        return _fail(as_json, "config_invalid", str(exc))
    try:
        config.process_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if command == "catalog":
            with ProcessLock(config.process_dir / ".audio-transcript.lock"):
                summary = rebuild_all(config)
            if as_json:
                _print_json(
                    {
                        "ok": True,
                        "command": "catalog",
                        "sources": summary["sources"],
                        "versions": summary["versions"],
                    }
                )
            else:
                print(
                    f"Catalog rebuilt: {summary['sources']} source(s), "
                    f"{summary['versions']} version(s)"
                )
            return EXIT_OK
        items = plan_migration(config)
        plan_lines = list(format_plan(items))
        movable = [item for item in items if item.status == "move"]
        refused = [item for item in items if item.status == "conflict"]
        untouched = len(items) - len(movable) - len(refused)
        result = {
            "command": "migrate-layout",
            "dry_run": not args.apply,
            "plan": plan_lines,
            "to_migrate": len(movable),
            "refused": len(refused),
            "untouched": untouched,
        }
        if not as_json:
            for line in plan_lines:
                print(line)
            print(
                f"{len(movable)} job(s) to migrate, {len(refused)} refused, "
                f"{untouched} left untouched"
            )
        if not args.apply:
            if as_json:
                _print_json({"ok": True, **result})
            else:
                print("Dry run: nothing was moved. Re-run with --apply to perform the moves.")
            return EXIT_OK
        with ProcessLock(config.process_dir / ".audio-transcript.lock"):
            log_path, summary = apply_migration(config, items)
        result.update(
            migrated=summary["migrated"],
            failed=summary["failed"],
            refused=summary["refused"],
            log=str(log_path),
            catalog=summary["catalog"],
        )
        if as_json:
            if summary["failed"]:
                return _fail(
                    True,
                    "migration_failed",
                    f"{summary['failed']} job(s) could not be moved",
                    extra=result,
                )
            _print_json({"ok": True, **result})
            return EXIT_OK
        print(
            f"Migrated {summary['migrated']} job(s), {summary['failed']} failed, "
            f"{summary['refused']} refused. Move log: {log_path}"
        )
        print(
            f"Catalog rebuilt: {summary['catalog']['sources']} source(s), "
            f"{summary['catalog']['versions']} version(s)"
        )
        return EXIT_FAILED if summary["failed"] else EXIT_OK
    except Exception as exc:
        return _fail(as_json, _classify(exc), f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ server-profile


def _server_profile_command(args) -> int:
    as_json = args.json
    try:
        config = load_config(args.config)
    except ValueError as exc:
        return _fail(as_json, "config_invalid", str(exc))
    server = config.server
    if server is None:
        if args.optional:
            print("null")
            return EXIT_OK
        return _fail(
            as_json,
            "server_not_configured",
            f"{args.config} has no [server] table; add one (see config.example.toml)",
        )
    try:
        selection = _select_backend(config)
    except BackendSelectionError as exc:
        return _fail(as_json, "backend_unavailable", str(exc))
    print(
        json.dumps(
            {
                "runtime_dir": str(selection.runtime_dir),
                "backend": selection.backend,
                "device": selection.device,
                "threads": selection.threads,
                "model_path": str(server.model_path),
                "projector_path": str(server.projector_path),
                "alias": server.alias,
                "state_dir": str(server.state_dir),
                "port": server.port,
                "context_size": server.context_size,
                "parallel": server.parallel,
                "gpu_layers": server.gpu_layers,
                "base_url": config.base_url,
                "selection": selection.to_dict(),
            },
            indent=2,
        )
    )
    return EXIT_OK


# ------------------------------------------------------------------ setup / init-config


def _progress_printer(name: str):
    def show(done: int, total: int, speed: float) -> None:
        sys.stdout.write(
            f"\r  {name}: {100 * done / total:5.1f}%  {speed / 1_000_000:6.1f} MB/s   "
        )
        sys.stdout.flush()

    return show


def _quiet_progress(done: int, total: int, speed: float) -> None:
    return None


def _setup_backends(requested: str | None, config: AppConfig | None) -> tuple[list[str], str]:
    """Backends whose runtimes setup installs, with the reason for the choice."""
    from .application import resources as res

    if requested is not None:
        names = [item.strip().lower() for item in requested.split(",") if item.strip()]
        unknown = sorted(set(names) - set(res.BACKENDS))
        if unknown or not names:
            raise ValueError(
                "--backends must be a comma-separated list of cuda, vulkan, cpu"
                + (f" (unknown: {', '.join(unknown)})" if unknown else "")
            )
        return [name for name in res.BACKENDS if name in names], "from --backends"
    server = config.server if config is not None else None
    if server is not None and server.backend != "auto":
        return [server.backend], f'from [server].backend = "{server.backend}"'
    candidates = list(server.fallback) if server is not None else list(res.BACKENDS)
    chosen = [name for name in res.BACKENDS if name in candidates and name != "cuda"]
    if "cuda" in candidates:
        devices = importlib.import_module(".adapters.devices", "audio_transcript")
        if devices.nvidia_gpu_present():
            chosen.insert(0, "cuda")
            return chosen, "automatic: NVIDIA GPU detected"
        return chosen, "automatic: no NVIDIA GPU detected, so CUDA is not installed"
    return chosen, "from [server].fallback"


def _setup_command(args) -> int:
    from .application import resources as res

    as_json = args.json
    say = (lambda *_a, **_k: None) if as_json else print
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
        return _fail(as_json, "config_invalid", str(exc))
    try:
        backends, backend_note = _setup_backends(args.backends, config)
    except ValueError as exc:
        return _fail(as_json, "usage", str(exc))
    say(f"Profile {profile.name}: {profile.description}")
    say(f"Runtime backends: {', '.join(backends) or 'none'} ({backend_note})")
    items = res.build_plan(manifest, profile, store, hasher=download.sha256_file, backends=backends)
    plan_lines = list(res.format_plan(items, store, res.free_space(store)))
    for line in plan_lines:
        say(line)
    summary = {
        "command": "setup",
        "profile": profile.name,
        "store": str(store),
        "dry_run": bool(args.dry_run),
        "backends": backends,
        "plan": plan_lines,
        "downloaded": [],
        "installed_runtimes": [],
        "failed": [],
    }
    conflicts = [item for item in items if item.status == "conflict"]
    for item in conflicts:
        if not as_json:
            print(
                f"ERROR: {item.path} exists but does not match the pinned file; it is never "
                "overwritten. Move it away and run setup again.",
                file=sys.stderr,
            )
    conflict_message = "; ".join(
        f"{item.path} exists but does not match the pinned file; it is never overwritten"
        for item in conflicts
    )
    todo = [item for item in items if item.status in {"download", "resume", "extract"}]
    if any(item.status == "installer" for item in items):
        runtime_dir = config.server.runtimes.get("cuda") if config and config.server else None
        say(
            "\nInstall the CUDA llama.cpp runtime (PowerShell installer; Vulkan and CPU "
            f"runtimes are installed by this command) with:\n  "
            f"{res.runtime_command(store, runtime_dir)}"
        )
    if args.dry_run:
        say("\nDry run: nothing was downloaded.")
        if conflicts:
            if as_json:
                return _fail(True, "resource_conflict", conflict_message, extra=summary)
            return EXIT_FAILED
        if as_json:
            _print_json({"ok": True, **summary})
        return EXIT_OK
    if conflicts:
        if as_json:
            return _fail(True, "resource_conflict", conflict_message, extra=summary)
        return EXIT_FAILED
    if not todo:
        say("\nNothing to download.")
        if as_json:
            _print_json({"ok": True, **summary})
        return EXIT_OK
    total = sum(item.download_bytes for item in todo)
    if total > res.free_space(store):
        return _fail(
            as_json,
            "insufficient_space",
            "not enough free disk space at the store.",
            extra=summary if as_json else None,
        )
    if not args.yes:
        if as_json:
            return _fail(
                True, "usage", "--json cannot ask for confirmation; pass --yes or --dry-run"
            )
        try:
            answer = input(f"\nDownload {total / 1_000_000:,.1f} MB into {store}? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled; nothing was downloaded.")
            return EXIT_FAILED
    failures = 0
    for item in todo:
        try:
            if item.status == "extract":
                folder = res.install_runtime(
                    item.resource,
                    store,
                    fetch=download.download_verified,
                    hasher=download.sha256_file,
                    progress=_quiet_progress if as_json else _progress_printer(item.resource.id),
                )
                summary["installed_runtimes"].append(str(folder))
                say(f"\n  verified, extracted and installed: {folder}")
                continue
            download.download_verified(
                item.resource.url,
                item.path,
                sha256=item.resource.sha256,
                size_bytes=item.resource.size_bytes,
                progress=_quiet_progress if as_json else _progress_printer(item.path.name),
            )
            res.write_provenance(item.path, item.resource)
            summary["downloaded"].append(str(item.path))
            say(f"\n  verified and installed: {item.path}")
        except (download.DownloadError, res.RuntimeInstallError, OSError) as exc:
            failures += 1
            summary["failed"].append(item.resource.id)
            if not as_json:
                print(f"\nERROR: {item.resource.id}: {exc}", file=sys.stderr)
    if as_json:
        if failures:
            return _fail(True, "download_failed", f"{failures} download(s) failed", extra=summary)
        _print_json({"ok": True, **summary})
        return EXIT_OK
    return EXIT_FAILED if failures else EXIT_OK


def _init_config_command(args) -> int:
    import tempfile

    from .application import resources as res

    as_json = args.json
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
            return _fail(
                as_json,
                "config_exists",
                f"{args.output} already exists; use --force to overwrite it",
            )
        with tempfile.TemporaryDirectory() as scratch:
            probe = Path(scratch) / "config.toml"
            probe.write_text(text, encoding="utf-8")
            load_config(probe)
    except ValueError as exc:
        return _fail(as_json, "config_invalid", str(exc))
    args.output.write_text(text, encoding="utf-8")
    if as_json:
        _print_json(
            {
                "ok": True,
                "command": "init-config",
                "output": str(args.output),
                "profile": profile.name,
                "store": str(args.store.resolve()),
            }
        )
        return EXIT_OK
    print(f"Wrote {args.output} for profile {profile.name} (store {args.store.resolve()})")
    print(f"Next: audio-transcript setup --config {args.output} --profile {profile.name}")
    return EXIT_OK


# ------------------------------------------------------------------ run support


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
        runtime=_runtime_info(config),
    )


def _attach(pipeline, session: RunSession) -> None:
    attach = getattr(pipeline, "attach", None)
    if callable(attach):
        attach(events=session.bus, status=session.status_out, runtime=session.runtime)


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


def _say(session: RunSession | None, text: str, as_json: bool = False) -> None:
    """Print a message without disturbing the console mode of the run."""
    if as_json:
        return
    if session is not None and session.ui_mode == "jsonl":
        print(text, file=sys.stderr)
    else:
        print(text)


class _BackendUnavailable(Exception):
    """The backend could not even be constructed (missing runtime, bad model path)."""


def _classify(exc: BaseException) -> str:
    """Stable error code for an unexpected exception."""
    if isinstance(exc, RuntimeError) and "already running" in str(exc):
        return "lock_held"
    trace = exc.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code.co_name in {"check", "_make_backend"}:
            return "backend_unavailable"
        trace = trace.tb_next
    return "runtime_error"


def _job_summary(result: dict, config: AppConfig) -> dict:
    job = {key: result.get(key) for key in ("job_id", "source", "status")}
    if result.get("error"):
        job["error"] = result["error"]
    job_id = result.get("job_id")
    if isinstance(job_id, str) and result.get("status") in {"completed", "skipped"}:
        transcript = config.output_dir / job_id / "transcript.txt"
        if transcript.is_file():
            job["transcript"] = transcript.as_posix()
    return job


def _emit_result(command: str, status: int, config: AppConfig, results: list[dict] | None) -> None:
    """Final JSON object of a streaming command (JSON mode)."""
    payload: dict = {
        "type": "result",
        "schema_version": clidoc.SCHEMA_VERSION,
        "ok": status == EXIT_OK,
        "command": command,
        "exit_status": status,
    }
    if results is not None:
        payload["jobs"] = [_job_summary(result, config) for result in results]
        if status != EXIT_OK:
            failed = any(result.get("status") == "failed" for result in results)
            payload["error"] = {
                "code": "job_failed" if failed else "job_incomplete",
                "message": "at least one job "
                + ("failed" if failed else "ended incomplete; re-run to retry the gaps"),
                "exit_status": status,
            }
    _print_json(payload)


# ------------------------------------------------------------------ events


def _events_command(args) -> int:
    from .application.eventlog import events_path, follow_events

    as_json = args.json
    try:
        config = load_config(args.config, {"process_dir": args.process_dir})
    except ValueError as exc:
        return _fail(as_json, "config_invalid", str(exc))
    path = events_path(config.process_dir, args.run)
    if path is None:
        return _fail(
            as_json,
            "run_not_found",
            f"no run found for '{args.run}' in {config.process_dir / 'runs'}",
        )
    try:
        code = follow_events(
            path,
            lambda line: print(line, flush=True),
            type_prefix=args.type,
            follow=args.follow,
            poll_interval=max(0.05, args.poll_interval),
        )
    except KeyboardInterrupt:
        return EXIT_OK
    except BrokenPipeError:
        return EXIT_OK
    if code:
        return _fail(as_json, "events_missing", f"no events file at {path}")
    return code


# ------------------------------------------------------------------ run / watch / wizard


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

    as_json = args.json
    scripted = None
    if args.answers is not None:
        try:
            scripted = json.loads(args.answers.read_text(encoding="utf-8"))
            if not isinstance(scripted, dict):
                raise ValueError("the answers file must contain a JSON object")
        except (OSError, ValueError) as exc:
            return _fail(
                as_json,
                "answers_invalid",
                f"Could not read the answers file {args.answers}: {exc}",
            )
    elif as_json:
        return _fail(
            True,
            "interactive_required",
            "with --json the wizard needs --answers <json file> (it cannot ask questions)",
        )
    elif not (sys.stdin and sys.stdin.isatty()):
        return _fail(
            False,
            "interactive_required",
            "the wizard needs an interactive console. Use "
            "'audio-transcript run' for unattended runs, or pass --answers <json file>.",
        )
    lock_path = config.process_dir / ".audio-transcript.lock"
    options = {}
    if as_json:
        options["say"] = lambda text="": print(text, file=sys.stderr)
    wizard = Wizard(
        config,
        _make_pipeline,
        lock_factory=lambda: ProcessLock(lock_path),
        open_folder=_open_folder if scripted is None and sys.platform == "win32" else None,
        scripted=scripted,
        sources=[path.expanduser() for path in args.file] if args.file else None,
        session_factory=lambda queue_size: _open_session(config, "wizard", queue_size),
        after_session=_after_session,
        **options,
    )
    try:
        status = wizard.run()
    except KeyboardInterrupt:
        if as_json:
            return _fail(True, "interrupted", "interrupted; job state remains recoverable")
        print("\nInterrupted; job state remains recoverable.", file=sys.stderr)
        return EXIT_INTERRUPTED
    if as_json:
        _emit_result("wizard", status, config, None)
    return status


def _help_command(parser: argparse.ArgumentParser, args) -> int:
    if args.json:
        _write_stdout(climan.json_text(args.topic))
    elif args.topic:
        _subparser(parser, args.topic).print_help()
    else:
        parser.print_help()
    return EXIT_OK


def _man_command(args) -> int:
    fmt = "json" if args.json else args.format
    _write_stdout(climan.render(fmt, args.topic))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["run"]
    if raw[0] not in {*COMMAND_NAMES, "-h", "--help", "--version"}:
        raw.insert(0, "run")
    _Parser.json_errors = "--json" in raw
    parser = _parser()
    args = parser.parse_args(raw)
    command = args.command or "run"
    as_json = bool(getattr(args, "json", False))
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
    if command == "help":
        return _help_command(parser, args)
    if command == "man":
        return _man_command(args)
    if command == "repl":
        from .repl import run_repl

        return run_repl(args.config, as_json)
    try:
        if args.prompt_file is not None:
            try:
                prompt_text = args.prompt_file.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                return _fail(
                    as_json,
                    "usage",
                    f"Could not read UTF-8 prompt file {args.prompt_file}: {exc}",
                )
        else:
            prompt_text = args.prompt
        config_values = {
            key: value
            for key, value in vars(args).items()
            if key not in {"command", "config", "file", "limit", "prompt_file", "answers", "json"}
        }
        config_values["prompt"] = prompt_text
        if as_json and command in _RUN_COMMANDS:
            config_values["ui"] = "jsonl"
        config = load_config(args.config, config_values)
    except ValueError as exc:
        return _fail(as_json, "config_invalid", str(exc))

    if command == "doctor":
        return _doctor(config, as_json)
    if command == "wizard":
        return _wizard_command(args, config)
    try:
        processor = _make_processor(config)
        try:
            backend = None if config.prepare_only else _make_backend(config)
        except Exception as exc:
            raise _BackendUnavailable(f"{type(exc).__name__}: {exc}") from exc
        exporter = None if config.prepare_only else _make_exporter()
        pipeline = TranscriptionPipeline(config, processor, backend, exporter)
        lock_path = config.process_dir / ".audio-transcript.lock"
        with ProcessLock(lock_path):
            if command == "watch":
                if args.file or args.limit is not None:
                    return _fail(
                        as_json, "usage", "--file and --limit are only supported by the run command"
                    )
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
                _say(session, "Watch stopped", as_json)
                if as_json:
                    _emit_result("watch", EXIT_OK, config, None)
                return EXIT_OK
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
                    as_json,
                )
        status = exit_status_of(results)
        if as_json:
            _emit_result("run", status, config, results)
        return status
    except _BackendUnavailable as exc:
        return _fail(as_json, "backend_unavailable", str(exc))
    except Exception as exc:
        return _fail(as_json, _classify(exc), f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        return _fail(as_json, "interrupted", "interrupted; job state remains recoverable")

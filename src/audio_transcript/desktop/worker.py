"""Isolated desktop worker; requests arrive on stdin, JSON events leave on stdout."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from .. import cli
from ..application.pipeline import ProcessLock, TranscriptionPipeline
from ..application.session import RunSession, exit_status_of
from ..config import load_config
from .runtime import managed_runtime


def emit(payload):
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def main() -> int:
    try:
        request = json.load(sys.stdin)
        config = load_config(Path(request["config"]), request["overrides"])
        # load_config treats None overrides as absent; a full GUI selection must
        # explicitly clear a duration bound from the base configuration.
        config = replace(config, max_duration=request["overrides"]["max_duration"])
        if "prompt" in request["overrides"]:
            config = replace(config, prompt=request["overrides"]["prompt"])
        source = Path(request["source"])
        with ProcessLock(config.process_dir / ".audio-transcript.lock"):
            emit(
                {
                    "type": "desktop.stage",
                    "data": {"message": "Checking tools and starting the model…"},
                }
            )
            processor = cli._make_processor(config)
            processor.check()
            with managed_runtime(config) as runtime:
                checks = cli._doctor_checks(config)
                failed = [check["detail"] for check in checks if not check["ok"]]
                if failed:
                    raise RuntimeError("; ".join(failed))
                backend = cli._make_backend(config)
                pipeline = TranscriptionPipeline(config, processor, backend, cli._make_exporter())
                with RunSession(
                    config,
                    command="desktop",
                    ui_mode="jsonl",
                    queue_size=1,
                    runtime=runtime,
                    monitor_factory=cli._monitor_factory(config),
                ) as session:
                    pipeline.attach(events=session.bus, status=session.status_out, runtime=runtime)
                    if runtime and runtime.get("fallback_used"):
                        session.bus.emit(
                            "warning", {"message": "GPU backend fallback: " + runtime["backend"]}
                        )
                    results = pipeline.run([source])
                    code = exit_status_of(results)
                    session.finish(code, results)
                cli._emit_result("desktop", code, config, results)
                return code
    except Exception as exc:
        emit(
            {
                "type": "result",
                "ok": False,
                "exit_status": 1,
                "error": {"message": f"{type(exc).__name__}: {exc}"},
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

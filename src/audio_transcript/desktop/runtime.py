"""Desktop model lifecycle. Windows reuses the established ownership-safe launcher."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .. import cli
from ..adapters.desktop_media import hidden_process
from ..application.backend_select import find_executable


def verify_placement(text: str, selection) -> dict:
    """Require model buffers and the audio encoder on the chosen device, not just health."""
    device = "CPU" if selection.backend == "cpu" else selection.device
    model = re.search(rf"\b{re.escape(device)}(?:_Mapped)?\s+model buffer", text)
    encoder = re.search(rf"CLIP using {re.escape(device)} backend", text)
    if not model or not encoder:
        raise RuntimeError(
            f"Cannot verify both model and audio encoder placement on {device}. "
            "Inspect the local llama.cpp startup logs; no transcription was started."
        )
    return {**selection.run_info(), "model_device": device, "audio_encoder_device": device}


@contextmanager
def managed_runtime(config):
    if config.backend != "llamacpp" or config.server is None or config.prepare_only:
        yield None
        return
    server = config.server
    endpoint = urlsplit(config.base_url)
    if (endpoint.port or (443 if endpoint.scheme == "https" else 80)) != server.port:
        raise ValueError("base_url and the configured model server must use the same port")
    selection = cli._select_backend(config)
    if selection.warning:
        print(json.dumps({"type": "warning", "data": {"message": selection.warning}}), flush=True)
    launcher = Path(__file__).resolve().parents[3] / "scripts" / "start-llamacpp.ps1"
    if os.name == "nt" and launcher.is_file():
        args = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-ModelPath",
            str(server.model_path),
            "-ProjectorPath",
            str(server.projector_path),
            "-RuntimeDirectory",
            str(selection.runtime_dir),
            "-Alias",
            server.alias,
            "-StateDirectory",
            str(server.state_dir),
            "-Port",
            str(server.port),
            "-ContextSize",
            str(server.context_size),
            "-Parallel",
            str(server.parallel),
            "-Backend",
            selection.backend,
            "-Device",
            selection.device,
        ]
        if selection.threads:
            args.extend(["-Threads", str(selection.threads)])
        if server.gpu_layers:
            args.extend(["-GpuLayers", str(server.gpu_layers)])
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=660, **hidden_process()
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout)[-4000:])
        # The existing service is shared with the CLI; never stop it on GUI exit.
        manifest = json.loads((server.state_dir / "server.json").read_text(encoding="utf-8-sig"))
        text = "\n".join(
            Path(manifest[key]).read_text(encoding="utf-8", errors="replace")
            for key in ("stdoutPath", "stderrPath")
        )
        yield verify_placement(text, selection)
        return
    # Installed packages and other systems use a child owned only by this worker.
    # Refuse occupied ports rather than adopting or terminating unknown services.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", server.port))
        except OSError as exc:
            raise RuntimeError(
                "Model port is occupied. Stop the existing server or choose another port."
            ) from exc
    executable = find_executable(selection.runtime_dir)
    if executable is None:
        raise RuntimeError("The configured llama-server executable is missing")
    for path in (server.model_path, server.projector_path):
        if not path.is_file():
            raise RuntimeError("A configured model file is missing; run setup explicitly")
    logs = config.process_dir / "desktop-runtime"
    logs.mkdir(parents=True, exist_ok=True)
    args = [
        str(executable),
        "-m",
        str(server.model_path),
        "--mmproj",
        str(server.projector_path),
        "--device",
        selection.device,
        "-ngl",
        str(0 if selection.backend == "cpu" else (server.gpu_layers or 999)),
        "-c",
        str(server.context_size),
        "--host",
        "127.0.0.1",
        "--port",
        str(server.port),
        "-np",
        str(server.parallel),
        "--alias",
        server.alias,
        "--timeout",
        "600",
        "-lv",
        "4",
    ]
    if selection.backend == "cpu":
        args.extend(["--no-mmproj-offload", "-t", str(selection.threads or 1)])
    with (logs / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args, cwd=selection.runtime_dir, stdout=log, stderr=log, **hidden_process()
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        "The model server exited during startup; see process/desktop-runtime/server.log"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{server.port}/health", timeout=2
                    ) as response:
                        ready = json.load(response).get("status") == "ok"
                except (OSError, urllib.error.URLError, ValueError):
                    ready = False
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Model startup timed out; inspect the local runtime log")
                time.sleep(0.25)
            text = (logs / "server.log").read_text(encoding="utf-8", errors="replace")
            yield verify_placement(text, selection)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

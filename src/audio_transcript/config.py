"""Configuration loading for the command line application."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ResourcesConfig:
    """Root folder for downloaded resources (models, VAD, runtimes)."""

    store: Path


@dataclass(frozen=True)
class ServerConfig:
    """Resolved llama.cpp server profile; every path is absolute."""

    runtime_dir: Path
    model_path: Path
    projector_path: Path
    alias: str
    state_dir: Path
    port: int
    context_size: int
    parallel: int
    gpu_layers: int | None = None


@dataclass(frozen=True)
class AppConfig:
    input_dir: Path = Path("input")
    processed_dir: Path = Path("processed")
    process_dir: Path = Path("process")
    output_dir: Path = Path("output")
    backend: str = "nexa"
    model: str = "qwen2audio"
    base_url: str = "http://127.0.0.1:8088"
    timeout: float = 180.0
    max_tokens: int = 1024
    prompt: str | None = None
    temperature: float = 0.0
    seed: int = 42
    response_mode: str = "json"
    model_path: Path | None = None
    projector_path: Path | None = None
    device: str = "auto"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    language: str | None = None
    chunk_seconds: float = 15.0
    sample_rate: int = 16000
    max_duration: float | None = None
    max_file_size: int | None = None
    retries: int = 2
    force: bool = False
    prepare_only: bool = False
    archive_inputs: bool = True
    watch_interval: float = 5.0
    stable_scans: int = 2
    vad: str = "silero"
    vad_model_path: Path = Path("models/silero_vad.onnx")
    vad_threshold: float = 0.5
    vad_min_speech_seconds: float = 0.25
    vad_min_silence_seconds: float = 0.1
    vad_speech_pad_seconds: float = 0.2
    vad_max_merge_gap_seconds: float = 1.0
    vad_energy_margin_db: float = 15.0
    fallback_temperatures: tuple[float, ...] = (0.2, 0.4)
    split_on_failure: bool = True
    context_free_fallback: bool = True
    min_tokens: int = 48
    tokens_per_second: float = 8.0
    compression_ratio_threshold: float = 2.4
    repeat_penalty: float = 1.0
    dry_multiplier: float = 0.0
    force_language: bool = True
    collect_logprobs: bool = True
    parallel_requests: int = 1
    intermediate_interval_seconds: float = 10.0
    # Presentation and monitoring only: none of these is part of job identity.
    ui: str = "auto"
    monitor: bool = True
    monitor_interval: float = 1.0
    resources: ResourcesConfig | None = None
    server: ServerConfig | None = None


_EXTRA_TABLES = ("resources", "server")
_SERVER_KEYS = (
    "runtime_dir",
    "model_path",
    "projector_path",
    "alias",
    "state_dir",
    "port",
    "context_size",
    "parallel",
    "gpu_layers",
)
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _parse_resources(table: object) -> ResourcesConfig:
    if not isinstance(table, dict):
        raise ValueError("[resources] must be a TOML table")
    unknown = sorted(set(table) - {"store"})
    if unknown:
        raise ValueError("Unknown [resources] option(s): " + ", ".join(unknown))
    store = table.get("store")
    if not isinstance(store, str) or not store.strip():
        raise ValueError(
            '[resources].store must be a non-empty path string, for example store = "D:/ai-models"'
        )
    return ResourcesConfig(store=Path(store).expanduser().resolve())


def _parse_server(table: object, resources: ResourcesConfig | None) -> ServerConfig:
    if not isinstance(table, dict):
        raise ValueError("[server] must be a TOML table")
    unknown = sorted(set(table) - set(_SERVER_KEYS))
    if unknown:
        raise ValueError("Unknown [server] option(s): " + ", ".join(unknown))
    missing = [key for key in _SERVER_KEYS if key != "gpu_layers" and key not in table]
    if missing:
        raise ValueError("[server] is missing required option(s): " + ", ".join(missing))
    paths: dict[str, Path] = {}
    for key in ("runtime_dir", "model_path", "projector_path", "state_dir"):
        raw = table[key]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"[server].{key} must be a non-empty path string")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            if resources is None:
                raise ValueError(
                    f"[server].{key} is relative ({raw}); add a [resources] table with a store "
                    "folder or use an absolute path"
                )
            path = resources.store / path
        paths[key] = path.resolve()
    alias = table["alias"]
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("[server].alias must be a non-empty string")
    numbers: dict[str, int] = {}
    limits = {
        "port": (1, 65535),
        "context_size": (1, 1048576),
        "parallel": (1, 8),
        "gpu_layers": (1, 999),
    }
    for key, (low, high) in limits.items():
        if key not in table:
            continue
        number = table[key]
        if isinstance(number, bool) or not isinstance(number, int) or not low <= number <= high:
            raise ValueError(f"[server].{key} must be an integer between {low} and {high}")
        numbers[key] = number
    return ServerConfig(
        runtime_dir=paths["runtime_dir"],
        model_path=paths["model_path"],
        projector_path=paths["projector_path"],
        alias=alias,
        state_dir=paths["state_dir"],
        port=numbers["port"],
        context_size=numbers["context_size"],
        parallel=numbers["parallel"],
        gpu_layers=numbers.get("gpu_layers"),
    )


def _validate_server(cfg: AppConfig, server: ServerConfig) -> None:
    if cfg.backend != "llamacpp":
        raise ValueError('[server] requires backend = "llamacpp" in [audio_transcript]')
    if server.alias != cfg.model:
        raise ValueError(
            f"[server].alias ({server.alias}) must equal [audio_transcript].model ({cfg.model})"
        )
    url = urlsplit(cfg.base_url)
    if url.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            "Server management starts a local server; base_url must point to 127.0.0.1 or "
            "localhost (remove the [server] table to use a remote server)"
        )
    url_port = url.port or (443 if url.scheme == "https" else 80)
    if server.port != url_port:
        raise ValueError(
            f"[server].port ({server.port}) must match the port of base_url ({url_port})"
        )
    if server.parallel < cfg.parallel_requests:
        raise ValueError(
            f"[server].parallel ({server.parallel}) must be at least "
            f"parallel_requests ({cfg.parallel_requests})"
        )


def load_config(path: Path | None = None, overrides: dict | None = None) -> AppConfig:
    """Load optional TOML settings, then apply explicitly supplied CLI values."""
    values: dict = {}
    resources: ResourcesConfig | None = None
    tables: dict = {}
    if path is not None:
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except FileNotFoundError as exc:
            raise ValueError(f"Configuration file not found: {path}") from exc
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise ValueError(f"Could not read configuration file {path}: {exc}") from exc
        tables = {name: raw[name] for name in _EXTRA_TABLES if name in raw}
        if "audio_transcript" in raw:
            section = raw["audio_transcript"]
            unknown_tables = sorted(set(raw) - {"audio_transcript", *_EXTRA_TABLES})
            if unknown_tables:
                raise ValueError("Unknown configuration table(s): " + ", ".join(unknown_tables))
        else:
            section = {k: v for k, v in raw.items() if k not in _EXTRA_TABLES}
        if not isinstance(section, dict):
            raise ValueError("The [audio_transcript] configuration must be a TOML table")
        misplaced = sorted(set(section) & set(_EXTRA_TABLES))
        if misplaced:
            raise ValueError(
                "Unknown configuration option(s): "
                + ", ".join(misplaced)
                + " (use top-level [resources] and [server] tables)"
            )
        values.update(section)
        if "resources" in tables:
            resources = _parse_resources(tables["resources"])
        if (
            resources is not None
            and isinstance(values.get("vad_model_path"), str)
            and not Path(values["vad_model_path"]).is_absolute()
        ):
            values["vad_model_path"] = resources.store / values["vad_model_path"]
    values.update({key: val for key, val in (overrides or {}).items() if val is not None})
    allowed = set(AppConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("Unknown configuration option(s): " + ", ".join(unknown))
    for key in ("input_dir", "processed_dir", "process_dir", "output_dir"):
        if key in values:
            if not isinstance(values[key], (str, Path)):
                raise ValueError(f"{key} must be a path or string")
            values[key] = Path(values[key])
    for key in ("model_path", "projector_path", "vad_model_path"):
        if key in values and values[key] is not None:
            if not isinstance(values[key], (str, Path)):
                raise ValueError(f"{key} must be a path or string")
            values[key] = Path(values[key])
    if "vad_model_path" in values and values["vad_model_path"] is None:
        del values["vad_model_path"]
    for key in (
        "backend",
        "model",
        "base_url",
        "device",
        "ffmpeg",
        "ffprobe",
        "response_mode",
        "vad",
        "ui",
    ):
        if key in values and not isinstance(values[key], str):
            raise ValueError(f"{key} must be a string")
    if (
        "language" in values
        and values["language"] is not None
        and not isinstance(values["language"], str)
    ):
        raise ValueError("language must be a string or null")
    if isinstance(values.get("language"), str) and values["language"].strip().casefold() == "auto":
        # "auto" means no language hint (auto-detect); it is recorded as None.
        values["language"] = None
    if (
        "prompt" in values
        and values["prompt"] is not None
        and not isinstance(values["prompt"], str)
    ):
        raise ValueError("prompt must be a string or null")
    for key in (
        "chunk_seconds",
        "max_duration",
        "watch_interval",
        "timeout",
        "temperature",
        "vad_threshold",
        "vad_min_speech_seconds",
        "vad_min_silence_seconds",
        "vad_speech_pad_seconds",
        "vad_max_merge_gap_seconds",
        "vad_energy_margin_db",
        "tokens_per_second",
        "compression_ratio_threshold",
        "repeat_penalty",
        "dry_multiplier",
        "intermediate_interval_seconds",
        "monitor_interval",
    ):
        if key in values and values[key] is not None:
            if isinstance(values[key], bool) or not isinstance(values[key], (int, float)):
                raise ValueError(f"{key} must be a number")
            values[key] = float(values[key])
    for key in (
        "sample_rate",
        "max_file_size",
        "retries",
        "stable_scans",
        "max_tokens",
        "seed",
        "min_tokens",
        "parallel_requests",
    ):
        if key in values and values[key] is not None:
            if isinstance(values[key], bool) or not isinstance(values[key], int):
                raise ValueError(f"{key} must be an integer")
    for key in (
        "force",
        "prepare_only",
        "archive_inputs",
        "split_on_failure",
        "context_free_fallback",
        "force_language",
        "collect_logprobs",
        "monitor",
    ):
        if key in values and not isinstance(values[key], bool):
            raise ValueError(f"{key} must be true or false")
    if "fallback_temperatures" in values:
        raw_temperatures = values["fallback_temperatures"]
        if not isinstance(raw_temperatures, (list, tuple)):
            raise ValueError("fallback_temperatures must be a list of numbers")
        parsed_temperatures = []
        for item in raw_temperatures:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("fallback_temperatures must be a list of numbers")
            if not math.isfinite(item) or not 0 <= item <= 2:
                raise ValueError("each fallback temperature must be finite and between 0 and 2")
            parsed_temperatures.append(float(item))
        values["fallback_temperatures"] = tuple(parsed_temperatures)
    if resources is not None:
        values["resources"] = resources
    cfg = AppConfig(**values)
    if cfg.backend not in {"nexa", "mock", "llamacpp"}:
        raise ValueError("backend must be 'nexa', 'mock', or 'llamacpp'")
    if cfg.backend == "mock" and "model" not in values:
        cfg = AppConfig(**{**cfg.__dict__, "model": "mock"})
    if cfg.backend == "llamacpp" and "model" not in values:
        cfg = AppConfig(**{**cfg.__dict__, "model": "qwen2-audio-7b"})
    try:
        parsed_url = urlsplit(cfg.base_url)
        valid_url = (
            parsed_url.scheme in {"http", "https"}
            and parsed_url.hostname is not None
            and parsed_url.port != 0
            and not parsed_url.username
            and not parsed_url.password
            and not parsed_url.query
            and not parsed_url.fragment
        )
    except ValueError:
        valid_url = False
    if not valid_url:
        raise ValueError(
            "base_url must be a valid HTTP or HTTPS URL without credentials, query, or fragment"
        )
    if not math.isfinite(cfg.timeout) or cfg.timeout <= 0 or cfg.max_tokens <= 0:
        raise ValueError("timeout and max_tokens must be positive")
    if not math.isfinite(cfg.temperature) or not 0 <= cfg.temperature <= 2:
        raise ValueError("temperature must be finite and between 0 and 2")
    if not 0 <= cfg.seed <= 4294967295:
        raise ValueError("seed must be between 0 and 4294967295")
    if cfg.response_mode not in {"json", "plain", "qwen3-asr"}:
        raise ValueError("response_mode must be 'json', 'plain', or 'qwen3-asr'")
    if cfg.ui not in {"auto", "live", "plain", "jsonl"}:
        raise ValueError("ui must be 'auto', 'live', 'plain', or 'jsonl'")
    if not math.isfinite(cfg.monitor_interval) or not 0.1 <= cfg.monitor_interval <= 3600:
        raise ValueError("monitor_interval must be between 0.1 and 3600 seconds")
    if cfg.vad not in {"silero", "energy", "none"}:
        raise ValueError("vad must be 'silero', 'energy', or 'none'")
    if not math.isfinite(cfg.vad_threshold) or not 0 < cfg.vad_threshold < 1:
        raise ValueError("vad_threshold must be greater than 0 and less than 1")
    for name in (
        "vad_min_speech_seconds",
        "vad_min_silence_seconds",
        "vad_speech_pad_seconds",
        "vad_max_merge_gap_seconds",
        "vad_energy_margin_db",
    ):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 1 <= cfg.parallel_requests <= 8:
        raise ValueError("parallel_requests must be an integer between 1 and 8")
    if cfg.parallel_requests > 1 and cfg.backend == "nexa":
        raise ValueError("parallel_requests above 1 requires the llamacpp backend")
    if (
        not math.isfinite(cfg.intermediate_interval_seconds)
        or cfg.intermediate_interval_seconds < 0
    ):
        raise ValueError("intermediate_interval_seconds must be finite and nonnegative")
    if cfg.min_tokens <= 0 or isinstance(cfg.min_tokens, bool):
        raise ValueError("min_tokens must be a positive integer")
    if not math.isfinite(cfg.tokens_per_second) or cfg.tokens_per_second < 0:
        raise ValueError("tokens_per_second must be finite and nonnegative")
    if not math.isfinite(cfg.compression_ratio_threshold) or cfg.compression_ratio_threshold <= 0:
        raise ValueError("compression_ratio_threshold must be positive")
    if not math.isfinite(cfg.repeat_penalty) or cfg.repeat_penalty <= 0:
        raise ValueError("repeat_penalty must be positive (1.0 disables it)")
    if not math.isfinite(cfg.dry_multiplier) or cfg.dry_multiplier < 0:
        raise ValueError("dry_multiplier must be nonnegative (0 disables it)")
    if not math.isfinite(cfg.chunk_seconds) or cfg.chunk_seconds <= 0 or cfg.sample_rate <= 0:
        raise ValueError("chunk_seconds and sample_rate must be positive")
    if cfg.max_duration is not None and (
        not math.isfinite(cfg.max_duration) or cfg.max_duration <= 0
    ):
        raise ValueError("max_duration must be positive")
    if cfg.max_file_size is not None and cfg.max_file_size <= 0:
        raise ValueError("max_file_size must be positive")
    if (
        cfg.retries < 0
        or not math.isfinite(cfg.watch_interval)
        or cfg.watch_interval <= 0
        or cfg.stable_scans < 2
    ):
        raise ValueError(
            "retries must be nonnegative, watch_interval positive, and stable_scans at least 2"
        )
    roots = {
        name: getattr(cfg, name).resolve()
        for name in ("input_dir", "processed_dir", "process_dir", "output_dir")
    }
    if len(set(roots.values())) != len(roots):
        raise ValueError(
            "input_dir, processed_dir, process_dir, and output_dir must be distinct directories"
        )
    names = tuple(roots)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            first_path, second_path = roots[first], roots[second]
            try:
                first_path.relative_to(second_path)
                nested = True
            except ValueError:
                try:
                    second_path.relative_to(first_path)
                    nested = True
                except ValueError:
                    nested = False
            if nested:
                raise ValueError(f"{first} and {second} must not contain one another")
    if "server" in tables:
        server = _parse_server(tables["server"], resources)
        _validate_server(cfg, server)
        cfg = AppConfig(**{**cfg.__dict__, "server": server})
    return cfg

"""Configuration loading for the command line application."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


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
    chunk_seconds: float = 30.0
    sample_rate: int = 16000
    max_duration: float | None = None
    max_file_size: int | None = None
    retries: int = 2
    force: bool = False
    prepare_only: bool = False
    archive_inputs: bool = True
    watch_interval: float = 5.0
    stable_scans: int = 2


def load_config(path: Path | None = None, overrides: dict | None = None) -> AppConfig:
    """Load optional TOML settings, then apply explicitly supplied CLI values."""
    values: dict = {}
    if path is not None:
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except FileNotFoundError as exc:
            raise ValueError(f"Configuration file not found: {path}") from exc
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise ValueError(f"Could not read configuration file {path}: {exc}") from exc
        section = raw.get("audio_transcript", raw)
        if not isinstance(section, dict):
            raise ValueError("The [audio_transcript] configuration must be a TOML table")
        values.update(section)
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
    for key in ("model_path", "projector_path"):
        if key in values and values[key] is not None:
            if not isinstance(values[key], (str, Path)):
                raise ValueError(f"{key} must be a path or string")
            values[key] = Path(values[key])
    for key in ("backend", "model", "base_url", "device", "ffmpeg", "ffprobe", "response_mode"):
        if key in values and not isinstance(values[key], str):
            raise ValueError(f"{key} must be a string")
    if (
        "language" in values
        and values["language"] is not None
        and not isinstance(values["language"], str)
    ):
        raise ValueError("language must be a string or null")
    if (
        "prompt" in values
        and values["prompt"] is not None
        and not isinstance(values["prompt"], str)
    ):
        raise ValueError("prompt must be a string or null")
    for key in ("chunk_seconds", "max_duration", "watch_interval", "timeout", "temperature"):
        if key in values and values[key] is not None:
            if isinstance(values[key], bool) or not isinstance(values[key], (int, float)):
                raise ValueError(f"{key} must be a number")
            values[key] = float(values[key])
    for key in ("sample_rate", "max_file_size", "retries", "stable_scans", "max_tokens", "seed"):
        if key in values and values[key] is not None:
            if isinstance(values[key], bool) or not isinstance(values[key], int):
                raise ValueError(f"{key} must be an integer")
    for key in ("force", "prepare_only", "archive_inputs"):
        if key in values and not isinstance(values[key], bool):
            raise ValueError(f"{key} must be true or false")
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
    if cfg.response_mode not in {"json", "plain"}:
        raise ValueError("response_mode must be 'json' or 'plain'")
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
    return cfg

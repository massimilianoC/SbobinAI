# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project name **SbobinAI** (command `sbobinai`, alias `audio-transcript`).
- Pause-aware speech segmentation with Silero VAD (CPU), chunks of at most 15 s.
- Qwen3-ASR response mode for llama.cpp with forced language and optional context.
- Per-chunk recovery ladder (temperature fallback, split) and `incomplete` job status.
- Versioned per-source output layout, `source.json`, `catalog.json`, Markdown run reports,
  `catalog` and `migrate-layout` commands.
- Uncalibrated confidence score from token log-probabilities.
- Optional concurrent chunk requests (`parallel_requests`).
- One-command run (`transcribe.cmd`) that also manages the local llama.cpp server.
- Guided wizard (`transcribe.cmd`, `audio-transcript wizard`): spoken language or auto,
  optional context/glossary, scope, live progress and a results table; `transcribe-batch.cmd`
  for unattended runs; `language = "auto"`.
- Single local configuration with `[resources]` and `[server]` tables, pinned resource
  manifest (`resources.json`), `setup`, `init-config` and `server-profile` commands.
- AGPL-3.0-only license with a commercial licensing option.

### Fixed
- Long recordings no longer stop on a single looping chunk.
- Silence no longer produces invented sentences or prompt echoes.
- Fully processed files are archived despite millisecond rounding of the source duration.

## [0.1.0] - 2026-10-02

### Added
- Initial local pipeline: FFmpeg preparation, llama.cpp Qwen2-Audio backend,
  resumable jobs, exports (TXT, Markdown, SRT, VTT, JSON) and input archival.

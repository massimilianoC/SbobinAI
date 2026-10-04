# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Guided wizard in the system language: English, Italian, Spanish, French, German
  and Portuguese, detected from the Windows display language or `LC_ALL`/`LANG`,
  with English as fallback; `--ui-language` and `ui_language` choose one
  (`-UiLanguage` in `process-input.ps1`). Texts live in one JSON file per language
  (`src/audio_transcript/locales/wizard/<code>.json`); a test keeps keys and
  placeholders aligned with English. Yes/no and "full file" answers are accepted
  in any of these languages.
- Language names are accepted wherever a spoken language is asked: `it`,
  `Italian`, `italiano`, `it-IT` (also `--language italiano` for Qwen3-ASR).
- README **Quick start**: install by asking an AI coding agent (for example Claude Code)
  or by hand, then transcribe; the manual steps now include Git/Python prerequisites,
  `git clone` and a per-window execution-policy bypass. `AGENTS.md` gained rules for
  agents that install SbobinAI for a user.
- README **Support status** table: what is stable (Windows + NVIDIA CUDA, the
  reference setup), beta (Vulkan/CPU fallback, auto language, parallel requests,
  context, confidence, setup downloads), untested (Linux, macOS, AMD/Intel GPUs,
  bf16 profile, legacy Nexa backend) and planned.
- Automatic inference-backend fallback CUDA -> Vulkan -> CPU: `[server].backend`
  (`auto`, `cuda`, `vulkan`, `cpu`), `fallback`, `[server.runtimes]`, `device`, `threads`;
  `server-profile` and `doctor` report the selection and warn on fallback; the launcher
  accepts `-Backend`, `-Device`, `-Threads` and prints a prominent warning with the expected
  slowdown; the backend, device and fallback flag are recorded in `inference_configuration`,
  `report.json`, `run-report.md`, `source.json`, the `run.started` event and the live header.
  Not part of job identity (identical transcripts were measured on all backends).
- `setup` installs the pinned llama.cpp b11389 Vulkan and CPU runtimes (download, SHA-256,
  atomic extraction, `runtime-manifest.json`, `PROVENANCE.txt`); new `setup --backends`.
- Agent-native CLI (CLI-Anything conventions): full `--help` per command with
  examples and exit status, `sbobinai help`, `sbobinai man` (text, markdown, json,
  skill), `--json` on every command with structured error codes, `doctor --json`,
  `repl` shell, generated `docs/cli-reference.md` and `skills/sbobinai/SKILL.md`,
  `TEST.md`, alias `cli-anything-sbobinai`, `--version`.
- Live progress console, event stream (`events.jsonl`, `status.json`), resource
  monitor and `events --follow`.
- Multiplatform porting analysis with measured CUDA, Vulkan and CPU runs.
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
- Real-time monitoring: live console (`--ui auto|live|plain|jsonl`) with stage marks, chunk
  progress bar, ETA, counters and CPU/RAM/GPU/VRAM/power/temperature sparklines; a versioned
  event stream (`process/runs/<run-id>/events.jsonl`, `status.json`, `latest.json`, per-version
  `events.jsonl`), a pipeline-written `pipeline.log`, the `events [--follow]` command and
  `--monitor-interval` / `--no-monitor`. See `docs/events.md`.

### Changed
- The wizard no longer asks for a transcript language: the transcript is always in
  the spoken language and translation is out of scope (a separate step for
  applications built around SbobinAI).
- `scripts/process-input.ps1` no longer relies on `Start-Transcript` for pipeline output
  (PowerShell does not capture native-process output); it prints the paths of the run's
  `pipeline.log` and `events.jsonl` and accepts `-Ui` and `-NoMonitor`.

### Fixed
- The wizard rejected the spoken language itself (`it`, `italiano`) as the
  transcript language, and stored language names verbatim (`italian`), which
  created a separate version for the same language; answers are now normalized
  to the ISO code.
- Atomic state, export and audit writes retry brief Windows sharing locks
  instead of failing the job (`PermissionError` on `checkpoint.json`).
- `transcribe.cmd` / `transcribe-batch.cmd` no longer run twice when started
  from a command line containing `&&`.
- Server readiness polling no longer writes false "fatal error" lines to
  launcher transcripts while the model loads.
- CI: Silero VAD tests no longer re-import NumPy inside a patched `sys.modules`
  ("cannot load module more than once per process"); all matrix jobs run to completion
  and actions moved off the deprecated Node 20 runtime.
- Long recordings no longer stop on a single looping chunk.
- Silence no longer produces invented sentences or prompt echoes.
- Fully processed files are archived despite millisecond rounding of the source duration.

## [0.1.0] - 2026-10-02

### Added
- Initial local pipeline: FFmpeg preparation, llama.cpp Qwen2-Audio backend,
  resumable jobs, exports (TXT, Markdown, SRT, VTT, JSON) and input archival.

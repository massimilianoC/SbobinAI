# Audio Transcript

Local audio/video transcription through FFmpeg and a shared Qwen2-Audio
llama.cpp NVIDIA CUDA server. The Python core uses the standard library.
Source media bytes are preserved; real results export JSON, TXT, Markdown,
SRT, VTT and a report. Subtitle times are coarse chunk boundaries.

## Setup

Requires Python 3.11+, FFmpeg/FFprobe and an NVIDIA CUDA-capable runtime.
See [shared runtime setup](docs/runtime-setup.md) for installation, existing
GGUF reuse, projector conversion and shared server startup.

```powershell
.\scripts\install-ffmpeg.ps1
.\scripts\setup.ps1 -Dev
.\scripts\start-llamacpp.ps1 -ModelPath models\qwen2audio-q8.gguf -ProjectorPath models\llamacpp-projector.gguf
Copy-Item config.example.toml config.local.toml
.\scripts\run.ps1 doctor --config config.local.toml
```

The run wrapper uses `python -m audio_transcript` and refreshes PATH. No cloud
API key is required. CLI settings override TOML. Relative paths resolve from
the project root when using the Windows wrapper. Local configuration is ignored.

## Operational queue

Drop one or several files into `input`, then run:

```powershell
.\scripts\process-input.ps1
# Or watch for stable new files until Ctrl+C:
.\scripts\run.ps1 watch --config config.local.toml
```

Jobs run sequentially. Successful complete real jobs move their original into
`processed/<job-id>/<original-name>`. Metadata, WAV chunks, checkpoints and raw
responses live in `process/<job-id>/`; exports live in `output/<job-id>/`.
Failures, bounded runs, preparation-only jobs and mock jobs remain in input.
External `--file` sources stay in place. Archive conflicts never overwrite
different data and retain a recoverable warning. Original media bytes do not change.

Move an archived file or a chunk back into input and use `--force` to explicitly
reprocess it. Unchanged completed jobs otherwise skip model work. For prompt
comparisons, use `--no-archive-inputs`. Changed prompt/settings create a distinct
job identity. See [inference controls](docs/inference-controls.md).

```powershell
.\scripts\run.ps1 run --config config.local.toml --file input\sample.wav --force
.\scripts\run.ps1 run --config config.local.toml --max-duration 60 --no-archive-inputs
.\scripts\run.ps1 run --prepare-only --file input\sample.wav
```

## Select a meaningful test sample

```powershell
.\.venv\Scripts\python.exe scripts\sample-audio.py input\recording.mp4 --count 1 --seconds 180 --destination process\validation-long --seed 20261002
Copy-Item process\validation-long\central_sample_01.wav input\central_validation_180s.wav
.\scripts\run.ps1 run --config config.local.toml --file input\central_validation_180s.wav
```

The sampler chooses central 35–65% windows and rejects low-energy audio.
Its local report records offsets and levels. Energy establishes signal, not
speech accuracy. Samples use mono PCM16 WAV at 16 kHz. Chunk settings bound
each inference request; default 30 seconds, current local configuration 10.

Default Italian instructions and constrained JSON output avoid a prefatory
sentence on tested samples. Only the transcript field is exported. Raw model
responses and request settings remain available for review. Empty, malformed
or truncated responses fail. Speaker identities, exact alignment and rich
analysis remain future work. Mock text is explicitly synthetic and never
substitutes for inference.

## Development and documentation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe scripts\smoke-test.py
```

Read the [documentation index](docs/README.md), [specification](docs/specification.md),
[verification](docs/verification.md), [TODO](docs/TODO.md) and [AGENTS.md](AGENTS.md).
At session completion, revise affected documents using the
[source-preserving documentation policy](docs/documentation-policy.md).
Supplied originals are archived unchanged; public documentation stays in English.
Media, transcripts, local paths, environments, logs, models, vision and private
reviews are excluded from Git. No commit or publication is automatic.

Intermediate exports update after each chunk under `output/<job-id>/intermediate`.
Their report distinguishes processing/failure/completion. Output JSON records
the effective prompt and generation settings. The structured pipeline script
validates dependencies and stores an execution log under `.local/pipeline-runs`.

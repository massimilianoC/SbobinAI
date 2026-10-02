# SbobinAI

*From the Italian* sbobinare *— to transcribe a recorded lecture or meeting.*

Local, private transcription of long audio and video recordings on your own
NVIDIA GPU. Drop a file into a folder, run one command, get a clean transcript
(TXT, Markdown, SRT, VTT, JSON) plus a readable run report.

- **Accurate on long recordings.** Speech is detected with Silero VAD and cut at
  natural pauses into chunks of at most 15 seconds; silence never reaches the
  model, so it cannot invent text there.
- **Fast and light.** Default model: [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
  (Q8_0 GGUF) on a local [llama.cpp](https://github.com/ggml-org/llama.cpp) CUDA
  server. A 2-hour talk takes about 4–5 minutes and ~4 GB of VRAM on an RTX 5070 Ti.
- **Robust.** Looping or unusable model answers are retried with a fallback
  ladder; a chunk that still fails is reported as a gap instead of stopping the job.
  Interrupted jobs resume.
- **Versioned and traceable.** Each input gets its own folder; each run with
  different settings becomes a separate version with model, settings, timings,
  completion and an (uncalibrated) confidence score.
- **Private.** Everything runs locally. No cloud API, no account, no telemetry.

> **Status:** early release, tested on Windows 11 with an NVIDIA GPU (CUDA).
> Linux/macOS, an HTTP API and a web UI are planned — see the
> [distribution design](docs/distribution-design.md).

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation (Windows + NVIDIA)](#installation-windows--nvidia)
- [Transcribe a file](#transcribe-a-file)
- [Where the results are](#where-the-results-are)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## How it works

```text
input/<file>  ->  FFmpeg (16 kHz mono)  ->  Silero VAD (CPU): speech chunks <= 15 s
              ->  llama.cpp server (CUDA) + Qwen3-ASR, one or two chunks at a time
              ->  output/<source>/<version>/ transcript.* + report.json + run-report.md
              ->  input file archived to processed/<source>/
```

The Python application uses only the standard library at its core (VAD needs
`onnxruntime` and `numpy`). FFmpeg and llama.cpp run as separate programs.

## Requirements

| Component | Requirement |
| --- | --- |
| OS | Windows 10/11 x64 (Linux/macOS planned) |
| GPU | NVIDIA, 6 GB VRAM or more recommended, current driver (the CUDA 13.x runtime is downloaded) |
| Python | 3.11 or newer |
| Disk | ~4 GB for runtime and models, plus space for your recordings |
| Network | Only during setup, to download the runtime and model files |

Download sizes (Windows + NVIDIA, default profile): llama.cpp CUDA runtime
~575 MB, Qwen3-ASR-1.7B Q8_0 + audio projector ~2.8 GB, Silero VAD 2.3 MB,
FFmpeg ~100 MB. All files are pinned to exact versions and verified by SHA-256
(see [`resources.json`](resources.json)).

## Installation (Windows + NVIDIA)

Run these once in PowerShell from the repository folder. Replace `D:\ai-models`
with the folder where downloaded models should live (any drive).

```powershell
# 1. Python environment with the speech-detection extra
.\scripts\setup.ps1 -Dev
.\.venv\Scripts\python.exe -m pip install -e '.[vad]'

# 2. FFmpeg (via winget)
.\scripts\install-ffmpeg.ps1

# 3. Local configuration (config.local.toml, git-ignored) with all paths filled in
.\scripts\run.ps1 init-config --store D:\ai-models --profile qwen3-asr-1.7b-q8

# 4. Model files: shows the plan and total download size, asks before downloading,
#    verifies SHA-256, and prints the command for the llama.cpp CUDA runtime
.\scripts\run.ps1 setup --config config.local.toml
.\scripts\install-llamacpp-cuda.ps1 -RuntimeDirectory D:\ai-models\runtimes\llama.cpp-cuda

# 5. Check everything
.\scripts\run.ps1 doctor --config config.local.toml
```

`config.local.toml` is the single place that connects the application to your
machine: queue folders and transcription settings in `[audio_transcript]`, the
resource folder in `[resources]`, and the llama.cpp server (runtime, model
files, alias, port, slots) in `[server]`. Relative paths resolve against the
resource folder. [`resources.json`](resources.json) lists every downloadable
file with its pinned URL, size, SHA-256 and license.

## Transcribe a file

1. Copy one or more audio/video files into the `input` folder
   (`mp4`, `mkv`, `mov`, `mp3`, `wav`, `m4a`, `flac`, `ogg`, `webm`, …).
2. Double-click **`transcribe.cmd`** (or run `.\transcribe.cmd` in a terminal).
3. Answer the questions (press Enter to keep the value in brackets):

   | Question | Answer |
   | --- | --- |
   | Spoken language | `auto` to detect it, or an ISO code such as `it`, `en`, `de` |
   | Transcript language | Only "same as spoken" for now — translation is not supported by the speech model |
   | Context | Optional: topic, names, product names or jargon (one line, or `@path\to\glossary.txt`). It helps the model spell them correctly |
   | Scope | Whole file, or only the first N minutes for a quick check |

   The wizard shows the queue with durations and an estimated time, a summary
   to confirm, then the progress and a results table (status, version,
   chunks, confidence, total time, paths to `transcript.txt` and
   `run-report.md`), and can open the output folder.

The script starts the model server if needed (or restarts it when the
configuration changed), checks dependencies, transcribes every queued file and
prints one line per chunk:

```text
Prepared talk.mp4: 7546.0s analysed, detector=silero, speech 6873.8s (91%), 581 chunks, ...
Chunk 58/581 [574.6-588.9s, 14.3s audio] t=0 ok 0.3s 48 tok
Execution: chunks ok=581 no_speech=0 failed=0, inference 203.9s, concurrency 1.93x, ...
Timings: total 4 min 25 s | preparation 1 min 01 s | inference 3 min 24 s | export 48 ms
```

Useful variants:

```powershell
.\transcribe-batch.cmd                                             # unattended: use config.local.toml settings, no questions
.\transcribe-batch.cmd -MaxDuration 600 -NoArchive -Label quick    # first 10 minutes only, keep the file in input
.\scripts\process-input.ps1 -Interactive -AnswersFile answers.json # guided run with answers from a JSON file
.\scripts\run.ps1 run --config config.local.toml --language auto  # override settings on the command line
.\scripts\run.ps1 watch --config config.local.toml                 # keep watching input until Ctrl+C
```

An answers file looks like
`{"language": "it", "context": "Names: Rossi, Bianchi", "max_minutes": null, "confirm": true}`.
The wizard remembers your last answers in `.local/wizard/last-answers.json`.
Different language, context or scope settings produce a new version of the
transcript; earlier versions are kept for comparison.

Every run is logged to `.local/pipeline-runs/<time>[-label].log` (the launcher's own
messages) and, written by the pipeline itself, to `process/runs/<run-id>/pipeline.log`.

### Live progress and monitoring

On an interactive terminal the run shows a live display: the current file and
stages, a chunk progress bar with ETA and x-realtime speed, counters (ok, no
speech, failed, fallbacks, splits), audio processed, tokens, confidence so far,
sparklines of CPU, RAM, GPU, VRAM, power and temperature, and the latest
events. Pipes, CI and logs get the plain status lines instead.

```powershell
.\scripts\run.ps1 run --config config.local.toml --ui plain      # plain status lines (default when not a terminal)
.\scripts\run.ps1 run --config config.local.toml --ui jsonl      # one JSON event per line on stdout
.\scripts\run.ps1 run --config config.local.toml --no-monitor    # no CPU/GPU sampling
.\scripts\run.ps1 events --follow                                # follow the latest run's events (JSONL)
```

Every run also writes `process/runs/<run-id>/events.jsonl` (all events),
`status.json` (latest snapshot, for polling) and `process/runs/latest.json`; each
version keeps its own `events.jsonl`. GPU numbers come from `nvidia-smi` and are
system-wide. See [run events and monitoring](docs/events.md).

## Where the results are

```text
output/catalog.json                          every input file, latest activity first
output/<name>-<id>/source.json               all versions of one input, newest first
output/<name>-<id>/<version>/
    transcript.txt  transcript.md  transcript.srt  transcript.vtt  transcript.json
    report.json                              settings, timings, counts, confidence
    run-report.md                            the same, human-readable (minutes, seconds)
    intermediate/                            live partial results while a job runs
process/<name>-<id>/<version>/               working files: audio chunks, checkpoint, raw model responses
processed/<name>-<id>/<original file>        your original file after a complete transcription
```

- `<id>` is derived from the file content, so renaming or moving a file keeps
  its folder. `<version>` reads `<UTC time>_<model>_<full|firstNs>_<settings id>`.
- Changing the model, language, context or other settings creates a **new
  version**; earlier versions are never deleted.
- A job that could not transcribe some chunks ends **incomplete**: the partial
  result lists the gaps, the file stays in `input`, and re-running retries only
  the missing chunks.
- Subtitle times are speech-chunk boundaries (≤ 15 s), not word alignment.
  Speakers are not identified.

## Configuration

Main settings in `config.local.toml` (full reference in
[`config.example.toml`](config.example.toml) and
[inference controls](docs/inference-controls.md)):

| Setting | Default | Meaning |
| --- | --- | --- |
| `language` | `it` | Expected spoken language (ISO code). Forces the output language for Qwen3-ASR. |
| `prompt` | none | Optional context for Qwen3-ASR (topic, names, terms). |
| `chunk_seconds` | `15` | Maximum chunk length. |
| `vad` | `silero` | Speech detector: `silero`, `energy` or `none`. |
| `parallel_requests` | `1` | Chunks sent at the same time (needs as many server slots). `2` is ~28 % faster; output may differ slightly, so it is a separate version. |
| `collect_logprobs` | `true` | Confidence score; costs some speed. |
| `ui` | `auto` | Console output: `auto`, `live`, `plain` or `jsonl`. |
| `monitor`, `monitor_interval` | `true`, `1.0` | Resource sampling and its interval in seconds. |
| `fallback_temperatures` | `[0.2, 0.4]` | Retry temperatures for looping/unusable answers. |

The command-line tool is `sbobinai` (alias `audio-transcript`; the Windows
wrapper `.\scripts\run.ps1` calls it). Command-line options override the file,
for example
`.\scripts\run.ps1 run --config config.local.toml --language en --file input\talk.mp4`.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `doctor` says the model is not loaded | Run `.\transcribe.cmd` (it starts the server) or check `[server]` in `config.local.toml`. |
| "Port 8088 is already occupied" | Another program uses the port. Stop it or change `port` and `base_url`. |
| "No NVIDIA CUDA device" | Update the NVIDIA driver; check `nvidia-smi`. |
| Job ends `incomplete` | Open `output/<source>/<version>/intermediate/run-report.md`; re-run to retry the gaps. |
| Names or jargon misrecognized | Add them as context (`prompt`), which creates a new version to compare. |
| File stays in `input` after success | Read `archive_warning` in `process/<source>/<version>/metadata.json`. |

## Documentation

| Document | Content |
| --- | --- |
| [Documentation index](docs/README.md) | All documents and their roles |
| [Specification](docs/specification.md) | Behaviour and acceptance criteria |
| [Inference controls](docs/inference-controls.md) | Segmentation, recovery ladder, response modes, metrics |
| [Run events and monitoring](docs/events.md) | Live console, event stream, status files, resource monitor |
| [Runtime setup](docs/runtime-setup.md) | llama.cpp CUDA runtime, server lifecycle, models |
| [Backend compatibility](docs/backend-compatibility.md) | Model comparison and decision |
| [Verification](docs/verification.md) | Dated test and real-run evidence |
| [Distribution design](docs/distribution-design.md) | Packaging, API, UI and cloud plans |

## Development

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

Tests need no network, model weights or private media. See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, and
[AGENTS.md](AGENTS.md) for the project rules followed by AI coding agents.

## License

Copyright (C) 2026 Massimiliano Camillucci.

SbobinAI is free software licensed under the
[GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0-only). You may use,
study, modify and share it; if you distribute it, or let others use a modified
version over a network, you must publish the corresponding source code under
the same license.

**Commercial license:** organisations that want to use or embed it without the
AGPL obligations can obtain a separate commercial license from the author —
see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

Models and tools downloaded during setup keep their own licenses
(Qwen3-ASR: Apache-2.0, llama.cpp: MIT, Silero VAD: MIT, FFmpeg: LGPL/GPL); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the
[licensing review](docs/licensing-review.md). They are downloaded from their
original publishers and are not redistributed by this repository.

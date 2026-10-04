# SbobinAI

*From the Italian* sbobinare *— to transcribe a recorded lecture or meeting.*

Local, private transcription of long audio and video recordings on your own
computer, fastest on an NVIDIA GPU (with automatic Vulkan and CPU fallback). Drop a file into a folder, run one command, get a clean transcript
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

> **Status:** early release. The **recommended, verified setup is Windows 11 +
> NVIDIA GPU (CUDA)**: it transcribes full 2-hour recordings end to end. Vulkan
> and CPU fallbacks work but are **beta**; Linux, macOS, AMD and Intel GPUs are
> **untested**. See [Support status](#support-status) before relying on anything
> else.

## Quick start

**Install** (Windows + NVIDIA GPU, about 4 GB of downloads), either way:

- **Let an AI coding agent do it.** Install [Claude Code](https://claude.com/claude-code)
  (or another coding agent that can run commands), open a terminal in an empty
  folder, start it (`claude`) and paste:

  ```text
  Clone https://github.com/massimilianoC/SbobinAI and install it on this PC by
  following its README.md and AGENTS.md. Put the models in D:\ai-models (ask me
  if that drive does not exist). Show me the download sizes and ask before
  downloading. When done, run doctor and a 2-minute test on a file I put in the
  input folder, then explain how to use it.
  ```

  The repository's [`AGENTS.md`](AGENTS.md) tells the agent the project rules
  (models outside the system drive, downloads only through the pinned `setup`,
  private files never committed). Check its commands before approving them.
- **Do it yourself** with the commands in [Installation](#installation-windows--nvidia).

**Transcribe:** copy a recording into `input`, double-click `transcribe.cmd`,
answer the questions, read the result in `output\<file>-<id>\<version>\transcript.txt`.
For a quick check on a long file, run `.\transcribe.cmd -MaxDuration 120 -NoArchive -Label quick`
(first 2 minutes only, the file stays in `input`).

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Support status](#support-status)
- [Requirements](#requirements)
- [Installation (Windows + NVIDIA)](#installation-windows--nvidia)
- [Transcribe a file](#transcribe-a-file)
- [Where the results are](#where-the-results-are)
- [Configuration](#configuration)
- [Command-line reference](#command-line-reference)
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

## Support status

What has actually been run on real recordings, and what has not. Automated
tests (no model, no GPU) pass on Windows and Linux with Python 3.11 and 3.14 for
everything below; this table is about real use.

| Level | Meaning |
| --- | --- |
| **Stable** | Verified end to end on real recordings with the production pipeline. Use it. |
| **Beta** | Works in real runs, but tested on one machine or on short samples only. Expect rough edges; reports welcome. |
| **Untested** | Should work by design, never run for real. Try it at your own risk and please report. |
| **Planned** | Not implemented. |

| Area | Level | Evidence and limits |
| --- | --- | --- |
| Windows 11 + NVIDIA GPU (CUDA), Qwen3-ASR-1.7B Q8_0 | **Stable** | Full 2-hour recording, 581/581 chunks in about 4–5 min on an RTX 5070 Ti (~4 GB VRAM). This is the reference setup. |
| Silero VAD segmentation, recovery ladder, resume, `incomplete` status | **Stable** | Exercised on the full recording, including interrupted and resumed jobs. |
| `transcribe.cmd` wizard, `transcribe-batch.cmd`, `run`, live console | **Stable** | Used for all real runs. |
| Exports (TXT, MD, SRT, VTT, JSON), run report, versioned outputs, catalog | **Stable** | Produced by every real run. |
| Italian with forced language (`language = "it"`) | **Stable** | Reviewed qualitatively on real recordings; no word-error-rate measurement yet. |
| Context / glossary (`prompt`) | **Beta** | Fixed all product-name misspellings on one reference recording; prompt echo on noise is handled by a fallback rung. |
| `parallel_requests = 2` | **Beta** | ~28 % faster on the full recording; about 9 % of lines differ slightly (punctuation, minor words), so it is a separate version. |
| Other languages, `language = "auto"` | **Beta** | Auto detection measured on 10 minutes of Italian only (one short greeting mislabelled). Other languages are supported by the model but not reviewed here. |
| Vulkan fallback (automatic when CUDA is missing) | **Beta** | Verified on the same NVIDIA GPU through its Vulkan driver: identical text, about 2× slower. **Not tested on AMD or Intel GPUs.** |
| CPU fallback | **Beta** | Verified on a 12-core desktop CPU: identical text, about 9× slower than CUDA. Fine for short files, slow for long ones. |
| Confidence score | **Beta** | Computed from token log-probabilities; **not calibrated**, use it only to compare versions. |
| `watch`, `repl`, `events --follow`, `migrate-layout`, agent `--json` output | **Beta** | Covered by automated tests; not yet used in long real sessions. |
| `setup` downloads and SHA-256 checks | **Beta** | Verified downloading the Vulkan and CPU runtimes; the CUDA runtime still uses its own install script. |
| First-time install on a new PC, manual or by an AI coding agent | **Untested** | The steps are documented but have not yet been run on a clean machine. Reports welcome. |
| Windows 10, other NVIDIA GPUs, GPUs with less VRAM | **Untested** | Expected to work (needs about 4 GB of free VRAM and a current driver). |
| AMD and Intel GPUs (Vulkan) on Windows | **Untested** | The automatic fallback should pick them; never run. |
| Linux (any backend) | **Untested** | Python code and tests run on Linux in CI, but the launchers are PowerShell scripts and no real transcription has been done. |
| macOS / Apple Silicon (Metal) | **Untested** | Not wired up; see [multiplatform porting](docs/multiplatform-porting.md). |
| Qwen3-ASR bf16 profile | **Untested** in production | Measured once on 10 minutes (same text as Q8_0); not used for full runs. |
| Legacy Nexa / Qwen2-Audio backend | **Untested** | Kept as an optional adapter; superseded by Qwen3-ASR and not maintained. |
| Transcribing part of a file (`--start`/`--end`), HTTP API, web UI, portable installer, translation, speaker identification, word-level timestamps | **Planned** | See the [distribution design](docs/distribution-design.md) and [TODO](docs/TODO.md). Only "first N minutes" (`--max-duration`) exists today. |

If you run SbobinAI on an untested setup, please open an
[issue](https://github.com/massimilianoC/SbobinAI/issues) with the output of
`sbobinai doctor --json` (remove any private paths) and whether the transcript
looked right.

## Requirements

| Component | Requirement |
| --- | --- |
| OS | Windows 10/11 x64 (Linux/macOS planned) |
| GPU | NVIDIA recommended (6 GB VRAM or more, current driver; the CUDA 13.x runtime is downloaded). Without one the run falls back automatically to Vulkan (any GPU with a Vulkan driver, about 2x slower) or the CPU (about 9x slower), with a clear warning; transcripts were identical on all three |
| Python | 3.11 or newer |
| Disk | ~4 GB for runtime and models, plus space for your recordings |
| Network | Only during setup, to download the runtime and model files |

Download sizes (Windows, default profile): llama.cpp CUDA runtime
~575 MB (only when an NVIDIA GPU is detected), Vulkan runtime ~33 MB, CPU runtime
~19 MB, Qwen3-ASR-1.7B Q8_0 + audio projector ~2.8 GB, Silero VAD 2.3 MB,
FFmpeg ~100 MB. All files are pinned to exact versions and verified by SHA-256
(see [`resources.json`](resources.json)).

## Installation (Windows + NVIDIA)

Prerequisites: an up-to-date NVIDIA driver, Git and Python 3.11 or newer. If
Git or Python are missing, install them and then open a **new** PowerShell
window, so they are on the `PATH`:

```powershell
winget install --id Git.Git --exact
winget install --id Python.Python.3.12 --exact
python --version    # must print 3.11 or newer; if not found, add Python to PATH or reinstall it with that option
```

Then run these once in PowerShell. Replace `D:\ai-models` with the folder where
downloaded models should live (any drive; avoid the system drive if space is short).

```powershell
# 0. Get the code and allow this window to run the project's scripts
git clone https://github.com/massimilianoC/SbobinAI.git
cd SbobinAI
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # this window only

# 1. Python environment with the speech-detection extra
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m pip install -e '.[vad]'

# 2. FFmpeg (via winget)
.\scripts\install-ffmpeg.ps1

# 3. Local configuration (config.local.toml, git-ignored) with all paths filled in
.\scripts\run.ps1 init-config --store D:\ai-models --profile qwen3-asr-1.7b-q8

# 4. Model files: shows the plan and total download size, asks before downloading,
#    verifies SHA-256, installs the small Vulkan and CPU runtimes itself and prints the
#    command for the (large) CUDA runtime when an NVIDIA GPU is present
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

## Command-line reference

Every command and option is documented in the program itself, from one source:

```powershell
sbobinai --help                  # overview: workflow, commands by group, global notes
sbobinai help run                # same as `sbobinai run --help`: options, examples, exit status
sbobinai man                     # full manual (configuration keys, files, exit codes, recipes)
sbobinai man --format json       # machine-readable interface description (schema_version 1)
sbobinai repl --config config.local.toml   # interactive shell with history
```

The generated manual is committed as [`docs/cli-reference.md`](docs/cli-reference.md)
(regenerate with `python scripts/gen-cli-docs.py`). **Agents and scripts:** every
command accepts `--json`, results and errors have stable shapes and exit codes, and
the manual's *Agent usage* section lists copy-paste recipes; see also the agent
manifest [`skills/sbobinai/SKILL.md`](skills/sbobinai/SKILL.md). The console script
is also installed as `audio-transcript` and `cli-anything-sbobinai`.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `doctor` says the model is not loaded | Run `.\transcribe.cmd` (it starts the server) or check `[server]` in `config.local.toml`. |
| "Port 8088 is already occupied" | Another program uses the port. Stop it or change `port` and `base_url`. |
| Yellow "GPU backend fallback" warning | CUDA was not usable, so Vulkan or the CPU runs instead (slower, same transcript). Update the NVIDIA driver, check `nvidia-smi`, or see [runtime setup](docs/runtime-setup.md#inference-backends-and-automatic-fallback). |
| "backend = cuda is not usable" | `[server].backend` is fixed and never falls back; fix the runtime/driver or set `backend = "auto"`. |
| Job ends `incomplete` | Open `output/<source>/<version>/intermediate/run-report.md`; re-run to retry the gaps. |
| Names or jargon misrecognized | Add them as context (`prompt`), which creates a new version to compare. |
| File stays in `input` after success | Read `archive_warning` in `process/<source>/<version>/metadata.json`. |

## Documentation

| Document | Content |
| --- | --- |
| [Documentation index](docs/README.md) | All documents and their roles |
| [Command-line reference](docs/cli-reference.md) | Every command, option, config key, exit code and agent recipe |
| [Specification](docs/specification.md) | Behaviour and acceptance criteria |
| [Inference controls](docs/inference-controls.md) | Segmentation, recovery ladder, response modes, metrics |
| [Run events and monitoring](docs/events.md) | Live console, event stream, status files, resource monitor |
| [Runtime setup](docs/runtime-setup.md) | llama.cpp CUDA runtime, server lifecycle, models |
| [Backend compatibility](docs/backend-compatibility.md) | Model comparison and decision |
| [Verification](docs/verification.md) | Dated test and real-run evidence |
| [Distribution design](docs/distribution-design.md) | Packaging, API, UI and cloud plans |

## Development

```powershell
.\scripts\setup.ps1 -Dev     # adds the test and lint tools
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

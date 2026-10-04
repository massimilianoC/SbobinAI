# Test plan

SbobinAI is tested in three layers. Layers 1 and 2 need no network, model weights,
GPU or private media and run on Windows and Linux in CI. Layer 3 uses a real GPU
and is run by the maintainer.

## Layer 1: unit and component tests (synthetic data)

Standard-library `unittest` with fakes (`tests/fakes.py`), a mock backend and
synthetic media bytes. They cover configuration validation, segmentation, the
recovery ladder, exporters, the pipeline (resume, idempotency, job identity),
catalog and migration, events and live console, resource download planning, the
wizard, and the command line documentation.

| Module group | What it proves |
| --- | --- |
| `test_core`, `test_fallback`, `test_parallel`, `test_segmentation`, `test_sampling`, `test_confidence` | Pipeline behaviour, recovery ladder, concurrency, segmentation |
| `test_adapters`, `test_backends`, `test_llamacpp`, `test_vad`, `test_exporters`, `test_server_config` | Adapters and their failure modes, exports, server profile |
| `test_layout`, `test_resources`, `test_events`, `test_sysmon`, `test_ui`, `test_wizard` | Layout and migration, downloads, event stream, console, wizard |
| `test_languages`, `test_wizard_text` | Language codes and names (`it`, `Italian`, `italiano`, `it-IT`), wizard translations (same keys and placeholders in every JSON file), system-language detection, Italian guide end to end |
| `test_cli`, `test_monitoring_cli` | CLI contracts: exit statuses, flag overrides, events command |
| `test_cli_docs` | Help text for every command and option, `help` equals `--help`, 80-column ASCII help, manual and JSON interface description match the parser and `AppConfig`, committed `docs/cli-reference.md` and both `SKILL.md` copies equal the generated output, job-identity column verified against the fingerprint, exit and error code tables |
| `test_cli_json` | `--json` for every command, `doctor --json` with fake checks, structured errors, run result line, idempotent re-run, the interactive shell with scripted stdin |

## Layer 2: CLI subprocess tests

`tests/test_cli_subprocess.py` starts real processes (`python -m audio_transcript`
and, when installed, the console scripts `sbobinai`, `audio-transcript` and
`cli-anything-sbobinai`) and asserts exit status, stdout/stderr separation and JSON
shape for success (0), usage error (2), configuration error (2), runtime error (1)
and the JSON error envelope.

## Layer 3: real end-to-end runs (maintainer, NVIDIA GPU)

Run through the launcher with the real llama.cpp CUDA server and Qwen3-ASR:

```powershell
.\scripts\process-input.ps1                       # unattended run of input/
.\scripts\process-input.ps1 -Interactive -AnswersFile answers.json
.\.venv\Scripts\sbobinai.exe doctor --config config.local.toml --json
```

Check: exit status 0, `output/<source>/<version>/transcript.txt` and
`run-report.md` present, `output/catalog.json` updated, a second run reports the
version as skipped, and `process/runs/<run-id>/events.jsonl` ends with
`run.finished`. Record the date, model, audio length and wall time in
[docs/verification.md](docs/verification.md).

Last real GPU end-to-end run: 2026-10-04 — first 20 minutes of a 2-hour
Italian recording through `transcribe-batch.cmd` in a visible console
(Qwen3-ASR-1.7B Q8_0, CUDA, 2 parallel requests): 73/73 chunks, 30 s wall time.
Full 2-hour recording: 2026-10-02, 581/581 chunks.
Backend fallback: 2026-10-04 — CUDA, Vulkan (CUDA hidden) and CPU (CUDA and
Vulkan hidden) through `transcribe-batch.cmd` in visible consoles, identical
overlapping text. Simulate missing devices with `CUDA_VISIBLE_DEVICES=-1` and
`GGML_VK_VISIBLE_DEVICES=99` (do not use empty values on Windows).

## How to run

```powershell
# everything, sequentially (set SILERO_VAD_MODEL to run the optional real-VAD tests)
$env:SILERO_VAD_MODEL = 'path\to\silero_vad.onnx'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# one layer or module
.\.venv\Scripts\python.exe -m unittest tests.test_cli_docs tests.test_cli_json
.\.venv\Scripts\python.exe -m unittest tests.test_cli_subprocess

# lint and format
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts

# regenerate or verify the generated CLI documentation
.\.venv\Scripts\python.exe scripts\gen-cli-docs.py [--check]
```

## Current results

Full suite, sequential, with `SILERO_VAD_MODEL` set (date 2026-10-04):

| Metric | Value |
| --- | --- |
| Tests run | 406 |
| Passed | 406 |
| Failed | 0 |
| Skipped | 0 |
| `ruff check` | clean |
| `ruff format --check` | clean |

New in this layer set: `test_cli_docs` 27, `test_cli_json` 28,
`test_cli_subprocess` 11.

Real GPU end-to-end: not part of the automated count; maintainer runs on
2026-10-02 (full recording) and 2026-10-04 (bounded runs on CUDA, Vulkan and CPU).

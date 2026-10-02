# Transcription instructions, segmentation and reproducibility

Reviewed: 2026-10-02. Applies to the `llamacpp` backend.

## Reading paths

- [Configure a run](#configure-a-run)
- [Pause-aware segmentation (SEG)](#pause-aware-segmentation-seg)
- [Per-chunk recovery ladder (REC)](#per-chunk-recovery-ladder-rec)
- [Response modes](#response-modes)
- [Execution metrics](#execution-metrics)
- [Reproducibility](#reproducibility)

## Configure a run

CLI options override TOML. Every setting below participates in job identity,
so a changed prompt, model, segmentation or fallback setting cannot silently
reuse a transcript produced under different settings.

| Setting | Default | Purpose |
| --- | --- | --- |
| `model` / `--model` | `qwen2-audio-7b` | Server alias that must be loaded in llama.cpp. |
| `response_mode` / `--response-mode` | `json` | `json`, `plain` or `qwen3-asr` (see [response modes](#response-modes)). |
| `prompt` / `--prompt` / `--prompt-file` | Built-in Italian instruction (`json`/`plain`); none (`qwen3-asr`) | Task instruction; in `qwen3-asr` mode it is optional system context. |
| `language` / `--language` | Not set | Expected language (ISO code), or `auto` for detection. Folded into the built-in prompt; forces the output language in `qwen3-asr` mode. |
| `force_language` / `--[no-]force-language` | `true` | `qwen3-asr` only: constrain output to `language` instead of auto-detection. |
| `temperature` / `--temperature` | `0.0` | First-attempt temperature (greedy). |
| `seed` / `--seed` | `42` | Sampler seed for controlled comparisons. |
| `max_tokens` / `--max-tokens` | `1024` | Hard ceiling for any request. |
| `min_tokens`, `tokens_per_second` | `48`, `8.0` | Per-chunk cap `min(max_tokens, ceil(min_tokens + tokens_per_second × seconds))`. |
| `compression_ratio_threshold` | `2.4` | Whisper-style zlib ratio above which text counts as a repetition loop. |
| `repeat_penalty`, `dry_multiplier` | `1.0`, `0.0` | Optional llama.cpp repetition controls; defaults leave them off. |
| `fallback_temperatures` / `--fallback-temperatures` | `[0.2, 0.4]` | Temperatures tried after degenerate output. |
| `split_on_failure` / `--[no-]split-on-failure` | `true` | Split a still-failing chunk once at its quietest pause. |
| `context_free_fallback` / `--[no-]context-free-fallback` | `true` | Retry once without the configured context when the model echoed it. Part of job identity only when disabled. |
| `chunk_seconds` / `--chunk-seconds` | `15` | **Maximum** chunk length in seconds (not a fixed length). |
| `vad` / `--vad` | `silero` | Speech detector: `silero`, `energy` or `none` (fixed contiguous chunks). |
| `vad_model_path` | `silero-vad/silero_vad.onnx` | Silero VAD ONNX file; a relative path resolves against `[resources].store`. |
| `parallel_requests` / `--parallel-requests` | `1` | Concurrent chunk requests (1–8); must not exceed the server slots (`-Parallel`). Values above 1 are a distinct version, because batched GPU execution may change greedy output slightly. |
| `intermediate_interval_seconds` | `10` | Minimum seconds between intermediate export rewrites (0 = every chunk); the checkpoint is still written after every chunk. Not part of job identity. |
| `collect_logprobs` / `--[no-]collect-logprobs` | `true` | Request token log-probabilities for the uncalibrated confidence proxy (mean token probability); part of job identity. |
| `vad_threshold`, `vad_min_speech_seconds`, `vad_min_silence_seconds`, `vad_speech_pad_seconds`, `vad_max_merge_gap_seconds`, `vad_energy_margin_db` | `0.5`, `0.25`, `0.1`, `0.2`, `1.0`, `15` | Segmentation tuning (see below). |

`config.example.toml` lists every key with comments. Keep machine-specific
paths in the ignored `config.local.toml`.

## Pause-aware segmentation (SEG)

**SEG-01 — speech-only chunks.** Preparation extracts one mono 16-bit WAV at
16 kHz, scores it with a speech detector and sends only speech to the model.
Silence never reaches inference: in the reviewed failure, near-silent fixed
chunks made Qwen2-Audio invent stock sentences or repeat its own prompt.

**SEG-02 — cut at pauses, at most `chunk_seconds`.** The planner
(`src/audio_transcript/domain/segmentation.py`) follows Silero VAD
`get_speech_timestamps` semantics: onset at `vad_threshold`, offset at
`threshold − 0.15` (hysteresis), runs shorter than `vad_min_speech_seconds`
dropped. A run that would exceed the limit is cut at its last pause of at least
`vad_min_silence_seconds` (about 100 ms), otherwise at the lowest-probability
frame; a hard cut is the last resort. Regions are padded by
`vad_speech_pad_seconds` and neighbours merged while the gap is at most
`vad_max_merge_gap_seconds` and the merged span stays within `chunk_seconds`.
Every chunk is therefore at most `chunk_seconds` long, ordered and
non-overlapping, with source-relative start/end times.

**SEG-03 — detectors.** `silero` runs the Silero VAD v6 ONNX model with
onnxruntime on CPU (optional extra `pip install -e .[vad]`, model installed by
`scripts/install-silero-vad.ps1` with a pinned tag and SHA-256). It needs no
GPU memory and processed a two-hour recording in well under a minute.
`energy` uses FFmpeg frame energy with an adaptive threshold and needs no extra
dependency, but it treats non-speech noise (keyboard, room) as speech more often.
`none` restores the historical contiguous fixed chunks for comparison.

**SEG-04 — durations.** Reports distinguish the analysed audio duration from the
speech seconds sent to the model. Archival compares the analysed duration with
the source, so trailing silence does not keep a fully processed input queued.
A recording without detected speech completes with zero segments and a
"No speech detected" warning.

Qwen's [official Qwen2-Audio repository](https://github.com/QwenLM/Qwen2-Audio)
recommends clips under 30 seconds. llama.cpp pads every audio clip to its
30-second encoder window (`audio_chunk_len: 30`), so shorter chunks do not
reduce the audio token count; the 15-second limit exists to keep generation
short and boundaries at natural pauses.

## Per-chunk recovery ladder (REC)

**REC-01 — classify failures.** Connection errors, timeouts and HTTP 5xx are
transient: the identical request is retried up to `retries` times, then the job
fails with its checkpoint kept (a dead server must not mark every chunk failed).
Token-limit termination, invalid structure, repetition loops (compression ratio)
and prompt echoes (six or more consecutive prompt words) are **degenerate
output**: an identical greedy request would fail identically, because the
server answers from its prompt cache.

**REC-02 — ladder.** Degenerate output moves down: configured temperature →
each `fallback_temperatures` value → (only when the model echoed the configured
prompt/context and `context_free_fallback` is on) one attempt **without the
context** → one split of the chunk at its quietest pause (each half runs the
temperature ladder). The no-context rung exists because Qwen3-ASR returns the
context text on sub-second noise; it is recorded as stage `no_context` with
`context: omitted` in the attempt history. Every attempt saves its raw
response as `responses/<index><part>_t<temperature×100>.json`; nothing is
overwritten.

**REC-03 — incomplete instead of aborted.** A chunk that still fails is
recorded under `failed` in the checkpoint with its attempt history, and the job
continues. A job with failed chunks ends `incomplete`: intermediate exports and
report list the gaps with time ranges, no final exports are written and the
input stays queued. A rerun with the same settings skips completed chunks and
retries failed ones. HTTP 4xx and unexpected errors still fail the job.

Empty model output for a chunk (`{"transcript": ""}`, empty plain text or
Qwen3-ASR `language None`) is recorded as **no speech**, not as a failure.

## Response modes

| Mode | Request | Accepted output |
| --- | --- | --- |
| `json` | Instruction + audio; JSON schema with one `transcript` string | Exactly `{"transcript": "..."}`. |
| `plain` | Instruction + audio | Plain text. |
| `qwen3-asr` | Audio only; optional prompt as system context; with `language` and `force_language`, assistant prefill `language <Name><asr_text>` | Qwen3-ASR protocol `language <Name><asr_text><text>`, or the continuation after the forced prefix. |

The JSON schema controls shape, not truth. The adapter exports only the
transcript text and never removes arbitrary words with regular expressions.

**Language hint.** For `json`/`plain` without a custom prompt, the language is
folded into the built-in sentence ("…nella lingua originale (it)."). A custom
prompt is sent unchanged. The former separate "Lingua indicata…" sentence was
removed: the model returned it verbatim as a transcript for silent chunks.

**Qwen3-ASR forced language.** Auto-detection mislabelled a 0.7-second Italian
greeting as Chinese in the bounded comparison. The official Qwen3-ASR toolkit
fixes the language by prefixing the model's own output protocol; the adapter
does the same through llama.cpp assistant prefill. The audit records the
forced and detected language. A reply naming a different language is
degenerate output. Language codes are mapped to Qwen3-ASR names
(`it` → `Italian`); unsupported values fall back to auto-detection.

## Execution metrics

The checkpoint keeps one history entry per attempt: part, ladder stage,
temperature, audio seconds, wall seconds, outcome and backend metrics
(prompt/completion tokens, finish reason, requested token cap, llama.cpp
prompt/generation milliseconds, tokens per second, cache hits, compression
ratio). Metrics never contain transcript or prompt text. Intermediate and
final `report.json` contain an `execution` block (`schema_version` 1): chunk
counts (ok/no speech/failed), audio seconds sent, inference wall seconds,
real-time factor, completion-token sum/max/p95, attempts, temperature fallbacks
and splits. The run log prints one line per attempt, for example:

```text
Chunk 58/581 [574.6-588.9s, 14.3s audio] t=0 ok 0.3s 48 tok
Chunk 9/24 [446.0-452.7s, 6.7s audio] t=0 DegenerateOutputError(length) 1.2s 102 tok -> fallback t=0.2
```

GPU telemetry and stage timings beyond preparation/inference remain part of
[OBS-01](TODO.md#obs-01--execution-and-resource-report-requested-not-urgent).

## Reproducibility

```powershell
.\scripts\process-input.ps1 -MaxDuration 600 -NoArchive -Label bounded-check
.\scripts\process-input.ps1 -ConfigPath .local\configs\other-model.toml -MaxDuration 600 -NoArchive -Label other-model
.\scripts\run.ps1 run --config config.local.toml --file input\sample.wav --prompt-file prompts\transcription.txt --force
```

`process-input.ps1` runs `doctor`, then `run`, and stores the console transcript
with start/end time, arguments and wall time under `.local/pipeline-runs/`.
`-MaxDuration` creates a bounded job with a distinct identity; `-NoArchive`
keeps the source queued for comparisons. Run comparisons one model at a time
through this entry point rather than through side scripts.

Greedy decoding and a fixed seed improve repeatability on the same stack. They
do not guarantee identical output across runtime versions, GPU kernels or
chunk boundaries. Preserve versions, settings, original audio and raw responses
when comparing runs. Keep private prompts in ignored local storage.

## Later analysis

Summarization, correction, terminology, speaker proposals or sound analysis
must be separate explicit stages/artifacts that preserve the raw transcript and
record model, prompt, settings and segment references.

See [runtime setup](runtime-setup.md), [specification](specification.md),
[verification](verification.md) and [execution review](execution-review.md).

## Revision record

- **2026-10-02:** initial prompt, structured output and reproducibility controls.
- **2026-10-02:** documented Qwen's under-30-second guidance and the fixed-chunk failure modes.
- **2026-10-02:** implemented SEG-01–SEG-04 (Silero VAD, ≤15 s pause-aware chunks),
  REC-01–REC-03 (failure classes, fallback ladder, `incomplete` status), the
  `qwen3-asr` response mode with forced language, removal of the echoed language
  sentence, dynamic token caps and per-attempt execution metrics.

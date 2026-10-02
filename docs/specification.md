# MVP specification

Date: 2026-10-02. Planning: Sol. Implementation: Luna agents with coordinating review.

## Reading paths

- [Goal](#goal)
- [Workflow and boundaries](#workflow-and-boundaries)
- [Backends and truthful output](#backends-and-truthful-output)
- [Acceptance](#acceptance)
- [Documentation maintenance](#documentation-maintenance)
- [Out of scope](#out-of-scope)

Use the [documentation index](README.md) for related contracts and evidence.

## Goal

Provide a lightweight local CLI for audio/video transcription. A batch command
discovers files in `input`; watch mode discovers files after their size and
modification time stabilize. Process sequentially to bound GPU and memory use.
Preserve original media bytes. `input` is the operational queue, and `processed`
stores original queued sources after a successful complete real transcription.

## Workflow and boundaries

1. Register supported media under a **source key** computed from the file
   content (size plus first and last 4 MiB), stable across rename, archive move
   and timestamp changes. A **version** of that source is identified by the run
   configuration fingerprint (model, prompt, language, segmentation, fallback,
   scope…). Job identity is `<source-folder>/<version-folder>` (`LAY-01`).
2. Store metadata, chunks, raw responses and checkpoints under
   `process/<stem>-<sourcekey12>/<version>/` and artifacts under
   `output/<stem>-<sourcekey12>/<version>/`, where `<version>` is
   `<created UTC YYYYMMDDTHHMMZ>_<model>_<full|first<N>s>_<fingerprint8>`.
   Keep `input` free of job metadata. Never delete earlier versions (`LAY-02`).
   `output/<source>/source.json` lists every version newest first (status,
   scope, model, completion %, durations, chunk counts, real-time factor,
   fallbacks, uncalibrated confidence, artifact paths) and `output/catalog.json`
   lists all sources, so a frontend can offer versions of one input for
   comparison (`LAY-03`). `audio-transcript catalog` rebuilds both; legacy flat
   job folders are moved with the logged, dry-run-first `migrate-layout` command.
3. Probe with FFprobe, select the first audio stream, and extract mono 16-bit PCM
   WAV at 16 kHz with FFmpeg. Detect speech (Silero VAD on CPU by default) and
   send only speech chunks, cut at pauses and at most `chunk_seconds` (default
   15 s) long (`SEG-01`–`SEG-04` in [inference controls](inference-controls.md)).
4. Call a replaceable backend through a domain protocol, one chunk at a time.
   Retry transient backend errors unchanged; move degenerate output (token
   limit, loop, prompt echo, invalid structure) down a temperature/split
   fallback ladder; record still-failing chunks and continue (`REC-01`–`REC-03`).
5. Export JSON, UTF-8 TXT, Markdown, SRT, VTT, and a machine-readable report.
6. Persist progress atomically. Skip completed unchanged jobs, recover interrupted
   jobs, and continue a batch after a per-file failure. Prevent concurrent writers.
7. Move fully transcribed real direct queue files atomically to
   `processed/<stem>-<sourcekey12>/<original-name>`. Never overwrite different archived data.
   The analysed duration must match the probed source duration within 0.05 s
   (ffprobe rounding and codec padding); bounded runs never archive.
   Failed, bounded, preparation-only and mock jobs remain in input. External
   `--file` sources stay in their original location. Archive problems retain the
   source and expose a recoverable warning. Allow `--no-archive-inputs` for tests.

Configuration and CLI flags select paths, backend, model, language, device,
maximum chunk length, speech detector and its tuning, maximum duration, prompt,
response mode, temperature, seed, fallback ladder and token caps.
CLI values override TOML configuration; a UTF-8 prompt file can supply reusable
instructions. Prompt and generation settings participate in job identity.
`doctor` checks dependencies without model downloads. `--prepare-only` produces
real normalized audio without inference; this state must never count as a
completed transcript. A bounded run must have a distinct identity from a full run.

## Backends and truthful output

The supplied reference targets historical Nexa SDK Qwen2-Audio (`qwen2audio`).
Keep that adapter optional and lazy. The current execution route is the
`llamacpp` adapter: a separate local server loads existing Qwen2-Audio GGUF
weights and a compatible audio projector; the application sends base64 WAV to
`/v1/chat/completions` and validates health/model metadata before use. Configure
the loopback URL, timeout and token limit explicitly. Empty or truncated
completions fail rather than count as successful transcripts.
The deployment requires NVIDIA CUDA for model and projector inference. Shared
startup validates CUDA availability and records GPU placement evidence.
Default JSON-schema responses contain exactly one transcript string; export
that field while preserving the raw response and settings for review. Do not
silently strip possibly spoken words or add a second rewriting model.
The `qwen3-asr` response mode serves Qwen3-ASR GGUF models on the same server:
audio-only requests, the model's `language <Name><asr_text>` protocol parsed
strictly, and the configured language forced through that prefix. Model
choice and evidence are recorded in [compatibility](backend-compatibility.md).

**Observability requirement OBS-01 (partly implemented):** reports now include
an `execution` block with per-attempt wall time, tokens, llama.cpp timings,
real-time factor and fallback counts, plus preparation wall time in the run
log. Still missing: end-to-end stage timings in the report, execution origin,
detailed runtime/model provenance and sampled GPU utilization/memory.
Add these as a versioned report schema under [TODO](TODO.md#obs-01--execution-and-resource-report-requested-not-urgent).
GPU readings must disclose their sampling/source and limits on attributing
shared-device use to one process. Never invent unavailable measurements or
include credentials, audio, transcript text or private machine paths in public
artifacts.

Do not assume today's Nexa package implements the historical API. Do not
silently substitute a different model. The inspected LM Studio endpoint accepts
text but rejects audio content for this model with HTTP 400. Generic text chat
compatibility is insufficient evidence. Future backends must satisfy the same
domain interface without changes to orchestration. See
[runtime setup](runtime-setup.md) and [compatibility evidence](backend-compatibility.md).

Chunk start/end times provide coarse subtitle timing, not word alignment.
Speaker identities remain absent unless an adapter can provide supported
diarization evidence. Rich analysis is deferred until a verified backend contract
exists; the MVP must preserve a faithful transcript rather than fabricate facts.
Mock output is explicitly synthetic and only validates plumbing.

## Acceptance

- Audio and video reach inference as normalized WAV chunks of at most
  `chunk_seconds`, cut at pauses; detected non-speech is not sent to the model.
- One unusable model answer does not abort a long job: degenerate chunks go
  through the fallback ladder, remaining failures end the job `incomplete` with
  listed gaps, no final exports and the input kept queued; a rerun retries only
  failed chunks. Every attempt's raw response is preserved.
- Original media bytes remain unchanged; spaces and Unicode filenames work.
- Full real queue success archives the source; failures and partial/test jobs
  remain queued. Moving a file back to input plus `--force` explicitly reprocesses.
- Sources with matching stems or differing settings cannot reuse incorrect output.
- Successful reruns skip work; failed/interrupted runs can resume checkpoints.
- A failed job does not prevent processing another eligible file.
- Missing tools/runtime, invalid configuration, and corrupt audio fail clearly.
- JSON and subtitles preserve Unicode; timestamps are finite, ordered, within
  duration, and explicitly identified as coarse when derived from chunk boundaries.
- Tests need no network, model weights, or private media.
- Real media/model tests must use the documented production pipeline. Preserve
  the command/settings, execution log, checkpoints, raw responses and intermediate
  exports so the operator can inspect and reproduce them. Output metadata must
  include the effective prompt, language hint, temperature, seed and response mode.
- Update readable intermediate exports after each completed chunk under
  `output/<source>/<version>/intermediate`, explicitly marked as incomplete; retain
  them with failure status when later inference fails. Final exports require full
  success. Each version also has a human-readable `run-report.md` (times in
  minutes and seconds, stage timings, counts, confidence) besides the JSON report.
- Verify synthetic end-to-end, actual media preparation, and actual model inference
  separately. Record the last as blocked if runtime/weights are unavailable.

## Documentation maintenance

Documentation revision is a completion requirement for each development/review
session, including vision and framework working documents when affected.
Follow the [documentation policy](documentation-policy.md) and its `DOC-01` to
`DOC-05` acceptance requirements:

1. Archive supplied originals without content changes before editorial rewriting.
2. Restructure working documents with readable headings, navigation, semantic
   requirement IDs and an explicit source-to-intent mapping.
3. Add concise concept/development hints and links to specifications, design,
   implementation and verification evidence.
4. Distinguish original intent, user follow-ups, source claims, implementation
   decisions, verified facts, proposals and deferred requirements.
5. Reconcile related documents, update the index and revision record, and verify
   source coverage, relative links/anchors and existing privacy boundaries.

Public documents remain in English; private vision/review may remain in Italian.
Original archives preserve their supplied language and remain immutable. Missing
original content or misleading verification status fails documentation acceptance.

## Out of scope

An application HTTP server, parallel GPU jobs, exact alignment, verified
diarization and cloud providers are future extensions. The separate inference
server is implemented. CLI runs do not download weights automatically; explicit
user-authorized runtime/model setup may download required components while
preserving existing originals and recording provenance.

## Revision record

- **2026-10-02:** added mandatory source-preserving document revision, semantic
  navigation, development hints, traceability and completion checks.
- **2026-10-02:** added the local llama.cpp HTTP backend and shared runtime
  lifecycle; separated explicit setup downloads from automatic CLI behavior.
- **2026-10-02:** recorded mandatory NVIDIA CUDA, operational input/processed
  queue semantics and configurable structured transcription controls.
- **2026-10-02:** added the per-source versioned layout (`LAY-01`–`LAY-03`),
  catalog, Markdown run report, archive-duration tolerance and logprob-based
  confidence proxy.
- **2026-10-02:** added pause-aware speech segmentation (`SEG`), the per-chunk
  recovery ladder and `incomplete` status (`REC`), per-attempt execution
  metrics, and the `qwen3-asr` response mode; `chunk_seconds` is now a maximum.
- **2026-10-02:** added deferred `OBS-01` for end-to-end, stage, chunk, runtime,
  execution-origin and optional GPU-resource reporting.

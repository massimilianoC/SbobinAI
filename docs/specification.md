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

1. Register supported media with a safe filename stem and a collision-resistant
   identity based on source path, size, modification time, and run configuration.
2. Store metadata, chunks, raw responses and checkpoints under `process/<job-id>/`,
   and final artifacts under `output/<job-id>/`. Keep `input` free of job metadata.
3. Probe with FFprobe, select the first audio stream, and extract mono 16-bit PCM
   WAV with FFmpeg. Default to 16 kHz and chunks no longer than 30 seconds.
4. Call a replaceable backend through a domain protocol, one chunk at a time.
5. Export JSON, UTF-8 TXT, Markdown, SRT, VTT, and a machine-readable report.
6. Persist progress atomically. Skip completed unchanged jobs, recover interrupted
   jobs, and continue a batch after a per-file failure. Prevent concurrent writers.
7. Move fully transcribed real direct queue files atomically to
   `processed/<job-id>/<original-name>`. Never overwrite different archived data.
   Failed, bounded, preparation-only and mock jobs remain in input. External
   `--file` sources stay in their original location. Archive problems retain the
   source and expose a recoverable warning. Allow `--no-archive-inputs` for tests.

Configuration and CLI flags select paths, backend, model, language, device,
chunk length, maximum duration, prompt, response mode, temperature and seed.
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

- Audio and video reach inference as bounded normalized WAV chunks.
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
  `output/<job-id>/intermediate`, explicitly marked as incomplete; retain them
  with failure status when later inference fails. Final exports require full success.
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

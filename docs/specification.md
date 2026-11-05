# MVP specification

Date: 2026-10-02 (revised since; see the revision record at the end).

## Reading paths

- [Goal](#goal)
- [Workflow and boundaries](#workflow-and-boundaries)
- [Backends and truthful output](#backends-and-truthful-output)
- [Acceptance](#acceptance)
- [Guided desktop contract](#guided-desktop-contract)
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
   Delivery selection may expose only the requested formats (GUI-07); complete
   canonical text/report artifacts remain recoverable in the job's process folder.
6. Persist progress atomically. Skip completed unchanged jobs, recover interrupted
   jobs, and continue a batch after a per-file failure. Prevent concurrent writers.
7. Move fully transcribed real direct queue files atomically to
   `processed/<stem>-<sourcekey12>/<original-name>`. Never overwrite different archived data.
   The analysed duration must match the probed source duration within 0.05 s
   (ffprobe rounding and codec padding); bounded runs never archive.
   Failed, bounded, preparation-only and mock jobs remain in input. External
   `--file` sources stay in their original location. Archive problems retain the
   source and expose a recoverable warning. Allow `--no-archive-inputs` for tests.
   **After success (2026-10-04):** `after_success` decides what happens to an
   archived source: `keep-all` (default, the archive above), `keep-audio` (write
   `<stem>.audio.flac`, the 16 kHz mono audio the model heard, beside it; delete
   the original only when the FLAC duration matches) or `delete-all` (delete the
   original). Both delete the version's chunk WAVs and record
   `source_disposition` in `metadata.json`; any failure keeps the original and
   records `disposition_warning`. The wizard asks it for full-length runs and
   never remembers a deleting answer as the next default.

Configuration and CLI flags select paths, backend, model, language, device,
maximum chunk length, speech detector and its tuning, maximum duration, prompt,
response mode, temperature, seed, fallback ladder and token caps.
CLI values override TOML configuration; a UTF-8 prompt file can supply reusable
instructions. Prompt and generation settings participate in job identity.
`doctor` checks dependencies without model downloads. `--prepare-only` produces
real normalized audio without inference; this state must never count as a
completed transcript. A bounded run must have a distinct identity from a full run.

## Backends and truthful output

The execution route is the `llamacpp` adapter (the only real backend; `mock` is
for tests, and the earlier optional Nexa adapter was removed): a separate local
server loads the GGUF model and its audio projector; the application sends base64 WAV to
`/v1/chat/completions` and validates health/model metadata before use. Configure
the loopback URL, timeout and token limit explicitly. Empty or truncated
completions fail rather than count as successful transcripts.
NVIDIA CUDA is the preferred backend for model and projector inference.
With `[server].backend = "auto"` the launcher falls back to Vulkan, then to
CPU, only when the preferred backend has no usable device; the fallback is
shown as a prominent warning and recorded (backend, device, reason) in
reports, events and versions. An explicitly configured backend never falls
back. Each backend was verified end to end with identical transcripts
([multiplatform porting](multiplatform-porting.md)).
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
Add these as a versioned report schema under [TODO](TODO.md#obs-01--execution-and-resource-report-largely-implemented-2026-10-02).
GPU readings must disclose their sampling/source and limits on attributing
shared-device use to one process. Never invent unavailable measurements or
include credentials, audio, transcript text or private machine paths in public
artifacts.

Do not silently substitute a different model. The inspected LM Studio endpoint accepts
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
  success. Each version has canonical human-readable and JSON reports; their
  public copies are selectable. After success the desktop verifies and compresses
  intermediate exports into `intermediate.zip`; failed/incomplete jobs retain the
  readable folder. Checkpoints and raw model responses remain in process.
- Verify synthetic end-to-end, actual media preparation, and actual model inference
  separately. Record the last as blocked if runtime/weights are unavailable.

## Guided desktop contract

**Accepted scope, 2026-11-05:** optional Python + pywebview desktop preview using
the existing transcription pipeline. [Desktop guide](desktop.md) defines the
operator flow and limits. This extends the CLI contract without requiring the
desktop dependency for headless use.

| ID | Required behavior |
| --- | --- |
| GUI-01 | List the existing input queue. Auto-select one file when it is the only item; otherwise require one explicit choice. Import one audio/video file through a native picker, copying bytes without altering the external original. Publish only complete imports and never overwrite collisions. GUI batch is deferred. |
| GUI-02 | Present five steps: recording; language/context; interval/retention; recap/start/progress; results. After a completed or skipped job, offer an optional sixth step, **Review with subtitles** (GUI-09), reached from the results screen. Validate effective options before confirmation and again before execution. Reuse the existing language validation, pipeline and export semantics. |
| GUI-03 | Use the existing run lock and recoverable state, sequential source jobs and configurable chunk concurrency. Process outside the UI event loop. Consume structured events rather than console prompts. Expose startup/preparation errors and incomplete jobs. Give feedback in every path: the normal and the reuse path emit the same `stage.started`/`stage.finished` events with seconds for audio, video, archive, packaging and catalog; a job whose finished transcription is reused emits `job.reused` and the UI says that no model work is needed and marks the model steps as reused; the MP4 render reports throttled `stage.progress` (percent, ETA, x realtime) and bars never carry over between phases. |
| GUI-04 | Offer two horizontal start/end handles with the selected middle highlighted and discarded ends grey. Calculate the FFmpeg waveform automatically in the background as soon as a recording is selected (including auto-selection and after import): it never blocks navigation, selection, review or import, is cached in memory by source identity (bounded), is cancelled and awaited whenever another recording is selected, a run starts or the window closes (so no FFmpeg handle can block archiving or deletion), and shows a spinner or a non-blocking "Try again" message instead of a global error. `start_seconds` and duration define a distinct interval identity; timestamps are selection-relative and the source offset is recorded. Any bounded interval retains input, regardless of retention policy. Deleting choices require full-run confirmation and verified requested deliveries; the CLI keep-audio policy retains its verified-FLAC safeguard. |
| GUI-05 | Show facts about the work done rather than the user's own inputs: outcome (status, backend and device, speed as x realtime, chunk counts, durations), what actually happened to the queued file (deleted, archived or kept, and why, read from the job's records and never from the chosen option), time per section from this run's `stage.finished` events (hidden when a section did not run; for a reused transcription the earlier run's timings are labelled separately), and a table of the files in the exact version with an honest mark: audio, video and ZIP files are re-probed or read back now ("Verified"; audio and video also check that the stream data reaches the declared duration), text and report files are "Complete" (present, non-empty, equal in size to the canonical copy, JSON parsed) which does not vouch for transcript content, and anything failing shows a red reason without breaking the screen. *Outcome* and *Time per section* are accordions, closed by default, whose headers carry a one-line summary (status plus the queued-file outcome; the total time) so the retention result stays visible; files are the main element. A failed or incomplete job leads with its actual error, opens *Outcome* and lists what exists. Open the exact output version or available intermediate results. Do not open a parent folder as if it were the result. Block normal window closure while processing; GUI cancellation is deferred. |
| GUI-06 | Keep models/runtimes outside the UI package and downloads explicit. Provide a desktop command and Windows launcher. Verify model and audio-encoder placement, visibly report automatic fallback and preserve runtime provenance. Cross-platform source design does not establish real inference support; Linux/macOS and standalone bundles require separate acceptance. |
| GUI-07 | Select TXT, Markdown, SRT, VTT, transcript JSON, report Markdown, report JSON, MP3 and FLAC independently in step 3, with at least one delivery. Default desktop selection is all text/report formats plus MP3; FLAC is opt-in. Preserve canonical exports in process, keep inference identity unchanged and rematerialize formats without new inference. Verify selected audio codec/duration before disposing of a queued source. Offer an optional `mp4` delivery (`video.mp4`), **unchecked by default**: a preview render, not a preservation copy, of the selected interval (zero-based), scaled down to at most 720p without upscaling, H.264/AAC after transcription as part of delivery, encoded on the GPU when possible (NVENC with GPU decoding and scaling, then NVENC with CPU decoding, then CPU libx264; the first working path wins, NVENC only after a real probe encode, every fall-through is a visible warning, the path used is recorded in the delivery record and the `video_export` stage event and shown in the results), with the canonical SRT as a soft `mov_text` track when available (never burned in). It needs a video stream (checked with FFprobe, not the file suffix) and an FFmpeg with `libx264` or a working `h264_nvenc` (plus `aac` and `mov_text`), both validated before inference and before Start in the desktop; the render is verified (video stream, height <= 720, duration within the audio-delivery tolerance) before any source may be disposed of, a failed render retains the source and a re-run restores outputs without new inference. It does not change inference identity. Step 3 groups outputs by type (text, subtitles, audio, video, data and reports) beside a compact retention control. Verify intermediate ZIP contents and unchanged source-file signatures before removing loose intermediates; packaging failure keeps recoverable files and emits a visible warning. |
| GUI-08 | Separate queued-file retention from format selection. The desktop default depends on how the file reached the queue (confirmed by the user 2026-11-05): a copy the GUI imported itself in the current session defaults to delete-all, displayed as deleting the file in input only after full success, with the final start button stating that deletion and a note that the original chosen in the picker stays untouched; a file that was already in input, including one picked from inside input, defaults to keep-all, and choosing delete for it shows a plain warning that it may be the only copy. The origin is session-scoped: after a restart an earlier import counts as already queued (the safe direction). A future GUI batch mixing both kinds must default to keep. Keep-all archives the queued copy in processed. Partial/failed/incomplete/mock runs retain input, and external sources are never deleted. CLI defaults remain keep-all. Place the recap in an initially open accordion, close it automatically on start, and prioritize bounded live activity, stages, chunk progress, ETA, speed and device/resource feedback. The collapsed running view fits the default window. During the run a row of four cards (CPU, GPU, VRAM, RAM: label, large number, level bar, amber above 90 %) shows whole-system figures from the resource monitor, with GPU temperature and NVENC encoder use while the preview renders; unavailable values show a dash and a note says the figures include other applications. |
| GUI-09 | **Review with subtitles** (optional step 6, accepted 2026-11-05; offered only for a completed or skipped job). A single player shows live subtitles from the canonical `transcript.json` (so it works even if SRT/JSON were not selected), a clickable waveform of the selection (hidden when no peaks exist) and a clickable cue list. Media is served only through a loopback HTTP server on `127.0.0.1` with a random token path, one allow-listed file at a time, GET/HEAD with Range (206/416), no path taken from the request and no output on stdout; it is never read through `file://`. Candidates in order: the selected queued file; its archived copy in processed; the original recorded at import; `video.mp4`; `audio.mp3`; `audio.flac`. A source candidate is offered only if size and mtime match the selection-time identity (a different file under these subtitles would mislead); a bounded selection starts at its offset and pauses at its end. If a candidate cannot be decoded the next is tried automatically and the one in use is named; with none playable the cue list stays with an explanation. Subtitle text is shown with `textContent` only. The UI states that timing follows audio chunks (up to about 15 s) and is not word alignment. The page is laid out in normal flow so nothing can overlap at any size: the waveform first at full width (time axis, playhead, click to seek), then a 16:9 player with the live subtitle on the left and the cue list on the right (it follows the left column's height and scrolls on its own), then the hint and buttons. The subtitle box has a fixed height that does not change with its text (long text scrolls inside it). It fits 1100x830 without page scroll; one column below 900 px; no picture box for audio-only media. Leaving the step, selecting another recording or starting a run releases the media element and the served file, so an open handle never blocks archive or deletion. Deletion order is unchanged: after deletion the original or the MP3/MP4 serves as fallback. |

The CLI `run --start-seconds N --max-duration M` shares the interval semantics;
the console wizard rejects nonzero offsets because it does not display them.
No thumbnails, GUI batch or standalone installers are included in this preview. See [verification](verification.md#guided-desktop-preview-2026-11-05).

## Documentation maintenance

Documentation revision is a completion requirement for each development/review
session, including the working documents it affects.
Follow the [documentation policy](documentation-policy.md) and its `DOC-01` to
`DOC-05` acceptance requirements:

1. Keep supplied originals unchanged (in ignored local folders) before editorial rewriting.
2. Restructure working documents with readable headings, navigation, semantic
   requirement IDs and an explicit source-to-intent mapping.
3. Add concise concept/development hints and links to specifications, design,
   implementation and verification evidence.
4. Distinguish original intent, user follow-ups, source claims, implementation
   decisions, verified facts, proposals and deferred requirements.
5. Reconcile related documents, update the index and revision record, and verify
   relative links/anchors and existing privacy boundaries.

Public documents remain in English. Misleading verification status fails
documentation acceptance.

## Out of scope

An application HTTP server, parallel GPU jobs, exact alignment, verified
diarization and cloud providers are future extensions. The separate inference
server is implemented. CLI runs do not download weights automatically; explicit
user-authorized runtime/model setup may download required components while
preserving existing originals and recording provenance.

## Revision record

- **2026-11-05 (review pass):** origin-based retention default (GUI-08, confirmed
  by the user), automatic background waveform (GUI-04), optional sixth step
  (GUI-02) and new GUI-09 review playback, optional 720p MP4 preview delivery and
  grouped step-3 layout (GUI-07). Implementation decisions and the note that the
  loaded ASR model cannot summarize are recorded in [TODO](TODO.md).
- **2026-11-05 (output controls):** accepted GUI-07/GUI-08, selectable public
  artifacts with private recovery data, MP3/FLAC exports, verified intermediate
  ZIP, explicit desktop deletion default and the focused processing view.
- **2026-11-05:** accepted GUI-01–GUI-06 for the five-step single-file pywebview
  preview; added interval selection semantics, retention and platform boundaries.
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

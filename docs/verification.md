# Verification evidence

Date: 2026-10-02 (updated after the full-recording run). Windows x64, Python 3.14.7, FFmpeg 9.0.2.
The core supports Python 3.11+. Remote CI has not been executed.

## Automated and infrastructure checks

- 54 local tests passed with real FFmpeg integration and GGUF mapping available.
- Ruff lint/format passed; package installation and pip check passed.
- Synthetic smoke test passed through actual FFmpeg and the explicitly mock backend.
- Tests cover identities, resume, locking, source preservation, queue archival,
  collisions, bounded/mock retention, intermediate failures and effective prompts.
- Shared llama.cpp b11193/CUDA 13.4 installed from official ZIPs with SHA-256
  validation, manifest and licenses. Lifecycle startup/shutdown/idempotence tested.
- CUDA logs confirm 33/33 model layers offloaded and audio encoder on CUDA0.
- Existing main GGUF reused; original matching projector downloaded separately
  and converted to 489 retained MMPROJ tensors without requantization.
- LM Studio loaded-model text request returned 200; two audio input forms returned
  400. This supersedes the initial snapshot that did not list the model.

## Actual recording preparation

Three random central 20-second samples passed measured energy thresholds and
were prepared as mono PCM16/16 kHz. The silent-beginning test was superseded.
A further random central 180-second excerpt passed energy checks. Offset,
levels and provenance stay in ignored local sampling reports. Energy does not
establish speech accuracy. Original source bytes were preserved.

## Real production-pipeline test

The 180-second excerpt completed through `scripts/process-input.ps1` with CUDA,
18 ten-second chunks, JSON transcript mode, temperature 0 and seed 42. Its
six final exports are present with report status completed. Queue archival
preserved its SHA-256 and removed the test source from input. The repeated
structured-script test also generated intermediate exports, raw responses,
checkpoint state, an execution transcript log, and effective prompt metadata.

The initial plain output had prefatory text. A malformed schema payload was
rejected and fixed; a 20-second chunk encountered repetitive generation and was
rejected at the token limit. The ten-second/short-Italian-prompt test succeeded.
No regex transcript cleanup or second model was used. This establishes actual
compatibility and successful bounded execution, not a word-error-rate guarantee.

## Full recording run

The complete 7545.963-second recording was started through the same structured
script, in a detached hidden process. It has 755 prepared ten-second chunks.
Intermediate output and raw responses update during execution. This is not
reported as a complete transcript before the final report succeeds.

The observed full run stopped at chunk 58 (zero-based 57) after 57 successful
chunks: Qwen repeated text until max_tokens=1024. The latest local final report,
checked on 2026-10-02, records `failed`, 57/755 completed chunks and two warnings.
The original remains in input, and the 57 completed checkpoints are retained.
Full-recording completion and model stability are follow-up work, not a claimed
success. The 180-second pipeline test remains complete. The later full run
below supersedes this failure.

Inspect `output/<source>/<version>/intermediate/report.json` (or `run-report.md`)
for live progress/failure and `output/<source>/<version>/report.json` for final
success; `output/catalog.json` lists every source and version. Execution logs remain under
`.local/pipeline-runs`. Resume with the same command and configuration, without
force, after correcting a failure. Completed chunk checkpoints are retained.

## Segmentation, recovery and model comparison (2026-10-02, later)

- 155 local tests passed (`unittest discover -v`, verbose log kept locally);
  Ruff lint and format passed. New suites cover segmentation invariants
  (300 seeded random trials), Silero/energy detectors with mocked sessions,
  the fallback ladder, `incomplete` status and resume, Qwen3-ASR parsing and
  forced language, dynamic token caps and execution metrics.
- Silero VAD v6.2.3 (pinned SHA-256) on CPU, 1 thread: 3.5 s for a synthetic
  10-minute WAV; preparation of the 2-hour recording including FFmpeg
  extraction took 49.4 s.
- Bounded comparison through `scripts/process-input.ps1` on the first 600 s
  (24 speech chunks): Qwen2-Audio-7B Q8 20.4 s inference and ~10.7 GiB server
  VRAM; Qwen3-ASR-1.7B Q8_0 2.8–4.2 s and ~3.7 GiB; bf16 5.3 s and ~5.3 GiB
  with practically identical text. All runs completed without fallbacks.
- Qwen3-ASR startup logs: 29/29 layers offloaded, `qwen3a` audio encoder on CUDA0.

## Full recording with Qwen3-ASR (2026-10-02)

The complete 7,546-second recording completed through `scripts/process-input.ps1`:
581/581 chunks ok, 0 fallbacks, 0 failed, 6,874 s of speech sent, inference
188.5 s, total wall time 262.4 s, peak ~3.8 GiB model-server VRAM. The source
stayed in input because of an archive-duration tolerance defect (0.3 ms ffprobe
rounding), recorded as RUN-05 and fixed afterwards.
Quality was reviewed qualitatively on sampled passages; no word error rate is
claimed.

After the archive-tolerance fix and the versioned layout, `migrate-layout`
moved 8 legacy jobs (dry run first, move log kept, 6 non-job folders left
untouched, nothing deleted) and a second full run completed in 5 min 30 s with
confidence collection, wrote `run-report.md`, updated `source.json`/`catalog.json`
and archived the source to `processed/<source>/`. 185 local tests passed.

One-command check: with the source back in `input/`, `scripts/process-input.ps1`
without arguments restarted the owned server from its local profile (2 slots; now the `[server]` table),
transcribed with two parallel requests (inference 203.9 s, effective concurrency
1.93×), archived the source and wrote all reports. 205 local tests passed.

Guided end-to-end check (2026-10-02): `process-input.ps1 -Interactive
-AnswersFile` with language and context ran the wizard, transcribed 579/581
chunks, ended `incomplete` on two context echoes, and after the no-context rung
was added an identical rerun resumed only those chunks, completed and archived
the source. 272 local tests passed.

Live monitoring check (2026-10-02, branch `feat/live-monitoring`): a bounded
600-second run through `scripts/process-input.ps1` wrote
`process/runs/<run_id>/events.jsonl` (153 events, strictly increasing `seq`,
10 `resource.sample` events with CPU, RAM, GPU utilization, VRAM, power and
temperature from `nvidia-smi`), `status.json`, `pipeline.log` and
`runs/latest.json`; the launcher printed their paths. Sampled transcript
phrases and the context text were absent from events and log. Replaying the
real events into the live renderer produced a correct 120-column frame. The
animated console itself was not observed by the reviewer (non-TTY tooling).
340 local tests passed.

Backend comparison (2026-10-04): the same 600-second bounded job through the
production pipeline produced identical transcripts with CUDA (6.8 s inference),
Vulkan on the same GPU (13.0 s) and CPU only on a 12-core Ryzen 9 3900X
(60.3 s); see [multiplatform porting](multiplatform-porting.md).

Agent-native CLI (2026-10-04): 406 local tests passed, including subprocess
exit-code and JSON-shape tests; generated reference and SKILL.md verified in
sync; `doctor --json`, structured `config_invalid` error and `man --format json`
checked on the installed command; a 20-minute bounded run through
`transcribe-batch.cmd` in a visible console completed 73/73 chunks in 30 s.

Backend fallback (2026-10-04, visible consoles, `backend = "auto"`): with
`CUDA_VISIBLE_DEVICES=-1` the launcher warned, restarted the owned server on
the Vulkan runtime installed by `setup` and completed 47/47 chunks (after a
resume, see below); with Vulkan also hidden (`GGML_VK_VISIBLE_DEVICES=99`) it
restarted on the CPU runtime (12 threads) and completed 13/13; a normal run
then restarted CUDA without warnings. Overlapping lines were identical across
the three backends. Two defects found and fixed during these runs: a
transient Windows `PermissionError` when replacing `checkpoint.json` failed
a job (atomic writes now retry briefly); the `.cmd` launchers re-executed
themselves when started from a command line containing `&&` (unsafe
`%CMDCMDLINE%` expansion). 451 local tests passed.

Wizard languages (2026-10-04, automated only): answering the transcript-language
question with the spoken language (`it`, `italiano`, `italian`) was rejected as
a translation request. The question was removed (translation is out of scope),
spoken-language answers accept names and region tags and are stored as ISO codes,
and the wizard speaks the system language (en, it, es, fr, de, pt; English
fallback) or the `ui_language` setting. 465 local tests passed; the translated
guide has not yet been used in a real session other than by the unit tests.

Guided choices and after_success (2026-10-04): numbered language menu with
examples, explained scope/context syntax and the original-file question.
`export_audio` was run for real on the 2 h 6 min reference recording (read
only, output outside the repository): 9.59 GB of video gave a 104 MB 16 kHz
mono FLAC in 7.7 s with the same 7546.0 s duration (about 50 MB per hour). The
deletion paths (`keep-audio`, `delete-all`) are covered by automated tests
only. Review fix: a bounded run on a file shorter than its bound archived the
input, contrary to the specification; it now stays in the queue. 479 local
tests passed.

## Guided desktop preview (2026-11-05)

Scope: [GUI-01–GUI-06](specification.md#guided-desktop-contract), implemented as
the optional [pywebview desktop preview](desktop.md). Evidence is separated:

- **Headless suite:** 492 tests ran successfully, with one optional dependency
  check skipped. Thirteen new desktop tests cover atomic import/failure,
  collisions, external-source preservation, interval validation and identity,
  retention, worker failure, exact result paths and placement validation.
- **Actual media preparation:** generated PCM fixtures were decoded with FFmpeg.
  Extracted samples match the selected source interval; waveform peaks reflect
  the fixture's silent and non-silent regions. This is not inference evidence.
- **Real GUI with mock inference:** `scripts/check-desktop-ui.py` passed on
  Windows with pywebview 6.2.1. It opens the real WebView, auto-selects one queued
  generated recording, traverses five screens, calculates its waveform, sets a
  nonzero start, validates retention/recap, executes the isolated mock worker and
  verifies that the exact version contains exports. This does not measure ASR
  accuracy. It also verifies that two queued recordings require an explicit
  choice, and exercises Unicode recording names. Native Computer Use inspection was unavailable because access to the
  Python window was not approved; this check uses the app's own test interface.
- **Real inference, separate check:** the desktop bridge/worker completed a
  bounded 20-second interval from an available recording on CUDA, with two
  successful speech chunks, zero failed chunks and no backend fallback. Startup
  logs confirmed both model buffers and the audio encoder on CUDA0; runtime
  provenance records both placements. The source stayed unchanged in place.
  Private reports, source names and transcript text remain in ignored storage.
- **Static/dependency checks:** Ruff lint/format, generated CLI documentation
  and `pip check` pass. The Python wheel builds and includes all three UI assets
  and the desktop entry point; archive inspection excludes media, models and
  private operational folders. Final `doctor --json` passed and selected CUDA
  without fallback. No model or inference-runtime downloads were performed.

Limits: the GUI's native picker and folder-opening OS handoff have unit-tested
contracts but were not exercised through approved native automation. GUI text
is English, selection is single-file, and graceful cancellation, thumbnails,
standalone installers and Linux/macOS acceptance remain deferred. CUDA evidence
above does not establish GUI support on other devices or backends.

## Selectable desktop deliveries and focused progress (2026-11-05)

Scope: [GUI-07/GUI-08](specification.md#guided-desktop-contract). The GUI now
offers independent transcript/report/audio checkboxes and a reviewed delete-input
default for full-length jobs, while CLI retention defaults remain unchanged.

- **Automated contracts:** 503 tests ran successfully, with one optional test
  skipped. Eleven new delivery tests cover canonical recovery data, TXT-only
  output, SRT/report reselection without inference, catalog metrics without a
  public report, failed/incomplete retention, ZIP verification failure and
  packaging warnings, audio export/codec failures, and external/bounded sources.
- **Actual media preparation, fake inference:** FFmpeg produced and probed MP3
  and FLAC from generated audio. Full queued-copy deletion was exercised only
  after selected deliveries existed and passed verification. These fixtures do
  not establish speech recognition accuracy.
- **Real GUI, mock inference:** the real WebView smoke check selected SRT alone,
  confirmed the new MP3/deletion defaults before overriding the selection,
  calculated a waveform, selected a bounded interval, verified automatic recap
  collapse and that the running page fits the default desktop viewport, and
  checked that the final folder contained SRT and verified intermediate ZIP,
  with no loose intermediate directory or unwanted TXT export.
- **Real CUDA inference:** an isolated queued copy of previously prepared real
  speech completed through the desktop bridge/worker with SRT, MP3, report
  Markdown and intermediate ZIP selected. The queued test copy was deleted after
  full success; its external source was compared byte-for-byte and preserved.
  Model and audio encoder placement were verified on CUDA0 with no fallback.
  Operational identifiers, transcripts and full evidence remain ignored.

Changing delivery selection preserves inference identity. Private canonical
exports and existing checkpoints/raw responses remain in process; completed
versions are not retroactively reorganized. Packaging warnings preserve loose
intermediates rather than hiding a failed ZIP operation. Layout acceptance is
for the default window size, not every possible resized desktop geometry.

## Desktop final review pass (2026-11-05)

Scope: read-through of the uncommitted desktop and delivery changes, run on
temporary directories and generated media only; the operational queue and
`config.local.toml` were not used, and no inference was started.

- **Re-run after review fixes:** 506 tests ran successfully with the same one
  optional skip (Silero model unavailable); the three additions cover stray
  non-event worker output and launcher messages for a missing configuration or
  missing optional UI dependency. Ruff lint and format checks, generated CLI
  documentation check and `git diff --check` passed. `git diff --check` only
  prints CRLF-normalisation warnings for four files; line endings were not
  rewritten.
- **Documentation checks:** local Markdown links and anchors, heading levels and
  code-fence balance passed for all tracked and new Markdown files. A privacy
  scan of changed and new documents found only generic placeholder media names
  and the public repository address.
- **Fixes:** a failed job's result screen showed the generic "at least one job
  failed" instead of the specific cause; the bridge now ignores worker output
  lines that are valid JSON but not events, instead of aborting the monitored
  run; a configured language missing from the six-entry dropdown (for example
  Japanese) left the selector blank and broke the recap, so it is now added as an
  option; launcher messages are clearer. The superseded keep-all wording in the
  [distribution design](distribution-design.md#proposed-requirements-and-coverage)
  GUI-04 row is labelled and points to GUI-08.
- **Not run:** `scripts/check-desktop-ui.py` was not re-run in this pass (the
  automated session was not permitted to launch it), so the real-WebView evidence
  above predates the final edits to `app.js`; the changed file passes
  `node --check` only. Re-run it before relying on the UI.

## Desktop review pass: retention origin, background waveform, review playback, MP4 preview (2026-11-05)

Scope: [GUI-02, GUI-04, GUI-07, GUI-08 and GUI-09](specification.md#guided-desktop-contract),
run on temporary directories and generated media only. The operational queue,
`config.local.toml` and the inference runtime were not used and no inference was
started. FFmpeg in this environment is a full build that includes `libx264`, `aac`
and `mov_text`; pywebview 6.2.1 with the Edge WebView2 runtime.

- **Headless suite:** `python -m unittest discover -s tests` ran 529 tests with
  one skip (the same optional dependency check as before); the 23 additions are
  in `tests/test_desktop.py`. They cover the retention default for an imported
  versus a found file and a picker result already inside the queue; waveform
  cancellation (a child process holding the source is killed, exits before
  cleanup and the source can then be renamed); that preview does not hold the
  operation lock while decoding and that `select`, `start` and `can_close`
  cancel and wait for it; the bounded in-memory cache; playback candidate order,
  rejection of a same-named file with different size or time, the bounded
  offset and end, and that review needs a completed or skipped job; the loopback
  server (200, 206 for all range forms, 416, wrong token, traversal and other
  paths 404, HEAD, POST rejected, release frees the file even while a response
  is half-read); and a unique media URL per review.
- **Real FFmpeg on generated video (mock inference):** 1080p downscales to 720p
  for a 2 s interval taken from a 4 s clip with H.264, AAC and a `mov_text`
  track; a 480p source stays 480p; an audio-only source and a cover-art MP3 are
  classified as having no video and the audio-only case is refused before any
  transcription; a failing render retains the source and the re-run restores the
  outputs without new model calls; `mp4` is not a default format and does not
  change the configuration fingerprint.
- **Real WebView, mock inference:** `scripts/check-desktop-ui.py` passed after
  these changes (log kept locally). It verified the keep default for a queued
  file, the disabled MP4 chip with a reason for audio-only media, the removed
  waveform button and `data-state="ready"`, the step-3 layout (five labelled
  groups, retention left of the wider outputs, one column at 820x700 with the
  footer reachable, no scrolling at 1100x830), the mock run, then the review step:
  cues rendered, the media element reached `readyState >= 1`, clicking cues
  seeked and the subtitle element showed the matching cue, and Back released the
  media element and the served file. A generated H.264 video also went through
  the MP4 chip, a rendered `video.mp4`, and review playback with the picture
  shown. The first runs of the check failed because a video review reused the
  previous audio review's URL and the WebView showed no picture; media URLs are now
  unique per review, which fixed it. The cause was inferred from that fix, not
  isolated further.
- **Static checks:** Ruff lint and format, the generated CLI reference check,
  `git diff --check` (CRLF warnings only) and `node --check` passed.

Not verified: real inference with the MP4 option, MKV/HEVC or other formats that
WebView2 may not decode (only the fallback logic is tested headlessly), the
native file picker and folder opening, a multi-hour render, other operating
systems, and window sizes other than 1100x830 and 820x700. The MP4 is a preview
render and was never compared with an original for fidelity. The indeterminate
progress bar during a long render has not been judged on a real long video.

## Desktop feedback round: reuse path, render progress, results and review layout (2026-11-05)

Scope: user hand-testing found that re-running identical settings with MP4 newly
added sat on "Starting the model" for about a minute: the reuse path emitted no
stage events, no progress and no timings. Fixed and re-checked on temporary
directories and generated media only; no real inference, no long render, and the
operational queue and `llama-server` were not touched.

- **Headless suite:** `python -m unittest discover -s tests` ran 542 tests, one
  skipped (the same optional dependency check). The 13 additions cover: a
  re-run with MP4 added reporting `job.reused`, `video_export` start/finish,
  progress events ending at 100 %, `reused` and timings in `job.finished`, and no
  new model calls; a failed render in the reuse path retaining the source;
  `-progress` parsing including a missing or invalid duration; the real render
  reporting throttled progress; the file inspection on good, empty, corrupt-JSON,
  corrupt-ZIP, size-mismatched and truncated-audio inputs; the queued-file
  outcome from job records; and stage collection, progress reset and reuse state
  in the bridge. A truncated MP3 keeps its header duration, so the check also
  reads the last packet time with FFprobe; on a generated 2-hour MP3 that check
  took about 1.2 s.
- **Real WebView, mock inference:** `scripts/check-desktop-ui.py` passed. It ran a
  generated 2-minute 1080p H.264 video twice with identical settings, the second
  time with MP4 added, and asserted that the reuse note and the "reused" rail
  marks appeared (and did not on the first run), that a determinate
  "Rendering preview video · N % · about ... remaining" detail was shown, that
  the result says the transcription was reused, lists this run's sections and the
  original run's timings, shows every file as verified or complete (including
  `video.mp4` as 720p), and states what happened to the queued file. It also
  asserted the review button style (filled green, larger than the secondary
  "Open result folder", contrast of at least 4.5:1), no page scroll for steps 5 and 6
  at 1100x830, and single columns at 820x700. Screenshots of steps 3, 5 and 6 were
  captured with the Win32 PrintWindow call and viewed
  (kept locally, not tracked).
- **Static checks:** Ruff lint and format, the generated CLI reference check,
  `git diff --check` (CRLF warnings only) and `node --check` passed.

Not verified: a render long enough to judge the progress bar over minutes, real
transcription followed by a reuse, results for a failed job in the real window,
other window sizes, and WebView2 decoding of files other than the generated
H.264 clips. The progress ETA is a straight-line estimate from elapsed time.

**Correction after review, same date:** reading the step 5 screenshot exposed a
wrong statement. Any kept full-length file was described as "only part of the
recording", because the check compared the structured `scope` record with a string,
and its test used the same wrong shape. The outcome now reads the record's real
shape and checks the files on disk first, so metadata from an earlier run of a reused
version cannot report a stale deletion. A media file whose last packet cannot be read
is now labelled "data end not checked". After this correction the full suite passed
(547 tests, 1 skipped), as did the static checks, and the real-WebView check passed
on the corrected source.

## Desktop round: GPU preview render, resource cards, results accordions, step 6 layout (2026-11-05)

Scope: user hand-testing asked to use the GPU for the MP4 preview, show resources
during processing, shorten the results screen and fix the overlap in step 6.
Checked on temporary directories and generated media only; no real transcription,
no long render, `llama-server` left running and untouched.

- **Encoder measurements (generated, not representative):** a 60 s 1080p `testsrc2`
  clip (flat synthetic colours) scaled to 720p, video only, on an RTX 2080 SUPER
  while `llama-server` held about 7.2 of 8 GiB of VRAM. `libx264` veryfast crf 26:
  2.9 s (21x realtime), 6.7 MB, PSNR 41.3 dB against the scaled source. NVENC
  (GPU decode and `scale_cuda`, p4, hq, VBR) at cq 28 / 32 / 34 / 36 / 38: about
  2.2 s each (27x), 15.2 / 9.0 / 7.1 / 5.5 / 4.5 MB, 43.6 / 41.6 / 40.7 / 39.8 /
  38.9 dB. NVENC with CPU decoding at cq 34: 1.8 s (33x), 7.7 MB, 41.5 dB. **cq 34
  was chosen** (about the size of the libx264 recipe at 0.6 dB lower PSNR); a
  cropped frame from both looked identical, which says little for flat synthetic
  content. A 480p source stayed 854x480 on the GPU path, and a generated 10-bit
  HEVC MKV rendered to 720p yuv420p H.264.
- **Headless suite:** `python -m unittest discover -s tests` ran 558 tests, one
  skipped; 11 are new. They cover the encoder chain order, `check_video_encoder`
  with only libx264, only NVENC and neither, that a machine with libx264 never
  runs the GPU probe, the cached real probe, fall-through using an injected bad
  encoder name (warning, no partial file, progress restarting, recorded path),
  failure of every path publishing nothing, no upscaling and the selected
  interval on every available path, the encoder path in events, the delivery
  record and the results row, and the new `utilization.encoder` field including
  `[N/A]`. The GPU cases ran here because NVENC works on this machine and skip
  where it does not.
- **Real WebView, mock inference:** `scripts/check-desktop-ui.py` passed with the
  resource monitor enabled. In the run that reused a transcription and added the
  MP4, the generated 1080p clip was rendered with NVENC and GPU decoding (the
  progress line and the file row said so); CPU, RAM, VRAM and GPU cards showed real
  values, the level bar was filled, and the GPU card showed encoder use during the
  render; the processing view, results, failed-results and step 6 all fit 1100x830
  without page scroll. Results: both accordions closed with summaries visible
  (status plus the queued-file outcome, and the total), the folder button
  outlined at 2 px with an icon; the failed-run screen was exercised by feeding
  the page a synthetic failure, not by a real failed job. Step 6: no overlapping
  blocks and the hint and buttons below the grid after loading, while a muted H.264
  video and an audio file played, narrow, and with a 400-word cue; the subtitle box
  kept the same height and nothing moved for an empty, short or very long cue.
  Screenshots of steps 3 to 6 were taken and viewed (kept locally).
- **Bug found by the check:** the audio data-end check (`ffprobe` packets) returned
  nothing for MP4 files because packets with side data end in an extra comma; it
  silently passed before and now reports correctly.

Not verified: a hardware failure on a real GPU mid-file (only an injected bad
encoder), NVENC session limits or out-of-memory, AMD or Intel hardware encoders
(deliberately unsupported), a long render, real transcription followed by a render,
and fidelity on real footage rather than generated clips.

## Limits and follow-up

llama.cpp audio support remains experimental and model text requires review.
Coarse subtitle timing, no verified diarization and no measured accuracy remain
explicit limitations. See [TODO](TODO.md), [runtime setup](runtime-setup.md),
[inference controls](inference-controls.md) and [compatibility](backend-compatibility.md).
Private operational files remain excluded from version control. Remote
publication has not occurred. The development state is recorded locally in Git.

Revision: 2026-10-02. Added the latest local full-run report status.
2026-10-02 (later): added segmentation/recovery tests, the bounded model
comparison and the completed full-recording run.

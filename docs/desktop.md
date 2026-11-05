# Guided desktop app

**Date:** 2026-11-05  
**Role:** User guide and implementation boundaries  
**Status:** Windows desktop preview; Linux/macOS inference and packaged releases
are not verified. See [verification](verification.md#guided-desktop-preview-2026-11-05).

## Reading paths

- [Start the app](#start-the-app)
- [Distribution status](#distribution-status)
- [Five steps](#five-steps)
- [Results screen](#results-screen)
- [Review with subtitles (optional step 6)](#review-with-subtitles-optional-step-6)
- [Interval semantics](#interval-semantics)
- [Output formats and intermediates](#output-formats-and-intermediates)
- [Implementation and limits](#implementation-and-limits)
- [Verification](#verification)

## Start the app

Complete the existing [README installation](../README.md#installation-windows--nvidia)
first. Add the optional UI dependency to the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[desktop]'
.\desktop.cmd
```

For an installed Python package, use `sbobinai-desktop --config config.local.toml`
or `python -m audio_transcript.desktop --config config.local.toml`.
Relative queue roots retain the CLI's current-working-directory semantics;
`desktop.cmd` starts in the project directory. Models and inference runtimes
are external resources configured through the normal explicit setup flow.
Opening the app never downloads them.

The console wizard and batch launchers remain available. The desktop entry point
is separate, and importing the core does not load pywebview. The preview UI is in
English; it does not yet inherit the console wizard's localized labels.

## Distribution status

**Clarification, 2026-11-05:** the built `.whl` is an installable Python package,
not a clean-machine desktop installer. It can be installed on another prepared
machine, but does not provision Python, FFmpeg, WebView2, models, the inference
runtime or machine-local configuration. `desktop.cmd` explicitly requires the
project's existing virtual environment. The repository checkout is not required
by the wheel itself; its dependencies and resource configuration are required.

The requested next deliverable is a Windows installer/bootstrap executable:
choose app and resource folders, show disk/download requirements, ask consent,
provision pinned resources, create configuration/shortcuts, run doctor and open
the existing GUI. This remains **proposed, not implemented**. A practical design
bundles a private Python interpreter plus Python dependencies with the app,
instead of installing Python globally or requiring pip on the destination PC.
The model and accelerator runtime can then be downloaded through setup and the
extended pinned manifest after explicit approval. WebView2 must be checked before
starting a GUI that depends on it. NVIDIA drivers remain a detected prerequisite,
not an automatically installed component.

Packaging must also replace the development worker invocation
`sys.executable -m audio_transcript.desktop.worker` with a bundled-worker entry
or executable dispatch mode, include lazy adapters/assets, remove checkout-script
assumptions and test lifecycle/configuration with no repository or development
environment present. Existing GUI/CLI tests do not establish clean-machine setup.
Build/signing/release and install-repair/uninstall behavior require separate
acceptance. See [D0/D1](distribution-design.md#d0--one-step-bootstrap-installer-prerequisite-of-d1).

## Five steps

Screenshots from one real run: [1 recording](images/step-1-recording.png),
[3 selection](images/step-3-selection.png), [4 progress](images/step-4-progress.png),
[5 results](images/step-5-results.png) and [6 review](images/step-6-review.png)
(transcript text and sensitive picture details blurred).

1. **Recording:** show supported media already in `input`. Automatically select
   the only recording; when several are queued, require a radio-button choice.
   **Import recording** opens a native picker and copies one file into the queue.
   Refresh to discover files added by another application. The external original
   remains untouched. An incomplete import is hidden from run/watch discovery,
   and a name collision creates another name rather than overwriting a file.
2. **Language & context:** choose the spoken language or automatic detection and
   optional context/glossary text (maximum 2,000 characters). These settings feed
   the existing pipeline and participate in job identity.
3. **Selection & storage:** drag start/end handles on a horizontal bar, or enter
   source seconds directly. The middle selection is highlighted, with discarded
   ends grey. The waveform is calculated automatically in the background as soon
   as a recording is selected (a spinner overlays it until it appears); it uses
   FFmpeg only, never the model, and works on audio and on the first audio stream
   of a video. It never blocks navigation. If it fails, a short message with a
   "Try again" link appears and the handles and numbers stay usable. Below it,
   a compact **File in input** control sits beside the outputs grouped by type
   (text, subtitles, audio, video, data and reports); the layout collapses to one
   column in narrow windows.
4. **Review & transcribe:** review the recording, language, context, interval,
   retention and model. Start explicitly. A separate Python worker reports
   preparation, inference, exports, progress, warnings and backend/device.
   The configuration recap is an accordion: open before confirmation, collapsed
   automatically when processing starts. The larger processing panel shows stages,
   chunk progress, ETA, speed, backend and live activity. A row of four cards
   (CPU, GPU, VRAM, RAM; label, large number and a level bar, tinted amber above
   90 %) stays visible for the whole run, with GPU temperature and, while the preview
   video renders, NVENC encoder use. Unavailable values show a dash, and a note says
   the figures are whole-system and include other applications. Keep the window open during processing; close is blocked while a run or import
   is active. There is no GUI cancel button in this preview.
5. **Your results:** facts about the work done (see [Results screen](#results-screen)).
   **Review with subtitles**, the prominent action for a completed or skipped job,
   opens the optional sixth step; **Open result folder** opens the exact version
   containing the exports, not the output root. Failed/incomplete jobs lead with
   the actual error and expose the intermediate folder when available. Another
   recording can be selected after completion.

The preview is intentionally **single-file**. The existing CLI supports batch
transcription; multiple-file GUI selection and shared batch settings are deferred.
The retention default (2026-11-05) depends on how the file
reached the queue. A recording **imported through the picker in this session**
defaults to **delete the queued copy after success**; the recap and final start
button name that action, and the original chosen in the picker is untouched. A
file that was **already in input** (including one picked from inside input)
defaults to **Keep**, i.e. archive in `processed`; choosing delete for it shows a
warning that it may be the only copy. The origin is remembered per session: after
a restart an earlier import counts as already queued. A future batch mixing both
kinds must default to keep. Any bounded selection forces retention; failures, incomplete jobs
and mock inference also retain input. Selected audio deliveries must pass codec
and duration verification before the queued copy can be deleted. The CLI default
remains keep-all, and its legacy keep-audio policy still requires verified FLAC.

## Output formats and intermediates

Step 3 has independent checkboxes for TXT, Markdown, SRT, VTT, transcript JSON,
MP3, FLAC, report Markdown and report JSON. All transcript/report formats plus
MP3 are initially selected; FLAC is optional. At least one output is required.
The result screen lists the files actually produced. MP3 is a compact, lossy
mono audio copy; FLAC losslessly encodes the normalized mono signal. Both use
the selected transcription interval and are verified with FFprobe.

`intermediate/` is a **cumulative readable transcript snapshot**, updated as
chunks complete, with progress reports. It is not the directory of individual
audio chunks, and it is not a translation. Chunk WAVs, checkpoints and raw model
responses live in `process`. Once the job succeeds, the desktop writes and
reads back `intermediate.zip` and checks that the loose files did not change
before removing the loose directory. If packaging fails, the folder remains
available and the UI reports the warning. Failed/incomplete jobs retain readable
intermediates for recovery.

The final version folder contains only the selected deliveries plus the
intermediate ZIP. Complete canonical text/report artifacts are preserved in
`process/<source>/<version>/artifacts`, with `delivery.json` recording the
delivery selection. Catalogs read the private report when the public JSON report
is disabled. Changing formats does not change inference identity: re-running
the same source/settings restores the requested exports from canonical data
without redoing model inference. Audio still requires an available source if a
new audio format has not already been generated. Existing versions are not
retroactively changed by this update.

**MP4 preview (optional, unchecked by default).** A 720p-or-smaller H.264/AAC
render of the selected interval (`video.mp4`, zero-based, never upscaled), made
after transcription. It is a preview for presentation, **not a preservation
copy**; originals are never replaced by it. **Encoder chain, first success wins:**
(1) NVENC with GPU decoding and GPU scaling, (2) NVENC with CPU decoding, (3) CPU
`libx264`. NVENC is used only if a real one-frame probe encode works (the encoder
list alone proves nothing), cached for the process; asking whether MP4 is possible
never opens a GPU session when `libx264` exists. A failed attempt, including a
failure in the middle of the file, leaves no partial file and falls through with a
visible warning that names the reason; progress restarts for the retry. Which path
ran is recorded in the delivery record and the `video_export` stage event, shown
in the progress line ("... · NVENC · GPU") and in the file row ("720p · NVENC, GPU
decode"). Only NVENC is supported; other hardware encoders are not attempted.
The canonical SRT is added as a soft, toggleable subtitle track (not burned in)
when available. It needs a recording with a video stream (an audio-only chip is
disabled with a reason) and an FFmpeg build with `libx264` or a working
`h264_nvenc` (plus `aac` and `mov_text`); both are checked before inference and
before Start. The render is verified (video stream, height,
duration) before a queued source may be deleted, and a failed render retains the
source. FFmpeg reports progress to a file about once per second, so the
processing panel shows a determinate bar: "Rendering preview video · 43 % · about
02:10 remaining · 14x realtime". The percentage and ETA are measured against the
selection duration. Audio exports are not instrumented (they are short).

The CLI can use the same controls explicitly:

```powershell
.\scripts\run.ps1 run --config config.local.toml --output-formats txt,srt,mp3 --archive-intermediate
```

## Interval semantics

`start_seconds` is a nonnegative offset into the first audio stream.
`max_duration` is the maximum number of seconds to process from that offset.
Selecting the full file clears any base-configuration duration limit in the GUI.

```powershell
.\scripts\run.ps1 run --config config.local.toml --start-seconds 30 --max-duration 90
```

This processes source seconds 30–120. **Exported timestamps start at zero within
the selection**; they remain coarse chunk boundaries. `audio_metadata` records
`selection_start_seconds` and `timestamp_origin = "selection"`. A nonzero offset
is also shown in scope metadata, the version name and the readable run report.
It participates in the configuration fingerprint, so ranges cannot reuse a
different interval's checkpoints or completed output. Zero preserves existing
full-run fingerprints. Every nonzero start or duration bound blocks archival
and deletion even if it happens to reach the source end.

The console wizard does not present start-offset controls and rejects a nonzero
`start_seconds`; use `run` or the desktop selector for that case.

## Results screen

**Status:** redesigned 2026-11-05 after hand-testing; it no longer repeats the
user's own settings. Only the recording name is kept from them.

- **Layout:** *Outcome* and *Time per section* are accordions, closed by default,
  stacked above the files. Each closed header carries a one-line summary so
  nothing important is hidden: the status plus what happened to the queued file
  ("Completed · Deleted from input after a complete transcription"), and the total
  time. For a failed or incomplete job *Outcome* starts open and the error stays at
  the top. **Files produced** is the main element and scrolls internally.
  **Review with subtitles** is the filled primary button; **Open result folder**
  has a green 2 px outline and a folder icon; *Choose another recording* stays quiet.
- **Outcome:** status, backend and device, speed in x realtime, chunks
  transcribed/silent/failed, recording and speech duration when known.
- **Queued file:** what actually happened (deleted, archived in processed or
  kept in input, and why: failure, partial interval, mock output, subfolder or an
  archive warning). It is read from the job's own records, not from the option
  chosen before Start.
- **Time per section:** inspect, audio extraction, speech detection, chunking,
  transcription, exports, audio, preview video, file management, packaging and
  catalog, plus the total. Sections that did not run are hidden. For a **reused**
  transcription (identical settings already finished) the list covers this run,
  and the original run's timings are shown separately with their date.
- **Files produced:** name, type, size and a mark computed when the screen opens,
  off the UI thread. **Verified** is used only for audio, video and ZIP files that
  were re-probed or read back now (audio and video must also contain stream data
  up to the declared duration; MP4 must be 720p or smaller). Text and report files
  are **Complete**: present, non-empty, equal in size to their canonical copy and
  parseable when JSON; this does not vouch for the transcript's content. A failure
  shows a red mark and reason and never breaks the screen. A summary line reads
  "All N files checked" or "N files need attention".

**Reuse and progress feedback.** If the same source and settings already have a
finished transcription, nothing is transcribed again; only newly requested
outputs (for example an MP4) are made. The processing panel then says "Reusing the
existing transcription: no model work needed", marks the model steps as reused,
names the earlier version in the activity log, and the results screen says that
the transcription was reused. Every delivery stage (audio, video, archive,
packaging, catalog) is reported in both the normal and the reuse path, and a bar
never keeps another phase's percentage.

## Review with subtitles (optional step 6)

**Status:** accepted 2026-11-05 as [GUI-09](specification.md#guided-desktop-contract).
After a completed or skipped job, **Review with subtitles** opens a player with
live subtitles, a clickable waveform of the selection and a clickable cue list.
Cues come from the canonical `transcript.json`, so the step works even when SRT
or JSON were not selected as outputs. The player tries, in order, the queued
recording, its archived copy, the original recorded at import, `video.mp4`,
`audio.mp3` and `audio.flac`; a source is used only when its size and
modification time still match what was transcribed, never a different file under
these subtitles. If the browser engine cannot decode one (for example some MKV
or HEVC files) the next is tried automatically and the one in use is named.
A bounded selection plays from its start offset and pauses at its end.
Everything is in normal flow, so blocks cannot overlap at any window size:
the **waveform** comes first at full width (time axis, playhead, played part
coloured, click to seek; hidden when no peaks exist); below it a smaller 16:9 player
(letterboxed for other ratios) with the live subtitle on the left and the cue list
on the right (it follows the left column's height and scrolls on its own); the
hint and buttons come last. The subtitle box has a **fixed height** (empty cue,
short cue and a very long one all keep it; long text scrolls inside) so it never
moves its neighbours. It fits 1100x830 without page scroll; below 900 px it
becomes one column, and for audio-only media there is no picture box and the
subtitle box is larger.

Subtitle timing follows audio chunks of up to about 15 seconds; it is not word
alignment. Leaving the step releases the media and the served file, so playback
never blocks a later archive or deletion. Media is served by a loopback-only
HTTP server with a random token, one file at a time, with Range support.

## Implementation and limits

The acceptance contract is [GUI-01–GUI-09](specification.md#guided-desktop-contract).
The original alternatives and current scope are recorded in
[distribution design](distribution-design.md#guided-desktop-workflow-review).

| Component | Responsibility |
| --- | --- |
| `desktop/assets/` | Local HTML/CSS/JavaScript, five-step presentation, range selection and event projection; no third-party web assets |
| `desktop/bridge.py` | Narrow native bridge: queue selection with origin-based retention default, recap validation, import, background waveform with cancellation and cache, worker lifecycle, result-folder opening and review playback candidates |
| `desktop/results.py` | Honest file checks and queued-file outcome for the results screen |
| `desktop/playback.py` | Loopback token-protected single-file media server with Range support for the review step |
| `desktop/worker.py` | Same processor/backend/exporter, pipeline, process lock, session, JSONL events and checkpoints as the CLI |
| `application/delivery.py` | Canonical recovery artifacts, selected public deliveries, verified audio prerequisites and intermediate ZIP |
| `desktop/runtime.py` | Windows source checkout reuses the established ownership-safe launcher; other installations use a worker-owned model process and refuse occupied ports |
| `adapters/desktop_media.py` | Atomic import publication, bounded-memory cancellable waveform preparation and platform folder opening |

Waveform preparation uses temporary low-rate PCM on disk and returns 800 peak
bins; memory does not scale with recording duration. It runs automatically
in the background, in one cancellable FFmpeg process at a time, and is cached in
memory (8 entries, by source identity) with no persistent cache. Selecting
another recording, starting a run or closing the window cancels the decode and
waits for FFmpeg to exit. Imports require hard-link support on the
queue filesystem for atomic no-overwrite publication; unsupported filesystems
fail while leaving the external source intact.

The worker checks model and audio-encoder placement in startup logs before
inference. The configured backend/fallback policy is preserved, warnings are
visible and runtime details are recorded in normal reports/events. No new
platform/backend is claimed supported based on GUI portability alone.

Deferred: video thumbnails, GUI batch, graceful GUI
cancellation, localized UI, packaged installers/portable bundles, code signing,
Metal backend selection and real Linux/macOS acceptance. This preview runs from
the installed project environment rather than a standalone executable bundle.

## Verification

Headless contract checks include retention defaults by origin, waveform
cancellation and cache, playback candidates, Range handling, MP4 rendering
(downscale, no upscale, interval, soft subtitles, audio-only rejection and
failed-render retention), import failure, collision and source
preservation; recap validation; intervals and identity; partial-retention rules;
worker failure recovery; and exact result-folder selection. Generated media
tests verify actual FFmpeg sample extraction and waveform peaks separately from
mock inference. The following optional test opens the real WebView and exercises
all five screens and the optional review step with generated recordings and
explicitly mock output, including the automatic waveform, the MP4 chip, the
step-3 layout at the default and minimum window sizes, and review playback of
audio and H.264 video:

```powershell
.\.venv\Scripts\python.exe scripts/check-desktop-ui.py
```

See [dated evidence](verification.md#guided-desktop-preview-2026-11-05) for the
separate real-inference check and its limits.

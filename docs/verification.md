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
success. The 180-second pipeline test remains complete. See the
[latest execution review](execution-review.md) for severity and recovery actions.

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
rounding), recorded as RUN-05 in the [execution review](execution-review.md).
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

## Limits and follow-up

Qwen2-Audio/llama.cpp audio remains experimental and model text requires review.
Coarse subtitle timing, no verified diarization and no measured accuracy remain
explicit limitations. See [TODO](TODO.md), [runtime setup](runtime-setup.md),
[inference controls](inference-controls.md) and [compatibility](backend-compatibility.md).
Vision/Framework originals are archived unchanged; working references and specs
were semantically restructured. Sol reviewed source coverage and navigation.
Private operational files remain excluded from version control. Remote
publication has not occurred. The development state is recorded locally in Git.

Revision: 2026-10-02. Added the latest local full-run report status and linked
the privacy-safe execution review.
2026-10-02 (later): added segmentation/recovery tests, the bounded model
comparison and the completed full-recording run.

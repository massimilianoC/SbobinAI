# Verification evidence

Date: 2026-10-02. Windows x64, Python 3.14.7, FFmpeg 9.0.2.
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
chunks: Qwen repeated text until max_tokens=1024. The intermediate report is
failed and includes the error; the original remains in input. The 57 completed
checkpoints are retained. Full-recording completion and model stability are
follow-up work, not a claimed success. The 180-second pipeline test remains complete.

Inspect `output/<job-id>/intermediate/report.json` for live progress/failure and
`output/<job-id>/report.json` for final success. Execution logs remain under
`.local/pipeline-runs`. Resume with the same command and configuration, without
force, after correcting a failure. Completed chunk checkpoints are retained.

## Limits and follow-up

Qwen2-Audio/llama.cpp audio remains experimental and model text requires review.
Coarse subtitle timing, no verified diarization and no measured accuracy remain
explicit limitations. See [TODO](TODO.md), [runtime setup](runtime-setup.md),
[inference controls](inference-controls.md) and [compatibility](backend-compatibility.md).
Vision/Framework originals are archived unchanged; working references and specs
were semantically restructured. Sol reviewed source coverage and navigation.
Private operational files remain excluded from version control. Remote
publication has not occurred. The development state is recorded locally in Git.

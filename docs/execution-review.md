# Latest execution review: full-recording transcription

**Date:** 2026-10-02 (second review of the day)  
**Role:** Review of real CUDA execution evidence after the segmentation and recovery fixes  
**Status:** Full 7,546-second recording **completed** through the production pipeline  
**Scope:** Evidence from local run logs, job reports and GPU telemetry. Private
media, transcript text, file names and machine paths are not included.

## Reading paths

- [Summary](#summary)
- [Root causes of the earlier failure](#root-causes-of-the-earlier-failure)
- [Findings and status](#findings-and-status)
- [Model comparison](#model-comparison)
- [Full-recording run](#full-recording-run)
- [Open items](#open-items)
- [Previous review](#previous-review-2026-10-02-before-the-fixes)

## Summary

The full recording that previously failed at chunk 58/755 now completes in
**4 min 22 s** wall time (preparation 49 s, inference 3 min 08 s) with 581
speech chunks, no fallbacks, no failed chunks and about 3.8 GiB of GPU memory
for the model server. Two changes made the difference:

1. **Pipeline:** speech-only chunks of at most 15 s cut at pauses (Silero VAD on
   CPU), a degenerate-output fallback ladder instead of identical retries, and an
   `incomplete` status instead of aborting the whole job.
2. **Model:** Qwen3-ASR-1.7B Q8_0 (dedicated speech recognition, LLM decoder)
   replaced Qwen2-Audio-7B-Instruct Q8 (general audio chat model) on the same
   llama.cpp CUDA server, with the configured language forced.

Quality evidence is a qualitative review against audio, not a measured word
error rate. Remaining visible errors are mostly proper nouns and product names.

## Root causes of the earlier failure

| Cause | Evidence from the failed run | Fix |
| --- | --- | --- |
| Fixed 10 s chunks without speech detection | About 34 of the first 38 chunks were near-silent or noise (about −55 to −84 dBFS); words were split across chunk edges. | SEG-01/02: Silero VAD, cuts at pauses, ≤ 15 s. |
| Silence sent to a chat model | 14 of the first 38 chunks returned the prompt's own language sentence; about 15 returned invented stock sentences. These were saved as valid transcript. | SEG-01 (no silence sent); removal of the separate language sentence; prompt-echo detection. |
| Deterministic identical retries | The failing chunk was retried with temperature 0 and seed 42; the server answered from its prompt cache (`cache_n` 835 of 836 tokens), so every retry failed identically. | REC-01/02: transient vs degenerate errors; temperature then split ladder. |
| One bad chunk aborted 755 | Speech chunk 57 was summarized instead of transcribed, then looped to 1,024 tokens (12 s); the job stopped with 698 chunks never attempted. | REC-03: record failed chunk, continue, end `incomplete`. Dynamic token cap (about 170 tokens for 15 s). |
| Model choice | llama.cpp documents poor results for quantized Qwen2-Audio; it is an instruction/chat model. | Qwen3-ASR-1.7B on the same runtime (see comparison). |
| Encoder padding | llama.cpp pads every clip to a 30 s window (constant 836 prompt tokens per 10 s chunk). | Informational: chunk length does not reduce encoder cost; ≤ 15 s keeps generation short. |

**Do not resume the failed job:** its 57 completed checkpoints contain the prompt
echoes and invented sentences listed above. The new settings give the recording
a new job identity; the failed job is retained only as evidence.

## Findings and status

| ID | Previous status | Current status | Evidence |
| --- | --- | --- | --- |
| RUN-01 | Blocker — unresolved | **Resolved** | Full recording completed: 581/581 chunks ok, 0 fallbacks, 0 splits, exit code 0. |
| RUN-02 | High risk — unresolved | **Mitigated** | Silence is not sent to the model; a sub-second noise blip now yields no speech or a filler sound instead of an invented sentence. Short isolated chunks remain the riskiest input. |
| RUN-03 | Open measurement gap | **Open** | No reviewed reference transcript or word error rate yet; chunk times are speech-chunk boundaries, not word alignment. |
| RUN-04 | Corrected for bounded test | **Superseded** | Qwen3-ASR uses its own output protocol, parsed strictly; JSON mode remains for Qwen2-Audio. |
| RUN-05 | — | **Resolved** | Archive tolerance: frame-exact analysed duration (…9627 s) vs ffprobe's millisecond-rounded duration (…963 s) differed by 0.3 ms, so the completed source was kept in input with "only part processed". Tolerance is now max(1 sample, 0.05 s); bounded runs are still rejected. The second full run archived the source. |
| RUN-06 | — | **Resolved** | Flat `output/<job-id>` folders made versions of one recording indistinguishable. Now `output/<source>/<version>/` with `source.json`, `output/catalog.json` and `run-report.md`; 8 legacy jobs were migrated by logged moves, none deleted. |
| RUN-07 | — | **Correction** | The earlier claim that `tests/test_core.py` never ran was wrong: a `load_tests` hook already executed its tests. They are now plain `unittest.TestCase` classes. |

## Model comparison

Same production pipeline (`scripts/process-input.ps1`), same first 600 s of the
recording (177 s of speech in 24 chunks), one model loaded at a time, CUDA0.
VRAM excludes the desktop baseline of about 3.0 GiB.

| Model | Server VRAM | Inference | Real-time factor | Notes |
| --- | --- | --- | --- | --- |
| Qwen2-Audio-7B Q8 | ~10.7 GiB | 20.4 s | 0.034 | One invented sentence on a noise blip; non-Latin output on a short greeting. |
| Qwen3-ASR-1.7B Q8_0, auto language | ~3.7 GiB | 4.2 s | 0.007 | More literal; greeting mislabelled as Chinese. |
| **Qwen3-ASR-1.7B Q8_0, forced Italian** | **~3.7 GiB** | **2.8 s** | **0.005** | **Selected.** Greeting correct. |
| Qwen3-ASR-1.7B bf16, forced Italian | ~5.3 GiB | 5.3 s | 0.009 | Text identical to Q8_0 except punctuation. |

Details and the decision record: [backend compatibility](backend-compatibility.md#model-selection-on-the-llamacpp-runtime-2026-10-02).

## Full-recording run

| Item | Value |
| --- | --- |
| Entry point | `scripts/process-input.ps1` (doctor + run, console transcript kept locally) |
| Analysed audio | 7,546 s (2 h 05 min 46 s) |
| Speech sent to the model | 6,874 s (91 %), 581 chunks, min/avg/max 0.7/11.8/15.0 s |
| Preparation | 49.4 s (FFmpeg extraction of a ~9.6 GB video + Silero VAD on CPU + slicing) |
| Inference | 188.5 s (3 min 08 s), real-time factor 0.025 including per-chunk overhead |
| Total wall time | 262.4 s (4 min 22 s), about 29× faster than real time |
| Outcome | 581 ok, 0 no speech, 0 failed, 0 fallbacks, 0 splits |
| Tokens | 24,743 generated, max 77 per chunk, p95 63 |
| GPU | Peak ~3.8 GiB above baseline, mean utilization 36 %, peak power 143 W |
| Archive | First run: kept in input (RUN-05). Second run after the fix: archived to `processed/<source>/` |

Per-chunk inference averaged about 0.3 s; the rest of the per-chunk time is
spent rewriting intermediate exports, whose cost grows with the number of
chunks (see [TODO](TODO.md)).

### Second full run (versioned layout, confidence enabled)

Same model and settings plus token log-probabilities: 581/581 chunks ok,
preparation 48.0 s (audio extraction 6.9 s, Silero speech detection 40.3 s,
slicing 0.5 s), inference 4 min 42 s, total 5 min 30 s, archived on success.
Uncalibrated confidence (mean token probability): mean 0.968, 10th-percentile
chunk 0.936, no chunk below 0.5. Requesting log-probabilities made inference
about 35–50 % slower; it can be disabled with `--no-collect-logprobs`.

### Third full run (one command, two parallel requests)

`scripts/process-input.ps1` with no arguments, file placed back in `input/`:
the script detected that the owned server ran with 1 slot, restarted it from
a local server profile (2 slots, context 8192; since replaced by the `[server]` table of `config.local.toml`), ran `doctor`, transcribed, wrote
all exports and reports and archived the source — no manual step.

| | Sequential (2nd run) | 2 parallel requests (3rd run) |
| --- | --- | --- |
| Inference stage wall time | 4 min 42 s (281.5 s) | **3 min 24 s (203.9 s)**, −28 % |
| Per-request time mean / p95 | 0.4 / 0.6 s | 0.7 / 1.0 s (GPU shared by two requests) |
| Effective concurrency | 1.0× | 1.93× |
| Total wall time | 5 min 30 s | 4 min 25 s |
| Mean GPU utilization | — | 22 % |
| Transcript lines differing | — | 52 of 581 (punctuation, a few words) |

Parallel requests are faster but **not output-identical** to sequential greedy
decoding (batched kernels). They are therefore recorded as a distinct version.
Use `parallel_requests = 1` when byte-reproducibility matters.

### Guided run with context (wizard, 2026-10-02)

`process-input.ps1 -Interactive -AnswersFile …` (language `it`, a one-line
context naming the products discussed, full file) through the official
launcher. Effect of the context on proper nouns, same model and chunks:

| Spelling in the transcript | Without context | With context |
| --- | --- | --- |
| Correct product name | 33 | **76** |
| Wrong variants (three spellings) | 16 | **0** |
| Capitalized second product name | 15 | **42** |

Two sub-second noise chunks (0.7 s, 0.8 s) returned the context text itself;
prompt-echo detection rejected them and the job ended `incomplete` instead of
storing invented text. A new ladder rung (retry without context) was added; a
rerun with identical answers resumed only those two chunks, which now yield a
short filler sound, completed the job in 2.8 s and archived the source.

## Open items

1. Measure word error rate on a reviewed excerpt; add a terminology/context
   prompt (Qwen3-ASR accepts system context) for recurring product names.
3. Short isolated chunks: minimum context duration or merging into neighbours.
4. Intermediate-export frequency; optional parallel slots (`-Parallel`), given
   36 % mean GPU utilization.
5. Decide which alias the shared server should expose to other services.

## Previous review (2026-10-02, before the fixes)

The earlier review recorded the full run as `failed` at chunk 58/755 after 57
chunks, with repetitive generation reaching `max_tokens=1024` (RUN-01, blocker),
prompt echoes or fabricated text on near-silent opening chunks (RUN-02, high
risk), no measured accuracy (RUN-03) and the bounded JSON-mode correction
(RUN-04). It recommended resuming the same job after a validated fix; this
review supersedes that recommendation because the retained checkpoints contain
fabricated text. The full previous text is preserved in the local review
archive.

## Revision record

- **2026-10-02:** initial privacy-safe review of the failed full run.
- **2026-10-02:** root causes, model comparison, completed full run, new
  findings RUN-05 to RUN-07 and superseded resume recommendation.

# Follow-up work

Updated: 2026-10-02 (segmentation and recovery implemented; see execution review).

## Immediate execution follow-up

**Resolved 2026-10-02 (implementation):** the fixed 10 s chunking, identical
deterministic retries and whole-job abort that stopped the full recording are
replaced by Silero VAD pause-aware chunks of at most 15 s, a degenerate-output
fallback ladder and an `incomplete` job status
([inference controls](inference-controls.md), [execution review](execution-review.md)).
The 57 checkpoints of the failed run contain prompt echoes and invented text
for silent chunks: **do not resume that job**; the changed settings give the
recording a new job identity.

Open items:

- **Short isolated chunks (≤ 1 s).** Very short speech blips are the riskiest
  inputs (wrong language, filler words). Evaluate a minimum chunk duration that
  borrows surrounding context, or merging blips into neighbours, against
  reviewed audio. Forced language already fixes the language error for Qwen3-ASR.
- **Accuracy measurement.** Review a reference excerpt by hand and measure word
  error rate for the chosen model; current evidence is qualitative.
- **Intermediate export cost.** Resolved: exports are throttled by
  `intermediate_interval_seconds` (default 10 s); the checkpoint is still written per chunk.
- **Parallel slots.** Evaluated: 2 requests cut inference wall time by 28 % but change
  about 9 % of lines (punctuation/minor words); kept as a distinct, opt-in version.
- **Run-report timing gaps.** Resolved: stages are recorded before the final render,
  both inference figures are labelled, and sub-0.1 s stages are shown in ms.
- **Log-probability overhead.** Confidence collection slowed inference by about
  35–50 %; measure and decide the default for long recordings.
- **Terminology context.** Implemented through the wizard/`prompt` (Qwen3-ASR system
  context): wrong product-name spellings dropped from 16 to 0 on the reference
  recording. A human/LLM review stage for remaining errors stays proposed below.
- **Legacy flat outputs without metadata** (`output/central_sample_*`, old
  sampler folders) were left untouched by the migration; decide whether to
  archive them manually.
- **Restore or retire the Qwen2-Audio server.** The shared port now serves the
  model selected for production; document the alias other services should use.

## Later extensions

### Contextual review stage (requested; proposal)

Proper nouns and domain terms are the main remaining errors. Add a separate,
explicit stage in which an LLM or a human reviews the transcript with context
(glossary, topic, low-confidence chunks from the checkpoint) and writes a
**derived** corrected file beside the version; the raw transcript and its
provenance stay unchanged, and the stage records model/prompt/reviewer.

**Summary with the loaded ASR model: not feasible (checked 2026-11-05).** A text-only probe of Qwen3-ASR-1.7B Q8 on llama.cpp (CPU runtime, no projector, one synthetic Italian meeting text, 8 requests at temperatures 0 and 0.3, four prompt variants including a system prompt and factual questions) produced no summary or answer: outputs were `language Italian<asr_text>` followed by the input repeated, or an empty `language None<asr_text>`. The user chose to drop the idea for now; a summary would need a separate text model as an explicit derived stage, and is not implemented.

### Distribution (requested; proposal)

Packaging, first-run resource downloads, HTTP API, storage adapters, UI and
cloud options are designed in [distribution design](distribution-design.md).

**2026-11-05:** a single-file [desktop preview](desktop.md) is implemented with
five screens, queue/import, start/end selection, optional waveform, progress and
exact result-folder access. Remaining desktop work: native picker/open-folder
acceptance, localization, GUI batch, graceful cancellation, video thumbnails,
standalone packaging and real Linux/macOS inference acceptance. These remain
separate from the implemented command/environment launcher. Standalone packaging
is planned in the [installer plan](installer-plan.md) (proposal, not implemented).

**Queued-file retention default (user request 2026-11-05; confirmed by the
user and implemented 2026-11-05).** GUI-08 now follows this table, tracked per
session in the desktop bridge (a file imported in an earlier session counts as
already queued, the safe direction); AGENTS.md and the desktop guide were
updated. The default depends on how the file reached the queue:

| Input | Default | Reason |
| --- | --- | --- |
| File already present in `input` | Keep | The user did not ask to remove it, and it may be the only copy. |
| File imported through the picker, which copies it into the queue | Delete the queued copy | The external original remains untouched. |
| Future GUI batch mixing both kinds | Keep | The safer choice applies to the whole batch (GUI batch does not exist yet; recorded in the spec only). |

The user can still change the choice before starting, and all existing guards
still apply: bounded, failed, incomplete and mock jobs retain input. An
alternative the user considered was an empty field requiring an explicit choice;
the origin-based default was preferred so that users do not have to interact with
an option they did not ask about.

### OBS-01 — Execution and resource report (largely implemented 2026-10-02)

**Status:** per-attempt metrics, stage timings, Markdown run reports, a
versioned event stream (`events.jsonl`, `status.json`, `pipeline.log`) and
sampled system/GPU resources are implemented — see [events](events.md).
Remaining: model/runtime provenance hashes in the report and GPU attribution
to the llama.cpp process. The original request follows.

Add a versioned JSON report beside each job's final and intermediate exports.
The report should describe the run from queue discovery to completion or
failure, so an operator can compare jobs and diagnose a slow or incomplete run.

**Proposed fields**

- Run/job start, end, elapsed wall time, outcome, failure/retry counts and
  configuration fingerprint.
- Time spent discovering/queueing, probing, decoding/extracting, preparing
  chunks, loading/waiting for the model, each model request, exporting and
  archiving. Include per-chunk index, source offset, duration, elapsed time,
  attempts, token usage and outcome where the backend supplies them.
- Model name/revision/quantization and file hash when available; runtime/backend
  name/version, endpoint type and whether execution was local or remote. Keep
  endpoints free of credentials and avoid exposing private source paths publicly.
- GPU make/model, selected device, model/projector offload, available/used memory,
  utilization and sampling interval during execution, when the platform exposes
  these metrics. Mark unavailable or sampled values explicitly; do not imply
  they represent exact per-process GPU use when other services share the device.
- Software versions, audio preparation settings, chunk count, prompt provenance,
  response mode, seed, temperature and relevant generation limits.

**Implementation guidance**

Keep a stable schema version and store raw event measurements in process/job
state. Generate a concise summary JSON in output after success or failure; update
intermediate status without losing prior events. Use a monotonic clock for
durations and UTC timestamps for correlation. Gather GPU metrics through an
optional NVIDIA telemetry adapter with a graceful unavailable state. For a
remote backend, collect only provider-reported timings and usage; label local
client wall time separately. Keep telemetry optional, private, and independent
of transcript text. Test mocked metrics plus an unavailable-GPU path before
relying on live hardware measurements.

**Acceptance**

- A report exists for completed and failed jobs and identifies schema/version,
  backend/model, execution origin and end-to-end duration.
- Stage/chunk timings and retry outcomes reconcile with pipeline progress and
  checkpoint state; absent measurements are null/reasoned, never fabricated.
- GPU metrics include source, sampling interval and availability; local/remote
  status is clear, and secrets/transcript/audio bytes are excluded.
- A resumed run preserves prior attempt history while distinguishing the new
  attempt and cumulative versus per-attempt durations.

- Measure transcription accuracy against reviewed reference speech; record error
  rate, domain terminology and effect of 10/20/30-second boundaries.
- Compare prompt variants using retained raw responses and explicit settings;
  add a separate evaluation artifact rather than rewriting raw transcripts.
- Add verified speech alignment and diarization only through explicit providers.
- Define separate schemas/stages for summary, sound events and rich analysis.
- Evaluate automatic Windows boot/service registration if needed by operators;
  shared runtime files, local API and lifecycle commands already exist.
- Consider application HTTP API, scheduling and parallelism after resource tests.
- Migrate legacy metadata from old input job folders safely when encountered;
  new jobs already store all metadata under process.
- Run configured remote CI; local checks do not establish remote execution.

Every follow-up must update affected Vision/Framework/specification documents
according to [documentation policy](documentation-policy.md), preserving sources.

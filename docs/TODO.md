# Follow-up work

Updated: 2026-10-02. Prioritize the documented real CUDA execution before extensions.

## Immediate execution follow-up

The structured 180-second test completed. The full recording stopped at chunk
58/755 because Qwen generated repetitive text to the token limit. Keep its
intermediate output and 57 completed checkpoints. Diagnose the retained raw
response; evaluate shorter fallback windows or repetition controls with explicit
recorded settings. Resume the identical pipeline only after validating the fix.
Do not present the partial transcript as complete. A dedicated ASR alternative
is an explicit model choice if Qwen remains unstable.

## Later extensions

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

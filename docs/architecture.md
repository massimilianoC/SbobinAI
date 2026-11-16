# Architecture

The composition root is the CLI. Domain dataclasses describe media, chunks,
speech activity, segmentation settings, prepared audio, segments and
transcripts; protocols describe speech detection, media preparation (including
splitting a chunk for recovery), inference and exporting. Chunk planning from
speech probabilities is pure domain logic (`domain/segmentation.py`); detectors
(Silero ONNX, FFmpeg energy) are lazy adapters. Backends signal
`TransientBackendError` or `DegenerateOutputError`, and the application owns
the recovery ladder and the `incomplete` job status. Application services orchestrate these contracts. Adapters own
FFmpeg subprocesses, the local llama.cpp HTTP client, filesystem persistence, and
output serialization. The core has no mandatory third-party runtime dependency.

The source media remains in place. Operational directories contain local paths,
metadata, model responses, and derived audio and are excluded from Git. Public
documentation uses generic examples. Shared agent policy lives in root AGENTS.md;
CLAUDE.md and GEMINI.md redirect to it.

Job folders are grouped per source (content-based source key) and per version
(configuration fingerprint) by `application/layout.py`; `application/catalog.py`
derives `source.json`/`catalog.json` from job metadata and reports, and
`application/migration.py` moves legacy flat folders with a move log.
`application/reporting.py` renders the Markdown run report.

Checkpoints separate audio preparation from inference completion. A configuration
fingerprint prevents a bounded run or different model from being mistaken for a
full completed run. Atomic writes and a process lock protect local job state.
Sequential orchestration is intentional; backend replacement does not imply
that the selected runtime can execute multiple GPU requests concurrently.

The llama.cpp model server lives outside the application's virtual environment.
Multiple local tools can reuse its loopback API and shared binaries. The CLI
owns job state; server lifecycle scripts own only their recorded server process.
Neither stopping a client nor closing a backend unloads the shared server.

Progress is reported through the same contracts rather than by parsing text.
`MediaProcessor.prepare` and `SpeechDetector.detect` accept an optional progress
callback that receives measured values only (FFmpeg's own progress file, windows
scored by the detector, chunks written); a detector without the parameter keeps
working. The pipeline turns these into `stage.progress` events, and every
consumer (live console, JSONL stream, desktop) reads the event stream described in
[run events](events.md). Transcript text never travels in events.

The desktop app (`desktop/`) is another consumer, not a second pipeline. Its
pywebview page talks to a narrow bridge; each run happens in a separate worker
process that uses the same processor, backend, exporter, process lock and
checkpoints as the CLI and writes JSON events to its standard output. The bridge
projects those events into a small live state (chunk plan and states, preparation
steps, heartbeat) and can stop the worker's whole process tree when the user
confirms closing during a run. See the [desktop guide](desktop.md#implementation-and-limits).

See [backend compatibility](backend-compatibility.md) for historical and current
interfaces, and [runtime setup](runtime-setup.md) for deployment and lifecycle.

Revision 2026-11-16: progress callbacks, the event-driven desktop consumer and its
worker lifecycle. Revision 2026-10-02: added speech detection, chunk planning,
split and failure-class contracts.

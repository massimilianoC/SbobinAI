# Implementation plan

Planning date: 2026-10-02. Planner: Sol.

1. Establish the English specification, domain protocols, agent policy, and Git
   exclusions. Preserve original analysis documents and media.
2. Delegate CLI/configuration/job orchestration and media/inference adapters to
   separate Luna agents with disjoint file ownership and shared contracts.
3. Integrate validated exporters, dependency diagnostics, checkpoint recovery,
   sequential batches, and stable-file watching.
4. Install FFmpeg, install the local Python package in an isolated environment,
   run meaningful unit and synthetic integration tests, and inspect output formats.
5. Probe the actual input and prepare a short bounded audio sample. Verify model
   inference only after a compatible runtime and Qwen2-Audio weights are available.
6. Review acceptance evidence, document unresolved limitations, and provide
   reproducible commands. Do not publish private assets or commit automatically.
7. Complete the [documentation maintenance procedure](documentation-policy.md):
   preserve new supplied sources, restructure affected vision/framework/spec
   documents, add semantic hints, reconcile statuses and verify navigation.

Tests should exercise observable behavior rather than mirror implementation.
The status and validation evidence belong in [verification](verification.md).

The current production test entry point is `scripts/process-input.ps1`.
It checks dependencies, logs the CLI run, and exposes intermediate/final exports.
Runtime is shared NVIDIA CUDA. Deferred work is tracked in [TODO](TODO.md).

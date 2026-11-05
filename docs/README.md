# Documentation index

The current project contracts and evidence are organized by purpose. Start with
the specification for behavior, compatibility for runtime setup, and verification
for actual outcomes. All public documents use generic examples.

| Document | Role | Read it for |
| --- | --- | --- |
| [Project README](../README.md) | Operational entry point | Setup, CLI commands, and the support status (verified, beta, untested, planned). |
| [MVP specification](specification.md) | Acceptance contract | Workflow, required behavior, scope and document maintenance. |
| [Architecture](architecture.md) | Design decisions | Domain interfaces, orchestration and adapter boundaries. |
| [Backend compatibility](backend-compatibility.md) | Runtime evidence | Verified interfaces, constraints and unresolved compatibility. |
| [Shared runtime setup](runtime-setup.md) | Operational contract | Global binaries, local API, model files, startup and shutdown. |
| [Inference controls](inference-controls.md) | Model task contract | Segmentation (SEG), recovery ladder (REC), response modes, metrics and experiment identity. |
| [Command-line reference](cli-reference.md) | Generated manual | Every command, option, configuration key (with job identity), exit status, JSON error code and agent recipe; regenerate with `scripts/gen-cli-docs.py`. |
| [Run events and monitoring](events.md) | Event schema and files | Live console modes, `events.jsonl`, `status.json`, resource monitor and the web UI/SSE design notes. |
| [Verification](verification.md) | Dated execution evidence | Automated checks, real audio preparation and inference status. |
| [Licensing review](licensing-review.md) | Third-party license audit | What can be sold, what must be credited, FFmpeg/CUDA/ownership caveats. |
| [Multiplatform porting](multiplatform-porting.md) | Research, spec; Windows backends implemented | CUDA/Vulkan/CPU measurements, AMD, Intel, Apple Silicon variants, runtime selection spec. |
| [Distribution design](distribution-design.md) | Proposal; desktop subset implemented | Portable package, first-run downloads, HTTP API, storage, UI and cloud options; [guided desktop workflow review](distribution-design.md#guided-desktop-workflow-review), GUI-01–GUI-06. |
| [Guided desktop app](desktop.md) | Windows preview guide | Five-step pywebview app plus optional review playback, queue/import, automatic waveform/range, selected transcript/report/audio/MP4-preview deliveries, verified intermediate ZIP, origin-based retention, focused live progress and installer boundary. |
| [Desktop screenshots](images/) | README images | Five screenshots of one real desktop run (steps 1, 3, 4, 5, 6); transcript text and sensitive picture details are blurred. |
| [Installer plan](installer-plan.md) | Proposal, not implemented | Single operational plan for Windows, macOS and Linux installers: package contents and sizes, hardware validation, first-run wizard, phases, effort and open decisions. |
| [Documentation policy](documentation-policy.md) | Editorial contract | Source preservation, semantic structure and completion checks. |
| [TODO](TODO.md) | Follow-up scope | Deferred extensions, including full execution timing and GPU telemetry (`OBS-01`). |
| [Contributor instructions](../AGENTS.md) | Shared agent policy | Development, privacy, documentation and collaboration rules. |
| [Changelog](CHANGELOG.md) | Release notes | Notable changes by version. |
| [Legal documents](legal/) | Licensing | [Commercial license](legal/COMMERCIAL-LICENSE.md), [contributor license agreement](legal/CLA.md) and [third-party notices](legal/THIRD_PARTY_NOTICES.md). |
| [Contributing](../.github/CONTRIBUTING.md) and [security policy](../.github/SECURITY.md) | Community files | How to contribute and report vulnerabilities. |

## Document roles and access

Private working material (vision notes, supplied references, real recordings,
transcripts and run outputs) stays in ignored local folders and is never
committed; the public index does not depend on it.

Semantic IDs identify intent and obligations. Status tables distinguish requested
capabilities, implemented behavior, verified results and future extensions.
At session completion, follow the documentation policy to keep these aligned.

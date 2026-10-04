# Contributor instructions

Use `docs/specification.md` as the acceptance contract. Write source code, tests,
specifications, and public documentation in English. Preserve supplied vision
and framework content in immutable source archives before restructuring their
working documents. Never publish private media, transcripts, filenames,
absolute machine paths, credentials, or runtime reports in tracked artifacts.

Keep this application lightweight: Python standard library for the core, FFmpeg
for media, and optional lazy inference dependencies behind domain protocols.
Domain code must not import adapters. Run jobs sequentially by default. Preserve
original media bytes. Treat input as the operational queue: archive fully
transcribed real queued sources into processed after success; retain failed,
bounded, preparation-only and synthetic jobs in input. Keep metadata in process.
Write job state atomically and keep failures recoverable.
Do not imply that coarse chunk times are word alignment or that model speaker
labels are verified diarization. Never substitute synthetic text for inference.

For delegated work, use Sol for planning/review and Luna for implementation, as
requested for this project. Give agents disjoint files and explicit interfaces.
Run meaningful tests for pipeline behavior, exports, and adapter failures. Keep
mock end-to-end tests separate from actual media preparation and real inference.
Do not install a changed Nexa runtime under the assumption that it implements
the historical Qwen2-Audio API. Record verified compatibility and unresolved
runtime requirements honestly. Do not commit or publish without instructions.

Prefer NVIDIA GPU inference through CUDA and verify both model and audio-encoder
GPU placement. When CUDA is unavailable, automatic fallback to Vulkan, then CPU,
is allowed only if it is visibly warned about and recorded (backend, device,
fallback flag) in reports and events; an explicitly configured backend never
falls back silently. Each backend must pass a real end-to-end check before it is
claimed as supported.
Keep prompt, response mode, seed and temperature configurable and part of job
identity. Preserve raw model responses; prefer constrained transcript output to
arbitrary removal of text. Derived analysis must remain a separate explicit stage.

At the end of every development/review session, revise affected vision,
framework, specification and review documents following
`docs/documentation-policy.md`. Keep original content recoverable, restructure
working documents for readability and semantic access, and update headings,
navigation, stable requirement IDs, development hints, cross-links, statuses
and dated evidence. Clearly label additions and proposals. Check source
coverage, local links and privacy exclusions before reporting completion.

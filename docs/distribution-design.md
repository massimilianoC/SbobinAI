# Distribution and deployment design (proposal)

**Date:** 2026-10-02  
**Role:** Design proposal — **nothing in this document is implemented yet**  
**Status:** For decision. Feasibility, confidence and effort are estimates.  
**Scope:** Packaging the transcription pipeline as an easy-to-install open-source
tool (portable desktop folder, local HTTP API, optional web UI, container/cloud),
and what the repository needs before a public GitHub release.

## Reading paths

- [Goals and constraints](#goals-and-constraints)
- [What the current design already provides](#what-the-current-design-already-provides)
- [Option overview](#option-overview)
- [D0 — One-step bootstrap installer](#d0--one-step-bootstrap-installer-prerequisite-of-d1)
- [D1 — Portable folder with first-run setup](#d1--portable-folder-with-first-run-setup)
- [D2 — Local HTTP API and job contract](#d2--local-http-api-and-job-contract)
- [D3 — Storage adapters](#d3--storage-adapters)
- [D4 — User interfaces](#d4--user-interfaces)
- [D5 — Containers and cloud GPU](#d5--containers-and-cloud-gpu)
- [D6 — Cross-platform inference runtime](#d6--cross-platform-inference-runtime)
- [Public open-source readiness](#public-open-source-readiness)
- [Feasibility summary](#feasibility-summary)
- [Recommended roadmap](#recommended-roadmap)
- [Open questions](#open-questions)

## Goals and constraints

User request (2026-10-02, summarized):

- Distribute the tool publicly on GitHub with clear instructions and a very
  simple deployment: an executable or portable folder **without the model**.
- On first start: ask where to place resources (and optionally input/output
  folders), **estimate the download size**, then download everything needed.
- Double-click start; desktop shortcuts to the input and output folders; a
  console window that shows runtime status; well-structured output folders.
- Alternatively or additionally, a server mode: submit files with a multipart
  POST (or a path on a shared folder), receive a job id, poll status, and find
  results at a server location or in configurable storage (local, FTP, S3-like).
- Later: a self-contained web UI, a desktop app, or a cloud deployment with GPU.
- Prefer the simplest, most versatile and cross-platform option first.

Constraints carried over from [AGENTS.md](../AGENTS.md) and the
[specification](specification.md): standard-library core, optional lazy
dependencies behind domain protocols, sequential GPU jobs by default, original
media preserved, no automatic downloads during a run (setup downloads must be
explicit, user-approved and provenance-recorded), privacy of media and paths.

## What the current design already provides

| Capability needed for distribution | Current state | Gap |
| --- | --- | --- |
| Queue folder workflow | `input/` → `process/` → `output/` → `processed/`, watch mode | Folder roots are configurable but default to the repo. |
| Versioned, self-describing output | `output/<source>/<version>/` + `source.json` + `catalog.json` + `run-report.md` | Ready for a UI/API to read. |
| Replaceable inference | `TranscriptionBackend` protocol; llama.cpp HTTP adapter | Adapter accepts loopback URLs only (deliberate). |
| Model server lifecycle | PowerShell start/stop scripts with ownership manifest; server profile | Windows-only scripts. |
| Resource provenance | Pinned URLs + SHA-256 for llama.cpp runtime, Silero VAD, Qwen3-ASR GGUF | Scattered across scripts and docs; no single manifest. |
| Resumable, crash-safe jobs | Atomic state, checkpoints, `incomplete` status, process lock | Ready for a long-running service. |
| Status reporting | Console lines per chunk, intermediate reports, run report | No machine-readable progress endpoint. |
| Dependencies | Core stdlib; extras `vad` (onnxruntime, numpy) | FFmpeg and llama.cpp are external executables. |

**Assessment:** the architecture is close to distributable. The missing pieces
are packaging, a single resource manifest with a first-run downloader, a
cross-platform server launcher (Python instead of PowerShell), and an optional
HTTP layer. None requires changing the domain or the pipeline contract.

## Option overview

| ID | Option | User experience | Relative effort | Feasibility |
| --- | --- | --- | --- | --- |
| D1 | Portable folder (zip) with first-run setup and watch console | Unzip, double-click, drop files | Low–medium | High |
| D2 | Local HTTP API (`serve`) with job contract | Upload via POST or path; poll job | Medium | High |
| D3 | Storage adapters (local, shared folder, S3-compatible, FTP/SFTP) | Inputs/outputs on remote storage | Medium per adapter | Medium–high |
| D4a | Web UI served by D2 | Browser: upload, progress, compare versions | Medium | High |
| D4b | Native desktop wrapper (pywebview/Tauri) around D4a | App window instead of browser | Medium | Medium |
| D5a | Docker image (Linux + NVIDIA) running D2 | `docker run --gpus all` | Medium | High (Linux), medium (Windows WSL2) |
| D5b | Remote GPU inference only (orchestration stays local) | Local tool, rented GPU endpoint | Low–medium | Medium (security/latency work) |
| D5c | Full cloud service (multi-user, autoscaling) | Hosted product | High | Medium; outside MVP |

## D0 — One-step bootstrap installer (prerequisite of D1)

**Question (user, 2026-10-02):** can one script download and install
everything (model, llama.cpp CUDA runtime, FFmpeg, Python packages) into a
self-contained folder, using what is already on PATH when present, instead of
asking people to follow manual README steps?

**Answer:** yes. It stays an explicit, user-confirmed setup step (never an
automatic download during a transcription run, per [AGENTS.md](../AGENTS.md)),
and it can be deterministic because every artefact is pinned in
[`resources.json`](../resources.json) by URL, size and SHA-256.

### Current coverage (2026-10-02)

| Dependency | How it is obtained today | Deterministic? | Gap for a one-step installer |
| --- | --- | --- | --- |
| Python 3.11+ | User installs | n/a | Detect on PATH; otherwise download a pinned portable CPython (python-build-standalone) into the folder. |
| Python packages (`vad` extra) | `scripts\setup.ps1` + `pip install -e .[vad]` | Version ranges | Pin exact versions (lock file / hashes) for reproducible installs. |
| FFmpeg | `install-ffmpeg.ps1` via winget (latest, GPL *full* build) | No | Use FFmpeg on PATH if present; otherwise download a pinned **LGPL** portable build listed in `resources.json` (also avoids GPL redistribution questions, see [licensing review](licensing-review.md)). |
| llama.cpp CUDA runtime | `install-llamacpp-cuda.ps1` (pinned release tag; hashes read from the GitHub API) | Mostly | Read URLs/hashes from `resources.json` (already listed there) so offline verification matches the manifest. |
| Qwen3-ASR + projector, Silero VAD | `audio-transcript setup` (plan, size, confirm, resume, SHA-256, provenance) | Yes | — |
| `config.local.toml` | `audio-transcript init-config --store … --profile …` | Yes | — |
| NVIDIA driver | User installs | n/a | Detect and explain only (drivers must not be installed by the tool). |

### Proposed flow (`install.ps1`, later `install.sh`)

1. Detect OS, GPU (`nvidia-smi`), free space; choose the resource folder
   (default: inside the project/portable folder → self-contained).
2. Detect Python and FFmpeg on PATH; plan downloads only for what is missing.
3. Print the full plan with sizes (~3.5 GB on Windows + NVIDIA with models),
   ask once for confirmation (`-Yes` for unattended installs).
4. Download and verify everything from `resources.json`, write provenance,
   create the virtual environment, run `init-config`, run `doctor`.
5. Optionally create shortcuts (transcribe, input folder, output folder).

Idempotent: re-running verifies hashes and downloads only what is missing.
Effort: 2–3 days on top of the current `setup` command; confidence high.

An "installation prompt for an AI coding agent" was considered and rejected as
the primary path: it is less reproducible than a pinned script; the README
manual steps remain the documented fallback.

## D1 — Portable folder with first-run setup

### Layout

```text
AudioTranscript/                 (portable root; can live on any drive or USB disk)
  AudioTranscript.exe            launcher (or .cmd/.sh on other platforms)
  app/                           bundled Python runtime + application (no models)
  config/settings.toml           created on first run
  resources/                     downloaded on first run (location selectable)
    llama.cpp/<variant>/         CUDA / Metal / Vulkan / CPU build
    ffmpeg/
    models/qwen3-asr-1.7b/…      GGUF + mmproj + PROVENANCE.txt
    models/silero-vad/…
  data/                          default queue roots (selectable)
    input/  process/  output/  processed/
  logs/
```

### First-run flow (console wizard; later the web UI)

1. Detect platform and GPU (NVIDIA via `nvidia-smi`, Apple Silicon, otherwise
   CPU/Vulkan) and recommend a runtime variant and model profile.
2. Show the **download plan with sizes** from a pinned resource manifest, and
   the free space at the chosen location:

   | Resource (Windows + NVIDIA example) | Download size |
   | --- | --- |
   | llama.cpp CUDA build + CUDA runtime (b11193, CUDA 13.4) | ~575 MB (~709 MB installed) |
   | Qwen3-ASR-1.7B Q8_0 + bf16 audio projector | ~2.8 GB |
   | Silero VAD ONNX v6.2.3 | ~2.3 MB |
   | FFmpeg (essentials build) | ~90–100 MB (full build ~230 MB/exe) |
   | **Total** | **~3.5 GB** |

3. Ask where to store resources and where the input/output folders live
   (defaults inside the portable folder → fully self-contained).
4. Download with resume, verify SHA-256, write `PROVENANCE.txt`, record the
   manifest version. Never download into user caches on the system drive
   unless chosen.
5. Create optional desktop shortcuts: *Transcribe (start)*, *Input folder*,
   *Output folder*.
6. Start: launch the model server from the profile, run `doctor`, then **watch
   mode** with the console showing per-chunk progress and the run report path.

### Packaging choices

| Choice | Pros | Cons | Recommendation |
| --- | --- | --- | --- |
| Embedded CPython (python-build-standalone) + wheels in `app/` + tiny launcher | Transparent, no extraction step, easy updates, fewer antivirus false positives | Folder rather than one file | **Preferred** |
| PyInstaller `--onedir` | Well known, single entry exe | Larger, AV false positives possible | Acceptable |
| PyInstaller/Nuitka `--onefile` | One file | Extracts to temp on every start, slow, AV-prone; models must stay external anyway | Avoid |
| Installer (Inno Setup / MSIX / .pkg) | Start-menu integration | Not portable; signing needed | Later, optional |

The bundle without models is small: Python runtime (~30 MB) + onnxruntime CPU
(~46 MB installed) + numpy (~34 MB) + application. FFmpeg and llama.cpp are
downloaded rather than bundled, which also avoids redistributing GPL FFmpeg
builds or CUDA runtime files inside the release archive.

### Required changes (estimates)

| Work item | Effort | Confidence |
| --- | --- | --- |
| Resource manifest (`resources.json`: URL, size, SHA-256, license, platform, variant) replacing per-script pins | 1–2 days | High |
| Python setup command (`audio-transcript setup`): plan, size estimate, download/resume/verify, provenance | 2–3 days | High |
| Cross-platform server launcher in Python (start/stop/health/ownership, replacing PowerShell for the portable build) | 2–3 days | High |
| Portable path resolution (all roots relative to the portable root or chosen paths) | 1 day | High |
| Launcher + console watch experience + shortcuts (Windows first) | 1–2 days | High |
| Release workflow building zips per platform (GitHub Actions) | 1–2 days | Medium–high |
| Code signing (optional, reduces SmartScreen warnings) | Cost + 1 day | Medium |

## D2 — Local HTTP API and job contract

A `serve` command exposes the same pipeline through a small HTTP API. One
worker processes jobs sequentially (GPU), as in the CLI; requests only enqueue.

### Proposed contract (v1)

| Method and path | Purpose | Response |
| --- | --- | --- |
| `POST /v1/jobs` (multipart: `file`, optional `options` JSON) | Upload a media file and enqueue | `202` `{job_id, source_key, version, status: "queued", links}` |
| `POST /v1/jobs` (JSON: `{"source": {"path" or "uri": …}, "options": {…}}`) | Enqueue a file already on a shared folder or storage | `202` same as above; `400` if the path is outside allowed roots |
| `GET /v1/jobs/{job_id}` | Status | `{status, progress: {completed_chunks, total_chunks, percent}, timings, confidence, error, links}` |
| `GET /v1/jobs/{job_id}/events` (Server-Sent Events) | Live progress for UIs | stream of chunk/status events |
| `GET /v1/jobs/{job_id}/artifacts/{name}` | Download `transcript.txt/json/srt/vtt/md`, `report.json`, `run-report.md` | file |
| `DELETE /v1/jobs/{job_id}` | Cancel queued/running job (results kept) | `202` |
| `GET /v1/sources` / `GET /v1/sources/{source_key}` | Catalog and versions newest first (from `catalog.json` / `source.json`) | JSON |
| `GET /v1/health` / `GET /v1/models` | Readiness, loaded model, runtime | JSON |

- `job_id` = `<source-folder>/<version-folder>` (URL-encoded) — the identity the
  pipeline already uses, so re-submitting identical bytes and settings returns
  the existing version instead of recomputing (idempotency for free).
- `options` may override only an allow-listed subset (language, model profile,
  max duration, fallback/VAD settings); they become part of the fingerprint.
- Output location returned as `links` (API URLs) plus the storage location
  (`output/<source>/<version>/` or a storage URI, see D3).
- Optional callback/webhook URL on completion (later).
- Security: bind to loopback by default; API token required for non-loopback
  binding; upload size limit; path submissions restricted to configured roots;
  no shell execution from parameters.

### Implementation notes

The standard library lacks multipart parsing since the `cgi` module was
removed (Python 3.13). Two paths:

| Approach | Pros | Cons |
| --- | --- | --- |
| Optional extra `server` with Starlette/FastAPI + uvicorn + python-multipart | Robust multipart/streaming, OpenAPI docs, SSE | Extra dependencies (optional, lazy) |
| Stdlib `http.server` + own streaming multipart parser | No dependencies | Security/robustness burden; more code |

**Recommendation:** optional `server` extra (FastAPI or Starlette), keeping the
core stdlib-only. Effort: 4–6 days including tests and SSE; confidence high.

## D3 — Storage adapters

Introduce a `Storage` protocol in the domain (read input object, write artifact
tree, list) and keep the local filesystem as default. Processing always happens
on local disk (`process/`), storage only receives inputs and published outputs.

| Adapter | Dependency | Effort | Confidence | Notes |
| --- | --- | --- | --- | --- |
| Local filesystem (default) | none | done | High | Current behaviour. |
| Shared folder (SMB/NFS mount) | none | ~0.5 day | High | It is a local path; document locking and stable-file detection on network shares. |
| S3-compatible (AWS S3, MinIO, Cloudflare R2, Backblaze B2) | `boto3` optional, or minimal SigV4 client | 2–3 days | High | Upload artifacts under `<prefix>/<source>/<version>/`; presigned URLs as output links. |
| FTP/FTPS | `ftplib` (stdlib) | 1–2 days | Medium–high | Plain FTP is insecure; prefer FTPS. |
| SFTP | `paramiko` optional | 2 days | Medium | Key management. |
| WebDAV / cloud drives | various | later | Medium | Only on demand. |

**Recommendation:** for a server the multipart POST plus local or S3-compatible
output covers most needs; shared folders work today through watch mode.

## D4 — User interfaces

| Option | Description | Effort | Confidence |
| --- | --- | --- | --- |
| Console (current + D1) | Watch mode with readable per-chunk lines | done / small | High |
| D4a Web UI served by D2 | Static single-page app: drop zone, queue, live progress (SSE), per-source version list newest first with model, scope, completion %, confidence, RTF; side-by-side transcript comparison; download links | 5–8 days | High |
| D4b Desktop wrapper | pywebview (Python, small) or Tauri (Rust, smallest) window around D4a; system tray, auto-start server | 3–5 days after D4a | Medium |
| Native GUI toolkit (Qt etc.) | Separate UI codebase | 10+ days | Medium; not recommended |

The catalog/`source.json` schema already contains what the comparison view
needs; per-chunk confidence (already in checkpoints) allows highlighting
low-confidence passages for human review.

## D5 — Containers and cloud GPU

| Option | Description | Effort | Confidence | Notes |
| --- | --- | --- | --- | --- |
| D5a Docker (Linux, NVIDIA) | Image with app + FFmpeg; llama.cpp from the official CUDA server image as a second container (compose); models on a mounted volume, downloaded by `setup` | 2–3 days | High | Needs NVIDIA Container Toolkit; on Windows via WSL2. |
| D5b Remote inference endpoint | Keep orchestration local; point the backend to a rented GPU running llama.cpp | 1–2 days | Medium | Requires relaxing the loopback-only adapter rule behind an explicit setting, TLS and an API key; audio leaves the machine (privacy decision). |
| D5c Managed cloud service | Multi-tenant API, object storage, queue, autoscaled GPU workers, auth, billing | weeks | Medium | Outside MVP; D2 + D3 + D5a are its building blocks. |

## D6 — Cross-platform inference runtime

| Platform | Runtime variant | Status/confidence |
| --- | --- | --- |
| Windows x64 + NVIDIA | llama.cpp CUDA release + cudart | Verified in this project (b11193, CUDA 13.4). |
| Linux x64 + NVIDIA | Official llama.cpp CUDA server container image, or a source build | High, to be verified. |
| macOS Apple Silicon | llama.cpp macOS release (Metal) | Medium–high; Qwen3-ASR on Metal to be verified. |
| Any, no NVIDIA | Vulkan or CPU builds | Works but slower; the current deployment rule (CUDA only) applies to this installation, not to a public tool — label the variant in reports. |

Silero VAD (onnxruntime CPU) and FFmpeg are available on all three platforms.
Model files (GGUF) are platform-independent.

## Public open-source readiness

| Item | Current state | Needed |
| --- | --- | --- |
| License | **Decided 2026-10-02:** AGPL-3.0-only + commercial license (dual licensing); `LICENSE`, `COMMERCIAL-LICENSE.md` added | Maintainer contact address and CLA text still to be filled in. |
| Third-party notices | **Added** `THIRD_PARTY_NOTICES.md` | Keep in sync with `resources.json`. |
| README | Operator-oriented, Windows/PowerShell-centred | Public quick start (portable zip), requirements matrix, screenshots/GIF, troubleshooting, privacy statement. |
| Contributor files | **Added** `CONTRIBUTING.md` (with CLA requirement), `SECURITY.md`, `CHANGELOG.md` | `CODE_OF_CONDUCT.md`, issue/PR templates; review `AGENTS.md` wording for a public audience. |
| Package metadata | `pyproject.toml` lacks license, authors, URLs, classifiers | Complete metadata; decide the PyPI name; semantic versioning and `CHANGELOG.md`. |
| CI | Lint + tests on Windows/Ubuntu, Python 3.11/3.14 | Release workflow building portable archives; artifact checksums. |
| Defaults | Paths such as `C:\ProgramData\AI\…` in scripts | Portable-root-relative defaults; platform-neutral Python launcher. |
| Privacy | Strong (git-ignored data, no private paths in docs) | Keep; add a pre-release check for absolute paths and media in tracked files. |
| Model download disclaimer | — | State model licenses and that weights are downloaded from their original publishers. |

## Feasibility summary

| Deliverable | Feasibility | Confidence in estimate | Rough effort |
| --- | --- | --- | --- |
| Public repo readiness (license, notices, README, metadata) | High | High | 1–2 days |
| D1 portable folder, Windows + NVIDIA, first-run downloader | High | High | 7–10 days |
| D1 for Linux and macOS | Medium–high | Medium | +3–5 days |
| D2 local HTTP API with job contract | High | High | 4–6 days |
| D3 S3-compatible + shared folder | High | High | 2–3 days |
| D3 FTP/SFTP | Medium–high | Medium | 2–4 days |
| D4a web UI with version comparison | High | Medium–high | 5–8 days |
| D4b desktop wrapper | Medium | Medium | 3–5 days |
| D5a Docker (Linux NVIDIA) | High | High | 2–3 days |
| D5b remote GPU inference | Medium | Medium | 1–2 days + security review |
| D5c managed cloud service | Medium | Low | weeks |

Effort assumes one developer familiar with the codebase and excludes design
iterations, code signing procurement and extensive cross-hardware testing.

## Recommended roadmap

1. **R1 — Public readiness:** license, third-party notices, public README,
   metadata, privacy check. Smallest step that makes the repository publishable.
2. **R2 — One-step bootstrap installer (D0)** on top of the existing resource manifest,
   `setup` and `init-config` commands, plus a Python server launcher. This
   is the foundation shared by every other option (portable, API, Docker).
3. **R3 — Portable Windows + NVIDIA zip (D1)** with first-run wizard, shortcuts
   and console watch mode; release workflow.
4. **R4 — Local HTTP API (D2)** using the same queue and catalog.
5. **R5 — Web UI (D4a)** with version comparison and confidence highlighting.
6. **R6 — Docker (D5a) and S3-compatible storage (D3)** for server installs.
7. **R7 — Linux/macOS portable builds, desktop wrapper, remote GPU** on demand.

The separate **contextual review stage** (an LLM or a human correcting proper
nouns and terminology into a derived file, never overwriting the raw
transcript) fits after R5, where the UI can present low-confidence passages.

## Open questions

1. ~~License preference~~ — decided: AGPL-3.0-only + commercial license. Still open: public contact address, CLA text, final project/package name.
2. Primary audience of the first release: Windows + NVIDIA only, or also macOS?
3. Should the portable build keep data inside its folder by default, or in the
   user's Documents folder?
4. Server mode users: single operator on a LAN, or several users needing
   accounts and quotas?
5. Is sending audio to a rented GPU (D5b) acceptable for the intended content?
6. Which output storage matters first: shared folder, S3-compatible or FTP?

## Revision record

- **2026-10-02:** initial proposal (user request for packaging, API, storage,
  UI and cloud options with feasibility estimates). Not implemented.

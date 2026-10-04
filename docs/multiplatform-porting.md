# Multiplatform porting: analysis and specification (proposal)

**Date:** 2026-10-04  
**Role:** Research, measured evidence and specification; **partly implemented**  
**Status:** Windows CUDA, Vulkan and CPU with automatic fallback are implemented
(2026-10-04, see [implementation status](#implementation-status)); the rest is a
proposal. Measurements below were taken through the production pipeline on one
machine; other hardware classes are evidence-based estimates.  
**Question (user):** can one configurable package download the right runtime
for the user's system — NVIDIA, AMD, laptop integrated GPUs (Intel/AMD), Apple
Silicon with Metal, Vulkan, or CPU only — instead of today's NVIDIA CUDA-only
setup? Fork, alternative runtime, or single package?

## Reading paths

- [Summary and recommendation](#summary-and-recommendation)
- [Measured evidence](#measured-evidence)
- [What the runtime already offers](#what-the-runtime-already-offers)
- [Hardware classes and recommended variants](#hardware-classes-and-recommended-variants)
- [Alternative runtimes](#alternative-runtimes)
- [Implementation status](#implementation-status)
- [Specification](#specification)
- [Changes required in the current code](#changes-required-in-the-current-code)
- [Risks and open questions](#risks-and-open-questions)
- [Roadmap and effort](#roadmap-and-effort)
- [Sources](#sources)

## Summary and recommendation

**One package, no fork.** The inference engine already used (llama.cpp
`llama-server`) publishes official, prebuilt binaries for every relevant
backend — CUDA, Vulkan, ROCm/HIP, SYCL, OpenVINO, Metal and CPU — on Windows,
Linux and macOS. The same GGUF model files work on all of them. The pipeline
talks to the server over HTTP, so nothing above the server changes. What is
needed is (1) hardware detection, (2) a resource manifest with one runtime
variant per platform/backend, (3) a cross-platform server launcher, and (4) a
per-variant acceptance check.

**Measured on this machine (2026-10-04):** Qwen3-ASR-1.7B Q8_0 produced
**identical transcripts** with CUDA, Vulkan and CPU-only execution; only speed
differs (CUDA ≈ 26×, Vulkan ≈ 14×, CPU ≈ 3× faster than real time on speech).
CPU-only is therefore a usable fallback (≈ 40 minutes for a 2-hour recording on
a 12-core desktop CPU), not just a degraded mode.

| Decision | Recommendation | Confidence |
| --- | --- | --- |
| Fork vs single package | Single package with runtime variants selected at setup | High |
| Default GPU path for non-NVIDIA | **Vulkan** (AMD, Intel Arc, integrated GPUs on Windows/Linux) | High |
| Apple Silicon | llama.cpp **Metal** build (official macOS arm64 asset); MLX as a later optional backend | Medium–high (not measured here) |
| AMD discrete performance option | ROCm/HIP build as an opt-in variant | Medium |
| Intel performance option | SYCL or OpenVINO build as opt-in variants | Medium |
| No usable GPU | CPU build; offer the 0.6B model as a faster profile | High (CPU measured) |

## Measured evidence

Same pipeline entry point (`scripts/process-input.ps1`), same first 600 s of
the reference recording (177 s of speech in 24 VAD chunks), one request at a
time, Qwen3-ASR-1.7B Q8_0 + bf16 projector. Host: AMD Ryzen 9 3900X (12 cores /
24 threads), 64 GB RAM, NVIDIA RTX 5070 Ti 16 GB.

| Backend | Runtime | Inference wall time | × real time (speech) | Projection, 2 h recording (6,874 s speech) | Transcript vs CUDA |
| --- | --- | --- | --- | --- | --- |
| CUDA (RTX 5070 Ti) | llama.cpp b11193 CUDA 13.4 | 6.8 s | ≈ 26× | ≈ 4.5 min | reference |
| Vulkan (same GPU) | llama.cpp b11389 Vulkan | 13.0 s | ≈ 14× | ≈ 8.5 min | **identical** (0/24 lines differ) |
| CPU only (12 threads) | llama.cpp b11389, `--device none` | 60.3 s | ≈ 2.9× | ≈ 40 min | **identical** (0/24 lines differ) |

Notes: all runs completed 24/24 chunks without fallbacks. Vulkan was measured on
an NVIDIA GPU (the only GPU available); on AMD/Intel GPUs absolute numbers will
differ. Preparation (FFmpeg + Silero VAD on CPU) is backend-independent:
≈ 4 s for 10 minutes, ≈ 50 s for 2 hours. The Vulkan build is 31 MB versus
≈ 575 MB for CUDA + CUDA runtime, which also makes it a lighter universal
download.

## What the runtime already offers

Official release assets of llama.cpp build `b11389` (2026-10-04):

| OS / arch | Backend assets (size) |
| --- | --- |
| Windows x64 | CPU (18 MB), CUDA 12.4 (251 MB) / 13.4 (145 MB) + cudart (373–403 MB), **Vulkan (31 MB)**, ROCm 10.0 (244 MB), SYCL (141 MB), OpenVINO (85 MB) |
| Windows arm64 | CPU (11 MB), CUDA 13.4 (138 MB), OpenCL Adreno (12 MB) |
| Linux x64 | CPU (16 MB), CUDA 12.8 / 13.4 (145–163 MB) + cudart, Vulkan (30 MB), ROCm 10.0 (232 MB), SYCL fp16/fp32 (51 MB), OpenVINO (105 MB) |
| Linux arm64 | CPU (13 MB), CUDA 13.4 (140 MB), Vulkan (23 MB), Snapdragon (18 MB) |
| macOS | arm64 **Metal** (11 MB), x64 (10 MB) |

Backend facts from the official build documentation:

- **Metal** is enabled by default on Apple Silicon; unified memory means the
  model shares RAM with the system.
- **Vulkan** is the cross-vendor GPU path (Windows, Linux; macOS through a
  translation layer, so Metal is preferred there); it covers integrated GPUs.
- **HIP/ROCm** targets AMD GPUs on Linux and Windows (`HIP_VISIBLE_DEVICES`).
- **SYCL** targets Intel GPUs including integrated ones; **OpenVINO** targets
  Intel CPU/GPU/NPU.
- `--list-devices` enumerates devices; `--device none` disables all GPUs.
- Operations missing on a backend fall back to the CPU automatically (graph
  splits): correctness is preserved, speed may drop. Audio encoders in
  `libmtmd` are still marked experimental upstream; per-backend verification is
  required (see acceptance checks).

Supporting components are already cross-platform: Silero VAD runs on
onnxruntime CPU (wheels for Windows, Linux and macOS, x64 and arm64); FFmpeg is
available on every platform; GGUF model files are platform-independent.

## Hardware classes and recommended variants

| Hardware class | Examples | Recommended variant | Fallback | Model profile | Memory notes |
| --- | --- | --- | --- | --- | --- |
| NVIDIA discrete | RTX 20xx–50xx | CUDA (13.x; 12.x for older drivers) | Vulkan | 1.7B Q8 (bf16 optional) | ~3.8 GB VRAM incl. 2 slots |
| AMD discrete | Radeon RX 6000–9000 | **Vulkan** | ROCm/HIP (opt-in, supported GPUs only) | 1.7B Q8 | similar VRAM |
| Intel discrete | Arc A/B series | **Vulkan** | SYCL / OpenVINO (opt-in) | 1.7B Q8 | similar VRAM |
| Integrated GPU, Windows/Linux laptop | Intel Iris Xe / Arc iGPU, AMD Radeon 7xxM/8xxM | **Vulkan** | CPU | 1.7B Q8, or 0.6B for speed | uses shared system RAM; ≥ 16 GB RAM advised |
| Apple Silicon | M1–M5 | **Metal** (macOS arm64 build) | CPU | 1.7B Q8 | unified memory; 8 GB Macs: prefer 0.6B |
| Intel Mac | x64 | CPU | — | 0.6B | slow; best effort |
| CPU only (any) | desktop/laptop | CPU (AVX2/AVX-512 dispatch) | — | 0.6B fast / 1.7B quality | 1.7B measured ≈ 3× real time on 12 cores |
| Windows on Arm | Snapdragon X | CPU (arm64); Adreno OpenCL experimental | — | 0.6B | not evaluated |

## Alternative runtimes

| Runtime | Platforms / accelerators | Qwen3-ASR support | Fit for SbobinAI | Verdict |
| --- | --- | --- | --- | --- |
| **llama.cpp `llama-server`** (current) | All of the above | Yes (`qwen3a` projector, official GGUF) | Same HTTP contract and adapter for every backend | **Primary for all platforms** |
| **MLX** (`mlx-audio`, `mlx-qwen3-asr`) | Apple Silicon only | Yes (0.6B/1.7B, 4/5/8-bit) | Python, in-process; would need a second backend adapter | Optional Mac performance backend later |
| **sherpa-onnx / ONNX Runtime** | Win/Linux/macOS/Android/iOS; CPU, CUDA, CoreML, Windows ML providers | Yes (exported ONNX, int8) | New adapter; broad device coverage (Windows ML: AMD, Intel, Qualcomm NPUs) | Candidate for NPUs/mobile; not needed for desktop GPUs |
| `qwen3-asr.cpp` | CPU + Metal (0.6B focus) | 0.6B; forced aligner | Small project (MIT, early stage) | Watch; aligner idea useful |
| whisper.cpp / faster-whisper | All (CUDA, Vulkan, Metal, CoreML) | No (Whisper models) | Different model family | Keep as an alternative model, not a runtime replacement |

ONNX Runtime note: DirectML is now a legacy provider inside Windows ML; vendor
execution providers (AMD, Intel OpenVINO, Qualcomm QNN, NVIDIA TensorRT) are
the forward path on Windows. This matters only if an ONNX backend is added.

## Implementation status

Implemented on 2026-10-04 for **Windows x64 (CUDA, Vulkan, CPU)**. Not implemented:
other operating systems and backends (ROCm, SYCL, OpenVINO, Metal), `detect`, and the
acceptance test.

| ID | Status | Where |
| --- | --- | --- |
| MP-01 hardware detection | Partial: `nvidia-smi` (setup default) and `llama-server --list-devices` (authoritative, at selection time); no `detect` command, no AMD/Intel/Apple probes | `adapters/devices.py`, `setup --backends` |
| MP-02 runtime variants | Implemented for Windows: `backend` field, pinned b11389 Vulkan and CPU entries, `setup` extracts the archives (SHA-256, atomic, `runtime-manifest.json`); CUDA keeps its installer | `resources.json`, `application/resources.py` |
| MP-03 selection policy | Implemented as `[server].backend` / `fallback` (default `auto`: cuda, vulkan, cpu); selection by real device listing instead of vendor guesses; no model-by-memory advice | `application/backend_select.py`, `server-profile`, `doctor` |
| MP-04 launcher | Windows PowerShell launcher extended (`-Backend`, `-Device`, `-Threads`, per-backend checks and arguments); the cross-platform Python launcher is still open | `scripts/start-llamacpp.ps1`, `scripts/process-input.ps1` |
| MP-05 acceptance check | Not implemented (verified by hand: identical transcripts on all three backends) | - |
| MP-06 reporting | Implemented: `runtime` block (backend, device, device name, fallback flag, threads, tag) in `inference_configuration`, `report.json`, `transcript.json`, `metadata.json`, `run-report.md`, `source.json`, `run.started` and the live header; not part of the fingerprint | `application/pipeline.py`, `reporting.py`, `catalog.py`, `ui/live.py` |
| MP-07 packaging | Not implemented | - |

How it behaves, how to force or simulate a backend and the verified environment
variables are in [runtime setup](runtime-setup.md#inference-backends-and-automatic-fallback).

## Specification

### MP-01 — Hardware detection (`sbobinai detect --json`)

Detect without installing anything; report facts and a recommendation:

| Signal | Windows | Linux | macOS |
| --- | --- | --- | --- |
| OS/arch | `platform` | `platform` | `platform` |
| NVIDIA | `nvidia-smi --query-gpu=name,memory.total,driver_version` | same | — |
| AMD | WMI `Win32_VideoController`, `rocm-smi` if present | `/sys/class/drm/*/device/vendor` (0x1002), `rocm-smi` | — |
| Intel GPU | WMI `Win32_VideoController` | `/sys/class/drm` vendor 0x8086 | — |
| Apple Silicon | — | — | `sysctl machdep.cpu.brand_string`, `hw.optional.arm64`, `hw.memsize` |
| Vulkan devices | `llama-server --list-devices` of the downloaded Vulkan build (authoritative) | same | — |
| RAM / CPU cores / AVX2 | ctypes / `/proc` / `sysctl` | same | same |

Final verification always uses the runtime's own `--list-devices`, because
driver presence is what matters.

### MP-02 — Runtime variants in `resources.json`

Extend the manifest with `runtime` resources keyed by `{os, arch, backend}`
(pinned tag, URL, size, SHA-256 from the GitHub release `digest`, license), plus
`requires` hints (minimum driver, CUDA version) and `priority`. Profiles gain a
`runtime` selector instead of a fixed CUDA asset. Example ids:
`llamacpp-b11389-win-x64-vulkan`, `llamacpp-b11389-win-x64-cuda13.4` (+ cudart),
`llamacpp-b11389-macos-arm64-metal`, `llamacpp-b11389-linux-x64-rocm10`,
`llamacpp-b11389-win-x64-cpu`. One pinned llama.cpp build tag for all variants
keeps behaviour comparable.

### MP-03 — Selection policy

1. User override (`setup --backend cuda|vulkan|rocm|sycl|openvino|metal|cpu`).
2. Otherwise by detected class: NVIDIA → CUDA; Apple Silicon → Metal; AMD or
   Intel GPU (discrete or integrated) → Vulkan; nothing usable → CPU.
3. Model profile by memory: ≥ 6 GB free VRAM/unified → 1.7B Q8; otherwise or on
   CPU → offer 0.6B (fast) vs 1.7B (quality) with time estimates from the table.
4. Print the plan (backend, model, download sizes) and ask once (`--yes`).

### MP-04 — Cross-platform server launcher

Replace the Windows-only, CUDA-only PowerShell start/stop scripts (they require
`ggml-cuda.dll` and pass `--device CUDA0`) with a Python launcher used by every
OS: start/stop/health, ownership manifest, `--device` from the selected backend
(`CUDA0`, `Vulkan0`, `ROCm0`, `SYCL0`, `none`), `-ngl`, `--no-mmproj-offload`
for CPU, threads = physical cores for CPU. Keep the PowerShell scripts as thin
wrappers on Windows.

### MP-05 — Acceptance check per variant (`sbobinai doctor --backend-test`)

After setup: start the server, transcribe a bundled 20-second public-domain
sample, require non-empty Italian/English output with the expected language,
record backend, device name, offloaded layers, graph splits (from the server
log), speed (× real time) and peak memory, and store the result with the
installation. This replaces the current "CUDA required" rule with "selected
backend verified".

### MP-06 — Reporting

Record the backend, device and runtime variant in `inference_configuration`,
`report.json`, `run-report.md`, events and `source.json`, so versions produced
on different backends are distinguishable and comparable. (Measured outputs are
identical across backends today, but this is not guaranteed for every model or
build.) The backend is **not** part of the job fingerprint by default; add a
config switch if a user wants per-backend versions.

### MP-07 — Packaging per platform

Same Python package everywhere; per-OS launchers (`transcribe.cmd`,
`transcribe.sh`, macOS `.command`); FFmpeg from PATH or a pinned LGPL portable
build per OS; portable bundles per OS/arch built by CI (see
[distribution design](distribution-design.md), D0/D1).

## Changes required in the current code

| Area | Today | Change |
| --- | --- | --- |
| `scripts/start-llamacpp.ps1` | Requires `ggml-cuda.dll`, NVIDIA in `--list-devices`, `--device CUDA0` | **Done (Windows):** `-Backend`, `-Device`, `-Threads`; per-backend device check; CPU flags (MP-04) |
| `scripts/install-llamacpp-cuda.ps1` | CUDA assets only | **Done:** `setup` installs the Vulkan and CPU variants from `resources.json`; the CUDA installer stays (MP-02) |
| `[server]` config | No backend field | **Done for `cuda`, `vulkan`, `cpu`:** `backend`, `fallback`, `[server.runtimes]`, `device`, `threads`; ROCm, SYCL, OpenVINO and Metal remain open |
| `AGENTS.md` / specification | "CUDA only; CPU or Vulkan does not satisfy acceptance" | Keep for this maintainer deployment profile; public acceptance becomes MP-05 per selected backend |
| Python pipeline, VAD, outputs | Backend-agnostic | No change |
| Resource monitor | NVIDIA via `nvidia-smi` | Add `rocm-smi`/sysfs for AMD, `powermetrics`-free macOS fallbacks; show `n/a` otherwise |

## Risks and open questions

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Audio encoder ops missing on a backend | Slower (CPU fallback via graph splits), rarely wrong output | MP-05 test per installation; record graph splits; prefer Vulkan/Metal builds with known-good results |
| Vulkan driver quality on some iGPUs | Crashes or slow paths | Automatic CPU fallback offer; keep a CPU variant installed |
| ROCm support limited to specific AMD GPUs | Install failures | Vulkan default for AMD; ROCm opt-in |
| Apple Silicon not measured | Unknown speed | Verify on a Mac before announcing; MLX as alternative |
| Unified/shared memory pressure on 8–16 GB machines | Swapping | Recommend 0.6B; `parallel_requests = 1`; context 4096 |
| Different builds produce slightly different text | Version comparability | MP-06 records backend; optional per-backend versions |

Open questions: which Mac/AMD/Intel machines are available for verification;
whether to ship the 0.6B profile by default on CPU; whether to support Windows
on Arm in the first multiplatform release.

## Roadmap and effort

| Step | Content | Effort | Confidence |
| --- | --- | --- | --- |
| P1 | MP-01 detect + MP-02 manifest variants (Windows CUDA/Vulkan/CPU) | 2 days | High |
| P2 | MP-04 Python launcher + `[server].backend`; Windows Vulkan/CPU end-to-end | 2–3 days | High |
| P3 | MP-05 backend acceptance test + MP-06 reporting | 1–2 days | High |
| P4 | Linux (CUDA, Vulkan, ROCm, CPU) incl. Docker | 2–3 days | Medium–high |
| P5 | macOS arm64 Metal + `transcribe.command`; verify on hardware | 2 days + hardware access | Medium |
| P6 | Optional: MLX backend for Mac; ONNX/Windows ML for NPUs | 3–5 days each | Medium |

This plan fits the one-step installer (D0) and portable bundles (D1) in the
[distribution design](distribution-design.md): the installer becomes
"detect → choose variant → download → verify".

## Sources

- llama.cpp release `b11389` assets (GitHub release page, listed 2026-10-04): <https://github.com/ggml-org/llama.cpp/releases>
- llama.cpp build and backend documentation: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>
- llama.cpp multimodal (Qwen3-ASR GGUF, audio support): <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md>
- Metal CPU-fallback example (audio encoder graph splits): <https://github.com/ggml-org/llama.cpp/pull/21421>
- Vulkan and ROCm performance discussions: <https://github.com/ggml-org/llama.cpp/discussions/10879>, <https://github.com/ggml-org/llama.cpp/discussions/15021>
- Qwen3-ASR on MLX: <https://github.com/moona3k/mlx-qwen3-asr>, <https://github.com/Blaizzy/mlx-audio>
- Qwen3-ASR on sherpa-onnx: <https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/index.html>
- qwen3-asr.cpp (GGML, Metal): <https://github.com/predict-woo/qwen3-asr.cpp>
- Windows ML execution providers: <https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers>

## Revision record

- **2026-10-04 (implementation):** automatic CUDA, Vulkan, CPU backend fallback for
  Windows (MP-02, MP-03, MP-04 partly, MP-06); see
  [implementation status](#implementation-status).
- **2026-10-04:** initial research, measured CUDA/Vulkan/CPU comparison through
  the production pipeline, and multiplatform specification proposal.

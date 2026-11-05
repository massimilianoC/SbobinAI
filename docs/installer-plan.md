# Installer and redistribution: operational plan (proposal)

**Date:** 2026-11-05  
**Role:** Single operational plan consolidating the earlier distribution analyses  
**Status:** Proposal, not implemented. No installer exists yet. Effort figures are
planning estimates; sizes are computed from publisher listings, not from a built
package.  
**Consolidates:** [distribution design](distribution-design.md) (D0, D1),
[multiplatform porting](multiplatform-porting.md) (MP-01 to MP-07),
[desktop distribution status](desktop.md#distribution-status) and
[licensing review](licensing-review.md). Where this plan differs from them, see
[Reconciled decisions](#reconciled-decisions).

## Reading paths

- [Request](#request)
- [Summary](#summary)
- [Installation flow](#installation-flow)
- [Package contents and sizes](#package-contents-and-sizes)
- [Hardware validation](#hardware-validation)
- [Requirements](#requirements)
- [Work plan and effort](#work-plan-and-effort)
- [Out of scope](#out-of-scope)
- [Reconciled decisions](#reconciled-decisions)
- [Risks](#risks)
- [Open decisions](#open-decisions)
- [Sources](#sources)

## Request

Requirement, 2026-11-05 (summarized): a real standalone installer for Windows,
macOS and Linux, with a small common package of at most 50–100 MB. Validate the
hardware first, so nothing heavy is downloaded on an unsuitable machine. Choose
the folder, offer one or two models of different weight, download everything
(in parallel where possible), create a desktop icon and open the application.
Keep it as simple as is reasonable, without going to extremes.

## Summary

| Question | Answer |
| --- | --- |
| Can one binary serve all platforms? | No. Native code differs per OS and architecture. The **application is common** (pure-Python wheel, UI assets, manifest, setup wizard); the **shell is per platform**: three release files. |
| Is 100 MB achievable? | Yes, without extreme measures: about 55–80 MB per platform, including Python, the UI and speech detection. |
| Is 50 MB worth pursuing? | No. About 30 MB is possible by downloading onnxruntime and numpy at install time, but that adds a package-resolution failure point to save about 30 MB against a download of 1–3.5 GB. |
| What is downloaded afterwards? | FFmpeg, the inference runtime for the detected hardware, the chosen model and Silero VAD: about 1.2–3.6 GB. |
| Effort | About 9–13 working days for a Windows release, about 15–22 in total with macOS and Linux, plus signing procurement and access to test hardware. |

**Core choice:** ship a real Python interpreter (python-build-standalone) with
pinned wheels inside a thin native package, and put all logic in one first-run
wizard, built with the existing pywebview UI. Everything after file placement is
the same code on every OS.

## Installation flow

1. **Native package:** places the app in a folder and creates the shortcut. Nothing
   else happens here, and it takes seconds.
2. **Hardware check:** runs first in the wizard, with no heavy downloads so far.
3. **Storage:** the models and data folder defaults to the chosen folder, with free
   space shown and an explicit warning if it is on the system drive.
4. **Model choice:** one recommended option, with a time estimate for each.
5. **Consent:** the complete plan with sizes and licenses. Nothing large is
   downloaded before this.
6. **Device confirmation:** download only the small runtime for the detected backend
   and run `llama-server --list-devices`. If the device is unusable, offer the next
   backend, having wasted at most tens of MB (the CUDA runtime is about 575 MB).
7. **Parallel downloads:** resumable, verified by SHA-256, with aggregated progress.
8. **Real check:** transcribe a short bundled sample, and record backend and speed.
9. **Open the application.** Re-running the wizard resumes or repairs an installation.

**Deviation from the requested order:** the app folder is chosen before the hardware
check, because the native package must place the wizard somewhere first. This meets
the intent of the request: an unsuitable machine loses only the small package.

| Platform | Shell | Folder choice | Icon |
| --- | --- | --- | --- |
| Windows x64 | Inno Setup, per user, no administrator rights | Installer page | Desktop and Start menu |
| macOS arm64 | Signed `.dmg` with an `.app` bundle | App in Applications; data folder in the wizard | Dock/Applications |
| Linux x64 | AppImage (`.tar.gz` as fallback) | Data folder in the wizard | `.desktop` entry created by the wizard |

## Package contents and sizes

Compressed sizes in MB, listed by the publishers on 2026-11-05 for Python 3.14.

| Component | Windows x64 | macOS arm64 | Linux x64 |
| --- | --- | --- | --- |
| python-build-standalone (stripped) | 23.3 | 26.8 | 36.8 |
| onnxruntime | 15.3 | 20.9 | 23.8 |
| numpy | 12.7 | about 7 | 16.8 |
| pywebview and platform bridge | about 1 | about 8 | 0 (browser mode) |
| Application, UI, manifest | about 2 | about 2 | about 2 |
| **Package, approximate** | **55–60** | **65** | **80** |

Downloaded after consent (MB):

| Scenario | Runtime | Model + projector | VAD | FFmpeg | Total |
| --- | --- | --- | --- | --- | --- |
| NVIDIA, Standard model | CUDA + cudart 575, CPU fallback 19 | 2,807 | 2 | about 175 | **about 3.6 GB** |
| AMD/Intel GPU, Standard model | Vulkan 33, CPU fallback 19 | 2,807 | 2 | about 175 | **about 3.0 GB** |
| CPU or low memory, Light model | CPU 19 | 1,019 | 2 | about 175 | **about 1.2 GB** |
| Apple Silicon, Standard model | Metal 11 | 2,807 | 2 | To be determined | **about 2.9 GB plus FFmpeg** |

- **Standard** = Qwen3-ASR-1.7B Q8_0 with the bf16 projector (the verified profile).
- **Light** = Qwen3-ASR-0.6B Q8_0 with the Q8_0 projector. **Not yet evaluated in this
  project.** It may be offered only after a real comparison on reference speech.
- FFmpeg figures are for the full LGPL archive; only `ffmpeg` and `ffprobe` need to
  be kept after extraction.

## Hardware validation

Each check reports a fact and a verdict. A slow machine is warned, not refused,
because CPU transcription was measured to produce identical text.

| Check | Source | Result |
| --- | --- | --- |
| OS and architecture | `platform` | Refuse if unsupported |
| Free disk space | Chosen folder, download plus extraction margin | Refuse if insufficient; offer another folder |
| RAM | OS query | Proposed: warn below 16 GB, prefer Light below 8 GB |
| NVIDIA GPU and **free** VRAM | `nvidia-smi` | About 4 GB free needed for Standard |
| AMD/Intel GPU | WMI on Windows, `/sys/class/drm` on Linux | Vulkan candidate |
| Apple Silicon and unified memory | `sysctl` | Metal; prefer Light on 8 GB |
| Actual device | `llama-server --list-devices` | Authoritative, after the small runtime download |

Free VRAM must be measured rather than total: another running GPU service can
leave too little room, and the wizard should say so instead of silently recommending
the CPU. Time estimates use the existing measurement for a 2-hour recording: about
4.5 minutes with CUDA, 8.5 with Vulkan and 40 with CPU on one 12-core machine.
These are indicative on any other hardware.

## Requirements

Proposed IDs. Existing MP and GUI IDs keep their meaning.

| ID | Requirement |
| --- | --- |
| INS-01 | One release file per supported platform, at most 100 MB, containing no model, inference runtime or FFmpeg. |
| INS-02 | No Python, pip, Git or administrator rights needed on the target machine. |
| INS-03 | Hardware check before any large download; explicit consent to a plan showing sizes and licenses. |
| INS-04 | Downloads only from the pinned `resources.json`, with resume, SHA-256 and provenance; at most 4 parallel transfers; a failed item does not discard completed ones. |
| INS-05 | Re-running the wizard resumes or repairs without downloading verified files again. |
| INS-06 | App folder separate from configuration, models and data, so that updating or uninstalling never touches user recordings, outputs or models unless the user explicitly asks. |
| INS-07 | Installation ends with a real transcription of a short sample and reports the selected backend; a non-CUDA result is stated with its reason. |
| INS-08 | The app runs with no repository checkout and no development tools present, verified on a clean machine. |
| INS-09 | A platform or backend is claimed as supported only after a real end-to-end check on that hardware (existing project rule). |
| INS-10 | The bundle includes third-party notices and meets AGPL source obligations; NVIDIA runtime and models are downloaded from their publishers, not redistributed. |

## Work plan and effort

Estimates are for one developer who knows the code, and exclude signing procurement.
Automation can shorten the coding, but not hardware acceptance or procurement.

| Phase | Content | Days |
| --- | --- | --- |
| **F1 Common base** | `detect --json` (MP-01). Manifest per OS/arch/backend, including FFmpeg and the Light profile (MP-02). `.tar.gz` extraction and executable permissions, which zip extraction loses. Parallel downloads over the existing resumable downloader. CUDA installed by `setup` rather than its PowerShell script. Per-user config and data locations. Automatic free port for the worker-owned server. Light model evaluation. | 4–5 |
| **F2 Wizard** | Wizard screens in the existing UI. Resume and repair. Device confirmation. Short-sample check (MP-05) using a redistributable speech sample. Tests with a fake download server. | 2–3 |
| **F3 Windows release** | Build script (pinned Python, wheels with hashes, app). Inno Setup. WebView2 and Visual C++ runtime handling. Shortcut, app identity, uninstaller. Windows Sandbox test for the clean-machine flow and CPU path, plus a real CUDA check in an isolated folder. CI release with checksums. | 3–5 |
| **Milestone** | **Windows release: about 9–13 days.** Publish here and collect feedback before starting F4. | |
| **F4 macOS** | Metal in backend selection and placement check. `.app`, `.dmg`, signing and notarization. FFmpeg source. Real check on an Apple Silicon Mac. | 3–5 |
| **F5 Linux** | Browser-mode transport (see below). AppImage and `.desktop` entry. Real NVIDIA and Vulkan checks. | 3–4 |
| **Total** | | **about 15–22 days** |

**Linux browser mode.** Embedded Python cannot use the distribution's GTK/WebKit
bindings, and bundling Qt WebEngine would exceed the size limit. All UI calls already
go through one JavaScript function (`call` in `app.js`), so a small loopback transport
can expose the same bridge to the default browser. It must use a random token and
check `Origin`/`Host`, because any web page can reach loopback ports. The same mode
would also serve as a fallback when WebView2 is missing on Windows.

## Out of scope

Deliberately excluded to keep the plan small: Tauri or another Rust shell, PyInstaller
one-file builds, MSIX, `.deb`/`.rpm`/Flatpak, Intel Mac, Windows on Arm, ROCm/SYCL/
OpenVINO variants, MLX, automatic updates, installing GPU drivers, HTTP API (D2),
Docker (D5) and cloud options. Each can be added later without changing this design.

## Reconciled decisions

| Earlier position | Decision here | Reason |
| --- | --- | --- |
| D1: portable folder first, installer "later, optional" | Native installer per platform | Superseded by the 2026-11-05 request. |
| D1: console wizard and watch mode | GUI wizard opening the desktop app | The GUI now exists and is preferred. |
| [Desktop guide](desktop.md#distribution-status): replace `sys.executable -m …` with a bundled worker entry | Not needed | With a real interpreter instead of a frozen build, `sys.executable` stays valid. Must use `python.exe`, not `pythonw.exe`, for the worker. |
| MP-04: Python server launcher still open | Mostly present, unverified | `desktop/runtime.py` already has a worker-owned path used when the PowerShell script is absent. It has not been exercised as a packaged install. |
| D0: use tools on `PATH` when present | Always use pinned private copies | One tested configuration makes support and reproducibility simpler. |
| D0: download a pinned Python at install time | Python bundled | Keeps the first step offline and free of a further failure point. |
| Licensing P1 and D0: LGPL FFmpeg | Kept, **with a conflict** | LGPL builds do not include `libx264`, which the new optional MP4 output uses. See [Open decisions](#open-decisions). |

## Risks

| Risk | Mitigation |
| --- | --- |
| Unsigned package: SmartScreen and Gatekeeper warnings, or blocking on macOS | Budget for signing before public release; see Open decisions. |
| Missing runtime on a clean Windows machine (Visual C++ runtime for onnxruntime or llama.cpp, or WebView2) | Test in Windows Sandbox in F3; ship only the required redistributables. |
| Antivirus false positives | Avoid one-file self-extractors; sign releases; use pinned downloads. |
| A host is unavailable or a file changes | Pinned revisions and SHA-256; clear retry and resume; mirroring permitted only where licenses allow. |
| Windows Sandbox has no CUDA | Use it for the flow and CPU path only. Real CUDA checks need the actual GPU. |
| No access to a Mac or Linux GPU machine | Stated as untested and not released as supported (INS-09). |
| Light model quality is insufficient | Offer only Standard; on low-end machines warn about time. |
| Dependence on `mediafoundation`/OS encoders for MP4 | To be verified per platform in F1 if that option is chosen. |

## Open decisions

1. **FFmpeg and MP4.** Recommended: keep the LGPL build and select an available H.264
   encoder per platform (Windows Media Foundation, macOS VideoToolbox, OpenH264 on
   Linux, each to be verified), or hide MP4 where none exists. The alternative is
   downloading a GPL build from its publisher with the user's consent; its effect on
   the commercial license needs legal review. H.264 patent exposure also needs review.
   NVENC is expected to be available on NVIDIA machines with an LGPL build (to be
   verified with the pinned build): the desktop render is moving to a chain of NVENC,
   then `libx264` where present (2026-11-05).
2. **Signing.** Microsoft's service is open to EU organizations but, per its current
   guidance, individuals only in the USA and Canada; an EU legal entity could use it
   from about $9.99/month, subject to eligibility. Otherwise an OV certificate costs
   about $150–300/year. Apple requires a paid developer membership (about $99/year,
   to be confirmed).
3. **Test hardware.** Whether an Apple Silicon Mac and a Linux GPU machine are available.
4. **Python version.** Pin one of the versions in CI (3.11 or 3.14). This plan assumes 3.14.
5. **Light model.** Whether to evaluate 0.6B now, or to start Windows with Standard only.

## Sources

Sizes checked on 2026-11-05 through publisher listings:
[python-build-standalone releases](https://github.com/astral-sh/python-build-standalone/releases),
[PyPI](https://pypi.org/),
[BtbN FFmpeg builds](https://github.com/BtbN/FFmpeg-Builds/releases),
[Qwen3-ASR 1.7B GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF),
[Qwen3-ASR 0.6B GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF).
llama.cpp runtime sizes come from [`resources.json`](../config/resources.json) and
[multiplatform porting](multiplatform-porting.md#what-the-runtime-already-offers).

Signing: [Windows code signing options](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options),
[Artifact Signing pricing](https://azure.microsoft.com/pricing/details/artifact-signing/).

## Revision record

- **2026-11-05:** initial consolidated plan for a standalone installer. No
  application behavior changed.

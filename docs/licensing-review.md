# Licensing review for open-source and commercial distribution

**Date:** 2026-10-02  
**Role:** Engineering review of third-party licenses (not legal advice)  
**Status:** Findings checked against the installed packages, model cards and the
installed FFmpeg build on this date. Have a lawyer confirm before selling.

## Summary

The project is offered under **AGPL-3.0-only** and, separately, under a
**commercial license** sold by the copyright holder (dual licensing). A
commercial license covers only the code the maintainer owns. Every third-party
component keeps its own license, and every component used today allows
commercial use **without buying a license from anyone**, provided its notice
obligations are respected. Three points need attention when a commercial
package is distributed:

1. **FFmpeg build:** the build used during development is GPL-3.0 (`--enable-gpl
   --enable-version3`, x264/x265). Calling it as a separate program is fine;
   shipping that binary inside a package means shipping GPL software (with its
   source-offer duties) and possibly codec patent exposure. Prefer letting users
   install FFmpeg themselves, or ship an LGPL build.
2. **NVIDIA CUDA runtime DLLs:** governed by the NVIDIA CUDA EULA, not an
   open-source license. Downloading them at setup (current design) is simplest;
   bundling them is allowed only for the redistributable files and under the
   EULA's conditions.
3. **Ownership of the project code:** dual licensing requires the maintainer to
   hold the rights to all project code: contributions need a CLA, and
   AI-assisted code should be reviewed for ownership under the AI provider's
   terms and local copyright law.

## How dual licensing works with third-party components

- The maintainer sells rights to **their own code** (the `audio_transcript`
  package, scripts, docs). Customers receive third-party components under
  those components' own licenses, unchanged.
- Permissive licenses (MIT, BSD, Apache-2.0, PSF, zlib, 0BSD) allow use,
  modification and redistribution in commercial and closed products. Their
  duties are to keep copyright/license notices (and, for Apache-2.0, the NOTICE
  file and change statements).
- Weak copyleft (MPL-2.0, LGPL) is file- or library-level: unmodified use is
  fine; modified files/libraries must stay open; LGPL libraries must remain
  replaceable.
- Strong copyleft (GPL/AGPL) applies to the program that is distributed. Using
  a GPL program through its command line from separate software is generally
  considered aggregation, not a derived work.
- All permissive components below are compatible with distributing this project
  under AGPL-3.0 (Apache-2.0 is compatible with GPLv3/AGPLv3).

## Component inventory

### Runtime components

| Component | How it is used | License | Distributed by this repo? | Commercial use | Obligations |
| --- | --- | --- | --- | --- | --- |
| Project code (`audio_transcript`) | The product | AGPL-3.0-only / commercial | Yes | Sold by maintainer | Own copyright; CLA for contributions |
| Python standard library / CPython | Interpreter | PSF-2.0 | No (user installs; a portable build would bundle it) | Yes | Keep PSF notice if bundled |
| llama.cpp / ggml (`llama-server`) | Separate inference server | MIT | No (downloaded at setup) | Yes | Keep MIT notice if bundled |
| NVIDIA CUDA runtime (cudart, cuBLAS DLLs) | GPU libraries of llama.cpp | NVIDIA CUDA EULA (proprietary) | No (downloaded at setup) | Yes, as a user of the runtime | If bundled: only EULA redistributable files, under EULA terms |
| NVIDIA driver | GPU | NVIDIA driver license | No (user installs) | Yes | None for us |
| Qwen3-ASR-1.7B weights (GGUF by ggml-org) | Speech model | Apache-2.0 (base model card; the GGUF card has no license field and inherits the base model license) | No (downloaded at setup) | Yes | Attribution, license text, state modifications if you modify weights |
| Silero VAD v6.2.3 ONNX | Speech detection | MIT | No (downloaded at setup) | Yes | Keep notice |
| onnxruntime 1.30 | Runs Silero on CPU | MIT | No (pip) | Yes | Keep notice |
| NumPy 2.5 | Buffers for VAD | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (bundles OpenBLAS and the MSVC runtime DLL on Windows) | No (pip) | Yes | Keep notices |
| flatbuffers, protobuf (onnxruntime dependencies) | Runtime libraries | Apache-2.0, BSD-3-Clause | No (pip) | Yes | Keep notices |
| FFmpeg / FFprobe | Separate executables | LGPL-2.1+ by default; the development build (gyan.dev *full*) is **GPL-3.0** | No (user installs, e.g. winget) | Yes as a separate program | If bundled: comply with LGPL/GPL (license, source offer); consider codec patents |

### Optional, development and legacy components

| Component | Use | License | Note |
| --- | --- | --- | --- |
| ruff | Lint (development only) | MIT | Not shipped. |
| gguf (+ requests, urllib3, idna, charset-normalizer, PyYAML, tqdm, certifi) | Legacy projector conversion extra | MIT / Apache-2.0 / BSD / MPL-2.0 (certifi, tqdm) | MPL files used unmodified: fine. |
| Qwen2-Audio-7B-Instruct and NexaAI GGUF | Legacy backend | Apache-2.0 (both model cards) | Optional. |
| Silero / faster-whisper algorithm ideas | Segmentation design | MIT | Credited in source; no code copied verbatim beyond the documented approach. |

## Points requiring attention

### P1 — FFmpeg build and codec patents

- **Today:** FFmpeg is installed by the user and called as a subprocess. The
  project does not link to it or redistribute it, so its GPL does not extend to
  the project code.
- **Commercial bundle:** either (a) keep FFmpeg out of the package and ask the
  user to install it, or (b) bundle an **LGPL-only** build (no `--enable-gpl`,
  no `--enable-nonfree`), include its license and offer the corresponding
  source. The pipeline only needs demuxing/decoding and PCM WAV output.
- **Patents:** decoding H.264/HEVC/AAC in a commercially distributed product
  may require patent licenses in some jurisdictions (patent pools). Shipping no
  codec binaries, or relying on OS/user-installed decoders, reduces exposure.
  Confirm with counsel for the target markets.

### P2 — NVIDIA CUDA runtime

The `cudart` package published with llama.cpp contains NVIDIA CUDA libraries
under the NVIDIA CUDA Toolkit EULA. The EULA lists runtime files that may be
redistributed together with an application, under conditions (unmodified,
with your application, under terms at least as protective as the EULA).
Keeping the download in the setup step avoids redistribution entirely.

### P3 — Rights in the project code

- Dual licensing works only if the maintainer can license **all** project code
  commercially. External contributions therefore require a CLA (see
  [CONTRIBUTING.md](../CONTRIBUTING.md)) before merge.
- Much of the code was written with AI coding assistants. Provider terms
  generally assign outputs to the user, but the copyright status of purely
  machine-generated material is unsettled in several jurisdictions. Human
  review, selection and modification strengthen ownership; ask counsel if the
  commercial value depends on exclusivity.

### P4 — Notices and trademarks

- Ship [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the license texts
  of any bundled component (portable builds: PSF, MIT texts, NumPy/onnxruntime
  notices, Apache-2.0 for the model if bundled).
- Names such as Qwen, NVIDIA, CUDA and llama.cpp may be used to describe
  compatibility, but not to imply endorsement or as product branding.

## Do we need to buy any license?

| Scenario | Purchase needed? |
| --- | --- |
| Free AGPL distribution on GitHub, components downloaded at setup | No |
| Selling commercial licenses for the project code, components downloaded by the user at setup | No (respect notices) |
| Commercial package bundling CPython, onnxruntime, NumPy, llama.cpp, Silero, Qwen3-ASR | No (permissive licenses; include notices) |
| Bundling NVIDIA CUDA DLLs | No purchase; comply with the CUDA EULA redistribution terms |
| Bundling a GPL FFmpeg build | No purchase; GPL compliance for that binary; consider an LGPL build |
| Distributing H.264/HEVC/AAC decoders commercially | Possibly codec patent licenses in some jurisdictions — legal check |

## Maintenance

Re-run this review when adding a dependency, model or bundled binary; record
licenses in `resources.json` and `THIRD_PARTY_NOTICES.md`.

## Revision record

- **2026-10-02:** initial review after choosing AGPL-3.0-only + commercial licensing.

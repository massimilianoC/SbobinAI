# Qwen2-Audio with Nexa SDK — Framework reference

**Reviewed:** 2026-10-02  
**Purpose:** preserve the supplied model reference and connect it to development  
**Source:** [complete original text, unchanged](sources/nexaAI_Qwen2-Audio.original.md)  
**Status:** current llama.cpp CUDA execution verified on real audio; legacy Nexa adapter retained

The supplied text describes a historical Nexa SDK integration. This document
organizes its contents and adds project-specific compatibility, implementation,
and validation guidance. Source claims, verified contracts, and proposed
extensions remain distinct. The archive preserves the complete supplied text.

## Contents and reading paths

- [Model and runtime roles](#model-and-runtime-roles)
- [Capabilities and project scope](#capabilities-and-project-scope)
- [Historical usage](#historical-usage)
- [Verified adapter contract](#verified-adapter-contract)
- [Project execution recipe](#project-execution-recipe)
- [Validation and open requirements](#validation-and-open-requirements)
- [Extension hints](#extension-hints)
- [Sources and revision record](#sources-and-revision-record)

For **a model overview**, read capabilities. For **installation decisions**,
read the historical compatibility distinction and adapter contract. For **project
commands**, follow the execution recipe. For **future development**, read hints
and establish capability evidence before extending the implementation.

## Model and runtime roles

| ID | Concept | Meaning and provenance |
| --- | --- | --- |
| FW-01 | Qwen2-Audio | The source describes a multimodal audio-language model accepting audio and text, with voice interaction without a separate ASR module. |
| FW-02 | Languages | The reference names English, Chinese, and major European languages. Coverage does not establish accuracy for a particular recording, accent, or domain. |
| FW-03 | Nexa SDK | The reference describes local inference on edge devices, GGUF quantization choices, and Python-package or executable-installer installation. |
| FW-04 | Wider SDK scope | The source lists text/image generation, vision-language, audio-language, ASR, and TTS. These are SDK capabilities, not all features of this model or project. |

The model, its quantized weights, the audio projector, the runtime, and an HTTP
server are separate dependencies. LM Studio was mentioned in the vision; this
framework reference does not establish its audio compatibility with Qwen2-Audio.
See [backend compatibility](../backend-compatibility.md).

## Capabilities and project scope

| ID | Supplied capability or use case | Project treatment |
| --- | --- | --- |
| FW-05a | Speaker identification and response | Preserve as a model claim and future validation target. No verified speaker identity or diarization in the MVP. |
| FW-05b | Speech translation and transcription | Faithful original-language transcription is the initial task. Translation requires a future explicit mode. |
| FW-05c | Mixed audio/noise detection and background-noise response | Preserve the model claim. The sampler measures energy independently; it does not classify sound or prove speech. |
| FW-05d | Music and sound analysis | Future capability requiring a structured response contract and actual-model validation. |
| FW-05e | Voice chat, daily questions and suggestions | Preserve these use cases outside the batch transcription MVP. |
| FW-05f | Information extraction, audio summary and transcription expansion | Proposed analysis artifacts must remain separate from the transcript and refer to supporting segments. |

**FW-06 — Performance:** the supplied text claims improvement over previous
state-of-the-art systems or Qwen-Audio across tasks. It supplies no benchmark
table, metrics, conditions, or project measurements. Preserve this as a source
claim, not a local quality guarantee. The
[official Qwen repository](https://github.com/QwenLM/Qwen2-Audio) is a primary
starting point for additional model research.

## Historical usage

**FW-07 — Installation:** the original asks the reader to install Nexa SDK,
mentioning a Python package and executable installer, but provides no package
version, download URL, or complete installation command.

**FW-08 — Terminal:** the supplied example is:

```bash
nexa run qwen2audio
```

The source says it uses `q4_K_M` by default. Users drag an audio file into the
terminal, or enter a path on Linux, then supply a guiding text prompt or leave
it empty for direct voice input.

**FW-09 — Streamlit:** the source also gives:

```bash
nexa run qwen2audio -st
```

It mentions choosing quantizations and checking a RAM list.
**FW-10 — Memory:** the source reports **4.2 GB RAM** for the default
`q4_K_M`. Measurement conditions are unspecified; validate additional runtime,
projector, context, audio, and device overhead before sizing a deployment.

These examples are historical reference material, not commands verified against
today's packages. The originally linked Nexa repository redirects to GenieX,
whose current platform scope differs. Do not assume a current package exposes
the historical Qwen2-Audio interface.

**Other retained material:** the source mentions demos, blogs, benchmarks, and
Discord/X community resources without supplying destination URLs. The
[original archive](sources/nexaAI_Qwen2-Audio.original.md) retains all of these
references; no missing links or demo content are invented.

## Verified adapter contract

The project targets the historical class in
[this exact archived source revision](https://github.com/NexaAI/nexa-sdk/blob/c84b03fc0a741102c86f3bdc6d3935d25c108aa6/nexa/gguf/nexa_inference_audio_lm.py).
This verifies an interface, not inference success in the current environment.

| Concern | Contract and implementation consequence |
| --- | --- |
| Import | `nexa.gguf.nexa_inference_audio_lm.NexaAudioLMInference`, loaded lazily so the core can run without Nexa. |
| Construction | `model_path`, `local_path`, `projector_local_path`, and `device`; this project's adapter requires existing local model and projector files. |
| Inference | `inference(audio_path, prompt)` returns text. The archived class also exposes `inference_streaming`; the project uses the non-streaming method. |
| Audio | Historical normalization is 16 kHz. Default preparation is mono PCM16 WAV at 16 kHz with bounded, sample-exact chunks. |
| Resources | A native context is created per inference. Release it through `cleanup()` or `close()` after each chunk, including failures. |
| Downloads | Both local paths are passed to avoid implicit model/projector downloads through the historical hub. |
| Time and speakers | Returned text does not establish word alignment or speaker identity. MVP segments use chunk boundaries and no speaker ID. |
| Failures | Missing/incompatible runtime, missing files, initialization errors and empty responses are failures. Never substitute the mock automatically. |

Implementation: [Nexa adapter](../../src/audio_transcript/adapters/nexa.py),
[media adapter](../../src/audio_transcript/adapters/ffmpeg.py),
[domain interfaces](../../src/audio_transcript/domain/ports.py).

## Project execution recipe

The current route uses the [shared CUDA runtime](../runtime-setup.md) and
[structured inference controls](../inference-controls.md). Run
`run --config config.local.toml`; input is a queue, complete real jobs archive
to processed, and metadata/chunks remain in process. The following Nexa recipe
is retained specifically for the optional historical adapter.


1. Install the lightweight project and FFmpeg using the [README](../../README.md).
2. Provide a Python environment with a compatible historical runtime. Python
   3.11+ support in the core does not prove that a particular old runtime build
   supports the same interpreter or device.
3. Supply matching existing GGUF model and audio-projector files through
   `model_path`/`projector_path` in ignored TOML or the corresponding CLI flags.
4. Run `doctor` before inference. It checks dependencies and files without
   constructing/downloading a model; inference success needs a separate test.
5. Use an excerpt with actual audio signal. If the beginning is quiet, select
   random central samples and preserve their source offsets in the local report.
6. Review real-model output before processing the entire recording.

```powershell
# These example files must already exist.
.\scripts\run.ps1 doctor --model-path models\qwen2audio.gguf --projector-path models\audio-projector.gguf
.\scripts\run.ps1 run --config config.local.toml --max-duration 20

# Sample central windows and prepare their audio without inference.
.\.venv\Scripts\python.exe scripts\sample-audio.py input\recording.mp4 --count 3 --seconds 20
.\scripts\run.ps1 run --prepare-only --input-dir .local\samples\input --process-dir .local\samples\process --output-dir .local\samples\output --chunk-seconds 10
```

After runtime setup, transcribe samples with the same directories and
`--config config.local.toml --no-prepare-only`. Sample transcript timestamps
are relative to the excerpt; original-video offsets remain in the local
`sampling-report.json`.

## Validation and open requirements

| Check | Recorded status | Next evidence |
| --- | --- | --- |
| Core, adapter contracts and exports | Local automated checks passed | Keep checks current after code changes. |
| Actual audio preparation | FFmpeg and non-silent central sampling passed | Verify backend output on those excerpts. |
| Legacy runtime | Missing or incompatible in the inspected environment | Record a compatible version and device/interpreter support. |
| Qwen2-Audio inference | Real WAV inference succeeded through llama.cpp CUDA | Review accuracy and repeat with changed inputs/settings. |
| LM Studio audio | Loaded model text request succeeded; input_audio/audio_url rejected with HTTP 400 | This inspected endpoint is unsuitable for the required audio request. |
| Diarization, alignment and rich analysis | Not implemented or verified | Explicit schemas, capability evidence and acceptance checks. |

See the dated [verification evidence](../verification.md). Mock output,
audio preparation and actual inference must remain separate outcomes.

## Extension hints

- **Analysis:** propose a separate artifact for summary, sound events, segment
  references, uncertainty and provenance. These concepts are not current outputs.
- **Speakers:** define supported diarization separately from textual speaker guesses.
- **Timing:** use a verified alignment provider before claiming precise subtitles.
- **Other models/devices:** declare formats, duration limits, tasks, hardware and
  lifecycle requirements at the adapter boundary.
- **Long recordings:** measure resource use, throughput and recovery before parallelism.
- **Documentation:** maintain source references, stable semantic IDs, status
  tables and a revision record using the [documentation policy](../documentation-policy.md).

## Sources and revision record

- [Supplied source archive](sources/nexaAI_Qwen2-Audio.original.md): complete text preserved byte for byte.
- [Official Qwen2-Audio repository](https://github.com/QwenLM/Qwen2-Audio): model reference.
- [Nexa model card](https://huggingface.co/NexaAI/Qwen2-Audio-7B-GGUF): historical model/runtime usage.
- [Archived Nexa class](https://github.com/NexaAI/nexa-sdk/blob/c84b03fc0a741102c86f3bdc6d3935d25c108aa6/nexa/gguf/nexa_inference_audio_lm.py): verified legacy interface.
- [Repository now reached by the Nexa link](https://github.com/qualcomm/GenieX): compatibility context from startup review.
- [LM Studio endpoints](https://lmstudio.ai/docs/developer/openai-compat): no established Qwen2-Audio audio contract in that review.

**2026-10-02 revision:** organized every supplied topic and preserved historical
commands and claims. Added implementation references, runtime requirements,
validation boundaries and extension hints. This editorial revision does not
claim a new real-inference result.

### Current execution extension — 2026-10-02

The existing Q8 language-model GGUF was reused; only its missing matching
audio projector was downloaded and converted into a separate current MMPROJ
layout without requantization. The working runtime is llama.cpp b11193/CUDA
13.4, with 33/33 model layers and audio encoder on CUDA0. NVIDIA CUDA is a
deployment requirement. This setup supersedes the historical runtime blockage
for the current adapter, without changing archived source claims. Prompt, seed,
temperature and JSON/plain mode are configurable per run; raw responses remain
reviewable. The [TODO](../TODO.md) distinguishes later analysis from transcription.

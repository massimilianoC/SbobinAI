# Backend compatibility and runtime selection

Current route: a shared **llama.cpp CUDA** server with Qwen3-ASR-1.7B Q8_0 and its
bf16 audio projector (the decision is recorded below). The first checks, dated
2026-10-02, used Qwen2-Audio weights; they are kept as history. See
[runtime setup](runtime-setup.md) and [verification](verification.md) for
reproducible commands and limits.

## Runtime comparison

| Runtime | Verified or documented evidence | Decision |
| --- | --- | --- |
| llama.cpp b11193, CUDA 13.4 | HTTP audio requests succeed; model and audio encoder use CUDA0 (first verified with a Qwen2-Audio Q8 model and a converted projector, then with Qwen3-ASR through the `qwen3a` projector type). | Current execution runtime. |
| LM Studio local endpoint | Qwen2-Audio was loaded; text control returned HTTP 200. Both `input_audio` and `audio_url` returned HTTP 400: content supports only text/image URLs. | This inspected endpoint cannot serve the required audio request. |
| Historical Nexa SDK | Archived class contract identified, but compatible historical distribution unavailable in this environment. | Removed (see [Removed Nexa backend](#removed-nexa-backend)). |
| Ollama | Already installed; documented chat messages do not establish this Qwen2-Audio projector/audio contract. | No speculative migration or second installation. |
| vLLM | Official supported-model table includes Qwen2AudioForConditionalGeneration with audio. | Alternative requiring a separately verified deployment and weight format; unnecessary for the working GGUF route. |

The LM Studio result supersedes the initial startup snapshot that did not list
the model. A loaded text model and a reachable `/v1/models` endpoint alone do
not prove audio support. In the first check, the existing 8 GB Qwen2-Audio GGUF
was reused without modification and its missing projector was converted into a
separate file with provenance hashes; this route is no longer used.

## Model selection on the llama.cpp runtime (2026-10-02)

The [llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
lists Qwen3-ASR (0.6B, 1.7B) among supported audio models and notes that
pre-quantized Qwen2-Audio GGUF files give poor results. Qwen2-Audio-7B-Instruct
is a chat model: on silence it repeated its prompt or invented stock sentences,
and on speech it once summarized instead of transcribing and looped to the token
limit. Qwen3-ASR is an LLM-decoder model trained only for speech recognition
(52 languages and dialects, including Italian; Apache-2.0).

Same production pipeline, same first 600 s of the reviewed recording (177 s of
speech in 24 VAD chunks), one model loaded at a time, CUDA0, desktop baseline
about 3.0 GiB excluded from the VRAM figures:

| Model (GGUF) | Server VRAM (approx.) | Inference time | Fallbacks | Observed output |
| --- | --- | --- | --- | --- |
| Qwen2-Audio-7B Q8 + converted projector | ~10.7 GiB | 20.4 s (RTF 0.034) | 0 | Readable; one invented sentence on a sub-second noise chunk, non-Latin output on a short greeting, some paraphrase. |
| Qwen3-ASR-1.7B Q8_0 + bf16 mmproj, auto language | ~3.7 GiB | 4.2 s (RTF 0.007) | 0 | More literal (keeps repetitions and fillers); noise chunk reported as no speech; short greeting mislabelled as Chinese. |
| Qwen3-ASR-1.7B Q8_0 + bf16 mmproj, forced Italian | ~3.7 GiB | 2.8 s (RTF 0.005) | 0 | As above, greeting correct; noise chunk transcribed as a filler sound. |
| Qwen3-ASR-1.7B bf16 + bf16 mmproj, forced Italian | ~5.3 GiB | 5.3 s (RTF 0.009) | 0 | Text identical to Q8_0 except two final periods. |

**Decision:** Qwen3-ASR-1.7B Q8_0 with the bf16 projector and forced language is
the production model for this deployment (alias `qwen3-asr-1.7b-q8`). bf16
weights brought no visible benefit for 1.4 GiB more memory. Quality evidence is
qualitative review of one recording, not a word error rate. Files come from
[ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF)
at a pinned revision; provenance and SHA-256 live beside the files in the
shared model store.

Dedicated non-LLM ASR (Whisper large-v3-turbo through whisper.cpp or
faster-whisper; NVIDIA Parakeet TDT 0.6B v3 or Canary-1B-v2 for European
languages) remains an option behind the same backend protocol. Note that
Parakeet TDT 0.6B v2, Canary-1B-flash and Canary-Qwen-2.5B do not support
Italian. Desktop applications reviewed as integration targets were rejected:
they are GUI-centred, Whisper-only or use Vulkan rather than CUDA on Windows.

**Summary with the loaded ASR model: not feasible (checked 2026-11-05).** A
text-only probe of Qwen3-ASR-1.7B Q8 on llama.cpp (CPU runtime, no projector)
did not produce summaries or answers; see the full note in [TODO](TODO.md#contextual-review-stage-requested-proposal).

## Removed Nexa backend

An optional adapter for the historical Nexa SDK (`nexa run qwen2audio`) existed
during the first iteration. It was never verified end to end in this environment
(no compatible runtime distribution was available), was superseded by Qwen3-ASR
on llama.cpp and has been removed together with the Qwen2-Audio projector converter
and the supplied framework reference. A configuration that still says
`backend = "nexa"` is rejected with an explanatory error. The supplied
reference remains in the git history. The remaining options are `llamacpp` (the
production backend) and `mock` (tests only).

## LM Studio

[LM Studio's documented compatibility endpoints](https://lmstudio.ai/docs/developer/openai-compat)
describe chat input as text and images. They do not establish an audio
transcription contract for Qwen2-Audio. The MVP therefore does not implement a
speculative LM Studio audio adapter. A reachable `/v1/models` response alone is
insufficient to enable one.

## Current llama.cpp contract

The stdlib HTTP adapter checks `/health` and requires the configured model alias
in `/v1/models`; it never silently selects another model. It sends base64 PCM WAV
using `input_audio` to `/v1/chat/completions`. The server owns the loaded model,
projector and GPU lifecycle; client shutdown leaves that shared server running.
NVIDIA CUDA is the preferred and only **verified** backend. Since 2026-10-04,
`[server].backend = "auto"` falls back to Vulkan, then CPU, with a visible
warning and the backend recorded in every report; both fallbacks are **beta**
(verified on one machine, not on AMD or Intel GPUs). See the
[support status](../README.md#support-status) and
[runtime setup](runtime-setup.md#inference-backends-and-automatic-fallback). JSON-schema output, instructions, seed and temperature are
documented in [inference controls](inference-controls.md).

Successful audio inference does not prove a measured transcription accuracy or
verified diarization. The upstream audio path is experimental. Keep raw responses
for review.

## Primary references

- [llama.cpp multimodal support](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [llama.cpp server audio API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Official CUDA release assets](https://github.com/ggml-org/llama.cpp/releases/tag/b11193)
- [LM Studio compatibility](https://lmstudio.ai/docs/developer/openai-compat)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)

Revision: recorded live endpoint rejection, GGUF reuse/projector conversion and
CUDA inference. 2026-10-02: added the Qwen2-Audio vs Qwen3-ASR comparison and the
production model decision. 2026-11-05: removed the legacy Nexa backend and its
contract section.

# Shared NVIDIA CUDA runtime

Reviewed: 2026-10-02. Runtime: llama.cpp `b11193`, Windows x64, CUDA 13.4.
The deployment requirement is **NVIDIA GPU inference through CUDA**. CPU and
Vulkan runs are diagnostic history and do not satisfy current acceptance.

## Reading paths

- [Install shared binaries](#install-shared-binaries)
- [Reuse the model and prepare its projector](#reuse-the-model-and-prepare-its-projector)
- [Start, check and stop](#start-check-and-stop)
- [Use from other services](#use-from-other-services)
- [Scope and quality limits](#scope-and-quality-limits)

## Install shared binaries

The installer obtains the official Windows CUDA binary and companion CUDA
runtime ZIPs from the [pinned release](https://github.com/ggml-org/llama.cpp/releases/tag/b11193).
It verifies their GitHub SHA-256 digests before installation. The driver must
support the selected CUDA runtime; `nvidia-smi` reports driver compatibility,
not the presence of a compiler toolkit. Prebuilt binaries do not require `nvcc`.

```powershell
.\scripts\install-llamacpp-cuda.ps1
& "$env:ProgramData\AI\runtimes\llama.cpp-cuda\llama-server.exe" --list-devices
```

The default shared directory contains executables, CUDA libraries and a
`runtime-manifest.json` with release provenance. Keep each runtime variant in
its own directory; never mix Vulkan and CUDA DLLs. The installer refuses to
replace a runtime with a running server. It does not change unrelated runtimes.
The current installation allows ordinary Windows users to read/execute binaries.

## Reuse the model and prepare its projector

The existing Nexa Qwen2-Audio Q8 GGUF language model can be reused. Its original
audio-projector GGUF uses a legacy layout; current llama.cpp needs a separate
MMPROJ file. Obtain the matching original projector from the
[Nexa model repository](https://huggingface.co/NexaAI/Qwen2-Audio-7B-GGUF/tree/main)
when missing. Store it beside, rather than over, existing weights.

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[conversion]'
.\.venv\Scripts\python.exe scripts\convert-nexa-projector.py --input models\nexa-projector.original.gguf --output models\llamacpp-projector.gguf
```

Conversion validates the source architecture, metadata and tensor shapes;
maps 489 tensors to current names; preserves F16/F32 values; reshapes two
convolution biases; and omits the legacy stored mel filter table, which the
runtime computes itself. It does not requantize or overwrite the source. The
adjacent conversion report records source/target SHA-256 hashes and tensor count.
The converter is specific to the inspected Nexa Qwen2-Audio projector.

## Start, check and stop

Use your actual existing model path and converted projector path:

```powershell
.\scripts\start-llamacpp.ps1 -ModelPath models\qwen2audio-q8.gguf -ProjectorPath models\llamacpp-projector.gguf
.\scripts\run.ps1 doctor --backend llamacpp --model qwen2-audio-7b --base-url http://127.0.0.1:8088
.\scripts\run.ps1 run --config config.local.toml --file input\recording.mp4
.\scripts\stop-llamacpp.ps1
```

The start script requires `ggml-cuda.dll` and an enumerated NVIDIA CUDA device.
It requests GPU model offload and keeps projector offload enabled. Defaults:
loopback port 8088, model alias `qwen2-audio-7b`, context 4096, one server slot.
The current proof logs show **33/33 model layers offloaded** and the audio
encoder using **CUDA0**. Confirm these markers after changing runtime/hardware.

The server runs hidden in the background. Its state, logs and ownership
manifest live under the shared service directory. Startup is idempotent for an
owned server with matching configuration. A busy port belonging to another
process is a failure; scripts never stop that process. Shutdown verifies PID,
executable path and creation time before stopping its own server.

Copy `config.example.toml` to ignored `config.local.toml` for the HTTP backend.
The server, rather than the Python client, selects CUDA. The CLI `device`
setting cannot move an already running HTTP server between devices.
Use sequential requests when other services share this GPU.

## Use from other services

The shared files are independent of this repository and its virtual environment:

| Shared resource | Default location or interface |
| --- | --- |
| CUDA runtime | `%ProgramData%\AI\runtimes\llama.cpp-cuda` |
| Lifecycle scripts and operator guide | `%ProgramData%\AI\tools` and `%ProgramData%\AI\README.md` |
| State and logs | `%ProgramData%\AI\services\qwen2-audio` |
| Readiness | `GET http://127.0.0.1:8088/health` |
| Loaded model | `GET http://127.0.0.1:8088/v1/models` |
| Inference | `POST http://127.0.0.1:8088/v1/chat/completions` |

Standalone lifecycle scripts were also copied to the shared tools directory.
To reuse the current model paths without this repository, read `server.json`
and pass its `modelPath` and `projectorPath` to the shared start script. The
manifest is local operational data, not a public model-download prescription.

Send `input_audio` with base64 WAV data and model `qwen2-audio-7b` as described
in the [official server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
No private media paths need to be sent over the API. Loopback clients need no
cloud API key. Keep the model alias explicit; a different loaded model fails
the application check rather than being substituted automatically.

## Scope and quality limits

The installation is system-shared files plus a running local model server.
CUDA commands were added to the current user's persistent PATH. Machine PATH
and Windows Service Control Manager registration were not changed; Windows
services can call the explicit shared executable/API. There is no automatic
boot startup. Lifecycle scripts provide repeatable startup and shutdown.

Real audio inference succeeded on three central excerpts and a complete
180-second production-pipeline test. This establishes
end-to-end compatibility, not a measured word-error rate. llama.cpp marks audio
as experimental; the observed model occasionally adds a prefatory sentence or
quotation marks despite transcript-only prompting. Preserve and review model
output rather than silently deleting possibly spoken words. The current JSON
mode avoided prefatory text on the 180-second test. A full-recording run later
hit repetitive generation and is explicitly partial; see the execution TODO.
Subtitle timings
remain coarse chunk boundaries; speaker identities and alignment are unverified.

See [verification](verification.md) for dated evidence and
[compatibility](backend-compatibility.md) for rejected/alternative runtimes.

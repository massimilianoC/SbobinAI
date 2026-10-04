# Shared NVIDIA CUDA runtime

Reviewed: 2026-10-02. Runtime: llama.cpp `b11193`, Windows x64, CUDA 13.4.
The maintainer deployment requirement is **NVIDIA GPU inference through CUDA**;
acceptance of that deployment is still CUDA only. Other machines get an automatic,
clearly warned fallback to Vulkan or the CPU (see
[Inference backends](#inference-backends-and-automatic-fallback)); the fallback
output is identical but slower and does not replace CUDA acceptance.

## Reading paths

- [Install shared binaries](#install-shared-binaries)
- [Inference backends and automatic fallback](#inference-backends-and-automatic-fallback)
- [Reuse the model and prepare its projector](#reuse-the-model-and-prepare-its-projector)
- [Start, check and stop](#start-check-and-stop)
- [Production model: Qwen3-ASR](#production-model-qwen3-asr)
- [Speech detector model](#speech-detector-model)
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

## Inference backends and automatic fallback

Added 2026-10-04 (implements MP-02, MP-03, MP-04 for Windows and MP-06 of the
[multiplatform analysis](multiplatform-porting.md)). The same GGUF files run on
three llama.cpp runtime variants; measured on the reference machine (RTX 5070 Ti,
12-core CPU) through the production pipeline, the transcripts were identical:

| Backend | Runtime folder (default, relative to `[resources].store`) | Size | Speed vs CUDA |
| --- | --- | --- | --- |
| `cuda` | `runtimes/llama.cpp-cuda` (b11193, CUDA 13.4 + cudart) | ~575 MB | reference |
| `vulkan` | `runtimes/llama.cpp-vulkan-b11389` | ~33 MB | about 2x slower |
| `cpu` | `runtimes/llama.cpp-cpu-b11389` | ~19 MB | about 9x slower |

Keep every variant in its own folder; never mix the DLLs of two backends.

### Install the runtimes

`sbobinai setup --config config.local.toml` installs the Vulkan and CPU runtimes
itself: it downloads the pinned ZIP, verifies size and SHA-256, extracts it into a
temporary folder and renames that folder atomically, then writes
`runtime-manifest.json` (tag, backend, asset, SHA-256, file list) and
`PROVENANCE.txt` next to the binaries. A verified installation is skipped; a
non-empty folder that setup did not create is never overwritten. The CUDA runtime
keeps its PowerShell installer (`scripts/install-llamacpp-cuda.ps1`); setup prints
the exact command. `--backends cuda,vulkan,cpu` picks the runtimes. The default is
`vulkan,cpu` plus `cuda` when `nvidia-smi` lists a GPU (or only the backend fixed in
`[server].backend`). `--dry-run` prints each backend with its size.

### Choose the backend

```toml
[server]
backend = "auto"                       # auto | cuda | vulkan | cpu
fallback = ["cuda", "vulkan", "cpu"]   # order tried by "auto"
# device = "Vulkan1"                   # optional, see llama-server --list-devices
# threads = 8                          # CPU backend; default: physical cores

[server.runtimes]                      # optional; these are the defaults
cuda = "runtimes/llama.cpp-cuda"
vulkan = "runtimes/llama.cpp-vulkan-b11389"
cpu = "runtimes/llama.cpp-cpu-b11389"
```

The legacy `runtime_dir` key still works and means the CUDA runtime. With `auto`
the first backend of `fallback` whose runtime exists and works wins:

- `cuda` needs `ggml-cuda.dll` and a `CUDA<n>:` device in `llama-server --list-devices`.
- `vulkan` needs `ggml-vulkan.dll` and a `Vulkan<n>:` device (software renderers such
  as llvmpipe do not count).
- `cpu` needs only a runtime; without a CPU runtime folder another installed runtime
  is reused (`--device none`).

A fixed `backend = "cuda"` (or `vulkan`, `cpu`) never falls back and fails with the
reason. `sbobinai server-profile --config config.local.toml` prints the selection
(chosen backend, device, `tried` list with a reason for each skipped backend) and
`doctor` shows it. The backend keys are not part of job identity.

When the selection is a fallback, `scripts/process-input.ps1` prints a multi-line
yellow warning (backend chosen, why the others were skipped, expected slowdown),
`doctor` repeats it, the live header shows `Vulkan0 · RTX 5070 Ti (fallback)` or
`CPU (fallback)` in yellow, plain mode prints one `Backend: ...` line, and the
backend, device, threads and `fallback_used` are recorded in
`inference_configuration` (`runtime`) of `report.json`, `transcript.json`,
`metadata.json`, `run-report.md`, `source.json` and the `run.started` event. The
backend recorded is the one chosen by the launcher's selection, not queried from
the server.

### Force or simulate a backend (verified 2026-10-04 with b11193 CUDA and b11389 Vulkan)

| Goal | How |
| --- | --- |
| Force CUDA, Vulkan or CPU | `[server].backend = "cuda"`, `"vulkan"` or `"cpu"` (no fallback; error if unusable) |
| Prefer another order | `fallback = ["vulkan", "cuda", "cpu"]` |
| Simulate no CUDA device | set `CUDA_VISIBLE_DEVICES=-1` (PowerShell: `$env:CUDA_VISIBLE_DEVICES = '-1'`); `--list-devices` of the CUDA build then prints `(none)` |
| Simulate no Vulkan device | set `GGML_VK_VISIBLE_DEVICES=99` (an index that does not exist; the runtime prints `Invalid device index` and lists `(none)`) |
| Simulate CPU only | set both variables, or `backend = "cpu"` |
| Simulate a missing runtime | rename the runtime folder or point `[server.runtimes]` elsewhere |

Do not use an empty value: on Windows `$env:GGML_VK_VISIBLE_DEVICES = ''` deletes the
variable. The variables are inherited by the selection probe and by the server the
launcher starts, so set them in the same console before `transcribe.cmd`, then remove
them (`Remove-Item Env:CUDA_VISIBLE_DEVICES`). A running owned server whose backend
differs from the selection is restarted automatically.

### Launcher

`scripts/start-llamacpp.ps1` accepts `-Backend cuda|vulkan|cpu` (default `cuda`,
unchanged behaviour), `-Device` and `-Threads`. It checks the backend DLL and a listed
device for `cuda` and `vulkan` (no checks for `cpu`), then starts the server with
`--device CUDA0|Vulkan0`, or for the CPU `--device none -ngl 0 --no-mmproj-offload -t <threads>`.
`server.json` records `backend`, `device` and `threads` in its configuration; a
difference restarts the owned server. Startup logs must confirm the backend (CUDA or
Vulkan initialisation; a CPU run only warns). `stop-llamacpp.ps1` is unchanged.

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

With its default `-Backend cuda` the start script requires `ggml-cuda.dll` and an
enumerated NVIDIA CUDA device (`-Backend vulkan|cpu` are described above).
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

## Production model: Qwen3-ASR

Since 2026-10-02 the production model is Qwen3-ASR-1.7B Q8_0 with its bf16
audio projector (see the [comparison](backend-compatibility.md#model-selection-on-the-llamacpp-runtime-2026-10-02)).
Download GGUF files explicitly into the shared model store (never through a
`-hf` cache on the system drive) and keep provenance and SHA-256 beside them.
Use a separate state directory per model so manifests do not collide:

```powershell
.\scripts\start-llamacpp.ps1 -ModelPath <model-store>\Qwen3-ASR-1.7B\Qwen3-ASR-1.7B-Q8_0.gguf `
    -ProjectorPath <model-store>\Qwen3-ASR-1.7B\mmproj-Qwen3-ASR-1.7B-bf16.gguf `
    -Alias qwen3-asr-1.7b-q8 -StateDirectory $env:ProgramData\AI\services\qwen3-asr-q8
.\scripts\stop-llamacpp.ps1 -StateDirectory $env:ProgramData\AI\services\qwen3-asr-q8
```

Configure `model = "qwen3-asr-1.7b-q8"`, `response_mode = "qwen3-asr"` and
`language`. Startup logs showed 29/29 layers offloaded and the audio encoder
(`qwen3a`) on CUDA0; the server used about 3.7 GiB. `start-llamacpp.ps1` also
accepts `-ContextSize` (default 4096) and `-Parallel` (default 1). Run one model
per port at a time; stop the previous server before starting another on 8088.

### Managed by the `[server]` table (recommended)

Instead of calling the start/stop scripts by hand, describe the server once in
the `[server]` table of `config.local.toml` (runtime folder, model and
projector files, alias, state folder, port, context size, parallel slots;
relative paths resolve against `[resources].store`). `transcribe.cmd`,
`transcribe-batch.cmd` and `scripts/process-input.ps1` read it through
`audio-transcript server-profile`, start the owned server when needed and
restart it when its recorded configuration differs. A port owned by another
process is never touched. `-NoServerManagement` uses the running server as is.

## Speech detector model

The default Silero VAD detector runs on CPU through onnxruntime and needs the
optional extra plus a pinned model file:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[vad]'
.\scripts\install-silero-vad.ps1 -Destination <model-store>\silero-vad\silero_vad.onnx
```

The installer downloads tag `v6.2.3` from the official snakers4/silero-vad
repository, verifies SHA-256 `1A153A22…88E3` (full value in the script), refuses
to overwrite a different file and writes nothing to user caches. Set
`vad_model_path` in the local configuration.

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
180-second production-pipeline test; see [verification](verification.md) for
the later bounded comparisons and the full-recording run. This establishes
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

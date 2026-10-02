# Third-party notices

SbobinAI does not bundle model weights or native runtimes. The setup
commands download them from their original publishers into a folder you choose;
each download is pinned to an exact version and verified by SHA-256 (see
[`resources.json`](resources.json)). Their licenses apply to those files.

| Component | Used for | License | Source |
| --- | --- | --- | --- |
| Qwen3-ASR-1.7B (GGUF conversion by ggml-org) | Speech recognition model | Apache-2.0 | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF) |
| llama.cpp (Windows CUDA build) | Local inference server | MIT | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| NVIDIA CUDA runtime libraries (cudart package published with llama.cpp) | GPU execution | NVIDIA CUDA EULA | [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/) |
| Silero VAD (ONNX) | Speech detection | MIT | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| ONNX Runtime | Running Silero VAD on CPU | MIT | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| NumPy | Audio buffers for VAD | BSD-3-Clause | [numpy/numpy](https://github.com/numpy/numpy) |
| FFmpeg (installed separately, e.g. via winget) | Audio extraction and probing, called as a separate program | LGPL-2.1+ by default; GPL-2.0+/GPL-3.0 for builds with GPL components (e.g. gyan.dev *full*) | [ffmpeg.org](https://ffmpeg.org/legal.html) |
| Qwen2-Audio-7B (optional, legacy backend) | Alternative model | Apache-2.0 | [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |

The segmentation algorithm follows the approach of Silero VAD's
`get_speech_timestamps` and of faster-whisper's VAD wrapper (MIT), credited in
the source code.

# Release notes

A short note per release; the full list of changes is in the
[changelog](../CHANGELOG.md).

| Release | Highlights |
| --- | --- |
| [2026.11.16-critical](2026.11.16-critical.md) | The desktop app no longer hangs when it starts the model itself; live progress map, close confirmation. |
| [2026.11.05-feature](2026.11.05-feature.md) | Guided desktop app (Windows preview), selectable outputs, MP4 preview, intervals; public alpha. |
| [2026.10.04-feature](2026.10.04-feature.md) | Agent-friendly CLI, automatic CUDA → Vulkan → CPU fallback, wizard in six languages. |
| [2026.10.02-feature](2026.10.02-feature.md) | First local pipeline: Qwen3-ASR on llama.cpp, Silero VAD chunks, versioned outputs. |

Notes before 2026.11.16 were written afterwards from the git history.

## Versions

`YYYY.MM.DD-<kind>`, for example `2026.11.16-critical`: the release date plus
one word about the release, such as `feature`, `fix`, `critical` or `security`.
A second release on the same day adds `.2` after the date. The package version
(`sbobinai --version`) is just the date: `2026.11.16`.

## Making a release

1. Rename the changelog's *Unreleased* section to the new label.
2. Add a note here, copying the shape of the previous one.
3. Set the version in `pyproject.toml` and `src/audio_transcript/__init__.py`, then
   run `python scripts/gen-cli-docs.py`.

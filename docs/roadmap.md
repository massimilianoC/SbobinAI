# Roadmap ideas

**Status:** proposals only. Nothing here is implemented or promised, and priorities
may change. Current behavior is described in the [README](../README.md) and the
[desktop guide](desktop.md); the installer has its own [plan](installer-plan.md).

## Chunk length and subtitle readability

**Idea:** let users choose the maximum chunk length within a safe range, for example
10–20 seconds, around today's 15-second default.

- `chunk_seconds` already exists in the configuration file and is part of job
  identity, so a different value produces a separate version. The desktop app does
  not expose it yet.
- Accuracy comes first. Shorter chunks give the model less context, and longer ones
  may raise the risk of looping or truncated answers. Measure each value on reviewed
  reference speech before offering it, and keep any range limited to what the
  measurements support.
- Chunk length is not the only way to get shorter subtitles, and may not be the best.
  An alternative is to keep the chunk size chosen for accuracy and split long cues
  only at export time, on sentence boundaries. Without word alignment, those split
  times would be estimates and must be labeled as such.

## Settings screen

**Idea:** a gear icon in the desktop app that opens a settings screen and writes the
configuration file, so preferences do not need manual TOML editing.

- First candidates: default output formats, default language, maximum chunk length
  and maximum subtitle length.
- `output_formats` already exists in the configuration file, but the desktop app
  starts from its own preselected formats. The app should read these defaults.
- Show which settings change job identity, since those cause a new transcription.
- Validate with the same rules as the command line, write atomically and preserve
  comments or keep a backup.
- Deleting a recording stays a per-run choice. Settings must not preselect deletion
  for files found in the input folder.

## Summary and information extraction

**Idea:** an optional seventh desktop step that produces a short summary, highlights,
decisions or other extracted information from a finished transcript.

- The ASR model cannot do this: in a text-only check it repeated its input or returned
  nothing ([backend compatibility](backend-compatibility.md)). A separate text model
  is required.
- **Local first:** evaluate compact local text models, such as the Gemma family,
  through the same llama.cpp runtime. Choose the model from measured quality, Italian
  and multilingual results, speed and memory use.
- **External API, opt-in only:** an endpoint such as a hosted model could be offered
  as an explicit choice, clearly stating that the transcript leaves the computer. It
  must never be the default.
- **Memory:** on smaller GPUs the ASR model may need to be unloaded before the text
  model loads. Keep execution sequential and report which model and device were used.
- **Derived output:** write the result as a separate artifact, such as `summary.md`,
  recording model, prompt and source version. The raw transcript is never changed.
- **Quality:** long transcripts will not fit in one request, so this needs a
  chunked summarization method and a real evaluation of omissions and invented content.
  A summary must be presented as a draft to check against the transcript.

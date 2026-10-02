# Transcription instructions and reproducibility

Reviewed: 2026-10-02. Applies to the `llamacpp` backend.

## Configure a run

| Setting | Default | Purpose |
| --- | --- | --- |
| `prompt` / `--prompt` | Built-in Italian transcript-only instruction | Define the task, preserve the spoken language, avoid introductions and invented speakers. |
| `--prompt-file` | Not set | Load UTF-8 instructions from a reusable local text file. |
| `response_mode` / `--response-mode` | `json` | Constrain the response to an object with a `transcript` string. `plain` is available for comparison. |
| `temperature` / `--temperature` | `0.0` | Greedy generation; values above zero allow sampling variability. |
| `seed` / `--seed` | `42` | Fix the sampler seed for controlled comparisons. |
| `max_tokens` / `--max-tokens` | `1024` | Bound output size; reaching the limit is a failure, not a complete transcript. |
| `language` / `--language` | Not set | Give an expected language hint while preserving the actual spoken language. |
| `chunk_seconds` / `--chunk-seconds` | `30` | Fixed-duration fallback bound for audio requests. |

## Audio segmentation guidance

Qwen's [official Qwen2-Audio repository](https://github.com/QwenLM/Qwen2-Audio)
says the models currently perform best with clips under 30 seconds. This is a
maximum-duration guideline, not a recommendation to make every segment exactly
30 seconds; it does not specify a minimum duration or a VAD policy. The current
local configuration uses fixed 10-second chunks, which satisfies that guidance.

However, fixed-duration chunks can contain silence and split a phrase at an
arbitrary point. The inspected run showed both failure modes: nearly silent
opening chunks elicited invented text or a copy of the prompt, while a later
speech chunk entered a repetitive-generation loop and hit the token limit.
These are separate from GPU memory and from the 4,096-token context limit.
The model receives each audio chunk independently; earlier transcript text is
not appended to later requests.

For future long recordings, evaluate voice-activity detection and pause-aware
variable boundaries, with segments kept below 30 seconds and a conservative
fallback for quiet or uncertain speech. A small overlap can preserve words at
boundaries, but requires transcript de-duplication and source-relative timing.
This is a project evaluation direction, not a Qwen-published overlap rule.
Do not increase `max_tokens` as the only response to a repetition loop: reaching
the limit should continue to fail instead of saving an incomplete transcript.
Compare segmentation on the same reviewed audio before choosing defaults.

CLI options override TOML. Instructions are sent with each audio chunk; the
shared server has no persistent conversation that carries instructions between
jobs. Run settings participate in job identity, so a changed prompt or seed
cannot silently reuse a transcript from another experiment.

```powershell
.\scripts\run.ps1 run --config config.local.toml --file input\sample.wav --chunk-seconds 20 --response-mode json --temperature 0 --seed 42
.\scripts\run.ps1 run --config config.local.toml --file input\sample.wav --prompt-file prompts\transcription.txt --force
```

Keep private prompts in ignored local storage if they contain sensitive data.
Use `--no-archive-inputs` for comparisons that need the same source to remain
in the input queue. `--force` explicitly recomputes a completed unchanged job.

## Why structured output

The initial plain-text prompt occasionally produced an English or Italian
preface before the actual Italian transcript. A bounded local comparison found
that Italian instructions plus a constrained JSON schema avoided that behavior
on the inspected sample. The adapter exports only the `transcript` field; it
does not remove arbitrary text with regular expressions. Invalid JSON, missing
or extra fields, empty transcript and token-limit termination fail clearly.

The schema controls shape, not truth: a model could still put commentary or
incorrect words inside the transcript field. Review real output after changes
to prompts, weights, runtime or recordings. Raw responses and request settings
are saved inside each ignored job's `responses` directory, without embedding
the base64 audio in the audit record.

Greedy decoding and a fixed seed improve repeatability on the same stack.
They do not guarantee identical output across runtime versions, GPU kernels,
hardware or chunk boundaries. Preserve versions, settings, original audio and
raw responses when comparing runs. Avoid judging accuracy on synthetic tests.

## Later analysis

The MVP uses the same audio model for transcription and adds no second model.
Summarization, correction, terminology, speaker proposals or sound analysis
should be separate explicit stages/artifacts. Preserve the raw transcript and
record model, prompt, settings and segment references for any derived analysis.
Do not silently replace the spoken words with a polished summary.

See [runtime setup](runtime-setup.md), [specification](specification.md), and
[verification](verification.md) for deployment, obligations and actual evidence.

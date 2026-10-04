# Run events, live console and resource monitoring

Status: implemented (schema version 1). Section [Web UI and API](#web-ui-and-api-design-notes)
is a design note, not implemented.

Every observable step of a run is published as an **event**. The same stream feeds the
live console, the log files and, later, the web UI of the
[distribution design](distribution-design.md) (D2 API and D4a UI). Events carry numbers,
labels and the existing status lines only: never transcript or prompt text, and never
absolute paths outside the project (artifacts are named `output/<source>/<version>/...`).

## Contents

- [Console modes](#console-modes)
- [Files written for every run](#files-written-for-every-run)
- [Event envelope](#event-envelope)
- [Event types](#event-types)
- [Polling and following](#polling-and-following)
- [Resource monitor](#resource-monitor)
- [Web UI and API design notes](#web-ui-and-api-design-notes)

## Console modes

`--ui auto|live|plain|jsonl` for `run`, `watch` and `wizard` (config key `ui`, default
`auto`; not part of job identity).

| Mode | What you see |
| --- | --- |
| `live` | In-place display (redrawn at most 8 times per second): header, current file `i/n` and version, stage marks with durations, chunk progress bar with ETA and x-realtime, counters (ok, no speech, failed, fallbacks, splits, chunks in flight), audio processed, tokens, confidence, sparklines for CPU, RAM, GPU, VRAM, power and temperature, and the last events. The final frame stays on screen and is followed by a run summary. The cursor is hidden while it runs and always restored. |
| `plain` | Exactly the status lines of earlier versions (one line per chunk attempt, `Execution:` and `Timings:` lines). |
| `jsonl` | One event per line on stdout for piping into other tools; other messages go to stderr. |
| `auto` | `live` only when stdout is a terminal, `TERM` is not `dumb`, no `CI` environment variable is set and the console can render escape sequences (on Windows, virtual terminal processing is enabled through `SetConsoleMode`); otherwise `plain`. An explicit `live` still falls back to `plain` if escape sequences cannot be enabled. |

`NO_COLOR` removes colors. If the console encoding cannot encode block characters, the
display switches to ASCII (`#`, `-`, `.:-=+*#%`).

`--monitor-interval <seconds>` (default 1.0) sets the resource sampling interval and
`--no-monitor` disables sampling (config keys `monitor_interval`, `monitor`).

## Files written for every run

The files are written whatever the console mode, because native-process output is not
captured by wrappers such as PowerShell `Start-Transcript`.

| File | Content |
| --- | --- |
| `process/runs/<run_id>/events.jsonl` | Every event of the run, one JSON object per line, flushed per event. |
| `process/runs/<run_id>/status.json` | Latest snapshot (run, current job, stages, progress, resources, last activity). Rewritten atomically at most every 0.5 s and on every `run.*`, `job.*` and `stage.*` event. |
| `process/runs/<run_id>/pipeline.log` | Timestamped status lines (`[info]`, `[warning]`, `[error]`), warnings, and a final `[summary]`. |
| `process/runs/latest.json` | Pointer: `run_id`, `state` (`running` or `finished`), `started_at`, `updated_at`, `process_dir` and relative `paths`. |
| `process/<source>/<version>/events.jsonl` | The events of that job, appended across runs and resumes (each line carries its `run_id`). Resource samples are not copied. |

`run_id` is `YYYYMMDDTHHMMSSZ-<4 hex>`. The run records in `metadata.json` carry the
`run_id` of the run that produced them. These files are runtime data under `process/` and
are never committed.

## Event envelope

```json
{"schema_version":1,"seq":17,"ts":"2026-10-02T21:14:15.535Z","elapsed_s":0.057,
 "run_id":"20261002T211415Z-58da","job_id":"talk-5b1a804db346/20261002T2114Z_mock_full_5fd977fb",
 "type":"chunk.finished","data":{}}
```

| Field | Meaning |
| --- | --- |
| `schema_version` | `1`. Fields are only added within a version. |
| `seq` | Monotonic per run, starting at 1, unique across threads. |
| `ts` | UTC ISO-8601 with milliseconds. |
| `elapsed_s` | Monotonic seconds since the run started. |
| `run_id` | The run. |
| `job_id` | `<source>/<version>`; absent on run-level events. |
| `type`, `data` | See below. |

Emitting never raises into the pipeline. A failing sink is counted and reported once on
stderr; it never fails a job.

## Event types

Examples are abbreviated to `data`. Chunk indexes are zero-based (the status lines show
`index + 1`).

| Type | Payload |
| --- | --- |
| `run.started` | `command`, `model`, `backend`, `response_mode`, `language`, `parallel_requests`, `queue_size`, `ui`, optional `runtime` (the inference backend of a managed server: `backend` cuda/vulkan/cpu, `device`, `device_name`, `fallback_used`, `requested`, `threads`, `runtime_tag`, `skipped` with a reason per skipped backend; no paths) |
| `job.started` | `source` (file name), `version`, `scope` (`full` or `first600s`), `index`, `total` |
| `stage.started` | `stage` |
| `stage.finished` | `stage`, `seconds` (null when reused), optional `reused` |
| `job.prepared` | `analysed_seconds`, `speech_seconds`, `chunk_count`, `chunk_seconds_min/avg/max`, `detector`, `reused` |
| `chunk.started` | `index`, `part`, `audio_seconds`, `in_flight` (indexes), `in_flight_count` |
| `chunk.attempt` | `index`, `part`, `stage` (`base`, `temperature`, `no_context`), `temperature`, `outcome`, `reason`, `wall_seconds`, `completion_tokens`, `finish_reason`, `mean_token_prob`, `context_omitted` |
| `chunk.finished` | `index`, `outcome` (`ok`, `no_speech`, `failed`), `attempts`, `wall_seconds`, cumulative `ok`, `no_speech`, `failed`, `fallbacks`, `splits` |
| `progress` | `done_chunks`, `total_chunks`, `percent`, `audio_done_s`, `audio_total_s`, `elapsed_s`, `eta_s`, `real_time_factor`, `throughput_x`, `tokens`, `ok`, `no_speech`, `failed`, `fallbacks`, `splits`, `confidence_mean`, `in_flight` |
| `resource.sample` | see [Resource monitor](#resource-monitor) |
| `log` | `level` (`info`, `warning`, `error`), `message` (the status line text) |
| `warning` | `code`, `message` |
| `job.finished` | `source`, `status`, `counts`, `timings`, `confidence`, `artifacts`, `archived`, optional `error` (at most 200 characters) |
| `run.finished` | `exit_status`, `jobs` (`job_id`, `source`, `status`), `wall_seconds` |

Stages: `probe`, `audio_extraction`, `speech_detection`, `chunk_slicing`, `preparation`,
`inference`, `export`, `archive`, `catalog`. `stage.started` is emitted for stages that
run live (`probe`, `preparation`, `inference`, `export`, `archive`); the sub-stages of
preparation and `catalog` are reported with `stage.finished` only. When the prepared audio
of an earlier run is reused, the preparation sub-stages finish with `"seconds":null,"reused":true`.

`progress` is emitted once at the start of inference (so a resumed job shows its earlier
progress) and once per finished chunk. `eta_s`, `throughput_x` (audio seconds per wall
second) and `real_time_factor` are measured over the chunks processed in this run and are
`null` until the first chunk finishes. `percent` counts chunks that are done, silent or
failed.

Examples:

```json
{"type":"run.started","data":{"command":"run","model":"qwen3-asr-1.7b-q8","backend":"llamacpp","response_mode":"qwen3-asr","language":"it","parallel_requests":2,"queue_size":3,"ui":"live","runtime":{"backend":"vulkan","device":"Vulkan0","device_name":"NVIDIA GeForce RTX 5070 Ti","fallback_used":true,"requested":"auto","threads":null,"runtime_tag":"b11389","skipped":[{"backend":"cuda","reason":"no CUDA device reported"}]}}}
{"type":"job.started","data":{"source":"interview.m4a","version":"20261002T1200Z_qwen3-asr-1.7b-q8_full_ab12cd34","scope":"full","index":2,"total":3}}
{"type":"stage.finished","data":{"stage":"speech_detection","seconds":5.2}}
{"type":"job.prepared","data":{"analysed_seconds":850.0,"speech_seconds":662.0,"chunk_count":57,"chunk_seconds_min":0.8,"chunk_seconds_avg":11.6,"chunk_seconds_max":15.0,"detector":"silero","reused":false}}
{"type":"chunk.started","data":{"index":23,"part":"","audio_seconds":13.5,"in_flight":[22,23],"in_flight_count":2}}
{"type":"chunk.attempt","data":{"index":23,"part":"","stage":"base","temperature":0.0,"outcome":"ok","reason":null,"wall_seconds":3.8,"completion_tokens":51,"finish_reason":"stop","mean_token_prob":0.94,"context_omitted":false}}
{"type":"chunk.finished","data":{"index":23,"outcome":"ok","attempts":1,"wall_seconds":3.8,"ok":22,"no_speech":2,"failed":0,"fallbacks":1,"splits":0}}
{"type":"progress","data":{"done_chunks":24,"total_chunks":57,"percent":42.1,"audio_done_s":354.5,"audio_total_s":850.0,"elapsed_s":43.6,"eta_s":60.9,"real_time_factor":0.123,"throughput_x":8.13,"tokens":1291.0,"ok":22,"no_speech":2,"failed":0,"fallbacks":1,"splits":0,"confidence_mean":0.9312,"in_flight":[24]}}
{"type":"resource.sample","data":{"scope":"system","source":{"host":"windows","gpu":"nvidia-smi"},"cpu_percent":23.4,"ram_used_mb":12288.0,"ram_total_mb":32768.0,"process_rss_mb":410.0,"gpu_util_percent":87.0,"gpu_mem_used_mb":4123.0,"gpu_mem_total_mb":16303.0,"gpu_power_w":182.5,"gpu_temp_c":61.0}}
{"type":"log","data":{"level":"info","message":"Chunk 24/57 [338.5-351.0s, 12.5s audio] t=0 ok 3.8s 51 tok"}}
{"type":"warning","data":{"code":"archive_not_done","message":"The input was not archived; see archive_warning in metadata.json"}}
{"type":"job.finished","data":{"source":"interview.m4a","status":"completed","counts":{"total_chunks":57,"ok":55,"no_speech":2,"failed":0,"fallbacks":1,"splits":0,"tokens":2310.0},"timings":{"wall_seconds":210.4,"stages":{"probe":0.4,"preparation":8.7,"inference":196.2,"export":0.05,"catalog":0.3}},"confidence":0.9312,"artifacts":["output/interview-1a2b3c4d5e6f/20261002T1200Z_qwen3-asr-1.7b-q8_full_ab12cd34/transcript.txt"],"archived":true}}
{"type":"run.finished","data":{"exit_status":0,"jobs":[{"job_id":"interview-1a2b3c4d5e6f/20261002T1200Z_qwen3-asr-1.7b-q8_full_ab12cd34","source":"interview.m4a","status":"completed"}],"wall_seconds":212.0}}
```

## Polling and following

- **Poll** `process/runs/latest.json` for the active `run_id`, then read
  `process/runs/<run_id>/status.json` every second or so. It always holds the latest
  progress and resources of the current job and the finished jobs of the run.
- **Follow** with the command line:

  ```powershell
  audio-transcript events --follow                          # latest run, until run.finished
  audio-transcript events --run 20261002T211415Z-58da       # a finished run, all events
  audio-transcript events --follow --type progress          # only events whose type starts with 'progress'
  ```

  `events` prints the JSONL lines unchanged, polls the file like `tail -f`, and stops after
  `run.finished`. Use `--config` or `--process-dir` when the process folder is not the
  default one.

## Resource monitor

A background thread samples every `monitor_interval` seconds and publishes
`resource.sample`. Fields that cannot be read are `null`; a failing sampler never fails the
run and produces one `warning`.

| Field | Source |
| --- | --- |
| `cpu_percent`, `ram_used_mb`, `ram_total_mb`, `process_rss_mb` | `psutil` if importable; otherwise `GetSystemTimes`, `GlobalMemoryStatusEx` and `GetProcessMemoryInfo` through `ctypes` (Windows) or `/proc/stat`, `/proc/meminfo` and `/proc/self/status` (Linux). |
| `gpu_util_percent`, `gpu_mem_used_mb`, `gpu_mem_total_mb`, `gpu_power_w`, `gpu_temp_c` | One long-lived `nvidia-smi --query-gpu=... --format=csv,noheader,nounits -lms <ms>` process when `nvidia-smi` is on `PATH` (restarted up to three times if it exits, terminated at the end of the run). The first GPU is reported. |
| `scope` | Always `"system"`: CPU, RAM and the GPU are shared with every other process, so these numbers are not attributable to this run. `process_rss_mb` is the pipeline process only (not llama-server). |
| `source` | Which sampler produced the values, for example `{"host":"windows","gpu":"nvidia-smi"}`. |

## Web UI and API design notes

Design only; nothing below is implemented.

- The D2 `serve` worker owns one `EventBus` per run and adds an in-memory ring buffer
  sink (the last N events per job) next to the file sinks. The `events.jsonl` files remain
  the source of truth for replay.
- `GET /v1/jobs/{job_id}/events` is a Server-Sent Events stream. Each event is sent as
  `id: <seq>`, `event: <type>` and `data: <the JSON envelope>`. A client reconnects with
  `Last-Event-ID` and the server replays from the per-job `events.jsonl` (or the ring
  buffer) before switching to live events. `resource.sample` events are run-level and can be
  offered as a separate `GET /v1/runs/{run_id}/events` stream.
- `GET /v1/jobs/{job_id}` maps to the job part of `status.json` (`progress` is
  `{done_chunks, total_chunks, percent}` plus ETA); a plain polling UI needs nothing else.
- A WebSocket endpoint would carry the same envelopes; SSE is preferred for the browser UI
  because it needs no extra dependency, reconnects by itself and passes proxies. Both can
  be served from the optional `server` extra while the core stays standard library only.
- The D4a UI renders the same state as the console: it reduces events with the same rules
  as `RunState` (see `application/progress.py`), so the console, `status.json` and the web
  display cannot disagree.
- Security follows D2: loopback by default, token for other bindings, no transcript text in
  events.

"""Human-readable duration formatting and the per-version Markdown run report.

The run report contains numbers, names and timestamps only; it never includes transcript,
prompt or model output text.
"""

from __future__ import annotations

import math

from .backend_select import describe_runtime

STAGE_ORDER = (
    "probe",
    "audio_extraction",
    "speech_detection",
    "chunk_slicing",
    "preparation",
    "inference",
    "export",
    "archive",
    "catalog",
)


def format_duration(seconds: float | None) -> str:
    """Compact human time: ``48 ms``, ``0.4 s``, ``59.9 s``, ``1 min 01 s``, ``1 h 05 min 03 s``."""
    if seconds is None or isinstance(seconds, bool) or not math.isfinite(seconds):
        return "n/a"
    seconds = max(0.0, float(seconds))
    if seconds < 0.1:
        # Sub-0.1 s stages (export, catalog) would otherwise all read "0.0 s".
        return f"{seconds * 1000:.0f} ms"
    if round(seconds, 1) < 59.95 and seconds < 60:
        return f"{seconds:.1f} s"
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, second = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {second:02d} s"
    return f"{minutes} min {second:02d} s"


def format_timing(seconds: float | None) -> str:
    """Human time followed by exact seconds, e.g. ``4 min 22 s (262.4 s)``."""
    if seconds is None or isinstance(seconds, bool) or not math.isfinite(seconds):
        return "n/a"
    human = format_duration(seconds)
    if human.endswith(" ms"):
        return human
    exact = f"{max(0.0, float(seconds)):.1f} s"
    return exact if human == exact else f"{human} ({exact})"


def sum_stages(runs: list[dict]) -> dict:
    """Add the per-run stage timings; a stage no run could measure stays None."""
    totals: dict[str, float | None] = {}
    for run in runs:
        stages = run.get("stages_seconds") if isinstance(run, dict) else None
        if not isinstance(stages, dict):
            continue
        for key, value in stages.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                totals.setdefault(key, None)
                continue
            totals[key] = round((totals.get(key) or 0.0) + float(value), 3)
    return totals


def timings_line(total: float | None, stages: dict) -> str:
    parts = [f"total {format_duration(total)}"]
    for label, key in (("preparation", "preparation"), ("inference", "inference")):
        if stages.get(key) is not None:
            parts.append(f"{label} {format_duration(stages[key])}")
    if stages.get("export") is not None:
        parts.append(f"export {format_duration(stages['export'])}")
    return "Timings: " + " | ".join(parts)


def _scope_text(scope: object) -> str:
    if isinstance(scope, dict) and scope.get("kind") == "bounded":
        limit = scope.get("max_duration_seconds")
        return f"first {limit:g} s (bounded run)" if isinstance(limit, (int, float)) else "bounded"
    if isinstance(scope, dict) and scope.get("kind") == "full":
        return "full"
    return "unknown"


def _percent(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "unknown"
    return f"{value:g}%"


def _stats(block: object) -> str:
    if not isinstance(block, dict):
        return "n/a"
    return (
        f"mean {format_timing(block.get('mean'))}, p95 {format_timing(block.get('p95'))}, "
        f"max {format_timing(block.get('max'))}"
    )


def _backend_lines(meta: dict) -> list[str]:
    """Run-report line for the inference backend; empty for an unmanaged server."""
    configuration = meta.get("inference_configuration")
    runtime = configuration.get("runtime") if isinstance(configuration, dict) else None
    line = describe_runtime(runtime if isinstance(runtime, dict) else None)
    if line is None:
        return []
    return [f"- Inference {line[0].lower()}{line[1:]}"]


def render_run_report(
    meta: dict, execution: dict | None, *, completion_percent: float | None
) -> str:
    """Render the Markdown run report for one version from its metadata and execution block."""
    execution = execution if isinstance(execution, dict) else {}
    runs = [run for run in meta.get("runs", []) if isinstance(run, dict)]
    stages = execution.get("stages_seconds")
    if not isinstance(stages, dict):
        stages = sum_stages(runs)
    total_wall = sum(
        float(run["wall_seconds"])
        for run in runs
        if isinstance(run.get("wall_seconds"), (int, float))
    )
    audio = meta.get("audio_metadata") if isinstance(meta.get("audio_metadata"), dict) else {}
    analysed = execution.get("analysed_audio_seconds", audio.get("duration_seconds"))
    speech = audio.get("speech_seconds")
    inference = stages.get("inference")
    rtf = execution.get("real_time_factor")
    lines = [
        "# Run report",
        "",
        f"- Source file: {meta.get('source_name', 'unknown')}",
        f"- Version: {meta.get('version_folder', 'unknown')}",
        f"- Model: {meta.get('model', 'unknown')}",
        f"- Response mode: {meta.get('response_mode') or 'n/a'}",
        *_backend_lines(meta),
        f"- Language: {meta.get('language') or 'auto'}",
        f"- Scope: {_scope_text(meta.get('scope'))}",
        f"- Status: {meta.get('status', 'unknown')}",
        f"- Completion: {_percent(completion_percent)}",
        "",
        "## Timings",
        "",
        "| Stage | Time |",
        "| --- | --- |",
        f"| Total wall time of all runs | {format_timing(total_wall if runs else None)} |",
    ]
    separable = any(stages.get(key) is not None for key in STAGE_ORDER[1:4])
    for label, key in (
        ("Preparation (probe + extraction + detection + slicing)", "preparation"),
        ("- probe", "probe"),
        ("- audio extraction", "audio_extraction"),
        ("- speech detection", "speech_detection"),
        ("- chunk slicing", "chunk_slicing"),
        ("Inference stage wall time (incl. per-chunk bookkeeping)", "inference"),
        ("Export", "export"),
        ("Archive", "archive"),
        ("Catalog bookkeeping", "catalog"),
    ):
        value = stages.get(key)
        if value is None and key in {"audio_extraction", "speech_detection", "chunk_slicing"}:
            if not separable and stages.get("preparation") is not None:
                lines.append(f"| {label} | not separable (processor reports only the total) |")
                continue
        if value is None and key not in {"preparation", "inference"}:
            continue
        lines.append(f"| {label} | {format_timing(value)} |")
    requests = execution.get("request_seconds_sum", execution.get("inference_wall_seconds"))
    parallel = execution.get("parallel_requests")
    overlapping = isinstance(parallel, int) and parallel > 1
    if isinstance(requests, (int, float)) and not isinstance(requests, bool):
        label = "Sum of model request times"
        if overlapping:
            label += " (concurrent requests overlap, so this exceeds wall time)"
        lines.append(f"| {label} | {format_timing(requests)} |")
    lines.append(f"| Per chunk (inference) | {_stats(execution.get('chunk_inference_seconds'))} |")
    lines.extend(
        [
            "",
            f"- Analysed audio: {format_timing(analysed)}",
            f"- Speech detected: {format_timing(speech)}",
        ]
    )
    if isinstance(rtf, (int, float)) and rtf > 0:
        speed = 1 / rtf
        word = "faster" if speed >= 1 else "slower"
        factor = speed if speed >= 1 else rtf
        lines.append(f"- Real-time factor: {rtf:g} (inference {factor:.1f}x {word} than real time)")
    elif inference is not None:
        lines.append("- Real-time factor: n/a")
    if isinstance(parallel, int) and not isinstance(parallel, bool):
        lines.append(f"- Parallel requests: {parallel}")
    concurrency = execution.get("effective_concurrency")
    if isinstance(concurrency, (int, float)) and not isinstance(concurrency, bool):
        lines.append(
            f"- Effective concurrency: {concurrency:g}x "
            "(sum of model request times / inference stage wall time)"
        )
    if overlapping:
        lines.append(
            "- Note: batched GPU execution of concurrent requests may change greedy output "
            "slightly compared with sequential requests."
        )
    tokens = execution.get("completion_tokens")
    tokens = tokens if isinstance(tokens, dict) else {}
    confidence = execution.get("confidence")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Chunks: total {execution.get('total_chunks', 'n/a')}, "
            f"ok {execution.get('chunks_ok', 'n/a')}, "
            f"no speech {execution.get('chunks_no_speech', 'n/a')}, "
            f"failed {execution.get('chunks_failed', 'n/a')}",
            f"- Temperature fallbacks used: {execution.get('temperature_fallbacks_used', 'n/a')}",
            f"- Splits used: {execution.get('splits_used', 'n/a')}",
            f"- Completion tokens: sum {tokens.get('sum', 'n/a')}, "
            f"max {tokens.get('max', 'n/a')}, p95 {tokens.get('p95', 'n/a')}",
        ]
    )
    if isinstance(confidence, dict):
        lines.append(
            "- Confidence (uncalibrated mean token probability): "
            f"mean {confidence.get('mean')}, 10th percentile chunk {confidence.get('p10_chunk')}, "
            f"chunks below 0.5: {confidence.get('low_chunks')}"
        )
    else:
        lines.append("- Confidence: unavailable (token probabilities were not collected)")
    if meta.get("error"):
        lines.extend(["", f"- Last error: {meta['error']}"])
    lines.extend(["", "## Runs", ""])
    if runs:
        lines.extend(
            ["| Start (UTC) | End (UTC) | Wall time | Outcome |", "| --- | --- | --- | --- |"]
        )
        for run in runs:
            wall = run.get("wall_seconds")
            lines.append(
                f"| {run.get('started_at', 'n/a')} | {run.get('ended_at') or 'in progress'} | "
                f"{format_timing(wall if isinstance(wall, (int, float)) else None)} | "
                f"{run.get('status', 'n/a')} |"
            )
        lines.append("")
        lines.append(f"Cumulative wall time over {len(runs)} run(s): {format_timing(total_wall)}")
    else:
        lines.append("No run recorded.")
    return "\n".join(lines) + "\n"

"""Pure pause-aware chunk planning from frame-level speech probabilities.

Speech onset/offset hysteresis and the "cut a too-long run at its last short
silence" idea follow Silero VAD's ``get_speech_timestamps`` (MIT,
https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py).
"""

from __future__ import annotations

import math

from .models import SegmentationSettings, SpeechRegion, VoiceActivity

HYSTERESIS = 0.15
_EPSILON = 1e-6


def plan_chunks(
    activity: VoiceActivity, total_seconds: float, settings: SegmentationSettings
) -> list[SpeechRegion]:
    """Return ordered, non-overlapping speech chunks of at most ``max_chunk_seconds``."""
    fs = activity.frame_seconds
    max_chunk = settings.max_chunk_seconds
    if not (math.isfinite(fs) and fs > 0):
        raise ValueError("frame_seconds must be a finite number greater than zero.")
    if not (math.isfinite(max_chunk) and max_chunk > 0):
        raise ValueError("max_chunk_seconds must be a finite number greater than zero.")
    if not (math.isfinite(total_seconds) and total_seconds > 0):
        return []

    count = min(len(activity.probabilities), math.ceil(total_seconds / fs - 1e-9))
    probabilities = activity.probabilities[:count]
    onset = activity.threshold
    offset = max(0.0, onset - HYSTERESIS)

    pad = min(max(0.0, settings.speech_pad_seconds), max_chunk / 4)
    longest_speech = max(fs, max_chunk - 2 * pad - _EPSILON)
    max_frames = max(1, math.floor(longest_speech / fs + 1e-9))
    end_frames = max(1, math.ceil(settings.min_silence_seconds / fs - 1e-9))
    cut_frames = max(1, math.floor(settings.min_silence_seconds / fs + 1e-9))
    min_speech_frames = max(1, math.ceil(settings.min_speech_seconds / fs - 1e-9))

    runs: list[tuple[int, int]] = []
    for start, end in _speech_runs(probabilities, onset, offset, end_frames):
        runs.extend(_split_long_run(probabilities, start, end, max_frames, offset, cut_frames))
    runs = [run for run in runs if run[1] - run[0] >= min_speech_frames]
    if not runs:
        return []

    spans = [(min(a * fs, total_seconds), min(b * fs, total_seconds)) for a, b in runs]
    padded = _pad(spans, pad, total_seconds)
    merged = _merge(padded, settings.max_merge_gap_seconds, max_chunk)
    return [SpeechRegion(start, end) for start, end in merged if end > start]


def _speech_runs(
    probabilities: tuple[float, ...] | list[float], onset: float, offset: float, end_frames: int
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    triggered = False
    run_start = 0
    low_start: int | None = None
    for index, probability in enumerate(probabilities):
        if not triggered:
            if probability >= onset:
                triggered, run_start, low_start = True, index, None
            continue
        if probability < offset:
            if low_start is None:
                low_start = index
            if index - low_start + 1 >= end_frames:
                runs.append((run_start, low_start))
                triggered, low_start = False, None
        else:
            low_start = None
    if triggered:
        runs.append((run_start, low_start if low_start is not None else len(probabilities)))
    return runs


def _split_long_run(
    probabilities: tuple[float, ...] | list[float],
    start: int,
    end: int,
    max_frames: int,
    offset: float,
    cut_frames: int,
) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    while end - start > max_frames:
        window_end = start + max_frames
        cut_start = cut_end = None
        index = start + 1
        while index <= window_end:
            if index < end and probabilities[index] < offset:
                dip_end = index
                while dip_end < end and probabilities[dip_end] < offset:
                    dip_end += 1
                if dip_end - index >= cut_frames:
                    cut_start, cut_end = index, dip_end
                index = dip_end
            else:
                index += 1
        if cut_start is not None:
            pieces.append((start, cut_start))
            start = cut_end
            continue
        lower = max(start + 1, start + max_frames // 2)
        best = lower
        for index in range(lower, window_end + 1):
            if probabilities[index] <= probabilities[best]:
                best = index
        pieces.append((start, best))
        start = best
    if end > start:
        pieces.append((start, end))
    return pieces


def _pad(
    spans: list[tuple[float, float]], pad: float, total_seconds: float
) -> list[tuple[float, float]]:
    starts = [max(0.0, start - pad) for start, _ in spans]
    ends = [min(total_seconds, end + pad) for _, end in spans]
    for index in range(len(spans) - 1):
        if starts[index + 1] < ends[index]:
            boundary = (spans[index][1] + spans[index + 1][0]) / 2
            ends[index] = starts[index + 1] = boundary
    return list(zip(starts, ends, strict=True))


def _merge(
    spans: list[tuple[float, float]], max_gap: float, max_chunk: float
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged:
            previous_start, previous_end = merged[-1]
            if start - previous_end <= max_gap and end - previous_start <= max_chunk:
                merged[-1] = (previous_start, end)
                continue
        merged.append((start, end))
    return [(start, min(end, start + max_chunk)) for start, end in merged]

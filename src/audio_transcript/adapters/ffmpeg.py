"""FFmpeg based media probing and speech-aware audio chunk preparation."""

from __future__ import annotations

import array
import dataclasses
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from audio_transcript.domain.models import (
    AudioChunk,
    MediaInfo,
    PreparedAudio,
    SegmentationSettings,
    TranscriptionError,
)
from audio_transcript.domain.ports import SpeechDetector
from audio_transcript.domain.segmentation import plan_chunks

MIN_SPLIT_SECONDS = 2.0


class FFmpegProcessor:
    """Extract mono PCM WAV chunks without loading source media into memory."""

    def __init__(
        self,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        detector: SpeechDetector | None = None,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.detector = detector

    def check(self) -> None:
        """Raise an actionable error when either FFmpeg executable is missing."""
        for executable in (self.ffmpeg, self.ffprobe):
            try:
                result = subprocess.run(
                    [executable, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise TranscriptionError(
                    f"Required executable '{executable}' was not found. Install FFmpeg "
                    "and ensure both ffmpeg and ffprobe are available on PATH."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise TranscriptionError(f"'{executable}' did not respond to -version.") from exc
            if result.returncode != 0:
                raise TranscriptionError(
                    f"'{executable} -version' failed: {_diagnostic(result.stderr or result.stdout)}"
                )

    def probe(self, source: Path) -> MediaInfo:
        source = Path(source)
        if not source.is_file():
            raise TranscriptionError(f"Media source does not exist or is not a file: {source}")
        command = [
            self.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,sample_rate,channels,duration",
            "-of",
            "json",
            str(source),
        ]
        result = _run(command, timeout=30, operation="probe media")
        try:
            metadata = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise TranscriptionError("ffprobe returned invalid media metadata.") from exc
        if not isinstance(metadata, dict):
            raise TranscriptionError("ffprobe returned invalid media metadata.")
        streams = metadata.get("streams", [])
        if not isinstance(streams, list):
            raise TranscriptionError("ffprobe returned invalid stream metadata.")
        audio = next(
            (
                item
                for item in streams
                if isinstance(item, dict) and item.get("codec_type") == "audio"
            ),
            None,
        )
        if audio is None:
            raise TranscriptionError(f"Media contains no audio stream: {source.name}")
        format_info = metadata.get("format", {})
        if not isinstance(format_info, dict):
            format_info = {}
        duration = _positive_float(audio.get("duration")) or _positive_float(
            format_info.get("duration")
        )
        if duration is None:
            raise TranscriptionError(
                "ffprobe could not determine a finite, positive audio duration."
            )
        try:
            sample_rate = int(audio["sample_rate"])
            channels = int(audio["channels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError("ffprobe returned incomplete audio stream metadata.") from exc
        if sample_rate <= 0 or channels <= 0:
            raise TranscriptionError("ffprobe returned invalid audio sample rate or channel count.")
        return MediaInfo(
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            codec=str(audio.get("codec_name") or "unknown"),
        )

    def prepare(
        self,
        source: Path,
        destination: Path,
        *,
        sample_rate: int,
        segmentation: SegmentationSettings,
        max_duration: float | None = None,
    ) -> PreparedAudio:
        source, destination = Path(source), Path(destination)
        if not source.is_file():
            raise TranscriptionError(f"Media source does not exist or is not a file: {source}")
        max_chunk = segmentation.max_chunk_seconds
        if not math.isfinite(max_chunk) or max_chunk <= 0:
            raise TranscriptionError("max_chunk_seconds must be a finite number greater than zero.")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise TranscriptionError("sample_rate must be a positive integer.")
        if max_duration is not None and (not math.isfinite(max_duration) or max_duration <= 0):
            raise TranscriptionError("max_duration must be a finite number greater than zero.")
        max_chunk_frames = math.floor(max_chunk * sample_rate)
        if max_chunk_frames <= 0:
            raise TranscriptionError("max_chunk_seconds is shorter than one output audio frame.")
        max_frames = None
        if max_duration is not None:
            max_frames = math.floor(max_duration * sample_rate)
            if max_frames <= 0:
                raise TranscriptionError(
                    "The duration limit is shorter than one output audio frame."
                )
        detector = None
        if segmentation.detector != "none":
            detector = self.detector
            if detector is None or detector.name != segmentation.detector:
                raise TranscriptionError(
                    f"Segmentation detector '{segmentation.detector}' is not available in this "
                    "media processor. Configure it or choose segmentation 'none'."
                )
            detector.check()

        destination.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
        ]
        trim_filter = ""
        if max_frames is not None:
            command.extend(["-t", _format_seconds(max_frames / sample_rate)])
            trim_filter = f",atrim=end_sample={max_frames}"
        command.extend(
            [
                "-af",
                f"aresample={sample_rate}{trim_filter}",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
            ]
        )

        with tempfile.TemporaryDirectory(prefix="audio-transcript-", dir=destination) as temp_name:
            temp = Path(temp_name)
            full_path = temp / "full.wav"
            timings: dict[str, float] = {}
            stage_started = time.monotonic()
            _run(command + [str(full_path)], timeout=3600, operation="prepare audio")
            total_frames, rate = _wav_shape(full_path, sample_rate)
            if total_frames <= 0:
                raise TranscriptionError("FFmpeg produced no audio; the source may be empty.")
            if max_frames is not None and total_frames > max_frames:
                raise TranscriptionError("Prepared audio exceeded max_duration.")
            duration = total_frames / rate
            timings["audio_extraction"] = round(time.monotonic() - stage_started, 3)
            stage_started = time.monotonic()

            if detector is None:
                spans = [
                    (start, min(start + max_chunk_frames, total_frames))
                    for start in range(0, total_frames, max_chunk_frames)
                ]
                record = {"detector": "none", "regions": len(spans)}
            else:
                activity = detector.detect(full_path)
                regions = plan_chunks(activity, duration, segmentation)
                spans = []
                cursor = 0
                for region in regions:
                    start = max(cursor, round(region.start * rate))
                    end = min(round(region.end * rate), start + max_chunk_frames, total_frames)
                    if end > start:
                        spans.append((start, end))
                        cursor = end
                record = {
                    "detector": detector.name,
                    "activity_threshold": activity.threshold,
                    "frame_seconds": activity.frame_seconds,
                    "details": activity.details,
                    "regions": len(spans),
                }

            timings["speech_detection"] = round(time.monotonic() - stage_started, 3)
            stage_started = time.monotonic()
            staged = _slice_wav(full_path, temp, spans, rate)
            full_path.unlink()
            chunks: list[AudioChunk] = []
            for index, (path, (start, end)) in enumerate(zip(staged, spans, strict=True)):
                final_path = destination / f"chunk_{index:06d}.wav"
                os.replace(path, final_path)
                chunks.append(
                    AudioChunk(path=final_path, start=start / rate, end=end / rate, index=index)
                )

            timings["chunk_slicing"] = round(time.monotonic() - stage_started, 3)

        record["settings"] = dataclasses.asdict(segmentation)
        return PreparedAudio(
            chunks=tuple(chunks),
            duration=duration,
            speech_seconds=sum(chunk.end - chunk.start for chunk in chunks),
            segmentation=record,
            timings=timings,
        )

    def export_audio(self, source: Path, destination: Path, *, sample_rate: int) -> MediaInfo:
        """Write the first audio stream as 16-bit mono FLAC (lossless) and probe it."""
        source, destination = Path(source), Path(destination)
        if not source.is_file():
            raise TranscriptionError(f"Media source does not exist or is not a file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            "-c:a",
            "flac",
            "-compression_level",
            "8",
            "-f",
            "flac",
            str(partial),
        ]
        try:
            _run(command, timeout=3600, operation="export audio")
            info = self.probe(partial)
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
        return info

    def split(self, chunk: AudioChunk, destination: Path) -> list[AudioChunk]:
        """Split a prepared chunk at its quietest ~30 ms point in the middle 30-70 %."""
        destination = Path(destination)
        try:
            with wave.open(str(chunk.path), "rb") as wav:
                rate = wav.getframerate()
                width = wav.getsampwidth()
                channels = wav.getnchannels()
                frames = wav.getnframes()
                raw = wav.readframes(frames)
        except (wave.Error, OSError) as exc:
            raise TranscriptionError(f"Could not read chunk audio to split it: {exc}") from exc
        if width != 2 or channels != 1:
            raise TranscriptionError("Only mono 16-bit PCM chunks can be split.")
        if frames / rate < MIN_SPLIT_SECONDS:
            raise TranscriptionError(
                f"Chunk {chunk.index}{chunk.part} is shorter than {MIN_SPLIT_SECONDS:.1f} s "
                "and cannot be split further."
            )
        samples = array.array("h")
        samples.frombytes(raw[: frames * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        cut = _quietest_point(samples, rate)
        cut_time = chunk.start + cut / rate
        destination.mkdir(parents=True, exist_ok=True)
        halves = []
        for suffix, (first, last), (start, end) in (
            ("a", (0, cut), (chunk.start, cut_time)),
            ("b", (cut, frames), (cut_time, chunk.end)),
        ):
            part = chunk.part + suffix
            path = destination / f"chunk_{chunk.index:06d}{part}.wav"
            _write_wav(path, rate, raw[first * 2 : last * 2])
            halves.append(AudioChunk(path=path, start=start, end=end, index=chunk.index, part=part))
        return halves


def _wav_shape(path: Path, sample_rate: int) -> tuple[int, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
            channels, width = wav.getnchannels(), wav.getsampwidth()
    except (wave.Error, OSError) as exc:
        raise TranscriptionError("FFmpeg created an invalid WAV file.") from exc
    if rate != sample_rate or channels != 1 or width != 2:
        raise TranscriptionError("FFmpeg produced an unexpected WAV format.")
    return frames, rate


def _write_wav(path: Path, rate: int, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)


def _slice_wav(
    source: Path, directory: Path, spans: list[tuple[int, int]], rate: int
) -> list[Path]:
    paths: list[Path] = []
    with wave.open(str(source), "rb") as wav:
        for index, (start, end) in enumerate(spans):
            wav.setpos(start)
            path = directory / f"slice_{index:06d}.wav"
            _write_wav(path, rate, wav.readframes(end - start))
            paths.append(path)
    return paths


def _quietest_point(samples: array.array, rate: int) -> int:
    """Return the centre sample of the quietest ~30 ms window in the middle 30-70 %."""
    count = len(samples)
    window = max(1, round(0.03 * rate))
    low = max(0, int(count * 0.3) - window // 2)
    high = min(count - window, int(count * 0.7) - window // 2)
    if high < low:
        return count // 2
    energy = [0]
    energy.extend(itertools.accumulate(sample * sample for sample in samples))
    middle = count // 2 - window // 2
    step = max(1, rate // 1000)
    best = min(
        range(low, high + 1, step),
        key=lambda start: (energy[start + window] - energy[start], abs(start - middle)),
    )
    return min(max(best + window // 2, 1), count - 1)


def _run(command: list[str], *, timeout: float, operation: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        raise TranscriptionError(
            f"Could not {operation}: executable '{command[0]}' was not found. "
            "Install FFmpeg and ensure it is on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"Timed out while trying to {operation}.") from exc
    if result.returncode != 0:
        message = _diagnostic(result.stderr or result.stdout)
        raise TranscriptionError(
            f"Could not {operation}: {message or 'FFmpeg exited with an error.'}"
        )
    return result


def _diagnostic(message: str | None) -> str:
    if not message:
        return ""
    return message.strip()[-1200:]


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _format_seconds(value: float) -> str:
    return format(value, ".9g")

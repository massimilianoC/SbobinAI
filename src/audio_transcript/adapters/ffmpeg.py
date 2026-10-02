"""FFmpeg based media probing and bounded audio chunk preparation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from audio_transcript.domain.models import AudioChunk, MediaInfo, TranscriptionError


class FFmpegProcessor:
    """Extract mono PCM WAV chunks without loading source media into memory."""

    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

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
        chunk_seconds: float,
        sample_rate: int,
        max_duration: float | None = None,
    ) -> list[AudioChunk]:
        source, destination = Path(source), Path(destination)
        if not source.is_file():
            raise TranscriptionError(f"Media source does not exist or is not a file: {source}")
        if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
            raise TranscriptionError("chunk_seconds must be a finite number greater than zero.")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise TranscriptionError("sample_rate must be a positive integer.")
        if max_duration is not None and (not math.isfinite(max_duration) or max_duration <= 0):
            raise TranscriptionError("max_duration must be a finite number greater than zero.")

        chunk_frames = math.floor(chunk_seconds * sample_rate)
        if chunk_frames <= 0:
            raise TranscriptionError("chunk_seconds is shorter than one output audio frame.")
        max_frames = None
        if max_duration is not None:
            max_frames = math.floor(max_duration * sample_rate)
            if max_frames <= 0:
                raise TranscriptionError(
                    "The duration limit is shorter than one output audio frame."
                )
            chunk_frames = min(chunk_frames, max_frames)
        effective_chunk_seconds = chunk_frames / sample_rate

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
            trim_filter = f"atrim=end_sample={max_frames},"
        command.extend(
            [
                "-af",
                f"aresample={sample_rate},{trim_filter}"
                f"asetnsamples=n={chunk_frames}:p=0,asetpts=N/SR/TB",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "segment",
                "-segment_time",
                _format_seconds(effective_chunk_seconds),
                "-segment_format",
                "wav",
                "-reset_timestamps",
                "1",
            ]
        )

        try:
            with tempfile.TemporaryDirectory(
                prefix="audio-transcript-", dir=destination
            ) as temp_name:
                pattern = str(Path(temp_name) / "segment_%08d.wav")
                result = _run(command + [pattern], timeout=3600, operation="prepare audio")
                del result
                generated = sorted(Path(temp_name).glob("segment_*.wav"))
                if not generated:
                    raise TranscriptionError(
                        "FFmpeg produced no audio chunks; the source may be silent."
                    )
                max_chunks = (
                    math.ceil(max_frames / chunk_frames) if max_frames is not None else None
                )
                if max_chunks is not None and len(generated) > max_chunks:
                    raise TranscriptionError(
                        "FFmpeg produced more chunks than the configured maximum duration permits."
                    )

                chunks: list[AudioChunk] = []
                cursor_frames = 0
                for index, item in enumerate(generated):
                    try:
                        with wave.open(str(item), "rb") as wav:
                            frames = wav.getnframes()
                            actual_rate = wav.getframerate()
                            channels = wav.getnchannels()
                            width = wav.getsampwidth()
                    except (wave.Error, OSError) as exc:
                        raise TranscriptionError(
                            f"FFmpeg created an invalid WAV chunk: {item.name}"
                        ) from exc
                    if frames <= 0:
                        continue
                    if actual_rate != sample_rate or channels != 1 or width != 2:
                        raise TranscriptionError(
                            f"FFmpeg produced unexpected WAV format in chunk {item.name}."
                        )
                    if frames > chunk_frames:
                        raise TranscriptionError(
                            "FFmpeg produced a chunk longer than the configured chunk duration."
                        )
                    if max_frames is not None and cursor_frames + frames > max_frames:
                        raise TranscriptionError("Prepared audio exceeded max_duration.")
                    final_path = destination / f"chunk_{index:06d}.wav"
                    os.replace(item, final_path)
                    start = cursor_frames / actual_rate
                    cursor_frames += frames
                    end = cursor_frames / actual_rate
                    chunks.append(AudioChunk(path=final_path, start=start, end=end, index=index))
                if not chunks:
                    raise TranscriptionError(
                        "FFmpeg produced no non-empty audio chunks; the source may be silent."
                    )
                return chunks
        except TranscriptionError:
            raise


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

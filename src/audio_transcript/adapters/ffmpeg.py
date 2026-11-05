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
from collections.abc import Callable
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
# NVENC variable-bitrate quality for the preview render. Measured on a generated 1080p
# clip scaled to 720p: cq 34 gave about the size of libx264 crf 26 at 0.6 dB lower PSNR.
NVENC_CQ = 34
_NVENC_PROBE: dict[str, bool] = {}


def _path_label(attempt: dict) -> str:
    if attempt["encoder"] == "libx264":
        return "the CPU encoder (libx264)"
    return "NVENC with " + ("GPU decoding" if attempt["gpu_decode"] else "CPU decoding")


def _short_reason(exc: Exception) -> str:
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    return (lines[-1] if lines else "unknown error")[:160]


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
            "format=duration:stream=codec_type,codec_name,sample_rate,channels,duration,height"
            ":stream_disposition=attached_pic",
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
        video = next(
            (
                item
                for item in streams
                if isinstance(item, dict)
                and item.get("codec_type") == "video"
                and not (item.get("disposition") or {}).get("attached_pic")
            ),
            None,
        )
        height = video.get("height") if video else None
        return MediaInfo(
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            codec=str(audio.get("codec_name") or "unknown"),
            video_codec=str(video.get("codec_name") or "unknown") if video else None,
            video_height=height if isinstance(height, int) and height > 0 else None,
        )

    def _encoder_listing(self) -> str:
        return _run(
            [self.ffmpeg, "-hide_banner", "-encoders"], timeout=30, operation="list encoders"
        ).stdout

    def nvenc_usable(self) -> bool:
        """Whether h264_nvenc really encodes here (one real 1-frame probe, cached per process).

        The encoder list alone says nothing about a driver, a GPU or a free encode session.
        """
        key = self.ffmpeg
        if key not in _NVENC_PROBE:
            try:
                _NVENC_PROBE[key] = " h264_nvenc " in self._encoder_listing() and (
                    subprocess.run(
                        [self.ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                         "color=black:s=256x144:d=0.1", "-frames:v", "1", "-c:v", "h264_nvenc",
                         "-f", "null", "-"],
                        capture_output=True, timeout=60, check=False,
                    ).returncode == 0
                )  # fmt: skip
            except (OSError, subprocess.SubprocessError, TranscriptionError):
                _NVENC_PROBE[key] = False
        return _NVENC_PROBE[key]

    def check_video_encoder(self) -> None:
        """Raise before inference when this FFmpeg build cannot render the MP4 preview.

        Accepts libx264 or a working h264_nvenc, plus aac and mov_text. libx264 is checked
        first so a machine with it never opens a GPU session just to answer this question.
        """
        listing = self._encoder_listing()
        missing = [n for n in ("aac", "mov_text") if f" {n} " not in listing]
        if " libx264 " not in listing and not self.nvenc_usable():
            missing.insert(0, "libx264 or a working h264_nvenc")
        if missing:
            raise TranscriptionError(
                "The configured FFmpeg cannot render the MP4 preview; missing encoder: "
                + ", ".join(missing)
            )

    def _video_attempts(self) -> list[dict]:
        """Encoder chain, best first: GPU decode+scale+NVENC, CPU decode+NVENC, libx264."""
        scale = "scale=-2:'min(720,ih)'"  # never upscales
        nvenc = [
            "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr",
            "-cq", str(NVENC_CQ), "-b:v", "0",
        ]  # fmt: skip
        chain = []
        if self.nvenc_usable():
            chain.append(
                {
                    "encoder": "h264_nvenc", "gpu_decode": True,
                    "pre": ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
                    "codec": ["-vf", "scale_cuda=-2:'min(720,ih)':format=yuv420p", *nvenc],
                }
            )  # fmt: skip
            chain.append(
                {
                    "encoder": "h264_nvenc", "gpu_decode": False, "pre": [],
                    "codec": ["-vf", f"{scale},format=yuv420p", *nvenc],
                }
            )  # fmt: skip
        chain.append(
            {
                "encoder": "libx264", "gpu_decode": False, "pre": [],
                "codec": [
                    "-vf", scale, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                    "-pix_fmt", "yuv420p",
                ],
            }
        )  # fmt: skip
        return chain

    def export_preview_video(
        self,
        source: Path,
        destination: Path,
        *,
        start_seconds: float = 0.0,
        max_duration: float | None = None,
        subtitles: Path | None = None,
        progress: Callable[[dict], None] | None = None,
        warn: Callable[[str], None] | None = None,
    ) -> MediaInfo:
        """Render a 720p-or-smaller H.264/AAC preview of the selection, zero-based.

        Tries NVENC with GPU decode and scaling, NVENC with CPU decode, then libx264; the
        first success wins and every fall-through is announced through ``warn``. The SRT,
        when given, is a soft mov_text track and is never burned in. Publishes atomically;
        the returned ``MediaInfo.render`` records which path produced the file.
        """
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise ValueError("start_seconds must be finite and nonnegative")
        if max_duration is not None and (not math.isfinite(max_duration) or max_duration <= 0):
            raise ValueError("max_duration must be finite and positive")
        info = self.probe(source)
        if not info.has_video:
            raise TranscriptionError(f"Media contains no video stream: {Path(source).name}")
        span = max_duration if max_duration is not None else max(1.0, info.duration - start_seconds)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        attempts = self._video_attempts()
        failure: TranscriptionError | None = None
        for position, attempt in enumerate(attempts):
            command = [self.ffmpeg, "-v", "error", "-nostdin", "-y", *attempt["pre"]]
            if start_seconds:
                command.extend(["-ss", _format_seconds(start_seconds)])
            command.extend(["-i", str(source)])
            if subtitles is not None:
                command.extend(["-i", str(subtitles)])
            command.extend(["-map", "0:v:0", "-map", "0:a:0"])
            if subtitles is not None:
                command.extend(["-map", "1:0"])
            if max_duration is not None:
                command.extend(["-t", _format_seconds(max_duration)])
            command.extend([*attempt["codec"], "-c:a", "aac", "-b:a", "128k"])
            if subtitles is not None:
                command.extend(["-c:s", "mov_text"])
            command.extend(["-movflags", "+faststart", "-f", "mp4", str(partial)])
            render = {"encoder": attempt["encoder"], "gpu_decode": attempt["gpu_decode"]}
            try:
                # A software render is slower than a copy; allow 8x the selection (at least 2 h).
                _run_with_progress(
                    command,
                    timeout=max(7200.0, 8 * span),
                    operation="render preview video",
                    total=span,
                    on_progress=progress,
                    tag=render,
                )
                rendered = self.probe(partial)
                os.replace(partial, destination)
                return dataclasses.replace(rendered, render=render)
            except TranscriptionError as exc:
                failure = exc
                if position + 1 < len(attempts) and warn is not None:
                    following = attempts[position + 1]
                    warn(
                        f"{_path_label(attempt)} failed ({_short_reason(exc)}); "
                        f"using {_path_label(following)} instead"
                    )
            finally:
                partial.unlink(missing_ok=True)
        assert failure is not None
        raise failure

    def prepare(
        self,
        source: Path,
        destination: Path,
        *,
        sample_rate: int,
        segmentation: SegmentationSettings,
        max_duration: float | None = None,
        start_seconds: float = 0.0,
    ) -> PreparedAudio:
        source, destination = Path(source), Path(destination)
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise TranscriptionError("start_seconds must be finite and nonnegative.")
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
            *(["-ss", _format_seconds(start_seconds)] if start_seconds else []),
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

    def export_selection_audio(
        self,
        source: Path,
        destination: Path,
        *,
        sample_rate: int,
        audio_format: str,
        start_seconds: float = 0.0,
        max_duration: float | None = None,
    ) -> MediaInfo:
        """Prepare a playable selection without modifying the source; publish atomically."""
        if audio_format not in {"mp3", "flac"}:
            raise ValueError("audio_format must be mp3 or flac")
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise ValueError("start_seconds must be finite and nonnegative")
        if max_duration is not None and (not math.isfinite(max_duration) or max_duration <= 0):
            raise ValueError("max_duration must be finite and positive")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        command = [self.ffmpeg, "-v", "error", "-nostdin", "-y"]
        if start_seconds:
            command.extend(["-ss", _format_seconds(start_seconds)])
        command.extend(["-i", str(source), "-map", "0:a:0", "-vn"])
        if max_duration is not None:
            command.extend(["-t", _format_seconds(max_duration)])
        command.extend(["-ac", "1", "-ar", str(sample_rate)])
        command.extend(
            ["-c:a", "libmp3lame", "-b:a", "64k"]
            if audio_format == "mp3"
            else ["-sample_fmt", "s16", "-c:a", "flac", "-compression_level", "8"]
        )
        command.extend(["-f", audio_format, str(partial)])
        try:
            _run(command, timeout=3600, operation=f"export {audio_format} audio")
            info = self.probe(partial)
            os.replace(partial, destination)
            return info
        finally:
            partial.unlink(missing_ok=True)

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


def parse_progress(text: str, total: float | None) -> dict | None:
    """Latest ``-progress`` block as ``done_s``/``percent``/``eta_s``; None before any output.

    Without a usable ``total`` the percentage and ETA stay None rather than guessed.
    """
    done = None
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key in {"out_time_us", "out_time_ms"}:  # FFmpeg reports microseconds in both
            try:
                done = max(0.0, int(value) / 1_000_000)
            except ValueError:
                pass
    if done is None:
        return None
    if not total or not math.isfinite(total) or total <= 0:
        return {"done_s": done, "total_s": None, "percent": None, "eta_s": None}
    done = min(done, total)
    return {
        "done_s": done,
        "total_s": total,
        "percent": round(done / total * 100, 1),
        "eta_s": None,
    }


def _run_with_progress(
    command: list[str],
    *,
    timeout: float,
    operation: str,
    total: float | None,
    on_progress: Callable[[dict], None] | None,
    tag: dict | None = None,
) -> None:
    """Run FFmpeg with ``-progress`` written to a file (never an unread pipe), about 1 report/s."""
    with tempfile.TemporaryDirectory(prefix="ffmpeg-run-") as scratch:
        progress_file, log = Path(scratch) / "progress.txt", Path(scratch) / "stderr.txt"
        full = [command[0], "-progress", str(progress_file), "-nostats", *command[1:]]
        started = time.monotonic()
        if on_progress is not None and total:  # a retry starts its bar from zero again
            on_progress(
                {"done_s": 0.0, "total_s": total, "percent": 0.0, "eta_s": None, "speed_x": None,
                 **(tag or {})}
            )  # fmt: skip
        try:
            with log.open("wb") as errors:
                process = subprocess.Popen(
                    full, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors
                )
                try:
                    while True:
                        try:
                            process.wait(timeout=1.0)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                        if time.monotonic() - started > timeout:
                            raise TranscriptionError(f"Timed out while trying to {operation}.")
                        _report(progress_file, total, started, on_progress, tag)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
        except FileNotFoundError as exc:
            raise TranscriptionError(
                f"Could not {operation}: executable '{command[0]}' was not found. "
                "Install FFmpeg and ensure it is on PATH."
            ) from exc
        if process.returncode != 0:
            message = _diagnostic(log.read_text(encoding="utf-8", errors="replace"))
            raise TranscriptionError(
                f"Could not {operation}: {message or 'FFmpeg exited with an error.'}"
            )
        if on_progress is not None and total:
            on_progress(
                {"done_s": total, "total_s": total, "percent": 100.0, "eta_s": 0.0,
                 "speed_x": None, **(tag or {})}
            )  # fmt: skip


def _report(
    progress_file: Path, total: float | None, started: float, callback, tag: dict | None = None
) -> None:
    if callback is None:
        return
    try:
        with progress_file.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 4096))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return
    info = parse_progress(text, total)
    if info is None:
        return
    elapsed = max(time.monotonic() - started, 0.001)
    info["speed_x"] = round(info["done_s"] / elapsed, 2) if info["done_s"] else None
    if info["total_s"] and info["speed_x"]:
        info["eta_s"] = round((info["total_s"] - info["done_s"]) / info["speed_x"], 1)
    callback({**info, **(tag or {})})


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

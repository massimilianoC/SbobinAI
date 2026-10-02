"""Speech detectors that score audio frame by frame on the CPU.

The Silero model I/O contract (``input`` of 64 context samples plus a 512 sample
window at 16 kHz, recurrent ``state`` of shape (2, 1, 128), ``sr`` scalar) follows
these MIT-licensed references:
https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py
https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
"""

from __future__ import annotations

import math
import re
import subprocess
import wave
from pathlib import Path

from audio_transcript.domain.models import TranscriptionError, VoiceActivity

SILERO_SAMPLE_RATE = 16000
SILERO_WINDOW = 512
SILERO_CONTEXT = 64
_READ_WINDOWS = 2048  # about 65 s of audio per block

ENERGY_FRAME_SECONDS = 0.03
_DB_FLOOR = -90.0
_MIN_THRESHOLD_DB = -55.0
_SPEECH_HEADROOM_DB = 6.0
_RAMP_DB = 10.0
_LEVEL_PATTERN = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(\S+)")


def make_detector(
    name: str,
    *,
    model_path: Path | None,
    threshold: float,
    energy_margin_db: float,
    ffmpeg: str = "ffmpeg",
):
    """Build the detector called ``name``; ``"none"`` returns ``None``."""
    if name == "none":
        return None
    if name == "silero":
        return SileroVadDetector(Path(model_path or "models/silero_vad.onnx"), threshold=threshold)
    if name == "energy":
        return EnergyDetector(ffmpeg=ffmpeg, threshold=threshold, margin_db=energy_margin_db)
    raise TranscriptionError(f"Unknown speech detector '{name}'. Use 'silero', 'energy' or 'none'.")


class SileroVadDetector:
    """Silero VAD (ONNX) on onnxruntime's CPU provider. Needs the ``vad`` extra."""

    name = "silero"

    def __init__(self, model_path: Path, *, threshold: float = 0.5, threads: int = 1) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.threads = threads
        self._session = None

    def check(self) -> None:
        if not self.model_path.is_file():
            raise TranscriptionError(
                f"Silero VAD model not found: {self.model_path}. Run "
                "scripts/install-silero-vad.ps1 or choose another segmentation detector."
            )
        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise TranscriptionError(
                "Silero VAD needs onnxruntime and numpy. Install them with "
                "`pip install -e .[vad]`, or use the 'energy' detector."
            ) from exc

    def _load(self):
        if self._session is None:
            self.check()
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            self._session = onnxruntime.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        return self._session

    def detect(self, wav_path: Path) -> VoiceActivity:
        session = self._load()
        import numpy as np

        probabilities: list[float] = []
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, SILERO_CONTEXT), dtype=np.float32)
        rate = np.array(SILERO_SAMPLE_RATE, dtype=np.int64)
        try:
            with wave.open(str(wav_path), "rb") as wav:
                if (
                    wav.getframerate() != SILERO_SAMPLE_RATE
                    or wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                ):
                    raise TranscriptionError(
                        "Silero VAD needs 16 kHz mono 16-bit PCM audio; got "
                        f"{wav.getframerate()} Hz, {wav.getnchannels()} channel(s)."
                    )
                carry = np.zeros(0, dtype=np.float32)
                while True:
                    raw = wav.readframes(SILERO_WINDOW * _READ_WINDOWS)
                    if raw:
                        block = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                        carry = np.concatenate((carry, block)) if carry.size else block
                    final = not raw
                    usable = carry.size - carry.size % SILERO_WINDOW
                    if final and carry.size > usable:
                        padded = np.zeros(usable + SILERO_WINDOW, dtype=np.float32)
                        padded[: carry.size] = carry
                        carry, usable = padded, usable + SILERO_WINDOW
                    for offset in range(0, usable, SILERO_WINDOW):
                        window = carry[offset : offset + SILERO_WINDOW][None, :]
                        model_input = np.concatenate((context, window), axis=1)
                        output, state = session.run(
                            None, {"input": model_input, "state": state, "sr": rate}
                        )
                        context = model_input[:, -SILERO_CONTEXT:]
                        probabilities.append(float(np.asarray(output).reshape(-1)[0]))
                    carry = carry[usable:]
                    if final:
                        break
        except (wave.Error, OSError) as exc:
            raise TranscriptionError(f"Could not read audio for voice detection: {exc}") from exc
        return VoiceActivity(
            frame_seconds=SILERO_WINDOW / SILERO_SAMPLE_RATE,
            probabilities=tuple(probabilities),
            threshold=self.threshold,
            detector=self.name,
            details={"model": self.model_path.name, "threads": self.threads},
        )


class EnergyDetector:
    """Adaptive frame-energy detector computed by FFmpeg; no extra dependencies."""

    name = "energy"

    def __init__(
        self,
        ffmpeg: str = "ffmpeg",
        *,
        threshold: float = 0.5,
        margin_db: float = 15.0,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.threshold = threshold
        self.margin_db = margin_db

    def check(self) -> None:
        try:
            result = subprocess.run(
                [self.ffmpeg, "-version"], capture_output=True, timeout=10, check=False
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise TranscriptionError(
                f"The energy detector needs FFmpeg; '{self.ffmpeg}' is not usable."
            ) from exc
        if result.returncode != 0:
            raise TranscriptionError(f"'{self.ffmpeg} -version' failed.")

    def detect(self, wav_path: Path) -> VoiceActivity:
        try:
            with wave.open(str(wav_path), "rb") as wav:
                rate = wav.getframerate()
        except (wave.Error, OSError) as exc:
            raise TranscriptionError(f"Could not read audio for voice detection: {exc}") from exc
        frame_samples = max(1, round(ENERGY_FRAME_SECONDS * rate))
        levels = self._frame_levels(wav_path, frame_samples)
        frame_seconds = frame_samples / rate
        if not levels:
            return VoiceActivity(frame_seconds, (), self.threshold, self.name, {"frames": 0})
        ordered = sorted(levels)
        noise_floor = _percentile(ordered, 0.10)
        speech_level = _percentile(ordered, 0.95)
        threshold_db = max(
            _MIN_THRESHOLD_DB,
            min(noise_floor + self.margin_db, speech_level - _SPEECH_HEADROOM_DB),
        )
        probabilities = tuple(_level_probability(db, threshold_db, self.threshold) for db in levels)
        return VoiceActivity(
            frame_seconds=frame_seconds,
            probabilities=probabilities,
            threshold=self.threshold,
            detector=self.name,
            details={
                "noise_floor_db": round(noise_floor, 2),
                "speech_level_db": round(speech_level, 2),
                "threshold_db": round(threshold_db, 2),
                "margin_db": self.margin_db,
            },
        )

    def _frame_levels(self, wav_path: Path, frame_samples: int) -> list[float]:
        graph = (
            f"asetnsamples=n={frame_samples}:p=0,"
            "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=RMS_level,"
            "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-"
        )
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(wav_path),
            "-af",
            graph,
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=3600, check=False
            )
        except FileNotFoundError as exc:
            raise TranscriptionError(
                f"Could not analyse energy: executable '{self.ffmpeg}' was not found."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TranscriptionError("Timed out while analysing audio energy.") from exc
        if result.returncode != 0:
            raise TranscriptionError(
                f"Could not analyse energy: {(result.stderr or '').strip()[-600:]}"
            )
        levels: list[float] = []
        for match in _LEVEL_PATTERN.finditer(result.stdout):
            value = match.group(1)
            level = _DB_FLOOR if "inf" in value or "nan" in value else float(value)
            levels.append(max(_DB_FLOOR, level))
        return levels


def _percentile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _level_probability(level_db: float, threshold_db: float, threshold: float) -> float:
    """Map dB to [0, 1] piecewise linearly so ``threshold_db`` maps to ``threshold``."""
    delta = (level_db - threshold_db) / _RAMP_DB
    if delta >= 0:
        return threshold + (1.0 - threshold) * min(1.0, delta)
    return threshold * max(0.0, 1.0 + delta)
